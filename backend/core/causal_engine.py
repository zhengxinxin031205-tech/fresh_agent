"""
因果推断与干预模拟引擎
======================
1. 因果发现:通过偏相关分析 + 后门准则,自动识别价格、促销、节假日、
   气温、降水等因素对销量的真实因果效应(排除混淆变量干扰)
2. 因果效应估计:控制混淆变量后的净效应(ATE),价格弹性系数
3. 反事实仿真:输入拟执行的经营动作(调价/促销/天气假设),
   输出对应的销量、库存、损耗变化模拟结果
"""
import numpy as np
import pandas as pd
from typing import Optional
from backend.core.stat_utils import SimpleLinearRegression as LinearRegression
from backend.core.parallel_utils import parallel_map
from backend.core.np_utils import to_native


# 因果图定义:混淆变量关系(哪些变量同时影响干预和结果)
CONFOUNDERS = {
    "is_promo":     ["is_weekend", "is_holiday"],       # 促销常安排在周末/节假日
    "price":        ["is_promo", "is_weekend"],         # 价格受促销和时段影响
    "temperature":  [],                                  # 天气是外生变量
    "precipitation": [],
    "is_holiday":   [],
    "is_weekend":   [],
}

# 核心列(结果/库存/成本等,不是影响因素)
CORE_COLUMNS = {
    "date", "sku_id", "sku_name", "category", "sales", "demand", "recovered_demand",
    "price", "base_price", "inventory", "in_transit", "stockout",
    "waste_qty", "near_expiry", "near_expiry_price",
    "cost", "shelf_life", "lead_time", "unit", "gross_margin", "stockout_loss",
    "holiday_name",
}

# 已知影响因素(有专门的因果图和特征工程逻辑)
KNOWN_FACTORS = {"is_promo", "discount", "temperature", "precipitation", "is_holiday", "is_weekend"}


class CausalEngine:
    """因果推断引擎"""

    def __init__(self, df: pd.DataFrame, excluded_factors: Optional[set] = None):
        self.df = df.copy()
        self.effects: dict = {}
        self.elasticity: dict = {}
        self._fitted = False
        self.excluded = excluded_factors or set()
        self._custom_factors = self._detect_custom_factors(df)

    def _detect_custom_factors(self, df: pd.DataFrame) -> list:
        """检测数据中的自定义影响因素(不在核心列和已知因素中的数值列,商家传什么就分析什么)"""
        custom = []
        for col in df.columns:
            if col in CORE_COLUMNS or col in KNOWN_FACTORS:
                continue
            if col in self.excluded:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                custom.append(col)
        return custom

    def get_custom_factors(self) -> list:
        """返回当前数据中检测到的自定义影响因素列表"""
        return list(self._custom_factors)

    # ──────────── 特征工程 ────────────
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建因果分析特征矩阵"""
        feat = pd.DataFrame(index=df.index)
        if "is_weekend" in df.columns:
            feat["weekend"] = df["is_weekend"].astype(float)
        else:
            feat["weekend"] = 0.0
        if "is_holiday" in df.columns:
            feat["holiday"] = df["is_holiday"].astype(float)
        else:
            feat["holiday"] = 0.0
        if "is_promo" in df.columns:
            feat["promo"] = df["is_promo"].astype(float)
        else:
            feat["promo"] = 0.0
        if "temperature" in df.columns:
            feat["temp"] = df["temperature"].fillna(df["temperature"].mean() if df["temperature"].notna().any() else 22)
            feat["temp_dev"] = (feat["temp"] - feat["temp"].mean()) / 10  # 偏离均值每10℃
        else:
            feat["temp"] = 22.0
            feat["temp_dev"] = 0.0
        if "precipitation" in df.columns:
            feat["rain"] = (df["precipitation"].fillna(0) > 5).astype(float)  # 大雨标记
        else:
            feat["rain"] = 0.0
        if "price" in df.columns and "base_price" in df.columns:
            bp = df["base_price"].fillna(df["price"])
            feat["log_price_ratio"] = np.log(df["price"].fillna(bp) / bp.clip(lower=0.01))
        elif "discount" in df.columns:
            feat["log_price_ratio"] = np.log(df["discount"].fillna(1.0).clip(lower=0.3))
        else:
            feat["log_price_ratio"] = 0.0
        # 周几哑变量
        if "date" in df.columns:
            for d in range(1, 7):
                feat[f"dow_{d}"] = (df["date"].dt.dayofweek == d).astype(float)
        # 自定义影响因素(商家上传什么因素就分析什么)
        for col in self._custom_factors:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                uniq = set(vals.dropna().unique())
                if uniq <= {0.0, 1.0}:
                    feat[col] = vals.fillna(0).astype(float)      # 二元因素缺失视为"无"
                else:
                    med = vals.median() if vals.notna().any() else 0.0
                    feat[col] = vals.fillna(med).astype(float)    # 连续因素缺失填中位数
        return feat

    # ──────────── 因果效应估计(后门调整) ────────────
    def estimate_effects(self) -> dict:
        """
        对每个干预变量,通过后门调整估计平均处理效应(ATE):
        1. 识别该干预的混淆变量集 Z
        2. 用回归控制 Z,得到干预变量的净系数
        3. 转换为业务可解释的效应量
        """
        df = self.df
        if "sku_id" not in df.columns:
            return {}
        # 先一次性预计算各商品的特征矩阵(保持按商品归一化,与原逻辑一致),
        # 再把特征传入并行任务,避免在并行函数内重复构建
        groups = list(df.groupby("sku_id"))
        feat_by_sku = {sid: self._prepare_features(grp) for sid, grp in groups}

        # 并行处理每个商品的全部因素估计
        outcomes = parallel_map(
            lambda t: self._estimate_single_sku(t[0], t[1], t[2]),
            [(sid, grp, feat_by_sku[sid]) for sid, grp in groups],
            desc="estimate_effects",
        )

        # 合并结果(价格弹性写回 self.elasticity)
        results = {}
        for sku_id, sku_result, elasticity in outcomes:
            results[sku_id] = sku_result
            if elasticity is not None:
                self.elasticity[sku_id] = elasticity

        self.effects = results
        self._fitted = True
        return results

    def _estimate_single_sku(self, sku_id, grp, feat) -> tuple:
        """处理单个商品的全部因素估计,返回 (商品编号, 结果dict, 价格弹性或None)"""
        sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sku_id
        y = grp["sales"].astype(float).values
        sku_result = {"sku_name": sku_name, "effects": {}}
        elasticity = None

        # ── 促销效应(控制周末/节假日混淆) ──
        if "is_promo" in grp.columns and grp["is_promo"].nunique() > 1 and "promo" not in self.excluded:
            Z_cols = [c for c in ["weekend", "holiday"] if c in feat.columns]
            X = feat[["promo"] + Z_cols].values
            if np.std(feat["promo"].values) > 0:
                model = LinearRegression().fit(X, y)
                promo_ate = float(model.coef_[0])
                # 对照组均值
                ctrl_mean = float(y[grp["is_promo"].values == 0].mean()) if (grp["is_promo"].values == 0).any() else float(y.mean())
                sku_result["effects"]["promo"] = {
                    "ate": round(promo_ate, 1),
                    "ate_pct": round(promo_ate / max(ctrl_mean, 1) * 100, 1),
                    "ctrl_mean": round(ctrl_mean, 1),
                    "confounders_controlled": Z_cols,
                }

        # ── 价格弹性(控制促销/周末混淆) ──
        if "log_price_ratio" in feat.columns and np.std(feat["log_price_ratio"].values) > 0.01:
            Z_cols = [c for c in ["promo", "weekend"] if c in feat.columns]
            X = feat[["log_price_ratio"] + Z_cols].values
            model = LinearRegression().fit(X, y)
            elasticity = float(model.coef_[0])  # log-log 弹性
            sku_result["effects"]["price_elasticity"] = round(elasticity, 2)

        # ── 节假日效应 ──
        if "is_holiday" in grp.columns and grp["is_holiday"].sum() > 0 and "holiday" not in self.excluded:
            Z_cols = [c for c in ["weekend"] if c in feat.columns]
            X = feat[["holiday"] + Z_cols].values
            model = LinearRegression().fit(X, y)
            holiday_ate = float(model.coef_[0])
            ctrl = float(y[grp["is_holiday"].values == 0].mean()) if (grp["is_holiday"].values == 0).any() else float(y.mean())
            sku_result["effects"]["holiday"] = {
                "ate": round(holiday_ate, 1),
                "ate_pct": round(holiday_ate / max(ctrl, 1) * 100, 1),
            }

        # ── 周末效应 ──
        if "weekend" in feat.columns and np.std(feat["weekend"].values) > 0 and "weekend" not in self.excluded:
            X = feat[["weekend"]].values
            model = LinearRegression().fit(X, y)
            weekend_ate = float(model.coef_[0])
            ctrl = float(y[feat["weekend"].values == 0].mean())
            sku_result["effects"]["weekend"] = {
                "ate": round(weekend_ate, 1),
                "ate_pct": round(weekend_ate / max(ctrl, 1) * 100, 1),
            }

        # ── 天气效应 ──
        if "temp_dev" in feat.columns and np.std(feat["temp_dev"].values) > 0 and "temperature" not in self.excluded:
            X = feat[["temp_dev", "rain"]].values
            model = LinearRegression().fit(X, y)
            sku_result["effects"]["temperature"] = {
                "ate_per_10c": round(float(model.coef_[0]), 1),
                "ate_pct_per_10c": round(float(model.coef_[0]) / max(y.mean(), 1) * 100, 1),
            }
            if np.std(feat["rain"].values) > 0:
                rain_ate = float(model.coef_[1])
                sku_result["effects"]["heavy_rain"] = {
                    "ate": round(rain_ate, 1),
                    "ate_pct": round(rain_ate / max(y.mean(), 1) * 100, 1),
                }

        # ── 自定义因素效应(商家上传什么因素就分析什么) ──
        for col in self._custom_factors:
            if col not in feat.columns:
                continue
            vals = feat[col].values
            if np.std(vals) == 0:
                continue
            Z_cols = [c for c in ["weekend", "holiday"] if c in feat.columns]
            X = feat[[col] + Z_cols].values
            model = LinearRegression().fit(X, y)
            ate = float(model.coef_[0])
            uniq = set(np.unique(vals))
            if uniq <= {0.0, 1.0}:
                # 二元因素:有/无该因素对销量的影响
                ctrl_mean = float(y[vals == 0].mean()) if (vals == 0).any() else float(y.mean())
                sku_result["effects"][col] = {
                    "ate": round(ate, 1),
                    "ate_pct": round(ate / max(ctrl_mean, 1) * 100, 1),
                    "ctrl_mean": round(ctrl_mean, 1),
                    "factor_type": "binary",
                }
            else:
                # 连续因素:每增加1个单位对销量的影响
                sku_result["effects"][col] = {
                    "ate_per_unit": round(ate, 2),
                    "ate_pct_per_unit": round(ate / max(y.mean(), 1) * 100, 2),
                    "factor_type": "continuous",
                }

        return sku_id, sku_result, elasticity

    # ──────────── 因果发现(因素贡献占比) ────────────
    def discover_causal_factors(self, sku_id: Optional[str] = None) -> dict:
        """
        基于回归的方差分解:计算各因果因素对销量变异的贡献占比
        """
        if not self._fitted:
            self.estimate_effects()
        df = self.df
        target_skus = [sku_id] if sku_id else list(df["sku_id"].unique())

        # 先一次性预计算目标商品的数据和特征(过滤掉数据量不足的),再并行计算
        task_items = []
        for sid in target_skus:
            grp = df[df["sku_id"] == sid]
            if len(grp) < 10:
                continue
            feat = self._prepare_features(grp)
            task_items.append((sid, grp, feat))

        # 并行计算各商品的因素贡献占比
        outcomes = parallel_map(
            lambda t: self._discover_single_sku(t[0], t[1], t[2]),
            task_items,
            desc="discover_causal_factors",
        )

        # 合并结果(数据不足或无可用因素的商品会被跳过)
        contributions = {}
        for sid, contrib in outcomes:
            if contrib is not None:
                contributions[sid] = contrib
        return contributions

    def _discover_single_sku(self, sid, grp, feat) -> tuple:
        """计算单个商品的各因素贡献占比,返回 (商品编号, 贡献dict或None)"""
        sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sid
        y = grp["sales"].astype(float).values
        avail = [c for c in ["promo", "log_price_ratio", "holiday", "weekend", "temp_dev", "rain"]
                 if c in feat.columns and c not in self.excluded]
        avail += [c for c in self._custom_factors if c in feat.columns and np.std(feat[c].values) > 0]
        if not avail:
            return sid, None
        X = feat[avail].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        ss_total = max(float(np.sum((y - y.mean()) ** 2)), 1e-6)
        # 各因素贡献 = 该系数 × 该变量变异
        factor_contrib = {}
        for i, col in enumerate(avail):
            contrib = abs(model.coef_[i]) * float(np.std(feat[col].values))
            factor_contrib[col] = contrib
        total_contrib = sum(factor_contrib.values()) + 1e-6
        # 归一化为百分比
        label_map = {"promo": "促销活动", "log_price_ratio": "价格变动", "holiday": "节假日",
                     "weekend": "周末效应", "temp_dev": "气温变化", "rain": "大雨天气"}
        contrib_dict = {
            "sku_name": sku_name,
            "factors": {label_map.get(k, k): round(v / total_contrib * 100, 1) for k, v in factor_contrib.items()},
            "r_squared": round(float(model.score(X, y)), 3),
        }
        return sid, contrib_dict

    # ──────────── 反事实仿真 ────────────
    def simulate_intervention(self, sku_id: str, interventions: dict) -> dict:
        """
        反事实仿真:输入拟执行的经营动作,预测销量变化
        interventions 示例:
            {"discount": 0.8, "is_promo": 1, "temperature": 30, "heavy_rain": 0, "is_holiday": 0}
        """
        if not self._fitted:
            self.estimate_effects()
        df = self.df
        grp = df[df["sku_id"] == sku_id]
        if len(grp) == 0:
            return {"error": f"SKU {sku_id} 不存在"}
        sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sku_id
        category = grp["category"].iloc[0] if "category" in grp.columns else ""

        # 基准需求(近14天恢复后销量的均值,若有)
        recent = grp.tail(14)
        if "recovered_demand" in recent.columns:
            base_demand = float(recent["recovered_demand"].mean())
        else:
            base_demand = float(recent["sales"].mean())
        base_demand = max(base_demand, 1.0)

        # 基准价格
        base_price = float(grp["price"].mean()) if "price" in grp.columns else 5.0
        base_price_std = float(grp["base_price"].iloc[0]) if "base_price" in grp.columns else base_price

        # 逐项计算因果效应
        demand_multiplier = 1.0
        detail = []
        elasticity = self.elasticity.get(sku_id, -1.5)
        if elasticity >= 0:  # 正弹性不合理(涨价反而多卖),说明回归质量差,回退默认值
            elasticity = -1.5

        # 价格/折扣干预
        if "discount" in interventions:
            disc = float(interventions["discount"])
            new_price = base_price_std * disc  # 折后价=标准价×折扣,用于展示
            price_ratio = disc  # 折扣率即价格比率(相对标准价);弹性估计时用的log(price/base_price)中base_price也是标准价,口径一致
            demand_multiplier *= price_ratio ** elasticity
            detail.append({"factor": "价格调整", "change": f"折扣{disc*100:.0f}%",
                           "impact_pct": round((price_ratio ** elasticity - 1) * 100, 1)})

        if "price_change_pct" in interventions:
            pct = float(interventions["price_change_pct"]) / 100
            demand_multiplier *= (1 + pct) ** elasticity
            detail.append({"factor": "价格调整", "change": f"价格变动{pct*100:+.0f}%",
                           "impact_pct": round(((1 + pct) ** elasticity - 1) * 100, 1)})

        # 促销干预
        if interventions.get("is_promo") and "promo" not in self.excluded:
            promo_effect = self.effects.get(sku_id, {}).get("effects", {}).get("promo", {})
            promo_pct = promo_effect.get("ate_pct", 15.0)
            demand_multiplier *= (1 + promo_pct / 100)
            detail.append({"factor": "促销活动", "change": "开启促销",
                           "impact_pct": round(promo_pct, 1)})

        # 节假日干预
        if interventions.get("is_holiday") and "holiday" not in self.excluded:
            hol_effect = self.effects.get(sku_id, {}).get("effects", {}).get("holiday", {})
            hol_pct = hol_effect.get("ate_pct", 30.0)
            demand_multiplier *= (1 + hol_pct / 100)
            detail.append({"factor": "节假日", "change": "节假日效应",
                           "impact_pct": round(hol_pct, 1)})

        # 天气干预
        if "temperature" in interventions and "temperature" not in self.excluded:
            temp = float(interventions["temperature"])
            temp_effect = self.effects.get(sku_id, {}).get("effects", {}).get("temperature", {})
            per_10c = temp_effect.get("ate_pct_per_10c", 3.0)
            temp_dev = (temp - 22) / 10
            demand_multiplier *= (1 + per_10c / 100 * temp_dev)
            detail.append({"factor": "气温变化", "change": f"气温{temp}℃",
                           "impact_pct": round(per_10c * temp_dev, 1)})

        if interventions.get("heavy_rain") and "heavy_rain" not in self.excluded:
            rain_effect = self.effects.get(sku_id, {}).get("effects", {}).get("heavy_rain", {})
            rain_pct = rain_effect.get("ate_pct", -12.0)
            demand_multiplier *= (1 + rain_pct / 100)
            detail.append({"factor": "大雨天气", "change": "有大雨",
                           "impact_pct": round(rain_pct, 1)})

        # 自定义因素干预(商家数据里有什么因素,就能模拟什么因素)
        for col in self._custom_factors:
            if col not in interventions:
                continue
            val = float(interventions[col])
            eff = self.effects.get(sku_id, {}).get("effects", {}).get(col, {})
            if eff.get("factor_type") == "binary":
                if val > 0:
                    pct = eff.get("ate_pct", 0)
                    demand_multiplier *= (1 + pct / 100)
                    detail.append({"factor": col, "change": "开启",
                                   "impact_pct": round(pct, 1)})
            else:
                pct_per_unit = eff.get("ate_pct_per_unit", 0)
                base_val = float(pd.to_numeric(grp[col], errors="coerce").mean()) if col in grp.columns else 0
                delta = val - base_val
                demand_multiplier *= (1 + pct_per_unit / 100 * delta)
                detail.append({"factor": col, "change": f"设为{val:g}",
                               "impact_pct": round(pct_per_unit * delta, 1)})

        simulated_demand = base_demand * demand_multiplier
        # 库存与损耗影响
        current_inv = float(grp["inventory"].iloc[-1]) if "inventory" in grp.columns else 0
        shelf_life = float(grp["shelf_life"].iloc[0]) if "shelf_life" in grp.columns else 5
        cost = float(grp["cost"].iloc[0]) if "cost" in grp.columns else base_price * 0.5
        margin = base_price - cost

        new_sales = min(simulated_demand, current_inv + simulated_demand * 0.8)  # 简化库存约束
        stockout_risk = max(0, simulated_demand - current_inv) / max(simulated_demand, 1)
        overstock_risk = max(0, current_inv - simulated_demand * 1.3) / max(current_inv, 1)
        # 损耗变化:过量库存增加损耗
        waste_delta = overstock_risk * current_inv * (3.0 / shelf_life) * 0.5
        revenue_delta = (simulated_demand - base_demand) * margin
        waste_cost_delta = waste_delta * cost

        return to_native({
            "sku_id": sku_id, "sku_name": sku_name, "category": category,
            "base_demand": round(base_demand, 1),
            "simulated_demand": round(simulated_demand, 1),
            "demand_change_pct": round((demand_multiplier - 1) * 100, 1),
            "demand_change_abs": round(simulated_demand - base_demand, 1),
            "current_inventory": round(current_inv, 1),
            "suggested_replenishment_adjustment": round(max(0, simulated_demand - current_inv), 1),
            "stockout_risk_pct": round(stockout_risk * 100, 1),
            "overstock_risk_pct": round(overstock_risk * 100, 1),
            "expected_revenue_change": round(revenue_delta, 1),
            "expected_waste_change": round(waste_delta, 1),
            "expected_waste_cost_change": round(waste_cost_delta, 1),
            "factor_details": detail,
        })

    # ──────────── 全局因果效应摘要 ────────────
    def get_global_summary(self) -> dict:
        """跨SKU的因果效应汇总"""
        if not self._fitted:
            self.estimate_effects()
        promo_effects = [v["effects"]["promo"]["ate_pct"] for v in self.effects.values()
                         if "promo" in v.get("effects", {})] if "promo" not in self.excluded else []
        holiday_effects = [v["effects"]["holiday"]["ate_pct"] for v in self.effects.values()
                           if "holiday" in v.get("effects", {})] if "holiday" not in self.excluded else []
        weekend_effects = [v["effects"]["weekend"]["ate_pct"] for v in self.effects.values()
                           if "weekend" in v.get("effects", {})] if "weekend" not in self.excluded else []
        rain_effects = [v["effects"]["heavy_rain"]["ate_pct"] for v in self.effects.values()
                        if "heavy_rain" in v.get("effects", {})] if "heavy_rain" not in self.excluded else []
        elasticities = [v["effects"]["price_elasticity"] for v in self.effects.values()
                        if "price_elasticity" in v.get("effects", {})]
        result = {
            "avg_promo_uplift_pct": round(float(np.mean(promo_effects)), 1) if promo_effects else 0,
            "avg_holiday_uplift_pct": round(float(np.mean(holiday_effects)), 1) if holiday_effects else 0,
            "avg_weekend_uplift_pct": round(float(np.mean(weekend_effects)), 1) if weekend_effects else 0,
            "avg_rain_impact_pct": round(float(np.mean(rain_effects)), 1) if rain_effects else 0,
            "avg_price_elasticity": round(float(np.mean(elasticities)), 2) if elasticities else -1.5,
            "n_skus_analyzed": len(self.effects),
        }
        # 自定义因素汇总
        custom_summary = {}
        for col in self._custom_factors:
            pcts = []
            for v in self.effects.values():
                eff = v.get("effects", {}).get(col)
                if eff:
                    pcts.append(eff.get("ate_pct") if eff.get("factor_type") == "binary" else eff.get("ate_pct_per_unit", 0))
            if pcts:
                custom_summary[col] = round(float(np.mean(pcts)), 1)
        if custom_summary:
            result["custom_factors"] = custom_summary
        return result