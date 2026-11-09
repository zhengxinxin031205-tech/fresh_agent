"""
不对称损失智能补货决策
======================
基于因果校正后的需求分布,以「缺货销售损失 + 过剩过期损耗」的
总期望损失最小化为目标,求解最优补货量(Newsvendor 模型)。

核心公式:
    临界比率 CR = Cu / (Cu + Co)
    其中 Cu = 单位缺货损失(毛利 + 缺货额外流失,数据无记录时额外流失为0)
          Co = 单位过剩损失 = 采购成本 × (1 − 残值率) × 保质期折减因子
               残值率默认0.3(临期处理能回收30%成本),保质期折减因子 clip(7/保质期, 0.5, 2.0)
               保质期越短 → 过剩惩罚越大,将「易腐程度」显式编码进成本函数
    最优补货量 Q* = F^{-1}(CR),F 为需求 CDF(正态近似)

安全库存系数根据需求波动率、补货周期动态调整(保质期影响已编码进成本函数 Co)。
"""
import numpy as np
import pandas as pd
from backend.core.stat_utils import norm_ppf, norm_cdf, norm_pdf
from backend.core.parallel_utils import parallel_map
from backend.core.param_fallback import ParamFallback
from backend.core.np_utils import to_native
from typing import Optional


class ReplenishmentOptimizer:
    """不对称损失补货优化器"""

    # 默认参数(完整版数据缺失时的极端兜底,正常走三级回退从数据取值)
    DEFAULT_PARAMS = {
        "stockout_customer_loss": 2.0,   # 单次缺货客损估值(元/件,数据无记录时的兜底)
        "salvage_value_ratio": 0.3,      # 残值率:临期处理能回收的采购成本比例
        "min_service_level": 0.30,       # 薄利易腐品(如散装叶菜)经济最优CR可低至0.4
        "max_service_level": 0.99,
    }

    def __init__(self, df: pd.DataFrame, tier: str = "basic"):
        self.df = df.copy()
        self.tier = tier
        self.params = dict(self.DEFAULT_PARAMS)
        # 三级参数回退: SKU级→品类级→全局级, __init__ 一次性构建,后续只读(并行安全)
        self._fallback = ParamFallback(df)

    # ──────────── 单SKU最优补货量 ────────────
    def optimize_sku(self, sku_id: str, demand_mean: float, demand_std: float,
                     current_inventory: float = 0, in_transit: float = 0) -> dict:
        """
        计算单个SKU的最优补货量
        Args:
            demand_mean: 恢复后的日均需求
            demand_std:  需求标准差
            current_inventory: 当前库存
            in_transit: 在途库存
        """
        grp = self.df[self.df["sku_id"] == sku_id]
        return self._optimize_sku_with_grp(sku_id, grp, demand_mean, demand_std,
                                           current_inventory, in_transit)

    def _optimize_sku_with_grp(self, sku_id: str, grp: pd.DataFrame,
                               demand_mean: float, demand_std: float,
                               current_inventory: float = 0, in_transit: float = 0) -> dict:
        """optimize_sku 的计算内核:直接接收预取的 SKU 元数据组(grp),
        避免并行时每个 SKU 重复扫描全表 self.df。"""
        sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns and len(grp) > 0 else sku_id
        category = grp["category"].iloc[0] if "category" in grp.columns and len(grp) > 0 else ""
        unit = grp["unit"].iloc[0] if "unit" in grp.columns and len(grp) > 0 else "件"

        # ── 成本参数提取(三级回退: SKU级→品类级→全局级) ──
        fb = self._fallback.get_params(sku_id)
        price = fb["price"]
        cost = fb["cost"]
        shelf_life = fb["shelf_life"]
        lead_time = fb["lead_time"]
        margin = max(price - cost, 0.1)
        # 缺货额外流失(客损): 从数据取,无数据默认0
        stockout_loss = fb.get("stockout_loss", 0.0)
        # ── 不对称损失参数 ──
        Cu = margin + stockout_loss           # 缺货单位损失(毛利+缺货额外流失)
        # 残值率: 临期处理能回收的采购成本比例,从数据取(临期价/采购成本),无数据默认0.3
        salvage_value = fb.get("salvage_value", cost * self.params["salvage_value_ratio"])
        salvage_value_ratio = salvage_value / cost if cost > 0 else self.params["salvage_value_ratio"]
        # 保质期折减因子: 保质期越短 → 过剩惩罚越大(易腐程度编码进成本函数)
        shelf_factor = np.clip(7.0 / shelf_life, 0.5, 2.0)       # 7天为基准
        # 过剩单位损失 = 采购成本 × (1 − 残值率) × 保质期折减因子,防负值
        Co = max(cost * (1 - salvage_value_ratio) * shelf_factor, 0.1)
        critical_ratio = Cu / (Cu + Co)
        critical_ratio = float(np.clip(critical_ratio, 0.30, 0.99))

        # ── 需求分布参数(覆盖补货周期+提前期) ──
        total_days = max(lead_time, 1)
        period_mean = demand_mean * total_days
        period_std = max(demand_std * np.sqrt(total_days), 0.5)

        # ── 动态安全库存系数 ──
        # (保质期的影响已通过 Co 编码进成本函数,不再重复调整 CR)
        # 需求波动越大 → 缺货风险越高 → 提高安全库存
        # 补货周期越长 → 不确定性越高 → 适度提高
        cv = period_std / max(period_mean, 1)                      # 变异系数
        volatility_factor = np.clip(1 + cv * 0.3, 1.0, 1.5)
        lead_factor = np.clip(1 + (lead_time - 1) * 0.05, 1.0, 1.2)
        # 综合调整临界比率
        adjusted_cr = critical_ratio * volatility_factor * lead_factor
        adjusted_cr = float(np.clip(adjusted_cr, 0.30, 0.99))

        # ── 求解最优库存水平 ──
        z = norm_ppf(adjusted_cr)
        optimal_stock = period_mean + z * period_std
        optimal_stock = max(optimal_stock, 0)

        # ── 补货量 = 最优库存 - 现有库存 - 在途 ──
        net_inventory = current_inventory + in_transit
        reorder_qty = max(0, optimal_stock - net_inventory)

        # ── 服务水平与风险估算 ──
        service_level = float(norm_cdf((optimal_stock - period_mean) / period_std))
        # 期望缺货量
        expected_shortage = float(period_std * norm_pdf(z) - (optimal_stock - period_mean) * (1 - service_level))
        expected_shortage = max(expected_shortage, 0)
        # 期望过剩量
        expected_overage = max(0, optimal_stock - period_mean * (1 + z * cv * 0.3))
        # 期望总损失
        expected_stockout_loss = expected_shortage * Cu
        expected_overstock_loss = expected_overage * Co
        total_expected_loss = expected_stockout_loss + expected_overstock_loss

        # ── 损耗风险等级 ──
        waste_risk = self._waste_risk_level(shelf_life, cv, reorder_qty, demand_mean, lead_time)

        # to_native: 将 numpy 标量(int64列的category、np.sqrt/clip产出的float64等)
        # 统一转为 Python 原生类型,避免 FastAPI jsonable_encoder 序列化失败返回 500
        return to_native({
            "sku_id": sku_id,
            "sku_name": sku_name,
            "category": category,
            "unit": unit,
            "demand_mean": round(demand_mean, 1),
            "demand_std": round(demand_std, 1),
            "period_demand": round(period_mean, 1),
            "current_inventory": round(current_inventory, 1),
            "in_transit": round(in_transit, 1),
            "optimal_stock_level": round(optimal_stock, 1),
            "reorder_qty": int(round(reorder_qty)),
            "service_level": round(service_level * 100, 1),
            "critical_ratio": round(adjusted_cr * 100, 1),
            "underage_cost": round(Cu, 2),
            "overage_cost": round(Co, 2),
            "expected_stockout_loss": round(expected_stockout_loss, 2),
            "expected_overstock_loss": round(expected_overstock_loss, 2),
            "total_expected_loss": round(total_expected_loss, 2),
            "waste_risk_level": waste_risk,
            "shelf_life": int(shelf_life),
            "lead_time": int(lead_time),
            "price": round(price, 2),
            "cost": round(cost, 2),
            "param_level": fb["level"],
            "param_level_label": fb["level_label"],
        })

    @staticmethod
    def _waste_risk_level(shelf_life: float, cv: float, reorder_qty: float,
                          demand_mean: float, lead_time: float) -> str:
        """判定损耗风险等级"""
        risk_score = 0
        if shelf_life <= 3:
            risk_score += 3
        elif shelf_life <= 5:
            risk_score += 2
        elif shelf_life <= 7:
            risk_score += 1
        if cv > 0.4:
            risk_score += 2
        elif cv > 0.25:
            risk_score += 1
        if demand_mean > 0 and reorder_qty / max(demand_mean * max(lead_time, 1), 1) > 1.5:
            risk_score += 1
        if risk_score >= 5:
            return "高"
        elif risk_score >= 3:
            return "中"
        return "低"

    # ──────────── 全量补货清单 ────────────
    def generate_replenishment_list(self, demand_df: Optional[pd.DataFrame] = None,
                                    forecast_days: int = 1) -> list:
        """
        生成所有SKU的补货清单
        Args:
            demand_df: 需求恢复后的DataFrame(含 recovered_demand 列)
            forecast_days: 预测天数
        """
        df = demand_df if demand_df is not None else self.df
        # 预计算 self.df 的 SKU 元数据分组(optimize_sku 原本每个SKU都全表扫描一次),
        # 循环外一次性建好索引,避免并行任务内重复扫描全表
        # 注:不用 dict(groupby) —— pandas 3.x 的 GroupBy 带有非可调用的 keys 属性,
        #     会触发 dict() 的 mapping 协议分支报错,改用推导式走纯迭代
        grp_meta_by_sku = {sid: g for sid, g in self.df.groupby("sku_id")}
        empty_meta = self.df.iloc[0:0]  # demand_df 中存在而 self.df 中没有的SKU的兜底空组
        groups = list(df.groupby("sku_id"))
        # 并行计算各SKU:_replenish_single_sku 只读 grp/预计算分组,不写共享状态,线程安全
        outcomes = parallel_map(
            lambda t: self._replenish_single_sku(t[0], t[1], grp_meta_by_sku, empty_meta),
            groups,
            desc="generate_replenishment_list",
        )
        # 主线程合并结果,过滤异常返回(None 表示该SKU计算失败)
        results = [r for r in outcomes if r is not None]
        # 按建议补货量降序排列
        results.sort(key=lambda x: x["reorder_qty"], reverse=True)
        return results

    def _replenish_single_sku(self, sku_id: str, grp: pd.DataFrame,
                              grp_meta_by_sku: dict, empty_meta: pd.DataFrame) -> Optional[dict]:
        """generate_replenishment_list 的单SKU并行任务:
        从需求侧分组 grp 计算需求参数与库存,再调用 _optimize_sku_with_grp 求最优补货量。
        只读输入、不写共享状态,结果通过返回值传回;异常时返回 None 由主线程过滤。"""
        try:
            # 需求参数(三级回退: SKU级用自身历史,品类级/全局级用回退参数)
            fb = self._fallback.get_params(sku_id)
            if fb["level"] == "sku":
                # SKU级: 用自己的历史数据
                recent = grp.tail(30)
                if "recovered_demand" in recent.columns:
                    demand_mean = float(recent["recovered_demand"].mean())
                    demand_std = float(recent["recovered_demand"].std())
                else:
                    demand_mean = float(recent["sales"].mean())
                    demand_std = float(recent["sales"].std())
                if np.isnan(demand_mean):
                    demand_mean = fb["demand_mean"]
                if np.isnan(demand_std) or demand_std < 0.5:
                    demand_std = max(demand_mean * 0.2, 0.5)
            else:
                # 品类级/全局级: 用回退参数
                demand_mean = fb["demand_mean"]
                demand_std = fb["demand_std"]
                if demand_std < 0.5:
                    demand_std = max(demand_mean * 0.2, 0.5)

            # 当前库存
            current_inv = float(grp["inventory"].iloc[-1]) if "inventory" in grp.columns and grp["inventory"].notna().any() else 0
            in_transit = float(grp["in_transit"].iloc[-1]) if "in_transit" in grp.columns else 0

            # SKU 元数据取预计算分组(与 optimize_sku 内 self.df 过滤等价)
            grp_meta = grp_meta_by_sku.get(sku_id, empty_meta)
            return self._optimize_sku_with_grp(sku_id, grp_meta, demand_mean, demand_std,
                                               current_inv, in_transit)
        except Exception:
            return None