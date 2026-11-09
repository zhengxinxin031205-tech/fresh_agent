"""
因果需求预测
============
基于时间序列特征 + 因果因素回归的混合预测模型:
- 去混淆:训练前剔除促销/节假日效应,防止高估污染基线(规格1)
- 趋势项:近期加权移动平均(基于去效应后的基线需求)
- 周期项:星期几效应
- 时间窗口:历史数据按时间先后划分,最后 horizon 天作为未来窗口验证(规格3)
- 输出:未来N天分SKU的需求预测(均值+置信区间),标注缺货校正前后对比
"""
import numpy as np
import pandas as pd
from typing import Optional
from backend.core.parallel_utils import parallel_map
from backend.core.stat_utils import norm_ppf
from backend.core.np_utils import to_native


class DemandForecaster:
    """因果需求预测器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _deconfound_series(self, df: pd.DataFrame, demand_col: str) -> tuple:
        """从需求序列中剔除促销/节假日等因果效应,返回 (基线需求ndarray, 效应摘要dict)
        
        混淆未剔除时,促销期的需求高估会污染预测训练集(EWMA基准被促销拉高)。
        用 OLS 估计促销/节假日系数,从观测需求中减去,得到"无干预基线需求"。
        """
        demand = df[demand_col].astype(float).values
        effects_summary = {}

        factors = []
        names = []
        if "is_promo" in df.columns and df["is_promo"].nunique() > 1:
            factors.append(df["is_promo"].astype(float).values)
            names.append("promo")
        if "is_holiday" in df.columns and df["is_holiday"].sum() > 0:
            factors.append(df["is_holiday"].astype(float).values)
            names.append("holiday")

        if not factors or len(demand) < 5:
            return demand, effects_summary

        X = np.column_stack(factors)
        try:
            X_const = np.column_stack([np.ones(len(demand)), X])
            coef, *_ = np.linalg.lstsq(X_const, demand, rcond=None)
            baseline = demand - X @ coef[1:]
            baseline = np.maximum(baseline, 0)
            for i, name in enumerate(names):
                effects_summary[name] = round(float(coef[i + 1]), 1)
            return baseline, effects_summary
        except Exception:
            return demand, effects_summary

    def _validate_holdout(self, series: pd.DataFrame, baseline: np.ndarray, horizon: int) -> dict:
        """时间窗口划分验证:历史数据按时间先后划分,最后 horizon 天作为"未来窗口"不参与训练,
        用之前数据训练 EWMA,对未来窗口预测,计算 MAPE 评估模型质量。"""
        if "date" not in series.columns or len(series) < horizon * 3:
            return {}

        n = len(series)
        train_n = n - horizon
        train_base = baseline[:train_n]
        valid_base = baseline[train_n:]
        valid_dates = series["date"].iloc[train_n:].reset_index(drop=True)

        if len(train_base) < 3:
            return {}

        # 训练窗口 EWMA 基准(与 forecast_sku 同款权重)
        weights = np.exp(np.linspace(-2, 0, len(train_base[-21:])))
        recent = train_base[-21:]
        base = float(np.average(recent, weights=weights[-len(recent):]))

        # 训练窗口星期几效应
        train_dows = series["date"].iloc[:train_n].dt.dayofweek.values
        overall = float(np.mean(train_base))
        dow_map = {}
        for d in range(7):
            mask = train_dows == d
            if mask.sum() >= 2:
                dow_map[d] = float(np.mean(train_base[mask]) / max(overall, 0.1))
            else:
                dow_map[d] = 1.0

        # 对未来窗口预测并计算误差
        last_train_date = series["date"].iloc[train_n - 1]
        errors = []
        for h in range(horizon):
            f_date = last_train_date + pd.Timedelta(days=h + 1)
            dow = f_date.dayofweek
            pred = max(base * dow_map.get(dow, 1.0), 0)
            match = valid_dates == f_date
            if match.any():
                actual = float(valid_base[match.values][0])
                if actual > 0:
                    errors.append(abs(pred - actual) / actual)

        if not errors:
            return {}
        mape = float(np.mean(errors) * 100)
        return {
            "validation_mape_pct": round(mape, 1),
            "validation_window_days": horizon,
            "train_window_days": train_n,
        }

    def forecast_sku(self, sku_id: str, horizon: int = 7,
                     recovered_df: Optional[pd.DataFrame] = None) -> dict:
        """
        预测指定SKU未来 horizon 天的需求
        Args:
            recovered_df: 需求恢复后的数据(含 recovered_demand 列)
        """
        src = recovered_df if recovered_df is not None else self.df
        grp = src[src["sku_id"] == sku_id]
        if len(grp) == 0:
            return {"error": f"SKU {sku_id} 不存在"}

        sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sku_id
        category = grp["category"].iloc[0] if "category" in grp.columns else ""
        unit = grp["unit"].iloc[0] if "unit" in grp.columns else "件"

        # 使用恢复后的需求(若有)
        demand_col = "recovered_demand" if "recovered_demand" in grp.columns else "sales"
        _cols = ["date", demand_col, "sales"] if "date" in grp.columns else [demand_col, "sales"]
        series = grp[list(dict.fromkeys(_cols))].copy()  # 去重(避免 demand_col=="sales" 时列重复)
        series = series.dropna(subset=[demand_col])

        # ── 因果去混淆:剔除促销/节假日效应,防止污染训练集 ──
        for aux_col in ["is_promo", "is_holiday"]:
            if aux_col in grp.columns:
                series[aux_col] = grp.loc[series.index, aux_col].values
        baseline_values, deconfound_effects = self._deconfound_series(series, demand_col)
        values = baseline_values

        # ── 基准:指数加权移动平均 ──
        if len(values) < 3:
            base_level = float(np.mean(values)) if len(values) > 0 else 1.0
        else:
            weights = np.exp(np.linspace(-2, 0, len(values[-21:])))  # 近21天指数权重
            recent = values[-21:]
            base_level = float(np.average(recent, weights=weights[-len(recent):]))

        # ── 星期几效应(基于去效应后的基线需求) ──
        dow_factors = {}
        if "date" in series.columns:
            s_dow = series["date"].dt.dayofweek.values
            overall_mean = float(np.mean(values))
            for d in range(7):
                mask = s_dow == d
                if mask.sum() >= 2:
                    dow_factors[d] = float(np.mean(values[mask]) / max(overall_mean, 0.1))
                else:
                    dow_factors[d] = 1.0
        else:
            dow_factors = {d: 1.0 for d in range(7)}

        # ── 波动率 ──
        if len(values) >= 5:
            residual_std = float(np.std(values[-14:] - np.mean(values[-14:])))
            demand_std = max(residual_std, base_level * 0.15, 0.5)
        else:
            demand_std = max(base_level * 0.2, 0.5)

        # ── 经验残差分位数(q10/q50/q90) ──
        n_res = min(21, len(values))
        recent_vals = values[-n_res:].astype(float)
        # 滚动EWMA拟合:对每天计算截至当天的指数加权均值(与base_level同款权重)
        ewma_fitted = np.empty(n_res)
        for i in range(n_res):
            w = np.exp(np.linspace(-2, 0, i + 1))
            ewma_fitted[i] = np.average(recent_vals[:i + 1], weights=w)
        residuals = recent_vals - ewma_fitted
        residuals = residuals[~np.isnan(residuals)]
        if len(residuals) >= 5:
            q10_res = float(np.percentile(residuals, 10))
            q50_res = float(np.percentile(residuals, 50))
            q90_res = float(np.percentile(residuals, 90))
            quantile_method = "empirical_residual"
        else:
            # 残差样本不足,回退到正态近似
            q10_res = norm_ppf(0.1) * demand_std
            q50_res = norm_ppf(0.5) * demand_std
            q90_res = norm_ppf(0.9) * demand_std
            quantile_method = "normal_approximation"

        # ── 时间窗口划分验证(最后 horizon 天作为未来窗口,不参与训练) ──
        validation = self._validate_holdout(series, values, horizon)

        # ── 生成预测 ──
        last_date = grp["date"].max() if "date" in grp.columns else pd.Timestamp.now()
        forecasts = []
        for h in range(horizon):
            f_date = last_date + pd.Timedelta(days=h + 1)
            dow = f_date.dayofweek
            point = base_level * dow_factors.get(dow, 1.0)
            point = max(point, 0)
            # 置信区间随预测步长扩大
            ci_width = demand_std * (1 + 0.15 * h)
            # 分位数预测:残差分位数 × 随步长扩大的缩放因子
            scale = 1 + 0.15 * h
            q10 = max(0, point + q10_res * scale)
            q50 = max(0, point + q50_res * scale)
            q90 = max(0, point + q90_res * scale)
            # 保证单调性 q10 <= q50 <= q90
            q50 = max(q10, min(q50, q90))
            forecasts.append({
                "date": f_date.strftime("%Y-%m-%d"),
                "dow": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dow],
                "forecast": round(point, 1),
                "lower": round(max(0, point - 1.96 * ci_width), 1),
                "upper": round(point + 1.96 * ci_width, 1),
                "q10": round(q10, 1),
                "q50": round(q50, 1),
                "q90": round(q90, 1),
            })

        # ── 历史对比数据(校正前 vs 校正后) ──
        history = []
        if "date" in grp.columns:
            recent_hist = grp.tail(30)
            for _, row in recent_hist.iterrows():
                entry = {"date": row["date"].strftime("%Y-%m-%d"), "sales": round(float(row["sales"]), 1)}
                if "recovered_demand" in grp.columns and pd.notna(row.get("recovered_demand")):
                    entry["recovered"] = round(float(row["recovered_demand"]), 1)
                history.append(entry)

        # 缺货校正提升
        uplift_pct = 0.0
        if "recovered_demand" in grp.columns:
            recent = grp.tail(14)
            raw_mean = recent["sales"].mean()
            rec_mean = recent["recovered_demand"].mean()
            if raw_mean > 0:
                uplift_pct = round(float((rec_mean / raw_mean - 1) * 100), 1)

        # to_native: category 等字段可能从 int64 列取出为 numpy.int64,
        # 统一转原生类型避免 FastAPI jsonable_encoder 序列化失败
        return to_native({
            "sku_id": sku_id, "sku_name": sku_name, "category": category, "unit": unit,
            "base_level": round(base_level, 1),
            "demand_std": round(demand_std, 1),
            "dow_factors": {str(k): round(v, 2) for k, v in dow_factors.items()},
            "forecasts": forecasts,
            "history": history,
            "correction_uplift_pct": uplift_pct,
            "total_forecast_qty": round(sum(f["forecast"] for f in forecasts), 1),
            "quantile_method": quantile_method,
            "deconfound_effects": deconfound_effects,
            "validation": validation,
        })

    def forecast_all(self, horizon: int = 7,
                     recovered_df: Optional[pd.DataFrame] = None) -> list:
        """预测所有SKU"""
        src = recovered_df if recovered_df is not None else self.df
        sku_ids = list(src["sku_id"].unique())
        # 并行预测各SKU:forecast_sku 只读 src/self.df,不写共享状态,线程安全
        outcomes = parallel_map(
            lambda sku_id: self.forecast_sku(sku_id, horizon, recovered_df),
            sku_ids,
            desc="forecast_all",
        )
        # 主线程合并结果(过滤数据不足的SKU)
        return [r for r in outcomes if "error" not in r]