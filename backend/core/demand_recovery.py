"""
缺货截断下的真实需求恢复
========================
核心问题:历史销量 ≠ 真实需求。当商品卖断货时,销量被库存截断,
传统预测会系统性低估需求。

算法方案:
1. 缺货时段识别:销量=0且库存=0,或销量触顶(达到库存上限)
2. 截尾回归(Tobit思想):对非截断样本拟合需求模型,推断截断处的潜在需求
3. 因果校正:利用同期同品类非缺货SKU的关联关系,校正缺货日的需求估计
4. 输出:校正后的需求分布(均值+标准差),修正系统性低估偏差
"""
import numpy as np
import pandas as pd
from typing import Optional


class DemandRecovery:
    """缺货截断下的真实需求恢复引擎"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.recovered_df: Optional[pd.DataFrame] = None
        self.stockout_stats: dict = {}

    # ──────────── 缺货检测 ────────────
    def detect_stockout(self) -> pd.DataFrame:
        """
        识别缺货/截断时段:
        - 显式标记:stockout 列 = 1
        - 隐式推断:销量=0 且 (库存=0 或 同SKU其他日销量显著>0)
        - 触顶检测:销量 >= 库存(卖光了当天所有库存)
        """
        df = self.df
        if "stockout" not in df.columns:
            df["stockout"] = 0

        # 向量化:一次性计算各SKU均值,推断隐式缺货与触顶(替代逐SKU循环)
        sku_mean = df.groupby("sku_id")["sales"].transform("mean")
        implicit_zero = (df["sales"] == 0) & (sku_mean > 5)
        if "inventory" in df.columns:
            capped = (df["sales"] >= df["inventory"]) & (df["inventory"] > 0)
        else:
            capped = pd.Series(False, index=df.index)
        df["stockout"] = (
            df["stockout"].astype(int)
            | implicit_zero.astype(int)
            | capped.astype(int)
        )

        # 统计
        total = len(df)
        n_stockout = int(df["stockout"].sum())
        self.stockout_stats = {
            "total_records": total,
            "stockout_records": n_stockout,
            "stockout_rate_pct": round(n_stockout / max(1, total) * 100, 2),
            "affected_skus": int(df[df["stockout"] == 1]["sku_id"].nunique()) if "sku_id" in df.columns else 0,
        }
        return df

    # ──────────── 截尾回归需求恢复 ────────────
    def recover(self) -> pd.DataFrame:
        """
        执行需求恢复:
        1. 对每个SKU,用非截断日数据建立基准需求模型(均值+波动)
        2. 对截断日,用同期同品类非缺货SKU的销量变化比率进行因果校正
        3. 融合基准估计与因果校正,输出恢复后的需求
        """
        df = self.detect_stockout()
        df["recovered_demand"] = df["sales"].astype(float)
        df["demand_mean"] = np.nan
        df["demand_std"] = np.nan
        df["correction_applied"] = 0

        has_category = "category" in df.columns
        has_date = "date" in df.columns
        nc_mask = df["stockout"] == 0
        censored_mask = df["stockout"] == 1

        # ── 1. 向量化计算各SKU基准需求(均值+波动),替代逐SKU循环 ──
        nc_stats = df.loc[nc_mask].groupby("sku_id")["sales"].agg(["mean", "std", "count"])
        all_stats = df.groupby("sku_id")["sales"].agg(["mean", "std"])

        nc_mean = df["sku_id"].map(nc_stats["mean"])
        nc_std = df["sku_id"].map(nc_stats["std"])
        nc_count = df["sku_id"].map(nc_stats["count"]).fillna(0)
        all_mean = df["sku_id"].map(all_stats["mean"])
        all_std = df["sku_id"].map(all_stats["std"])

        enough = nc_count >= 3
        mu = np.where(enough, nc_mean, all_mean)
        sigma_nc = np.maximum(nc_std, np.maximum(nc_mean * 0.15, 1.0))
        sigma_all = np.maximum(all_std, np.maximum(all_mean * 0.2, 1.0))
        sigma = np.where(enough, sigma_nc, sigma_all)
        df["demand_mean"] = mu
        df["demand_std"] = sigma

        # ── 2. 向量化因果校正:同期同品类非缺货SKU的销量比率,替代 iterrows 嵌套循环 ──
        # 原逻辑:对每个缺货日,遍历同天同品类非缺货SKU,计算"当天销量/自身历史均值"
        # 并取均值。向量化等价:按(date,category)聚合全部比率,再减去自身SKU的贡献。
        correction_ratio = pd.Series(1.0, index=df.index)
        if has_category and has_date:
            sku_nc_mean = df["sku_id"].map(nc_stats["mean"])
            sku_nc_count = df["sku_id"].map(nc_stats["count"]).fillna(0)

            # 仅非缺货行且SKU历史有效(count>=3 且 mean>0)时,比率才有意义
            ratio_valid = nc_mask & (sku_nc_count >= 3) & (sku_nc_mean > 0)
            safe_mean = sku_nc_mean.fillna(1.0).replace(0.0, 1.0)
            ratio_val = np.where(ratio_valid, df["sales"] / safe_mean, 0.0)

            # 辅助DataFrame承载分组统计量(缺货行贡献为0),不污染输出
            _g = pd.DataFrame(index=df.index)
            _g["nc"] = nc_mask.astype(int)                 # 非缺货行计数
            _g["rsum"] = np.where(nc_mask, ratio_val, 0.0)  # 有效比率之和
            _g["rcnt"] = ratio_valid.astype(int)           # 有效比率个数

            # 按(date, category)分组: 同期同品类的全部非缺货贡献
            dc_key = [df["date"], df["category"]]
            dc_cnt = _g.groupby(dc_key)["nc"].transform("sum")
            dc_rsum = _g.groupby(dc_key)["rsum"].transform("sum")
            dc_rcnt = _g.groupby(dc_key)["rcnt"].transform("sum")

            # 按(date, category, sku_id)分组: 当前SKU自身在同组的贡献(需排除)
            dcs_key = [df["date"], df["category"], df["sku_id"]]
            dcs_cnt = _g.groupby(dcs_key)["nc"].transform("sum")
            dcs_rsum = _g.groupby(dcs_key)["rsum"].transform("sum")
            dcs_rcnt = _g.groupby(dcs_key)["rcnt"].transform("sum")

            # 排除自身SKU后,同期同品类非缺货SKU的比率统计
            same_day_cnt = (dc_cnt - dcs_cnt).fillna(0)
            ratios_cnt = (dc_rcnt - dcs_rcnt).fillna(0)
            ratios_sum = (dc_rsum - dcs_rsum).fillna(0)

            # len(same_day)>=2 且 ratios非空 时应用校正
            has_corr = (same_day_cnt >= 2) & (ratios_cnt > 0)
            mean_ratio = ratios_sum / ratios_cnt.replace(0.0, 1.0)
            correction_ratio = pd.Series(
                np.where(has_corr, np.clip(mean_ratio, 0.5, 2.0), 1.0),
                index=df.index,
            )

        # ── 3. 向量化周末/节假日校正 ──
        if "is_weekend" in df.columns:
            nc_df = df.loc[nc_mask]
            wknd_stats = nc_df[nc_df["is_weekend"] == 1].groupby("sku_id")["sales"].agg(["mean", "count"])
            wkdy_stats = nc_df[nc_df["is_weekend"] == 0].groupby("sku_id")["sales"].agg(["mean", "count"])

            wknd_mean = df["sku_id"].map(wknd_stats["mean"]).fillna(0.0)
            wknd_count = df["sku_id"].map(wknd_stats["count"]).fillna(0)
            wkdy_mean = df["sku_id"].map(wkdy_stats["mean"]).fillna(0.0)
            wkdy_count = df["sku_id"].map(wkdy_stats["count"]).fillna(0)

            weekend_ok = (wknd_count >= 2) & (wkdy_count >= 2)
            weekend_ratio = wknd_mean / np.maximum(wkdy_mean, 0.1)
            weekend_factor = np.where(weekend_ok, np.clip(weekend_ratio, 0.7, 1.5), 1.0)

            # 仅对 is_weekend==1 的截断行应用
            weekend_apply = censored_mask & (df["is_weekend"] == 1)
            correction_ratio = correction_ratio * np.where(weekend_apply, weekend_factor, 1.0)

        # ── 4. 融合基准与校正,输出恢复后的需求 ──
        base = np.asarray(df["demand_mean"])
        sales_f = df["sales"].to_numpy(dtype=float)
        cens_np = censored_mask.to_numpy()
        recovered = base * np.asarray(correction_ratio)
        # 截尾回归逻辑:观测销量>0(触顶型)时,取 max(恢复值, 观测*1.15)
        recovered = np.where(
            cens_np & (sales_f > 0),
            np.maximum(recovered, sales_f * 1.15),
            recovered,
        )
        df.loc[censored_mask, "recovered_demand"] = np.round(recovered[cens_np], 1)
        df.loc[censored_mask, "correction_applied"] = 1

        # 非截断日:恢复需求 = 销量(可能有小幅校正)
        df.loc[~censored_mask, "recovered_demand"] = df.loc[~censored_mask, "sales"].astype(float)

        self.recovered_df = df
        return df

    # ──────────── 结果摘要 ────────────
    def get_summary(self) -> dict:
        """获取需求恢复的统计摘要"""
        if self.recovered_df is None:
            self.recover()
        df = self.recovered_df
        corrected = df[df["correction_applied"] == 1]
        total_underestimate = 0.0
        if len(corrected) > 0:
            total_underestimate = float((corrected["recovered_demand"] - corrected["sales"]).sum())
        return {
            "stockout_stats": self.stockout_stats,
            "corrected_records": int(len(corrected)),
            "total_recovered_demand": round(total_underestimate, 1),
            "avg_uplift_pct": round(
                float(((corrected["recovered_demand"] / corrected["sales"].clip(lower=0.1) - 1) * 100).mean())
                if len(corrected) > 0 else 0, 1
            ),
        }

    # ──────────── 获取SKU级别的恢复需求分布 ────────────
    def get_sku_demand_distribution(self, sku_id: str) -> dict:
        """获取指定SKU恢复后的需求分布参数"""
        if self.recovered_df is None:
            self.recover()
        df = self.recovered_df
        grp = df[df["sku_id"] == sku_id]
        if len(grp) == 0:
            return {"sku_id": sku_id, "mean": 0, "std": 1, "n_stockout": 0}
        recent = grp.tail(30)  # 近30天
        return {
            "sku_id": sku_id,
            "mean": round(float(recent["recovered_demand"].mean()), 1),
            "std": round(float(recent["recovered_demand"].std()), 1),
            "raw_mean": round(float(recent["sales"].mean()), 1),
            "n_stockout": int(grp["stockout"].sum()),
            "uplift_pct": round(
                float((recent["recovered_demand"].mean() / max(recent["sales"].mean(), 0.1) - 1) * 100), 1
            ),
        }