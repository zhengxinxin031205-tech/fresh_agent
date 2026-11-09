"""
损耗归因与临期优化
==================
1. 损耗归因:拆解历史损耗的构成,量化订货过量、需求波动、天气异常、
   促销不及预期、保质期偏短等因素的贡献占比
2. 临期折扣优化:基于价格因果弹性计算最优折扣力度,
   保证"减少损耗的收益 > 折扣造成的毛利损失"
"""
import numpy as np
import pandas as pd
from typing import Optional
from backend.core.np_utils import to_native


class WasteAnalyzer:
    """损耗归因与临期优化引擎"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.has_waste_data = "waste_qty" in df.columns and df["waste_qty"].notna().any()
        self.has_cost_data = "cost" in df.columns and df["cost"].notna().any()

    # ──────────── 损耗概览 ────────────
    def get_overview(self, period: str = "month") -> dict:
        """获取损耗概览:总损耗金额、损耗率走势"""
        df = self.df
        if not self.has_waste_data:
            return self._estimate_waste_overview()

        df = df.copy()
        df["waste_cost"] = df["waste_qty"].fillna(0) * df["cost"].fillna(df.get("price", 5) * 0.5)
        df["sales_revenue"] = df["sales"] * df.get("price", pd.Series([1] * len(df)))

        # 按周期聚合
        if "date" in df.columns:
            df["period"] = df["date"].dt.to_period("W" if period == "week" else "M").astype(str)
        else:
            df["period"] = "all"

        trend = []
        for per, grp in df.groupby("period"):
            total_waste_cost = float(grp["waste_cost"].sum())
            total_revenue = float(grp["sales_revenue"].sum())
            total_waste_qty = int(grp["waste_qty"].sum())
            trend.append({
                "period": per,
                "waste_cost": round(total_waste_cost, 2),
                "waste_qty": total_waste_qty,
                "waste_rate_pct": round(total_waste_cost / max(total_revenue, 1) * 100, 2),
            })

        # 各商品每日损耗走势(取总损耗最多的前8个商品,避免图表过乱)
        sku_trends = []
        if "date" in df.columns and "sku_id" in df.columns:
            waste_df = df[df["waste_qty"].fillna(0) > 0]
            if len(waste_df) > 0:
                top_sku_ids = (waste_df.groupby("sku_id")["waste_qty"].sum()
                               .sort_values(ascending=False).head(8).index.tolist())
                for sid in top_sku_ids:
                    grp = waste_df[waste_df["sku_id"] == sid]
                    sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sid
                    daily_rows = (grp.groupby(grp["date"].dt.strftime("%Y-%m-%d"))["waste_qty"]
                                  .sum().reset_index())
                    daily = [{"date": str(r["date"]), "waste_qty": int(r["waste_qty"])}
                             for _, r in daily_rows.iterrows()]
                    sku_trends.append({"sku_id": sid, "sku_name": sku_name, "daily": daily})

        total_waste = float(df["waste_cost"].sum())
        total_rev = float(df["sales_revenue"].sum())
        return {
            "total_waste_cost": round(total_waste, 2),
            "total_waste_qty": int(df["waste_qty"].sum()),
            "overall_waste_rate_pct": round(total_waste / max(total_rev, 1) * 100, 2),
            "trend": trend,
            "sku_trends": sku_trends,
            "n_skus_with_waste": int(df[df["waste_qty"] > 0]["sku_id"].nunique()),
        }

    def _estimate_waste_overview(self) -> dict:
        """基础版/进阶版:基于保质期和过量库存估算损耗"""
        df = self.df
        estimates = []
        for sku_id, grp in df.groupby("sku_id"):
            shelf = float(grp["shelf_life"].iloc[0]) if "shelf_life" in grp.columns and grp["shelf_life"].notna().any() else 5
            base_rate = {3: 0.06, 5: 0.04, 7: 0.025, 15: 0.01, 20: 0.008}.get(int(shelf), 0.04)
            recent = grp.tail(14)
            avg_sales = float(recent["sales"].mean()) if len(recent) > 0 else 1
            avg_inv = float(recent["inventory"].mean()) if "inventory" in recent.columns else avg_sales * 1.5
            overstock = max(0, avg_inv - avg_sales * 1.2)
            est_waste_qty = int(round(avg_inv * base_rate + overstock * 0.05))
            price = float(grp["price"].mean()) if "price" in grp.columns else 5.0
            cost = price * 0.5
            estimates.append({"sku_id": sku_id, "est_waste_qty": est_waste_qty,
                              "est_waste_cost": round(est_waste_qty * cost, 2)})
        total_qty = sum(e["est_waste_qty"] for e in estimates)
        total_cost = sum(e["est_waste_cost"] for e in estimates)
        return {
            "total_waste_cost": round(total_cost, 2),
            "total_waste_qty": total_qty,
            "overall_waste_rate_pct": 0.0,
            "trend": [],
            "sku_trends": [],
            "estimated": True,
            "note": "当前数据不含损耗记录,以上为基于保质期与库存水平的估算值",
        }

    # ──────────── 损耗归因分析 ────────────
    def attribute_waste(self, sku_id: Optional[str] = None) -> dict:
        """
        拆解损耗构成,量化各因素贡献占比:
        - 订货过量:库存/销量比持续偏高
        - 需求波动:销量变异系数大
        - 天气异常:大雨天销量骤降
        - 促销不及预期:促销日销量未达预期
        - 保质期偏短:保质期≤3天的SKU固有损耗
        - 随机残差:不可解释部分
        """
        df = self.df
        if not self.has_waste_data:
            return {"error": "损耗归因分析需要完整版数据(含损耗记录与成本字段),请上传损耗数据或切换完整版示例数据集"}

        target_skus = [sku_id] if sku_id else list(df["sku_id"].unique())
        results = {}

        for sid in target_skus:
            grp = df[df["sku_id"] == sid].copy()
            if len(grp) < 5 or grp["waste_qty"].sum() == 0:
                continue
            sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sid
            total_waste = float(grp["waste_qty"].sum())

            # 因素1:订货过量(库存/销量比 > 1.5 的天数的损耗贡献)
            if "inventory" in grp.columns:
                grp["inv_sales_ratio"] = grp["inventory"] / grp["sales"].clip(lower=0.5)
                overstock_mask = grp["inv_sales_ratio"] > 1.5
                overstock_waste = float(grp.loc[overstock_mask, "waste_qty"].sum())
            else:
                overstock_waste = total_waste * 0.2

            # 因素2:需求波动(销量低于均值-1σ的天数的损耗贡献)
            sales_mean = grp["sales"].mean()
            sales_std = grp["sales"].std()
            low_demand_mask = grp["sales"] < (sales_mean - sales_std)
            volatility_waste = float(grp.loc[low_demand_mask, "waste_qty"].sum())

            # 因素3:天气异常(大雨天)
            if "precipitation" in grp.columns:
                rain_mask = grp["precipitation"].fillna(0) > 5
                rain_waste = float(grp.loc[rain_mask, "waste_qty"].sum())
            else:
                rain_waste = 0

            # 因素4:促销不及预期(促销日销量 < 非促销日均值)
            if "is_promo" in grp.columns and grp["is_promo"].sum() > 0:
                non_promo_mean = grp.loc[grp["is_promo"] == 0, "sales"].mean()
                bad_promo_mask = (grp["is_promo"] == 1) & (grp["sales"] < non_promo_mean)
                promo_waste = float(grp.loc[bad_promo_mask, "waste_qty"].sum())
            else:
                promo_waste = 0

            # 因素5:保质期偏短(保质期≤3天的固有损耗基线)
            shelf = float(grp["shelf_life"].iloc[0]) if "shelf_life" in grp.columns else 7
            shelf_baseline_rate = max(0, (7 - shelf) / 7) * 0.15  # 保质期越短基线越高
            shelf_waste = total_waste * shelf_baseline_rate

            # 因素6:随机残差(扣除以上后)
            explained = overstock_waste + volatility_waste + rain_waste + promo_waste + shelf_waste
            residual = max(0, total_waste - explained)

            # 归一化为占比(防止重叠计算导致超100%)
            raw = {
                "订货过量": overstock_waste,
                "需求波动": volatility_waste,
                "天气异常": rain_waste,
                "促销不及预期": promo_waste,
                "保质期偏短": shelf_waste,
                "随机残差": residual,
            }
            total_raw = sum(raw.values()) + 1e-6
            attribution = {k: round(v / total_raw * 100, 1) for k, v in raw.items()}

            # 成本
            cost = float(grp["cost"].iloc[0]) if "cost" in grp.columns else 3.0
            results[sid] = {
                "sku_name": sku_name,
                "total_waste_qty": int(total_waste),
                "total_waste_cost": round(total_waste * cost, 2),
                "attribution": attribution,
                "shelf_life": int(shelf),
            }

        return {"sku_attributions": results, "has_data": True}

    # ──────────── Top高损耗SKU ────────────
    def top_waste_skus(self, top_n: int = 5) -> list:
        """获取损耗最高的SKU排名及优化建议"""
        if not self.has_waste_data:
            overview = self._estimate_waste_overview()
            return []
        df = self.df
        result = []
        for sku_id, grp in df.groupby("sku_id"):
            total_waste = float(grp["waste_qty"].sum())
            cost = float(grp["cost"].iloc[0]) if "cost" in grp.columns else 3.0
            price = float(grp["price"].mean()) if "price" in grp.columns else 5.0
            shelf = int(grp["shelf_life"].iloc[0]) if "shelf_life" in grp.columns else 7
            sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sku_id
            category = grp["category"].iloc[0] if "category" in grp.columns else ""
            waste_cost = total_waste * cost
            waste_rate = total_waste / max(float(grp["sales"].sum()), 1) * 100
            # 生成针对性建议
            suggestions = self._generate_suggestions(grp, shelf, waste_rate)
            result.append({
                "sku_id": sku_id, "sku_name": sku_name, "category": category,
                "total_waste_qty": int(total_waste), "waste_cost": round(waste_cost, 2),
                "waste_rate_pct": round(waste_rate, 1), "shelf_life": shelf,
                "suggestions": suggestions,
            })
        result.sort(key=lambda x: x["waste_cost"], reverse=True)
        return to_native(result[:top_n])

    @staticmethod
    def _generate_suggestions(grp: pd.DataFrame, shelf: int, waste_rate: float) -> list:
        """生成针对性优化建议"""
        suggestions = []
        if shelf <= 3:
            suggestions.append("保质期极短,建议采用「小批量高频次」补货策略,单次补货量不超过日均需求的1.2倍")
        if "inventory" in grp.columns:
            avg_ratio = float((grp["inventory"] / grp["sales"].clip(lower=0.5)).mean())
            if avg_ratio > 2.0:
                suggestions.append(f"库存/销量比达{avg_ratio:.1f},订货明显过量,建议下调补货量15-25%")
        cv = float(grp["sales"].std() / max(grp["sales"].mean(), 1))
        if cv > 0.35:
            suggestions.append(f"需求波动率高(CV={cv:.2f}),建议结合天气预报动态调整订货量")
        if "is_promo" in grp.columns and grp["is_promo"].sum() > 0:
            promo_waste = float(grp.loc[grp["is_promo"] == 1, "waste_qty"].sum()) if "waste_qty" in grp.columns else 0
            if promo_waste > 0:
                suggestions.append("促销期间存在损耗,建议促销前精准预测增量,避免促销结束后积压")
        if waste_rate > 10:
            suggestions.append("损耗率超10%,建议尽快启用临期折扣机制,在过期前24-48小时启动阶梯降价")
        if not suggestions:
            suggestions.append("损耗处于可控范围,维持当前补货策略并持续监控")
        return suggestions

    # ──────────── 临期折扣优化 ────────────
    def optimize_discount(self, sku_id: str, near_expiry_qty: float,
                          elasticity: float = -1.5) -> dict:
        """
        基于价格因果弹性计算最优临期折扣力度
        目标: max( 折扣带来的额外销量收益 + 减少损耗的节省 - 折扣毛利损失 )

        模型: 折扣 d ∈ [0.3, 0.9](售价比例)
          额外销量 = Q_base * (d^elasticity - 1)  (弹性效应)
          折扣损失 = Q_base * price * (1 - d)      (原有销量的毛利损失)
          损耗节省 = min(额外销量, 临期数量) * cost  (卖出去就不用报废)
        """
        grp = self.df[self.df["sku_id"] == sku_id]
        if len(grp) == 0:
            return {"error": f"SKU {sku_id} 不存在"}
        sku_name = grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sku_id
        price = float(grp["price"].mean()) if "price" in grp.columns else 5.0
        cost = float(grp["cost"].iloc[0]) if "cost" in grp.columns else price * 0.5
        base_daily_sales = float(grp["sales"].tail(14).mean())
        shelf = int(grp["shelf_life"].iloc[0]) if "shelf_life" in grp.columns else 5

        # 网格搜索最优折扣
        best = {"discount": 1.0, "net_benefit": 0, "extra_sales": 0, "waste_saved": 0}
        for d in np.arange(0.3, 0.95, 0.05):
            d = round(float(d), 2)
            # 弹性效应:折扣越深,需求增幅越大
            demand_mult = d ** elasticity
            extra_sales = base_daily_sales * (demand_mult - 1)
            extra_sales = max(0, extra_sales)
            # 能卖出去的临期品 = min(额外需求, 临期数量)
            waste_saved = min(extra_sales, near_expiry_qty)
            # 折扣导致的原有销量毛利损失
            discount_loss = base_daily_sales * price * (1 - d) * 0.3  # 只有部分原价销量受影响
            # 净收益 = 损耗节省(避免报废的成本) + 额外收入 - 折扣损失
            extra_revenue = waste_saved * price * d
            net_benefit = waste_saved * cost + extra_revenue - discount_loss
            if net_benefit > best["net_benefit"]:
                best = {
                    "discount": d, "net_benefit": round(net_benefit, 2),
                    "extra_sales": round(extra_sales, 1),
                    "waste_saved": round(waste_saved, 1),
                    "discount_loss": round(discount_loss, 2),
                    "extra_revenue": round(extra_revenue, 2),
                }

        # 不打折的基线:全部临期品报废
        baseline_loss = near_expiry_qty * cost
        return {
            "sku_id": sku_id, "sku_name": sku_name,
            "near_expiry_qty": near_expiry_qty,
            "base_price": round(price, 2),
            "cost": round(cost, 2),
            "shelf_life": shelf,
            "price_elasticity": elasticity,
            "optimal_discount": best["discount"],
            "discounted_price": round(price * best["discount"], 2),
            "expected_extra_sales": best["extra_sales"],
            "waste_saved_qty": best["waste_saved"],
            "net_benefit": round(best["net_benefit"] - baseline_loss + baseline_loss, 2),
            "baseline_waste_loss": round(baseline_loss, 2),
            "benefit_vs_no_discount": round(best["net_benefit"] + baseline_loss - baseline_loss, 2),
            "recommendation": (
                f"建议对 {sku_name} 的 {int(near_expiry_qty)} 件临期商品执行 {best['discount']*100:.0f} 折 "
                f"(折后价 {price * best['discount']:.1f} 元),"
                f"预计可多售出 {best['extra_sales']:.0f} 件,避免 {best['waste_saved']:.0f} 件报废,"
                f"净收益比不打折多 {best['net_benefit']:.1f} 元。"
            ),
        }