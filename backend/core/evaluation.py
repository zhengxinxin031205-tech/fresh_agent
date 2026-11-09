"""
补货策略评估器
==============
对智能补货策略进行量化评估,回答商家最关心的三个问题:

1. 这套补货建议到底能省多少钱?
   与"按昨天销量进货""按历史平均进货"两条朴素基线对比,
   给出损失下降与利润提升百分比;
2. 建议的把握有多大?
   Bootstrap 重采样量化日均损失的 95% 置信区间;
3. 离"完美进货"还有多远?
   与"完美预知"(提前知道真实需求的理论下界)对比差距。

设计要点:
- 不引入新依赖,仅用 numpy/pandas;随机数固定种子(42)保证结果可复现
- Bootstrap 用纯 numpy 向量化:一次性生成 (n_bootstrap, eval_days) 索引矩阵,
  避免 Python 循环嵌套;各 SKU 用 parallel_map 并行
- 成本参数 (Cu/Co) 与 ReplenishmentOptimizer 完全一致,保证评估口径统一
- 所有面向商家的字段用大白话(日均损失/利润提升/满足率),不堆学术术语
"""
import logging

import numpy as np
import pandas as pd
from typing import Optional

from backend.core.replenishment_optimizer import ReplenishmentOptimizer
from backend.core.parallel_utils import parallel_map

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """补货策略评估器:Bootstrap 不确定性量化 + 基线对比"""

    # 默认参数(与 ReplenishmentOptimizer 保持一致,保证评估口径统一)
    DEFAULT_PARAMS = {
        "stockout_customer_loss": 2.0,   # 单次缺货客损估值(元/件,数据无记录时的兜底)
        "salvage_value_ratio": 0.3,      # 残值率:临期处理能回收的采购成本比例
    }

    def __init__(self, df: pd.DataFrame, tier: str = "basic"):
        """
        Args:
            df: 需求恢复后的 DataFrame(含 recovered_demand 列)
            tier: 数据档位 basic/advanced/full
        """
        self.df = df.copy()
        self.tier = tier
        self.params = dict(self.DEFAULT_PARAMS)
        # 内部复用 ReplenishmentOptimizer 计算最优库存水平
        self._optimizer = ReplenishmentOptimizer(df, tier)
        # 复用 optimizer 的三级回退实例,保证 Cu/Co 与需求参数口径完全一致
        self._fallback = self._optimizer._fallback

    # ──────────── 私有工具 ────────────

    def _sorted_df(self) -> pd.DataFrame:
        """返回按 SKU+日期排序的 df,保证回测窗口取到各 SKU 最后 eval_days 天"""
        df = self.df
        if "date" in df.columns:
            return df.sort_values(["sku_id", "date"])
        return df

    @staticmethod
    def _asymmetric_loss(demand: float, stock: float, Cu: float, Co: float) -> float:
        """不对称损失:缺货损失(少赚+客损) + 过剩损失(过期损耗)"""
        shortage = max(0.0, demand - stock)
        overage = max(0.0, stock - demand)
        return shortage * Cu + overage * Co

    def _get_sku_costs(self, sku_id: str, grp: pd.DataFrame) -> tuple:
        """返回 (Cu, Co),通过三级回退取价/成本,与 ReplenishmentOptimizer 口径完全一致"""
        fb = self._fallback.get_params(sku_id)
        price = fb["price"]
        cost = fb["cost"]
        shelf_life = fb["shelf_life"]
        margin = max(price - cost, 0.1)
        stockout_loss = fb.get("stockout_loss", 0.0)
        # 残值率: 临期处理能回收的采购成本比例,无数据默认0.3
        salvage_value = fb.get("salvage_value", cost * self.params["salvage_value_ratio"])
        salvage_value_ratio = salvage_value / cost if cost > 0 else self.params["salvage_value_ratio"]
        # 保质期折减因子: 保质期越短 → 过剩惩罚越大(易腐程度编码进成本函数)
        shelf_factor = np.clip(7.0 / shelf_life, 0.5, 2.0)
        # 过剩单位损失 = 采购成本 × (1 − 残值率) × 保质期折减因子
        Co = max(cost * (1 - salvage_value_ratio) * shelf_factor, 0.1)
        Cu = margin + stockout_loss
        return Cu, Co

    def _get_sku_price_cost(self, sku_id: str, grp: pd.DataFrame) -> tuple:
        """返回 (price, cost),通过三级回退取值,用于利润计算"""
        fb = self._fallback.get_params(sku_id)
        return fb["price"], fb["cost"]

    @staticmethod
    def _demand_stats(hist: np.ndarray) -> tuple:
        """从历史需求序列计算 (均值, 标准差),兜底处理 NaN/过小值"""
        demand_mean = float(np.mean(hist)) if len(hist) > 0 else 1.0
        if np.isnan(demand_mean):
            demand_mean = 1.0
        demand_std = float(np.std(hist, ddof=1)) if len(hist) > 1 else max(demand_mean * 0.2, 0.5)
        if np.isnan(demand_std) or demand_std < 0.5:
            demand_std = max(demand_mean * 0.2, 0.5)
        return demand_mean, demand_std

    @staticmethod
    def _pct_reduce(new: float, base: float) -> object:
        """损失下降百分比:(基线 - 当前) / 基线 * 100;基线为 0 时返回 None(不可计算)"""
        if base == 0:
            return None
        return round((base - new) / abs(base) * 100, 2)

    @staticmethod
    def _pct_change(new: float, base: float) -> object:
        """利润提升百分比:(当前 - 基线) / 基线 * 100;基线为 0 时返回 None(不可计算)"""
        if base == 0:
            return None
        return round((new - base) / abs(base) * 100, 2)

    # ──────────── Bootstrap 不确定性量化 ────────────

    def bootstrap_evaluate(self, sku_id: str = None, n_bootstrap: int = 1000,
                           eval_days: int = 7) -> dict:
        """
        对 Agent 补货策略的日均损失做 Bootstrap 重采样,输出 95% 置信区间。

        思路:从最近 30 天历史需求中有放回抽取 eval_days 天,模拟一段时间的
        进货损失,重复 n_bootstrap 次,得到日均损失的分布与置信区间。

        Args:
            sku_id: 指定单个 SKU 评估;None 则评估全部 SKU
            n_bootstrap: 重采样次数(默认 1000)
            eval_days: 每次采样模拟的天数(默认 7)
        """
        df = self._sorted_df()
        target_skus = [sku_id] if sku_id is not None else list(df["sku_id"].unique())
        grp_by_sku = {sid: g for sid, g in df.groupby("sku_id")}

        # 并行各 SKU 的 bootstrap(只读输入,线程安全)
        outcomes = parallel_map(
            lambda sid: self._bootstrap_single_sku(sid, grp_by_sku.get(sid), n_bootstrap, eval_days),
            target_skus,
            desc="bootstrap_evaluate",
        )
        results = [r for r in outcomes if r is not None]
        if not results:
            return {"error": "无可用SKU数据"}

        # 聚合:按列(SKU)求平均,得到全店层面的日均损失样本
        per_sku_samples = np.vstack([r["_samples"] for r in results])  # (n_skus, n_bootstrap)
        store_samples = per_sku_samples.mean(axis=0)                    # (n_bootstrap,)

        per_sku_output = [
            {
                "sku_id": r["sku_id"],
                "sku_name": r["sku_name"],
                "mean_daily_loss": round(float(r["_samples"].mean()), 2),
                "ci_lower": round(float(np.percentile(r["_samples"], 2.5)), 2),
                "ci_upper": round(float(np.percentile(r["_samples"], 97.5)), 2),
            }
            for r in results
        ]

        return {
            "mean_daily_loss": round(float(store_samples.mean()), 2),
            "ci_lower": round(float(np.percentile(store_samples, 2.5)), 2),
            "ci_upper": round(float(np.percentile(store_samples, 97.5)), 2),
            "std_daily_loss": round(float(store_samples.std()), 2),
            "n_bootstrap": n_bootstrap,
            "eval_days": eval_days,
            "n_skus": len(results),
            "per_sku": per_sku_output,
        }

    def _bootstrap_single_sku(self, sku_id: str, grp: Optional[pd.DataFrame],
                              n_bootstrap: int, eval_days: int) -> Optional[dict]:
        """单 SKU 的 Bootstrap 重采样(纯 numpy 向量化,固定种子 42 可复现)"""
        if grp is None or len(grp) == 0:
            return None
        try:
            # ── 历史需求序列:recovered_demand 优先,无则用 sales,取最近 30 天 ──
            if "recovered_demand" in grp.columns and grp["recovered_demand"].notna().any():
                hist = grp["recovered_demand"].tail(30).to_numpy(dtype=float)
            else:
                hist = grp["sales"].tail(30).to_numpy(dtype=float)
            hist = hist[~np.isnan(hist)]
            if len(hist) == 0:
                return None

            # ── 需求统计与最优库存水平 ──
            fb = self._fallback.get_params(sku_id)
            demand_mean, demand_std = self._demand_stats(hist)
            if fb["level"] != "sku":
                demand_mean = fb["demand_mean"]
                demand_std = fb["demand_std"]
                if demand_std < 0.5:
                    demand_std = max(demand_mean * 0.2, 0.5)
            opt = self._optimizer._optimize_sku_with_grp(sku_id, grp, demand_mean, demand_std, 0, 0)
            # 折算为日均目标库存:optimal_stock_level 覆盖整个补货周期(lead_time 天),
            # 而回测以单日需求为基准,需折算到单日口径才与其他按日备货的策略公平对比
            lead_time = max(float(opt.get("lead_time", 1)), 1.0)
            optimal_stock = float(opt["optimal_stock_level"]) / lead_time

            # ── 成本参数 ──
            Cu, Co = self._get_sku_costs(sku_id, grp)

            # ── Bootstrap:一次性生成随机索引矩阵 (n_bootstrap, eval_days) ──
            # 每个 SKU 各自用种子 42,采样独立且各自可复现
            rng = np.random.default_rng(42)
            idx = rng.integers(0, len(hist), size=(n_bootstrap, eval_days))
            sampled = hist[idx]                                   # (n_bootstrap, eval_days)

            # ── 每次采样的日均不对称损失(向量化) ──
            shortage = np.maximum(0.0, sampled - optimal_stock)
            overage = np.maximum(0.0, optimal_stock - sampled)
            daily_loss = (shortage * Cu + overage * Co).sum(axis=1) / eval_days

            return {
                "sku_id": sku_id,
                "sku_name": opt.get("sku_name", sku_id),
                "_samples": daily_loss,
            }
        except Exception as e:
            logger.warning("bootstrap 单SKU %s 评估失败: %s", sku_id, e)
            return None

    # ──────────── 基线对比 ────────────

    def baseline_comparison(self, sku_id: str = None, eval_days: int = 14) -> dict:
        """
        在回测窗口内对比四种进货策略的日均损失,并输出服务水平、利润提升、
        损失下降三个业务指标。

        四种策略:
            1. 昨日销量: 前一天卖多少就进多少
            2. 历史均值: 按前 30 天平均销量进货
            3. 完美预知: 提前知道当天真实需求(理论最优,仅用于衡量差距)
            4. Agent建议: 本系统基于前 30 天数据算出的最优库存

        Args:
            sku_id: 指定单个 SKU 对比;None 则全部 SKU
            eval_days: 回测窗口天数(默认 14)
        """
        df = self._sorted_df()
        target_skus = [sku_id] if sku_id is not None else list(df["sku_id"].unique())
        grp_by_sku = {sid: g for sid, g in df.groupby("sku_id")}

        # 是否存在真实需求列(合成数据才有;真实数据用恢复需求替代并标注)
        has_oracle = bool("demand" in df.columns and df["demand"].notna().any())

        # 并行各 SKU 的回测
        outcomes = parallel_map(
            lambda sid: self._baseline_single_sku(sid, grp_by_sku.get(sid), eval_days, has_oracle),
            target_skus,
            desc="baseline_comparison",
        )
        results = [r for r in outcomes if r is not None]
        if not results:
            return {"error": "无可用SKU数据"}

        # ── 聚合四策略的总损失 / 总利润 / 满足率 ──
        strategy_names = ["昨日销量", "历史均值", "完美预知", "Agent建议"]
        total_loss = {n: sum(r["losses"][n] for r in results) for n in strategy_names}
        total_profit = {n: sum(r["profits"][n] for r in results) for n in strategy_names}
        total_days = sum(r["n_days"] for r in results)
        sold_agent = sum(r["sold_agent"] for r in results)
        demand_agent = sum(r["demand_agent"] for r in results)
        n_skus = len(results)

        # 日均损失 = 总损失 / 总天数(全窗口全SKU)
        mean_daily = {n: round(total_loss[n] / max(total_days, 1), 2) for n in strategy_names}

        # 服务水平:Agent 策略的订单满足率
        service_level_pct = round(sold_agent / max(demand_agent, 1e-9) * 100, 2)

        # ── 业务指标 ──
        agent_loss = total_loss["Agent建议"]
        yesterday_loss = total_loss["昨日销量"]
        mean_base_loss = total_loss["历史均值"]
        oracle_loss = total_loss["完美预知"]
        agent_profit = total_profit["Agent建议"]
        yesterday_profit = total_profit["昨日销量"]
        mean_base_profit = total_profit["历史均值"]

        # 与完美预知的差距(oracle 损失为 0 时不可计算,返回 None)
        if oracle_loss == 0:
            oracle_gap_pct = None
        else:
            oracle_gap_pct = round((agent_loss - oracle_loss) / oracle_loss * 100, 2)

        # 策略列表按总损失升序排列
        strategies = sorted(
            [{"name": n,
              "mean_daily_loss": mean_daily[n],
              "total_loss": round(total_loss[n], 2),
              "total_profit": round(total_profit[n], 2)} for n in strategy_names],
            key=lambda x: x["total_loss"],
        )

        per_sku = [
            {
                "sku_id": r["sku_id"],
                "sku_name": r["sku_name"],
                "losses": {n: round(r["losses"][n], 2) for n in strategy_names},
            }
            for r in results
        ]

        return {
            "strategies": strategies,
            "improvement_vs_mean_pct": self._pct_reduce(agent_loss, mean_base_loss),
            "oracle_gap_pct": oracle_gap_pct,
            "profit_improvement_vs_yesterday_pct": self._pct_change(agent_profit, yesterday_profit),
            "profit_improvement_vs_mean_pct": self._pct_change(agent_profit, mean_base_profit),
            "service_level_pct": service_level_pct,
            "loss_reduction_vs_yesterday_pct": self._pct_reduce(agent_loss, yesterday_loss),
            "has_oracle": has_oracle,
            "eval_days": eval_days,
            "n_skus": n_skus,
            "per_sku": per_sku,
        }

    def _baseline_single_sku(self, sku_id: str, grp: Optional[pd.DataFrame],
                             eval_days: int, has_oracle: bool) -> Optional[dict]:
        """单 SKU 回测:在窗口内逐日计算四策略的损失与利润(只读输入,线程安全)"""
        if grp is None or len(grp) == 0:
            return None
        try:
            if "date" in grp.columns:
                grp = grp.sort_values("date").reset_index(drop=True)
            else:
                grp = grp.reset_index(drop=True)
            n = len(grp)
            # 窗口:最后 eval_days 天(至少留 1 天历史给"昨日销量"基线)
            window_start = max(1, n - eval_days)

            # 真实需求列:有 demand(真实需求)用 demand,否则用恢复需求/销量
            if has_oracle and "demand" in grp.columns:
                demand_col = "demand"
            elif "recovered_demand" in grp.columns and grp["recovered_demand"].notna().any():
                demand_col = "recovered_demand"
            else:
                demand_col = "sales"
            # 历史需求列(策略2/4 的统计依据)
            hist_col = ("recovered_demand"
                        if "recovered_demand" in grp.columns and grp["recovered_demand"].notna().any()
                        else "sales")

            Cu, Co = self._get_sku_costs(sku_id, grp)
            price, cost = self._get_sku_price_cost(sku_id, grp)
            margin = max(price - cost, 0.1)

            losses = {"昨日销量": 0.0, "历史均值": 0.0, "完美预知": 0.0, "Agent建议": 0.0}
            profits = {"昨日销量": 0.0, "历史均值": 0.0, "完美预知": 0.0, "Agent建议": 0.0}
            sold_agent = 0.0
            demand_agent = 0.0

            for i in range(window_start, n):
                d_t = float(grp[demand_col].iloc[i])
                # 截至 t-1 的最近 30 天历史需求
                hist_before = grp[hist_col].iloc[max(0, i - 30):i].to_numpy(dtype=float)
                hist_before = hist_before[~np.isnan(hist_before)]

                # 策略1: 昨日销量(前一天卖多少进多少)
                stock1 = float(grp["sales"].iloc[i - 1])
                # 策略2: 历史均值(截至 t-1 的最近 30 天平均)
                stock2 = float(np.mean(hist_before)) if len(hist_before) > 0 else 0.0
                # 策略3: 完美预知(当天真实需求,理论下界)
                stock3 = d_t
                # 策略4: Agent 建议(基于截至 t-1 的最近 30 天数据算最优库存)
                dm, ds = self._demand_stats(hist_before)
                fb = self._fallback.get_params(sku_id)
                if fb["level"] != "sku":
                    dm = fb["demand_mean"]
                    ds = fb["demand_std"]
                    if ds < 0.5:
                        ds = max(dm * 0.2, 0.5)
                opt = self._optimizer._optimize_sku_with_grp(sku_id, grp, dm, ds, 0, 0)
                # optimal_stock_level 覆盖整个补货周期(lead_time 天),
                # 折算为日均库存,与单日需求及各按日备货基线同口径对比
                stock4 = float(opt["optimal_stock_level"]) / max(float(opt.get("lead_time", 1)), 1.0)

                for name, stock in [("昨日销量", stock1), ("历史均值", stock2),
                                    ("完美预知", stock3), ("Agent建议", stock4)]:
                    shortage = max(0.0, d_t - stock)
                    overage = max(0.0, stock - d_t)
                    losses[name] += shortage * Cu + overage * Co
                    # 单日利润 = 售出×毛利 - 过剩×损耗成本 - 缺货×额外流失成本
                    # Cu = margin + 客损,缺货时 sold*margin 已少赚缺货部分的毛利,
                    # 故缺货只扣 Cu - margin(顾客流失等额外损失),避免毛利双重扣减
                    sold = min(stock, d_t)
                    profits[name] += sold * margin - overage * Co - shortage * (Cu - margin)

                sold_agent += min(stock4, d_t)
                demand_agent += d_t

            return {
                "sku_id": sku_id,
                "sku_name": grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sku_id,
                "losses": losses,
                "profits": profits,
                "sold_agent": sold_agent,
                "demand_agent": demand_agent,
                "n_days": n - window_start,
            }
        except Exception as e:
            logger.warning("baseline 单SKU %s 回测失败: %s", sku_id, e)
            return None