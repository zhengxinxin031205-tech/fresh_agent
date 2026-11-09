"""
分析服务编排层
==============
统一调度各核心算法模块,管理计算结果的缓存,
为API路由层提供简洁的高层接口。
"""
import threading
import pandas as pd
from typing import Optional
from backend.core.data_adapter import DataSession
from backend.core.demand_recovery import DemandRecovery
from backend.core.causal_engine import CausalEngine
from backend.core.replenishment_optimizer import ReplenishmentOptimizer
from backend.core.waste_analyzer import WasteAnalyzer
from backend.core.forecaster import DemandForecaster
from backend.core.nlp_engine import NLPEngine
from backend.core.evaluation import StrategyEvaluator
from backend.core.np_utils import to_native


class AnalysisService:
    """分析服务单例:编排所有核心算法"""

    _instance = None

    def __init__(self):
        self.session = DataSession.get()
        self._cache = {}
        self._preloading = False
        # 预热进度:供前端轮询展示分析进度条
        # step=当前步骤描述(面向商家的大白话), step_index=当前步骤序号(1-based),
        # total_steps=总步骤数, done=是否全部完成, started_at=预热启动时间
        self._preload_progress = {
            "step": "", "step_index": 0, "total_steps": 5,
            "done": False, "started_at": None,
        }

    @classmethod
    def get(cls) -> "AnalysisService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _invalidate_cache(self):
        self._cache = {}
        # 数据变了,预热进度重置,前端会重新显示进度条
        self._preload_progress["done"] = False

    def get_preload_status(self) -> dict:
        """返回当前预热进度,供前端轮询展示进度条。

        进度字典只在 _run 线程中写入、此方法只读,无竞争问题。
        未加载数据时直接返回已完成状态,前端不显示进度条。
        """
        if not self.is_loaded():
            return {"step": "", "step_index": 0, "total_steps": 5, "done": True, "started_at": None}
        return dict(self._preload_progress)

    # ──────────── 后台预计算 ────────────
    def preload_async(self):
        """数据加载完成后,在后台线程预热全量计算缓存。
        用户首次请求时直接命中缓存,秒级返回。
        用 _preloading 标志防止重复启动;预热异常只记录日志,不影响主服务。
        """
        if self._preloading:
            print("[预热] 已有预计算任务进行中,跳过本次预热")
            return
        if not self.is_loaded():
            print("[预热] 数据尚未加载,跳过预热")
            return
        self._preloading = True

        def _set_progress(step_index, step_text):
            """更新预热进度(面向商家的大白话文案)"""
            self._preload_progress["step"] = step_text
            self._preload_progress["step_index"] = step_index
            self._preload_progress["done"] = False

        def _run():
            try:
                from datetime import datetime
                self._preload_progress["started_at"] = datetime.now().isoformat(timespec="seconds")
                print("[预热] 后台预计算开始...")
                _set_progress(1, "正在把缺货那几天没统计到的需求补回来")
                self.get_recovered_df()
                print("[预热] 缺货恢复完成")
                _set_progress(2, "正在分析促销、节假日、天气对销量的影响")
                self.get_causal_engine().estimate_effects()
                print("[预热] 因果推断完成")
                _set_progress(3, "正在算每样商品该进多少")
                self.get_replenishment_list()
                print("[预热] 补货决策完成")
                _set_progress(4, "正在预测未来一周能卖多少")
                self.get_forecasts(7)
                print("[预热] 需求预测完成")
                _set_progress(5, "正在对比不同进货策略的损失")
                self.get_evaluator()
                print("[预热] 策略评估器就绪")
                self._preload_progress["done"] = True
                print("[预热] 全部预计算完成,缓存已就绪")
            except Exception as e:
                print(f"[预热] 预计算出错(不影响主服务): {e}")
                self._preload_progress["done"] = True
            finally:
                self._preloading = False

        threading.Thread(target=_run, daemon=True).start()

    # ──────────── 数据加载 ────────────
    def get_df(self) -> pd.DataFrame:
        return self.session.get_df()

    def get_tier(self) -> str:
        return self.session.get_tier()

    def is_loaded(self) -> bool:
        return self.session.is_loaded()

    # ──────────── 需求恢复 ────────────
    def get_recovered_df(self) -> pd.DataFrame:
        if "recovered_df" not in self._cache:
            recovery = DemandRecovery(self.get_df())
            self._cache["recovered_df"] = recovery.recover()
            self._cache["recovery_summary"] = recovery.get_summary()
            self._cache["recovery_instance"] = recovery
        return self._cache["recovered_df"]

    def get_recovery_summary(self) -> dict:
        self.get_recovered_df()
        return to_native(self._cache["recovery_summary"])

    # ──────────── 因果推断 ────────────
    def get_causal_engine(self) -> CausalEngine:
        if "causal_engine" not in self._cache:
            engine = CausalEngine(self.get_df(), excluded_factors=self.session.excluded_factors)
            engine.estimate_effects()
            self._cache["causal_engine"] = engine
        return self._cache["causal_engine"]

    def set_excluded_factors(self, factors: list):
        """更新排除因素列表并清除相关缓存"""
        self.session.set_excluded_factors(factors)
        self._cache.pop("causal_engine", None)

    # ──────────── 补货优化 ────────────
    def get_replenishment_list(self) -> list:
        if "replenishment_list" not in self._cache:
            recovered = self.get_recovered_df()
            optimizer = ReplenishmentOptimizer(recovered, self.get_tier())
            # to_native 兜底:确保缓存中存的是 JSON 安全的原生类型
            self._cache["replenishment_list"] = to_native(optimizer.generate_replenishment_list(recovered))
        return self._cache["replenishment_list"]

    # ──────────── 需求预测 ────────────
    def get_forecasts(self, horizon: int = 7) -> list:
        cache_key = f"forecasts_{horizon}"
        if cache_key not in self._cache:
            recovered = self.get_recovered_df()
            forecaster = DemandForecaster(recovered)
            self._cache[cache_key] = to_native(forecaster.forecast_all(horizon, recovered))
        return self._cache[cache_key]

    def get_forecast_sku(self, sku_id: str, horizon: int = 7) -> dict:
        recovered = self.get_recovered_df()
        forecaster = DemandForecaster(recovered)
        return to_native(forecaster.forecast_sku(sku_id, horizon, recovered))

    # ──────────── 损耗分析 ────────────
    def get_waste_analyzer(self) -> WasteAnalyzer:
        if "waste_analyzer" not in self._cache:
            self._cache["waste_analyzer"] = WasteAnalyzer(self.get_df())
        return self._cache["waste_analyzer"]

    # ──────────── 策略评估 ────────────
    def get_evaluator(self) -> StrategyEvaluator:
        """获取策略评估器(缓存)"""
        if "evaluator" not in self._cache:
            recovered = self.get_recovered_df()
            self._cache["evaluator"] = StrategyEvaluator(recovered, tier=self.get_tier())
        return self._cache["evaluator"]

    def get_bootstrap_evaluate(self, sku_id: str = None, n_bootstrap: int = 1000, eval_days: int = 7) -> dict:
        """Bootstrap 重采样评估 Agent 补货策略的日均损失与 95% 置信区间"""
        evaluator = self.get_evaluator()
        return to_native(evaluator.bootstrap_evaluate(sku_id=sku_id, n_bootstrap=n_bootstrap, eval_days=eval_days))

    def get_baseline_comparison(self, sku_id: str = None, eval_days: int = 14) -> dict:
        """三条基线对比:昨日销量/历史均值/完美预知 vs Agent建议"""
        evaluator = self.get_evaluator()
        return to_native(evaluator.baseline_comparison(sku_id=sku_id, eval_days=eval_days))

    def get_estimated_elasticity(self, sku_id: str) -> dict:
        """AI估算指定SKU的价格敏感度(弹性系数)
        优先用因果引擎基于历史价格变动回归出的SKU级弹性;
        数据不含价格信息或回归质量差时,回退到行业经验默认值。
        """
        default_elasticity = -1.5
        tier = self.get_tier()
        # 仅进阶版以上数据(含价格/促销列)才能做因果弹性估算
        if tier in ("advanced", "full"):
            try:
                engine = self.get_causal_engine()
                est = engine.elasticity.get(sku_id)
                if est is not None and est < 0:  # 负值才经济合理(降价多卖)
                    return {
                        "elasticity": round(float(est), 2),
                        "source": "ai",
                        "source_label": "AI根据这个商品的历史调价记录估算",
                        "tier": tier,
                    }
            except Exception:
                pass
        return {
            "elasticity": default_elasticity,
            "source": "default",
            "source_label": "没有足够的价格变动记录,用了生鲜行业常用经验值",
            "tier": tier,
        }

    # ──────────── NLP 引擎 ────────────
    def get_nlp_engine(self) -> NLPEngine:
        if "nlp_engine" not in self._cache:
            df = self.get_df()
            sku_names = {}
            if "sku_name" in df.columns and "sku_id" in df.columns:
                for _, row in df[["sku_id", "sku_name"]].drop_duplicates().iterrows():
                    sku_names[row["sku_name"]] = row["sku_id"]
            self._cache["nlp_engine"] = NLPEngine(sku_names)
        return self._cache["nlp_engine"]

    # ──────────── 仪表盘数据 ────────────
    def get_dashboard_data(self, period: str = "week") -> dict:
        df_all = self.get_df()
        tier = self.get_tier()
        recovery_summary = self.get_recovery_summary()

        # 按时段过滤数据(today=最后一天 / week=最近7天 / month=最近30天 / all=全部)
        period_labels = {"today": "今天", "week": "最近7天", "month": "最近30天", "all": "全部时间"}
        df = df_all
        if "date" in df_all.columns and period != "all":
            try:
                max_date = pd.to_datetime(df_all["date"]).max()
                dates = pd.to_datetime(df_all["date"])
                if period == "today":
                    df = df_all[dates == max_date]
                elif period == "week":
                    df = df_all[dates > max_date - pd.Timedelta(days=7)]
                elif period == "month":
                    df = df_all[dates > max_date - pd.Timedelta(days=30)]
            except Exception:
                df = df_all  # 日期解析失败时回退到全量数据

        n_skus = df["sku_id"].nunique() if "sku_id" in df.columns else 0
        total_sales = int(df["sales"].sum()) if "sales" in df.columns else 0
        avg_daily_sales = round(float(df["sales"].mean()), 1) if "sales" in df.columns else 0

        # 品类销售分布(按时段过滤后的数据)
        category_dist = {}
        if "category" in df.columns and "sales" in df.columns:
            for cat, grp in df.groupby("category"):
                category_dist[cat] = int(grp["sales"].sum())

        # 因果摘要(基于全量数据估算,不受时段影响)
        causal_summary = {}
        if tier in ("advanced", "full"):
            engine = self.get_causal_engine()
            causal_summary = engine.get_global_summary()

        # 损耗摘要(按时段过滤后的数据计算)
        waste_overview = {}
        if "waste_qty" in df.columns and len(df) > 0:
            waste_period = "week" if period in ("today", "week") else "month"
            if df is df_all:
                analyzer = self.get_waste_analyzer()
            else:
                analyzer = WasteAnalyzer(df.copy())
            waste_overview = analyzer.get_overview(waste_period)

        # 补货摘要(基于全量数据,不受时段影响)
        repl_list = self.get_replenishment_list()
        total_reorder = sum(r["reorder_qty"] for r in repl_list)
        high_risk = sum(1 for r in repl_list if r["waste_risk_level"] == "高")

        # 日期范围(展示全量数据覆盖的时间范围)
        date_range = {}
        if "date" in df_all.columns:
            date_range = {"start": str(df_all["date"].min()), "end": str(df_all["date"].max())}

        # to_native: category_dist 键来自 groupby(int64 列 → numpy.int64)、
        # causal_summary/waste_overview 来自其他模块,统一转原生类型确保 JSON 安全
        return to_native({
            "tier": tier,
            "period": period,
            "period_label": period_labels.get(period, period),
            "n_skus": int(n_skus),
            "total_records": len(df),
            "total_sales": total_sales,
            "avg_daily_sales": avg_daily_sales,
            "date_range": date_range,
            "category_dist": category_dist,
            "recovery_summary": recovery_summary,
            "causal_summary": causal_summary,
            "waste_overview": waste_overview,
            "replenishment_summary": {
                "total_reorder_qty": total_reorder,
                "n_skus_to_reorder": sum(1 for r in repl_list if r["reorder_qty"] > 0),
                "high_waste_risk_count": high_risk,
            },
        })

    # ──────────── 对话处理 ────────────
    def process_chat(self, text: str) -> dict:
        """处理自然语言对话请求"""
        nlp = self.get_nlp_engine()
        parsed = nlp.parse(text)
        intent = parsed["intent"]
        params = parsed["params"]
        tier = self.get_tier()

        reply = ""
        data = None

        if intent == "replenishment":
            if params.get("sku_id"):
                recovered = self.get_recovered_df()
                optimizer = ReplenishmentOptimizer(recovered, tier)
                grp = recovered[recovered["sku_id"] == params["sku_id"]]
                recent = grp.tail(30)
                dm = float(recent["recovered_demand"].mean()) if "recovered_demand" in recent.columns else float(recent["sales"].mean())
                ds = float(recent["recovered_demand"].std()) if "recovered_demand" in recent.columns else float(recent["sales"].std())
                inv = float(grp["inventory"].iloc[-1]) if "inventory" in grp.columns else 0
                result = optimizer.optimize_sku(params["sku_id"], dm, ds, inv)
                reply = nlp.format_replenishment_reply(result, params)
                data = result
            else:
                repl = self.get_replenishment_list()
                top5 = repl[:5]
                lines = ["📊 **今天最需要进的5样货:**", ""]
                for r in top5:
                    lines.append(f"- {r['sku_name']}:进 **{r['reorder_qty']}** {r['unit']}(每天卖{r['demand_mean']},过期风险{r['waste_risk_level']})")
                reply = "\n".join(lines)
                data = top5

        elif intent == "waste_attribution":
            analyzer = self.get_waste_analyzer()
            result = analyzer.attribute_waste(params.get("sku_id"))
            reply = nlp.format_waste_reply(result, params)
            data = result

        elif intent == "simulation":
            if not params.get("sku_id"):
                reply = "请告诉我您想试哪个商品,例如:「牛奶做8折促销能多卖多少?」"
            else:
                engine = self.get_causal_engine()
                interventions = {}
                if params.get("discount"):
                    interventions["discount"] = params["discount"]
                if params.get("price_change_pct"):
                    interventions["price_change_pct"] = params["price_change_pct"]
                interventions["is_promo"] = 1 if params.get("discount") else 0
                result = engine.simulate_intervention(params["sku_id"], interventions)
                reply = nlp.format_simulation_reply(result, params)
                data = result

        elif intent == "forecast":
            if params.get("sku_id"):
                result = self.get_forecast_sku(params["sku_id"], max(params.get("days", 7), 1))
                reply = nlp.format_forecast_reply(result, params)
                data = result
            else:
                reply = "请告诉我您想知道哪个商品能卖多少,例如:「下周番茄能卖多少?」"

        elif intent == "inventory":
            df = self.get_df()
            if params.get("sku_id"):
                grp = df[df["sku_id"] == params["sku_id"]]
                inv = float(grp["inventory"].iloc[-1]) if "inventory" in grp.columns else 0
                name = params.get("sku_name", "")
                reply = f"📦 **{name}** 现在还有:**{inv}** 件"
            else:
                reply = "请告诉我您想知道哪个商品还剩多少,例如:「草莓还剩多少?」"

        elif intent == "discount":
            if not params.get("sku_id"):
                reply = "请告诉我哪个商品快过期了,例如:「草莓快过期了,打几折能卖掉?」"
            else:
                analyzer = self.get_waste_analyzer()
                df = self.get_df()
                grp = df[df["sku_id"] == params["sku_id"]]
                near_qty = float(grp["near_expiry"].iloc[-1]) if "near_expiry" in grp.columns else float(grp["inventory"].iloc[-1]) * 0.15
                engine = self.get_causal_engine()
                elasticity = engine.elasticity.get(params["sku_id"], -1.5)
                result = analyzer.optimize_discount(params["sku_id"], near_qty, elasticity)
                if "error" in result:
                    reply = result["error"]
                else:
                    reply = f"💡 {result['recommendation']}"
                data = result

        elif intent == "overview":
            dash = self.get_dashboard_data()
            lines = [
                f"📋 **您的店最近怎么样:**",
                f"",
                f"- 在卖的商品:{dash['n_skus']} 种",
                f"- 一共卖了:{dash['total_sales']} 件",
                f"- 断货天数占比:{dash['recovery_summary']['stockout_stats']['stockout_rate_pct']}%",
                f"- 今天总共要进:{dash['replenishment_summary']['total_reorder_qty']} 件",
                f"- 容易过期的商品:{dash['replenishment_summary']['high_waste_risk_count']} 种",
            ]
            if dash.get("causal_summary"):
                cs = dash["causal_summary"]
                lines.append(f"- 做促销平均多卖:{cs['avg_promo_uplift_pct']}%")
                lines.append(f"- 价格敏感度:{cs['avg_price_elasticity']}(数字越大,调价影响越大)")
            reply = "\n".join(lines)
            data = dash

        else:
            reply = (
                "抱歉,我没太听懂您的问题。您可以这样问我:\n\n"
                "- 「明天生菜要进多少?」(问进货)\n"
                "- 「上周苹果为什么扔了那么多?」(问损耗)\n"
                "- 「牛奶做8折能多卖多少?」(试促销)\n"
                "- 「下周番茄能卖多少?」(问销量)\n"
                "- 「草莓还剩多少?」(查库存)\n"
                "- 「草莓快过期了,打几折能卖掉?」(临期处理)\n"
                "- 「最近生意怎么样?」(看整体)"
            )

        # to_native: data 字段可能包含各算法模块的返回结果(含 numpy 标量),统一转原生
        return to_native({
            "reply": reply,
            "intent": intent,
            "intent_label": parsed["intent_label"],
            "params": params,
            "data": data,
        })