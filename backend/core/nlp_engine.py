"""
对话式自然语言交互引擎
======================
支持中文自然语言提问,自动识别意图并调用对应算法模块:
- 补货查询:"明天生菜要补多少斤?"
- 损耗归因:"上周苹果损耗高是什么原因?"
- 促销模拟:"中秋做8折促销,牛奶要多补多少?"
- 库存查询:"草莓还有多少库存?"
- 预测查询:"下周番茄销量预测?"
- 综合建议:"帮我看看明天该进什么货?"

技术方案:规则+关键词匹配的轻量级中文意图识别(无需外部NLP服务)
"""
import re
from typing import Optional


# 意图定义
INTENT_RULES = [
    {
        "intent": "replenishment",
        "label": "该进什么货",
        "keywords": ["补货", "补多少", "进货", "订货", "要补", "该进", "进什么货", "下单", "采购多少", "要订",
                     "不够卖", "缺货", "断货", "少货", "备货", "要备", "上货", "拿货", "要多少货", "要进", "进多少"],
        "patterns": [r"补.*多少", r"进.*货", r"订.*货", r"不够.*卖", r"断.*货"],
    },
    {
        "intent": "waste_attribution",
        "label": "为什么扔了那么多",
        "keywords": ["损耗", "报废", "过期", "扔掉", "坏了", "损耗高", "损耗原因", "为什么损耗", "损耗多",
                     "亏钱", "亏了", "浪费", "卖不掉", "烂了", "蔫了", "不新鲜", "扔了多少"],
        "patterns": [r"损耗.*原因", r"为什么.*损耗", r"损耗.*高", r"为什么.*扔", r"为什么.*坏"],
    },
    {
        "intent": "simulation",
        "label": "促销试算",
        "keywords": ["促销", "打折", "折扣", "调价", "降价", "涨价", "模拟", "如果", "假设", "做活动",
                     "8折", "7折", "9折", "半价", "试算", "试试", "划算吗", "能多卖", "多赚",
                     "搞活动", "做特价", "便宜点"],
        "patterns": [r"\d+折", r"促销.*多补", r"打折.*补", r"调价.*影响", r"如果.*打折", r"做.*活动"],
    },
    {
        "intent": "forecast",
        "label": "能卖多少",
        "keywords": ["预测", "销量会", "预计", "明天卖", "下周卖", "能卖多少", "需求多少", "趋势",
                     "能卖几", "好卖吗", "卖得动吗", "能卖完吗", "卖多少"],
        "patterns": [r"预测.*销量", r"能卖多少", r"销量.*趋势", r"能卖.*多少"],
    },
    {
        "intent": "inventory",
        "label": "还剩多少货",
        "keywords": ["库存", "还有多少", "存货", "剩多少", "仓里", "在手", "还剩", "有多少货", "剩下多少"],
        "patterns": [r"库存.*多少", r"还有.*多少", r"还剩.*多少"],
    },
    {
        "intent": "discount",
        "label": "临期怎么处理",
        "keywords": ["临期", "快过期", "折扣建议", "怎么打折", "清仓", "甩卖",
                     "快坏了", "卖不完怎么办", "怎么处理", "快到期的", "马上过期"],
        "patterns": [r"临期.*折", r"快过期.*怎么办", r"清仓.*折", r"快过期.*打折", r"卖不完.*怎么办"],
    },
    {
        "intent": "overview",
        "label": "生意怎么样",
        "keywords": ["概览", "整体情况", "经营状况", "总结", "报告", "帮我看看", "总览",
                     "今天怎么样", "生意怎么样", "挣钱了吗", "赚钱了吗", "赔了吗", "生意如何",
                     "最近怎么样", "情况如何", "做得怎么样"],
        "patterns": [r"整体.*情况", r"经营.*状况", r"生意.*怎么样", r"生意.*如何"],
    },
]

# 时间词映射
TIME_KEYWORDS = {
    "今天": 0, "今日": 0, "现在": 0,
    "明天": 1, "明日": 1, "后天": 2,
    "这周": 7, "本周": 7, "这七天": 7,
    "下周": 7, "未来一周": 7, "接下来一周": 7,
    "上周": -7, "上个月": -30, "上周": -7, "前一周": -7,
    "这个月": 30, "本月": 30, "未来三天": 3, "三天": 3,
}

# 折扣词解析
DISCOUNT_PATTERNS = [
    (r"(\d+(?:\.\d+)?)\s*折", lambda m: float(m.group(1)) / 10),
    (r"打\s*(\d+(?:\.\d+)?)\s*折", lambda m: float(m.group(1)) / 10),
    (r"半价", lambda m: 0.5),
    (r"(\d+(?:\.\d+)?)\s*%?\s*off", lambda m: 1 - float(m.group(1)) / 100),
]


class NLPEngine:
    """中文自然语言意图识别引擎"""

    def __init__(self, sku_names: Optional[dict] = None):
        """
        Args:
            sku_names: {sku_name: sku_id} 映射表
        """
        self.sku_names = sku_names or {}

    def update_sku_names(self, sku_names: dict):
        self.sku_names = sku_names

    # ──────────── 意图识别 ────────────
    def recognize_intent(self, text: str) -> dict:
        """识别用户输入的意图"""
        text = text.strip()
        scores = {}
        for rule in INTENT_RULES:
            score = 0
            for kw in rule["keywords"]:
                if kw in text:
                    score += 2
            for pat in rule["patterns"]:
                if re.search(pat, text):
                    score += 3
            if score > 0:
                scores[rule["intent"]] = (score, rule["label"])

        if not scores:
            # 默认:如果提到商品名,尝试补货查询
            sku = self._extract_sku(text)
            if sku:
                best_intent, best_label = "replenishment", "该进什么货"
            else:
                best_intent, best_label = "unknown", "未识别"
        else:
            best_intent = max(scores, key=lambda k: scores[k][0])
            best_label = scores[best_intent][1]

        return {
            "intent": best_intent,
            "label": best_label,
            "all_scores": {k: v[0] for k, v in scores.items()},
        }

    # ──────────── SKU 提取 ────────────
    def _extract_sku(self, text: str) -> Optional[str]:
        """从文本中提取SKU名称并返回sku_id"""
        # 精确匹配
        for name, sid in self.sku_names.items():
            if name in text:
                return sid
        # 模糊匹配(包含关系)
        for name, sid in self.sku_names.items():
            if len(name) >= 2 and any(ch in text for ch in name):
                # 检查至少一半的字符匹配
                matched = sum(1 for ch in name if ch in text)
                if matched >= len(name) * 0.5:
                    return sid
        return None

    def _extract_sku_name(self, text: str) -> Optional[str]:
        for name in self.sku_names:
            if name in text:
                return name
        return None

    # ──────────── 参数提取 ────────────
    def extract_params(self, text: str) -> dict:
        """提取时间、折扣、数量等参数"""
        params = {}
        # 时间
        for time_word, days in TIME_KEYWORDS.items():
            if time_word in text:
                params["time_word"] = time_word
                params["days"] = days
                break
        if "days" not in params:
            params["days"] = 1  # 默认明天
            params["time_word"] = "明天"

        # 折扣
        for pat, converter in DISCOUNT_PATTERNS:
            m = re.search(pat, text)
            if m:
                params["discount"] = converter(m)
                break

        # 数量(斤/个/盒/瓶等)
        m = re.search(r"(\d+(?:\.\d+)?)\s*(斤|个|盒|瓶|件|份|公斤|kg)", text)
        if m:
            params["quantity"] = float(m.group(1))
            params["quantity_unit"] = m.group(2)

        # 价格变动百分比
        m = re.search(r"(涨|降|升|跌|提价|降价)\s*(\d+(?:\.\d+)?)\s*%", text)
        if m:
            direction = 1 if m.group(1) in ("涨", "升", "提价") else -1
            params["price_change_pct"] = direction * float(m.group(2))

        # SKU
        sku_id = self._extract_sku(text)
        if sku_id:
            params["sku_id"] = sku_id
            params["sku_name"] = self._extract_sku_name(text)

        return params

    # ──────────── 完整解析 ────────────
    def parse(self, text: str) -> dict:
        """完整解析用户输入:意图 + 参数"""
        intent = self.recognize_intent(text)
        params = self.extract_params(text)
        return {
            "text": text,
            "intent": intent["intent"],
            "intent_label": intent["label"],
            "params": params,
        }

    # ──────────── 回复生成 ────────────
    @staticmethod
    def format_replenishment_reply(result: dict, params: dict) -> str:
        """格式化补货查询回复"""
        sku_name = result.get("sku_name", params.get("sku_name", "该商品"))
        unit = result.get("unit", "件")
        time_word = params.get("time_word", "明天")
        reorder = result.get("reorder_qty", 0)
        demand = result.get("demand_mean", 0)
        inv = result.get("current_inventory", 0)
        service = result.get("service_level", 95)
        risk = result.get("waste_risk_level", "低")
        lines = [
            f"📊 **{sku_name}** {time_word}进货建议:",
            f"",
            f"- 每天大概能卖:**{demand}** {unit}",
            f"- 现在还有:**{inv}** {unit}",
            f"- 建议进:**{reorder}** {unit}",
            f"- 照这个量进货,不缺货的把握:**{service}%**",
            f"- 过期风险:**{risk}**",
        ]
        if result.get("expected_stockout_loss", 0) > 0:
            lines.append(f"- 万一断货要少赚:{result['expected_stockout_loss']} 元")
        if result.get("expected_overstock_loss", 0) > 0:
            lines.append(f"- 万一进多了要亏:{result['expected_overstock_loss']} 元")
        return "\n".join(lines)

    @staticmethod
    def format_waste_reply(result: dict, params: dict) -> str:
        """格式化损耗归因回复"""
        sku_name = params.get("sku_name", "")
        if "error" in result:
            return result["error"]
        if sku_name and sku_name in str(result.get("sku_attributions", {})):
            # 单SKU
            for sid, attr in result["sku_attributions"].items():
                lines = [
                    f"📉 **{attr['sku_name']}** 为什么扔了这么多:",
                    f"",
                    f"- 总共扔掉:{attr['total_waste_qty']} 件",
                    f"- 扔掉的钱:{attr['total_waste_cost']} 元",
                    f"",
                    f"**主要原因:**",
                ]
                for factor, pct in sorted(attr["attribution"].items(), key=lambda x: -x[1]):
                    if pct > 0:
                        lines.append(f"- {factor}:占 **{pct}%**")
                return "\n".join(lines)
        # 全局
        lines = ["📉 **全店为什么扔了这么多:**", ""]
        total_by_factor = {}
        for sid, attr in result.get("sku_attributions", {}).items():
            for factor, pct in attr["attribution"].items():
                total_by_factor[factor] = total_by_factor.get(factor, 0) + pct
        n = max(len(result.get("sku_attributions", {})), 1)
        for factor, total_pct in sorted(total_by_factor.items(), key=lambda x: -x[1]):
            lines.append(f"- {factor}:平均占 **{total_pct/n:.1f}%**")
        return "\n".join(lines)

    @staticmethod
    def format_simulation_reply(result: dict, params: dict) -> str:
        """格式化策略模拟回复"""
        if "error" in result:
            return result["error"]
        sku_name = result.get("sku_name", "")
        rev_change = result.get('expected_revenue_change', 0)
        waste_change = result.get('expected_waste_change', 0)
        lines = [
            f"🔬 **{sku_name}** 试算结果:",
            f"",
            f"- 现在每天能卖:{result['base_demand']}",
            f"- 调整后每天能卖:**{result['simulated_demand']}** (变化 {result['demand_change_pct']:+.1f}%)",
            f"- 现在还有库存:{result['current_inventory']}",
            f"- 进货要不要变:**{result['suggested_replenishment_adjustment']:+.1f}**",
            f"- 预计{'多赚' if rev_change >= 0 else '少赚'}:**{abs(rev_change):.1f}** 元",
            f"- 预计扔掉的{'增加' if waste_change >= 0 else '减少'}:**{abs(waste_change):.1f}** 件",
            f"- 可能断货:{result['stockout_risk_pct']}% | 可能进多了:{result['overstock_risk_pct']}%",
        ]
        if result.get("factor_details"):
            lines.append("")
            lines.append("**各项调整的影响:**")
            for d in result["factor_details"]:
                lines.append(f"- {d['factor']}({d['change']}):{d['impact_pct']:+.1f}%")
        return "\n".join(lines)

    @staticmethod
    def format_forecast_reply(result: dict, params: dict) -> str:
        """格式化预测回复"""
        if "error" in result:
            return result["error"]
        sku_name = result.get("sku_name", "")
        unit = result.get("unit", "件")
        days = params.get("days", 7)
        lines = [f"📈 **{sku_name}** 未来{days}天能卖多少:", ""]
        for f in result.get("forecasts", [])[:days]:
            lines.append(f"- {f['date']}({f['dow']}):预计卖 **{f['forecast']}** {unit}(最少 {f['lower']},最多 {f['upper']})")
        if result.get("correction_uplift_pct", 0) > 0:
            lines.append(f"\n💡 之前断货导致销量被低估,已修正,实际上调了 **{result['correction_uplift_pct']}%**")
        return "\n".join(lines)