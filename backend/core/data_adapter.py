"""
可插拔多源数据适配器
====================
- 支持 CSV / Excel 自动加载
- 字段模糊匹配:中英文表头自动映射到标准字段
- 数据质量画像:识别字段完整性、缺失率、异常值
- 分档检测:自动判定 基础版/进阶版/完整版,匹配对应算法复杂度
- 内置5套公开数据集适配器
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ──────────────────── 标准字段定义 ────────────────────
STANDARD_FIELDS = {
    "date":          ["日期", "date", "销售日期", "交易日期", "下单日期", "salesdate", "transactiondate", "orderdate"],
    "store_id":      ["门店编号", "store_id", "store", "shop_id", "shop", "店铺编号", "门店", "店铺"],
    "sku_id":        ["SKU编码", "sku_id", "sku", "商品编码", "商品id", "货号", "编码", "条码",
                      "item", "item_id", "product_id", "商品编号"],
    "sku_name":      ["SKU名称", "sku_name", "商品名称", "品名", "名称", "商品", "product", "product_name", "商品名"],
    "category":      ["品类", "分类", "类别", "category", "商品分类", "大类", "product_category"],
    "sales":         ["当日销量", "销量", "销售数量", "销售量", "销售", "sales", "quantity", "qty", "售出数量", "出货量",
                      "units", "unitssold", "units_sold", "订单量", "订单数", "单量"],
    "demand":        ["真实需求", "需求量", "demand", "实际需求"],
    "price":         ["售价", "价格", "单价", "price", "销售价格", "实际售价", "零售价", "unit_price"],
    "base_price":    ["标准售价", "原价", "日常售价", "base_price", "标准价格", "list_price"],
    "is_promo":      ["是否促销", "促销标记", "促销", "is_promo", "promotion", "活动标记", "promo", "onpromotion"],
    "discount":      ["折扣率", "折扣", "discount", "discount_rate", "打折力度"],
    "temperature":   ["气温", "温度", "temperature", "temp", "平均气温", "avg_temp"],
    "precipitation": ["降水量", "降雨量", "降水", "precipitation", "rain", "降雨"],
    "is_weekend":    ["是否周末", "周末", "is_weekend", "weekend"],
    "is_holiday":    ["是否节假日", "节假日", "节日", "is_holiday", "holiday", "stateholiday", "publicholiday"],
    "holiday_name":  ["节假日名称", "节日名称", "holiday_name", "节日名"],
    "inventory":     ["当日库存", "库存", "库存数量", "inventory", "stock", "现有库存", "结存"],
    "in_transit":    ["在途库存", "在途", "in_transit", "已下单未到"],
    "stockout":      ["是否缺货", "缺货标记", "缺货", "stockout", "断货"],
    "waste_qty":     ["损耗数量", "报废数量", "过期数量", "损耗", "waste", "报损", "丢弃数量"],
    "near_expiry":   ["临期处理数量", "临期数量", "near_expiry", "临期处理"],
    "near_expiry_price": ["临期处理售价", "临期售价", "临期价格", "discount_price", "salvage_value", "残值"],
    "cost":          ["采购成本", "成本", "进价", "cost", "purchase_cost", "进货价", "unit_cost", "进货单价"],
    "shelf_life":    ["保质期天数", "保质期", "shelf_life", "有效期", "保存天数", "shelf_life_days"],
    "lead_time":     ["补货周期天", "补货周期", "lead_time", "到货周期", "供货周期"],
    "unit":          ["单位", "计量单位", "unit", "规格单位"],
    "gross_margin":  ["单品毛利", "毛利", "gross_margin", "利润"],
    "stockout_loss": ["缺货客损估值", "缺货损失", "客损", "stockout_loss"],
}

# ──────────────────── 语义关键词权重表 ────────────────────
# 核心识别词权重高(5),修饰词权重低(1-2),组合后自然消歧
FIELD_KEYWORDS = {
    "date":          {"日期": 5, "date": 5, "时间": 4, "time": 4, "day": 3, "dt": 5, "日": 1},
    "store_id":      {"门店": 5, "store": 5, "shop": 5, "店铺": 5, "店": 1, "outlet": 5, "nbr": 4},
    "sku_id":        {"编码": 4, "编号": 4, "货号": 5, "条码": 5, "barcode": 5, "code": 4, "sku": 5, "id": 3, "商品": 1, "货": 1, "item": 2, "product": 2},
    "sku_name":      {"名称": 5, "name": 5, "品名": 5, "商品名": 5, "商品": 1, "title": 4, "desc": 3},
    "category":      {"分类": 5, "类别": 5, "品类": 5, "category": 5, "大类": 4, "type": 3, "class": 3, "cat": 4},
    "sales":         {"销量": 5, "sales": 5, "数量": 4, "qty": 5, "quantity": 5, "售出": 4, "卖出": 4, "出货": 4, "销售": 3, "sale": 4, "sold": 4, "unit_sales": 5, "amt": 3, "amount": 3, "额": 2, "volume": 3},
    "demand":        {"需求": 5, "demand": 5},
    "price":         {"售价": 5, "价格": 5, "单价": 5, "price": 5, "零售价": 5, "unit_price": 5, "价": 2},
    "base_price":    {"原价": 5, "标准售价": 5, "base_price": 5, "list_price": 5, "日常售价": 5, "regular": 3, "base": 3},
    "is_promo":      {"促销": 5, "promo": 5, "promotion": 5, "活动": 3, "onpromotion": 5},
    "discount":      {"折扣": 5, "discount": 5, "打折": 5},
    "temperature":   {"温度": 5, "气温": 5, "temperature": 5, "temp": 4},
    "precipitation": {"降水": 5, "降雨": 5, "rain": 5, "precipitation": 5, "雨": 3},
    "is_weekend":    {"周末": 5, "weekend": 5},
    "is_holiday":    {"节假日": 5, "节日": 4, "holiday": 5, "stateholiday": 5},
    "holiday_name":  {"节假日名称": 5, "节日名": 5, "节日名称": 5, "holiday_name": 5},
    "inventory":     {"库存": 5, "inventory": 5, "stock": 5, "结存": 4},
    "in_transit":    {"在途": 5, "in_transit": 5, "未到": 4, "transit": 4},
    "stockout":      {"缺货": 5, "断货": 5, "stockout": 5, "out_of_stock": 5},
    "waste_qty":     {"损耗": 5, "报废": 5, "报损": 5, "过期数量": 5, "丢弃": 4, "waste": 5, "loss": 3, "spoilage": 5, "shrink": 4},
    "near_expiry":   {"临期": 5, "near_expiry": 5, "临近": 3, "near": 3},
    "near_expiry_price": {"临期售价": 5, "临期价格": 5, "残值": 5, "salvage": 5, "discount_price": 4, "clearance": 4},
    "cost":          {"成本": 5, "进价": 5, "cost": 5, "采购": 3, "unit_cost": 5, "进货价": 5, "wholesale": 4},
    "shelf_life":    {"保质期": 5, "shelf_life": 5, "有效期": 4, "保存期": 4, "shelf": 3, "expiry_days": 5},
    "lead_time":     {"补货周期": 5, "到货": 4, "lead_time": 5, "供货周期": 5, "补货": 3, "lead": 3},
    "unit":          {"单位": 5, "计量单位": 5, "unit": 4, "规格": 3, "uom": 4},
    "gross_margin":  {"毛利": 5, "利润": 4, "margin": 5, "profit": 4},
    "stockout_loss": {"客损": 5, "缺货损失": 5, "stockout_loss": 5, "lost_sale": 4},
}

# 分档检测规则
TIER_RULES = {
    "basic":    ["date", "sku_id", "sales"],
    "advanced": ["date", "sku_id", "sales", "price"],
    "full":     ["date", "sku_id", "sales", "waste_qty", "cost"],
}

# 补充上传建议:关键字段缺了会损失什么分析能力(面向商户的通俗说明,禁用专业术语)
SUPPLEMENT_FIELD_INFO = {
    "price":       {"label": "售价", "benefit": "能算出降价/涨价对销量的影响，促销试算更准",
                    "example": "比如: 售价, 单价, 价格"},
    "is_promo":    {"label": "是否促销", "benefit": "能分析做促销和不促销的销量差别",
                    "example": "比如: 是否促销, 促销, 促销标记"},
    "temperature": {"label": "温度", "benefit": "能看天气冷热对您商品销量的影响",
                    "example": "比如: 温度, 气温, temperature"},
    "is_holiday":  {"label": "是否节假日", "benefit": "能分析节假日对销量的影响",
                    "example": "比如: 是否节假日, 节假日, 节日"},
    "waste_qty":   {"label": "损耗数量", "benefit": "能算出每天扔掉多少钱的货，找出最浪费的商品",
                    "example": "比如: 损耗数量, 报损, 报废数量"},
    "cost":        {"label": "成本", "benefit": "能算出赚了多少钱，而不只是卖了多少",
                    "example": "比如: 成本, 进价, 采购成本"},
}

# ──────────────────── 品类关键词→保质期映射表(行业常识) ────────────────────
# 按保质期从短到长排列:同时命中多个关键词时,取更短(更保险)的保质期
# 格式: [(关键词列表, 保质期天数), ...]
SHELF_LIFE_BY_CATEGORY = [
    (["水产", "海鲜", "活鱼", "鲜鱼", "鱼类", "虾", "蟹", "贝类", "鱿鱼", "带鱼"], 1),
    (["鲜肉", "猪肉", "牛肉", "羊肉", "禽肉", "鸡肉", "鸭肉", "排骨", "五花肉", "内脏", "腊肠"], 2),
    (["叶菜", "生菜", "菠菜", "青菜", "韭菜", "白菜", "油菜", "茼蒿", "空心菜", "芹菜"], 2),
    (["豆腐", "豆制品", "豆干", "豆浆", "豆皮"], 2),
    (["草莓", "杨梅", "樱桃", "荔枝", "蓝莓", "树莓", "桑葚"], 2),
    (["预制", "熟食", "卤味", "凉菜", "便当", "快餐", "净菜"], 2),
    (["番茄", "黄瓜", "茄子", "西兰花", "菌菇", "蘑菇", "青椒", "辣椒", "豆角", "秋葵", "冬瓜"], 5),
    (["香蕉", "葡萄", "桃子", "李子", "猕猴桃", "芒果", "杏", "枇杷", "石榴"], 5),
    (["牛奶", "酸奶", "乳制品", "鲜奶", "鲜牛奶", "巴士奶", "milk", "yogurt"], 7),
    (["苹果", "梨", "橙子", "柑橘", "柚子", "柠檬", "桔子"], 15),
    (["土豆", "洋葱", "萝卜", "南瓜", "红薯", "山药", "生姜", "大蒜", "胡萝卜", "芋头"], 20),
    (["鸡蛋", "土鸡蛋", "鲜蛋", "蛋类", "egg"], 20),
    (["饮料", "果汁", "汽水", "包装食品", "零食", "罐头", "可乐", "矿泉水", "茶饮", "咖啡",
      "饮品", "drink", "juice", "beverage", "soda"], 90),
    (["冷冻", "速冻", "水饺", "汤圆", "冰品", "雪糕", "冰淇淋", "冻肉", "冻鱼", "frozen"], 90),
    (["调味品", "酱料", "粮油", "酱油", "醋", "食用油", "大米", "面粉", "挂面", "料酒", "香料"], 180),
]

# ──────────────────── 品类关键词→补货周期映射表(行业常识) ────────────────────
# 生鲜冷藏类需要每天补货,耐储商品补货周期更长
LEAD_TIME_BY_CATEGORY = [
    (["水产", "海鲜", "活鱼", "鲜鱼", "虾", "蟹",
      "鲜肉", "猪肉", "牛肉", "羊肉", "禽肉", "鸡肉", "鸭肉",
      "叶菜", "生菜", "菠菜", "青菜", "韭菜", "白菜",
      "预制", "熟食", "卤味", "凉菜", "便当"], 1),
    (["牛奶", "酸奶", "乳制品", "鲜奶",
      "豆腐", "豆制品",
      "草莓", "杨梅", "樱桃", "荔枝", "蓝莓",
      "香蕉", "葡萄", "桃子", "李子", "猕猴桃", "芒果"], 1),
    (["苹果", "梨", "橙子", "柑橘", "柚子",
      "土豆", "洋葱", "萝卜", "南瓜", "红薯", "山药", "胡萝卜",
      "番茄", "黄瓜", "茄子", "西兰花", "菌菇", "蘑菇",
      "鸡蛋"], 2),
    (["饮料", "果汁", "汽水", "包装食品", "零食", "罐头", "可乐", "矿泉水", "茶饮", "咖啡", "饮品",
      "冷冻", "速冻", "水饺", "汤圆", "冰品", "雪糕"], 3),
    (["调味品", "酱料", "粮油", "酱油", "醋", "食用油", "大米", "面粉", "挂面"], 5),
]


def _normalize_col(col: str) -> str:
    """列名标准化:去空格、转小写"""
    return str(col).strip().lower().replace(" ", "").replace("_", "")


def _min_substr_len(s: str) -> int:
    """包含匹配时列名最小长度:纯ASCII需>=4字符,含中文>=2字符"""
    return 4 if all(ord(c) < 128 for c in s) else 2


def _semantic_match(col: str) -> Optional[str]:
    """语义关键词评分匹配: 列名含某字段关键词则加分,总分最高且达阈值者胜出"""
    norm = _normalize_col(col)
    if not norm:
        return None
    col_lower = norm.lower()
    best_field, best_score = None, 0
    for field, kws in FIELD_KEYWORDS.items():
        score = 0
        for kw, w in kws.items():
            kl = kw.lower()
            if len(kl) <= 3 and kl.isascii():
                # 短英文词需词边界匹配(防"id"匹配"paid")
                if re.search(r'(?<![a-z])' + re.escape(kl) + r'(?![a-z])', col_lower):
                    score += w
            else:
                # 中文词和长英文词用子串匹配
                if kl in col_lower:
                    score += w
        if score > best_score:
            best_score = score
            best_field = field
    return best_field if best_score >= 4 else None


def _fuzzy_match(col: str) -> Optional[str]:
    """将原始列名模糊匹配到标准字段"""
    norm = _normalize_col(col)
    # 第一轮:归一化后精确匹配
    for std_field, aliases in STANDARD_FIELDS.items():
        for alias in aliases:
            if _normalize_col(alias) == norm:
                return std_field
    # 第二轮:语义关键词评分匹配(智能识别各种中英文列名变体)
    sem = _semantic_match(col)
    if sem:
        return sem
    # 第三轮:包含匹配(要求别名归一化后>=4字符,避免day等短词误命中)
    for std_field, aliases in STANDARD_FIELDS.items():
        for alias in aliases:
            a = _normalize_col(alias)
            if len(a) >= 4 and (a in norm or (len(norm) >= _min_substr_len(norm) and norm in a)):
                return std_field
    return None


def _infer_shelf_life(sku_name: str, category: str) -> Optional[int]:
    """根据商品名称/分类关键词推断保质期天数

    在 sku_name 和 category 两个字符串中搜索关键词(转小写后匹配),
    命中则返回对应保质期天数,都没命中返回 None(表示无法推断)。
    多个关键词同时命中时,取更短(更保险)的保质期。
    """
    text = f"{sku_name or ''} {category or ''}".strip().lower()
    if not text or text == "nan":
        return None
    for keywords, days in SHELF_LIFE_BY_CATEGORY:
        for kw in keywords:
            if kw.lower() in text:
                return days
    return None


def _infer_lead_time(sku_name: str, category: str) -> Optional[int]:
    """根据商品名称/分类关键词推断补货周期天数

    在 sku_name 和 category 两个字符串中搜索关键词(转小写后匹配),
    命中则返回对应补货周期天数,都没命中返回 None(表示无法推断)。
    """
    text = f"{sku_name or ''} {category or ''}".strip().lower()
    if not text or text == "nan":
        return None
    for keywords, days in LEAD_TIME_BY_CATEGORY:
        for kw in keywords:
            if kw.lower() in text:
                return days
    return None


def _apply_field_inference(df: pd.DataFrame, field: str, default_days: int,
                           infer_func, label: str) -> list:
    """对缺失的保质期/补货周期按商品名称/分类关键词智能推断

    优先级: 同商品已有值 > 关键词推断 > 默认值。
    已有值不覆盖,只填空缺。

    Args:
        df: 数据 DataFrame(会被原地修改)
        field: 标准字段名("shelf_life" 或 "lead_time")
        default_days: 无法推断时的默认天数
        infer_func: 推断函数(_infer_shelf_life 或 _infer_lead_time)
        label: 面向商户的字段中文名(如"保质期")
    Returns:
        推断报告条目列表(面向商户的通俗描述,无专业术语)
    """
    reports = []
    if field not in df.columns:
        df[field] = np.nan
    # 1) 优先用同商品已有值填补(同一商品的保质期/补货周期应当一致)
    _group_col = "sku_id" if "sku_id" in df.columns else ("sku_name" if "sku_name" in df.columns else None)
    if _group_col and df[field].isna().any():
        try:
            df[field] = df.groupby(_group_col)[field].transform(lambda s: s.ffill().bfill())
        except Exception as e:
            print(f"[推断跳过] 按{_group_col}分组填补{field}失败:{type(e).__name__}: {e}")
    # 2) 按商品名称/分类关键词推断(同一SKU只推断一次)
    missing = df[field].isna()
    inferred_detail = {}   # {商品名: 推断天数} 用于生成报告
    if missing.any() and "sku_name" in df.columns:
        has_cat = "category" in df.columns
        sku_key = df["sku_id"] if "sku_id" in df.columns else df["sku_name"].astype(str)
        for sku in sku_key[missing].unique():
            mask = (sku_key == sku) & missing
            if not mask.any():
                continue    # 防御:mask为空时iloc[0]会越界,跳过该商品
            _row = df.loc[mask]
            name = str(_row["sku_name"].iloc[0])
            cat = str(_row["category"].iloc[0]) if has_cat else ""
            days = infer_func(name, cat)
            if days is not None:
                df.loc[mask, field] = days
                inferred_detail[name] = days
    # 3) 兜底:仍未推断出来的用默认值
    remaining_na = int(df[field].isna().sum())
    if remaining_na > 0:
        df[field] = df[field].fillna(default_days)
    # 4) 生成面向商户的报告
    if inferred_detail:
        examples = "、".join(f"{n}{d}天" for n, d in list(inferred_detail.items())[:5])
        if len(inferred_detail) > 5:
            examples += "等"
        msg = f"{label}(根据商品名称推断: {examples}"
        if remaining_na > 0:
            msg += f"；未能识别的商品按默认{default_days}天"
        msg += ")"
        reports.append(msg)
    elif remaining_na > 0:
        reports.append(f"{label}(按默认{default_days}天)")
    return reports


class DataAdapter:
    """多源数据适配器"""

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None      # 标准化后的 DataFrame
        self.field_mapping: dict = {}                # 原始列名 -> 标准字段名
        self.tier: str = "basic"                     # basic / advanced / full
        self.profile: dict = {}
        self.source_name: str = ""
        self.column_overrides: dict = {}             # 商户手动指定的列映射 {原始列名: 标准字段名或None}
        self.cleaning_report: list = []              # 数据清洗报告(通俗中文描述,供画像展示)
        self.inferred_field_names: set = set()       # 系统推断/估算的字段名(机器可读,供合并逻辑使用)

    # ──────────── 加载 ────────────
    def load_file(self, filepath: str) -> "DataAdapter":
        """加载 CSV / Excel / Parquet / JSON 文件"""
        path = Path(filepath)
        self.source_name = path.name
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            self.raw_df = pd.read_excel(path)
        elif suffix == ".parquet":
            self.raw_df = pd.read_parquet(path)
        elif suffix == ".json":
            self.raw_df = pd.read_json(path)
        elif suffix == ".jsonl":
            self.raw_df = pd.read_json(path, lines=True)
        else:
            self.raw_df = pd.read_csv(path)
        return self

    def load_dataframe(self, df: pd.DataFrame, name: str = "uploaded") -> "DataAdapter":
        """直接加载 DataFrame"""
        self.raw_df = df.copy()
        self.source_name = name
        return self

    # ──────────── 标准化 ────────────
    def standardize(self) -> "DataAdapter":
        """字段自动映射与数据清洗"""
        if self.raw_df is None:
            raise ValueError("未加载数据")
        df = self.raw_df.copy()
        # 字段映射
        mapping = {}
        self.column_mapping = {}   # {原始列名: 标准字段名或None} 记录每个列名的识别结果
        for col in df.columns:
            # 商户手动指定的映射优先于自动识别
            if col in self.column_overrides:
                std = self.column_overrides[col]
            else:
                std = _fuzzy_match(col)
            self.column_mapping[col] = std  # std可能是None(未识别)
            if std and std not in mapping.values():
                mapping[col] = std
        self.field_mapping = mapping
        # 重命名
        rename = {k: v for k, v in mapping.items()}
        df = df.rename(columns=rename)
        # 门店/商品编号兜底:只有门店编号没有商品编号时,把门店当商品分析(如Rossmann)
        if "sku_id" not in df.columns and "store_id" in df.columns:
            df["sku_id"] = df["store_id"]
        # SKU兜底:没有商品编号时用商品名代替,没有商品名时用编号代替
        if "sku_id" not in df.columns and "sku_name" in df.columns:
            df["sku_id"] = df["sku_name"]
        if "sku_name" not in df.columns and "sku_id" in df.columns:
            df["sku_name"] = df["sku_id"]
        # 编号统一转字符串,避免numpy整数类型导致JSON序列化和跨接口查找失败
        if "store_id" in df.columns:
            df["store_id"] = df["store_id"].astype(str).str.strip()
        if "sku_id" in df.columns:
            df["sku_id"] = df["sku_id"].astype(str).str.strip()
        if "sku_name" in df.columns:
            df["sku_name"] = df["sku_name"].astype(str).str.strip()
        # 类型转换与清洗
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)
        if "sales" in df.columns:
            df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0).clip(lower=0)
        for num_col in ["price", "temperature", "precipitation", "inventory", "waste_qty",
                        "cost", "discount", "base_price", "near_expiry", "shelf_life", "lead_time"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
        # 明显不该为负的数值列,负值置0
        for non_neg_col in ["price", "base_price", "cost", "inventory", "waste_qty", "near_expiry", "discount"]:
            if non_neg_col in df.columns:
                df[non_neg_col] = df[non_neg_col].clip(lower=0)
        # 丢弃完全为空的列(列名存在但无任何数据,只会干扰下游分析)
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            # 记录被丢弃的空列,供画像展示
            self.empty_columns = empty_cols
        else:
            self.empty_columns = []
        # ── 数据清洗:去重 / 异常值修正 / 缺失值补齐(全空列已丢弃) ──
        self.cleaning_report = []
        if len(df) > 0:
            # 1) 去重:完全重复的行只保留一条
            #    含不可哈希类型(list/dict/ndarray)的列会让 drop_duplicates 崩溃,
            #    先检测,有则跳过去重(parquet 等格式导出的嵌套列常见此情况)
            try:
                _unhashable = [c for c in df.columns
                               if any(isinstance(v, (list, dict, set, np.ndarray))
                                      for v in df[c].dropna().head(3))]
                if _unhashable:
                    print(f"[清洗跳过] 去重:列{_unhashable}含列表/嵌套类型,无法去重,已跳过")
                else:
                    _n_before = len(df)
                    df = df.drop_duplicates()
                    _n_dup = _n_before - len(df)
                    if _n_dup > 0:
                        self.cleaning_report.append(f"去掉了 {_n_dup} 行完全重复的数据")
            except Exception as e:
                print(f"[清洗跳过] 去重失败:{type(e).__name__}: {e}")
            # 列名→通俗叫法(清洗报告面向商户,不用专业术语)
            _col_labels = {"sales": "销量", "price": "售价", "base_price": "标准售价",
                           "cost": "成本", "inventory": "库存", "waste_qty": "损耗数量",
                           "temperature": "气温", "precipitation": "降水量",
                           "discount": "折扣", "near_expiry": "临期数量",
                           "shelf_life": "保质期", "lead_time": "补货周期"}
            # 2) 异常值截断(IQR法):只动明显不对的数字,正常数据不碰
            for _col in ["sales", "price", "base_price", "cost", "inventory", "waste_qty"]:
                if _col not in df.columns:
                    continue
                try:
                    _s = df[_col]
                    if not pd.api.types.is_numeric_dtype(_s) or _s.dropna().empty:
                        continue
                    _q1, _q3 = _s.quantile(0.25), _s.quantile(0.75)
                    _iqr = _q3 - _q1
                    if pd.isna(_iqr) or _iqr == 0:
                        continue
                    _lo, _hi = _q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr
                    _n_clip = int(((_s < _lo) | (_s > _hi)).sum())
                    if _n_clip > 0:
                        df[_col] = _s.clip(_lo, _hi)
                        self.cleaning_report.append(
                            f"{_col_labels[_col]}列有 {_n_clip} 个明显不对的数字，已自动修正到合理范围")
                except Exception as e:
                    print(f"[清洗跳过] 异常值截断({_col})失败:{type(e).__name__}: {e}")
            # 3) 缺失值补齐(只补部分缺失的列;缺失比例≥50%的列保持空,让下游用默认值兜底)
            #    3a) 数值列:按中间值(中位数)补
            #    (保质期/补货周期不在此列——不同商品差异太大,中位数没意义,
            #     改由下方按商品名称/分类关键词逐个推断)
            for _col in ["price", "base_price", "cost", "inventory", "waste_qty",
                         "temperature", "precipitation", "discount", "near_expiry"]:
                if _col not in df.columns:
                    continue
                try:
                    _s = df[_col]
                    if not pd.api.types.is_numeric_dtype(_s):
                        continue
                    _n_miss = int(_s.isna().sum())
                    if _n_miss == 0 or _n_miss / len(df) >= 0.5:
                        continue
                    _med = _s.median()
                    if pd.isna(_med):
                        continue
                    df[_col] = _s.fillna(_med)
                    self.cleaning_report.append(
                        f"{_col_labels.get(_col, _col)}列有 {_n_miss} 个空缺，已按中间值补上")
                except Exception as e:
                    print(f"[清洗跳过] 缺失值补齐({_col})失败:{type(e).__name__}: {e}")
            #    3b) 分类列:按最常见的值(众数)补
            for _col, _label in [("category", "商品分类"), ("holiday_name", "节假日名称"), ("unit", "单位")]:
                if _col not in df.columns:
                    continue
                try:
                    _s = df[_col]
                    _n_miss = int(_s.isna().sum())
                    if _n_miss == 0 or _n_miss / len(df) >= 0.5:
                        continue
                    _mode = _s.mode()
                    if _mode.empty:
                        continue
                    df[_col] = _s.fillna(_mode.iloc[0])
                    self.cleaning_report.append(
                        f"{_label}列有 {_n_miss} 个空缺，已按最常见的值补上")
                except Exception as e:
                    print(f"[清洗跳过] 缺失值补齐({_col})失败:{type(e).__name__}: {e}")
            #    3c) 布尔列:缺失按"否"处理
            for _col, _label, _msg in [("is_promo", "促销标记", "按'没有促销'处理"),
                                       ("is_holiday", "节假日标记", "按'不是节假日'处理"),
                                       ("stockout", "缺货标记", "按'没有缺货'处理")]:
                if _col not in df.columns:
                    continue
                try:
                    _s = df[_col]
                    _n_miss = int(_s.isna().sum())
                    if _n_miss == 0:
                        continue
                    df[_col] = _s.fillna(0)
                    self.cleaning_report.append(f"{_label}列有 {_n_miss} 个空缺，{_msg}")
                except Exception as e:
                    print(f"[清洗跳过] 缺失值补齐({_col})失败:{type(e).__name__}: {e}")
        # 智能推断缺失字段(有依据才推,不凭空造)
        inferred = []
        inferred_field_names = set()
        if "cost" not in df.columns and "base_price" in df.columns:
            df["cost"] = (df["base_price"] * 0.6).round(2)
            inferred.append("成本(按标准售价×0.6估算)")
            inferred_field_names.add("cost")
        elif "cost" not in df.columns and "price" in df.columns:
            df["cost"] = (df["price"] * 0.6).round(2)
            inferred.append("成本(按售价×0.6估算)")
            inferred_field_names.add("cost")
        if "near_expiry_price" not in df.columns and "price" in df.columns:
            df["near_expiry_price"] = (df["price"] * 0.5).round(2)
            inferred.append("临期处理售价(按售价×0.5估算)")
            inferred_field_names.add("near_expiry_price")
        elif "near_expiry_price" not in df.columns and "cost" in df.columns:
            df["near_expiry_price"] = (df["cost"] * 0.5).round(2)
            inferred.append("临期处理售价(按成本×0.5估算)")
            inferred_field_names.add("near_expiry_price")
        # 保质期/补货周期:按商品名称/分类关键词智能推断(饮料90天、肉类2天等)
        #    推断失败只跳过该字段,不影响数据本身可用
        for _field, _default, _infer_fn, _label in [
            ("shelf_life", 3, _infer_shelf_life, "保质期"),
            ("lead_time", 1, _infer_lead_time, "补货周期"),
        ]:
            try:
                _reports = _apply_field_inference(df, _field, _default, _infer_fn, _label)
                if _reports:
                    inferred.extend(_reports)
                    inferred_field_names.add(_field)
            except Exception as e:
                print(f"[推断跳过] {_label}推断失败:{type(e).__name__}: {e}")
                # 兜底:推断失败时该字段若存在仍补默认值,避免下游拿到全空列
                try:
                    if _field in df.columns and df[_field].isna().any():
                        df[_field] = df[_field].fillna(_default)
                        inferred.append(f"{_label}(按默认{_default}天)")
                        inferred_field_names.add(_field)
                except Exception:
                    pass
        self.inferred_fields = inferred
        self.inferred_field_names = inferred_field_names
        for bool_col in ["is_promo", "is_weekend", "is_holiday", "stockout"]:
            if bool_col in df.columns:
                try:
                    df[bool_col] = df[bool_col].apply(self._to_bool)
                except Exception as e:
                    print(f"[清洗跳过] 布尔转换({bool_col})失败:{type(e).__name__}: {e}")
        # 自动补充周末标记
        if "date" in df.columns and "is_weekend" not in df.columns:
            df["is_weekend"] = df["date"].dt.dayofweek >= 5
            df["is_weekend"] = df["is_weekend"].astype(int)
        self.df = df
        self._detect_tier()
        self._build_profile()
        return self

    def remap_column(self, column: str, field) -> "DataAdapter":
        """商户手动指定某列的含义,然后重新标准化整个数据集

        Args:
            column: 原始列名(必须存在于raw_df中)
            field:  标准字段名(如"sales"),传None/空字符串表示标记为"不使用"
        Returns:
            self(链式调用)
        """
        if self.raw_df is None:
            raise ValueError("未加载数据")
        if column not in self.raw_df.columns:
            raise ValueError(f"列「{column}」不存在")
        if field and field not in STANDARD_FIELDS:
            raise ValueError(f"不支持的字段类型: {field}")
        # 检查目标字段是否已被其他列占用(自动识别或手动指定的)
        if field:
            for col, std in self.column_mapping.items():
                if col != column and std == field:
                    raise ValueError(f"「{field}」已被列「{col}」使用,请先取消该列的映射")
        self.column_overrides[column] = field if field else None
        return self.standardize()

    # ──────────── 补充上传:建议与合并 ────────────
    def get_supplement_suggestions(self) -> dict:
        """分析当前数据缺什么关键信息,返回给商户的补充上传建议(通俗语言,不用专业术语)

        Returns:
            {"suggestions": [{field,label,benefit,example},...],
             "has_suggestions": bool, "message": str}
        """
        if self.df is None:
            raise ValueError("未加载数据")
        cols = set(self.df.columns)
        # 系统估算出来的字段也算"缺失":估算值不如商户的真实数据准,仍建议补充
        inferred_names = getattr(self, "inferred_field_names", set())

        def _has_data(col: str) -> bool:
            """列存在且至少有一个非空值才算有数据"""
            return col in cols and self.df[col].notna().any()

        missing = []
        # 进阶字段:有任意一个就能支撑价格/促销/天气/节假日分析
        for f in ["price", "is_promo", "temperature", "is_holiday"]:
            if not _has_data(f):
                missing.append(f)
        # 完整字段:损耗+成本,缺任一个都到不了完整版
        for f in ["waste_qty", "cost"]:
            if not _has_data(f) or f in inferred_names:
                missing.append(f)

        suggestions = []
        for f in missing:
            info = SUPPLEMENT_FIELD_INFO[f]
            suggestions.append({"field": f, "label": info["label"],
                                "benefit": info["benefit"], "example": info["example"]})
        if not suggestions:
            return {"suggestions": [], "has_suggestions": False,
                    "message": "您的数据很完整，所有分析都能用"}
        return {"suggestions": suggestions, "has_suggestions": True,
                "message": "如果您的表格里有这些信息，上传补充后能解锁更多分析"}

    def _recognize_supplement_columns(self, supp_raw: pd.DataFrame) -> pd.DataFrame:
        """识别补充数据的列名并做最小限度的类型转换

        只做列名识别+类型转换,不走完整清洗流程,避免混入系统估算值
        (估算值应该让位给商户补充的真实数据)。
        """
        df = supp_raw.copy()
        # 列名识别:与主流程同一套模糊匹配逻辑
        rename = {}
        for col in df.columns:
            std = _fuzzy_match(col)
            if std and std not in rename.values():
                rename[col] = std
        df = df.rename(columns=rename)
        # 编号兜底:与主流程一致(只有门店编号时把门店当商品)
        if "sku_id" not in df.columns and "store_id" in df.columns:
            df["sku_id"] = df["store_id"]
        # 类型转换(日期/编号/数值/布尔)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "sku_id" in df.columns:
            df["sku_id"] = df["sku_id"].astype(str).str.strip()
        for num_col in ["sales", "price", "base_price", "temperature", "precipitation",
                        "inventory", "waste_qty", "cost", "discount", "near_expiry",
                        "near_expiry_price", "shelf_life", "lead_time",
                        "gross_margin", "stockout_loss"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
        for bool_col in ["is_promo", "is_holiday", "is_weekend", "stockout"]:
            if bool_col in df.columns:
                df[bool_col] = df[bool_col].apply(self._to_bool)
        return df

    def merge_supplement(self, supp_raw: pd.DataFrame, supp_name: str = "补充数据") -> dict:
        """将补充数据合并到当前数据(按日期+商品编号匹配,只填空位,不覆盖已有数据)

        合并后自动重新走完整标准化流程(清洗/分档/画像)。
        补充数据没有日期或商品编号时,若行数与当前数据完全一致则按行序对齐,
        否则报错提示需要包含日期和商品编号。

        Args:
            supp_raw: 补充上传的原始 DataFrame(列名尚未识别)
            supp_name: 补充数据来源名(用于更新 source_name)
        Returns:
            {"new_fields": [新增的字段名], "merged_fields": [真正补进数据的字段名],
             "filled_rows": 补了多少行}
        Raises:
            ValueError: 补充数据无法识别/无法对齐/完全对不上号
        """
        if self.df is None:
            raise ValueError("未加载数据")
        supp = self._recognize_supplement_columns(supp_raw)

        # 补充数据能提供的字段(排除关联用的键字段)
        KEY_FIELDS = {"date", "sku_id", "store_id"}
        supp_fields = [c for c in supp.columns if c not in KEY_FIELDS]
        if not supp_fields:
            raise ValueError("补充数据里没有能识别的列，请检查列名（比如：日期、商品编号、售价、是否促销）")

        current_df = self.df.copy()
        # 系统估算出来的字段先移除,让补充的真实数据顶上(重新标准化时会按需再估算)
        inferred_names = getattr(self, "inferred_field_names", set())
        for f in list(inferred_names):
            if f in current_df.columns and f in supp_fields:
                current_df = current_df.drop(columns=[f])

        existing_fields = [f for f in supp_fields if f in current_df.columns]
        new_fields = [f for f in supp_fields if f not in current_df.columns]

        if "date" in supp.columns and "sku_id" in supp.columns:
            # ── 按日期+商品编号匹配 ──
            supp = supp.dropna(subset=["date"])
            if len(supp) == 0:
                raise ValueError("补充数据里的日期无法识别，请检查日期格式（比如: 2026-08-01）")
            supp_merge = supp[["date", "sku_id"] + supp_fields].copy()
            supp_merge = supp_merge.dropna(subset=supp_fields, how="all")
            supp_merge = supp_merge.drop_duplicates(subset=["date", "sku_id"], keep="first")
            if len(supp_merge) == 0:
                raise ValueError("补充数据里没有有效内容，请检查是否有空行")
            merged = current_df.merge(supp_merge, on=["date", "sku_id"], how="left",
                                      suffixes=("", "__supp"))
        elif len(supp) == len(current_df):
            # ── 没有日期/商品编号,但行数完全一致:按行序对齐 ──
            current_df = current_df.reset_index(drop=True)
            supp = supp.reset_index(drop=True)
            current_df["__row_id"] = range(len(current_df))
            supp_merge = supp[supp_fields].copy()
            supp_merge["__row_id"] = range(len(supp_merge))
            supp_merge = supp_merge.dropna(subset=supp_fields, how="all")
            merged = current_df.merge(supp_merge, on="__row_id", how="left",
                                      suffixes=("", "__supp"))
            merged = merged.drop(columns=["__row_id"])
        else:
            raise ValueError("补充数据也需要包含日期和商品编号，这样才能和您现有的数据对上号")

        # 统计并填充:已有字段只补空位,新字段整列加入
        merged_fields = []
        new_columns_added = []
        filled_row_mask = pd.Series(False, index=merged.index)
        matched_any = False
        for field in supp_fields:
            if field in existing_fields:
                supp_col = f"{field}__supp"
                if supp_col not in merged.columns:
                    continue
                if merged[supp_col].notna().any():
                    matched_any = True
                fill_mask = merged[field].isna() & merged[supp_col].notna()
                if fill_mask.any():
                    merged_fields.append(field)
                    filled_row_mask |= fill_mask
                merged[field] = merged[field].fillna(merged[supp_col])
                merged = merged.drop(columns=[supp_col])
            else:
                # 新字段:merge结果里直接叫field(未匹配的行是空值)
                if field not in merged.columns:
                    continue
                new_columns_added.append(field)
                if merged[field].notna().any():
                    matched_any = True
                    merged_fields.append(field)
                    filled_row_mask |= merged[field].notna()

        if not matched_any:
            raise ValueError("补充数据的日期和商品编号与现有数据对不上，请检查是否是同一批商品和日期范围")

        filled_rows = int(filled_row_mask.sum())
        # 用合并后的数据重新走完整标准化流程(清洗/分档/画像都会刷新)
        self.raw_df = merged
        # 只保留合并后仍然存在的列的手动映射(列名可能已变化)
        self.column_overrides = {k: v for k, v in self.column_overrides.items()
                                 if k in merged.columns}
        self.source_name = f"{self.source_name} + {supp_name}"
        self.standardize()
        return {"new_fields": new_columns_added, "merged_fields": merged_fields,
                "filled_rows": filled_rows}

    @staticmethod
    def _to_bool(val) -> int:
        """转布尔:0/false/否/空→0,其余(含a/b/c等标记)→1"""
        if isinstance(val, (bool, np.bool_)):
            return int(val)
        if isinstance(val, (int, float)):
            return int(val > 0)
        s = str(val).strip().lower()
        if s in ("0", "false", "否", "no", "n", "f", "nan", "", "none"):
            return 0
        return 1

    # ──────────── 分档检测 ────────────
    def _detect_tier(self):
        """根据字段完整性判定数据档次"""
        if self.df is None:
            self.tier = "basic"
            return
        cols = set(self.df.columns)
        def _has_data(col):
            """列存在且至少有一个非空值才算有效"""
            return col in cols and self.df[col].notna().any()
        has_advanced = _has_data("price") or _has_data("is_promo") or _has_data("temperature") or _has_data("is_holiday")
        has_full = _has_data("waste_qty") and _has_data("cost")
        if has_full:
            self.tier = "full"
        elif has_advanced:
            self.tier = "advanced"
        else:
            self.tier = "basic"

    # ──────────── 数据画像 ────────────
    def _build_profile(self):
        """构建数据质量画像"""
        if self.df is None:
            return
        df = self.df
        n_skus = df["sku_id"].nunique() if "sku_id" in df.columns else 0
        date_range = {}
        if "date" in df.columns:
            date_range = {"start": str(df["date"].min().date()), "end": str(df["date"].max().date()),
                          "days": int((df["date"].max() - df["date"].min()).days) + 1}
        missing = {col: round(float(df[col].isna().mean()) * 100, 1) for col in df.columns if df[col].isna().any()}
        # 缺货检测
        stockout_rate = 0.0
        if "sales" in df.columns:
            zero_sales = (df["sales"] == 0).mean()
            stockout_rate = round(float(zero_sales) * 100, 1)
        # 解锁功能
        unlocked = self._unlocked_features()
        self.profile = {
            "source": self.source_name,
            "tier": self.tier,
            "tier_label": {"basic": "基础版", "advanced": "进阶版", "full": "完整版"}[self.tier],
            "rows": len(df),
            "n_skus": int(n_skus),
            "n_stores": int(df["store_id"].nunique()) if "store_id" in df.columns else 1,
            "date_range": date_range,
            "columns": list(df.columns),
            "field_mapping": self.field_mapping,
            "missing_rates": missing,
            "stockout_rate_pct": stockout_rate,
            "unlocked_features": unlocked,
            "inferred_fields": getattr(self, "inferred_fields", []),
            "cleaning_report": getattr(self, "cleaning_report", []),
            "unmatched_columns": [c for c, v in getattr(self, "column_mapping", {}).items() if v is None],
            "empty_columns": getattr(self, "empty_columns", []),
            "column_mapping": getattr(self, "column_mapping", {}),
            "data_source_type": "custom",
        }

    def _unlocked_features(self) -> list:
        feats = ["基础需求预测", "缺货校正补货建议", "基础损耗风险估算"]
        if self.tier in ("advanced", "full"):
            feats += ["完整因果需求预测", "促销/价格干预模拟", "动态补货优化", "分品类损耗归因分析"]
        if self.tier == "full":
            feats += ["全链路损耗控制", "精准不对称损失计算", "最优临期折扣建议", "极端场景压力测试"]
        return feats

    # ──────────── 内置数据集 ────────────
    BUILTIN_DATASETS = [
        {"id": "full_retail", "name": "完整版示例数据(推荐)", "file": "full_retail_data.csv",
         "desc": "12个SKU×90天,含价格/促销/天气/节假日/损耗/成本全字段"},
        {"id": "advanced_retail", "name": "进阶版示例数据", "file": "advanced_retail_data.csv",
         "desc": "含价格/促销/天气/节假日,无损耗与成本字段"},
        {"id": "basic_retail", "name": "基础版示例数据", "file": "basic_retail_data.csv",
         "desc": "仅含日期/SKU/销量/库存,最低门槛体验"},
    ]

    @classmethod
    def list_builtin(cls, datasets_dir: Path) -> list:
        """列出可用的内置数据集"""
        result = []
        for ds in cls.BUILTIN_DATASETS:
            path = Path(datasets_dir) / ds["file"]
            entry = dict(ds)
            entry["available"] = path.exists()
            if path.exists():
                entry["size_kb"] = round(path.stat().st_size / 1024, 1)
            result.append(entry)
        return result

    @classmethod
    def load_builtin(cls, datasets_dir: Path, dataset_id: str) -> "DataAdapter":
        """加载内置数据集"""
        for ds in cls.BUILTIN_DATASETS:
            if ds["id"] == dataset_id:
                path = Path(datasets_dir) / ds["file"]
                if not path.exists():
                    raise FileNotFoundError(f"内置数据集不存在: {path}")
                return cls().load_file(str(path)).standardize()
        raise ValueError(f"未知的内置数据集: {dataset_id}")

    @classmethod
    def merge_adapters(cls, adapters: list) -> "DataAdapter":
        """合并多个已标准化的DataAdapter为一个(按日期+商品编号外连接,缺失字段留空)"""
        if not adapters:
            raise ValueError("没有可合并的数据")
        if len(adapters) == 1:
            return adapters[0]
        # 校验:每个数据集都必须有日期和商品编号才能按键合并
        for a in adapters:
            if a.df is None or "date" not in a.df.columns or "sku_id" not in a.df.columns:
                raise ValueError(f"文件「{a.source_name}」缺少日期或商品编号列,无法与其他文件合并")
        merged = adapters[0].df.copy()
        for a in adapters[1:]:
            right = a.df.copy()
            merged = pd.merge(merged, right, on=["date", "sku_id"], how="outer", suffixes=("", "_dup"))
            # 处理重名列:取非空值
            for col in list(merged.columns):
                if col.endswith("_dup"):
                    base = col[:-4]
                    if base in merged.columns:
                        merged[base] = merged[base].fillna(merged[col])
                        merged = merged.drop(columns=[col])
                    else:
                        merged = merged.rename(columns={col: base})
        # 同一商品内填充文本属性(商品名/品类/单位)
        for col in ["sku_name", "category", "unit"]:
            if col in merged.columns:
                merged[col] = merged.groupby("sku_id")[col].transform(lambda s: s.ffill().bfill())
        merged = merged.sort_values(["sku_id", "date"]).reset_index(drop=True)
        adapter = cls()
        adapter.raw_df = merged
        adapter.source_name = " + ".join(a.source_name for a in adapters)
        adapter.standardize()
        return adapter


# ──────────── 全局数据会话管理 ────────────────────
class DataSession:
    """管理当前活跃的数据集(单例)"""

    _instance = None

    def __init__(self):
        self.adapter: Optional[DataAdapter] = None
        self.excluded_factors: set = set()  # 用户排除不参与分析的因素
        # 优先读取环境变量(支持PyInstaller打包后写入exe同级目录)
        import os
        env_dir = os.environ.get("FRESH_AGENT_DATA_DIR")
        if env_dir:
            self.datasets_dir = Path(env_dir)
        else:
            self.datasets_dir = Path(__file__).parent.parent / "datasets"

    @classmethod
    def get(cls) -> "DataSession":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_adapter(self, adapter: DataAdapter):
        self.adapter = adapter

    def get_df(self) -> pd.DataFrame:
        if self.adapter is None or self.adapter.df is None:
            raise ValueError("尚未加载数据,请先上传数据或选择内置数据集")
        return self.adapter.df

    def get_tier(self) -> str:
        return self.adapter.tier if self.adapter else "basic"

    def is_loaded(self) -> bool:
        return self.adapter is not None and self.adapter.df is not None

    def set_excluded_factors(self, factors: list):
        """设置排除的因素列表(这些因素不参与因果分析和模拟)"""
        self.excluded_factors = set(factors)

    def get_excluded_factors(self) -> list:
        return list(self.excluded_factors)