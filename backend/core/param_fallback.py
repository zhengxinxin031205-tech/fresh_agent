"""
三级参数回退机制
================
当某个商品的历史数据不足时,逐级回退到更粗粒度的参数来源:

- SKU级: 历史数据≥30天的商品,用自身数据计算参数,精度最高
- 品类级: 数据不足30天但品类内有≥15天数据,回退到同类商品的平均参数
- 全局级: 新品、数据极少的商品,回退到全店通用默认参数

线程安全: __init__ 一次性构建全部统计量,后续所有方法只读,parallel_map 并行安全。
所有面向商家的文案(level_label)用大白话,不堆学术术语。
"""
import numpy as np
import pandas as pd


class ParamFallback:
    """三级参数回退:SKU级 → 品类级 → 全局级"""

    SKU_MIN_DAYS = 30       # SKU级阈值:自身有效数据天数
    CATEGORY_MIN_DAYS = 15  # 品类级阈值:品类内合计有效数据天数

    # 全店通用默认参数(所有统计都失败时的最终兜底)
    GLOBAL_DEFAULTS = {
        "price": 5.0, "cost_ratio": 0.5, "shelf_life": 5, "lead_time": 1,
        "salvage_value_ratio": 0.3,
        "near_expiry_price": float("nan"), "stockout_loss": 2.0,
        "demand_mean": 1.0, "demand_std": 0.5,
    }

    # 面向商家的去术语化说明
    LEVEL_LABELS = {
        "sku": "按这个商品自己的历史数据算的",
        "category": "这个商品记录不多，参考了同类商品的平均情况",
        "global": "这个商品是新面孔，先按全店平均水平来",
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # 需求列: recovered_demand 优先,无则用 sales
        if "recovered_demand" in df.columns and df["recovered_demand"].notna().any():
            self._demand_col = "recovered_demand"
        elif "sales" in df.columns:
            self._demand_col = "sales"
        else:
            self._demand_col = None

        # 一次性构建全部统计量(后续只读,线程安全)
        self._sku_stats = self._build_sku_stats()
        self._category_params = self._build_category_params()
        self._global_params = self._build_global_params()
        self._sku_level = self._build_sku_levels()

    # ──────────── 内部构建 ────────────

    @staticmethod
    def _clean_category(cat) -> str:
        """清洗品类:空值/空串/'nan' 视为无品类"""
        if cat is None:
            return ""
        if isinstance(cat, float) and np.isnan(cat):
            return ""
        cat = str(cat).strip()
        if cat.lower() == "nan" or cat == "":
            return ""
        return cat

    @staticmethod
    def _is_nan(val) -> bool:
        """判断值是否为空(NaN 或 None)"""
        if val is None:
            return True
        if isinstance(val, float) and np.isnan(val):
            return True
        return False

    @staticmethod
    def _first_valid(grp: pd.DataFrame, col: str, use_mean: bool = False) -> float:
        """提取某列的有效值(均值或首个非空);全空返回 NaN"""
        if col not in grp.columns or not grp[col].notna().any():
            return float("nan")
        if use_mean:
            return float(grp[col].mean())
        return float(grp[col].dropna().iloc[0])

    def _build_sku_stats(self) -> dict:
        """每个 SKU 的统计量: n_days/demand_mean/demand_std/category/price/cost/shelf_life/lead_time"""
        stats = {}
        df = self.df
        demand_col = self._demand_col
        for sku_id, grp in df.groupby("sku_id"):
            # 需求统计: tail(30) 的非空值; n_days 为全序列非空天数
            if demand_col is not None and demand_col in grp.columns:
                demand_series = grp[demand_col].tail(30).dropna()
                n_days = int(grp[demand_col].notna().sum())
                if len(demand_series) > 0:
                    demand_mean = float(demand_series.mean())
                    demand_std = float(demand_series.std()) if len(demand_series) > 1 else float("nan")
                else:
                    demand_mean = float("nan")
                    demand_std = float("nan")
            else:
                n_days = 0
                demand_mean = float("nan")
                demand_std = float("nan")

            category = ""
            if "category" in grp.columns and len(grp) > 0:
                category = self._clean_category(grp["category"].iloc[0])

            stats[sku_id] = {
                "n_days": n_days,
                "demand_mean": demand_mean,
                "demand_std": demand_std,
                "category": category,
                "price": self._first_valid(grp, "price", use_mean=True),
                "cost": self._first_valid(grp, "cost"),
                "shelf_life": self._first_valid(grp, "shelf_life"),
                "lead_time": self._first_valid(grp, "lead_time"),
                # 临期处理售价:随促销变动,取均值;缺货客损:取均值
                "near_expiry_price": self._first_valid(grp, "near_expiry_price", use_mean=True),
                "stockout_loss": self._first_valid(grp, "stockout_loss", use_mean=True),
            }
        return stats

    @staticmethod
    def _nanmean(vals) -> float:
        """忽略 NaN 求均值;全空返回 NaN"""
        arr = np.array([v for v in vals if not ParamFallback._is_nan(v)], dtype=float)
        return float(arr.mean()) if len(arr) > 0 else float("nan")

    def _mean_params(self, items: list) -> dict:
        """对一组 SKU 统计量求均值(NaN 跳过)"""
        mean_price = self._nanmean([it["price"] for it in items])
        mean_cost = self._nanmean([it["cost"] for it in items])
        # cost_ratio: 有价格和成本时取比值,否则用默认 0.5
        if not self._is_nan(mean_price) and mean_price > 0 and not self._is_nan(mean_cost):
            cost_ratio = mean_cost / mean_price
        else:
            cost_ratio = self.GLOBAL_DEFAULTS["cost_ratio"]
        return {
            "price": mean_price,
            "cost": mean_cost,
            "cost_ratio": cost_ratio,
            "shelf_life": self._nanmean([it["shelf_life"] for it in items]),
            "lead_time": self._nanmean([it["lead_time"] for it in items]),
            "demand_mean": self._nanmean([it["demand_mean"] for it in items]),
            "demand_std": self._nanmean([it["demand_std"] for it in items]),
            "near_expiry_price": self._nanmean([it["near_expiry_price"] for it in items]),
            "stockout_loss": self._nanmean([it["stockout_loss"] for it in items]),
        }

    def _build_category_params(self) -> dict:
        """按品类聚合的平均参数(品类内所有 SKU 的均值)"""
        cat_data = {}
        for st in self._sku_stats.values():
            cat = st["category"]
            if cat:
                cat_data.setdefault(cat, []).append(st)
        return {cat: self._mean_params(items) for cat, items in cat_data.items()}

    def _build_global_params(self) -> dict:
        """全店平均参数(所有 SKU 的均值,无值时用 GLOBAL_DEFAULTS 兜底)"""
        all_items = list(self._sku_stats.values())
        if not all_items:
            return dict(self.GLOBAL_DEFAULTS)
        gp = self._mean_params(all_items)
        # 逐字段兜底到 GLOBAL_DEFAULTS
        for key in ["price", "cost_ratio", "shelf_life", "lead_time",
                    "demand_mean", "demand_std", "near_expiry_price", "stockout_loss"]:
            if self._is_nan(gp.get(key)):
                gp[key] = self.GLOBAL_DEFAULTS[key]
        return gp

    def _build_sku_levels(self) -> dict:
        """每个 SKU 的回退级别判定"""
        # 品类总天数: 品类内所有 SKU 的 n_days 之和
        cat_total_days = {}
        for st in self._sku_stats.values():
            cat = st["category"]
            if cat:
                cat_total_days[cat] = cat_total_days.get(cat, 0) + st["n_days"]

        levels = {}
        for sku_id, st in self._sku_stats.items():
            if st["n_days"] >= self.SKU_MIN_DAYS:
                levels[sku_id] = "sku"
            elif st["category"] and cat_total_days.get(st["category"], 0) >= self.CATEGORY_MIN_DAYS:
                levels[sku_id] = "category"
            else:
                levels[sku_id] = "global"
        return levels

    # ──────────── 公开接口(只读,线程安全) ────────────

    def get_level(self, sku_id: str) -> str:
        """返回 'sku'/'category'/'global'"""
        return self._sku_level.get(sku_id, "global")

    def get_level_label(self, sku_id: str) -> str:
        """返回面向商家的去术语化说明文案"""
        return self.LEVEL_LABELS.get(self.get_level(sku_id), self.LEVEL_LABELS["global"])

    def get_params(self, sku_id: str) -> dict:
        """
        返回 {level, level_label, demand_mean, demand_std, price, cost, shelf_life, lead_time}
        - SKU级: 用自身数据为主,个别缺失字段用品类/全局补
        - 品类级: 用品类平均,缺失用全局补
        - 全局级: 用全局默认
        """
        level = self.get_level(sku_id)
        gd = self._global_params
        st = self._sku_stats.get(sku_id)
        cat = st["category"] if st else ""
        cp = self._category_params.get(cat, {}) if cat else {}

        # 按级别确定候选值来源顺序(前者优先)
        if level == "sku":
            candidates = [st, cp, gd]
        elif level == "category":
            candidates = [cp, gd]
        else:
            candidates = [gd]

        def pick(field: str):
            """按候选顺序取第一个非空值;全空用 GLOBAL_DEFAULTS 兜底"""
            for c in candidates:
                if not c:
                    continue
                val = c.get(field)
                if not self._is_nan(val):
                    return val
            return self.GLOBAL_DEFAULTS.get(field)

        price = pick("price")
        shelf_life = pick("shelf_life")
        lead_time = pick("lead_time")
        demand_mean = pick("demand_mean")
        demand_std = pick("demand_std")

        # cost: 直接取,取不到用 price × cost_ratio(与原 price*0.5 兜底逻辑一致)
        cost = pick("cost")
        if self._is_nan(cost):
            ratio = pick("cost_ratio")
            cost = price * ratio if not self._is_nan(ratio) else price * self.GLOBAL_DEFAULTS["cost_ratio"]

        # 临期处理售价(残值): 优先用数据中的 near_expiry_price(>0 有效),
        # 无数据则按残值率兜底(采购成本×0.3,临期处理能回收30%成本)
        near_expiry_price = pick("near_expiry_price")
        if self._is_nan(near_expiry_price) or near_expiry_price <= 0:
            salvage_value = cost * self.GLOBAL_DEFAULTS["salvage_value_ratio"]
        else:
            salvage_value = near_expiry_price

        # 缺货额外损失(客损): 从数据取,无数据兜底2.0元/件
        stockout_loss = pick("stockout_loss")
        if self._is_nan(stockout_loss):
            stockout_loss = self.GLOBAL_DEFAULTS["stockout_loss"]

        return {
            "level": level,
            "level_label": self.get_level_label(sku_id),
            "demand_mean": demand_mean,
            "demand_std": demand_std,
            "price": price,
            "cost": cost,
            "shelf_life": shelf_life,
            "lead_time": lead_time,
            "salvage_value": salvage_value,
            "stockout_loss": stockout_loss,
        }

    def get_summary(self) -> dict:
        """返回各级别 SKU 数量与阈值"""
        return {
            "n_sku_level": sum(1 for lv in self._sku_level.values() if lv == "sku"),
            "n_category_level": sum(1 for lv in self._sku_level.values() if lv == "category"),
            "n_global_level": sum(1 for lv in self._sku_level.values() if lv == "global"),
            "sku_min_days": self.SKU_MIN_DAYS,
            "category_min_days": self.CATEGORY_MIN_DAYS,
        }