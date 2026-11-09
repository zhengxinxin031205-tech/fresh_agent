"""
内置示例数据集生成器
====================
生成三档示例数据(基础版/进阶版/完整版),模拟真实社区生鲜零售场景:
- 12个SKU覆盖蔬菜/水果/肉禽/乳制品四大品类
- 90天历史数据,含日销量、价格、促销、天气、节假日、库存、损耗等字段
- 内置缺货事件、促销活动、天气异常等真实业务场景
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# SKU 主数据: (sku_id, 名称, 品类, 基础日销量, 基础售价, 采购成本, 保质期天数, 补货周期天, 单位)
SKU_MASTER = [
    ("SKU001", "生菜",   "蔬菜",   35,  4.5,  2.2, 3,  1, "斤"),
    ("SKU002", "番茄",   "蔬菜",   50,  5.0,  2.8, 5,  1, "斤"),
    ("SKU003", "黄瓜",   "蔬菜",   40,  3.5,  1.8, 5,  1, "斤"),
    ("SKU004", "土豆",   "蔬菜",   60,  3.0,  1.5, 15, 3, "斤"),
    ("SKU005", "西兰花", "蔬菜",   25,  6.0,  3.0, 5,  2, "斤"),
    ("SKU006", "苹果",   "水果",   45,  7.0,  4.0, 20, 3, "斤"),
    ("SKU007", "香蕉",   "水果",   55,  4.0,  2.2, 7,  2, "斤"),
    ("SKU008", "草莓",   "水果",   20, 15.0,  9.0, 3,  1, "盒"),
    ("SKU009", "葡萄",   "水果",   30,  9.0,  5.5, 5,  2, "斤"),
    ("SKU010", "猪肉",   "肉禽",   40, 16.0, 12.0, 3,  1, "斤"),
    ("SKU011", "鸡胸肉", "肉禽",   30, 10.0,  7.0, 3,  1, "斤"),
    ("SKU012", "鲜牛奶", "乳制品", 60,  6.5,  4.0, 7,  2, "瓶"),
]

# 促销计划: (sku_id, 起始天, 持续天数, 折扣率)
PROMO_PLAN = [
    ("SKU002", 15, 3, 0.80), ("SKU006", 20, 5, 0.85), ("SKU008", 25, 2, 0.75),
    ("SKU010", 30, 3, 0.90), ("SKU012", 35, 4, 0.88), ("SKU001", 40, 2, 0.85),
    ("SKU007", 50, 3, 0.80), ("SKU009", 55, 4, 0.85), ("SKU003", 65, 3, 0.80),
    ("SKU006", 70, 5, 0.80), ("SKU012", 78, 3, 0.85), ("SKU010", 82, 2, 0.88),
]

# 节假日: (日期偏移, 名称)
HOLIDAYS = {5: "端午节", 12: "周末大促", 22: "中秋国庆", 45: "店庆日", 60: "周末特卖", 85: "元旦预热"}


def _gen_weather(n_days):
    """生成模拟天气数据(气温℃、降水mm)"""
    temps = np.zeros(n_days)
    precip = np.zeros(n_days)
    base_temp = 22.0
    for i in range(n_days):
        # 气温缓慢波动 + 噪声
        seasonal = 5 * np.sin(2 * np.pi * i / 30)
        temps[i] = base_temp + seasonal + np.random.normal(0, 3)
        # 15%概率下雨
        if np.random.random() < 0.15:
            precip[i] = np.random.exponential(8)
    return np.round(temps, 1), np.round(precip, 1)


def _is_promo(sku_id, day):
    for sid, start, dur, disc in PROMO_PLAN:
        if sid == sku_id and start <= day < start + dur:
            return True, disc
    return False, 1.0


def generate_full_dataset(n_days=90):
    """生成完整版数据集(含所有字段)"""
    temps, precip = _gen_weather(n_days)
    records = []
    for sku_id, name, cat, base_sales, price, cost, shelf, lead, unit in SKU_MASTER:
        # 每个SKU的库存策略: 基础订货量 = base_sales * (lead+1) * 1.1
        base_order = int(base_sales * (lead + 1) * 1.15)
        inventory = base_order
        for day in range(n_days):
            date = pd.Timestamp("2026-08-01") + pd.Timedelta(days=day)
            dow = date.dayofweek  # 0=周一
            is_weekend = 1 if dow >= 5 else 0
            is_holiday = 1 if day in HOLIDAYS else 0
            holiday_name = HOLIDAYS.get(day, "")
            is_promo, disc = _is_promo(sku_id, day)
            actual_price = round(price * disc, 2)

            # ---- 真实需求生成(因果机制) ----
            demand = base_sales
            demand *= 1.0 + 0.25 * is_weekend                      # 周末效应
            demand *= 1.0 + 0.35 * is_holiday                       # 节假日效应
            if is_promo:
                elasticity = -2.0 if cat in ("水果", "乳制品") else -1.5
                demand *= 1.0 + elasticity * (disc - 1.0)           # 价格弹性
            temp_dev = (temps[day] - 22) / 10
            if cat == "水果":
                demand *= 1.0 + 0.08 * temp_dev                     # 热天水果多卖
            if cat == "乳制品":
                demand *= 1.0 + 0.05 * temp_dev
            if precip[day] > 5:
                demand *= 0.85                                       # 大雨减少客流
            demand *= max(0.5, 1.0 + np.random.normal(0, 0.15))     # 随机波动
            demand = max(0, int(round(demand)))

            # ---- 库存与销量(截断机制) ----
            sales = min(demand, inventory)
            stockout = 1 if demand > inventory else 0
            # 补货: 库存低于安全线时到货
            safety = int(base_sales * 0.5)
            if inventory <= safety and day % lead == 0:
                inventory = inventory + base_order
            inventory = inventory - sales

            # ---- 损耗生成 ----
            # 损耗率与保质期负相关、与过量订货正相关
            overstock_ratio = max(0, (inventory - base_sales * 1.2) / max(1, base_sales))
            base_waste_rate = {3: 0.06, 5: 0.04, 7: 0.025, 15: 0.01, 20: 0.008}.get(shelf, 0.04)
            waste_rate = base_waste_rate + 0.03 * overstock_ratio + np.random.normal(0, 0.005)
            waste_rate = max(0, waste_rate)
            waste_qty = int(round(inventory * waste_rate))
            waste_qty = min(waste_qty, inventory)
            inventory = inventory - waste_qty

            # ---- 临期折扣处理 ----
            near_expiry_qty = int(round(inventory * min(0.15, waste_rate * 2)))
            discount_sale_price = round(actual_price * 0.5, 2)  # 临期半价处理

            records.append({
                "日期": date.strftime("%Y-%m-%d"),
                "SKU编码": sku_id, "SKU名称": name, "品类": cat,
                "当日销量": sales, "真实需求": demand,
                "售价": actual_price, "标准售价": price,
                "是否促销": is_promo, "折扣率": disc,
                "气温": temps[day], "降水量": precip[day],
                "是否周末": is_weekend, "是否节假日": is_holiday, "节假日名称": holiday_name,
                "当日库存": max(0, inventory), "是否缺货": stockout,
                "损耗数量": waste_qty, "临期处理数量": near_expiry_qty,
                "临期处理售价": discount_sale_price,
                "采购成本": cost, "保质期天数": shelf,
                "补货周期天": lead, "单位": unit,
            })
    return pd.DataFrame(records)


def generate_advanced_dataset(full_df):
    """进阶版:去掉损耗/成本相关字段"""
    drop_cols = ["真实需求", "损耗数量", "临期处理数量", "临期处理售价", "采购成本"]
    return full_df.drop(columns=[c for c in drop_cols if c in full_df.columns])


def generate_basic_dataset(full_df):
    """基础版:仅保留日期/SKU/销量/库存"""
    keep_cols = ["日期", "SKU编码", "SKU名称", "品类", "当日销量", "当日库存", "单位"]
    return full_df[keep_cols].copy()


def generate_all_datasets(output_dir: Path):
    """生成全部三档数据集到指定目录"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full = generate_full_dataset()
    full.to_csv(output_dir / "full_retail_data.csv", index=False, encoding="utf-8-sig")
    adv = generate_advanced_dataset(full)
    adv.to_csv(output_dir / "advanced_retail_data.csv", index=False, encoding="utf-8-sig")
    basic = generate_basic_dataset(full)
    basic.to_csv(output_dir / "basic_retail_data.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ 基础版: {len(basic)} 条记录, {len(basic.columns)} 个字段")
    print(f"  ✓ 进阶版: {len(adv)} 条记录, {len(adv.columns)} 个字段")
    print(f"  ✓ 完整版: {len(full)} 条记录, {len(full.columns)} 个字段")