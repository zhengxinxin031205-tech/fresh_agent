"""
分层数据验证器
==============
对加载后的数据做四层检查,发现问题输出具体警告但不阻断运行(降级策略)。
层级: 字段级 → 记录级 → SKU级 → 门店级
"""
import pandas as pd
from typing import List

def validate_data(df: pd.DataFrame) -> dict:
    """
    对标准化后的DataFrame做分层验证,返回 {passed: bool, warnings: [...], errors: [...], quality_score: float}
    errors: 阻断性问题(缺必需字段) — 由调用方决定是否阻断
    warnings: 降级警告(样本不足、数据质量等) — 不阻断
    quality_score: 0~100 数据质量评分
    """
    errors = []
    warnings = []

    # ── 字段级 ──
    required = {"date": "日期", "sku_id": "商品编号", "sales": "销量"}
    for col, label in required.items():
        if col not in df.columns:
            errors.append(f"缺少必需字段:{label}(列名:{col})")

    # 数据类型检查
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        warnings.append("日期列格式可能不正确,建议检查")
    if "sales" in df.columns and not pd.api.types.is_numeric_dtype(df["sales"]):
        errors.append("销量列必须是数字")

    if errors:
        return {"passed": False, "warnings": warnings, "errors": errors, "quality_score": 0}

    # ── 记录级 ──
    n_rows = len(df)
    if n_rows < 30:
        warnings.append(f"数据量太少(仅{n_rows}条),分析结果可能不可靠,建议至少30条记录")

    # 日期连续性检查(每个SKU内日期是否有大段缺失)
    if "date" in df.columns and "sku_id" in df.columns and n_rows > 0:
        date_range_days = (df["date"].max() - df["date"].min()).days + 1
        if date_range_days > 0:
            expected_rows = date_range_days * df["sku_id"].nunique()
            coverage = n_rows / expected_rows
            if coverage < 0.5:
                warnings.append(f"日期覆盖率仅{coverage:.0%},数据中大段日期缺失,预测精度会受影响")

    # 负值检查(标准化后应该已清洗,再确认一次)
    for col in ["sales", "price", "base_price", "cost"]:
        if col in df.columns and (df[col] < 0).any():
            n_neg = int((df[col] < 0).sum())
            warnings.append(f"列「{col}」有{n_neg}个负值,已自动按0处理")

    # 空值检查
    for col in ["sales", "date", "sku_id"]:
        if col in df.columns and df[col].isna().any():
            n_na = int(df[col].isna().sum())
            warnings.append(f"列「{col}」有{n_na}个空值")

    # ── SKU级 ──
    if "sku_id" in df.columns:
        sku_counts = df.groupby("sku_id").size()
        min_samples = 30
        insufficient = sku_counts[sku_counts < min_samples]
        if len(insufficient) > 0:
            pct = len(insufficient) / len(sku_counts) * 100
            examples = ", ".join(insufficient.head(3).index.astype(str))
            if len(insufficient) <= 3:
                warnings.append(f"以下商品记录数不足{min_samples}条,因果估计可能不准确:{examples}")
            else:
                warnings.append(f"{len(insufficient)}个商品(占{pct:.0%})记录数不足{min_samples}条,因果估计可能不准确,如:{examples}等")

    # ── 门店级 ──
    if "store_id" in df.columns:
        n_stores = df["store_id"].nunique()
        n_skus = df["sku_id"].nunique()
        if n_stores > 100:
            warnings.append(f"门店数量较多({n_stores}家),分析可能较慢")
        # 门店-SKU组合检查
        combos = df.groupby(["store_id", "sku_id"]).size() if n_stores > 1 else None
        if combos is not None and len(combos) > 10000:
            warnings.append(f"门店×商品组合数较多({len(combos)}),建议缩小分析范围")

    # ── 质量评分 ──
    score = 100
    score -= len(warnings) * 10
    score = max(score, 10)
    if n_rows >= 1000:
        score += 5
    score = min(score, 100)

    return {"passed": True, "warnings": warnings, "errors": errors, "quality_score": score}