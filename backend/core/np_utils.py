"""
numpy → Python 原生类型转换工具
================================
核心算法模块基于 pandas/numpy 计算,产出的标量常为 numpy.int64 / numpy.float64
等类型。FastAPI 的 jsonable_encoder 无法识别这些类型(会尝试 vars() 和迭代,
均失败后抛 ValueError),导致所有返回计算结果的 API 端点 500。

典型泄漏场景:
- DataFrame.groupby 的键(sku_id/category 为 int64 列时 → numpy.int64)
- df[col].iloc[0](列 dtype 非 object 时返回 numpy 标量)
- np.clip / np.sqrt / np.mean 等运算结果(round() 不改变 numpy 类型)
- ParamFallback / 各分析模块返回 dict 中的计算字段

本模块提供 to_native() 递归转换,在数据离开计算层前统一转为 Python 原生类型。
"""
import math
import numpy as np


def to_native(obj):
    """递归将 numpy 标量/数组转换为 Python 原生类型,确保 JSON 可序列化。

    转换规则:
    - dict       → 递归转换值(键也转原生)
    - list/tuple → 递归转换元素
    - np.integer → int
    - np.floating→ float(NaN/Inf → None,因 JSON 标准不支持)
    - np.bool_   → bool
    - np.ndarray → list(递归)
    - np.str_ 等 → .item()
    - 其他类型   → 原样返回
    """
    if isinstance(obj, dict):
        return {_native_key(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _safe_float(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        # 其他 numpy 标量(np.str_ / np.datetime64 等)
        return to_native(obj.item())
    if isinstance(obj, float):
        # Python float 也做 NaN/Inf 防护(Starlette JSONResponse 用 allow_nan=False)
        return _safe_float(obj)
    return obj


def _safe_float(v: float):
    """NaN/Inf → None(JSON 无此值,用 null 表示);正常值原样返回"""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _native_key(k):
    """dict 键转原生类型(numpy 标量 → Python 标量,其他原样)"""
    if isinstance(k, np.integer):
        return int(k)
    if isinstance(k, np.floating):
        return _safe_float(float(k))
    if isinstance(k, np.generic):
        return k.item()
    return k