"""
因果推断路由
============
- 全局因果效应摘要(促销/节假日/周末/天气/价格弹性)
- 单SKU因果效应详情
- 因素贡献占比(方差分解)
"""
from fastapi import APIRouter, HTTPException, Query
from backend.core.analysis_service import AnalysisService
from backend.core.np_utils import to_native

router = APIRouter()


@router.get("/summary")
async def get_causal_summary():
    """获取跨SKU的因果效应汇总"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    tier = svc.get_tier()
    if tier == "basic":
        raise HTTPException(
            status_code=403,
            detail="因果推断分析需要进阶版或完整版数据(含价格/促销/天气字段),请上传进阶数据或切换进阶版示例数据集",
        )
    engine = svc.get_causal_engine()
    return to_native(engine.get_global_summary())


@router.get("/effects")
async def get_causal_effects(sku_id: str = Query(None, description="可选:指定SKU,不传则返回全部")):
    """获取因果效应估计详情(后门调整ATE)"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    tier = svc.get_tier()
    if tier == "basic":
        raise HTTPException(
            status_code=403,
            detail="因果推断分析需要进阶版或完整版数据,请上传进阶数据或切换进阶版示例数据集",
        )
    engine = svc.get_causal_engine()
    effects = engine.effects
    if sku_id:
        if sku_id not in effects:
            raise HTTPException(status_code=404, detail=f"SKU {sku_id} 无因果分析结果")
        return to_native({"sku_id": sku_id, **effects[sku_id]})
    return to_native({"effects": effects})


@router.get("/custom-factors")
async def get_custom_factors():
    """获取当前数据中检测到的自定义影响因素列表"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        return {"factors": []}
    engine = svc.get_causal_engine()
    return to_native({"factors": engine.get_custom_factors()})


# 已知因素的展示标签
KNOWN_FACTOR_LABELS = {
    "promo": "促销活动",
    "holiday": "节假日",
    "weekend": "周末效应",
    "temperature": "气温变化",
    "heavy_rain": "大雨天气",
}


@router.get("/factor-config")
async def get_factor_config():
    """获取全部可用因素及排除状态(供前端因素管理面板使用)"""
    import pandas as pd
    from backend.core.causal_engine import CORE_COLUMNS, KNOWN_FACTORS
    svc = AnalysisService.get()
    if not svc.is_loaded():
        return {"known_factors": [], "custom_factors": [], "excluded": []}
    df = svc.get_df()
    excluded = svc.session.get_excluded_factors()
    # 已知因素:数据中实际存在的才列出
    known = []
    col_map = {
        "promo": "is_promo", "holiday": "is_holiday", "weekend": "is_weekend",
        "temperature": "temperature", "heavy_rain": "precipitation",
    }
    for key, label in KNOWN_FACTOR_LABELS.items():
        col = col_map.get(key)
        if col and col in df.columns:
            known.append({"key": key, "label": label, "excluded": key in excluded})
    # 自定义因素:所有非核心非已知的数值列(不论是否排除,都列出供用户勾选)
    custom = []
    for col in df.columns:
        if col in CORE_COLUMNS or col in KNOWN_FACTORS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        custom.append({"key": col, "label": col, "excluded": col in excluded})
    return {"known_factors": known, "custom_factors": custom, "excluded": excluded}


@router.post("/factor-config")
async def set_factor_config(body: dict):
    """更新因素排除配置(排除的因素不参与因果分析和模拟)"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据")
    excluded = body.get("excluded", [])
    if not isinstance(excluded, list):
        raise HTTPException(status_code=422, detail="excluded 必须是列表")
    svc.set_excluded_factors(excluded)
    return {"ok": True, "excluded": excluded}


@router.get("/factors")
async def get_causal_factors(sku_id: str = Query(None, description="可选:指定SKU,不传则返回全部")):
    """获取各因果因素对销量变异的贡献占比(方差分解)"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    tier = svc.get_tier()
    if tier == "basic":
        raise HTTPException(
            status_code=403,
            detail="因果因素分析需要进阶版或完整版数据,请上传进阶数据或切换进阶版示例数据集",
        )
    engine = svc.get_causal_engine()
    return to_native(engine.discover_causal_factors(sku_id))