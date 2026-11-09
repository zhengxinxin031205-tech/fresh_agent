"""
需求预测路由
============
- 全量SKU需求预测
- 单SKU需求预测(含历史对比与置信区间)
"""
from fastapi import APIRouter, HTTPException, Query
from backend.core.analysis_service import AnalysisService

router = APIRouter()


@router.get("/all")
async def forecast_all(horizon: int = Query(7, ge=1, le=30, description="预测天数")):
    """预测所有SKU未来N天的需求"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    return {"forecasts": svc.get_forecasts(horizon), "horizon": horizon}


@router.get("/sku/{sku_id}")
async def forecast_sku(sku_id: str, horizon: int = Query(7, ge=1, le=30)):
    """预测单个SKU未来N天的需求(含历史对比与缺货校正提升)"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    result = svc.get_forecast_sku(sku_id, horizon)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result