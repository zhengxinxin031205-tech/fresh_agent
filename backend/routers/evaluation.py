"""
策略评估路由
============
- Bootstrap 重采样评估(日均损失与95%置信区间)
- 三条基线对比(昨日销量/历史均值/完美预知 vs Agent建议)
"""
from fastapi import APIRouter, HTTPException, Query
from backend.core.analysis_service import AnalysisService

router = APIRouter()


@router.get("/bootstrap")
async def evaluation_bootstrap(
    sku_id: str = Query(None, description="可选:指定SKU,不传则返回全部"),
    n_bootstrap: int = Query(1000, ge=100, le=5000),
    eval_days: int = Query(7, ge=1, le=30),
):
    """Bootstrap 重采样评估 Agent 补货策略的日均损失与 95% 置信区间"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    return svc.get_bootstrap_evaluate(sku_id, n_bootstrap, eval_days)


@router.get("/baseline")
async def evaluation_baseline(
    sku_id: str = Query(None, description="可选:指定SKU,不传则返回全部"),
    eval_days: int = Query(14, ge=1, le=30),
):
    """三条基线对比:昨日销量/历史均值/完美预知 vs Agent建议"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    return svc.get_baseline_comparison(sku_id, eval_days)