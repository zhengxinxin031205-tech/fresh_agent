"""
损耗分析路由
============
- 损耗概览(总损耗金额/损耗率走势)
- 损耗归因(六因素贡献占比)
- Top高损耗SKU排名与优化建议
- 临期折扣优化(弹性驱动网格搜索)
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from backend.core.analysis_service import AnalysisService
from backend.core.np_utils import to_native

router = APIRouter()


@router.get("/overview")
async def get_waste_overview(period: str = Query("week", pattern="^(week|month)$")):
    """获取损耗概览:总损耗金额、损耗率走势"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    analyzer = svc.get_waste_analyzer()
    return to_native(analyzer.get_overview(period))


@router.get("/attribution")
async def get_waste_attribution(sku_id: str = Query(None, description="可选:指定SKU,不传则返回全部")):
    """损耗归因分析:拆解各因素贡献占比"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    analyzer = svc.get_waste_analyzer()
    result = analyzer.attribute_waste(sku_id)
    if "error" in result:
        raise HTTPException(status_code=403, detail=result["error"])
    return to_native(result)


@router.get("/top-skus")
async def get_top_waste_skus(top_n: int = Query(5, ge=1, le=20)):
    """获取损耗最高的SKU排名及优化建议"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    analyzer = svc.get_waste_analyzer()
    return to_native({"items": analyzer.top_waste_skus(top_n)})


@router.get("/estimate-elasticity")
async def estimate_elasticity(sku_id: str = Query(..., description="SKU编码")):
    """AI估算指定商品的价格敏感度(弹性系数),供临期折扣试算使用"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    return to_native(svc.get_estimated_elasticity(sku_id))


@router.post("/optimize-discount")
async def optimize_near_expiry_discount(
    sku_id: str = Query(..., description="SKU编码"),
    near_expiry_qty: float = Query(..., gt=0, description="临期商品数量"),
    elasticity: Optional[float] = Query(None, description="价格敏感度(负值);不传则AI自动估算"),
):
    """基于价格因果弹性计算最优临期折扣力度"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    # 未手动指定价格敏感度时,AI自动估算
    elasticity_source = "manual"
    source_label = "您手动填写的价格敏感度"
    if elasticity is None:
        est = svc.get_estimated_elasticity(sku_id)
        elasticity = est["elasticity"]
        elasticity_source = est["source"]
        source_label = est["source_label"]
    analyzer = svc.get_waste_analyzer()
    result = analyzer.optimize_discount(sku_id, near_expiry_qty, elasticity)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    result["elasticity_source"] = elasticity_source
    result["elasticity_source_label"] = source_label
    return to_native(result)