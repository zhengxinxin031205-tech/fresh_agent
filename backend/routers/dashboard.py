"""
仪表盘路由
==========
提供经营概览汇总数据:数据档次、SKU概况、缺货统计、因果摘要、损耗摘要、补货摘要
"""
from fastapi import APIRouter, HTTPException, Query
from backend.core.analysis_service import AnalysisService
from backend.core.np_utils import to_native

router = APIRouter()


@router.get("/summary")
async def get_dashboard_summary(period: str = Query("week", pattern="^(today|week|month|all)$")):
    """获取仪表盘汇总数据,period: today=今天 / week=最近7天 / month=最近30天 / all=全部"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    return svc.get_dashboard_data(period)


@router.get("/recovery-summary")
async def get_recovery_summary():
    """获取需求恢复(缺货校正)统计摘要"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    return svc.get_recovery_summary()


@router.get("/sku-list")
async def get_sku_list():
    """获取当前数据集中所有SKU的基础信息"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    df = svc.get_df()
    skus = []
    if "sku_id" in df.columns:
        for sku_id, grp in df.groupby("sku_id"):
            entry = {
                "sku_id": sku_id,
                "sku_name": grp["sku_name"].iloc[0] if "sku_name" in grp.columns else sku_id,
                "category": grp["category"].iloc[0] if "category" in grp.columns else "",
                "unit": grp["unit"].iloc[0] if "unit" in grp.columns else "件",
                "avg_daily_sales": round(float(grp["sales"].mean()), 1) if "sales" in grp.columns else 0,
                "shelf_life": int(grp["shelf_life"].iloc[0]) if "shelf_life" in grp.columns and grp["shelf_life"].notna().any() else None,
                "records": len(grp),
            }
            current_inv = float(grp["inventory"].iloc[-1]) if "inventory" in grp.columns and grp["inventory"].notna().any() else None
            if current_inv is not None:
                entry["current_inventory"] = round(current_inv, 1)
            skus.append(entry)
    return to_native({"skus": skus, "total": len(skus)})