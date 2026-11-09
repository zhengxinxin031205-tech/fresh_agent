"""
补货决策路由
============
- 生成全量补货清单(Newsvendor不对称损失优化)
- 查询单个SKU的最优补货量
- 导出补货清单为CSV
"""
import csv
import io
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from backend.core.analysis_service import AnalysisService
from backend.core.replenishment_optimizer import ReplenishmentOptimizer

router = APIRouter()


@router.get("/list")
async def get_replenishment_list():
    """生成全量补货清单(按建议补货量降序)"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    return {"items": svc.get_replenishment_list()}


@router.get("/sku/{sku_id}")
async def get_sku_replenishment(sku_id: str):
    """查询单个SKU的最优补货量"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    recovered = svc.get_recovered_df()
    grp = recovered[recovered["sku_id"] == sku_id]
    if len(grp) == 0:
        raise HTTPException(status_code=404, detail=f"SKU {sku_id} 不存在")
    recent = grp.tail(30)
    demand_col = "recovered_demand" if "recovered_demand" in recent.columns else "sales"
    demand_mean = float(recent[demand_col].mean())
    demand_std = float(recent[demand_col].std())
    import numpy as np
    if np.isnan(demand_mean):
        demand_mean = 1.0
    if np.isnan(demand_std) or demand_std < 0.5:
        demand_std = max(demand_mean * 0.2, 0.5)
    current_inv = float(grp["inventory"].iloc[-1]) if "inventory" in grp.columns and grp["inventory"].notna().any() else 0
    in_transit = float(grp["in_transit"].iloc[-1]) if "in_transit" in grp.columns else 0
    optimizer = ReplenishmentOptimizer(recovered, svc.get_tier())
    return optimizer.optimize_sku(sku_id, demand_mean, demand_std, current_inv, in_transit)


@router.get("/export")
async def export_replenishment_csv():
    """导出补货清单为CSV文件"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    items = svc.get_replenishment_list()
    output = io.StringIO()
    writer = csv.writer(output)
    headers = ["SKU编码", "商品名称", "品类", "单位", "日均需求", "当前库存",
               "建议补货量", "服务水平%", "损耗风险", "保质期天", "补货周期天",
               "预期缺货损失", "预期过期损失"]
    writer.writerow(headers)
    for r in items:
        writer.writerow([
            r["sku_id"], r["sku_name"], r["category"], r["unit"],
            r["demand_mean"], r["current_inventory"], r["reorder_qty"],
            r["service_level"], r["waste_risk_level"], r["shelf_life"], r["lead_time"],
            r["expected_stockout_loss"], r["expected_overstock_loss"],
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=replenishment_list.csv"},
    )