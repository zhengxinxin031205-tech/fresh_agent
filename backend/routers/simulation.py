"""
策略模拟路由
============
反事实仿真:输入拟执行的经营动作(调价/促销/天气假设/节假日),
输出对应的销量、库存、损耗变化模拟结果。
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from backend.core.analysis_service import AnalysisService
from backend.core.np_utils import to_native

router = APIRouter()


class SimulationRequest(BaseModel):
    sku_id: str = Field(..., description="SKU编码")
    discount: Optional[float] = Field(None, ge=0.3, le=1.0, description="折扣率(如0.8表示8折)")
    price_change_pct: Optional[float] = Field(None, description="价格变动百分比(如-10表示降价10%)")
    is_promo: Optional[bool] = Field(None, description="是否开启促销")
    is_holiday: Optional[bool] = Field(None, description="是否为节假日")
    temperature: Optional[float] = Field(None, description="假设气温(℃)")
    heavy_rain: Optional[bool] = Field(None, description="是否有大雨")
    custom: Optional[dict] = Field(None, description="自定义因素干预,键为因素列名,值为目标值")


@router.post("/intervention")
async def simulate_intervention(req: SimulationRequest):
    """反事实仿真:模拟经营干预对销量/损耗的影响"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    tier = svc.get_tier()
    if tier == "basic":
        raise HTTPException(
            status_code=403,
            detail="策略模拟需要进阶版或完整版数据(含价格/促销字段),请上传进阶数据或切换进阶版示例数据集",
        )
    engine = svc.get_causal_engine()
    interventions = {}
    if req.discount is not None:
        interventions["discount"] = req.discount
    if req.price_change_pct is not None:
        interventions["price_change_pct"] = req.price_change_pct
    if req.is_promo is not None:
        interventions["is_promo"] = 1 if req.is_promo else 0
    if req.is_holiday is not None:
        interventions["is_holiday"] = 1 if req.is_holiday else 0
    if req.temperature is not None:
        interventions["temperature"] = req.temperature
    if req.heavy_rain is not None:
        interventions["heavy_rain"] = 1 if req.heavy_rain else 0
    if req.custom:
        interventions.update(req.custom)

    result = engine.simulate_intervention(req.sku_id, interventions)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return to_native(result)