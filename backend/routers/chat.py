"""
智能对话路由
============
接收中文自然语言提问,自动识别意图并调用对应算法模块,
返回格式化的分析结果与建议。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.core.analysis_service import AnalysisService

router = APIRouter()


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="用户自然语言输入")


class ChatResponse(BaseModel):
    reply: str
    intent: str
    intent_label: str
    params: dict
    data: object = None


@router.post("/message", response_model=ChatResponse)
async def process_chat_message(req: ChatRequest):
    """处理中文自然语言对话"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")
    result = svc.process_chat(req.text.strip())
    return ChatResponse(**result)


@router.get("/examples")
async def get_chat_examples():
    """获取对话示例问题"""
    return {
        "examples": [
            "明天生菜要进多少?",
            "帮我看看今天该进什么货?",
            "上周苹果为什么扔了那么多?",
            "牛奶做8折能多卖多少?",
            "下周番茄能卖多少?",
            "草莓还剩多少?",
            "草莓快过期了,打几折能卖掉?",
            "最近生意怎么样?",
        ]
    }