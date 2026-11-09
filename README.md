# 生鲜零售智能补货与损耗控制 Agent

基于**因果推断引擎**与**不对称损失补货优化**的轻量化决策智能体,面向中小生鲜零售主体,解决行业平均损耗率超15%的痛点。

## 核心能力

| 能力 | 说明 |
|------|------|
| 多源数据适配 | 支持CSV/Excel上传,26字段中英文别名自动映射,基础/进阶/完整三档自动检测 |
| 缺货截尾需求恢复 | Tobit思想截尾回归,恢复被缺货截断的真实需求,同品类因果校正 |
| 因果推断与干预模拟 | 后门调整准则估计促销/价格/节假日/天气的真实因果效应(排除混淆变量),支持反事实仿真 |
| 不对称损失补货决策 | Newsvendor模型,临界比率CR=Cu/(Cu+Co),动态安全库存,损耗风险分级 |
| 损耗归因与临期优化 | 六因素方差分解(订货过量/需求波动/天气/促销/保质期/残差),临期折扣网格搜索 |
| 对话式NLP交互 | 中文自然语言提问,关键词+正则意图识别,7种意图自动编排算法调用 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python main.py

# 3. 打开浏览器
# http://localhost:8000
```

首次启动自动生成三档示例数据集(基础版/进阶版/完整版),开箱即用。

## 数据档次说明

| 档次 | 必需字段 | 解锁功能 |
|------|---------|---------|
| 基础版 | 日期、SKU、销量 | 需求预测、基础补货建议 |
| 进阶版 | +价格、促销、天气、节假日 | 因果推断、干预模拟、弹性分析 |
| 完整版 | +损耗量、损耗金额、采购成本、售价 | 损耗归因、临期折扣优化、完整补货决策 |

## API 接口

| 模块 | 端点 | 说明 |
|------|------|------|
| 数据管理 | `GET /api/data/builtin` | 列出内置数据集 |
| | `POST /api/data/builtin/{id}` | 加载内置数据集 |
| | `POST /api/data/upload` | 上传CSV/Excel |
| | `GET /api/data/profile` | 数据画像 |
| | `GET /api/data/preview` | 数据预览 |
| 仪表盘 | `GET /api/dashboard/summary` | 经营汇总 |
| | `GET /api/dashboard/sku-list` | SKU列表 |
| 补货决策 | `GET /api/replenishment/list` | 全量补货清单 |
| | `GET /api/replenishment/sku/{id}` | 单SKU补货 |
| | `GET /api/replenishment/export` | 导出CSV |
| 需求预测 | `GET /api/forecast/all?horizon=N` | 全量预测 |
| | `GET /api/forecast/sku/{id}?horizon=N` | 单SKU预测 |
| 因果分析 | `GET /api/causal/summary` | 全局效应汇总 |
| | `GET /api/causal/effects` | ATE详情 |
| | `GET /api/causal/factors` | 因素贡献占比 |
| 策略模拟 | `POST /api/simulation/intervention` | 反事实仿真 |
| 损耗分析 | `GET /api/waste/overview` | 损耗概览 |
| | `GET /api/waste/attribution` | 六因素归因 |
| | `GET /api/waste/top-skus` | 高损耗排名 |
| | `POST /api/waste/optimize-discount` | 临期折扣优化 |
| 智能对话 | `POST /api/chat/message` | 对话交互 |
| | `GET /api/chat/examples` | 示例问题 |

## 项目结构

```
fresh_agent/
├── main.py                     # 入口:数据生成+FastAPI应用+uvicorn
├── requirements.txt            # 依赖清单
├── README.md
├── backend/
│   ├── core/                   # 核心算法层
│   │   ├── data_generator.py   # 示例数据生成(三档)
│   │   ├── data_adapter.py     # 多源数据适配(分档映射)
│   │   ├── demand_recovery.py  # 缺货截尾需求恢复
│   │   ├── causal_engine.py    # 因果推断引擎(后门调整)
│   │   ├── replenishment_optimizer.py  # Newsvendor补货优化
│   │   ├── waste_analyzer.py   # 损耗归因+临期折扣优化
│   │   ├── forecaster.py       # 需求预测(EWMA+星期效应)
│   │   ├── nlp_engine.py       # 中文NLP意图识别
│   │   └── analysis_service.py # 服务编排层(缓存管理)
│   └── routers/                # REST API层
│       ├── data.py             # 数据管理路由
│       ├── dashboard.py        # 仪表盘路由
│       ├── replenishment.py    # 补货决策路由
│       ├── forecast.py         # 需求预测路由
│       ├── causal.py           # 因果推断路由
│       ├── simulation.py       # 策略模拟路由
│       ├── waste.py            # 损耗分析路由
│       └── chat.py             # 智能对话路由
└── frontend/
    └── index.html              # 单页应用(Chart.js可视化)
```

## 技术栈

- **后端**: Python 3.9+, FastAPI, uvicorn, pandas, numpy, scikit-learn
- **前端**: 原生HTML/CSS/JS + Chart.js(CDN)
- **算法**: 截尾回归(Tobit)、后门调整因果推断、Newsvendor不对称损失、EWMA时间序列、log-log价格弹性

## 对话示例

- "明天生菜要补多少斤?"
- "为什么番茄这周损耗这么高?"
- "如果鸡蛋打8折,销量会涨多少?"
- "哪些商品受暴雨影响最大?"
- "给我一份全店补货建议"
- "生菜临期了50斤,打几折合适?"