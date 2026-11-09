# ============================================================
# 生鲜零售智能补货与损耗控制 Agent - Docker 部署配置
# 适用平台: JD DevCloud (京东云原生平台, 仅暴露 8080 端口)
# ============================================================

FROM python:3.11-slim

# scikit-learn / scipy 在 slim 镜像上需要 OpenMP 运行时库
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖清单并安装,利用 Docker 层缓存加速重复构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码(受 .dockerignore 过滤,测试/临时文件不会进入镜像)
COPY main.py .
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/

# 确保数据目录可写(用户上传缓存 + 示例数据生成),创建上传目录
RUN mkdir -p data/.uploads && chmod -R a+rwX data/

# DevCloud 平台约束: 必须监听 8080
ENV PORT=8080
EXPOSE 8080

# 前台启动;不传 --data,启动时不自动加载数据(由用户在页面手动选择数据集)
CMD ["python", "main.py", "--port", "8080"]