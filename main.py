"""
生鲜零售智能补货与损耗控制 Agent - 主入口
==========================================
基于因果推断引擎 + 不对称损失补货优化的轻量化因果决策智能体

启动方式:
    开发模式:  pip install -r requirements.txt && python main.py
    打包后:    双击 FreshAgent.exe,浏览器自动打开

    访问地址: http://localhost:8000
"""
import os
import sys
import webbrowser
import threading
from pathlib import Path

# Windows 打包环境下控制台默认 GBK 编码,打印 ✓ 等字符会崩溃,统一改为 UTF-8 容错输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ──────────── 路径适配(兼容PyInstaller打包) ────────────
def is_frozen() -> bool:
    """是否为PyInstaller打包环境"""
    return getattr(sys, "frozen", False)


def get_bundle_dir() -> Path:
    """打包资源目录(只读):PyInstaller解压目录 / 开发模式项目根目录"""
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).parent.resolve()


def get_app_dir() -> Path:
    """可写应用目录:打包后为exe所在目录 / 开发模式项目根目录"""
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).parent.resolve()


# 开发模式:确保项目根目录在 sys.path 中
ROOT_DIR = get_bundle_dir()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 数据目录(可写):打包后放在exe同级 data 文件夹,供示例数据生成与用户上传缓存
APP_DIR = get_app_dir()
DATA_DIR = APP_DIR / "data"
os.environ["FRESH_AGENT_DATA_DIR"] = str(DATA_DIR)


def ensure_sample_data():
    """首次启动时自动生成内置示例数据集"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    flag = DATA_DIR / ".generated"
    if not flag.exists():
        print("[启动] 首次运行,正在生成内置示例数据集...")
        from backend.core.data_generator import generate_all_datasets
        generate_all_datasets(DATA_DIR)
        flag.touch()
        print("[启动] 示例数据集生成完毕。")


def open_browser_later(delay: float = 2.0, port: int = 8000):
    """延迟打开浏览器,等待服务器就绪"""
    def _open():
        import time
        time.sleep(delay)
        webbrowser.open(f"http://localhost:{port}")
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def create_app():
    """创建并配置 FastAPI 应用"""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware

    from backend.routers import data, replenishment, forecast, causal, simulation, waste, chat, dashboard, evaluation

    app = FastAPI(
        title="生鲜零售智能补货与损耗控制 Agent",
        description="基于因果推断的智能补货决策与损耗控制系统",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册 API 路由
    app.include_router(data.router, prefix="/api/data", tags=["数据管理"])
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["仪表盘"])
    app.include_router(replenishment.router, prefix="/api/replenishment", tags=["补货决策"])
    app.include_router(forecast.router, prefix="/api/forecast", tags=["需求预测"])
    app.include_router(causal.router, prefix="/api/causal", tags=["因果推断"])
    app.include_router(simulation.router, prefix="/api/simulation", tags=["策略模拟"])
    app.include_router(waste.router, prefix="/api/waste", tags=["损耗分析"])
    app.include_router(chat.router, prefix="/api/chat", tags=["智能对话"])
    app.include_router(evaluation.router, prefix="/api/evaluation", tags=["策略评估"])

    # 前端静态文件(打包后从 _MEIPASS 读取)
    frontend_dir = get_bundle_dir() / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(str(frontend_dir / "index.html"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    return app


app = create_app()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="生鲜零售智能补货与损耗控制 Agent")
    parser.add_argument("--data", type=str, default=None, help="自定义数据文件路径(CSV/Excel/Parquet/JSON),启动时自动加载")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="服务端口(默认8000,可通过环境变量PORT覆盖)")
    args = parser.parse_args()

    ensure_sample_data()

    # 命令行指定自定义数据文件时,启动即加载(三重降级:自定义数据→内置数据→空启动)
    if args.data:
        from pathlib import Path as _P
        _data_path = _P(args.data)

        def _fallback_to_builtin(reason: str):
            """自定义数据不可用时,自动回退到推荐的内置示例数据集,保证服务开箱可用"""
            print(f"[警告] {reason},将自动加载内置示例数据集(完整版)")
            try:
                from backend.core.data_adapter import DataAdapter, DataSession
                from backend.core.analysis_service import AnalysisService
                _b = DataAdapter.load_builtin(DATA_DIR, "full_retail")
                _b.profile["data_source_type"] = "builtin"
                DataSession.get().set_adapter(_b)
                AnalysisService.get()._invalidate_cache()
                AnalysisService.get().preload_async()
                print(f"[启动] 已回退加载内置数据: {_b.source_name} ({len(_b.df)}行, {_b.df['sku_id'].nunique()}个商品)")
            except Exception as e2:
                print(f"[警告] 内置数据集加载也失败: {e2},请稍后在页面中手动选择数据集")

        if not _data_path.exists():
            _fallback_to_builtin(f"指定的数据文件不存在: {args.data}")
        else:
            try:
                from backend.core.data_adapter import DataAdapter, DataSession
                from backend.core.analysis_service import AnalysisService
                _adapter = DataAdapter().load_file(str(_data_path)).standardize()
                _cols = set(_adapter.df.columns)
                _missing = [c for c in ("sales", "date", "sku_id") if c not in _cols]
                if _missing:
                    _fallback_to_builtin(f"数据文件缺少必需字段:{_missing}")
                else:
                    DataSession.get().set_adapter(_adapter)
                    AnalysisService.get()._invalidate_cache()
                    AnalysisService.get().preload_async()
                    print(f"[启动] 已加载自定义数据: {_data_path.name} ({len(_adapter.df)}行, {_adapter.df['sku_id'].nunique()}个商品)")
            except Exception as e:
                _fallback_to_builtin(f"加载自定义数据失败: {e}")

    print("=" * 60)
    print("  生鲜零售智能补货与损耗控制 Agent v1.0")
    print(f"  访问地址: http://localhost:{args.port}")
    print(f"  API文档:  http://localhost:{args.port}/docs")
    print("  关闭此窗口即可停止服务")
    print("=" * 60)

    # 自动打开浏览器
    open_browser_later(2.0, args.port)

    try:
        # 直接传 app 对象(不用字符串导入,兼容PyInstaller)
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False, log_level="info")
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            print("\n[错误] 端口 8000 已被占用,可能已有一个实例在运行。")
            print("       请关闭其他实例后重试,或直接访问 http://localhost:8000")
        else:
            raise
    except KeyboardInterrupt:
        pass
    print("\n服务已停止。")
    if is_frozen():
        input("按回车键关闭窗口...")