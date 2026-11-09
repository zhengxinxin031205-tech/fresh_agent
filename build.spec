# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 - 生鲜零售智能补货与损耗控制 Agent
========================================================
打包命令:  pyinstaller build.spec
输出目录:  dist/FreshAgent/
分发方式:  将整个 dist/FreshAgent 文件夹拷贝到目标电脑,双击 FreshAgent.exe 即可运行
"""
import sys
from pathlib import Path

# ── pandas Cython 扩展手动收集 ──
# Windows 安全策略阻止 PyInstaller 的 collect_submodules 导入扫描,
# 改为直接扫描文件系统,收集 pandas/_libs 下所有 .pyd 扩展
_site_pkg = Path('build_env/Lib/site-packages')

pandas_libs_hiddenimports = []
pandas_libs_binaries = []
for _pyd in (_site_pkg / 'pandas' / '_libs').rglob('*.pyd'):
    _mod = _pyd.stem.split('.')[0]  # 去掉 .cp314-win_amd64 后缀
    _rel_parent = _pyd.parent.relative_to(_site_pkg)
    _full_mod = '.'.join(_rel_parent.parts + (_mod,))
    pandas_libs_hiddenimports.append(_full_mod)
    pandas_libs_binaries.append((str(_pyd), str(_rel_parent)))

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=pandas_libs_binaries,
    datas=[
        # 前端页面打包到 _MEIPASS/frontend/
        ('frontend/index.html', 'frontend'),
    ],
    hiddenimports=[
        # uvicorn 动态导入的协议模块
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'uvicorn.protocols.websockets.websockets_impl',
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        # FastAPI 文件上传依赖
        'multipart',
        # pandas 内部C扩展(全量收集,避免漏检)
        *pandas_libs_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的大模块,减小体积
        # scipy / sklearn 已由 stat_utils.py 替代(省约78MB)
        # 注意: joblib 不能排除,parallel_utils.py 依赖它做线程并行
        'scipy',
        'sklearn',
        'sklearnex',
        'statsmodels',
        'threadpoolctl',
        'tkinter',
        'matplotlib',
        'pytest',
        'setuptools',
        'pip',
        'IPython',
        'jupyter',
        'notebook',
        # 深度学习/大数据处理大库(本项目不需要,可减数GB)
        'torch',
        'tensorboard',
        'dask',
        'distributed',
        'numba',
        'llvmlite',
        'pyarrow',
        'fsspec',
        # 注意: cloudpickle 不能排除,joblib 的 loky 后端导入时依赖它
        # 文档/开发工具
        'sphinx',
        'docutils',
        'pygments',
        'jinja2',
        # Windows COM 自动化(本地Web服务不需要)
        'win32com',
        'pythoncom',
        'pywintypes',
        'win32evtlog',
        'win32api',
        # 加密库(localhost HTTP不需要)
        'cryptography',
        'bcrypt',
        'pycparser',
        'cffi',
        # 其他不需要的可选依赖
        'psutil',
        'chardet',
        'tornado',
        'zmq',
        'pyzmq',
        'Bottleneck',
        'gcsfs',
        's3fs',
        'sympy',
        # Qt GUI 框架(Web应用不需要,可减约300MB)
        'PyQt6',
        'PyQt5',
        'PySide6',
        'PySide2',
        'qtpy',
        'shiboken6',
        # AWS SDK
        'botocore',
        'boto3',
        's3transfer',
        # 图像/文档处理(本项目仅处理CSV,不需要)
        'PIL',
        'Pillow',
        'lxml',
        'tables',
        # 其他冗余
        'pyreadstat',
        'odf',
        'xarray',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir 模式:二进制文件放到 COLLECT 中
    name='FreshAgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,            # 保留控制台窗口显示日志,关闭窗口即停止服务
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='FreshAgent',
)