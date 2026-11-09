"""
并行计算工具
============
基于 joblib 线程池的通用并行执行工具。

为什么用线程而不是进程:
- Windows 下 FastAPI 应用用多进程有序列化(pickle)问题和 spawn 启动开销
- 核心计算是 numpy.linalg.lstsq,它会释放 GIL,线程并行同样能吃到多核加速
- 线程共享内存,DataFrame 等对象无需序列化即可直接传递
"""
import os
import time
import logging
from typing import Any, Callable, List, Sequence

from joblib import Parallel, delayed

# 性能日志记录器(统一用 "perf" 名,方便按名称过滤)
logger = logging.getLogger("perf")


def parallel_map(func: Callable, items: Sequence, desc: str = None) -> List[Any]:
    """
    并行执行 func(item),返回结果列表(顺序与 items 一致)。

    参数:
        func: 处理单个元素的函数
        items: 待处理的数据序列
        desc: 日志描述(如 "estimate_effects"),不填则不记录耗时日志

    规则:
        - 元素少于 4 个时直接串行执行,避免线程调度开销
        - 否则用全部 CPU 核心并行(线程模式)
        - 自动记录耗时日志,格式: [perf] 描述: N 项 in X.Xs (M 线程)
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    # 数据量太小,并行反而更慢,直接串行
    if n < 4:
        start = time.perf_counter()
        results = [func(item) for item in items]
        if desc:
            elapsed = time.perf_counter() - start
            logger.info("[perf] %s: %d 项 串行 in %.1fs", desc, n, elapsed)
        return results

    # 线程模式并行,占满所有 CPU 核心
    n_threads = min(n, os.cpu_count() or 1)
    start = time.perf_counter()
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(func)(item) for item in items
    )
    if desc:
        elapsed = time.perf_counter() - start
        logger.info("[perf] %s: %d 项 in %.1fs (%d 线程)", desc, n, elapsed, n_threads)
    return results