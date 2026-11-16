# -*- coding: utf-8 -*-
"""
一键跑通全流程：阶段 1~11。
运行：.venv\\Scripts\\python.exe src\\run_all.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGES = [
    ("阶段 1 · 数据探查与质量校验", "stage1_overview"),
    ("阶段 2 · 清洗与特征工程", "stage2_cleaning"),
    ("阶段 3 · 描述性分析与可视化", "stage3_descriptive"),
    ("阶段 4 · 漏斗分析", "stage4_funnel"),
    ("阶段 5 · RFM 分层", "stage5_rfm"),
    ("阶段 6 · 留存与复购", "stage6_retention"),
    ("阶段 7 · 统计检验与用户聚类", "stage7_tests_clustering"),
    ("阶段 8 · SQL 化", "stage8_sql"),
    ("阶段 11 · Session 阈值敏感性", "stage11_session_sensitivity"),
    ("阶段 9 · 交付物（报告）", "stage9_report"),
    ("阶段 9.2 · 交互看板", "stage10_dashboard"),
]


def main() -> None:
    import importlib

    total = time.perf_counter()
    timings = []
    for name, module in STAGES:
        print(f"\n{'=' * 68}\n>>> {name}\n{'=' * 68}", flush=True)
        t0 = time.perf_counter()
        mod = importlib.import_module(module)
        try:
            mod.main()
        except Exception as e:  # pragma: no cover
            print(f"!! {name} 失败：{type(e).__name__}: {e}", flush=True)
            raise
        dt = time.perf_counter() - t0
        timings.append((name, dt))
        print(f"<<< {name} 用时 {dt:.1f}s", flush=True)

    print(f"\n{'=' * 68}")
    for name, dt in timings:
        print(f"{name:<34s} {dt:8.1f}s")
    print(f"{'合计':<34s} {time.perf_counter() - total:8.1f}s")


if __name__ == "__main__":
    main()
