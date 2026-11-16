# -*- coding: utf-8 -*-
"""通用小工具：markdown 表格渲染、结果落盘。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:,.4g}") -> str:
    """把 DataFrame 渲染成 markdown 表格（不依赖 tabulate）。"""
    d = df.copy()

    def _fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return "-"
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            return floatfmt.format(v)
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
