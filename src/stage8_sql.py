# -*- coding: utf-8 -*-
"""
阶段 8 · SQL 化（DuckDB 直连 3.6 GB CSV / 中间层 parquet）
=========================================================
把所有核心指标写进 sql/*.sql，本脚本负责批量执行并落盘结果。

产出：outputs/sql_results/<sql名>.csv，reports/stage8_sql_summary.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import db  # noqa: E402
import utils  # noqa: E402

DESC = {
    "01_dau.sql": "DAU 趋势：逐日活跃用户数、行为量、人均行为",
    "02_hourly_activity.sql": "小时活跃分布：定位晚间资源投放窗口",
    "03_funnel_uv.sql": "漏斗（UV 口径）：pv → (cart|fav) → buy，含顺序约束",
    "04_funnel_session.sql": "漏斗（Session 口径）：30 分钟会话内且顺序正确",
    "05_rfm.sql": "RFM 分层：五分位打分 + 8 类客户 + 购买贡献度",
    "06_retention.sql": "留存：次日留存与 7 日内回访（含窗口截断说明）",
    "07_repeat_purchase.sql": "复购率与购买跨度：复购定义 = 购买天数 ≥ 2",
    "08_top_category.sql": "Top20 类目：浏览量 vs 转化率（识别高流量低转化）",
    "09_behavior_share.sql": "行为类型构成与用户/商品覆盖",
    "10_sampling_bias.sql": "采样偏差对照：前 100 万行 vs 1% 分层采样",
}


def main() -> None:
    outdir = C.OUTPUT_DIR / "sql_results"
    outdir.mkdir(parents=True, exist_ok=True)

    con = db.connect(memory_limit="6GB")
    con.execute(f"SET temp_directory='{(C.OUTPUT_DIR / 'duckdb_tmp').as_posix()}'")
    # `raw` 由 db.connect() 注册；events / suspect_users 走中间层 parquet
    db.attach_parquet_layer(con)

    rows = []
    for f in sorted(C.SQL_DIR.glob("*.sql")):
        sql = f.read_text(encoding="utf-8")
        t0 = time.perf_counter()
        try:
            df = con.execute(sql).fetchdf()
            dt = time.perf_counter() - t0
            df.to_csv(outdir / f"{f.stem}.csv", index=False, encoding="utf-8-sig")
            rows.append([f.name, DESC.get(f.name, ""), len(df), dt, "OK"])
            print(f"[OK] {f.name:28s} {len(df):>6} 行  {dt:6.1f}s", flush=True)
        except Exception as e:  # pragma: no cover
            dt = time.perf_counter() - t0
            rows.append([f.name, DESC.get(f.name, ""), 0, dt, f"失败: {e}"])
            print(f"[FAIL] {f.name}: {e}", flush=True)

    summary = pd.DataFrame(rows, columns=["SQL 文件", "说明", "结果行数", "耗时(秒)", "状态"])
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = f"""# 阶段 8 · SQL 化结果

所有指标均已用 SQL 表达并归档到 `sql/`，可直接用 DuckDB 对 **全量 3.6 GB CSV** 执行：
`events` 视图指向中间层 parquet，`raw` 视图指向原始 CSV，二者由 `src/db.py` 注册。

{utils.df_to_md(summary)}

## 怎么跑

```bash
.venv\\Scripts\\python.exe src\\stage8_sql.py
```

## 为什么用 DuckDB

- **零部署**：不需要起 Hive/Spark 集群，一个进程直接对 CSV 跑 SQL；
- **列式 + 向量化 + 多线程**：1 亿行聚合在秒级到十秒级完成；
- **可迁移**：同样的 SQL 思路可平移到 Hive/Spark，也避免了 pandas 全量读入的内存瓶颈。
"""
    (C.REPORT_DIR / "stage8_sql_summary.md").write_text(md, encoding="utf-8")
    con.close()
    print(utils.df_to_md(summary), flush=True)


if __name__ == "__main__":
    main()
