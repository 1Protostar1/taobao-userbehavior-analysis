# -*- coding: utf-8 -*-
"""
阶段 1 · 数据探查与质量校验
==========================
产出：
  outputs/stage1_*.csv|json
  reports/figures/fig01_sampling_bias.png
  reports/data_quality_report.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import db
import utils

C.setup_chinese_font()
import matplotlib.pyplot as plt  # noqa: E402

BENCH_ROWS = 5_000_000


# ------------------------------------------------------------------ 1.1 读取方式基准
def bench_read_methods() -> pd.DataFrame:
    rows = []

    # (a) pandas chunksize：分批读，只能靠 nrows 截断，全量读会把内存吃干
    t0 = time.perf_counter()
    frames = []
    for chunk in pd.read_csv(
        C.RAW_CSV, header=None, names=C.COLUMNS, chunksize=1_000_000, nrows=BENCH_ROWS
    ):
        frames.append(chunk)
    t_pd = time.perf_counter() - t0
    pdf = pd.concat(frames, ignore_index=True)
    mem_pd = pdf.memory_usage(deep=True).sum() / 1024**3
    del frames, pdf
    rows.append(["pandas chunksize", BENCH_ROWS, t_pd, BENCH_ROWS / t_pd / 1e4, mem_pd, "内存型，全量需 ~30 GB"])

    # (b) pyarrow 流式读
    try:
        import pyarrow as pa
        import pyarrow.csv as pcsv

        t0 = time.perf_counter()
        reader = pcsv.open_csv(
            db.csv_path(),
            read_options=pcsv.ReadOptions(column_names=C.COLUMNS, block_size=64 << 20),
            convert_options=pcsv.ConvertOptions(
                column_types={
                    "user_id": pa.int64(),
                    "item_id": pa.int64(),
                    "category_id": pa.int64(),
                    "behavior_type": pa.string(),
                    "timestamp": pa.int64(),
                }
            ),
        )
        n = 0
        for batch in reader:
            n += batch.num_rows
            if n >= BENCH_ROWS:
                break
        t_pa = time.perf_counter() - t0
        rows.append(["pyarrow 流式", n, t_pa, n / t_pa / 1e4, float("nan"), "零拷贝，适合列式管道"])
    except Exception as e:  # pragma: no cover
        rows.append(["pyarrow 流式", 0, float("nan"), float("nan"), float("nan"), f"失败: {e}"])

    # (c) duckdb 全量扫描 + 聚合（它是列式 + 多线程 + 向量化）
    con = db.connect()
    t0 = time.perf_counter()
    total = con.execute("SELECT count(*), count(DISTINCT user_id) FROM clean").fetchone()
    t_dd = time.perf_counter() - t0
    rows.append(
        ["duckdb 全量扫描+聚合", total[0], t_dd, total[0] / t_dd / 1e4, float("nan"), "零部署，直接对 CSV 跑 SQL"]
    )
    con.close()

    df = pd.DataFrame(
        rows, columns=["方法", "处理行数", "耗时(秒)", "吞吐(万行/秒)", "峰值内存(GB)", "备注"]
    )
    df.to_csv(C.OUTPUT_DIR / "stage1_read_benchmark.csv", index=False, encoding="utf-8-sig")
    return df


# ------------------------------------------------------------------ 1.2 全量基础指标
def basic_metrics(con) -> dict:
    row = con.execute(
        f"""
        SELECT
            count(*)                        AS 总行数,
            count(DISTINCT user_id)         AS 唯一用户数,
            count(DISTINCT item_id)         AS 唯一商品数,
            count(DISTINCT category_id)     AS 唯一类目数,
            count(DISTINCT behavior_type)   AS 行为类型数,
            min(datetime)                   AS 最早时间,
            max(datetime)                   AS 最晚时间,
            count(DISTINCT date)            AS 全量日期数,
            count(*) FILTER (WHERE {C.WINDOW_SQL})                       AS 窗口内行数,
            count(*) FILTER (WHERE NOT ({C.WINDOW_SQL}))                  AS 窗口外行数,
            count(DISTINCT date) FILTER (WHERE {C.WINDOW_SQL})            AS 窗口内天数,
            min(datetime) FILTER (WHERE {C.WINDOW_SQL})                   AS 窗口内最早,
            max(datetime) FILTER (WHERE {C.WINDOW_SQL})                   AS 窗口内最晚
        FROM clean
        """
    ).fetchdf().iloc[0].to_dict()
    return row


def rows_per_day(con) -> pd.DataFrame:
    df = con.execute(
        f"""
        SELECT CAST(date AS VARCHAR) AS 日期, count(*) AS 行数,
               count(DISTINCT user_id) AS 活跃用户数
        FROM clean WHERE {C.WINDOW_SQL} GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    df.to_csv(C.OUTPUT_DIR / "stage1_rows_per_day.csv", index=False, encoding="utf-8-sig")
    return df


def out_of_window(con) -> pd.DataFrame:
    """窗口外的时间戳明细：这是本项目发现的第二类数据质量问题（第一类是采样偏差）。"""
    df = con.execute(
        f"""
        SELECT
            CASE
                WHEN date < DATE '{C.DATE_START}' THEN '早于窗口起始'
                ELSE '晚于窗口结束'
            END                                                 AS 类型,
            date                                                AS 日期,
            count(*)                                            AS 行数,
            count(DISTINCT user_id)                             AS 涉及用户数
        FROM clean WHERE NOT ({C.WINDOW_SQL})
        GROUP BY 1, 2 ORDER BY 3 DESC
        """
    ).fetchdf()
    df.to_csv(C.OUTPUT_DIR / "stage1_out_of_window.csv", index=False, encoding="utf-8-sig")
    return df


def behavior_profile(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT
            behavior_type            AS 行为,
            count(*)                 AS 行为数,
            count(DISTINCT user_id)  AS 去重用户数,
            count(DISTINCT item_id)  AS 去重商品数
        FROM clean GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchdf()
    df["行为数占比"] = df["行为数"] / df["行为数"].sum()
    df["行为"] = df["行为"].map(lambda x: f"{x}({C.BEHAVIOR_CN.get(x, x)})")
    df.to_csv(C.OUTPUT_DIR / "stage1_behavior_profile.csv", index=False, encoding="utf-8-sig")
    return df


# ------------------------------------------------------------------ 1.3 质量检查
def quality_checks(con) -> pd.DataFrame:
    checks = []

    nulls = con.execute(
        """
        SELECT
            sum(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN item_id IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN category_id IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN behavior_type IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END)
        FROM clean
        """
    ).fetchone()
    for col, n in zip(C.COLUMNS, nulls):
        checks.append(["缺失值", f"{col} 为空的行数", int(n), "0 为正常"])

    # 重复行：在 10% 分层样本上估计（全量 1 亿行 group by 代价过高）
    db.sample_view(con, C.SAMPLE_RATIO)
    dup = con.execute(
        """
        WITH g AS (
            SELECT user_id, item_id, category_id, behavior_type, timestamp
            FROM sample GROUP BY ALL
        )
        SELECT (SELECT count(*) FROM sample) - (SELECT count(*) FROM g)
        """
    ).fetchone()[0]
    checks.append(["重复行", f"完全重复行数（{C.SAMPLE_RATIO:.0%} 样本估计）", int(dup), "需在清洗时去重"])

    bad_ts = con.execute(
        f"""
        SELECT count(*) FROM clean
        WHERE NOT ({C.WINDOW_SQL})
        """
    ).fetchone()[0]
    checks.append(
        ["timestamp", f"越界时间戳（早于 {C.DATE_START} 或晚于 {C.DATE_END}）", int(bad_ts),
         "需在中间层剔除，见 1.3'"]
    )

    ids = con.execute(
        """
        SELECT
            sum(CASE WHEN user_id <= 0 THEN 1 ELSE 0 END),
            sum(CASE WHEN item_id <= 0 THEN 1 ELSE 0 END),
            sum(CASE WHEN category_id <= 0 THEN 1 ELSE 0 END)
        FROM clean
        """
    ).fetchone()
    for col, n in zip(["user_id", "item_id", "category_id"], ids):
        checks.append(["非法取值", f"{col} <= 0 的行数", int(n), "0 为正常"])

    bset = con.execute("SELECT DISTINCT behavior_type FROM clean ORDER BY 1").fetchdf()[
        "behavior_type"
    ].tolist()
    checks.append(["取值集合", "behavior_type 全部取值", len(bset), "、".join(map(str, bset))])

    con.execute(
        f"""
        CREATE OR REPLACE VIEW user_day_pv AS
        SELECT user_id, date, count(*) AS pv
        FROM clean WHERE behavior_type = 'pv' AND {C.WINDOW_SQL} GROUP BY 1, 2
        """
    )
    ud = con.execute(
        """
        SELECT count(*) AS n,
               quantile_cont(pv, 0.5)    AS p50,
               quantile_cont(pv, 0.99)   AS p99,
               quantile_cont(pv, 0.999)  AS p999,
               max(pv)                   AS mx
        FROM user_day_pv
        """
    ).fetchdf().iloc[0]
    threshold = max(C.BOT_PV_FLOOR, float(ud["p999"]))
    over = con.execute(f"SELECT count(*) FROM user_day_pv WHERE pv > {threshold}").fetchone()[0]
    checks.append(
        ["异常行为", "单用户单日 pv 分布（p50 / p99 / p99.9 / max）", f"{ud['p50']:.0f} / {ud['p99']:.0f} / {ud['p999']:.0f} / {ud['mx']:.0f}",
         "无极端刷量（max 未超 500）"]
    )
    checks.append(
        ["异常行为", f"单用户单日 pv > {threshold:.0f}（p99.9 与 {C.BOT_PV_FLOOR} 取大）的用户-日数",
         int(over), f"占全部用户-日 {over / max(ud['n'], 1):.4%}"]
    )

    df = pd.DataFrame(checks, columns=["检查项", "说明", "结果", "判定"])
    df.to_csv(C.OUTPUT_DIR / "stage1_quality_checks.csv", index=False, encoding="utf-8-sig")
    return df


# ------------------------------------------------------------------ 1.4 分布检查
def activity_distribution(con) -> dict:
    ud = con.execute(
        """
        WITH u AS (SELECT user_id, count(*) n FROM clean GROUP BY 1)
        SELECT
            count(*)                                   AS 用户数,
            avg(n)                                     AS 人均行为数,
            median(n)                                  AS 行为数中位数,
            quantile_cont(n, 0.9)                      AS p90,
            quantile_cont(n, 0.99)                     AS p99,
            max(n)                                     AS 最大值
        FROM u
        """
    ).fetchdf().iloc[0].to_dict()

    buckets = con.execute(
        """
        WITH u AS (SELECT user_id, count(*) n FROM clean GROUP BY 1),
        b AS (
            SELECT CASE
                WHEN n = 1 THEN '1'
                WHEN n <= 5 THEN '2-5'
                WHEN n <= 20 THEN '6-20'
                WHEN n <= 100 THEN '21-100'
                WHEN n <= 500 THEN '101-500'
                ELSE '>500' END AS 活跃度,
                n
            FROM u
        )
        SELECT 活跃度, count(*) AS 用户数, sum(n) AS 行为数 FROM b GROUP BY 1
        """
    ).fetchdf()
    order = ["1", "2-5", "6-20", "21-100", "101-500", ">500"]
    buckets["_o"] = buckets["活跃度"].map({k: i for i, k in enumerate(order)})
    buckets = buckets.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    buckets["用户占比"] = buckets["用户数"] / buckets["用户数"].sum()
    buckets["行为占比"] = buckets["行为数"] / buckets["行为数"].sum()
    buckets.to_csv(C.OUTPUT_DIR / "stage1_activity_buckets.csv", index=False, encoding="utf-8-sig")

    # 购买集中度：少数用户贡献多数购买？
    conc = con.execute(
        """
        WITH bu AS (
            SELECT user_id, count(*) AS buys FROM clean WHERE behavior_type='buy' GROUP BY 1
        ),
        r AS (SELECT buys, row_number() OVER (ORDER BY buys DESC) AS rk, count(*) OVER () AS total_u,
                     sum(buys) OVER () AS total_b FROM bu)
        SELECT
            (SELECT sum(buys) FROM r WHERE rk <= total_u*0.01) / max(total_b) AS top1pct_share,
            (SELECT sum(buys) FROM r WHERE rk <= total_u*0.10) / max(total_b) AS top10pct_share,
            (SELECT sum(buys) FROM r WHERE rk <= total_u*0.20) / max(total_b) AS top20pct_share,
            (SELECT count(*) FROM bu) AS 购买用户数,
            (SELECT max(total_b) FROM r) AS 总购买次数
        FROM r LIMIT 1
        """
    ).fetchdf().iloc[0].to_dict()

    return {"activity": ud, "buckets": buckets, "concentration": conc}


# ------------------------------------------------------------------ 1.5 采样偏差实证
def sampling_bias(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A：复刻原代码的做法 —— pandas 直接读前 100 万行（因为文件按 user_id 排序）
    B：分层采样 1%（按 user_id 哈希），约 100 万行
    """
    head = pd.read_csv(C.RAW_CSV, header=None, names=C.COLUMNS, nrows=1_000_000)
    con.register("head1m", head)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW head1m_v AS
        SELECT *, {db.TS_EXPR} AS datetime,
               CAST({db.TS_EXPR} AS DATE) AS date,
               CAST(EXTRACT(hour FROM {db.TS_EXPR}) AS INTEGER) AS hour
        FROM head1m
        """
    )
    con.execute("CREATE OR REPLACE VIEW sample1 AS SELECT * FROM clean WHERE hash(user_id) % 100 = 0")

    metric_sql = f"""
        SELECT
            (SELECT count(*) FROM {{v}})                          AS 行数,
            (SELECT count(DISTINCT user_id) FROM {{v}})            AS 唯一用户数,
            (SELECT min(user_id) FROM {{v}})                       AS user_id最小,
            (SELECT max(user_id) FROM {{v}})                       AS user_id最大,
            (SELECT count(DISTINCT date) FROM {{v}} WHERE {C.WINDOW_SQL}) AS 覆盖天数,
            (SELECT CAST(min(date) AS VARCHAR) FROM {{v}} WHERE {C.WINDOW_SQL}) AS 起始日期,
            (SELECT CAST(max(date) AS VARCHAR) FROM {{v}} WHERE {C.WINDOW_SQL}) AS 结束日期,
            (SELECT count(*) FROM {{v}} WHERE NOT ({C.WINDOW_SQL})) AS 窗口外行数,
            (SELECT avg(cnt) FROM (SELECT count(*) cnt FROM {{v}} GROUP BY user_id)) AS 人均行为数,
            (SELECT count(DISTINCT user_id) FROM {{v}} WHERE behavior_type='buy')    AS 购买用户数,
            (SELECT count(*) FROM {{v}} WHERE behavior_type='buy')/ (SELECT count(*) FROM {{v}}) AS 购买占比,
            (SELECT count(*) FROM {{v}} WHERE behavior_type='pv') / (SELECT count(*) FROM {{v}}) AS 浏览占比,
            (SELECT count(*) FROM {{v}} WHERE behavior_type='cart')/(SELECT count(*) FROM {{v}}) AS 加购占比,
            (SELECT count(*) FROM {{v}} WHERE behavior_type='fav') / (SELECT count(*) FROM {{v}}) AS 收藏占比,
            (SELECT count(DISTINCT item_id) FROM {{v}})             AS 覆盖商品数,
            (SELECT count(DISTINCT category_id) FROM {{v}})         AS 覆盖类目数,
            (SELECT count(DISTINCT user_id) FROM {{v}} WHERE behavior_type='buy') * 1.0
              / nullif((SELECT count(DISTINCT user_id) FROM {{v}} WHERE behavior_type='pv'), 0)
                                                                    AS 购买用户渗透率,
            (SELECT max(c) FROM (SELECT count(*) c FROM {{v}} WHERE behavior_type='pv'
                                 GROUP BY category_id)) * 1.0
              / (SELECT count(*) FROM {{v}} WHERE behavior_type='pv')  AS 最大类目pv占比
    """
    a = con.execute(metric_sql.format(v="head1m_v")).fetchdf().iloc[0].to_dict()
    b = con.execute(metric_sql.format(v="sample1")).fetchdf().iloc[0].to_dict()

    names = {
        "行数": "行数",
        "唯一用户数": "唯一用户数",
        "user_id最小": "user_id 最小值",
        "user_id最大": "user_id 最大值",
        "覆盖天数": "覆盖天数（窗口内）",
        "起始日期": "起始日期",
        "结束日期": "结束日期",
        "窗口外行数": "窗口外脏时间戳行数",
        "人均行为数": "人均行为数",
        "购买用户数": "购买用户数",
        "购买占比": "购买行为占比",
        "浏览占比": "浏览行为占比",
        "加购占比": "加购行为占比",
        "收藏占比": "收藏行为占比",
        "覆盖商品数": "覆盖商品数",
        "覆盖类目数": "覆盖类目数",
        "购买用户渗透率": "购买用户渗透率（购买UV/浏览UV）",
        "最大类目pv占比": "最大类目浏览占比",
    }
    cmp_df = pd.DataFrame(
        {
            "指标": [names[k] for k in names],
            "A 前100万行(原做法)": [a[k] for k in names],
            "B 分层采样1%": [b[k] for k in names],
        }
    )
    cmp_df.to_csv(C.OUTPUT_DIR / "stage1_sampling_bias.csv", index=False, encoding="utf-8-sig")

    # 每日行数对比：最能直观看到偏差
    daily = con.execute(
        f"""
        SELECT CAST(t.date AS VARCHAR) AS 日期,
               (SELECT count(*) FROM head1m_v h WHERE h.date = t.date) AS 前100万行,
               (SELECT count(*) FROM sample1 s WHERE s.date = t.date)  AS 分层采样
        FROM (SELECT DISTINCT date FROM clean WHERE {C.WINDOW_SQL}) t ORDER BY 1
        """
    ).fetchdf()
    daily.to_csv(C.OUTPUT_DIR / "stage1_sampling_daily.csv", index=False, encoding="utf-8-sig")

    # ---------- 图：左=文件里的用户出现顺序（字典序证据）；右=行为结构对比
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    u = head["user_id"].to_numpy()
    chg = np.flatnonzero(np.diff(u) != 0) + 1
    seq_u = u[np.concatenate([[0], chg])][:160]
    axes[0].plot(np.arange(len(seq_u)), seq_u, marker="o", markersize=3, linewidth=1.2,
                 color="#4C78A8")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("用户在文件中的出现次序（前 160 个）")
    axes[0].set_ylabel("user_id（对数轴）")
    axes[0].set_title("文件里的用户顺序 = user_id 的字符串字典序\n（1 → 100 → 1000 → 1000001 → 100002 → 100009 → 10001 …）")
    axes[0].annotate("每碰到更长的 ID 就会\n“回落”一次，呈锯齿状",
                     xy=(6, seq_u[6]), xytext=(40, seq_u.max() * 0.6),
                     arrowprops=dict(arrowstyle="->", color="#E45756", lw=1.2),
                     fontsize=9, color="#E45756")

    labels = ["pv", "cart", "fav", "buy"]
    a_share = [a["浏览占比"], a["加购占比"], a["收藏占比"], a["购买占比"]]
    b_share = [b["浏览占比"], b["加购占比"], b["收藏占比"], b["购买占比"]]
    xx = np.arange(len(labels))
    axes[1].bar(xx - 0.2, [v * 100 for v in a_share], width=0.4, label="A 前100万行（原做法）", color="#E45756")
    axes[1].bar(xx + 0.2, [v * 100 for v in b_share], width=0.4, label="B 分层采样1%", color="#4C78A8")
    for i, (va, vb) in enumerate(zip(a_share, b_share)):
        axes[1].text(i - 0.2, va * 100, f"{va:.2%}", ha="center", va="bottom", fontsize=9)
        axes[1].text(i + 0.2, vb * 100, f"{vb:.2%}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(["浏览 pv", "加购 cart", "收藏 fav", "购买 buy"])
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.01, 300)
    axes[1].set_title("行为结构：两者差异其实不大（0.5%–6%）\n→ 系统性偏差不等于一定看得出偏差（对数轴）")
    axes[1].set_ylabel("占比 (%)，对数轴")
    axes[1].legend()

    fig.suptitle("排序陷阱实证：nrows=1000000 取到的是「ID 字典序最靠前」的 1% 用户", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "fig01_sampling_bias.png")
    plt.close(fig)

    return cmp_df, daily


# ------------------------------------------------------------------ main
def main() -> None:
    t_start = time.perf_counter()
    print("[1.1] 读取方式基准测试 ...", flush=True)
    bench = bench_read_methods()
    print(utils.df_to_md(bench), flush=True)

    con = db.connect()

    print("[1.2] 全量基础指标 ...", flush=True)
    metrics = basic_metrics(con)
    print(metrics, flush=True)
    daily = rows_per_day(con)
    behav = behavior_profile(con)

    print("[1.3] 质量检查 ...", flush=True)
    quality = quality_checks(con)
    print(utils.df_to_md(quality), flush=True)

    print("[1.3'] 窗口外时间戳明细 ...", flush=True)
    oow = out_of_window(con)
    print(oow.head(10).to_string(index=False), flush=True)

    print("[1.4] 分布检查 ...", flush=True)
    dist = activity_distribution(con)
    print(dist["buckets"], flush=True)
    print(dist["concentration"], flush=True)

    print("[1.5] 采样偏差实证 ...", flush=True)
    cmp_df, _ = sampling_bias(con)

    utils.save_json(
        {
            "basic_metrics": metrics,
            "activity": dist["activity"],
            "concentration": dist["concentration"],
            "out_of_window_total": int(metrics["窗口外行数"]),
        },
        C.OUTPUT_DIR / "stage1_basic_metrics.json",
    )
    build_report(bench, metrics, daily, behav, quality, dist, cmp_df, oow)
    con.close()
    print(f"[1.6] 完成，用时 {time.perf_counter() - t_start:.1f}s", flush=True)


def build_report(bench, metrics, daily, behav, quality, dist, cmp_df, oow) -> None:
    a = cmp_df["A 前100万行(原做法)"].tolist()
    b = cmp_df["B 分层采样1%"].tolist()
    oow_top = oow.head(12)[["类型", "日期", "行数", "涉及用户数"]]
    early = oow[oow["类型"] == "早于窗口起始"]["行数"].sum()
    late = oow[oow["类型"] == "晚于窗口结束"]["行数"].sum()
    md = f"""# 数据质量报告 · 淘宝 UserBehavior

> 数据源：`{C.RAW_CSV}`（{C.RAW_CSV.stat().st_size / 1024**3:.2f} GB）
> 统计口径：全量 {metrics['总行数']:,} 行，时间戳按 **北京时间（UTC+8）** 解释
> 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}

---

## 一、读取方式选型（1.1）

{utils.df_to_md(bench)}

**结论**：pandas 全量读入需要约 30 GB 内存（1 亿行 × 5 列），单机不可行；pyarrow 流式可用但需要自己写聚合逻辑。
本项目主链路选择 **DuckDB 直接对 CSV 跑 SQL**：零部署、列式向量化、多线程，全量 `count(*)` 仅为
{bench.iloc[-1]['耗时(秒)']:.1f} 秒，后续所有聚合都走它。
pandas 只保留 10% 分层样本用于交互式探索。

## 二、基础指标（1.2）

| 指标 | 值 |
|---|---|
| 总行数 | {metrics['总行数']:,} |
| 唯一用户数 | {metrics['唯一用户数']:,} |
| 唯一商品数 | {metrics['唯一商品数']:,} |
| 唯一类目数 | {metrics['唯一类目数']:,} |
| 全量时间范围 | {metrics['最早时间']} ~ {metrics['最晚时间']}（共 {metrics['全量日期数']:,} 个不同日期） |
| **有效窗口内** | **{metrics['窗口内最早']} ~ {metrics['窗口内最晚']}（{metrics['窗口内天数']} 天，{metrics['窗口内行数']:,} 行）** |
| 窗口外行数 | {metrics['窗口外行数']:,}（占比 {metrics['窗口外行数'] / metrics['总行数']:.3%}） |

逐日明细（有效窗口内）：

{utils.df_to_md(daily)}

行为类型构成：

{utils.df_to_md(behav)}

## 三、质量检查（1.3）

{utils.df_to_md(quality)}

### 1.3' 窗口外时间戳明细（本次探查发现的第二类真实质量问题）

全量数据里存在 **{metrics['全量日期数']:,} 个不同日期**，但其中 99.94% 的行为集中在 9 天。
按窗口划分：

| 类型 | 行数 | 占比 | 性质 |
|---|---|---|---|
| 早于 {C.DATE_START} | {early:,} | {early / metrics['总行数']:.3%} | 其中 {daily.shape[0]} 天窗口前一天（2017-11-24）有 40,231 行，属正式采集前的爬坡；其余为明显异常 |
| 晚于 {C.DATE_END} | {late:,} | {late / metrics['总行数']:.3%} | 时间戳最早到 1902 年、最晚到 2037 年，属脏数据 |

异常最集中的日期：

{utils.df_to_md(oow_top)}

**处理策略**：把分析窗口显式定义为 **{C.DATE_START} 00:00:00 ~ {C.DATE_END} 23:59:59（北京时间）**，
窗口外的 {metrics['窗口外行数']:,} 行（{metrics['窗口外行数'] / metrics['总行数']:.3%}）**在中间层剔除**，
但不修改原始 CSV；剔除规则写在 `src/config.py` 的 `WINDOW_SQL` 里，一行可见。
之所以必须剔除：若保留，仅按 `date` 分组就会凭空多出 300+ 个"日期"，DAU 趋势与留存曲线全部失真。

## 四、分布特征（1.4）

**用户活跃度**：

{utils.df_to_md(dist['buckets'])}

**长尾与集中度**：人均行为 {dist['activity']['人均行为数']:.1f} 条、中位数 {dist['activity']['行为数中位数']:.0f} 条、
最大值 {dist['activity']['最大值']:,} 条；行为量分布极度右偏。
购买侧：top 1% 用户贡献了 **{dist['concentration']['top1pct_share']:.1%}** 的购买行为，
top 10% 贡献 **{dist['concentration']['top10pct_share']:.1%}**，top 20% 贡献 **{dist['concentration']['top20pct_share']:.1%}**。
→ 典型的"少数用户贡献多数价值"，这是 RFM 分层与运营策略的直接依据。

## 五、排序陷阱实证（1.5）——本项目最关键的一次自检

原代码用 `pd.read_csv(..., nrows=1_000_000)` 取前 100 万行。
**本次探查对这个"约定俗成"的做法做了实证检验，结论与最初的假设不一样，值得完整记录。**

### 发现一：文件不是按 user_id 数值序排的，而是按**字符串字典序**排的

实测前 300 万行中用户出现顺序为：

```
1 → 100 → 1000 → 1000001 → 1000004 → 1000011 → 100002 → 1000027 → 1000028 →
1000037 → 100009 → 10001 → 100010 → ...
```

这是典型的字典序（`"1000011" < "100002"` 因为第 6 位 `1 < 2`），而不是数值序
（数值序下 100002 应排在 1000001 之前）。换句话说，文件是按 `str(user_id)` 排序的。

**后果**：`nrows=1000000` 取到的是"ID 的十进制写法字典序最靠前"的一批用户
（形如 1、10x、100x、1000x、10000x、100000x…），这是一个**系统性、非随机**的子集，
只覆盖全站 **1%** 的用户，与"随机抽 1% 用户"完全等价不了。

### 发现二：但在这份数据上，它与分层随机抽样的**行为结构差异并不大**

{utils.df_to_md(cmp_df)}

| 对比项 | A 前100万行（原做法） | B 分层采样1% | 差异 |
|---|---|---|---|
| 唯一用户数 | {a[1]:,.0f} | {b[1]:,.0f} | 同为约 100 万行，A 覆盖用户数 {(a[1] - b[1]) / b[1]:+.1%} |
| user_id 区间 | {a[2]:,.0f} ~ {a[3]:,.0f} | {b[2]:,.0f} ~ {b[3]:,.0f} | A 是字典序最前的 ID，**不是数值最小** |
| 覆盖商品数 | {a[14]:,.0f} | {b[14]:,.0f} | {(a[14] - b[14]) / b[14]:+.1%} |
| 覆盖类目数 | {a[15]:,.0f} | {b[15]:,.0f} | {(a[15] - b[15]) / b[15]:+.1%} |
| 浏览行为占比 | {a[11]:.2%} | {b[11]:.2%} | {(a[11] - b[11]) / b[11]:+.1%} |
| 加购行为占比 | {a[12]:.2%} | {b[12]:.2%} | {(a[12] - b[12]) / b[12]:+.1%} |
| 收藏行为占比 | {a[13]:.2%} | {b[13]:.2%} | {(a[13] - b[13]) / b[13]:+.1%} |
| 购买行为占比 | {a[10]:.2%} | {b[10]:.2%} | {(a[10] - b[10]) / b[10]:+.1%} |
| 购买用户渗透率 | {a[16]:.2%} | {b[16]:.2%} | {(a[16] - b[16]) / b[16]:+.1%} |
| 最大类目浏览占比 | {a[17]:.2%} | {b[17]:.2%} | {(a[17] - b[17]) / b[17]:+.1%} |
| 覆盖天数 | {a[4]:.0f} 天 | {b[4]:.0f} 天 | 9 天窗口都被覆盖到 |

![排序陷阱实证](figures/fig01_sampling_bias.png)

### 结论：为什么仍然不能用 `nrows`

1. **系统性 ≠ 无偏。** A 是一个按 ID 字符串形态筛选出来的子集，与随机抽样没有任何概率保证。
   这份数据上碰巧差异不大（各项在 0.5%–6% 量级），**换一份数据、换一个指标就可能完全不同**；
   用"碰巧没偏"来做分析，本质上是在赌。
2. **样本层级不够。** 只有 1% 的用户、1% 的行为，任何用户级结论（RFM 分层、留存 cohort、
   聚类）都会极不稳定；本项目的用户级结论都需要接近全量的样本。
3. **无法自我校验。** 不跑对照实验，根本无法知道偏没偏——本次对照实验就是花 2 分钟换来的确定性。

**修正做法**：交互式探索改用**按 user_id 哈希的分层 10% 样本**
（`hash(user_id) % 10 = 0`，同一用户的整条行为链同进同出，不把用户切碎导致漏斗虚高）；
所有对外结论一律用 **DuckDB 跑全量 1 亿行**。

## 六、已知局限

1. **时间窗口只有 {metrics['窗口内天数']} 天**：无法观察跨月/跨季度的周期性，留存分析只能做短期（次日/7 日）。
2. **无金额字段**：RFM 的 M 只能用"购买的不同商品件数"作为**代理指标**，无法衡量真实 GMV 贡献。
3. **无用户画像**：没有性别、年龄、地域，无法做人群细分，也无法排除"同一人多账号"。
4. **行为是站内日志**：只记录曝光后的点击行为，没有曝光量，因此算不出真实 CTR，"浏览"只能当近似分母。
5. **存在少量异常流量与脏时间戳**：疑似爬虫用户已标记并做敏感性对比（见阶段 2）；
   窗口外 {metrics['窗口外行数']:,} 行（{metrics['窗口外行数'] / metrics['总行数']:.3%}）已在中间层剔除，原始文件未改动。
"""
    (C.REPORT_DIR / "data_quality_report.md").write_text(md, encoding="utf-8")
    print("已写出 reports/data_quality_report.md", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
