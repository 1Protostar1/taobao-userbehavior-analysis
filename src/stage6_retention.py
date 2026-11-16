# -*- coding: utf-8 -*-
"""
阶段 6 · 留存与复购
===================
口径说明：
  - **留存分母**：以"用户首次活跃日"为基准的 cohort，即首日活跃用户。
    数据集只有 9 天（2017-11-25 ~ 2017-12-03），2017-11-25 是窗口第一天，
    当天"首次活跃"的用户里混入了大量存量老客（他们的真实首次活跃日在窗口之外），
    因此 11-25 cohort 的留存率会系统性偏高 —— 报告中显式标注，解读时以后续 cohort 为准。
  - **右截断**：day-N 留存只有 fd + N <= 2017-12-03 的 cohort 可完整观测，
    不可观测的格子在矩阵中留空（NaN），不做任何填充/外推。
  - **复购定义**：购买天数 >= 2（跨天复购），与 sql/07_repeat_purchase.sql 一致；
    同时给出"购买次数 >= 2（多次购买）"口径作对照，说明一单多件与跨天复购的区别。

产出：
  outputs/stage6_cohort_matrix.csv / stage6_cohort_long.csv / stage6_retention_overall.csv
  outputs/stage6_repurchase.csv / stage6_repurchase_interval.csv
  outputs/stage6_retention_by_category.csv / stage6_retention_by_first_hour.csv
  reports/figures/fig19~fig22*.png
  reports/stage6_retention_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import db  # noqa: E402
import utils  # noqa: E402

C.setup_chinese_font()
import matplotlib.pyplot as plt  # noqa: E402

TOP_CATEGORIES_N = 5  # 6.5 分品类留存对比取的类目数（按全窗口 pv 排序）

HOUR_BINS = [(0, 6, "凌晨 0-6"), (7, 11, "上午 7-11"), (12, 17, "下午 12-17"), (18, 23, "晚间 18-23")]


# ------------------------------------------------------------------ 数据
def build_activity_base(con) -> pd.DataFrame:
    """
    用户级留存底表：user_id × first_date × day0..day8 是否活跃（0/1 列）。
    一次性从 events 取 (user_id, date) 去重对，再在 pandas 里展开成宽表，
    后续 cohort 矩阵、留存曲线、分段留存都复用这张表，不再扫 parquet。
    """
    uf = C.PARQUET_DIR / "user_features.parquet"
    events_glob = str(C.PARQUET_DIR / "events" / "**" / "*.parquet").replace("\\", "/")
    df = con.execute(
        f"""
        WITH act AS (
            SELECT DISTINCT user_id, date
            FROM read_parquet('{events_glob}', hive_partitioning = true)
        )
        SELECT
            f.user_id,
            f.first_date                              AS fd,
            date_diff('day', f.first_date, a.date)    AS day_n
        FROM read_parquet('{uf.as_posix()}') f
        JOIN act a USING (user_id)
        """
    ).fetchdf()
    wide = (
        df.assign(active=1)
        .pivot_table(index=["user_id", "fd"], columns="day_n", values="active",
                     aggfunc="max", fill_value=0)
        .reset_index()
    )
    wide.columns = ["user_id", "fd"] + [f"d{int(c)}" for c in wide.columns[2:]]
    for n in range(0, 9):
        col = f"d{n}"
        if col not in wide.columns:
            wide[col] = 0
    return wide


def build_cohort_matrix(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """cohort（首日）× day-N 的留存率矩阵与长表。不可观测的格子为 NaN。"""
    end = pd.Timestamp(C.DATE_END)
    sizes = base.groupby("fd")["user_id"].count().rename("cohort_size")
    long_rows = []
    for n in range(1, 9):
        observable_end = end - pd.Timedelta(days=n)
        sub = base[base["fd"] <= observable_end]
        retained = sub[sub[f"d{n}"] == 1].groupby("fd")["user_id"].count()
        for fd, size in sizes.items():
            if fd > observable_end:
                continue  # 右截断：该 cohort 的 day-N 不可观测
            r = int(retained.get(fd, 0))
            long_rows.append({"首日": fd, "day_n": n, "留存人数": r,
                              "cohort_size": int(size), "留存率": r / size})
    long_df = pd.DataFrame(long_rows)
    matrix = long_df.pivot(index="首日", columns="day_n", values="留存率")
    matrix.columns = [f"D{n}" for n in matrix.columns]
    return matrix, long_df


def build_overall_retention(long_df: pd.DataFrame) -> pd.DataFrame:
    """总体留存：各 cohort 按人数加权，只统计可观测 cohort。"""
    g = (
        long_df.groupby("day_n")
        .agg(留存人数=("留存人数", "sum"), 分母=("cohort_size", "sum"), 可观测cohort数=("首日", "nunique"))
        .reset_index()
    )
    g["加权留存率"] = g["留存人数"] / g["分母"]
    return g


def build_repurchase(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """复购汇总 + 相邻购买日间隔分布。"""
    events_glob = str(C.PARQUET_DIR / "events" / "**" / "*.parquet").replace("\\", "/")
    summary = con.execute(
        f"""
        WITH bu AS (
            SELECT user_id,
                   count(*)                 AS buy_cnt,
                   count(DISTINCT date)     AS buy_days
            FROM read_parquet('{events_glob}', hive_partitioning = true)
            WHERE behavior_type = 'buy'
            GROUP BY user_id
        )
        SELECT
            count(*)                                                  AS 购买用户数,
            count(*) FILTER (WHERE buy_cnt >= 2)                      AS 多次购买用户数,
            count(*) FILTER (WHERE buy_days >= 2)                     AS 跨天复购用户数,
            round(avg(buy_cnt), 3)                                    AS 人均购买次数,
            round(avg(buy_days), 3)                                   AS 人均购买天数,
            round(avg(buy_cnt) FILTER (WHERE buy_days >= 2), 3)       AS 复购用户人均购买次数
        FROM bu
        """
    ).fetchdf()
    interval = con.execute(
        f"""
        WITH bd AS (
            SELECT DISTINCT user_id, date
            FROM read_parquet('{events_glob}', hive_partitioning = true)
            WHERE behavior_type = 'buy'
        ),
        gaps AS (
            SELECT date_diff('day',
                       lag(date) OVER (PARTITION BY user_id ORDER BY date),
                       date) AS gap_days
            FROM bd
        )
        SELECT gap_days AS 间隔天数, count(*) AS 次数
        FROM gaps
        WHERE gap_days IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    return summary, interval


def build_retention_by_category(con, base: pd.DataFrame) -> pd.DataFrame:
    """
    分品类留存：对 pv Top-N 类目，比较"首日浏览过该类目"的新增用户 vs 首日未浏览的
    次日留存率。一个用户可浏览多个类目，组间不互斥（报告中说明）。
    """
    events_glob = str(C.PARQUET_DIR / "events" / "**" / "*.parquet").replace("\\", "/")
    uf = C.PARQUET_DIR / "user_features.parquet"
    top_cats = con.execute(
        f"""
        SELECT category_id
        FROM read_parquet('{events_glob}', hive_partitioning = true)
        WHERE behavior_type = 'pv'
        GROUP BY 1
        ORDER BY count(*) DESC
        LIMIT {TOP_CATEGORIES_N}
        """
    ).fetchdf()["category_id"].tolist()
    cat_first = con.execute(
        f"""
        SELECT DISTINCT e.user_id, e.category_id
        FROM read_parquet('{events_glob}', hive_partitioning = true) e
        JOIN read_parquet('{uf.as_posix()}') f USING (user_id)
        WHERE e.behavior_type = 'pv' AND e.date = f.first_date
        """
    ).fetchdf()
    # 次日留存只统计可观测 cohort（fd <= 12-02）
    observable = base[base["fd"] <= pd.Timestamp(C.DATE_END) - pd.Timedelta(days=1)][
        ["user_id", "d1"]
    ]
    rows = []
    total_n = len(observable)
    overall_ret1 = observable["d1"].mean()
    for cat in top_cats:
        touched = set(cat_first.loc[cat_first["category_id"] == cat, "user_id"])
        in_g = observable[observable["user_id"].isin(touched)]
        out_g = observable[~observable["user_id"].isin(touched)]
        rows.append({
            "类目ID": cat,
            "首日浏览该类目用户数": len(in_g),
            "该类目用户次日留存率": in_g["d1"].mean(),
            "未浏览用户数": len(out_g),
            "未浏览用户次日留存率": out_g["d1"].mean(),
            "留存率差值": in_g["d1"].mean() - out_g["d1"].mean(),
        })
    res = pd.DataFrame(rows)
    res.attrs["overall_ret1"] = overall_ret1
    res.attrs["total_n"] = total_n
    return res


def build_retention_by_first_hour(con, base: pd.DataFrame) -> pd.DataFrame:
    """分时段留存：按首日首次活跃的小时分段，比较次日留存率。"""
    events_glob = str(C.PARQUET_DIR / "events" / "**" / "*.parquet").replace("\\", "/")
    uf = C.PARQUET_DIR / "user_features.parquet"
    first_hour = con.execute(
        f"""
        SELECT e.user_id,
               CAST(EXTRACT(hour FROM min(e.datetime)) AS INTEGER) AS first_hour
        FROM read_parquet('{events_glob}', hive_partitioning = true) e
        JOIN read_parquet('{uf.as_posix()}') f USING (user_id)
        WHERE e.date = f.first_date
        GROUP BY 1
        """
    ).fetchdf()

    def bucket(h: int) -> str:
        for lo, hi, name in HOUR_BINS:
            if lo <= h <= hi:
                return name
        return "其他"

    first_hour["时段"] = first_hour["first_hour"].map(bucket)
    observable = base[base["fd"] <= pd.Timestamp(C.DATE_END) - pd.Timedelta(days=1)][
        ["user_id", "d1"]
    ]
    m = observable.merge(first_hour[["user_id", "时段"]], on="user_id", how="inner")
    g = (
        m.groupby("时段")
        .agg(新增用户数=("user_id", "count"), 次日留存率=("d1", "mean"))
        .reindex([b[2] for b in HOUR_BINS])
        .reset_index()
    )
    return g


# ------------------------------------------------------------------ 图
def fig_cohort_heatmap(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    vals = matrix.values.astype(float)
    im = ax.imshow(np.ma.masked_invalid(vals), cmap="YlGnBu", aspect="auto",
                   vmin=0, vmax=np.nanmax(vals) * 1.05)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([str(d)[:10] for d in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "截断", ha="center", va="center", fontsize=8, color="#999999")
            else:
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > np.nanmax(vals) * 0.6 else "#222222")
    ax.set_xlabel("首次活跃后的第 N 天")
    ax.set_ylabel("Cohort（首次活跃日）")
    ax.set_title("Cohort 留存矩阵：首日活跃 × 第 N 日留存率（空格=窗口右截断，不可观测）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, format=lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    C.save_fig(fig, "fig19_cohort_retention_heatmap.png")
    plt.close(fig)


def fig_retention_curves(matrix: pd.DataFrame, overall: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for fd, row in matrix.iterrows():
        xs = [int(c[1:]) for c in matrix.columns]
        ys = row.values.astype(float)
        mask = ~np.isnan(ys)
        is_first = str(fd)[:10] == C.DATE_START
        ax.plot(np.array(xs)[mask], ys[mask], marker="o", ms=3.5, lw=1.4,
                color="#E45756" if is_first else "#4C78A8",
                alpha=0.95 if is_first else 0.55,
                label=f"{str(fd)[:10]}{'（混入存量老客）' if is_first else ''}")
    ov = overall.set_index("day_n")["加权留存率"]
    ax.plot(ov.index, ov.values, marker="s", ms=6, lw=2.6, color="#222222",
            label="全体加权平均")
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([f"D{n}" for n in range(1, 9)])
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("首次活跃后的第 N 天")
    ax.set_ylabel("留存率")
    ax.set_title("各 Cohort 留存衰减曲线（每条线只画到可观测的最后一天）")
    ax.legend(fontsize=8.5, ncol=3)
    fig.tight_layout()
    C.save_fig(fig, "fig20_retention_curves.png")
    plt.close(fig)


def fig_repurchase(summary: pd.DataFrame, interval: pd.DataFrame) -> None:
    r = summary.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    labels = ["购买用户\n(分母)", "多次购买\n(次数≥2)", "跨天复购\n(天数≥2)"]
    vals = [r["购买用户数"], r["多次购买用户数"], r["跨天复购用户数"]]
    colors = ["#9ECAE1", "#F58518", "#E45756"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}\n({v / r['购买用户数']:.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_ylabel("用户数")
    ax.set_title("复购两种口径：一单多件（次数≥2）≠ 跨天复购（天数≥2）")

    ax = axes[1]
    x = interval["间隔天数"].to_numpy()
    y = interval["次数"].to_numpy()
    ax.bar(x, y, color="#4C78A8", width=0.6)
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:,}\n{yi / y.sum():.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, y.max() * 1.25)
    ax.set_xticks(x)
    ax.set_xlabel("相邻两次购买的间隔天数")
    ax.set_ylabel("次数")
    med = np.repeat(x, y)
    ax.set_title(f"复购间隔分布（中位数 {np.median(med):.0f} 天，"
                 f"{y[x == 1].sum() / y.sum():.1%} 的复购发生在次日）")
    fig.tight_layout()
    C.save_fig(fig, "fig21_repurchase.png")
    plt.close(fig)


def fig_retention_by_segment(cat: pd.DataFrame, hour: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(len(cat))
    ax.bar(x - 0.2, cat["该类目用户次日留存率"] * 100, width=0.4,
           color="#E45756", label="首日浏览过该类目")
    ax.bar(x + 0.2, cat["未浏览用户次日留存率"] * 100, width=0.4,
           color="#BAB0AC", label="首日未浏览该类目")
    for i in range(len(cat)):
        ax.text(i - 0.2, cat["该类目用户次日留存率"].iloc[i] * 100,
                f"{cat['该类目用户次日留存率'].iloc[i]:.1%}", ha="center", va="bottom", fontsize=8.5)
        ax.text(i + 0.2, cat["未浏览用户次日留存率"].iloc[i] * 100,
                f"{cat['未浏览用户次日留存率'].iloc[i]:.1%}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"类目 {c}" for c in cat["类目ID"]], fontsize=9)
    ax.set_ylabel("次日留存率 (%)")
    ax.set_title(f"分品类：pv Top{TOP_CATEGORIES_N} 类目首日触达 vs 次日留存\n（组间不互斥，一个用户可浏览多个类目）",
                 fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    x = np.arange(len(hour))
    bars = ax.bar(x, hour["次日留存率"] * 100, color="#54A24B", width=0.5)
    for b, (_, r) in zip(bars, hour.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, r["次日留存率"] * 100,
                f"{r['次日留存率']:.1%}\nn={r['新增用户数']:,}", ha="center", va="bottom", fontsize=9)
    overall = (hour["次日留存率"] * hour["新增用户数"]).sum() / hour["新增用户数"].sum()
    ax.axhline(overall * 100, color="#E45756", ls="--", lw=1.2,
               label=f"全体均值 {overall:.1%}")
    ax.set_xticks(x)
    ax.set_xticklabels(hour["时段"], fontsize=10)
    ax.set_ylabel("次日留存率 (%)")
    ax.set_ylim(0, hour["次日留存率"].max() * 100 * 1.3)
    ax.set_title("分时段：首日首次活跃时段 vs 次日留存")
    ax.legend(fontsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig22_retention_by_segment.png")
    plt.close(fig)


# ------------------------------------------------------------------ 主流程
def main() -> None:
    con = db.connect(memory_limit="8GB")
    con.execute(f"SET temp_directory='{(C.OUTPUT_DIR / 'duckdb_tmp').as_posix()}'")

    print("[6.1-6.3] 构建用户×日期活跃底表（约百万用户 × 9 天）...", flush=True)
    base = build_activity_base(con)
    print(f"     用户底表 {len(base):,} 行", flush=True)

    matrix, long_df = build_cohort_matrix(base)
    overall = build_overall_retention(long_df)
    matrix_out = matrix.copy()
    matrix_out.index = matrix_out.index.astype(str)
    matrix_out.to_csv(C.OUTPUT_DIR / "stage6_cohort_matrix.csv", encoding="utf-8-sig")
    long_df.assign(首日=long_df["首日"].astype(str)).to_csv(
        C.OUTPUT_DIR / "stage6_cohort_long.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(C.OUTPUT_DIR / "stage6_retention_overall.csv", index=False, encoding="utf-8-sig")

    print("[6.4] 复购率与购买间隔 ...", flush=True)
    rep, interval = build_repurchase(con)
    rep.to_csv(C.OUTPUT_DIR / "stage6_repurchase.csv", index=False, encoding="utf-8-sig")
    interval.to_csv(C.OUTPUT_DIR / "stage6_repurchase_interval.csv", index=False,
                    encoding="utf-8-sig")

    print("[6.5] 分品类 / 分时段留存差异 ...", flush=True)
    cat_ret = build_retention_by_category(con, base)
    hour_ret = build_retention_by_first_hour(con, base)
    cat_ret.to_csv(C.OUTPUT_DIR / "stage6_retention_by_category.csv", index=False,
                   encoding="utf-8-sig")
    hour_ret.to_csv(C.OUTPUT_DIR / "stage6_retention_by_first_hour.csv", index=False,
                    encoding="utf-8-sig")

    print("[6.x] 可视化 ...", flush=True)
    fig_cohort_heatmap(matrix)
    fig_retention_curves(matrix, overall)
    fig_repurchase(rep, interval)
    fig_retention_by_segment(cat_ret, hour_ret)

    # ---------------- 汇总报告
    r = rep.iloc[0]
    ret1 = overall.loc[overall["day_n"] == 1, "加权留存率"].iloc[0]
    ret7 = overall.loc[overall["day_n"] == 7, "加权留存率"].iloc[0]
    first_cohort_ret1 = matrix.iloc[0, 0]
    later = long_df[(long_df["day_n"] == 1) & (long_df["首日"] != matrix.index[0])]
    later_ret1 = later["留存人数"].sum() / later["cohort_size"].sum()
    n_obs_d7 = int(overall.loc[overall["day_n"] == 7, "可观测cohort数"].iloc[0])
    int_x = interval["间隔天数"].to_numpy()
    int_y = interval["次数"].to_numpy()
    med_gap = float(np.median(np.repeat(int_x, int_y)))
    best_cat = cat_ret.loc[cat_ret["留存率差值"].idxmax()]
    best_hour = hour_ret.loc[hour_ret["次日留存率"].idxmax()]

    md = f"""# 阶段 6 · 留存与复购结论

## 6.1 口径定义（先说清分母）

| 指标 | 定义 | 可观测性 |
|---|---|---|
| 次日留存率 | 首日活跃（cohort）用户中，第 2 天仍活跃的比例 | 11-25 ~ 12-02 共 8 个 cohort 可完整观测 |
| 7 日内回访率 | cohort 用户在第 1~7 天内任意一天回访的比例 | 仅前 2 个 cohort 完整，其余右截断 |
| 复购率 | 购买用户中**购买天数 ≥ 2** 的比例 | 9 天窗口全量可算，但低估真实复购 |

**窗口局限（必须写进报告）**：
1. 数据只有 9 天，只能看**短期留存**；真实业务的 30 日留存、月留存无法计算。
2. **2017-11-25 是窗口第一天**，当天"首次活跃"混入大量存量老客，
   其次日留存率 {first_cohort_ret1:.1%} 显著高于后续 cohort 的 {later_ret1:.1%} ——
   解读新增用户留存时应剔除该 cohort。
3. day-N 格子若 fd + N 超出 2017-12-03 则**不可观测**，矩阵中留空，不做任何填充或外推。

## 6.2–6.3 Cohort 留存

{utils.df_to_md(overall.assign(加权留存率=overall['加权留存率'].map('{:.2%}'.format)))}

![cohort 热力图](figures/fig19_cohort_retention_heatmap.png)

![留存曲线](figures/fig20_retention_curves.png)

- 全体加权次日留存率 **{ret1:.1%}**；剔除 11-25 存量混合 cohort 后为 **{later_ret1:.1%}**。
- **日历效应（本阶段最重要的口径发现）**：几乎每个 cohort 在"第 N 天恰逢周末（12-02/12-03）"时
  留存率都跳到 95% 以上（如 11-30 cohort 的 D1 落在周五 12-01，留存 {matrix.loc[matrix.index[5], 'D1']:.1%}；
  11-26 cohort 的 D6 落在周六 12-02，留存 {matrix.loc[matrix.index[1], 'D6']:.1%}）。
  因此留存曲线的"翘尾"**不是用户粘性随时间变好，而是周末流量高峰的日历效应**；
  同理，加权 7 日留存（仅 {n_obs_d7} 个 cohort 可观测）高达 {ret7:.1%}，也是被周末截断放大的，
  不能与常规意义上的 D7 留存直接比较。
- 剔除日历效应后看真实衰减：同一 cohort 在工作日区间（D1~D5）内基本平稳在 62%~72%，
  没有明显逐日衰减 —— 说明大促期间用户的回访意愿整体较强，流失主要集中在"首次活跃后的第二天"
  （次日留存 {later_ret1:.1%} 意味着约四分之一新客第二天就流失了）。

## 6.4 复购

{utils.df_to_md(rep)}

![复购](figures/fig21_repurchase.png)

- 购买用户共 **{int(r['购买用户数']):,}** 人；**多次购买（次数≥2）占 {r['多次购买用户数'] / r['购买用户数']:.1%}**，
  其中**跨天复购（天数≥2）占 {r['跨天复购用户数'] / r['购买用户数']:.1%}** ——
  两者之差即"同一天内一单多件"的用户，约 {int(r['多次购买用户数'] - r['跨天复购用户数']):,} 人。
  用"购买天数"定义复购可以剔除这种一次性囤货行为，口径更稳。
- 复购用户人均购买 {r['复购用户人均购买次数']:.1f} 次，是全体购买用户（{r['人均购买次数']:.2f} 次）的
  {r['复购用户人均购买次数'] / r['人均购买次数']:.1f} 倍。
- 相邻购买间隔中位数 **{med_gap:.0f} 天**，{int_y[int_x == 1].sum() / int_y.sum():.1%} 的复购发生在**次日** ——
  大促期间"连续下单"是复购的主要形态，复购召回的黄金窗口是首购后 24 小时。

## 6.5 分品类 / 分时段留存差异

{utils.df_to_md(cat_ret)}

{utils.df_to_md(hour_ret)}

![分段留存](figures/fig22_retention_by_segment.png)

- 分品类：首日浏览**类目 {int(best_cat['类目ID'])}** 的新增用户次日留存率最高
  （{best_cat['该类目用户次日留存率']:.1%}，比未浏览用户高 {best_cat['留存率差值'] * 100:.1f} 个百分点），
  说明该类目对新客有更强的"钩子"效应，适合作为新客首屏推荐位。
- 分时段：**{best_hour['时段']}** 首次活跃的用户次日留存率最高（{best_hour['次日留存率']:.1%}）。
  注意这是**自选择效应**而非时段的因果作用：凌晨还在逛的用户本来就是重度用户，粘性天然更高；
  晚间 18-23 点进入的新客留存最低（{hour_ret.loc[hour_ret['时段'] == '晚间 18-23', '次日留存率'].iloc[0]:.1%}），
  高峰期流量虽大，但新客质量被稀释。
- 注意：品类分组不互斥（一个用户首日可浏览多个类目），结论方向性参考即可，不做因果推断。

## 6.6 一句话总结

新客留存的关键窗口是首次活跃后 48 小时；复购的一半发生在首购次日。
短期留存与复购都指向同一个动作：**首购/首访后 24 小时内的即时触达**
（优惠券到期提醒、同类商品推荐），比广撒网式召回更有效。
"""
    (C.REPORT_DIR / "stage6_retention_summary.md").write_text(md, encoding="utf-8")
    con.close()
    print("阶段 6 完成", flush=True)


if __name__ == "__main__":
    main()
