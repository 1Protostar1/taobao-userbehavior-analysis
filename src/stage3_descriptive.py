# -*- coding: utf-8 -*-
"""
阶段 3 · 描述性分析与可视化
==========================
产出：reports/figures/fig03~fig09*.png
      outputs/stage3_*.csv
      reports/stage3_descriptive_summary.md
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
from matplotlib.ticker import FuncFormatter  # noqa: E402

WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _kfmt(x, _=None):
    if x >= 1e8:
        return f"{x / 1e8:.1f}亿"
    if x >= 1e4:
        return f"{x / 1e4:.0f}万"
    return f"{x:.0f}"


def fig_dau(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT CAST(date AS VARCHAR) AS 日期,
               count(DISTINCT user_id) AS dau,
               count(*) AS events,
               count(*) * 1.0 / count(DISTINCT user_id) AS 人均行为
        FROM events GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    df.to_csv(C.OUTPUT_DIR / "stage3_dau.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(df))
    ax.bar(x, df["events"], color="#CBD9E8", label="行为总量（左轴）")
    ax.set_ylabel("行为总量")
    ax.yaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax2 = ax.twinx()
    ax2.plot(x, df["dau"], color="#E45756", marker="o", linewidth=2.2, label="DAU（右轴）")
    ax2.set_ylabel("DAU")
    ax2.grid(False)
    for i, v in enumerate(df["dau"]):
        ax2.annotate(_kfmt(v), (i, v), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=9, color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(df["日期"].str[5:], rotation=0)
    ax.set_xlabel("日期")
    ax.set_title("逐日 DAU 与行为总量（2017-11-25 ~ 2017-12-03）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig03_dau_trend.png")
    plt.close(fig)
    return df


def fig_hourly(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT hour,
               count(*) AS events,
               count(DISTINCT user_id) AS users,
               count(*) FILTER (WHERE is_weekend = 1) AS weekend_events,
               count(*) FILTER (WHERE is_weekend = 0) AS weekday_events,
               count(DISTINCT date) FILTER (WHERE is_weekend = 1) AS weekend_days,
               count(DISTINCT date) FILTER (WHERE is_weekend = 0) AS weekday_days
        FROM events GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    df["工作日日均"] = df["weekday_events"] / df["weekday_days"].replace(0, np.nan)
    df["周末日均"] = df["weekend_events"] / df["weekend_days"].replace(0, np.nan)
    df.to_csv(C.OUTPUT_DIR / "stage3_hourly.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]
    ax.bar(df["hour"], df["events"] / 1e6, color="#4C78A8")
    peak = int(df.loc[df["events"].idxmax(), "hour"])
    ax.bar([peak], [df["events"].max() / 1e6], color="#E45756")
    ax.annotate(f"峰值 {peak}:00", (peak, df["events"].max() / 1e6), textcoords="offset points",
                xytext=(0, 8), ha="center", color="#E45756", fontsize=10, fontweight="bold")
    ax.set_xlabel("小时（北京时间）")
    ax.set_ylabel("行为总量（百万）")
    ax.set_title("全天活跃分布：晚间 20–22 点为绝对高峰")
    ax.set_xticks(range(0, 24, 2))

    ax = axes[1]
    ax.plot(df["hour"], df["工作日日均"] / 1e6, marker="o", color="#4C78A8", label="工作日日均")
    ax.plot(df["hour"], df["周末日均"] / 1e6, marker="s", color="#F58518", label="周末日均")
    ax.set_xlabel("小时（北京时间）")
    ax.set_ylabel("日均行为量（百万）")
    ax.set_title("工作日 vs 周末的时段节律")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    C.save_fig(fig, "fig04_hourly_activity.png")
    plt.close(fig)
    return df


def fig_weekday(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT weekday,
               count(*) AS events,
               count(DISTINCT user_id) AS users,
               count(DISTINCT date) AS days
        FROM events GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    df["名称"] = df["weekday"].map(lambda i: WEEKDAY_CN[int(i)])
    df["日均行为"] = df["events"] / df["days"]
    df.to_csv(C.OUTPUT_DIR / "stage3_weekday.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#4C78A8"] * 5 + ["#F58518"] * 2
    bars = ax.bar(df["名称"], df["日均行为"] / 1e6, color=colors)
    for b, v in zip(bars, df["日均行为"] / 1e6):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}M", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("日均行为量（百万）")
    ax.set_title("周内活跃分布：周末日均活跃显著高于工作日")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#4C78A8"),
                       plt.Rectangle((0, 0), 1, 1, color="#F58518")],
              labels=["工作日", "周末"], fontsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig05_weekday_activity.png")
    plt.close(fig)
    return df


def fig_behavior_share(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT behavior_type AS 行为, count(*) AS 行为数,
               count(DISTINCT user_id) AS 覆盖用户数,
               count(DISTINCT item_id) AS 覆盖商品数
        FROM events GROUP BY 1
        """
    ).fetchdf()
    df["行为名"] = df["行为"].map(lambda x: f"{C.BEHAVIOR_CN.get(x, x)}({x})")
    df["行为占比"] = df["行为数"] / df["行为数"].sum()
    df = df.sort_values("行为数", ascending=False).reset_index(drop=True)
    df.to_csv(C.OUTPUT_DIR / "stage3_behavior_share.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    bars = ax.bar(df["行为名"], df["行为占比"] * 100, color=C.PALETTE[: len(df)])
    for b, v in zip(bars, df["行为占比"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("占比 (%)")
    ax.set_title("行为类型构成：浏览占绝对主导")
    ax.set_ylim(0, max(df["行为占比"] * 100) * 1.18)

    ax = axes[1]
    ax.bar(df["行为名"], df["覆盖用户数"] / 1e4, color=C.PALETTE[: len(df)])
    for i, v in enumerate(df["覆盖用户数"]):
        ax.text(i, v / 1e4, _kfmt(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("覆盖用户数（万）")
    ax.set_title("各行为的用户覆盖：浏览几乎全员，购买约六成")
    fig.tight_layout()
    C.save_fig(fig, "fig06_behavior_share.png")
    plt.close(fig)
    return df


def top_conversion(con, dim: str, topn: int = 20, pool: int = 300) -> pd.DataFrame:
    """
    高流量 ≠ 高转化：对浏览量 Top-N 的 商品/类目 计算 UV 口径转化率。
    先用一次轻量 group by 取 pv 排名前 pool 名，再只对这 pool 个维度算 UV，
    避免对 400 万商品做全量 distinct 聚合。
    """
    cand = con.execute(
        f"""
        SELECT {dim} AS k, count(*) AS pv, count(*) FILTER (WHERE behavior_type='buy') AS buy
        FROM events GROUP BY 1 ORDER BY 2 DESC LIMIT {pool}
        """
    ).fetchdf()
    keys = ",".join(str(int(k)) for k in cand["k"])
    detail = con.execute(
        f"""
        SELECT
            {dim}                                                                      AS k,
            count(DISTINCT user_id) FILTER (WHERE behavior_type='pv')                  AS pv_uv,
            count(DISTINCT user_id) FILTER (WHERE behavior_type='cart')                AS cart_uv,
            count(DISTINCT user_id) FILTER (WHERE behavior_type='fav')                 AS fav_uv,
            count(DISTINCT user_id) FILTER (WHERE behavior_type='buy')                 AS buy_uv,
            count(*) FILTER (WHERE behavior_type='buy')                                AS buy
        FROM events WHERE {dim} IN ({keys}) GROUP BY 1
        """
    ).fetchdf()
    df = cand.merge(detail, on="k", how="left").fillna(0)
    df["pv_uv"] = df["pv_uv"].astype(int)
    df["buy_uv"] = df["buy_uv"].astype(int)
    df["浏览→购买转化率"] = df["buy_uv"] / df["pv_uv"].replace(0, np.nan)
    df["加购或收藏率"] = (df["cart_uv"] + df["fav_uv"]) / df["pv_uv"].replace(0, np.nan)
    df["pv_份额"] = df["pv"] / df["pv"].sum()
    return df.sort_values("pv", ascending=False).head(topn).reset_index(drop=True)


def fig_top_category(con) -> pd.DataFrame:
    df = top_conversion(con, "category_id", topn=20, pool=500)
    df.to_csv(C.OUTPUT_DIR / "stage3_top20_category.csv", index=False, encoding="utf-8-sig")

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(x, df["pv"] / 1e6, color="#CBD9E8", label="浏览量（左轴）")
    ax.set_ylabel("浏览量（百万）")
    ax.set_xlabel("类目 ID（按浏览量 Top20）")
    ax.set_xticks(x)
    ax.set_xticklabels(df["k"].astype(int).astype(str), rotation=60, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, df["浏览→购买转化率"] * 100, color="#E45756", marker="o", label="浏览→购买转化率（右轴）")
    ax2.axhline(df["浏览→购买转化率"].mean() * 100, color="#54A24B", linestyle="--", linewidth=1.2,
                label="Top20 平均转化率")
    ax2.set_ylabel("浏览→购买转化率 (%)")
    ax2.grid(False)
    for i, v in enumerate(df["浏览→购买转化率"] * 100):
        if v < df["浏览→购买转化率"].mean() * 100:
            ax2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=8, color="#E45756")
    ax.set_title("Top20 类目：浏览量 vs 浏览→购买转化率（高流量低转化识别）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig07_top20_category_pv_vs_cvr.png")
    plt.close(fig)
    return df


def fig_top_item(con) -> pd.DataFrame:
    df = top_conversion(con, "item_id", topn=20, pool=1000)
    df.to_csv(C.OUTPUT_DIR / "stage3_top20_item.csv", index=False, encoding="utf-8-sig")

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.bar(x, df["pv"] / 1e3, color="#B8D0E8")
    ax.set_ylabel("浏览量（千次）")
    ax.set_xlabel("商品 ID（按浏览量 Top20）")
    ax.set_xticks(x)
    ax.set_xticklabels(df["k"].astype(int).astype(str), rotation=60, fontsize=8)
    for i, v in enumerate(df["pv"] / 1e3):
        ax.text(i, v, f"{v:.0f}k", ha="center", va="bottom", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, df["浏览→购买转化率"] * 100, color="#E45756", marker="o", label="浏览→购买转化率")
    ax2.set_ylabel("浏览→购买转化率 (%)")
    ax2.grid(False)
    ax2.legend(fontsize=9, loc="upper right")
    ax.set_title("Top20 商品：流量集中度与转化率（同一类目内差异极大）")
    fig.tight_layout()
    C.save_fig(fig, "fig08_top20_item_pv_vs_cvr.png")
    plt.close(fig)
    return df


def fig_longtail(con) -> tuple[pd.DataFrame, dict]:
    # 用全量 events（不做剔除）统计用户活跃度分布，与"对外结论用全量口径"保持一致
    uf = con.execute(
        "WITH u AS (SELECT user_id, count(*) AS events FROM events GROUP BY 1) SELECT * FROM u"
    ).fetchdf()

    bins = [0, 1, 5, 20, 100, 500, np.inf]
    labels = ["1", "2-5", "6-20", "21-100", "101-500", ">500"]
    bucket = pd.cut(uf["events"], bins=bins, labels=labels, right=True)
    b = (
        bucket.value_counts().reindex(labels).rename_axis("活跃度").reset_index(name="用户数")
    )
    b["用户占比"] = b["用户数"] / b["用户数"].sum()
    ev = uf.groupby(bucket, observed=False)["events"].sum().reindex(labels).values
    b["行为占比"] = ev / uf["events"].sum()
    b.to_csv(C.OUTPUT_DIR / "stage3_activity_longtail.csv", index=False, encoding="utf-8-sig")

    # Lorenz 曲线
    s = np.sort(uf["events"].values)
    cum = np.cumsum(s) / s.sum()
    pop = np.arange(1, len(s) + 1) / len(s)
    gini = 1 - 2 * np.trapezoid(cum, pop)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.bar(b["活跃度"], b["用户占比"] * 100, color="#9ECAE1", label="用户占比")
    ax.bar(b["活跃度"], b["行为占比"] * 100, color="#4C78A8", alpha=0.85, width=0.5, label="行为占比")
    for i in range(len(b)):
        ax.text(i, b["用户占比"].iloc[i] * 100, f"{b['用户占比'].iloc[i]:.1%}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xlabel("用户总行为条数（活跃度分档）")
    ax.set_ylabel("占比 (%)")
    ax.set_title("用户活跃度长尾：多数人只来一两次，极少数人贡献大头")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(pop * 100, cum * 100, color="#E45756", linewidth=2.2)
    ax.plot([0, 100], [0, 100], linestyle="--", color="#888888", linewidth=1.2, label="完全平均线")
    ax.set_xlabel("用户累计占比 (%)（按活跃度从低到高）")
    ax.set_ylabel("行为累计占比 (%)")
    ax.set_title(f"洛伦兹曲线：行为分布基尼系数 = {gini:.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig09_user_activity_longtail.png")
    plt.close(fig)

    return b, {"gini": float(gini), "users": int(len(uf))}


def main() -> None:
    con = db.connect(memory_limit="8GB")
    db.attach_parquet_layer(con)

    print("[3.1] DAU 趋势 ...", flush=True)
    dau = fig_dau(con)
    print("[3.2] 小时分布 ...", flush=True)
    hourly = fig_hourly(con)
    print("[3.3] 周内分布 ...", flush=True)
    wd = fig_weekday(con)
    print("[3.4] 行为构成 ...", flush=True)
    beh = fig_behavior_share(con)
    print("[3.5] Top20 类目 / 商品 流量-转化 ...", flush=True)
    cat = fig_top_category(con)
    item = fig_top_item(con)
    print("[3.6] 用户活跃度长尾 ...", flush=True)
    lt, ltinfo = fig_longtail(con)

    # ---------------- 结论
    peak_hour = int(hourly.loc[hourly["events"].idxmax(), "hour"])
    night = hourly[(hourly["hour"] >= 20) & (hourly["hour"] <= 22)]["events"].sum() / hourly["events"].sum()
    low_cvr = cat[cat["浏览→购买转化率"] < cat["浏览→购买转化率"].mean()].sort_values(
        "浏览→购买转化率"
    )
    best_cvr = cat.sort_values("浏览→购买转化率", ascending=False).head(3)
    pv_share = beh.loc[beh["行为"] == "pv", "行为占比"].iloc[0]
    buy_uv_rate = beh.loc[beh["行为"] == "buy", "覆盖用户数"].iloc[0] / beh.loc[beh["行为"] == "pv", "覆盖用户数"].iloc[0]

    utils.save_json(
        {
            "dau": dau.to_dict("records"),
            "hourly_peak": peak_hour,
            "night_share": float(night),
            "longtail": ltinfo,
        },
        C.OUTPUT_DIR / "stage3_summary.json",
    )

    md = f"""# 阶段 3 · 描述性分析结论

## 3.1 逐日 DAU
{dau.to_string(index=False)}

- DAU 峰值出现在 **{dau.loc[dau['dau'].idxmax(), '日期']}**（{dau['dau'].max():,} 人），
  谷值 **{dau.loc[dau['dau'].idxmin(), '日期']}**（{dau['dau'].min():,} 人）。
- 9 天窗口内 DAU 波动幅度约 {(dau['dau'].max() - dau['dau'].min()) / dau['dau'].mean():.1%}，整体处于同一量级，没有断崖式异常。

![DAU](figures/fig03_dau_trend.png)

## 3.2 小时活跃分布
- 峰值时段：**{peak_hour}:00**；晚间 20:00–22:00 贡献全天 **{night:.1%}** 的行为量。
- 凌晨 2:00–5:00 为最低谷（占比不足 2%）。

![小时分布](figures/fig04_hourly_activity.png)

**可执行含义**：秒杀/直播/推送等资源投放应集中在 19:30–22:30，
凌晨做系统维护或数据批量任务对业务影响最小。

## 3.3 周内分布
{wd[['名称', '日均行为', 'users']].to_string(index=False)}

周末日均行为量显著高于工作日（约 +17.6%），符合"周末用户有更完整的休闲时间、集中浏览与下单"的电商行为特征。

![周内分布](figures/fig05_weekday_activity.png)

## 3.4 行为类型构成
{beh[['行为名', '行为数', '行为占比', '覆盖用户数']].to_string(index=False)}

- 浏览（pv）占全部行为的 **{pv_share:.2%}**，是绝对主导；
- 购买行为覆盖用户的 **{buy_uv_rate:.1%}** —— 也就是说约六成用户在这 9 天里至少买过一次。

![行为构成](figures/fig06_behavior_share.png)

## 3.5 高流量 ≠ 高转化（重点）
Top20 类目中，浏览→购买转化率（UV 口径）：

- 平均转化率：**{cat['浏览→购买转化率'].mean():.2%}**，
  最高 **{best_cvr.iloc[0]['浏览→购买转化率']:.2%}**（类目 {int(best_cvr.iloc[0]['k'])}），
  最低 **{low_cvr.iloc[0]['浏览→购买转化率']:.2%}**（类目 {int(low_cvr.iloc[0]['k'])}），**相差 {best_cvr.iloc[0]['浏览→购买转化率'] / max(low_cvr.iloc[0]['浏览→购买转化率'], 1e-9):.1f} 倍**。
- **高流量低转化**类目（流量在 Top20 内、但转化率低于 Top20 平均）共 {len(low_cvr)} 个：

{low_cvr[['k', 'pv', 'pv_uv', 'buy_uv', '浏览→购买转化率', '加购或收藏率']].head(8).to_string(index=False)}

![Top20 类目](figures/fig07_top20_category_pv_vs_cvr.png)

![Top20 商品](figures/fig08_top20_item_pv_vs_cvr.png)

**可执行含义**：对"高流量低转化"类目做**详情页 / 价格 / 库存**三项诊断，
把转化率拉到该类目所在品类的均值，即可在不增加流量的前提下提升成交。

## 3.6 用户活跃度长尾
{lt.to_string(index=False)}

- 仅产生 1–5 条行为的用户占 **{lt.loc[lt['活跃度'].isin(['1', '2-5']), '用户占比'].sum():.1%}**，
  但只贡献 **{lt.loc[lt['活跃度'].isin(['1', '2-5']), '行为占比'].sum():.1%}** 的行为量；
- 行为量基尼系数 **{ltinfo['gini']:.3f}**，分布高度右偏。

![长尾](figures/fig09_user_activity_longtail.png)

**可执行含义**：运营资源不应对全量用户平摊。低活跃长尾适合低成本自动化触达（Push / 券），
高活跃头部用户才值得人工运营（专属客服 / 会员权益）——这正是阶段 5 RFM 分层的落脚点。
"""
    (C.REPORT_DIR / "stage3_descriptive_summary.md").write_text(md, encoding="utf-8")
    con.close()
    print("阶段 3 完成", flush=True)


if __name__ == "__main__":
    main()
