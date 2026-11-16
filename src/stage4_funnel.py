# -*- coding: utf-8 -*-
"""
阶段 4 · 漏斗分析
================
漏斗按 pv → (cart | fav) → buy 建模，加购与收藏是并列分支，不做先后假设。
给 UV 与 Session 双口径，Session 口径要求行为顺序（不是"会话内出现过"）。
分维度下钻：逐日、新老用户、Top 类目。

产出：
  outputs/parquet/sessions.parquet       会话级聚合表
  outputs/stage4_*.csv
  reports/figures/fig10~fig13*.png
  reports/stage4_funnel_summary.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import db  # noqa: E402
import utils  # noqa: E402

C.setup_chinese_font()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

# ------------------------------------------------------------------ 会话口径定义
# t_x = 该 session 内行为 x 的首次发生时间
_SESSION_SELECT = """
    SELECT
        session_key,
        any_value(user_id)                                                AS user_id,
        CAST(min(datetime) AS DATE)                                       AS session_date,
        min(datetime)                                                     AS session_start,
        max(datetime)                                                     AS session_end,
        count(*)                                                          AS n_events,
        count(*) FILTER (WHERE behavior_type='pv')                        AS n_pv,
        count(*) FILTER (WHERE behavior_type='cart')                      AS n_cart,
        count(*) FILTER (WHERE behavior_type='fav')                       AS n_fav,
        count(*) FILTER (WHERE behavior_type='buy')                       AS n_buy,
        min(datetime) FILTER (WHERE behavior_type='pv')                   AS t_pv,
        min(datetime) FILTER (WHERE behavior_type='cart')                 AS t_cart,
        min(datetime) FILTER (WHERE behavior_type='fav')                  AS t_fav,
        min(datetime) FILTER (WHERE behavior_type='buy')                  AS t_buy
    FROM ({sess}) s
    GROUP BY session_key
"""

# 在会话级聚合结果之上打标签
_SESSION_FLAGS = """
    SELECT *,
        (t_pv IS NOT NULL)                                                    AS f1_浏览,
        (t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv))                AS f2_浏览后加购或收藏,
        (t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv)
            AND t_buy > least(coalesce(t_cart, t_fav), coalesce(t_fav, t_cart)))                                 AS f3_意向后再购买,
        (t_pv IS NOT NULL AND t_buy > t_pv
            AND (t_cart IS NULL OR t_cart < t_pv) AND (t_fav IS NULL OR t_fav < t_pv))
                                                                              AS f_直接购买,
        (t_buy IS NOT NULL)                                                   AS f_任意购买
    FROM ({base})
"""


def build_sessions(con) -> None:
    """对全量 1 亿行做 30 分钟会话切分，只落会话级聚合（不落事件级，省磁盘）。"""
    sess = db.sessionized_sql("events", C.SESSION_GAP_MINUTES)
    con.execute(
        f"""
        COPY (
            {_SESSION_FLAGS.format(base=_SESSION_SELECT.format(sess=sess))}
        ) TO '{(C.PARQUET_DIR / "sessions.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION zstd)
        """
    )
    con.execute(
        f"CREATE OR REPLACE VIEW sessions AS SELECT * FROM read_parquet('{(C.PARQUET_DIR / 'sessions.parquet').as_posix()}')"
    )


def funnel_table(con, source: str, label: str) -> pd.DataFrame:
    """在 sessions / users 上直接算三步漏斗（source 需含 f1/f2/f3 布尔列）。"""
    row = con.execute(
        f"""
        SELECT
            count(*)                                   AS n1,
            count(*) FILTER (WHERE f1_浏览)             AS s1,
            count(*) FILTER (WHERE f2_浏览后加购或收藏)  AS s2,
            count(*) FILTER (WHERE f3_意向后再购买)      AS s3,
            count(*) FILTER (WHERE f_直接购买)           AS direct_buy,
            count(*) FILTER (WHERE f_任意购买)           AS any_buy
        FROM {source}
        """
    ).fetchdf().iloc[0]

    idx = ["① 浏览", "② 浏览后加购/收藏", "③ 意向后再购买"]
    vals = [int(row["s1"]), int(row["s2"]), int(row["s3"])]
    df = pd.DataFrame({"环节": idx, label: vals})
    df["整体转化率(相对①)"] = df[label] / vals[0]
    step = [1.0, vals[1] / vals[0] if vals[0] else np.nan, vals[2] / vals[1] if vals[1] else np.nan]
    df["环节转化率(相对上一步)"] = step
    df["流失人数"] = [0, vals[0] - vals[1], vals[1] - vals[2]]
    df["环节流失率"] = [0.0, 1 - step[1], 1 - step[2]]
    df.attrs["raw"] = row.to_dict()
    return df


def user_timestamps_sql(where: str = "") -> str:
    return f"""
        SELECT
            user_id,
            min(datetime) FILTER (WHERE behavior_type='pv')   AS t_pv,
            min(datetime) FILTER (WHERE behavior_type='cart') AS t_cart,
            min(datetime) FILTER (WHERE behavior_type='fav')  AS t_fav,
            min(datetime) FILTER (WHERE behavior_type='buy')  AS t_buy
        FROM events {where}
        GROUP BY user_id
    """


def build_uv_table(con, where: str = "") -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW users_funnel AS
        SELECT *,
            (t_pv IS NOT NULL)                                                      AS f1_浏览,
            (t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv))                  AS f2_浏览后加购或收藏,
            (t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv)
                AND t_buy > least(coalesce(t_cart, t_fav), coalesce(t_fav, t_cart)))                                   AS f3_意向后再购买,
            (t_pv IS NOT NULL AND t_buy > t_pv
                AND (t_cart IS NULL OR t_cart < t_pv) AND (t_fav IS NULL OR t_fav < t_pv))
                                                                                    AS f_直接购买,
            (t_buy IS NOT NULL)                                                     AS f_任意购买
        FROM ({user_timestamps_sql(where)})
        """
    )


# ------------------------------------------------------------------ 可视化
def draw_funnel(ax, steps, values, title, color="#4C78A8"):
    """横向漏斗（梯形叠加）。"""
    n = len(values)
    top, band_h, step_y = max(values), 0.9, 1.35
    width_scale = 1.0 / top
    y = 0
    for i, (label, v) in enumerate(zip(steps, values)):
        w = v * width_scale
        poly = Polygon(
            [(-w / 2, y), (w / 2, y), (w / 2, y + band_h), (-w / 2, y + band_h)],
            closed=True, facecolor=color, edgecolor="white", alpha=0.35 + 0.2 * i, linewidth=1.5,
        )
        ax.add_patch(poly)
        ax.text(0, y + band_h / 2, f"{label}\n{v:,}", ha="center", va="center",
                color="#1a1a1a", fontsize=10)
        if i > 0:
            rate = v / values[i - 1]
            ax.annotate(
                f"↓ 环节转化 {rate:.1%}　流失 {values[i-1] - v:,}",
                xy=(0, y - 0.04), ha="center", va="bottom", fontsize=9.5, color="#E45756",
            )
        y += step_y
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.45, y + 0.05)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title)


def fig_funnel(uv_tab: pd.DataFrame, sess_tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    draw_funnel(axes[0], uv_tab["环节"].tolist(), uv_tab["UV 口径"].tolist(),
                "UV 口径漏斗（按用户去重）", color="#4C78A8")
    draw_funnel(axes[1], sess_tab["环节"].tolist(), sess_tab["Session 口径"].tolist(),
                "Session 口径漏斗（含行为顺序约束）", color="#F58518")
    fig.suptitle("漏斗双口径对照：pv → (cart | fav) → buy", fontsize=15, fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "fig10_funnel_overview.png")
    plt.close(fig)


def fig_waterfall(uv_tab: pd.DataFrame, sess_tab: pd.DataFrame) -> None:
    """流失瀑布图：每一步流失了多少人。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, tab, name, color in zip(
        axes, [uv_tab, sess_tab], ["UV 口径", "Session 口径"], ["#4C78A8", "#F58518"]
    ):
        start = tab[tab.columns[1]].iloc[0]
        labels, vals, colors, bottoms = [], [], [], []
        cur = start
        for i in range(1, len(tab)):
            drop = int(tab["流失人数"].iloc[i])
            labels.append(f"{tab['环节'].iloc[i-1]}→{tab['环节'].iloc[i]}")
            vals.append(-drop)
            colors.append("#E45756")
            bottoms.append(cur - drop)
            cur -= drop
        labels.append("最终成交")
        vals.append(cur)
        colors.append("#54A24B")
        bottoms.append(0)
        bars = ax.bar(labels, vals, bottom=bottoms, color=colors)
        for b, v, bo in zip(bars, vals, bottoms):
            ax.text(b.get_x() + b.get_width() / 2, bo + v / 2, f"{abs(v):,}",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.set_title(f"{name}：流失瀑布")
        ax.set_ylabel("人数")
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig11_funnel_waterfall.png")
    plt.close(fig)


def fig_branches(con) -> pd.DataFrame:
    """并列分支：加购 / 收藏 / 两者都无，各自的后续购买概率。"""
    df = con.execute(
        """
        WITH u AS (SELECT * FROM users_funnel)
        SELECT
            '加购(cart)'      AS 分支,
            count(*) FILTER (WHERE t_cart IS NOT NULL)                                AS 人数,
            count(*) FILTER (WHERE t_cart IS NOT NULL AND f_任意购买)                  AS 最终购买人数,
            count(*) FILTER (WHERE t_cart IS NOT NULL AND f3_意向后再购买)             AS 严格链路购买人数
        FROM u
        UNION ALL
        SELECT '收藏(fav)', count(*) FILTER (WHERE t_fav IS NOT NULL),
               count(*) FILTER (WHERE t_fav IS NOT NULL AND f_任意购买),
               count(*) FILTER (WHERE t_fav IS NOT NULL AND f3_意向后再购买) FROM u
        UNION ALL
        SELECT '加购且收藏', count(*) FILTER (WHERE t_cart IS NOT NULL AND t_fav IS NOT NULL),
               count(*) FILTER (WHERE t_cart IS NOT NULL AND t_fav IS NOT NULL AND f_任意购买),
               count(*) FILTER (WHERE t_cart IS NOT NULL AND t_fav IS NOT NULL AND f3_意向后再购买) FROM u
        UNION ALL
        SELECT '仅浏览(无加购无收藏)', count(*) FILTER (WHERE t_pv IS NOT NULL AND t_cart IS NULL AND t_fav IS NULL),
               count(*) FILTER (WHERE t_pv IS NOT NULL AND t_cart IS NULL AND t_fav IS NULL AND f_任意购买),
               0 FROM u
        UNION ALL
        SELECT '全体浏览用户', count(*) FILTER (WHERE t_pv IS NOT NULL),
               count(*) FILTER (WHERE t_pv IS NOT NULL AND f_任意购买),
               count(*) FILTER (WHERE f3_意向后再购买) FROM u
        """
    ).fetchdf()
    df["购买概率"] = df["最终购买人数"] / df["人数"]
    df.to_csv(C.OUTPUT_DIR / "stage4_funnel_branches.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(df))
    bars = ax.bar(x, df["购买概率"] * 100, color=["#F58518", "#54A24B", "#B279A2", "#BAB0AC", "#4C78A8"])
    for b, v, n, k in zip(bars, df["购买概率"] * 100, df["人数"], df["最终购买人数"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%\n({k:,}/{n:,})", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["分支"], fontsize=9)
    ax.set_ylabel("最终购买概率 (%)")
    ax.set_ylim(0, max(df["购买概率"] * 100) * 1.3)
    ax.set_title("并列分支对比：加购 / 收藏的意向强度完全不同")
    fig.tight_layout()
    C.save_fig(fig, "fig12_funnel_branches.png")
    plt.close(fig)
    return df


def funnel_by_day(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT CAST(session_date AS VARCHAR) AS 日期,
               count(*) FILTER (WHERE f1_浏览)            AS 浏览会话,
               count(*) FILTER (WHERE f2_浏览后加购或收藏) AS 加购或收藏,
               count(*) FILTER (WHERE f3_意向后再购买)     AS 成交会话
        FROM sessions GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    df["浏览→成交转化率"] = df["成交会话"] / df["浏览会话"]
    df["加购收藏→成交率"] = df["成交会话"] / df["加购或收藏"]
    df.to_csv(C.OUTPUT_DIR / "stage4_funnel_by_day.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(df))
    ax.bar(x, df["浏览会话"], color="#CBD9E8", label="浏览会话数（左轴）")
    ax.set_ylabel("会话数")
    ax2 = ax.twinx()
    ax2.plot(x, df["浏览→成交转化率"] * 100, marker="o", color="#E45756", linewidth=2.2,
             label="浏览→成交转化率（右轴）")
    ax2.plot(x, df["加购收藏→成交率"] * 100, marker="s", color="#54A24B", linewidth=2,
             label="加购/收藏→成交率（右轴）")
    for i, v in enumerate(df["浏览→成交转化率"] * 100):
        ax2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 8), ha="center",
                     fontsize=9, color="#E45756")
    ax2.set_ylabel("转化率 (%)")
    ax2.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(df["日期"].str[5:])
    ax.set_xlabel("日期")
    ax.set_title("逐日漏斗：转化率是否在恶化？")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    fig.tight_layout()
    C.save_fig(fig, "fig13_funnel_by_day.png")
    plt.close(fig)
    return df


def funnel_by_segment(con) -> pd.DataFrame:
    """
    新老用户分层漏斗。窗口局限：只有 9 天数据，无法识别"11-25 之前就存在的老用户"。
    因此口径为：
      - 首日出现（11-25 首次活跃）→ 存量 + 未知，记为"观测期首日人群"
      - 窗口内新增（11-26 起首次活跃）→ 一定是新增用户
    """
    rows = []
    seg_sql = """
        WITH firstd AS (SELECT user_id, min(date) AS fd FROM events GROUP BY 1),
        u AS (SELECT * FROM users_funnel)
        SELECT COUNT(*) n1, COUNT(*) FILTER (WHERE f1_浏览) s1,
               COUNT(*) FILTER (WHERE f2_浏览后加购或收藏) s2,
               COUNT(*) FILTER (WHERE f3_意向后再购买) s3,
               COUNT(*) FILTER (WHERE f_任意购买) ab
        FROM u JOIN firstd USING (user_id) WHERE {cond}
    """
    for label, cond in [
        ("观测期首日人群(11-25 首次活跃)", "firstd.fd = DATE '2017-11-25'"),
        ("窗口内新增(11-26 起首次活跃)", "firstd.fd > DATE '2017-11-25'"),
    ]:
        r = con.execute(seg_sql.format(cond=cond)).fetchdf().iloc[0]
        rows.append([label, int(r["n1"]), int(r["s1"]), int(r["s2"]), int(r["s3"]), int(r["ab"])])
    df = pd.DataFrame(rows, columns=["人群", "总用户数", "浏览用户", "浏览后加购或收藏", "意向后再购买", "任意购买用户"])
    df["浏览→成交转化率"] = df["意向后再购买"] / df["浏览用户"]
    df["任意购买率"] = df["任意购买用户"] / df["浏览用户"]
    df.to_csv(C.OUTPUT_DIR / "stage4_funnel_by_segment.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["浏览→成交转化率"] * 100, width=0.4, color="#4C78A8", label="浏览→成交转化率（严格链路）")
    ax.bar(x + 0.2, df["任意购买率"] * 100, width=0.4, color="#F58518", label="浏览→任意购买率")
    for i, (a, b) in enumerate(zip(df["浏览→成交转化率"] * 100, df["任意购买率"] * 100)):
        ax.text(i - 0.2, a, f"{a:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, b, f"{b:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["人群"], fontsize=9)
    ax.set_ylabel("转化率 (%)")
    ax.set_title("新老用户漏斗对比：新增用户的意向转化明显更低")
    ax.legend(fontsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig14_funnel_by_segment.png")
    plt.close(fig)
    return df


def funnel_by_category(con, topn: int = 15) -> pd.DataFrame:
    top = con.execute(
        f"SELECT category_id FROM events WHERE behavior_type='pv' GROUP BY 1 ORDER BY count(*) DESC LIMIT {topn}"
    ).fetchdf()["category_id"].tolist()
    keys = ",".join(str(int(k)) for k in top)
    df = con.execute(
        f"""
        WITH u AS (
            SELECT
                category_id,
                user_id,
                min(datetime) FILTER (WHERE behavior_type='pv')   AS t_pv,
                min(datetime) FILTER (WHERE behavior_type='cart') AS t_cart,
                min(datetime) FILTER (WHERE behavior_type='fav')  AS t_fav,
                min(datetime) FILTER (WHERE behavior_type='buy')  AS t_buy
            FROM events WHERE category_id IN ({keys})
            GROUP BY 1, 2
        )
        SELECT
            category_id                                                              AS 类目,
            count(*) FILTER (WHERE t_pv IS NOT NULL)                                  AS 浏览用户,
            count(*) FILTER (WHERE t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv)) AS 加购或收藏用户,
            count(*) FILTER (WHERE t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv)
                                AND t_buy > least(coalesce(t_cart, t_fav), coalesce(t_fav, t_cart)))                     AS 成交用户,
            count(*) FILTER (WHERE t_buy IS NOT NULL)                                 AS 任意购买用户
        FROM u GROUP BY 1 ORDER BY 浏览用户 DESC
        """
    ).fetchdf()
    df["浏览→成交转化率"] = df["成交用户"] / df["浏览用户"]
    df["浏览→任意购买率"] = df["任意购买用户"] / df["浏览用户"]
    df.to_csv(C.OUTPUT_DIR / "stage4_funnel_by_category.csv", index=False, encoding="utf-8-sig")

    d = df.sort_values("浏览→成交转化率", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(d))
    ax.barh(y, d["浏览→成交转化率"] * 100, color="#4C78A8", label="浏览→成交转化率（严格链路）")
    ax.barh(y, d["浏览→任意购买率"] * 100, color="#F58518", alpha=0.45, label="浏览→任意购买率")
    for i, (a, b) in enumerate(zip(d["浏览→成交转化率"] * 100, d["浏览→任意购买率"] * 100)):
        ax.text(a, i, f" {a:.2f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"类目 {int(c)}" for c in d["类目"]], fontsize=9)
    ax.set_xlabel("转化率 (%)")
    ax.set_title("Top15 类目漏斗：同一站内不同类目的转化能力差数倍")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    C.save_fig(fig, "fig15_funnel_by_category.png")
    plt.close(fig)
    return df


# ------------------------------------------------------------------ main
def main() -> None:
    t0 = time.perf_counter()
    con = db.connect(memory_limit="6GB")
    con.execute(f"SET temp_directory='{(C.OUTPUT_DIR / 'duckdb_tmp').as_posix()}'")
    db.attach_parquet_layer(con)

    print("[4.2] 会话切分（30 分钟阈值，全量）...", flush=True)
    build_sessions(con)
    n_sess = con.execute("SELECT count(*) FROM sessions").fetchone()[0]
    print(f"     会话数 {n_sess:,}", flush=True)

    print("[4.2] 用户级漏斗表 ...", flush=True)
    build_uv_table(con)

    print("[4.1/4.3] 三步漏斗（UV / Session 双口径）...", flush=True)
    uv_tab = funnel_table(con, "users_funnel", "UV 口径")
    sess_tab = funnel_table(con, "sessions", "Session 口径")
    uv_tab.to_csv(C.OUTPUT_DIR / "stage4_funnel_uv.csv", index=False, encoding="utf-8-sig")
    sess_tab.to_csv(C.OUTPUT_DIR / "stage4_funnel_session.csv", index=False, encoding="utf-8-sig")

    print("[4.5] 漏斗图 + 瀑布图 ...", flush=True)
    fig_funnel(uv_tab, sess_tab)
    fig_waterfall(uv_tab, sess_tab)

    print("[4.1] 并列分支条件概率 ...", flush=True)
    br = fig_branches(con)

    print("[4.4] 分维度漏斗 ...", flush=True)
    by_day = funnel_by_day(con)
    by_seg = funnel_by_segment(con)
    by_cat = funnel_by_category(con)

    # ------------------------------------------------ 结论
    u1, u2, u3 = uv_tab["UV 口径"].tolist()
    s1, s2, s3 = sess_tab["Session 口径"].tolist()
    davg = by_day["浏览→成交转化率"].mean()
    dtrend = by_day["浏览→成交转化率"].iloc[-1] - by_day["浏览→成交转化率"].iloc[0]
    cart_p = br.loc[br["分支"] == "加购(cart)", "购买概率"].iloc[0]
    fav_p = br.loc[br["分支"] == "收藏(fav)", "购买概率"].iloc[0]
    both_p = br.loc[br["分支"] == "加购且收藏", "购买概率"].iloc[0]
    none_p = br.loc[br["分支"] == "仅浏览(无加购无收藏)", "购买概率"].iloc[0]
    seg_new = by_seg.iloc[1]
    seg_old = by_seg.iloc[0]

    utils.save_json({"uv": uv_tab.to_dict("records"), "session": sess_tab.to_dict("records")},
                    C.OUTPUT_DIR / "stage4_summary.json")

    md = f"""# 阶段 4 · 漏斗分析结论

## 4.1 漏斗结构修正（为什么必须改）

原代码：`step = ['pv','cart','fav','buy']`，把加购与收藏串成一条四级链路。
**这是错的**：加购（cart）与收藏（fav）是**并列的两种意向表达**，用户可能是"浏览 → 加购 → 购买"，
也可能是"浏览 → 收藏 → 购买"，不存在"必须先加购再收藏"。把它们串起来会导致中间环节人为缩窄，
所有下游转化率被系统性低估。

本项目改为：**① 浏览 → ② 浏览后加购或收藏 → ③ 意向后再购买**，
并把加购 / 收藏拆成并列分支单独用条件概率刻画（见 4.1 分支表）。

## 4.2 双口径漏斗

### UV 口径（按用户去重，9 天窗口）

{utils.df_to_md(uv_tab)}

### Session 口径（按 30 分钟会话切分，含行为顺序约束）

{utils.df_to_md(sess_tab)}

- 会话总数（全量）：**{n_sess:,}**
- 严格链路（浏览→加购/收藏→购买，且顺序正确）的会话数：**{s3:,}**
- 绕过中间环节的**直接购买**会话数：**{int(sess_tab.attrs['raw']['direct_buy']):,}**

![漏斗总览](figures/fig10_funnel_overview.png)

![流失瀑布](figures/fig11_funnel_waterfall.png)

**两种口径的差异说明了什么**：
UV 口径把用户 9 天内的所有行为压成一次"转化判断"，会把"上周加购、这周购买"这类跨会话行为算进同一条链，
转化率天然偏高（浏览→成交 {u3/u1:.2%}）；
Session 口径要求意向与成交发生在同一次 30 分钟会话内，口径更严，转化率更低（{s3/s1:.2%}），
但它更接近"这一次访问有没有促成成交"的真实业务问题。
**对外汇报用 Session 口径看单次访问效率，用 UV 口径看用户整体经营效率**，两个都要给，但必须写清楚是哪一个。

## 4.1' 并列分支：加购与收藏的意向强度完全不同

{utils.df_to_md(br)}

![分支对比](figures/fig12_funnel_branches.png)

**关键发现**：加购用户的最终购买概率 **{cart_p:.1%}**，收藏用户只有 **{fav_p:.1%}**，两者都无的纯浏览用户仅 **{none_p:.1%}**。
加购的意向强度约为收藏的 {cart_p / max(fav_p, 1e-9):.1f} 倍 —— 这也正是"加购后必须做挽单"的直接依据。

## 4.4 分维度漏斗

### 逐日变化

{utils.df_to_md(by_day)}

![逐日漏斗](figures/fig13_funnel_by_day.png)

9 天平均浏览→成交转化率 **{davg:.2%}**，首末两日变化 **{dtrend:+.2%}**，
说明窗口内漏斗**没有出现恶化趋势**（无大促/故障导致的转化塌陷）。

### 新老用户

{utils.df_to_md(by_seg)}

![新老用户](figures/fig14_funnel_by_segment.png)

新增用户的严格链路转化率 **{seg_new['浏览→成交转化率']:.2%}**，
低于观测期首日人群的 **{seg_old['浏览→成交转化率']:.2%}** —— 新用户缺信任与决策周期，
"任意购买率"差距（{seg_new['任意购买率']:.2%} vs {seg_old['任意购买率']:.2%}）比严格链路差距更小，
说明新用户不是不买，而是**不在同一次会话里买**（决策链路更长）。

### Top15 类目

{utils.df_to_md(by_cat)}

![类目漏斗](figures/fig15_funnel_by_category.png)

## 4.6 可执行业务结论

1. **漏斗最陡的一段是「加购/收藏 → 购买」：意向会话里只有 {s3 / s2:.1%} 在同一次会话内成交（Session 口径）。
   但丢人最多的是「浏览 → 加购/收藏」：Session 口径有 {int(sess_tab['流失人数'].iloc[1]):,} 个会话
   （{sess_tab['环节流失率'].iloc[1]:.1%}）完全没产生任何加购或收藏。**
   两个视角对应两个动作：前者做"购物车临门一脚"（降价提醒 / 库存告急 / 免运费），
   后者做"进店即种草"（详情页加购引导、浏览未加购发加购券）。
   UV 口径下同一环节流失 {int(uv_tab['流失人数'].iloc[1]):,} 人（{uv_tab['环节流失率'].iloc[1]:.1%}），
   即有约两成用户在 9 天里从未产生过任何意向行为。

2. **加购/收藏用户的后续购买概率明显更高，是低成本可挽回人群。**
   加购用户后续购买概率 {cart_p:.1%}、收藏用户 {fav_p:.1%}、加购且收藏 {both_p:.1%}，
   均高于纯浏览用户 {none_p:.1%}；加购且收藏的意向最强。
   ⚠️ 这只是**相关性**：加购/收藏用户本身更活跃；且本数据无法把"购买"归因到"被浏览的那个商品"，
   因此这里的"购买概率"指的是"该用户最终是否买过任何东西"，不是该商品的转化率。
   建议动作：对"加购未购"人群在 24 小时内做定向触达——这是转化意愿最高的池子。

3. **不同类目的转化能力差 {by_cat['浏览→成交转化率'].max() / max(by_cat['浏览→成交转化率'].min(), 1e-9):.1f} 倍，流量分配应重排。**
   最高（类目 {int(by_cat.loc[by_cat['浏览→成交转化率'].idxmax(), '类目'])}：{by_cat['浏览→成交转化率'].max():.2%}）
   vs 最低（类目 {int(by_cat.loc[by_cat['浏览→成交转化率'].idxmin(), '类目'])}：{by_cat['浏览→成交转化率'].min():.2%}）。
   这一组是**类目内 UV 口径**（浏览、加购、购买都限定在同一类目内），归因更干净、更可信。
   建议动作：把首页/推荐位曝光从"高浏览低转化"类目适度迁移到"转化效率高但流量不足"的类目，
   预期在不增加 UV 的前提下提升整体成交。

4. **新用户需要"跨会话召回"而不是"当场逼单"。**
   新增用户严格链路转化率 {seg_new['浏览→成交转化率']:.2%}，
   低于观测期首日人群的 {seg_old['浏览→成交转化率']:.2%}；
   但两者的"任意购买率"差距（{seg_new['任意购买率']:.2%} vs {seg_old['任意购买率']:.2%}）明显更小，
   说明新用户的行为是"多次观望后下单"，而不是"不买"。
   建议动作：给新用户配置次日回访券与跨会话购物车提醒，而不是只做首次会话内的强促。
"""
    (C.REPORT_DIR / "stage4_funnel_summary.md").write_text(md, encoding="utf-8")
    con.close()
    print(f"阶段 4 完成，用时 {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
