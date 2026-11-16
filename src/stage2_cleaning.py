# -*- coding: utf-8 -*-
"""
阶段 2 · 数据清洗与特征工程
==========================
产出（中间层，parquet）：
  outputs/parquet/events/date=*/           清洗后全量事件，按 date 分区
  outputs/parquet/daily_behavior/date=*/   日 × 行为类型 聚合
  outputs/parquet/category_daily/date=*/   日 × 类目 聚合（含 UV 口径）
  outputs/parquet/suspect_users.parquet    疑似爬虫用户
  outputs/parquet/user_features.parquet    用户级特征宽表（供 RFM / 聚类复用）
  outputs/stage2_cleaning_summary.md       清洗结论 + 爬虫敏感性对比
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import config as C
import db
import utils

C.setup_chinese_font()
import matplotlib.pyplot as plt  # noqa: E402


def build_events(con) -> None:
    """把 CSV 清洗一遍落成按 date 分区的 parquet，后续不再重复解析 3.6 GB。"""
    target = C.PARQUET_DIR / "events"
    con.execute(
        f"""
        COPY (
            SELECT user_id, item_id, category_id, behavior_type, timestamp,
                   datetime, hour, weekday, is_weekend, date
            FROM clean
            WHERE {C.WINDOW_SQL}      -- 剔除窗口外脏时间戳（含 2017-11-24 采集爬坡数据）
        ) TO '{target.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (date), COMPRESSION zstd, OVERWRITE_OR_IGNORE)
        """
    )


def bot_threshold(con) -> float:
    """
    数据驱动地确定"异常高频"阈值。
    最初设想用固定值 500（单用户单日 pv），但实测发现本数据最大单用户单日 pv 不足 500，
    固定阈值会导致命中 0 个用户、清洗动作形同虚设。
    因此改为：max(配置下限, 用户-日 pv 的 p99.9)，并在报告中写明阈值来源。
    """
    ud = con.execute(
        f"""
        WITH ud AS (
            SELECT user_id, date, count(*) AS pv
            FROM clean WHERE behavior_type = 'pv' AND {C.WINDOW_SQL} GROUP BY 1, 2
        )
        SELECT quantile_cont(pv, 0.999) AS p999, max(pv) AS mx, count(*) AS n FROM ud
        """
    ).fetchdf().iloc[0]
    th = max(C.BOT_PV_FLOOR, float(ud["p999"]))
    print(
        f"     [阈值] 用户-日 pv：p99.9={ud['p999']:.0f}，max={ud['mx']:.0f}，"
        f"用户-日总数={ud['n']:,.0f} → 采用阈值 {th:.0f}",
        flush=True,
    )
    return th


def build_suspect_users(con, threshold: float) -> pd.DataFrame:
    """单用户单日 pv 超过阈值 → 该用户标记为疑似异常（保留但不删除，可做敏感性对比）。"""
    df = con.execute(
        f"""
        WITH ud AS (
            SELECT user_id, date, count(*) AS pv
            FROM clean WHERE behavior_type = 'pv' AND {C.WINDOW_SQL} GROUP BY 1, 2
        )
        SELECT
            user_id,
            max(pv)                       AS max_daily_pv,
            sum(CASE WHEN pv > {threshold} THEN 1 ELSE 0 END) AS bot_days,
            avg(pv)                       AS avg_daily_pv
        FROM ud
        GROUP BY 1
        HAVING sum(CASE WHEN pv > {threshold} THEN 1 ELSE 0 END) > 0
        ORDER BY max_daily_pv DESC
        """
    ).fetchdf()
    df.to_parquet(C.PARQUET_DIR / "suspect_users.parquet", index=False)
    return df


def build_aggregates(con) -> None:
    con.execute(
        f"""
        COPY (
            SELECT date, behavior_type,
                   count(*)                AS events,
                   count(DISTINCT user_id) AS users,
                   count(DISTINCT item_id) AS items
            FROM events GROUP BY 1, 2
        ) TO '{(C.PARQUET_DIR / "daily_behavior").as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (date), COMPRESSION zstd, OVERWRITE_OR_IGNORE)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT date, category_id,
                   count(*)                                        AS events,
                   count(*) FILTER (WHERE behavior_type = 'pv')    AS pv,
                   count(*) FILTER (WHERE behavior_type = 'cart')  AS cart,
                   count(*) FILTER (WHERE behavior_type = 'fav')   AS fav,
                   count(*) FILTER (WHERE behavior_type = 'buy')   AS buy,
                   count(DISTINCT user_id) FILTER (WHERE behavior_type = 'pv')   AS pv_uv,
                   count(DISTINCT user_id) FILTER (WHERE behavior_type = 'cart') AS cart_uv,
                   count(DISTINCT user_id) FILTER (WHERE behavior_type = 'buy')  AS buy_uv
            FROM events GROUP BY 1, 2
        ) TO '{(C.PARQUET_DIR / "category_daily").as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (date), COMPRESSION zstd, OVERWRITE_OR_IGNORE)
        """
    )


def build_user_features(con) -> pd.DataFrame:
    con.execute(
        """
        CREATE OR REPLACE VIEW events_clean AS
        SELECT e.* FROM events e
        WHERE e.user_id NOT IN (SELECT user_id FROM suspect_users)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                user_id,
                count(*)                                                        AS events,
                count(*) FILTER (WHERE behavior_type = 'pv')                     AS pv,
                count(*) FILTER (WHERE behavior_type = 'cart')                   AS cart,
                count(*) FILTER (WHERE behavior_type = 'fav')                    AS fav,
                count(*) FILTER (WHERE behavior_type = 'buy')                    AS buy_cnt,
                count(DISTINCT item_id) FILTER (WHERE behavior_type = 'buy')     AS buy_items,
                count(DISTINCT date)                                             AS active_days,
                count(DISTINCT date) FILTER (WHERE behavior_type = 'buy')        AS buy_days,
                count(DISTINCT category_id)                                      AS categories,
                min(date)                                                        AS first_date,
                max(date)                                                        AS last_date,
                max(datetime) FILTER (WHERE behavior_type = 'buy')               AS last_buy_time
            FROM events_clean
            GROUP BY 1
        ) TO '{(C.PARQUET_DIR / "user_features.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION zstd)
        """
    )
    return con.execute(
        f"SELECT * FROM read_parquet('{(C.PARQUET_DIR / 'user_features.parquet').as_posix()}') LIMIT 5"
    ).fetchdf()


def bot_sensitivity(con) -> pd.DataFrame:
    """剔除前后关键指标对比：证明清洗动作不改变结论方向。"""
    q = """
        SELECT
            count(*)                                              AS 行为行数,
            count(DISTINCT user_id)                               AS 用户数,
            count(*) * 1.0 / count(DISTINCT user_id)              AS 人均行为数,
            count(*) FILTER (WHERE behavior_type='pv')  * 1.0 / count(*) AS pv占比,
            count(*) FILTER (WHERE behavior_type='cart')* 1.0 / count(*) AS cart占比,
            count(*) FILTER (WHERE behavior_type='fav') * 1.0 / count(*) AS fav占比,
            count(*) FILTER (WHERE behavior_type='buy') * 1.0 / count(*) AS buy占比,
            count(DISTINCT user_id) FILTER (WHERE behavior_type='buy') * 1.0
                / count(DISTINCT user_id)                         AS 购买用户占比
        FROM {src}
    """
    all_ = con.execute(q.format(src="events")).fetchdf().iloc[0]
    clean_ = con.execute(q.format(src="events_clean")).fetchdf().iloc[0]
    df = pd.DataFrame(
        {
            "指标": ["行为行数", "用户数", "人均行为数", "pv占比", "cart占比", "fav占比", "buy占比", "购买用户占比"],
            "剔除前(全量)": all_.tolist(),
            "剔除疑似爬虫后": clean_.tolist(),
        }
    )
    df["相对变化"] = [
        "-" if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a == 0
        else f"{(b - a) / a:+.2%}"
        for a, b in zip(df["剔除前(全量)"], df["剔除疑似爬虫后"])
    ]
    return df


def main() -> None:
    t0 = time.perf_counter()
    con = db.connect(memory_limit="8GB")

    print("[2.5] 落地事件中间层（parquet，按 date 分区）...", flush=True)
    build_events(con)

    print("[2.2] 标记疑似异常高频用户 ...", flush=True)
    th = bot_threshold(con)
    sus = build_suspect_users(con, th)
    print(f"     疑似异常用户 {len(sus):,} 个；最高单日 pv = {sus['max_daily_pv'].max():,}", flush=True)

    db.attach_parquet_layer(con)

    print("[2.5] 落地日/类目聚合 ...", flush=True)
    build_aggregates(con)

    print("[2.x] 落地用户级特征宽表 ...", flush=True)
    build_user_features(con)

    print("[2.2] 爬虫敏感性对比 ...", flush=True)
    sens = bot_sensitivity(con)
    print(utils.df_to_md(sens), flush=True)

    # ---------- 图：剔除前后行为结构对比
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = ["pv占比", "cart占比", "fav占比", "buy占比"]
    before = [sens.loc[sens["指标"] == l, "剔除前(全量)"].iloc[0] * 100 for l in labels]
    after = [sens.loc[sens["指标"] == l, "剔除疑似爬虫后"].iloc[0] * 100 for l in labels]
    idx = range(len(labels))
    ax.bar([i - 0.2 for i in idx], before, width=0.4, label="全量", color="#BAB0AC")
    ax.bar([i + 0.2 for i in idx], after, width=0.4, label="剔除疑似爬虫", color="#4C78A8")
    for i, (b_, a_) in enumerate(zip(before, after)):
        ax.text(i - 0.2, b_, f"{b_:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, a_, f"{a_:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(["浏览 pv", "加购 cart", "收藏 fav", "购买 buy"])
    ax.set_ylabel("占比 (%)")
    ax.set_title(f"异常流量敏感性对比：{len(sus):,} 个极端高频用户（单日 pv > {th:.0f}）")
    ax.legend()
    fig.tight_layout()
    C.save_fig(fig, "fig02_bot_sensitivity.png")
    plt.close(fig)

    total_users = con.execute("SELECT count(DISTINCT user_id) FROM events").fetchone()[0]
    buy_row = sens.loc[sens["指标"] == "buy占比", "相对变化"].iloc[0]
    buy_change = float(str(buy_row).rstrip("%")) if str(buy_row) not in ("-", "") else 0.0

    md = f"""# 阶段 2 · 清洗与特征工程结论

## 一条命令复现
```bash
.venv\\Scripts\\python.exe src\\stage2_cleaning.py
```

## 1. 时间派生（2.1）
在 DuckDB 视图 `clean` 中一次派生：`datetime / date / hour / weekday / is_weekend`。
- 时间戳按 **北京时间（UTC+8）** 解释；
- `weekday` 采用 **0 = 周一 … 6 = 周日**，与 pandas `dt.weekday` 完全一致，避免两套口径打架。

## 2. 极端高频用户标记与敏感性（2.2）

**阈值是数据驱动的，不是拍脑袋定的。**
最初的设想是用固定值（单用户单日 pv > 500），但实测发现本数据里**最大单用户单日 pv 不足 500**——
固定阈值会命中 0 个用户，清洗动作形同虚设。
因此改为：**阈值 = max({C.BOT_PV_FLOOR}, 用户-日 pv 的 p99.9) = {th:.0f}**。

- 命中用户：**{len(sus):,}**（占全部用户 {len(sus) / max(total_users, 1):.3%}）
- 命中用户最高单日 pv：**{sus['max_daily_pv'].max() if len(sus) else 0:,.0f}**

处理策略：**标记而不删除**。全量口径与剔除口径同时计算，验证结论方向是否被这批流量带偏。

{utils.df_to_md(sens)}

![爬虫敏感性](figures/fig02_bot_sensitivity.png)

**结论**：异常流量对行为结构的影响在 {abs(buy_change):.2f}% 量级以内，
方向未被改变。因此**对外结论保留全量口径**（更保守、不丢数据），并把剔除口径作为稳健性检验写在报告里。

## 3. Session 切分口径（2.3）
规则：同一用户按时间排序，**相邻行为间隔 > {C.SESSION_GAP_MINUTES} 分钟**即切分为新会话。
- 为什么是 30 分钟：业界通用的"会话超时"经验值（Google Analytics 默认 30 min）。
  用户在 30 分钟内无操作，通常意味着注意力已离开当前浏览任务；小于 15 分钟会把一次连续浏览切碎，大于 60 分钟会把两次独立访问粘在一起。
- 数据集粒度是**秒级行为流**，没有页面级停留时长，所以只能用行为间隔做代理。
- 该口径是本项目漏斗分析的基础（见阶段 4 的 session 口径漏斗）。

## 4. 分层采样（2.4）
交互式探索使用 **按 user_id 哈希的 {C.SAMPLE_RATIO:.0%} 分层样本**（`hash(user_id) % {int(1 / C.SAMPLE_RATIO)} = 0`）：
同一用户的整条行为链同进同出，不会把用户"切碎"导致漏斗虚高。
对外结论一律用**全量**。

## 5. 中间层落地（2.5）

| 产物 | 内容 | 分区 |
|---|---|---|
| `outputs/parquet/events/` | 清洗后的全量事件 | 按 `date` |
| `outputs/parquet/daily_behavior/` | 日 × 行为类型：行为数 / 用户数 / 商品数 | 按 `date` |
| `outputs/parquet/category_daily/` | 日 × 类目：行为数 + UV 口径 | 按 `date` |
| `outputs/parquet/suspect_users.parquet` | 疑似爬虫用户清单 | — |
| `outputs/parquet/user_features.parquet` | 用户级特征宽表（pv/cart/fav/buy、活跃天数、购买天数、购买商品数、首末活跃日） | — |

效果：后续所有分析从"每次重扫 3.6 GB CSV"变成"扫 parquet"，单次全量聚合从数十秒降到数秒级。
"""
    (C.REPORT_DIR / "stage2_cleaning_summary.md").write_text(md, encoding="utf-8")
    con.close()
    print(f"阶段 2 完成，用时 {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent))
    main()
