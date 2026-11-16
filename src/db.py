# -*- coding: utf-8 -*-
"""
DuckDB 连接与视图定义。

设计要点
--------
1. 原始 CSV 3.6 GB 原地只读，不复制、不改名；DuckDB 直接对它跑 SQL。
2. 统一两个视图：
   - `raw`   原始五列，不做任何清洗
   - `clean` 派生 datetime / date / hour / weekday 字段（字段口径与 pandas 对齐）
3. 时间戳统一换算为北京时间：数据集为淘宝站内日志，小时峰值应落在北京时间晚间，
   因此把 epoch 秒直接加 8 小时得到朴素时间戳（不引入 pytz / 会话时区依赖）。
"""
from __future__ import annotations

from typing import Optional

import duckdb

from config import COLUMNS, PARQUET_DIR, RAW_CSV, SESSION_GAP_MINUTES

_NAMES_LITERAL = "[" + ",".join(f"'{c}'" for c in COLUMNS) + "]"
_TYPES_LITERAL = (
    "{'user_id':'BIGINT','item_id':'BIGINT','category_id':'BIGINT',"
    "'behavior_type':'VARCHAR','timestamp':'BIGINT'}"
)

# epoch 秒 -> 北京时间（UTC+8）的"朴素"时间戳。
# 用 epoch_ms 而不是 to_timestamp：to_timestamp 返回带时区的 TIMESTAMPTZ，
# 转成 Python 对象时需要 pytz，且依赖会话时区；直接加 8 小时偏移更确定、更快、无额外依赖。
TZ_OFFSET_SECONDS = 8 * 3600
TS_EXPR = f"epoch_ms((timestamp + {TZ_OFFSET_SECONDS}) * 1000)"
_TS = TS_EXPR


def csv_path() -> str:
    return str(RAW_CSV).replace("\\", "/")


def connect(memory_limit: Optional[str] = None, threads: Optional[int] = None) -> duckdb.DuckDBPyConnection:
    """建立一个内存 DuckDB 连接，并注册 raw / clean 视图。"""
    con = duckdb.connect()
    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    if threads:
        con.execute(f"SET threads={threads}")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW raw AS
        SELECT * FROM read_csv(
            '{csv_path()}',
            header = false,
            names = {_NAMES_LITERAL},
            types = {_TYPES_LITERAL},
            sample_size = 200000
        )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW clean AS
        SELECT
            user_id,
            item_id,
            category_id,
            behavior_type,
            timestamp,
            {_TS}                                        AS datetime,
            CAST({_TS} AS DATE)                          AS date,
            CAST(EXTRACT(hour FROM {_TS}) AS INTEGER)    AS hour,
            CAST(EXTRACT(isodow FROM {_TS}) - 1 AS INTEGER) AS weekday,  -- 0=周一，与 pandas 一致
            CASE WHEN EXTRACT(isodow FROM {_TS}) >= 6 THEN 1 ELSE 0 END AS is_weekend
        FROM raw
        """
    )
    return con


def sample_view(con: duckdb.DuckDBPyConnection, ratio: float) -> None:
    """
    注册 `sample` 视图：按 user_id 哈希分层抽样。
    同一 user_id 的整条行为链要么全进、要么全不进，避免把用户"切碎"。
    """
    mod = max(int(round(1 / ratio)), 1)
    con.execute(
        f"CREATE OR REPLACE VIEW sample AS SELECT * FROM clean WHERE hash(user_id) % {mod} = 0"
    )


def attach_parquet_layer(con: duckdb.DuckDBPyConnection) -> None:
    """
    注册中间层视图（阶段 2 落地的 parquet）：
      events        —— 清洗后的全量事件，按 date 分区
      suspect_users —— 疑似爬虫用户
    后续所有分析只扫 parquet，不再重复解析 3.6 GB CSV。
    """
    events_glob = str(PARQUET_DIR / "events" / "**" / "*.parquet").replace("\\", "/")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW events AS
        SELECT * FROM read_parquet('{events_glob}', hive_partitioning = true)
        """
    )
    sus = PARQUET_DIR / "suspect_users.parquet"
    if sus.exists():
        con.execute(
            f"CREATE OR REPLACE VIEW suspect_users AS SELECT * FROM read_parquet('{sus.as_posix()}')"
        )


def sessionized_sql(source: str = "events", gap_minutes: int = SESSION_GAP_MINUTES) -> str:
    """
    会话切分：同一用户按时间排序，相邻行为间隔 > gap_minutes 则 session_id + 1。
    session_key = user_id * 10^6 + 会话序号
    """
    return f"""
    WITH ordered AS (
        SELECT
            *,
            CASE
                WHEN epoch(datetime) - lag(epoch(datetime)) OVER (PARTITION BY user_id ORDER BY datetime)
                     > {gap_minutes} * 60
                     OR lag(epoch(datetime)) OVER (PARTITION BY user_id ORDER BY datetime) IS NULL
                THEN 1 ELSE 0
            END AS is_new_session
        FROM {source}
    ),
    flagged AS (
        SELECT
            *,
            SUM(is_new_session) OVER (
                PARTITION BY user_id ORDER BY datetime
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS session_seq
        FROM ordered
    )
    SELECT
        *,
        user_id * 1000000 + session_seq AS session_key
    FROM flagged
    """
