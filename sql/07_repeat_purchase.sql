-- ============================================================
-- 07 · 复购率与购买间隔
-- 复购定义：窗口内购买天数 >= 2 天（用"购买天数"而不是"购买次数"，避免一单多件误判）
-- 依赖视图：events
-- ============================================================
WITH bu AS (
    SELECT
        user_id,
        count(*)                                   AS buy_cnt,
        count(DISTINCT date)                       AS buy_days,
        date_diff('day', min(CAST(datetime AS DATE)), max(CAST(datetime AS DATE))) AS span_days
    FROM events
    WHERE behavior_type = 'buy'
    GROUP BY user_id
)
SELECT
    count(*)                                                        AS 购买用户数,
    count(*) FILTER (WHERE buy_days >= 2)                            AS 复购用户数,
    round(count(*) FILTER (WHERE buy_days >= 2) * 1.0 / count(*), 4)  AS 复购率,
    round(avg(buy_cnt), 2)                                           AS 人均购买次数,
    round(avg(buy_days), 2)                                          AS 人均购买天数,
    round(avg(span_days) FILTER (WHERE buy_days >= 2), 2)             AS 复购用户平均购买跨度天
FROM bu;
