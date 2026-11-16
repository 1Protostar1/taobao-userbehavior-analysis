-- ============================================================
-- 01 · DAU 趋势
-- 用途：看出 9 天窗口内的活跃规模与波动，为后续留存/漏斗提供分母
-- 依赖视图：events
-- ============================================================
SELECT
    CAST(date AS VARCHAR)                    AS 日期,
    count(DISTINCT user_id)                  AS DAU,
    count(*)                                 AS 行为量,
    round(count(*) * 1.0 / count(DISTINCT user_id), 2) AS 人均行为数
FROM events
GROUP BY 1
ORDER BY 1;
