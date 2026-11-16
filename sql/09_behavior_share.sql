-- ============================================================
-- 09 · 行为类型构成
-- 用途：确认行为分布形态（pv 占比 > 89%），并给出各行为的用户/商品覆盖
-- 依赖视图：events
-- ============================================================
SELECT
    behavior_type                            AS 行为类型,
    count(*)                                 AS 行为数,
    round(count(*) * 1.0 / sum(count(*)) OVER (), 4) AS 行为占比,
    count(DISTINCT user_id)                  AS 覆盖用户数,
    count(DISTINCT item_id)                  AS 覆盖商品数
FROM events
GROUP BY 1
ORDER BY 行为数 DESC;
