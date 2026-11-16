-- ============================================================
-- 08 · Top 类目：浏览量 vs 转化率（UV 口径）
-- 目的：识别"高流量低转化"类目，为流量重排提供依据
-- 依赖视图：events
-- ============================================================
WITH cand AS (
    SELECT category_id, count(*) AS pv
    FROM events WHERE behavior_type = 'pv'
    GROUP BY 1 ORDER BY 2 DESC LIMIT 500
),
detail AS (
    SELECT
        category_id,
        count(DISTINCT user_id) FILTER (WHERE behavior_type = 'pv')   AS pv_uv,
        count(DISTINCT user_id) FILTER (WHERE behavior_type = 'cart') AS cart_uv,
        count(DISTINCT user_id) FILTER (WHERE behavior_type = 'fav')  AS fav_uv,
        count(DISTINCT user_id) FILTER (WHERE behavior_type = 'buy')  AS buy_uv
    FROM events
    WHERE category_id IN (SELECT category_id FROM cand)
    GROUP BY 1
)
SELECT
    c.category_id           AS 类目,
    c.pv                    AS 浏览量,
    d.pv_uv                 AS 浏览用户数,
    d.buy_uv                AS 购买用户数,
    round(d.buy_uv * 1.0 / nullif(d.pv_uv, 0), 4)                    AS "浏览→购买转化率",
    round((d.cart_uv + d.fav_uv) * 1.0 / nullif(d.pv_uv, 0), 4)      AS "加购或收藏率"
FROM cand c
JOIN detail d USING (category_id)
ORDER BY c.pv DESC
LIMIT 20;
