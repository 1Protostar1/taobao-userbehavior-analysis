-- ============================================================
-- 04 · 漏斗（Session 口径，主用）
-- 会话定义：同一用户相邻行为间隔 > 30 分钟切分为新会话
-- 相比 UV 口径，这里要求"浏览 → 加购/收藏 → 购买"发生在【同一次会话内且顺序正确】，
-- 更接近"这一次访问有没有促成成交"的业务问题
-- 依赖视图：events
-- ============================================================
WITH ordered AS (
    SELECT
        *,
        CASE
            WHEN lag(epoch(datetime)) OVER (PARTITION BY user_id ORDER BY datetime) IS NULL
              OR epoch(datetime) - lag(epoch(datetime)) OVER (
                     PARTITION BY user_id ORDER BY datetime) > 30 * 60
            THEN 1 ELSE 0
        END AS is_new_session
    FROM events
),
flagged AS (
    SELECT *,
           sum(is_new_session) OVER (
               PARTITION BY user_id ORDER BY datetime
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_seq
    FROM ordered
),
sess AS (
    SELECT
        user_id * 1000000 + session_seq                     AS session_key,
        min(datetime) FILTER (WHERE behavior_type = 'pv')   AS t_pv,
        min(datetime) FILTER (WHERE behavior_type = 'cart') AS t_cart,
        min(datetime) FILTER (WHERE behavior_type = 'fav')  AS t_fav,
        min(datetime) FILTER (WHERE behavior_type = 'buy')  AS t_buy
    FROM flagged
    GROUP BY 1
)
SELECT '① 浏览会话'                AS 环节,
       count(*) FILTER (WHERE t_pv IS NOT NULL)                                       AS 会话数
FROM sess
UNION ALL
SELECT '② 浏览后加购或收藏会话',
       count(*) FILTER (WHERE t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv))   FROM sess
UNION ALL
SELECT '③ 意向后再购买会话',
       count(*) FILTER (WHERE t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv)
                          AND t_buy > least(coalesce(t_cart, t_fav), coalesce(t_fav, t_cart))) FROM sess
UNION ALL
SELECT '参考·直接购买会话（跳过加购/收藏）',
       count(*) FILTER (WHERE t_buy > t_pv
                          AND (t_cart IS NULL OR t_cart < t_pv)
                          AND (t_fav IS NULL OR t_fav < t_pv)) FROM sess;
