-- ============================================================
-- 03 · 漏斗（UV 口径）
-- 口径修正：pv → (cart | fav) → buy
--   加购与收藏是【并列分支】，不是先后关系，因此第二步取"有任一一支"
-- 顺序约束：要求 加购/收藏 发生在 浏览 之后，购买 发生在 加购/收藏 之后
-- 依赖视图：events
-- ============================================================
WITH u AS (
    SELECT
        user_id,
        min(datetime) FILTER (WHERE behavior_type = 'pv')   AS t_pv,
        min(datetime) FILTER (WHERE behavior_type = 'cart') AS t_cart,
        min(datetime) FILTER (WHERE behavior_type = 'fav')  AS t_fav,
        min(datetime) FILTER (WHERE behavior_type = 'buy')  AS t_buy
    FROM events
    GROUP BY user_id
)
SELECT '① 浏览'               AS 环节,
       count(*) FILTER (WHERE t_pv IS NOT NULL)                                     AS 用户数
FROM u
UNION ALL
SELECT '② 浏览后加购或收藏',
       count(*) FILTER (WHERE t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv)) FROM u
UNION ALL
SELECT '③ 意向后再购买',
       count(*) FILTER (WHERE t_pv IS NOT NULL AND (t_cart > t_pv OR t_fav > t_pv)
                          AND t_buy > least(coalesce(t_cart, t_fav), coalesce(t_fav, t_cart))) FROM u
UNION ALL
-- 参考：不受顺序约束的"任意购买用户"，用于说明口径差异
SELECT '参考·任意购买用户', count(*) FILTER (WHERE t_buy IS NOT NULL) FROM u;
