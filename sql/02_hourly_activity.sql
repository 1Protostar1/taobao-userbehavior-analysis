-- ============================================================
-- 02 · 小时活跃分布
-- 用途：定位资源投放时段（本项目峰值落在晚间 20:00–22:00）
-- 注意：timestamp 按北京时间（UTC+8）解释（epoch 秒 + 8 小时）
-- 依赖视图：events
-- ============================================================
SELECT
    hour                                            AS 小时,
    count(*)                                        AS 行为量,
    count(DISTINCT user_id)                         AS 活跃用户数,
    round(count(*) * 1.0 / sum(count(*)) OVER (), 4) AS 时段行为占比
FROM events
GROUP BY 1
ORDER BY 1;
