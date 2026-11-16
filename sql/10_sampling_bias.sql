-- ============================================================
-- 10 · 采样偏差对照（把"取前 N 行"的坑写进 SQL，随时可复现）
-- A 组：前 100 万行（原始做法）—— 文件按 user_id 排序，等价于"只取 ID 最小的一小撮用户"
-- B 组：按 user_id 哈希的 1% 分层样本 —— 同一用户整链同进同出，不做切碎
-- 依赖视图：raw（原始 CSV 五列，未做任何清洗）
-- ============================================================
WITH a AS (
    SELECT * FROM raw LIMIT 1000000
),
b AS (
    SELECT * FROM raw WHERE hash(user_id) % 100 = 0
),
stat AS (
    SELECT 'A 前100万行' AS 样本,
           count(*)                                       AS 行数,
           count(DISTINCT user_id)                        AS 唯一用户数,
           min(user_id)                                   AS user_id最小,
           max(user_id)                                   AS user_id最大,
           count(DISTINCT CAST(epoch_ms((timestamp + 28800) * 1000) AS DATE)) AS 覆盖天数,
           round(count(*) * 1.0 / count(DISTINCT user_id), 1)    AS 人均行为数,
           round(count(*) FILTER (WHERE behavior_type = 'buy') * 1.0 / count(*), 4) AS 购买占比,
           round(count(*) FILTER (WHERE behavior_type = 'cart') * 1.0 / count(*), 4) AS 加购占比
    FROM a
    UNION ALL
    SELECT 'B 分层采样1%',
           count(*), count(DISTINCT user_id), min(user_id), max(user_id),
           count(DISTINCT CAST(epoch_ms((timestamp + 28800) * 1000) AS DATE)),
           round(count(*) * 1.0 / count(DISTINCT user_id), 1),
           round(count(*) FILTER (WHERE behavior_type = 'buy') * 1.0 / count(*), 4),
           round(count(*) FILTER (WHERE behavior_type = 'cart') * 1.0 / count(*), 4)
    FROM b
)
SELECT * FROM stat ORDER BY 样本;
