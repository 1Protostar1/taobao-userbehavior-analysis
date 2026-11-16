-- ============================================================
-- 05 · RFM 用户价值分层
-- R = 末次购买距数据集末端天数
-- F = 购买【天数】（比购买次数抗刷单）
-- M = 购买的不同商品件数（⚠️ 无金额字段，M 为代理指标）
-- 打分：五分位（NTILE(5)），R 反向、F/M 正向
-- 依赖视图：events
-- ============================================================
WITH base AS (
    SELECT
        user_id,
        date_diff('day', CAST(max(datetime) FILTER (WHERE behavior_type = 'buy') AS DATE),
                  DATE '2017-12-03')                                       AS r_days,
        count(DISTINCT date) FILTER (WHERE behavior_type = 'buy')          AS f_days,
        count(DISTINCT item_id) FILTER (WHERE behavior_type = 'buy')       AS m_items,
        count(*) FILTER (WHERE behavior_type = 'buy')                      AS buy_cnt
    FROM events
    GROUP BY user_id
),
scored AS (
    SELECT
        *,
        6 - ntile(5) OVER (ORDER BY r_days ASC)  AS r_score,   -- R 越小越好 → 反向
        ntile(5) OVER (ORDER BY f_days ASC)      AS f_score,
        ntile(5) OVER (ORDER BY m_items ASC)     AS m_score
    FROM base
    WHERE buy_cnt > 0
),
tagged AS (
    SELECT
        *,
        CASE WHEN r_score >= 4 THEN 1 ELSE 0 END AS r_high,
        CASE WHEN f_score >= 4 THEN 1 ELSE 0 END AS f_high,
        CASE WHEN m_score >= 4 THEN 1 ELSE 0 END AS m_high
    FROM scored
)
SELECT
    CASE concat(r_high, f_high, m_high)
        WHEN '111' THEN '重要价值客户' WHEN '101' THEN '重要发展客户'
        WHEN '011' THEN '重要保持客户' WHEN '001' THEN '重要挽留客户'
        WHEN '110' THEN '一般价值客户' WHEN '100' THEN '一般发展客户'
        WHEN '010' THEN '一般保持客户' ELSE '一般挽留客户'
    END                                                     AS 客户分层,
    count(*)                                                AS 用户数,
    round(count(*) * 1.0 / sum(count(*)) OVER (), 4)         AS 人数占比,
    round(sum(buy_cnt) * 1.0 / sum(sum(buy_cnt)) OVER (), 4) AS 购买贡献占比,
    round(avg(buy_cnt), 2)                                   AS 人均购买次数,
    round(avg(f_days), 2)                                    AS 人均购买天数,
    round(avg(m_items), 2)                                   AS 人均购买商品数,
    round(avg(r_days), 2)                                    AS 平均R
FROM tagged
GROUP BY 1
ORDER BY 购买贡献占比 DESC;
