-- ============================================================
-- 06 · 留存（次日 / 7 日）
-- 分母口径：以"用户首次活跃日"为基准（首日活跃用户），只看窗口内可以被观测到的留存
-- 窗口局限：数据集仅 9 天，11-25 之后的日期才有完整的次日留存
--            "7日内回访" 只有 11-25 / 11-26 两个 cohort 是可完整观测的，其余 cohort 被窗口右截断
-- 依赖视图：events
-- ============================================================
WITH first_day AS (
    SELECT user_id, min(date) AS fd FROM events GROUP BY user_id
),
pairs AS (
    SELECT f.user_id, f.fd, e.date AS d,
           date_diff('day', f.fd, e.date) AS day_n
    FROM events e
    JOIN first_day f USING (user_id)
),
base AS (
    SELECT fd AS 首日,
           count(DISTINCT user_id) AS 新增用户数
    FROM first_day
    WHERE fd < DATE '2017-12-03'
    GROUP BY 1
),
ret AS (
    SELECT fd AS 首日,
           count(DISTINCT user_id) FILTER (WHERE day_n = 1) AS 次日留存,
           count(DISTINCT user_id) FILTER (WHERE day_n BETWEEN 1 AND 7) AS "7日内回访"
    FROM pairs
    WHERE fd < DATE '2017-12-03'
    GROUP BY 1
)
SELECT CAST(b.首日 AS VARCHAR) AS 首日,
       b.新增用户数,
       r.次日留存,
       round(r.次日留存 * 1.0 / b.新增用户数, 4)      AS 次日留存率,
       r."7日内回访",
       round(r."7日内回访" * 1.0 / b.新增用户数, 4)   AS "7日内回访率"
FROM base b JOIN ret r USING (首日)
ORDER BY 1;
