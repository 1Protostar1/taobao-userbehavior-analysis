# 阶段 8 · SQL 化结果

所有指标均已用 SQL 表达并归档到 `sql/`，可直接用 DuckDB 对 **全量 3.6 GB CSV** 执行：
`events` 视图指向中间层 parquet，`raw` 视图指向原始 CSV，二者由 `src/db.py` 注册。

| SQL 文件 | 说明 | 结果行数 | 耗时(秒) | 状态 |
|---|---|---|---|---|
| 01_dau.sql | DAU 趋势：逐日活跃用户数、行为量、人均行为 | 9 | 2.343 | OK |
| 02_hourly_activity.sql | 小时活跃分布：定位晚间资源投放窗口 | 24 | 1.753 | OK |
| 03_funnel_uv.sql | 漏斗（UV 口径）：pv → (cart|fav) → buy，含顺序约束 | 4 | 2.444 | OK |
| 04_funnel_session.sql | 漏斗（Session 口径）：30 分钟会话内且顺序正确 | 4 | 107.5 | OK |
| 05_rfm.sql | RFM 分层：五分位打分 + 8 类客户 + 购买贡献度 | 8 | 6.777 | OK |
| 06_retention.sql | 留存：次日留存与 7 日内回访（含窗口截断说明） | 8 | 4.234 | OK |
| 07_repeat_purchase.sql | 复购率与购买跨度：复购定义 = 购买天数 ≥ 2 | 1 | 2.597 | OK |
| 08_top_category.sql | Top20 类目：浏览量 vs 转化率（识别高流量低转化） | 20 | 9.998 | OK |
| 09_behavior_share.sql | 行为类型构成与用户/商品覆盖 | 4 | 22.19 | OK |
| 10_sampling_bias.sql | 采样偏差对照：前 100 万行 vs 1% 分层采样 | 2 | 10.6 | OK |

## 怎么跑

```bash
.venv\Scripts\python.exe src\stage8_sql.py
```

## 为什么用 DuckDB

- **零部署**：不需要起 Hive/Spark 集群，一个进程直接对 CSV 跑 SQL；
- **列式 + 向量化 + 多线程**：1 亿行聚合在秒级到十秒级完成；
- **可迁移**：同样的 SQL 思路可平移到 Hive/Spark，也避免了 pandas 全量读入的内存瓶颈。
