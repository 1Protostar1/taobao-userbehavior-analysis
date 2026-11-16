# data/

**本项目不在本目录存放原始数据。**

原始数据 `UserBehavior.csv`（3.64 GB / 约 1 亿行）保存在：

```
C:\Users\zhao\Desktop\数据分析\UserBehavior.csv
```

之所以不移动它：

1. 3.6 GB 的大文件移动/复制成本高、容易中断，且会污染版本库（已加入 `.gitignore`）；
2. 统一通过 `src/config.py` 里的常量 `RAW_CSV` 指向它，路径只有一处，换机器只改一行；
3. DuckDB 可以直接对 CSV 原地跑 SQL，不需要先"导入数据库"。

## 字段说明（文件无表头）

| 列序 | 字段 | 类型 | 说明 |
|---|---|---|---|
| 1 | user_id | int | 用户 ID |
| 2 | item_id | int | 商品 ID |
| 3 | category_id | int | 商品类目 ID |
| 4 | behavior_type | string | 行为类型：`pv` 浏览 / `cart` 加购 / `fav` 收藏 / `buy` 购买 |
| 5 | timestamp | int | Unix 时间戳（秒），按北京时间（UTC+8）解释 |

读取方式（所有脚本统一走 `src/db.py`）：

```python
read_csv('UserBehavior.csv', header=false,
         names=['user_id','item_id','category_id','behavior_type','timestamp'])
```

## 中间层产物（可随时重建，不入版本库）

清洗与聚合结果落在 `outputs/parquet/`：

| 路径 | 内容 |
|---|---|
| `events/date=*/` | 清洗后全量事件，按 `date` 分区 |
| `daily_behavior/date=*/` | 日 × 行为类型聚合 |
| `category_daily/date=*/` | 日 × 类目聚合（含 UV 口径） |
| `sessions.parquet` | 会话级聚合表 |
| `user_features.parquet` | 用户级特征宽表 |
| `rfm_user.parquet` | 用户级 RFM 打分与分层 |
| `suspect_users.parquet` | 疑似爬虫用户清单 |
