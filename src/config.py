# -*- coding: utf-8 -*-
"""
全局配置：路径、口径常量、绘图规范。
所有脚本统一从这里取参数，避免路径/口径散落在各处。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始数据不做任何移动（3.6 GB），只在这里指向它
RAW_CSV = Path(r"C:\Users\zhao\Desktop\数据分析\UserBehavior.csv")

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PARQUET_DIR = OUTPUT_DIR / "parquet"          # 中间层：聚合结果，按 date 分区
SQL_DIR = PROJECT_ROOT / "sql"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

for _d in (DATA_DIR, OUTPUT_DIR, PARQUET_DIR, SQL_DIR, REPORT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 数据 schema
COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]

# ---------------------------------------------------------------- 口径常量
# 有效分析窗口（北京时间，含首尾）。窗口之外的行按"数据错误/采集爬坡"处理，不进入中间层。
# 实测：窗口内 100,095,231 行；窗口外 55,576 行（0.055%），
#       其中 2017-11-24 全天 40,231 行是正式采集前的爬坡数据，另有 15,345 行时间戳明显异常
#       （最早到 1902 年、最晚到 2037 年）。详见 reports/data_quality_report.md
DATE_START = "2017-11-25"
DATE_END = "2017-12-03"
WINDOW_SQL = (
    f"datetime >= TIMESTAMP '{DATE_START} 00:00:00' "
    f"AND datetime <= TIMESTAMP '{DATE_END} 23:59:59'"
)

# 分层采样比例：用于快速迭代（按 user_id 哈希，保证同一用户整体进入样本）
SAMPLE_RATIO = 0.10

# session 切分阈值：同一用户相邻行为间隔超过该分钟数则切分为新会话
SESSION_GAP_MINUTES = 30

# 疑似爬虫阈值：单用户单日 pv 次数上限。
# 实测数据中单用户单日 pv 最大值不足 500，说明这批日志里**没有**极端刷量用户，
# 因此阈值改为"数据驱动"： max(BOT_PV_FLOOR, 用户-日 pv 的 p99.9)，见 src/stage2_cleaning.py。
BOT_PV_FLOOR = 100

# 漏斗口径：加购与收藏为并列分支，不是先后关系
FUNNEL_TOP = "pv"
FUNNEL_TIERS = ["cart", "fav"]   # 并列分支
FUNNEL_BOTTOM = "buy"

BEHAVIOR_CN = {
    "pv": "浏览",
    "cart": "加购",
    "fav": "收藏",
    "buy": "购买",
}

# ---------------------------------------------------------------- 绘图规范
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D", "#BAB0AC"]
FIG_DPI = 150


def setup_chinese_font() -> None:
    """统一设置 matplotlib 中文字体与负号显示。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = FIG_DPI
    plt.rcParams["savefig.dpi"] = FIG_DPI
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["figure.facecolor"] = "white"


def save_fig(fig, name: str) -> str:
    """保存图片到 reports/figures/ 并返回路径。"""
    path = FIG_DIR / name
    fig.savefig(path)
    return str(path)
