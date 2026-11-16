# -*- coding: utf-8 -*-
"""
生成 notebooks/01~06 的骨架 notebook。
notebook 是"分析叙事层"，每一步都调用 src/ 下的模块（代码逻辑只有一份，不会两处漂移）。

运行：.venv\\Scripts\\python.exe src\\make_notebooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402

SETUP = """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))
import config as C
import db
C.setup_chinese_font()
%matplotlib inline"""

NOTEBOOKS = [
    (
        "01_overview",
        "阶段 1 · 数据探查与质量校验",
        "把数据说清楚：1 亿行、9 天窗口、字段质量、以及**采样偏差**这个最大的坑。",
        [
            ("## 1.1 读取方式基准测试\n\n对比 pandas chunksize / pyarrow 流式 / DuckDB 全量扫描，选定主链路。",
             "import stage1_overview as s1\nbench = s1.bench_read_methods()\nbench"),
            ("## 1.2 全量基础指标（DuckDB 直连 3.6 GB CSV）",
             "con = db.connect()\nmetrics = s1.basic_metrics(con)\nmetrics"),
            ("## 1.2' 逐日行数 / 行为构成",
             "daily = s1.rows_per_day(con)\nbehav = s1.behavior_profile(con)\nbehav"),
            ("## 1.3 质量检查：缺失、重复、越界、取值集合、异常高频用户",
             "quality = s1.quality_checks(con)\nquality"),
            ("## 1.4 分布检查：用户活跃度长尾与购买集中度",
             "dist = s1.activity_distribution(con)\ndist['buckets']"),
            ("## 1.5 采样偏差实证：前 100 万行 vs 分层采样 1%\n\n**这是本项目最关键的一次自检。**",
             "cmp_df, _ = s1.sampling_bias(con)\ncmp_df"),
            ("## 1.6 产出报告",
             "s1.build_report(bench, metrics, daily, behav, quality, dist, cmp_df)\ncon.close()\n# -> reports/data_quality_report.md"),
        ],
    ),
    (
        "02_cleaning",
        "阶段 2 · 数据清洗与特征工程",
        "时间派生 → 异常标记 → session 切分 → 分层采样 → parquet 中间层落地。",
        [
            ("## 2.1 时间派生\n\n在 DuckDB 的 `clean` 视图里一次派生 `datetime/date/hour/weekday/is_weekend`，\n"
             "时间戳按**北京时间**解释，`weekday` 与 pandas 口径一致（0=周一）。",
             "con = db.connect(memory_limit='8GB')\ncon.execute('SELECT * FROM clean LIMIT 5').fetchdf()"),
            ("## 2.2 异常流量标记：单用户单日 pv > 500 → 疑似爬虫（标记而不删除）",
             "import stage2_cleaning as s2\nsus = s2.build_suspect_users(con)\nprint('疑似爬虫用户数:', len(sus))\nsus.head()"),
            ("## 2.3 Session 切分口径\n\n间隔 > 30 分钟切分新会话。查看切分 SQL：",
             "print(db.sessionized_sql('events', 30))"),
            ("## 2.5 中间层落地：按 date 分区的 parquet（后续不再重扫 3.6 GB）",
             "s2.build_events(con)\ndb.attach_parquet_layer(con)\ns2.build_aggregates(con)\ns2.build_user_features(con)"),
            ("## 2.2' 剔除前后敏感性对比",
             "sens = s2.bot_sensitivity(con)\nsens"),
        ],
    ),
    (
        "03_descriptive",
        "阶段 3 · 描述性分析与可视化",
        "DAU / 小时 / 周内 / 行为构成 / Top 类目流量-转化 / 用户活跃长尾。",
        [
            ("## 3.1 DAU 趋势",
             "import stage3_descriptive as s3\ndau = s3.fig_dau(con)\ndau"),
            ("## 3.2 小时活跃分布（定位 20:00–22:00 高峰）",
             "hourly = s3.fig_hourly(con)\nhourly.head(24)"),
            ("## 3.3 周内活跃分布",
             "s3.fig_weekday(con)"),
            ("## 3.4 行为类型构成",
             "s3.fig_behavior_share(con)"),
            ("## 3.5 高流量 ≠ 高转化：Top20 类目 / 商品",
             "cat20 = s3.fig_top_category(con)\nitem20 = s3.fig_top_item(con)\ncat20"),
            ("## 3.6 用户活跃度长尾与洛伦兹曲线",
             "lt, info = s3.fig_longtail(con)\nprint(info)\nlt"),
            ("## 3.7 图表已全部导出到 reports/figures/（统一字体、标题、轴标签、数据标签）",
             "sorted(p.name for p in C.FIG_DIR.glob('*.png'))"),
        ],
    ),
    (
        "04_funnel",
        "阶段 4 · 漏斗分析",
        "修正漏斗结构为 `pv → (cart | fav) → buy`；双口径（UV / Session）；加顺序约束；分维度下钻。",
        [
            ("## 漏斗结构说明\n\n加购与收藏是并列分支，不能串成先后关系。\n"
             "二者是并列分支，串起来会让中间环节人为缩窄、下游转化率被系统性低估。",
             "import stage4_funnel as s4\ns4.build_sessions(con)\ns4.build_uv_table(con)"),
            ("## 4.1/4.3 三步漏斗：UV 口径 vs Session 口径",
             "uv_tab = s4.funnel_table(con, 'users_funnel', 'UV 口径')\nsess_tab = s4.funnel_table(con, 'sessions', 'Session 口径')\nuv_tab"),
            ("## 4.2 Session 口径（含行为顺序约束）",
             "sess_tab"),
            ("## 4.4 并列分支：加购 / 收藏 / 纯浏览的最终购买概率",
             "s4.fig_branches(con)"),
            ("## 4.4' 逐日漏斗",
             "s4.funnel_by_day(con)"),
            ("## 4.4'' 新老用户漏斗",
             "s4.funnel_by_segment(con)"),
            ("## 4.4''' Top15 类目漏斗",
             "s4.funnel_by_category(con)"),
            ("## 4.6 结论\n\n1. 最大流失在「浏览 → 加购/收藏」；\n2. 加购用户成交概率远高于纯浏览用户，是最值得挽单的人群；\n"
             "3. 类目间转化能力差数倍，流量应重排。",
             "# 详见 reports/stage4_funnel_summary.md"),
        ],
    ),
    (
        "05_rfm",
        "阶段 5 · RFM 用户价值分层",
        "F 改为购买天数、M 显式声明为代理指标、打分改五分位、8 类 → 3 大类并做贡献度验证。",
        [
            ("## 5.1–5.4 计算 R/F/M 并做五分位打分",
             "import stage5_rfm as s5\nraw = s5.build_rfm(con)\ndf = s5.build_layers(raw)\ndf.head()"),
            ("## 5.5 分层结果（标准 8 类客户模型）",
             "g = s5.layer_summary(df)\ncls = s5.class_summary(df)\ng"),
            ("## 5.6 分层验证：人数占比 vs 购买贡献占比",
             "cls"),
            ("## 5.7 可视化：R–F 分面散点 / 占比-贡献双柱 / 打分单调性热力图",
             "s5.fig_facet_scatter(df)\ns5.fig_share_vs_contribution(g, cls)\ns5.fig_rfm_heatmap(df)"),
            ("## 5.8 每层运营动作",
             "import pandas as pd\npd.DataFrame({'客户分层': list(s5.PLAYBOOK), '建议动作': list(s5.PLAYBOOK.values())})"),
        ],
    ),
    (
        "06_modeling",
        "阶段 6/7 · 留存复购与聚类",
        "Cohort 留存矩阵、复购间隔、假设检验（周末 vs 工作日）、KMeans 用户聚类。",
        [
            ("## 6.1 留存口径\n\n以「用户首次活跃日」为 cohort 基准，计算次日 / 7 日留存。\n"
             "**注意窗口局限**：只有 9 天数据，后续 cohort 存在右截断。",
             "# 见 sql/06_retention.sql\nprint(open(C.SQL_DIR / '06_retention.sql', encoding='utf-8').read())"),
            ("## 6.4 复购率（复购定义 = 购买天数 ≥ 2）",
             "print(open(C.SQL_DIR / '07_repeat_purchase.sql', encoding='utf-8').read())"),
            ("## 7.1 假设检验：周末 vs 工作日购买转化",
             "# from scipy import stats"),
            ("## 7.4 KMeans 聚类\n\n特征：pv/cart/buy 频次 + 活跃天数 + 转化率 → StandardScaler → 肘部法定 K。",
             "# from sklearn.cluster import KMeans"),
        ],
    ),
]


def main() -> None:
    C.NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for fname, title, intro, cells in NOTEBOOKS:
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}\n\n"
                                                 f"> 代码逻辑统一放在 `src/`，notebook 只做编排与解读。"))
        nb.cells.append(nbf.v4.new_code_cell(SETUP))
        for md, code in cells:
            nb.cells.append(nbf.v4.new_markdown_cell(md))
            nb.cells.append(nbf.v4.new_code_cell(code))
        nbf.write(nb, str(C.NOTEBOOK_DIR / f"{fname}.ipynb"))
        print(f"写出 notebooks/{fname}.ipynb", flush=True)


if __name__ == "__main__":
    main()
