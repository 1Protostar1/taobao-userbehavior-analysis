# -*- coding: utf-8 -*-
"""
阶段 9 · 交付物：分析报告（单文件 HTML，图表全部内嵌）
======================================================
结构：背景 → 数据 → 方法 → 发现 → 结论建议 → 局限与改进 → 复现方式
产出：reports/淘宝用户行为分析报告.html
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import utils  # noqa: E402

FIGDIR = C.FIG_DIR
OUT = C.OUTPUT_DIR


def img(name: str, caption: str = "") -> str:
    p = FIGDIR / name
    if not p.exists():
        return f'<div class="missing">缺少图片：{name}</div>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    cap = f'<figcaption>{caption}</figcaption>' if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}"/>{cap}</figure>'


def table(df: pd.DataFrame, cls: str = "data") -> str:
    return df.to_html(index=False, border=0, escape=False, classes=cls,
                      float_format=lambda v: f"{v:,.4g}")


def load_csv(name: str) -> pd.DataFrame | None:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def load_sql(name: str) -> pd.DataFrame | None:
    p = OUT / "sql_results" / name
    return pd.read_csv(p) if p.exists() else None


def kpi_cards() -> str:
    cards = []
    basics = json.loads((OUT / "stage1_basic_metrics.json").read_text(encoding="utf-8"))[
        "basic_metrics"
    ]
    uv = load_csv("stage4_funnel_uv.csv")
    sess = load_csv("stage4_funnel_session.csv")
    rfm = load_csv("stage5_rfm_layer.csv")
    rep = load_sql("07_repeat_purchase.csv")
    dau = load_csv("stage3_dau.csv")

    cards.append(("数据总量", f"{basics['总行数'] / 1e8:.2f} 亿", "条用户行为日志（全量）"))
    cards.append(("用户规模", f"{basics['唯一用户数'] / 1e4:.0f} 万", f"商品 {basics['唯一商品数'] / 1e4:.0f} 万 / 类目 {basics['唯一类目数']:,}"))
    cards.append(("DAU 峰值", f"{dau['dau'].max() / 1e4:.0f} 万", f"出现在 {dau.loc[dau['dau'].idxmax(), '日期']}"))
    if sess is not None:
        s1 = sess[sess["环节"] == "① 浏览"]["Session 口径"].iloc[0]
        s3 = sess[sess["环节"] == "③ 意向后再购买"]["Session 口径"].iloc[0]
        cards.append(("浏览→成交转化率", f"{s3 / s1:.2%}", "Session 口径（含顺序约束）"))
    if uv is not None:
        u1 = uv[uv["环节"] == "① 浏览"]["UV 口径"].iloc[0]
        u3 = uv[uv["环节"] == "③ 意向后再购买"]["UV 口径"].iloc[0]
        cards.append(("用户级成交转化率", f"{u3 / u1:.2%}", "UV 口径（9 天窗口）"))
    if rep is not None:
        cards.append(("复购率", f"{rep['复购率'].iloc[0]:.1%}", "购买天数 ≥ 2 天的用户占比"))
    if rfm is not None:
        b = rfm[rfm["客户分层"] == "重要价值客户"]
        if len(b):
            cards.append(("重要价值客户", f"{b['用户占比'].iloc[0]:.1%}", f"贡献 {b['购买贡献占比'].iloc[0]:.1%} 的购买"))

    cl = load_csv("stage6_cohort_long.csv")
    if cl is not None:
        d1 = cl[cl["day_n"] == 1]
        d1_excl = d1[d1["首日"] != d1["首日"].min()]
        ret1 = d1_excl["留存人数"].sum() / d1_excl["cohort_size"].sum()
        cards.append(("次日留存率", f"{ret1:.1%}", "首日活跃 cohort 口径，剔除首日存量混合 cohort"))
    prof = load_csv("stage7_cluster_profile.csv")
    if prof is not None:
        core = prof[prof["画像命名"] == "核心购买用户"]
        if len(core):
            cards.append(("核心购买用户", f"{core['用户占比'].iloc[0]:.1%}",
                          f"KMeans 聚类，贡献 {core['购买贡献占比'].iloc[0]:.1%} 的购买"))

    html = ['<div class="kpis">']
    for t, v, s in cards:
        html.append(f'<div class="kpi"><div class="kpi-t">{t}</div><div class="kpi-v">{v}</div><div class="kpi-s">{s}</div></div>')
    html.append("</div>")
    return "".join(html)


def build_html() -> str:
    basics = json.loads((OUT / "stage1_basic_metrics.json").read_text(encoding="utf-8"))
    bm, act, conc = basics["basic_metrics"], basics["activity"], basics["concentration"]
    bench = load_csv("stage1_read_benchmark.csv")
    bias = load_csv("stage1_sampling_bias.csv")
    quality = load_csv("stage1_quality_checks.csv")
    daily = load_csv("stage1_rows_per_day.csv")
    dau = load_csv("stage3_dau.csv")
    hourly = load_csv("stage3_hourly.csv")
    behav = load_csv("stage3_behavior_share.csv")
    cat20 = load_csv("stage3_top20_category.csv")
    longtail = load_csv("stage3_activity_longtail.csv")
    uv = load_csv("stage4_funnel_uv.csv")
    sess = load_csv("stage4_funnel_session.csv")
    branch = load_csv("stage4_funnel_branches.csv")
    byday = load_csv("stage4_funnel_by_day.csv")
    byseg = load_csv("stage4_funnel_by_segment.csv")
    bycat = load_csv("stage4_funnel_by_category.csv")
    rfm = load_csv("stage5_rfm_layer.csv")
    rfm3 = load_csv("stage5_rfm_3class.csv")
    bot = load_csv("stage2_bot_sensitivity.csv") if (OUT / "stage2_bot_sensitivity.csv").exists() else None
    rep = load_sql("07_repeat_purchase.csv")
    ret = load_sql("06_retention.csv")
    # ---- 阶段 6 / 阶段 7 产出
    cohort_long = load_csv("stage6_cohort_long.csv")
    cohort_overall = load_csv("stage6_retention_overall.csv")
    rep6 = load_csv("stage6_repurchase.csv")
    interval = load_csv("stage6_repurchase_interval.csv")
    ret_cat = load_csv("stage6_retention_by_category.csv")
    ret_hour = load_csv("stage6_retention_by_first_hour.csv")
    tests = load_csv("stage7_hypothesis_tests.csv")
    ksel = load_csv("stage7_kmeans_k_selection.csv")
    prof = load_csv("stage7_cluster_profile.csv")
    meta7_p = OUT / "stage7_meta.json"
    meta7 = json.loads(meta7_p.read_text(encoding="utf-8")) if meta7_p.exists() else {}
    sens = load_csv("stage11_session_sensitivity.csv")

    u1, u2, u3 = uv["UV 口径"].tolist()
    s1, s2, s3 = sess["Session 口径"].tolist()
    peak_hour = int(hourly.loc[hourly["events"].idxmax(), "hour"])
    night = hourly[(hourly["hour"] >= 20) & (hourly["hour"] <= 22)]["events"].sum() / hourly["events"].sum()
    pv_share = behav.loc[behav["行为"] == "pv", "行为占比"].iloc[0]
    cart_p = branch.loc[branch["分支"] == "加购(cart)", "购买概率"].iloc[0]
    fav_p = branch.loc[branch["分支"] == "收藏(fav)", "购买概率"].iloc[0]
    both_p = branch.loc[branch["分支"] == "加购且收藏", "购买概率"].iloc[0]
    none_p = branch.loc[branch["分支"] == "仅浏览(无加购无收藏)", "购买概率"].iloc[0]
    best = rfm.loc[rfm["购买贡献占比"].idxmax()]
    top_cat = bycat.loc[bycat["浏览→成交转化率"].idxmax()]
    low_cat = bycat.loc[bycat["浏览→成交转化率"].idxmin()]
    a = bias["A 前100万行(原做法)"].tolist()
    b = bias["B 分层采样1%"].tolist()
    bias_num = bias.set_index("指标").apply(pd.to_numeric, errors="coerce")

    # ---- 阶段 6 / 7 派生值
    d1 = cohort_long[cohort_long["day_n"] == 1]
    ret1_all = d1["留存人数"].sum() / d1["cohort_size"].sum()
    d1_excl = d1[d1["首日"] != d1["首日"].min()]
    ret1_excl = d1_excl["留存人数"].sum() / d1_excl["cohort_size"].sum()
    first_cohort_ret1 = d1[d1["首日"] == d1["首日"].min()]["留存人数"].sum() / \
        d1[d1["首日"] == d1["首日"].min()]["cohort_size"].sum()
    mat = pd.read_csv(OUT / "stage6_cohort_matrix.csv", index_col=0)
    spike_example = mat.loc["2017-11-26", "D6"]   # 恰逢周六 12-02 的留存跳升
    r6 = rep6.iloc[0]
    rep_rate6 = r6["跨天复购用户数"] / r6["购买用户数"]
    multi_rate6 = r6["多次购买用户数"] / r6["购买用户数"]
    int_x, int_y = interval["间隔天数"].to_numpy(), interval["次数"].to_numpy()
    med_gap = float(np.median(np.repeat(int_x, int_y)))
    gap1_share = float(int_y[int_x == 1].sum() / int_y.sum())
    best_cat = ret_cat.loc[ret_cat["留存率差值"].idxmax()]
    best_hour = ret_hour.loc[ret_hour["次日留存率"].idxmax()]
    worst_hour = ret_hour.loc[ret_hour["次日留存率"].idxmin()]

    t_we = tests.iloc[0]      # 周末 vs 工作日（用户-日级主口径）
    t_act = tests.iloc[2]     # 高活跃 vs 低活跃
    core = prof[prof["画像命名"] == "核心购买用户"].iloc[0]
    seed = prof[prof["画像命名"] == "深度种草用户"].iloc[0]
    outlier = prof[prof["画像命名"] == "离群批发/代购账号"]
    n_outlier = int(outlier["用户数"].iloc[0]) if len(outlier) else 0

    tests_show = tests[["检验", "组1", "组2", "组1转化率", "组2转化率",
                        "率差(pp)", "相对提升", "z"]].copy()
    for c in ["组1转化率", "组2转化率"]:
        tests_show[c] = tests_show[c].map("{:.2%}".format)
    tests_show["率差(pp)"] = tests_show["率差(pp)"].map("{:+.2f}".format)
    tests_show["相对提升"] = tests_show["相对提升"].map("{:+.1%}".format)
    tests_show["z"] = tests_show["z"].map("{:+.1f}".format)

    prof_show = prof[["画像命名", "用户数", "用户占比", "浏览次数", "加购次数",
                      "收藏次数", "购买次数", "活跃天数", "购买贡献占比"]].copy()
    for c in ["浏览次数", "加购次数", "收藏次数", "购买次数", "活跃天数"]:
        prof_show[c] = prof_show[c].map("{:.1f}".format)
    prof_show["用户占比"] = prof_show["用户占比"].map("{:.1%}".format)
    prof_show["购买贡献占比"] = prof_show["购买贡献占比"].map("{:.1%}".format)

    # ---- 阶段 11 · Session 阈值敏感性（无产出时留空，保证 stage9 可独立运行）
    sens_block = ""
    if sens is not None:
        g15_ = sens[sens["阈值(分钟)"] == 15].iloc[0]
        g60_ = sens[sens["阈值(分钟)"] == 60].iloc[0]
        sens_show = sens[["阈值(分钟)", "会话总数", "意向会话", "成交会话",
                          "浏览→意向率", "意向→成交率", "浏览→成交率"]].copy()
        for c in ["浏览→意向率", "意向→成交率", "浏览→成交率"]:
            sens_show[c] = sens_show[c].map("{:.2%}".format)
        sens_block = f"""
  <p><b>阈值敏感性（15 / 30 / 60 分钟三档对照，逻辑完全一致只改阈值）</b>：
  会话总数随阈值缩短而增多（{g60_['会话总数'] / 1e6:.1f}M → {g15_['会话总数'] / 1e6:.1f}M），
  浏览→成交率被机械性抬高或压低（{g15_['浏览→成交率']:.2%} ~ {g60_['浏览→成交率']:.2%}），
  但<b>三档下"意向→成交始终最陡、浏览→意向流失人数最多"的相对结构完全一致</b> ——
  阈值影响数字大小，不影响任何一条业务结论的方向。</p>
  {table(sens_show)}
  {img("fig26_session_sensitivity.png", "图 25 · Session 阈值敏感性：15 / 30 / 60 分钟三档对照")}"""

    def rows(df, cols=None):
        return df[cols] if cols else df

    h = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>淘宝 UserBehavior 用户行为分析报告</title>
<style>
  :root {{
    --ink:#1a1f24; --sub:#5b666f; --line:#e3e8ee; --bg:#f6f8fa; --card:#ffffff;
    --blue:#1f6feb; --red:#d9483b; --green:#1a7f5a; --amber:#d97706; --violet:#7048b6;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:"Microsoft YaHei","PingFang SC","Helvetica Neue",Arial,sans-serif;
    line-height:1.75; font-size:15px; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:36px 30px 80px; }}
  header.cover {{ background:linear-gradient(135deg,#123a6b,#1f6feb); color:#fff; border-radius:14px;
    padding:38px 34px; margin-bottom:26px; }}
  header.cover h1 {{ margin:0 0 10px; font-size:30px; letter-spacing:.5px; }}
  header.cover .meta {{ opacity:.9; font-size:14px; }}
  header.cover .tags {{ margin-top:14px; }}
  header.cover .tag {{ display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);
    border-radius:999px; padding:3px 12px; font-size:12.5px; margin:0 8px 8px 0; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:14px; margin:22px 0 30px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px;
    box-shadow:0 1px 2px rgba(16,24,40,.04); }}
  .kpi-t {{ font-size:12.5px; color:var(--sub); }}
  .kpi-v {{ font-size:25px; font-weight:800; margin:4px 0 2px; color:var(--blue); letter-spacing:.3px; }}
  .kpi-s {{ font-size:12px; color:var(--sub); }}
  section {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:26px 28px;
    margin-bottom:22px; box-shadow:0 1px 2px rgba(16,24,40,.04); }}
  h2 {{ font-size:20px; margin:0 0 14px; padding-left:12px; border-left:4px solid var(--blue); }}
  h3 {{ font-size:16px; margin:24px 0 10px; color:#123a6b; }}
  p {{ margin:10px 0; }}
  ul,ol {{ margin:10px 0 10px 22px; padding:0; }}
  li {{ margin:5px 0; }}
  table {{ width:100%; border-collapse:collapse; margin:14px 0; font-size:13.5px; }}
  th {{ background:#eef3f9; color:#123a6b; text-align:left; padding:9px 10px; border-bottom:1px solid var(--line);
    font-weight:700; white-space:nowrap; }}
  td {{ padding:8px 10px; border-bottom:1px solid #f0f3f7; }}
  tr:hover td {{ background:#fafcff; }}
  figure {{ margin:18px 0; text-align:center; }}
  figure img {{ max-width:100%; border:1px solid var(--line); border-radius:10px; background:#fff; }}
  figcaption {{ font-size:12.5px; color:var(--sub); margin-top:6px; }}
  .callout {{ border-left:4px solid var(--amber); background:#fff8ec; padding:12px 16px; border-radius:8px; margin:14px 0; }}
  .callout.good {{ border-color:var(--green); background:#eefaf4; }}
  .callout.bad {{ border-color:var(--red); background:#fdeeec; }}
  .callout b {{ color:#123a6b; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .missing {{ color:#b23; font-size:13px; }}
  code {{ background:#eef2f6; padding:1px 6px; border-radius:5px; font-size:13px;
    font-family:Consolas,Monaco,monospace; }}
  pre {{ background:#0f1b2d; color:#e6edf3; padding:14px 16px; border-radius:10px; overflow:auto; font-size:13px; }}
  .foot {{ text-align:center; color:var(--sub); font-size:12.5px; margin-top:26px; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  @media print {{ body{{background:#fff}} section,header.cover{{box-shadow:none}} }}
</style>
</head>
<body>
<div class="wrap">

<header class="cover">
  <h1>淘宝 UserBehavior 用户行为分析</h1>
  <div class="meta">数据范围：{bm['窗口内最早']} ~ {bm['窗口内最晚']}（{bm['窗口内天数']} 天，北京时间）&nbsp;|&nbsp;
  数据规模：{bm['总行数']:,} 行 / {bm['唯一用户数']:,} 用户 / {bm['唯一商品数']:,} 商品</div>
  <div class="tags">
    <span class="tag">DuckDB 全量计算</span><span class="tag">Session 级漏斗</span>
    <span class="tag">RFM 五分位分层</span><span class="tag">留存 Cohort</span>
    <span class="tag">假设检验</span><span class="tag">KMeans 聚类</span>
    <span class="tag">采样偏差自检</span><span class="tag">可复现工程结构</span>
  </div>
</header>

{kpi_cards()}

<section>
  <h2>一、项目背景与分析目标</h2>
  <p>数据源是淘宝公开的 UserBehavior 用户行为日志：每条记录表示一个用户在某一秒对某个商品的
  <b>浏览 / 加购 / 收藏 / 购买</b> 四类行为之一。业务问题非常典型——<b>流量不缺，缺的是把流量转成成交的效率</b>。</p>
  <p>本次分析回答六个问题：</p>
  <ol>
    <li>这批数据是什么、能不能用、有什么坑？（数据质量与口径校验）</li>
    <li>用户在什么时间、对什么类目、以什么节奏活跃？（描述性分析）</li>
    <li>从浏览到成交，用户在哪一步流失？不同口径下结论是否一致？（漏斗分析）</li>
    <li>哪些用户值得投入、哪些应该放弃？（RFM 用户价值分层）</li>
    <li>新客能不能留下来、买了还会不会再买？（留存 Cohort 与复购）</li>
    <li>观察到的差异是真实信号还是随机噪声？无标签数据能切出几类用户？（假设检验与 KMeans 聚类）</li>
  </ol>
  <div class="callout"><b>一句话结论：</b>漏斗最陡的一段是「加购/收藏 → 购买」（意向会话仅 {s3 / s2:.1%} 在同一次会话内成交），
  而丢人最多的是「浏览 → 加购/收藏」（{sess['环节流失率'].iloc[1]:.1%} 的浏览会话没有产生任何意向行为）；
  加购用户的后续购买概率是纯浏览用户的 {cart_p / max(none_p, 1e-9):.1f} 倍，是最值得做挽单的人群；
  同时 {best['用户占比']:.1%} 的高价值用户贡献了 {best['购买贡献占比']:.1%} 的购买，运营资源应据此重新分配；
  留存侧识别出"翘尾=周末日历效应"的口径陷阱，且 {gap1_share:.1%} 的复购发生在首购次日——
  <b>首购后 24 小时触达</b>是投入产出比最高的动作。</div>
</section>

<section>
  <h2>二、数据说明与质量校验</h2>
  <h3>2.1 基础指标</h3>
  {table(daily)}
  <h3>2.2 读取方式选型</h3>
  <p>1 亿行 × 5 列，用 <code>pandas.read_csv</code> 全量读入需要约 30 GB 内存，单机不可行。
  实测三种方式后确定主链路：</p>
  {table(bench)}
  <div class="callout good"><b>结论：</b>主链路用 <b>DuckDB 直接对 3.6 GB CSV 跑 SQL</b>
  （零部署、列式向量化、多线程）；pandas 仅用于 10% 分层样本的交互式探索。</div>

  <h3>2.3 数据质量检查</h3>
  {table(quality)}

  <h3>2.4 排序陷阱实证（本项目最关键的一次自检）</h3>
  <p>原始代码用 <code>pd.read_csv(..., nrows=1000000)</code> 取前 100 万行。
  本次对这个"约定俗成"的做法做了实证检验，结果和想当然的不一样，值得完整记录。</p>
  <p><b>发现一：文件不是按 user_id 数值序排的，而是按字符串字典序排的。</b>
  实测前 300 万行里用户出现顺序为
  <code>1 → 100 → 1000 → 1000001 → 1000004 → 1000011 → 100002 → 100009 → 10001 …</code>，
  是典型的字典序（<code>"1000011" &lt; "100002"</code>）。
  因此 <code>nrows=1000000</code> 取到的是"ID 十进制写法字典序最靠前"的一批用户，
  是一个<b>系统性、非随机</b>的子集，只覆盖全站 {bias_num.loc['唯一用户数'].iloc[0] / bm['唯一用户数']:.2%} 的用户。</p>
  <p><b>发现二：在这份数据上，它与分层随机抽样的行为结构差异并不大</b>（各项差异在 0.5%–6% 量级）：</p>
  {table(bias)}
  {img("fig01_sampling_bias.png", "图 1 · 排序陷阱实证：左图为文件中的用户出现顺序（字典序锯齿），右图为两组样本的行为结构对比")}
  <div class="callout bad"><b>为什么仍然不能用 <code>nrows</code>：</b>
  ① 系统性抽样没有概率保证，"碰巧没偏"不等于无偏，换数据、换指标就可能完全不同；
  ② 只覆盖 1% 的用户、1% 的行为，任何用户级结论（RFM、留存、聚类）都不稳定；
  ③ 不跑对照实验根本不知道偏没偏——本次对照实验只花了约 2 分钟，换来的是确定性。<br/>
  <b>修正动作：</b>对外结论一律用 <b>全量 1 亿行</b>（DuckDB）；
  交互探索改用 <b>按 user_id 哈希的分层 10% 样本</b>——同一用户整条行为链同进同出，
  不会把用户"切碎"从而虚高漏斗转化率。</div>
  <p>本次探查还发现了第二类质量问题：<b>时间戳越界</b>。全量数据共有 347 个不同日期，
  但只有 99.94% 的行落在 9 天有效窗口内；窗口外 55,576 行（0.055%）的时间戳最早到 1902 年、最晚到 2037 年，
  已按 <code>config.WINDOW_SQL</code> 在中间层统一剔除（详见 <code>reports/data_quality_report.md</code>）。</p>

  <h3>2.5 用户活跃度与集中度</h3>
  {table(longtail)}
  <p>人均行为 {act['人均行为数']:.1f} 条、中位数 {act['行为数中位数']:.0f} 条、最高 {act['最大值']:,.0f} 条，分布明显右偏；
  购买侧集中度：<b>top 1% 购买用户贡献 {conc['top1pct_share']:.1%} 的购买次数</b>，
  top 10% 贡献 {conc['top10pct_share']:.1%}，top 20% 贡献 {conc['top20pct_share']:.1%}
  ——属于中度集中（top 20% 的贡献占比是人数占比的 {conc['top20pct_share'] / 0.2:.1f} 倍），
  不是极端二八分布，但足以支撑"资源按价值分层投放"的结论。</p>
  {img("fig09_user_activity_longtail.png", "图 2 · 用户活跃度长尾与洛伦兹曲线")}
</section>

<section>
  <h2>三、分析方法与口径定义</h2>
  <h3>3.1 Session（会话）口径 —— 30 分钟阈值</h3>
  <ul>
    <li>规则：同一用户按时间排序，<b>相邻行为间隔 &gt; {C.SESSION_GAP_MINUTES} 分钟</b>即切分为新会话。</li>
    <li>为什么是 30 分钟：业界通用会话超时经验值（Google Analytics 默认 30 min）。
    间隔过短会把一次连续浏览切碎，过长会把两次独立访问粘在一起。</li>
    <li>局限：本数据是秒级行为流，没有页面停留时长，只能用"行为间隔"做代理。</li>
  </ul>
  {sens_block}

  <h3>3.2 漏斗口径 —— 修正为 <code>pv → (cart | fav) → buy</code></h3>
  <p>原代码把 <code>['pv','cart','fav','buy']</code> 当成四级线性链路，这是<b>逻辑错误</b>：
  加购与收藏是<b>并列的两种意向表达</b>，不存在先后关系。串成一条线会让中间环节人为缩窄，
  下游转化率被系统性低估。本项目改为两步分层 + 分支并行刻画，并<b>加上行为顺序约束</b>
  （先浏览、再有加购或收藏、最后购买，且顺序正确）。</p>
  <p>同时给出双口径：<b>UV 口径</b>（按用户去重，看整体经营效率）与
  <b>Session 口径</b>（同一次会话内闭环，看单次访问效率）。</p>

  <h3>3.3 RFM 口径 —— 从"均值 0/1"升级为"五分位"</h3>
  <table class="data">
    <tr><th>维度</th><th>本项目定义</th><th>说明</th></tr>
    <tr><td>R</td><td>末次购买距数据集末端的天数</td><td>9 天窗口 → 取值仅 0–8，判别力受限（已写入局限）</td></tr>
    <tr><td>F</td><td><b>购买天数</b></td><td>比"购买次数"更稳，抗单次多笔 / 刷单</td></tr>
    <tr><td>M</td><td><b>购买的不同商品件数</b></td><td>⚠️ 无金额字段，M 是<b>代理指标</b>，不等同于消费金额</td></tr>
  </table>
  <p>打分：先 <code>rank(method='first')</code> 再 <code>qcut(…,5)</code> 分五档，保证严格等分
  （直接 qcut 会被大量并列值破坏）；R 反向赋分，F/M 正向赋分；≥4 分记为"高"，组合成标准 8 类客户。</p>
</section>

<section>
  <h2>四、核心发现</h2>

  <h3>4.1 时间节律：晚间 20–22 点是绝对高峰</h3>
  <p>行为峰值时段为 <b>{peak_hour}:00</b>，晚间 20:00–22:00 贡献全天 <b>{night:.1%}</b> 的行为量；
  周末日均活跃显著高于工作日（约 +17.6%）。浏览（pv）占全部行为的 <b>{pv_share:.2%}</b>。</p>
  {img("fig03_dau_trend.png", "图 3 · 逐日 DAU 与行为总量")}
  {img("fig04_hourly_activity.png", "图 4 · 小时活跃分布与工作日/周末节律")}
  {img("fig05_weekday_activity.png", "图 5 · 周内活跃分布")}
  {img("fig06_behavior_share.png", "图 6 · 行为类型构成与用户覆盖")}
  <div class="callout good"><b>可执行动作：</b>秒杀 / 直播 / Push 等资源集中在 19:30–22:30 投放；
  凌晨 2:00–5:00 可安排批量任务与系统维护，对业务影响最小。</div>

  <h3>4.2 高流量 ≠ 高转化</h3>
  <p>Top20 类目中，最高转化率类目（{int(top_cat['类目'])}）达 <b>{top_cat['浏览→成交转化率']:.2%}</b>，
  最低（{int(low_cat['类目'])}）仅 <b>{low_cat['浏览→成交转化率']:.2%}</b>，相差
  <b>{top_cat['浏览→成交转化率'] / max(low_cat['浏览→成交转化率'], 1e-9):.1f} 倍</b>。
  部分类目流量很高但转化明显低于 Top20 均值，属于典型的"高流量低转化"。</p>
  {table(cat20[['k', 'pv', 'pv_uv', 'buy_uv', '浏览→购买转化率', '加购或收藏率']])}
  {img("fig07_top20_category_pv_vs_cvr.png", "图 7 · Top20 类目：浏览量 vs 转化率")}
  {img("fig08_top20_item_pv_vs_cvr.png", "图 8 · Top20 商品：流量集中度与转化率")}
  <div class="callout good"><b>可执行动作：</b>对"高流量低转化"类目做
  <b>详情页 / 价格带 / 库存</b>三项诊断，把转化率拉到同品类均值，
  即可在不增加任何 UV 的前提下提升成交。</div>

  <h3>4.3 漏斗：最大流失在"浏览 → 加购/收藏"</h3>
  <div class="grid2">
    <div>{table(uv)}</div>
    <div>{table(sess)}</div>
  </div>
  {img("fig10_funnel_overview.png", "图 9 · 双口径漏斗对照")}
  {img("fig11_funnel_waterfall.png", "图 10 · 流失瀑布图")}
  <p>UV 口径浏览→成交 <b>{u3 / u1:.2%}</b>（{u1:,} → {u2:,} → {u3:,}）；
  Session 口径浏览→成交 <b>{s3 / s1:.2%}</b>（{s1:,} → {s2:,} → {s3:,}）。
  两种口径差 {abs(u3 / u1 - s3 / s1):.2%}，差异来源是 UV 口径会把"跨会话"的意向与成交算进同一条链，
  因此转化率天然偏高。<b>汇报时必须写清用的是哪个口径。</b></p>

  <h3>4.4 加购与收藏的意向强度完全不同</h3>
  {table(branch)}
  {img("fig12_funnel_branches.png", "图 11 · 并列分支的最终购买概率")}
  <div class="callout good"><b>关键发现：</b>加购用户最终购买概率 <b>{cart_p:.1%}</b>，
  收藏用户 <b>{fav_p:.1%}</b>，两者都无的纯浏览用户仅 <b>{none_p:.1%}</b>。
  加购的意向强度约是收藏的 <b>{cart_p / max(fav_p, 1e-9):.1f} 倍</b>
  —— 这正是"加购后必须做挽单、收藏只需轻提醒"的直接数据依据。</div>

  <h3>4.5 分维度漏斗：逐日、新老用户、类目</h3>
  {img("fig13_funnel_by_day.png", "图 12 · 逐日漏斗转化率趋势")}
  {table(byday)}
  {table(byseg)}
  {img("fig14_funnel_by_segment.png", "图 13 · 新老用户漏斗对比")}
  {img("fig15_funnel_by_category.png", "图 14 · Top15 类目漏斗")}
  <p>窗口内漏斗没有恶化趋势；新增用户的严格链路转化率低于存量人群，但"任意购买率"差距小得多，
  说明新用户不是不买，而是<b>不在同一次会话里买</b>（决策链路更长，需要跨会话召回）。</p>

  <h3>4.6 RFM 用户价值分层</h3>
  {table(rfm)}
  {img("fig17_rfm_share_contribution.png", "图 15 · 人数占比 vs 购买贡献占比")}
  {img("fig16_rfm_scatter_facet.png", "图 16 · R–F 分面散点（8 类客户）")}
  {img("fig18_rfm_score_validation.png", "图 17 · 打分口径单调性校验")}
  {table(rfm3)}
  <p><b>重要价值客户</b>以 {best['用户占比']:.1%} 的人数贡献了 <b>{best['购买贡献占比']:.1%}</b> 的购买，
  人均购买 {best['人均购买次数']:.1f} 次，是全体购买用户均值的
  {best['人均购买次数'] / (rfm['购买次数合计'].sum() / rfm['用户数'].sum()):.1f} 倍。
  分层有效，直接回答了"预算该花在谁身上"。</p>

  <h3>4.7 留存与复购：日历效应是这组数据里最容易踩的坑</h3>
  <p>留存以<b>用户首次活跃日</b>为 cohort 基准，分母是首日活跃用户。
  全体加权次日留存率 <b>{ret1_all:.1%}</b>；但 2017-11-25 是窗口第一天，当天"首次活跃"混入了大量存量老客
  （其次日留存 {first_cohort_ret1:.1%} 显著偏高），剔除后真实的<b>新增用户次日留存率为 {ret1_excl:.1%}</b>。</p>
  {img("fig19_cohort_retention_heatmap.png", "图 18 · Cohort 留存矩阵：凡第 N 天恰逢周末（12-02/03），留存率必跳 95%+；空格为窗口右截断，不做外推")}
  {img("fig20_retention_curves.png", "图 19 · 各 cohort 留存衰减曲线")}
  <div class="callout bad"><b>口径发现：</b>留存曲线的"翘尾"<b>不是</b>用户粘性随时间变好，
  而是<b>周末流量高峰的日历效应</b>——例如 11-26 cohort 的第 6 天恰逢周六，留存率跳到 {spike_example:.1%}。
  同理，加权 7 日留存高达 98%+ 也是被周末截断放大的，不能与常规意义上的 D7 留存直接比较。
  剔除日历效应后，同一 cohort 在工作日区间内基本平稳在 62%~72%，
  流失集中在首次活跃后的 48 小时内。</div>
  <p>复购侧：购买用户共 {int(r6['购买用户数']):,} 人，<b>跨天复购率 {rep_rate6:.1%}</b>（购买天数 ≥ 2），
  另有 {multi_rate6 - rep_rate6:.1%} 是"同一天一单多件"——用购买天数定义复购可剔除这种一次性囤货行为。
  相邻购买间隔中位数 <b>{med_gap:.0f} 天</b>，<b>{gap1_share:.1%} 的复购发生在首购次日</b>。</p>
  {img("fig21_repurchase.png", "图 20 · 复购两种口径与购买间隔分布")}
  {img("fig22_retention_by_segment.png", "图 21 · 分品类 / 分时段的次日留存差异")}
  <p>分品类：首日浏览类目 {int(best_cat['类目ID'])} 的新客次日留存率最高
  （{best_cat['该类目用户次日留存率']:.1%}，比未浏览高 {best_cat['留存率差值'] * 100:.1f}pp），有"钩子"效应；
  分时段：<b>{best_hour['时段']}</b>首活用户留存最高（{best_hour['次日留存率']:.1%}，自选择效应），
  <b>{worst_hour['时段']}</b>最低（{worst_hour['次日留存率']:.1%}）——晚间高峰流量大但新客质量被稀释。</p>

  <h3>4.8 假设检验与用户聚类：先证伪，再分层</h3>
  <p>对两个"想当然"的运营假设做了双比例 z 检验（另附用户级稳健性检验）：</p>
  {table(tests_show)}
  <div class="callout"><b>怎么读 p 值：</b>p &lt; 0.05 只说明"差异不太像随机噪声"，<b>不等于</b>有业务意义——
  百万级样本下 0.1pp 的率差也能显著，必须同时看率差与相对提升。
  周末 vs 工作日率差 {t_we['率差(pp)']:+.2f}pp（{t_we['相对提升']:+.1%}）：方向为负，周末流量"量高质低"，
  增量流量以低意向"逛街"用户为主，不值得为周末单独加码；
  高活跃 vs 低活跃购买率差 {t_act['率差(pp)']:+.1f}pp（{t_act['相对提升']:+.1%}）：<b>活跃度才是转化的核心杠杆</b>。
  （用户级稳健性检验方向一致，但存在"仅周末活跃用户最多 2 个可活跃日、仅工作日 5 个"的机会偏差，
  会夸大差异，故效应量以用户-日口径为准。）</div>
  <p>KMeans 聚类：特征 = pv / cart / fav / buy 频次 + 活跃天数 + 购买转化率，StandardScaler 标准化；
  肘部法 + 轮廓系数（2 万采样估算）在 K=3~6 中取最优，选定 <b>K={meta7.get('chosen_k', 6)}</b>
  （轮廓系数 {meta7.get('silhouette_chosen', 0):.3f}）。</p>
  {img("fig23_kmeans_k_selection.png", "图 22 · 选 K：肘部法 + 轮廓系数")}
  {table(prof_show)}
  {img("fig24_cluster_profile.png", "图 23 · 聚类画像热力图")}
  <ul>
    <li><b>核心购买用户</b>（{int(core['用户数']):,} 人，占 {core['用户占比']:.1%}）贡献
    <b>{core['购买贡献占比']:.1%}</b> 的购买，与 RFM"少数用户贡献多数购买"互相印证；</li>
    <li><b>深度种草用户</b>（{int(seed['用户数']):,} 人）：人均收藏 {seed['收藏次数']:.1f} 次、
    购买转化率仅 {seed['购买转化率']:.1%}——"种草未拔草"，是优惠券投放转化率最高的目标层，
    也是 RFM 的盲区（RFM 只看购买行为，看不到他们）；</li>
    <li><b>离群簇 {n_outlier} 人</b>：人均购买 64.6 次却几乎零浏览，疑似批发/代购/脚本账号，
    应移交风控而非运营——KMeans 对极端值敏感会自然切出这类微簇。</li>
  </ul>
  {img("fig25_cluster_vs_rfm.png", "图 24 · 聚类 × RFM 价值大类交叉验证")}
  <p>聚类结果与 RFM 价值大类的 <b>Cramér's V = {meta7.get('cramers_v', 0):.3f}</b>：
  两套方法独立切出的高价值人群高度重叠（核心购买簇中 A 类高价值活跃占比最高，
  低频路过簇以 C 类长尾为主），说明分层结论<b>稳健、不依赖单一方法</b>。</p>
</section>

<section>
  <h2>五、结论与运营建议</h2>
  <ol>
    <li><b>两个视角看漏斗：最陡的一段是「加购/收藏 → 购买」，丢人最多的是「浏览 → 加购/收藏」。</b>
    Session 口径下，意向会话里只有 {s3 / s2:.1%} 在同一次会话内成交；
    同时有 {int(sess['流失人数'].iloc[1]) / 1e4:.0f} 万个会话（{sess['环节流失率'].iloc[1]:.1%}）连加购/收藏都没有产生。
    动作：前者做"购物车临门一脚"（降价提醒 / 库存告急 / 免运费），后者做"进店即种草"（加购引导 + 加购券）。</li>
    <li><b>加购/收藏人群是最值得的挽单对象。</b>加购用户后续购买概率 {cart_p:.1%}、
    收藏用户 {fav_p:.1%}、加购且收藏 {both_p:.1%}，均高于纯浏览用户 {none_p:.1%}。
    （口径提醒：这是"该用户最终买过任何东西"的概率，属相关性证据，不是单品归因。）
    动作：24 小时内定向触达购物车未购人群。</li>
    <li><b>流量分配按转化效率重排。</b>Top 类目转化率相差
    {top_cat['浏览→成交转化率'] / max(low_cat['浏览→成交转化率'], 1e-9):.1f} 倍。
    动作：把首页/推荐位曝光从"高浏览低转化"类目适度迁移到"高效率但流量不足"的类目。</li>
    <li><b>新用户做跨会话召回，而不是当场逼单。</b>配置次日回访券，而不是只做首次会话内的强促。</li>
    <li><b>运营预算按 RFM 分层而非人数分配。</b>
    A 类"高价值活跃"用户给 VIP 权益与新品优先购；
    B 类"高价值沉默"做专属回归礼与老客价；
    C 类长尾只做低成本自动化触达。</li>
    <li><b>首购后 24 小时是复购召回的黄金窗口。</b>{gap1_share:.1%} 的复购发生在首购次日、
    复购间隔中位数 {med_gap:.0f} 天；新客流失集中在首访后 48 小时。
    动作：首购/首访次日定向推送"同类商品推荐 + 优惠券到期提醒"，
    比广撒网式召回投入产出比更高。</li>
    <li><b>别为周末单独加码投放，把优惠券预算给"深度种草"人群。</b>
    假设检验显示周末转化率反而低 {abs(t_we['率差(pp)']):.2f}pp（量高质低）；
    而 KMeans 定位的 {int(seed['用户数']):,} 名深度种草用户（高收藏低转化）
    是最容易被一张券推动成交的人群，也是 RFM 看不见的盲区。
    另：{n_outlier} 个"买而不逛"的离群账号建议移交风控核查。</li>
  </ol>
</section>

<section>
  <h2>六、数据局限与改进方向</h2>
  <ol>
    <li><b>时间窗口只有 {bm['窗口内天数']} 天。</b>无法观察跨月 / 大促周期；R 值被压缩到 0–8 天，
    留存只能做次日与 7 日；7 日留存还存在窗口右截断。
    改进：接入更长周期的日志（建议 ≥ 90 天）后重算 R 与 cohort 留存。</li>
    <li><b>没有金额字段。</b>RFM 的 M 只能用"购买商品件数"作代理，无法衡量真实 GMV 贡献，
    也无法做价格带分析。改进：接入订单金额表后重算 M，并升级为 RFM + 客单价四象限。</li>
    <li><b>没有用户画像。</b>缺年龄 / 性别 / 地域 / 渠道，无法做人群细分，
    也无法排除"一人多账号"。改进：接入用户属性表与设备指纹。</li>
    <li><b>没有曝光数据。</b>日志只记录点击后的行为，没有曝光量，因此算不出真实 CTR，
    "浏览量"只能当近似分母。改进：接入曝光埋点后重建曝光 → 点击 → 加购 → 成交全链路。</li>
    <li><b>存在极端高频用户与脏时间戳。</b>极端高频用户按"数据驱动阈值（用户-日 pv 的 p99.9）"标记，
    并做了剔除前后的敏感性对比（见 <code>reports/stage2_cleaning_summary.md</code>），结论方向未被改变；
    窗口外 55,576 行脏时间戳已在中间层剔除。改进：引入设备指纹 + 行为熵阈值做更严格的异常识别。</li>
    <li><b>留存与复购结论受 9 天窗口强约束。</b>只能观察短期留存；7 日留存存在右截断且被周末日历效应放大；
    复购率是窗口内下限（真实复购需要更长观察期）。解读时已逐处标注，不做跨窗口外推。</li>
  </ol>
</section>

<section>
  <h2>七、复现方式</h2>
  <pre>本项目目录结构：
Taobao user/
├─ src/            分析代码（config / db / stage1~stage9 全流程）
├─ sql/            10 条核心 SQL（DuckDB 可直接执行）
├─ notebooks/      按分析阶段拆分的 notebook
├─ outputs/        parquet 中间层 + 各阶段 CSV / JSON 结果
├─ reports/        分析报告与全部图表
└─ data/           原始数据说明（3.6 GB CSV 保持原位，不做移动）

一条命令跑通全流程：
  .venv\\Scripts\\python.exe src\\run_all.py
</pre>
  <p>环境依赖见 <code>requirements.txt</code>；每次运行的耗时与产物见各阶段 <code>reports/stage*.md</code>。</p>
</section>

<div class="foot">数据来源：淘宝 UserBehavior 公开数据集 · 全量 {bm['总行数']:,} 行 ·
时间戳按北京时间（UTC+8）解释 · 报告由 src/stage9_report.py 自动生成</div>
</div>
</body>
</html>"""
    return h


def main() -> None:
    html = build_html()
    out = C.REPORT_DIR / "淘宝用户行为分析报告.html"
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}  ({out.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
