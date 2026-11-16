# -*- coding: utf-8 -*-
"""
阶段 9.2 · 交互看板（单文件 HTML，Plotly）
==========================================
看板用 Plotly 单文件 HTML：双击即开、无需起服务，数据来自 outputs/ 的聚合表，不重复扫原始 CSV。

产出：reports/用户行为看板.html（plotly.js 内联，离线可开）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402

OUT = C.OUTPUT_DIR
FONT = dict(family="Microsoft YaHei, PingFang SC, sans-serif")
BLUE, ORANGE, GREEN, RED, TEAL, PURPLE = "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


def kpi_html() -> str:
    basics = json.loads((OUT / "stage1_basic_metrics.json").read_text(encoding="utf-8"))["basic_metrics"]
    uv, sess = load("stage4_funnel_uv.csv"), load("stage4_funnel_session.csv")
    rep = pd.read_csv(OUT / "sql_results" / "07_repeat_purchase.csv")
    dau = load("stage3_dau.csv")
    cl = load("stage6_cohort_long.csv")
    d1 = cl[cl["day_n"] == 1]
    d1 = d1[d1["首日"] != d1["首日"].min()]
    ret1 = d1["留存人数"].sum() / d1["cohort_size"].sum()
    prof = load("stage7_cluster_profile.csv")
    core = prof[prof["画像命名"] == "核心购买用户"].iloc[0]

    u1, u3 = uv["UV 口径"].iloc[0], uv["UV 口径"].iloc[2]
    s1, s3 = sess["Session 口径"].iloc[0], sess["Session 口径"].iloc[2]
    cards = [
        ("数据总量", f"{basics['总行数'] / 1e8:.2f} 亿", "条行为日志（全量）"),
        ("DAU 峰值", f"{dau['dau'].max() / 1e4:.0f} 万", str(dau.loc[dau['dau'].idxmax(), '日期'])),
        ("浏览→成交（UV）", f"{u3 / u1:.1%}", "9 天窗口，含顺序约束"),
        ("浏览→成交（Session）", f"{s3 / s1:.2%}", "30 分钟会话内闭环"),
        ("次日留存率", f"{ret1:.1%}", "剔除首日存量混合 cohort"),
        ("复购率", f"{rep['复购率'].iloc[0]:.1%}", "购买天数 ≥ 2"),
        ("核心购买用户", f"{core['用户占比']:.1%}", f"贡献 {core['购买贡献占比']:.1%} 购买"),
    ]
    items = "".join(
        f'<div class="kpi"><div class="kpi-t">{t}</div><div class="kpi-v">{v}</div>'
        f'<div class="kpi-s">{s}</div></div>'
        for t, v, s in cards
    )
    return f'<div class="kpis">{items}</div>'


# ------------------------------------------------------------------ 各图
def fig_dau() -> go.Figure:
    d = load("stage3_dau.csv")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=d["日期"], y=d["events"], name="行为总量", marker_color="#D6E4F5",
                hovertemplate="%{x}<br>行为量 %{y:,.0f}<extra></extra>", secondary_y=False)
    fig.add_scatter(x=d["日期"], y=d["dau"], name="DAU", mode="lines+markers",
                    line=dict(color=BLUE, width=3), marker=dict(size=8),
                    hovertemplate="%{x}<br>DAU %{y:,.0f}<extra></extra>", secondary_y=True)
    fig.update_layout(title="逐日 DAU 与行为总量（周末 12-02/03 为流量高峰）",
                      legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="行为量", secondary_y=False)
    fig.update_yaxes(title_text="DAU", secondary_y=True)
    return fig


def fig_hourly() -> go.Figure:
    d = load("stage3_hourly.csv")
    fig = go.Figure()
    fig.add_bar(x=d["hour"], y=d["工作日日均"], name="工作日日均", marker_color=BLUE)
    fig.add_bar(x=d["hour"], y=d["周末日均"], name="周末日均", marker_color=ORANGE)
    fig.update_layout(title="小时活跃分布：20-22 点为绝对高峰（按日均可比）",
                      barmode="group", xaxis_title="小时", yaxis_title="日均行为量",
                      legend=dict(orientation="h", y=1.12))
    return fig


def fig_funnel() -> go.Figure:
    uv, sess = load("stage4_funnel_uv.csv"), load("stage4_funnel_session.csv")
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "funnel"}, {"type": "funnel"}]],
                        subplot_titles=("UV 口径（用户去重）", "Session 口径（单次会话）"))
    fig.add_trace(go.Funnel(y=uv["环节"], x=uv["UV 口径"],
                            textinfo="value+percent initial",
                            marker=dict(color=[BLUE, TEAL, GREEN])),
                  row=1, col=1)
    fig.add_trace(go.Funnel(y=sess["环节"], x=sess["Session 口径"],
                            textinfo="value+percent initial",
                            marker=dict(color=[BLUE, ORANGE, RED])),
                  row=1, col=2)
    fig.update_layout(title="转化漏斗双口径：UV 看整体经营效率，Session 看单次访问效率")
    return fig


def fig_cohort() -> go.Figure:
    m = pd.read_csv(OUT / "stage6_cohort_matrix.csv", index_col=0)
    fig = go.Figure(go.Heatmap(
        z=m.values, x=list(m.columns), y=[str(i)[:10] for i in m.index],
        colorscale="YlGnBu", zmin=0, zmax=1,
        text=[[f"{v:.1%}" if pd.notna(v) else "截断" for v in row] for row in m.values],
        texttemplate="%{text}", hovertemplate="cohort %{y} · %{x}<br>留存率 %{text}<extra></extra>",
        colorbar=dict(title="留存率", tickformat=".0%"),
    ))
    fig.update_layout(title="Cohort 留存矩阵：第 N 天恰逢周末必跳升（日历效应，非粘性变好）",
                      xaxis_title="首次活跃后第 N 天", yaxis_title="首次活跃日",
                      yaxis=dict(autorange="reversed"))
    return fig


def fig_repurchase() -> go.Figure:
    d = load("stage6_repurchase_interval.csv")
    total = d["次数"].sum()
    fig = go.Figure(go.Bar(
        x=d["间隔天数"], y=d["次数"], marker_color=BLUE,
        text=[f"{v / total:.1%}" for v in d["次数"]], textposition="outside",
        hovertemplate="间隔 %{x} 天：%{y:,.0f} 次<extra></extra>",
    ))
    fig.update_layout(title="复购间隔分布：41.9% 的复购发生在首购次日（召回黄金窗口 24h）",
                      xaxis_title="相邻两次购买间隔（天）", yaxis_title="次数")
    return fig


def fig_rfm() -> go.Figure:
    g = load("stage5_rfm_layer.csv")
    fig = go.Figure()
    fig.add_bar(x=g["客户分层"], y=g["用户占比"], name="用户占比", marker_color="#9ECAE1",
                text=g["用户占比"].map("{:.1%}".format), textposition="outside")
    fig.add_bar(x=g["客户分层"], y=g["购买贡献占比"], name="购买贡献占比", marker_color=RED,
                text=g["购买贡献占比"].map("{:.1%}".format), textposition="outside")
    fig.update_layout(title="RFM 8 类客户：人数占比 vs 购买贡献占比（资源不应按人数分配）",
                      barmode="group", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=1.12), xaxis_tickangle=-30)
    return fig


def fig_cluster() -> go.Figure:
    p = load("stage7_cluster_profile.csv")
    p = p[p["画像命名"] != "离群批发/代购账号"]  # 离群簇仅 33 人，单独标注不入图
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=p["画像命名"], y=p["用户数"], name="用户数", marker_color=TEAL,
                text=p["用户数"].map("{:,.0f}".format), textposition="outside",
                secondary_y=False)
    fig.add_scatter(x=p["画像命名"], y=p["购买贡献占比"], name="购买贡献占比",
                    mode="lines+markers+text", line=dict(color=RED, width=3),
                    marker=dict(size=10), text=p["购买贡献占比"].map("{:.1%}".format),
                    textposition="top center", secondary_y=True)
    fig.update_layout(title="KMeans 聚类（K=6）：核心购买用户 6.9% 的人贡献 29.8% 购买"
                            "（另有 33 个离群批发/代购账号未入图）",
                      legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="用户数", secondary_y=False)
    fig.update_yaxes(title_text="购买贡献占比", tickformat=".0%", secondary_y=True)
    return fig


def fig_category() -> go.Figure:
    d = load("stage3_top20_category.csv")
    fig = go.Figure(go.Scatter(
        x=d["pv"], y=d["浏览→购买转化率"], mode="markers+text",
        marker=dict(size=d["pv_份额"] * 400 + 8, color=d["浏览→购买转化率"],
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="转化率", tickformat=".1%")),
        text=d["k"].astype(str), textposition="top center", textfont=dict(size=9),
        hovertemplate="类目 %{text}<br>浏览量 %{x:,.0f}<br>转化率 %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(title="Top20 类目：浏览量 vs 转化率（气泡=流量份额；右下=高流量低转化）",
                      xaxis_title="浏览量", yaxis_title="浏览→购买转化率",
                      yaxis_tickformat=".1%")
    return fig


# ------------------------------------------------------------------ 组装
def main() -> None:
    figs = [
        ("流量节律", [fig_dau(), fig_hourly()]),
        ("转化漏斗", [fig_funnel()]),
        ("留存与复购", [fig_cohort(), fig_repurchase()]),
        ("用户分层", [fig_rfm(), fig_cluster()]),
        ("类目效率", [fig_category()]),
    ]
    layout = dict(font=FONT, template="plotly_white", height=430,
                  margin=dict(l=60, r=40, t=70, b=60))
    sections = []
    js_done = False
    for title, fl in figs:
        divs = []
        for f in fl:
            f.update_layout(**layout)
            divs.append(pio.to_html(f, full_html=False,
                                    include_plotlyjs=not js_done, div_id=None))
            js_done = True
        sections.append(f"<section><h2>{title}</h2>{''.join(divs)}</section>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>淘宝用户行为看板 · UserBehavior Dashboard</title>
<style>
  body {{ margin:0; background:#f6f8fa; color:#1a1f24;
    font-family:"Microsoft YaHei","PingFang SC",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 24px 60px; }}
  header {{ background:linear-gradient(135deg,#123a6b,#1f6feb); color:#fff;
    border-radius:14px; padding:26px 30px; margin-bottom:20px; }}
  header h1 {{ margin:0 0 6px; font-size:24px; }}
  header .meta {{ opacity:.9; font-size:13px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
    gap:12px; margin-bottom:20px; }}
  .kpi {{ background:#fff; border:1px solid #e3e8ee; border-radius:12px; padding:14px 16px; }}
  .kpi-t {{ font-size:12px; color:#5b666f; }}
  .kpi-v {{ font-size:22px; font-weight:800; color:#1f6feb; margin:3px 0; }}
  .kpi-s {{ font-size:11.5px; color:#5b666f; }}
  section {{ background:#fff; border:1px solid #e3e8ee; border-radius:12px;
    padding:18px 20px; margin-bottom:18px; }}
  h2 {{ font-size:17px; margin:4px 0 10px; padding-left:10px; border-left:4px solid #1f6feb; }}
  .foot {{ text-align:center; color:#5b666f; font-size:12px; margin-top:18px; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>淘宝 UserBehavior 用户行为看板</h1>
  <div class="meta">2017-11-25 ~ 2017-12-03（9 天，北京时间）· 全量 1 亿行 DuckDB 计算 ·
  由 src/stage10_dashboard.py 自动生成，数据来自 outputs/ 聚合结果表</div>
</header>
{kpi_html()}
{''.join(sections)}
<div class="foot">口径提醒：留存为首日活跃 cohort 口径；复购 = 购买天数 ≥ 2；
周末留存/流量高峰存在日历效应，详见主报告《淘宝用户行为分析报告.html》</div>
</div>
</body>
</html>"""
    out = C.REPORT_DIR / "用户行为看板.html"
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
