# -*- coding: utf-8 -*-
"""
阶段 5 · RFM 用户价值分层
========================
F 用购买天数（比次数抗单次多笔/刷单），M 用购买的不同商品件数作代理指标（数据无金额），
打分用 qcut 五分位，再按高/低二分归为 8 类并合并成 3 大类，附购买贡献度验证。

产出：
  outputs/stage5_rfm_user.parquet / stage5_rfm_layer.csv / stage5_rfm_3class.csv
  reports/figures/fig16~fig18*.png
  reports/stage5_rfm_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import db  # noqa: E402
import utils  # noqa: E402

C.setup_chinese_font()
import matplotlib.pyplot as plt  # noqa: E402

# 标准 8 类客户模型：R/F/M 是否高于 4 分（即进入前 40%）
LAYER_MAP = {
    (1, 1, 1): "重要价值客户",
    (1, 0, 1): "重要发展客户",
    (0, 1, 1): "重要保持客户",
    (0, 0, 1): "重要挽留客户",
    (1, 1, 0): "一般价值客户",
    (1, 0, 0): "一般发展客户",
    (0, 1, 0): "一般保持客户",
    (0, 0, 0): "一般挽留客户",
}

PLAYBOOK = {
    "重要价值客户": "VIP 专属客服 + 新品优先购 + 会员等级权益，目标：维持频次、提升客单",
    "重要保持客户": "沉默高价值，做「专属回归礼 + 老客价」触达，目标：唤醒复购",
    "重要发展客户": "购买频次不足但金额高，做「周期购/会员包」引导，目标：把一次性变常态",
    "重要挽留客户": "高价值但久未购买，人工外呼 + 大额回归券，最后一道挽留",
    "一般价值客户": "保持现有触达节奏，用满减凑单提升客单",
    "一般发展客户": "低价值新客，用首单/二单券培养习惯",
    "一般保持客户": "中低价值低频，全自动 Push 触达，控制成本",
    "一般挽留客户": "基本流失，仅做低成本召回（大促短信），不值得单独投放",
}


def build_rfm(con) -> pd.DataFrame:
    uf = C.PARQUET_DIR / "user_features.parquet"
    df = con.execute(
        f"""
        SELECT
            user_id,
            buy_cnt                                        AS 购买次数,
            buy_days                                       AS F,
            buy_items                                      AS M,
            date_diff('day', CAST(last_buy_time AS DATE), DATE '{C.DATE_END}') AS R,
            active_days
        FROM read_parquet('{uf.as_posix()}')
        WHERE buy_cnt > 0
        """
    ).fetchdf()
    return df


def qscore(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    五分位打分（1–5）。
    先 rank(method='first') 再 qcut：既保证 5 档等分，又避免大量并列值导致 qcut 边界重复。
    """
    r = s.rank(method="first")
    q = pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return q if higher_is_better else (6 - q)


def build_layers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["R_score"] = qscore(df["R"], higher_is_better=False)   # R 越小越好
    df["F_score"] = qscore(df["F"], higher_is_better=True)
    df["M_score"] = qscore(df["M"], higher_is_better=True)
    df["R_high"] = (df["R_score"] >= 4).astype(int)
    df["F_high"] = (df["F_score"] >= 4).astype(int)
    df["M_high"] = (df["M_score"] >= 4).astype(int)
    df["RFM"] = df["R_high"].astype(str) + df["F_high"].astype(str) + df["M_high"].astype(str)
    df["客户分层"] = [
        LAYER_MAP[(r, f, m)] for r, f, m in zip(df["R_high"], df["F_high"], df["M_high"])
    ]
    # 3 大类归并（按业务动作划分）
    def _cls(row):
        if row["F_high"] == 1 or row["M_high"] == 1:
            return "A 高价值活跃" if row["R_high"] == 1 else "B 高价值沉默(流失预警)"
        return "C 一般/长尾"

    df["价值大类"] = df.apply(_cls, axis=1)
    return df


def layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_buy = df["购买次数"].sum()
    g = (
        df.groupby("客户分层", observed=False)
        .agg(
            用户数=("user_id", "count"),
            人均购买次数=("购买次数", "mean"),
            购买次数合计=("购买次数", "sum"),
            人均购买天数=("F", "mean"),
            人均购买商品数=("M", "mean"),
            平均R=("R", "mean"),
        )
        .reset_index()
    )
    g["用户占比"] = g["用户数"] / g["用户数"].sum()
    g["购买贡献占比"] = g["购买次数合计"] / total_buy
    order = list(LAYER_MAP.values())
    g["_o"] = g["客户分层"].map({k: i for i, k in enumerate(order)})
    g = g.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    return g


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_buy = df["购买次数"].sum()
    g = (
        df.groupby("价值大类", observed=False)
        .agg(用户数=("user_id", "count"), 购买次数合计=("购买次数", "sum"),
             人均购买次数=("购买次数", "mean"))
        .reset_index()
        .sort_values("价值大类")
    )
    g["用户占比"] = g["用户数"] / g["用户数"].sum()
    g["购买贡献占比"] = g["购买次数合计"] / total_buy
    return g


# ------------------------------------------------------------------ 图
def fig_facet_scatter(df: pd.DataFrame) -> None:
    layers = list(LAYER_MAP.values())
    sample = df.sample(min(len(df), 200_000), random_state=42)
    # R / F 都是 0–8 的小整数，直接散点会画成网格；加抖动还原成"云"
    rng = np.random.default_rng(0)

    def jitter(s: pd.Series, scale: float = 0.22) -> np.ndarray:
        return s.to_numpy(dtype=float) + rng.uniform(-scale, scale, len(s))

    jr, jf = jitter(sample["R"]), jitter(sample["F"])
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.6), sharex=True, sharey=True)
    for ax, layer in zip(axes.ravel(), layers):
        mask = (sample["客户分层"] == layer).to_numpy()
        ax.scatter(jr[~mask], jf[~mask], s=3, color="#DDDDDD", alpha=0.35, linewidths=0)
        ax.scatter(jr[mask], jf[mask], s=5, color="#E45756", alpha=0.55, linewidths=0)
        ax.set_title(f"{layer}\nn={int(mask.sum()):,}", fontsize=10)
        ax.set_xlim(-0.8, 8.8)
        ax.set_ylim(0, min(9.5, df["F"].max()))
    for ax in axes[1]:
        ax.set_xlabel("R（距末次购买天数，越小越近）")
    for ax in axes[:, 0]:
        ax.set_ylabel("F（购买天数）")
    fig.suptitle("R–F 分面散点：8 类客户的分布位置（红点=该类客户，灰点=其他；已加抖动防重叠）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "fig16_rfm_scatter_facet.png")
    plt.close(fig)


def fig_share_vs_contribution(g: pd.DataFrame, cls: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    ax = axes[0]
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["用户占比"] * 100, width=0.4, color="#9ECAE1", label="用户占比")
    ax.bar(x + 0.2, g["购买贡献占比"] * 100, width=0.4, color="#E45756", label="购买贡献占比")
    for i in range(len(g)):
        ax.text(i - 0.2, g["用户占比"].iloc[i] * 100, f"{g['用户占比'].iloc[i]:.1%}", ha="center",
                va="bottom", fontsize=8)
        ax.text(i + 0.2, g["购买贡献占比"].iloc[i] * 100, f"{g['购买贡献占比'].iloc[i]:.1%}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(g["客户分层"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("占比 (%)")
    ax.set_title("8 类客户：人数占比 vs 购买贡献占比")
    ax.legend(fontsize=9)

    ax = axes[1]
    x = np.arange(len(cls))
    ax.bar(x - 0.2, cls["用户占比"] * 100, width=0.4, color="#9ECAE1", label="用户占比")
    ax.bar(x + 0.2, cls["购买贡献占比"] * 100, width=0.4, color="#54A24B", label="购买贡献占比")
    for i in range(len(cls)):
        ax.text(i - 0.2, cls["用户占比"].iloc[i] * 100, f"{cls['用户占比'].iloc[i]:.1%}", ha="center",
                va="bottom", fontsize=9)
        ax.text(i + 0.2, cls["购买贡献占比"].iloc[i] * 100, f"{cls['购买贡献占比'].iloc[i]:.1%}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cls["价值大类"], fontsize=9)
    ax.set_ylabel("占比 (%)")
    ax.set_title("归并为 3 大类后的价值分布")
    ax.legend(fontsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig17_rfm_share_contribution.png")
    plt.close(fig)


def fig_rfm_heatmap(df: pd.DataFrame) -> None:
    """R 分档 × F 分档 的人均购买次数热力图，验证打分口径单调性。"""
    piv_n = df.pivot_table(index="R_score", columns="F_score", values="user_id", aggfunc="count")
    piv_b = df.pivot_table(index="R_score", columns="F_score", values="购买次数", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, piv, title, fmt in [
        (axes[0], piv_n, "R 档 × F 档：用户数", "{:,.0f}"),
        (axes[1], piv_b, "R 档 × F 档：人均购买次数", "{:.1f}"),
    ]:
        im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"F={c}" for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"R={i}" for i in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, "-" if np.isnan(v) else fmt.format(v), ha="center", va="center",
                        fontsize=9,
                        color="white" if not np.isnan(v) and v > np.nanmax(piv.values) * 0.6 else "#222222")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("RFM 打分口径校验：R 越小（越近）、F 越高，人均购买次数单调上升", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "fig18_rfm_score_validation.png")
    plt.close(fig)


def main() -> None:
    con = db.connect(memory_limit="8GB")
    db.attach_parquet_layer(con)

    print("[5.1-5.4] 计算 R / F / M 与五分位打分 ...", flush=True)
    raw = build_rfm(con)
    df = build_layers(raw)
    df.to_parquet(C.PARQUET_DIR / "rfm_user.parquet", index=False)
    n_buy_users = len(df)
    print(f"     有购买行为的用户 {n_buy_users:,}", flush=True)

    print("[5.5] 归并 8 类 / 3 大类 ...", flush=True)
    g = layer_summary(df)
    cls = class_summary(df)
    g.to_csv(C.OUTPUT_DIR / "stage5_rfm_layer.csv", index=False, encoding="utf-8-sig")
    cls.to_csv(C.OUTPUT_DIR / "stage5_rfm_3class.csv", index=False, encoding="utf-8-sig")
    print(utils.df_to_md(g), flush=True)

    print("[5.7] 可视化 ...", flush=True)
    fig_facet_scatter(df)
    fig_share_vs_contribution(g, cls)
    fig_rfm_heatmap(df)

    # 与原口径（均值 0/1）的对比
    old_R = (df["R"] <= df["R"].mean()).astype(int)
    old_F = (df["F"] >= df["F"].mean()).astype(int)
    old_M = (df["M"] >= df["M"].mean()).astype(int)
    old_n = len(set(zip(old_R, old_F, old_M)))
    new_n = len(set(zip(df["R_score"], df["F_score"], df["M_score"])))

    top = g.sort_values("购买贡献占比", ascending=False)
    best = g.loc[g["客户分层"] == "重要价值客户"].iloc[0] if (g["客户分层"] == "重要价值客户").any() else top.iloc[0]

    md = f"""# 阶段 5 · RFM 用户价值分层结论

## 5.1–5.3 指标口径

| 指标 | 定义 | 说明 |
|---|---|---|
| R | 末次购买日距数据集末端（{C.DATE_END}）的天数 | 全部落在 0–8 天 |
| F | **购买天数**（一个用户有多少天产生过购买） | 比"购买次数"抗单次多笔与刷单 |
| M | **购买的不同商品件数** | ⚠️ 数据集**没有金额字段**，M 是**代理指标**，不能等同于消费金额 |

**窗口局限（必须写在报告里）**：数据集只有 9 天，R 的取值范围被压缩到 0–8，
无法区分"沉默 1 个月的流失用户"和"沉默 1 周的老客"。真实业务中 R 的判别力需要至少 90 天窗口。

## 5.4 打分方式：qcut 五分位（替代均值 0/1）

- 方法：先 `rank(method='first')` 再 `qcut(…, 5)` 分五档。
  直接 qcut 会因为大量并列值（例如 F=1 的用户极多）导致分箱边界重复、档位不足，先 rank 可保证严格等分。
- R 越小越好 → 反向赋分；F / M 越大越好 → 正向赋分。
- 与原口径对比：均值 0/1 二分类最多只能区分 **{old_n}** 种组合，
  五分位方案可区分 **{new_n}** 种组合（5×5×5=125 理论上限），
  且均值切分会被极端值拉偏（F 的均值远高于中位数，导致"高价值"标签过度集中）。

![打分校验](figures/fig18_rfm_score_validation.png)

## 5.5 分层结果（8 类客户模型）

分层规则：R_score / F_score / M_score ≥ 4 记为该维度"高"，组合成 8 类。

{utils.df_to_md(g)}

![分面散点](figures/fig16_rfm_scatter_facet.png)

## 5.6 分层验证：少数人贡献多数购买

{utils.df_to_md(cls)}

![占比 vs 贡献](figures/fig17_rfm_share_contribution.png)

- 有购买行为的用户共 **{n_buy_users:,}** 人。
- **重要价值客户** 人数占比 {best['用户占比']:.1%}，却贡献了 **{best['购买贡献占比']:.1%}** 的购买次数，
  人均购买 {best['人均购买次数']:.1f} 次 —— 是全体购买用户的 {best['人均购买次数'] / df['购买次数'].mean():.1f} 倍。
- 高价值活跃 + 高价值沉默（A + B 类）合计人数占比 {cls[cls['价值大类'].str[0].isin(['A','B'])]['用户占比'].sum():.1%}，
  贡献购买 {cls[cls['价值大类'].str[0].isin(['A','B'])]['购买贡献占比'].sum():.1%}。
- 一般/长尾（C 类）人数占比 {cls[cls['价值大类'] == 'C 一般/长尾']['用户占比'].iloc[0]:.1%}，
  仅贡献 {cls[cls['价值大类'] == 'C 一般/长尾']['购买贡献占比'].iloc[0]:.1%}。
  → **运营资源绝不应按人数平均分配。**

## 5.8 分层运营动作

| 客户分层 | 用户数 | 人数占比 | 建议动作 |
|---|---|---|---|
"""
    for _, r in g.iterrows():
        md += f"| {r['客户分层']} | {r['用户数']:,} | {r['用户占比']:.1%} | {PLAYBOOK[r['客户分层']]} |\n"

    md += f"""
## 5.9 一句话总结

把用户按 RFM 五分位切分后，"重要价值客户"以 {best['用户占比']:.1%} 的人数贡献了 {best['购买贡献占比']:.1%} 的购买，
而"一般挽留客户"以 {g.loc[g['客户分层'] == '一般挽留客户', '用户占比'].iloc[0]:.1%} 的人数只贡献了 {g.loc[g['客户分层'] == '一般挽留客户', '购买贡献占比'].iloc[0]:.1%} ——
分层有效，且直接给出了"把预算花在谁身上"的答案。
"""
    (C.REPORT_DIR / "stage5_rfm_summary.md").write_text(md, encoding="utf-8")
    con.close()
    print("阶段 5 完成", flush=True)


if __name__ == "__main__":
    main()
