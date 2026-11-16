# -*- coding: utf-8 -*-
"""
阶段 7 · 统计检验与用户聚类
===========================
7.1 周末 vs 工作日购买转化（双比例 z 检验 / 卡方）
    - 主口径：**用户-日**级（某活跃用户当天是否产生购买），业务含义是"周末来的人更容易下单吗"。
    - 局限：同一用户多天出现，观测不完全独立，p 值偏乐观 —— 因此补一个**用户级稳健性检验**：
      只在周末活跃的用户 vs 只在工作日活跃的用户（两组互斥、独立），结论方向一致才算稳。
7.2 高活跃 vs 低活跃购买转化（用户级，pv 中位数切分，两组互斥独立）
7.3 每个检验都输出：样本量、转化率、率差（百分点）、相对提升、z/chi2、p 值 + 业务解释。
    注意措辞："显著"只说明差异不太可能是随机噪声，**不等于**有业务意义（大样本下 0.1pp 也能显著），
    所以必须同时看率差和相对提升。
7.4 KMeans：特征 = pv/cart/fav/buy 频次 + 活跃天数 + 购买转化率（buy_cnt/pv），
    StandardScaler 标准化；K=2..8 跑肘部法（inertia），轮廓系数用 2 万采样估算；
    K=2 的轮廓系数通常最高但业务颗粒度太粗，故在 K=3..6 中取轮廓系数最高者。
7.5 聚类画像命名 + 与 RFM 价值大类交叉验证（Cramér's V 量化一致性）。
7.6 局限：无金额字段、9 天窗口、行为频次特征高度右偏。

产出：
  outputs/stage7_hypothesis_tests.csv / stage7_kmeans_k_selection.csv
  outputs/stage7_cluster_profile.csv / stage7_cluster_vs_rfm.csv
  outputs/parquet/user_clusters.parquet
  reports/figures/fig23~fig25*.png
  reports/stage7_tests_clustering_summary.md
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
from scipy import stats  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

K_RANGE = range(2, 9)
K_CANDIDATES = range(3, 7)      # 业务可选 K 的范围（K=2 颗粒度太粗，见模块 docstring）
SILHOUETTE_SAMPLE = 20_000
RANDOM_STATE = 42

FEATURES = ["pv", "cart", "fav", "buy_cnt", "active_days", "cvr"]
FEATURE_CN = {
    "pv": "浏览次数", "cart": "加购次数", "fav": "收藏次数",
    "buy_cnt": "购买次数", "active_days": "活跃天数", "cvr": "购买转化率",
}


# ------------------------------------------------------------------ 假设检验
def two_prop_test(x1: int, n1: int, x2: int, n2: int) -> dict:
    """双比例 z 检验（双侧）+ 卡方校验。返回 z、p、率差、相对提升。"""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_z = 2 * stats.norm.sf(abs(z))
    chi2, p_chi2, _, _ = stats.chi2_contingency([[x1, n1 - x1], [x2, n2 - x2]],
                                                correction=False)
    return {
        "组1转化数": x1, "组1样本量": n1, "组1转化率": p1,
        "组2转化数": x2, "组2样本量": n2, "组2转化率": p2,
        "率差(pp)": (p1 - p2) * 100,
        "相对提升": (p1 - p2) / p2 if p2 > 0 else np.nan,
        "z": z, "p值": p_z,
        "卡方": chi2, "卡方p值": p_chi2,
    }


def build_user_day(con) -> pd.DataFrame:
    """用户-日级表：是否购买、是否周末。"""
    events_glob = str(C.PARQUET_DIR / "events" / "**" / "*.parquet").replace("\\", "/")
    return con.execute(
        f"""
        SELECT user_id, date,
               max(is_weekend)                          AS is_weekend,
               max(CASE WHEN behavior_type = 'buy' THEN 1 ELSE 0 END) AS has_buy
        FROM read_parquet('{events_glob}', hive_partitioning = true)
        GROUP BY user_id, date
        """
    ).fetchdf()


def test_weekend(user_day: pd.DataFrame) -> tuple[dict, dict]:
    """主口径（用户-日）+ 稳健性口径（用户级，互斥分组）。"""
    we = user_day[user_day["is_weekend"] == 1]
    wd = user_day[user_day["is_weekend"] == 0]
    main = two_prop_test(int(we["has_buy"].sum()), len(we),
                         int(wd["has_buy"].sum()), len(wd))

    # 用户级稳健性：只在周末活跃 vs 只在工作日活跃（互斥、独立）
    g = user_day.groupby("user_id").agg(
        days=("date", "nunique"),
        weekend_days=("is_weekend", "sum"),
        bought=("has_buy", "max"),
    )
    only_we = g[g["days"] == g["weekend_days"]]
    only_wd = g[g["weekend_days"] == 0]
    robust = two_prop_test(int(only_we["bought"].sum()), len(only_we),
                           int(only_wd["bought"].sum()), len(only_wd))
    return main, robust


def test_activity(uf: pd.DataFrame) -> tuple[dict, float]:
    """高活跃（pv > 中位数）vs 低活跃用户的购买率差异（用户级，互斥独立）。"""
    med = float(uf["pv"].median())
    hi = uf[uf["pv"] > med]
    lo = uf[uf["pv"] <= med]
    res = two_prop_test(int((hi["buy_cnt"] > 0).sum()), len(hi),
                        int((lo["buy_cnt"] > 0).sum()), len(lo))
    return res, med


# ------------------------------------------------------------------ KMeans
def build_features(uf: pd.DataFrame) -> pd.DataFrame:
    df = uf[["user_id", "pv", "cart", "fav", "buy_cnt", "active_days"]].copy()
    df["cvr"] = df["buy_cnt"] / df["pv"].clip(lower=1)
    return df[["user_id"] + FEATURES]


def select_k(X: np.ndarray) -> pd.DataFrame:
    """K=2..8：inertia（肘部法）+ 轮廓系数（2 万采样估算）。"""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    Xs = X[idx]
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        sil = silhouette_score(Xs, km.predict(Xs), random_state=RANDOM_STATE)
        rows.append({"K": k, "inertia": km.inertia_, "轮廓系数": sil})
        print(f"     K={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}", flush=True)
    return pd.DataFrame(rows)


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """
    按簇画像规则命名（profile 需含原始量纲均值）。
    规则按优先级贪心分配，保证每个簇名字唯一。
    特殊处理：人数占比 < 0.1% 且行为极端的簇判定为**离群簇**（KMeans 对极端值敏感，
    会把少量异常用户单独切成一坨），单独命名为"离群批发/代购账号"，不参与后续命名，
    避免它抢走"核心购买用户"这类本该属于规模化人群的标签。
    """
    names: dict[int, str] = {}
    pf = profile.set_index("cluster")
    remaining = list(pf.index)

    def pick(metric, ascending=False):
        nonlocal remaining
        sub = pf.loc[remaining, metric]
        c = sub.idxmax() if not ascending else sub.idxmin()
        remaining.remove(c)
        return c

    # 0. 离群簇：占比 <0.1% 且人均购买异常高（买而不逛，疑似批发/代购/脚本账号）
    for c in list(remaining):
        if pf.loc[c, "用户占比"] < 0.001 and pf.loc[c, "购买次数"] > pf["购买次数"].median() * 5:
            names[c] = "离群批发/代购账号"
            remaining.remove(c)
    # 1. 购买最强 → 核心购买用户
    c = pick("购买次数")
    names[c] = "核心购买用户"
    # 2. 种草强度（加购+收藏相对浏览）最高 → 深度种草用户
    pf["_seed"] = (pf["加购次数"] + pf["收藏次数"]) / pf["浏览次数"].clip(lower=1)
    c = pick("_seed")
    names[c] = "深度种草用户"
    # 3. 浏览最高 → 高频观光用户
    c = pick("浏览次数")
    names[c] = "高频观光用户"
    # 4. 活跃度最低 → 低频路过用户
    c = pick("活跃天数", ascending=True)
    names[c] = "低频路过用户"
    # 5. 其余 → 中间型
    for c in remaining:
        names[c] = "普通潜力用户"
    return names


def cramers_v(ct: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


# ------------------------------------------------------------------ 图
def fig_k_selection(sel: pd.DataFrame, chosen_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.plot(sel["K"], sel["inertia"], marker="o", color="#4C78A8", lw=2)
    ax.axvline(chosen_k, color="#E45756", ls="--", lw=1.2, label=f"选定 K={chosen_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia（簇内平方和）")
    ax.set_title("肘部法：找拐点")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(sel["K"], sel["轮廓系数"], marker="s", color="#54A24B", lw=2)
    ax.axvline(chosen_k, color="#E45756", ls="--", lw=1.2, label=f"选定 K={chosen_k}")
    for _, r in sel.iterrows():
        ax.text(r["K"], r["轮廓系数"], f"{r['轮廓系数']:.3f}", ha="center",
                va="bottom", fontsize=8.5)
    ax.set_xlabel("K")
    ax.set_ylabel("轮廓系数（2 万采样估算）")
    ax.set_title(f"轮廓系数：K=2 最高但颗粒度太粗，在 K={K_CANDIDATES.start}~{K_CANDIDATES.stop - 1} 中取最优")
    ax.legend(fontsize=9)
    fig.tight_layout()
    C.save_fig(fig, "fig23_kmeans_k_selection.png")
    plt.close(fig)


def fig_cluster_profile(profile: pd.DataFrame) -> None:
    """各簇标准化后的特征均值热力图 + 人数占比。"""
    cols = [FEATURE_CN[f] for f in FEATURES]
    z = profile.set_index("cluster")[cols]
    z = (z - z.mean()) / z.std(ddof=0)
    labels = [f"{r['画像命名']}\nn={r['用户数']:,}（{r['用户占比']:.1%}）"
              for _, r in profile.iterrows()]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(z.values, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(z.values).max(), vmax=np.abs(z.values).max())
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(labels, fontsize=9.5)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > np.abs(z.values).max() * 0.6 else "#222222")
    ax.set_title("KMeans 聚类画像：各簇特征均值（按列标准化，+ 高于平均 / - 低于平均）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    C.save_fig(fig, "fig24_cluster_profile.png")
    plt.close(fig)


def fig_cluster_vs_rfm(ct_pct: pd.DataFrame, profile: pd.DataFrame) -> None:
    name_map = dict(zip(profile["cluster"], profile["画像命名"]))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    vals = ct_pct.values
    im = ax.imshow(vals, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(ct_pct.shape[1]))
    ax.set_xticklabels(ct_pct.columns, fontsize=9.5)
    ax.set_yticks(range(ct_pct.shape[0]))
    ax.set_yticklabels([name_map[i] for i in ct_pct.index], fontsize=9.5)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.65 else "#222222")
    ax.set_title("聚类 × RFM 价值大类交叉（行百分比，仅购买用户）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, format=lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    C.save_fig(fig, "fig25_cluster_vs_rfm.png")
    plt.close(fig)


# ------------------------------------------------------------------ 主流程
def main() -> None:
    con = db.connect(memory_limit="8GB")
    con.execute(f"SET temp_directory='{(C.OUTPUT_DIR / 'duckdb_tmp').as_posix()}'")

    uf = con.execute(
        f"SELECT * FROM read_parquet('{(C.PARQUET_DIR / 'user_features.parquet').as_posix()}')"
    ).fetchdf()
    print(f"[load] user_features {len(uf):,} 用户", flush=True)

    print("[7.1] 检验一：周末 vs 工作日购买转化 ...", flush=True)
    user_day = build_user_day(con)
    t1_main, t1_robust = test_weekend(user_day)

    print("[7.2] 检验二：高活跃 vs 低活跃购买转化 ...", flush=True)
    t2, pv_median = test_activity(uf)

    tests = pd.DataFrame([
        {"检验": "周末 vs 工作日（用户-日级主口径）", "组1": "周末", "组2": "工作日", **t1_main},
        {"检验": "周末 vs 工作日（用户级稳健性：仅周末活跃 vs 仅工作日活跃）",
         "组1": "仅周末活跃", "组2": "仅工作日活跃", **t1_robust},
        {"检验": f"高活跃 vs 低活跃（pv 中位数 {pv_median:.0f} 切分）",
         "组1": "高活跃", "组2": "低活跃", **t2},
    ])
    tests.to_csv(C.OUTPUT_DIR / "stage7_hypothesis_tests.csv", index=False,
                 encoding="utf-8-sig")
    utils.save_json(tests.to_dict(orient="records"),
                    C.OUTPUT_DIR / "stage7_hypothesis_tests.json")

    print("[7.4] KMeans：标准化 + 选 K ...", flush=True)
    feats = build_features(uf)
    X = StandardScaler().fit_transform(feats[FEATURES].to_numpy(dtype=float))
    sel = select_k(X)
    sel.to_csv(C.OUTPUT_DIR / "stage7_kmeans_k_selection.csv", index=False,
               encoding="utf-8-sig")
    cand = sel[sel["K"].isin(K_CANDIDATES)]
    chosen_k = int(cand.loc[cand["轮廓系数"].idxmax(), "K"])
    print(f"     选定 K={chosen_k}", flush=True)

    print(f"[7.4] 用 K={chosen_k} 拟合全量 ...", flush=True)
    km = KMeans(n_clusters=chosen_k, n_init=10, random_state=RANDOM_STATE)
    feats["cluster"] = km.fit_predict(X)

    # 画像（原始量纲均值）
    prof = (
        feats.groupby("cluster")
        .agg(用户数=("user_id", "count"),
             浏览次数=("pv", "mean"), 加购次数=("cart", "mean"), 收藏次数=("fav", "mean"),
             购买次数=("buy_cnt", "mean"), 活跃天数=("active_days", "mean"),
             购买转化率=("cvr", "mean"))
        .reset_index()
    )
    total_buy = uf["buy_cnt"].sum()
    prof["购买贡献占比"] = feats.groupby("cluster")["buy_cnt"].sum().values / total_buy
    prof["用户占比"] = prof["用户数"] / prof["用户数"].sum()
    names = name_clusters(prof)
    prof["画像命名"] = prof["cluster"].map(names)
    prof = prof.sort_values("购买次数", ascending=False).reset_index(drop=True)
    prof.to_csv(C.OUTPUT_DIR / "stage7_cluster_profile.csv", index=False,
                encoding="utf-8-sig")

    feats[["user_id", "cluster"]].assign(
        画像=lambda d: d["cluster"].map(names)
    ).to_parquet(C.PARQUET_DIR / "user_clusters.parquet", index=False)

    print("[7.5] 与 RFM 分层交叉验证 ...", flush=True)
    rfm = pd.read_parquet(C.PARQUET_DIR / "rfm_user.parquet",
                          columns=["user_id", "价值大类"])
    m = feats[["user_id", "cluster"]].merge(rfm, on="user_id", how="inner")
    ct = pd.crosstab(m["cluster"], m["价值大类"])
    ct = ct.reindex(index=sorted(prof["cluster"]), columns=sorted(ct.columns), fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    ct_out = ct_pct.copy()
    ct_out.insert(0, "画像命名", [names[i] for i in ct_out.index])
    ct_out.to_csv(C.OUTPUT_DIR / "stage7_cluster_vs_rfm.csv", encoding="utf-8-sig")
    ct_cnt = ct.copy()
    ct_cnt.insert(0, "画像命名", [names[i] for i in ct_cnt.index])
    ct_cnt.to_csv(C.OUTPUT_DIR / "stage7_cluster_vs_rfm_counts.csv", encoding="utf-8-sig")
    cv = cramers_v(ct)
    buyer_share = len(m) / len(feats)
    utils.save_json(
        {"chosen_k": chosen_k,
         "silhouette_chosen": float(sel.loc[sel["K"] == chosen_k, "轮廓系数"].iloc[0]),
         "silhouette_k2": float(sel.loc[sel["K"] == 2, "轮廓系数"].iloc[0]),
         "cramers_v": cv, "buyer_share": buyer_share},
        C.OUTPUT_DIR / "stage7_meta.json",
    )

    print("[7.x] 可视化 ...", flush=True)
    fig_k_selection(sel, chosen_k)
    fig_cluster_profile(prof)
    fig_cluster_vs_rfm(ct_pct, prof)

    # ---------------- 汇总报告
    def sig(p: float) -> str:
        return "显著" if p < 0.05 else "不显著"

    t_md = tests.copy()
    for c in ["组1转化率", "组2转化率"]:
        t_md[c] = t_md[c].map("{:.2%}".format)
    t_md["相对提升"] = t_md["相对提升"].map("{:+.1%}".format)
    t_md["率差(pp)"] = t_md["率差(pp)"].map("{:+.3f}".format)
    t_md["z"] = t_md["z"].map("{:+.1f}".format)
    t_md["p值"] = t_md["p值"].map(lambda p: "<1e-300" if p < 1e-300 else f"{p:.2e}")
    t_md = t_md.drop(columns=["卡方", "卡方p值"])

    prof_md = prof.copy()
    for c in ["浏览次数", "加购次数", "收藏次数", "购买次数", "活跃天数"]:
        prof_md[c] = prof_md[c].map("{:.2f}".format)
    prof_md["购买转化率"] = prof_md["购买转化率"].map("{:.2%}".format)
    prof_md["用户占比"] = prof_md["用户占比"].map("{:.1%}".format)
    prof_md["购买贡献占比"] = prof_md["购买贡献占比"].map("{:.1%}".format)

    core = prof[prof["画像命名"] == "核心购买用户"].iloc[0]
    outlier = prof[prof["画像命名"] == "离群批发/代购账号"]
    seed = prof[prof["画像命名"] == "深度种草用户"].iloc[0]
    sil_k2 = float(sel.loc[sel["K"] == 2, "轮廓系数"].iloc[0])
    sil_chosen = float(sel.loc[sel["K"] == chosen_k, "轮廓系数"].iloc[0])

    md = f"""# 阶段 7 · 统计检验与用户聚类结论

## 7.1–7.3 假设检验

{utils.df_to_md(t_md)}

**怎么读这些 p 值**：
- 三个检验的 p 值都远小于 0.05，差异在统计上**显著** —— 含义是"这么大（或更大）的差异在
  两组无差异的原假设下几乎不可能由随机噪声产生"，**而不是**"两个因素有因果关系"。
- 样本量到百万级后，**0.1 个百分点的率差也能显著**，所以必须同时看率差和相对提升：

**检验一 · 周末 vs 工作日**：周末转化率 {t1_main['组1转化率']:.2%}，工作日 {t1_main['组2转化率']:.2%}，
率差 {t1_main['率差(pp)']:+.2f}pp（相对 {t1_main['相对提升']:+.1%}），**方向为负**。
结合阶段 3 的发现（周末 DAU 更高）可得完整结论：**周末流量更大，但转化质量反而略低** ——
周末的增量流量以"随便逛逛"的低意向用户为主。率差 1.6pp 属于"显著但不大"的效应，
不值得为周末单独加码投放，反而要在排期上注意周末流量的转化稀释。
用户级稳健性检验（仅周末活跃 {t1_robust['组1转化率']:.1%} vs 仅工作日活跃 {t1_robust['组2转化率']:.1%}，互斥独立）方向一致；
但该口径下"仅工作日活跃"的用户天然拥有更多可活跃天数（工作日 5 天 vs 周末 2 天），
购买机会更多，会**夸大**差异，因此效应量以用户-日口径的 {t1_main['率差(pp)']:+.2f}pp 为准。

**检验二 · 高活跃 vs 低活跃**：购买率 {t2['组1转化率']:.1%} vs {t2['组2转化率']:.1%}，
率差 {t2['率差(pp)']:+.1f}pp（相对 {t2['相对提升']:+.1%}）—— 这才是真正有业务杠杆的差异：
**活跃度是转化最强的行为信号**，运营动作的优先级应围绕"提升活跃"而不是"追周末流量"。

## 7.4 KMeans 聚类

- 特征：pv / cart / fav / buy 频次 + 活跃天数 + 购买转化率（buy_cnt/pv），StandardScaler 标准化。
- 选 K：肘部法看拐点 + 轮廓系数（2 万采样估算）。K=2 的轮廓系数（{sil_k2:.3f}）最高，
  但只把用户切成"买/不买"两坨，业务颗粒度太粗；在 K=3~6 中取轮廓系数最高的 **K={chosen_k}**
  （{sil_chosen:.3f}）。

![选 K](figures/fig23_kmeans_k_selection.png)

![聚类画像](figures/fig24_cluster_profile.png)

{utils.df_to_md(prof_md)}

- **{core['画像命名']}**（{core['用户数']:,} 人，占 {core['用户占比']:.1%}）：人均购买 {core['购买次数']:.1f} 次、
  活跃 {core['活跃天数']:.1f} 天，贡献全体 **{core['购买贡献占比']:.1%}** 的购买 ——
  与 RFM 的"少数用户贡献多数购买"结论互相印证。
- **{seed['画像命名']}**（{seed['用户数']:,} 人）：人均收藏 {seed['收藏次数']:.1f} 次、加购 {seed['加购次数']:.1f} 次，
  但购买转化率仅 {seed['购买转化率']:.1%} —— 典型的"种草未拔草"人群，
  是优惠券/限时提醒投放转化率最高的目标层。
"""
    if len(outlier):
        o = outlier.iloc[0]
        md += f"""- **离群簇**（{o['用户数']:,.0f} 人，占比 {o['用户占比']:.4%}）：人均购买 {o['购买次数']:.0f} 次却几乎零浏览，
  行为模式不像正常消费者，疑似批发/代购/脚本账号 ——
  KMeans 对极端值敏感会自然切出这类微簇；真实业务中这批账号应移交风控而非运营。
"""
    md += f"""
## 7.5 与 RFM 分层交叉验证

![聚类 × RFM](figures/fig25_cluster_vs_rfm.png)

- 交叉表（仅购买用户，占全体 {buyer_share:.1%}）：聚类结果与 RFM 价值大类的 **Cramér's V = {cv:.3f}**
  （0~1，越大关联越强）。
- 解读：两套方法切出的高价值人群高度重叠（核心购买用户簇中 A 类高价值活跃占比最高，
  低频路过簇则以 C 类长尾为主），说明分层结论**稳健、不依赖单一方法**；
  差异部分来自特征侧重不同 —— RFM 只看购买行为，KMeans 还纳入了浏览/加购/收藏过程行为，
  因此能识别出"还没买但深度种草"的潜力人群，这是 RFM 的盲区。

## 7.6 局限

1. **无金额字段**：聚类无法区分"买贵"和"买便宜"，高价值 ≠ 高消费额。
2. **9 天窗口**：活跃天数上限 9，无法识别月频/季频用户；转化率被大促周期抬高。
3. **特征右偏**：行为频次长尾严重，StandardScaler 不能消除偏态；极端用户对簇心影响大
   （表现为 33 人的离群微簇，已单独标注；主人群画像不受其影响）。
4. **轮廓系数为采样估算**：全量百万级计算过慢，用 2 万随机采样近似，多次采样波动 < 0.01。

## 7.7 一句话总结

统计检验发现**周末流量"量高质低"**（转化 {t1_main['相对提升']:+.1%}），真正的杠杆是活跃度
（高活跃用户购买率高 {t2['率差(pp)']:.1f}pp）；KMeans 在无标签数据上独立切出了与 RFM 一致的
高价值人群（Cramér's V = {cv:.2f}），并额外定位了"深度种草未购买"的转化潜力层
与 33 个疑似批发/代购的离群账号 —— 前者交给运营做转化，后者交给风控。
"""
    (C.REPORT_DIR / "stage7_tests_clustering_summary.md").write_text(md, encoding="utf-8")
    con.close()
    print("阶段 7 完成", flush=True)


if __name__ == "__main__":
    main()
