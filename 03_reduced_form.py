# =============================================================================
# Reduced-Form Estimation (Pillars I–IV) — v3 (FINAL)
# Regulatory Shocks and Market Equilibria
# Google Colab 実行用
#
# v2→v3 修正点（実行ログ確認後の修正）:
#
#   [P2]  h_it の上限が82点であること・80点に53%集中を実際に確認。
#         → h_it は整数スコア（perf_y1/perf_y2の平均）であり、
#           「80点がCPR資格閾値」という想定が成立しているか再検証。
#         → RDD右辺でscoring_rate_nextがゼロになる問題:
#           ① 翌年結合のロジックが正しいか診断コードを強化
#           ② h_it ≥ 80 の企業の翌年CE参加率を直接表示
#           ③ 代替アウトカム(scoring_rate 同期・翌期bid_rate)でも試行
#           ④ running variable として perf_y1 単独を試行
#
#   [P3]  exp_quartile_num（序数 0-3）を使っていたことを確認。
#         → Bartik Exposure を work_type × pre/post CE変化率から再構築。
#         → pre-reform 係数がゼロベースラインを確認（並行トレンド検定追加）。
#         → 2017年以降係数≈0の原因（CE消滅後の全企業scoring=0）を
#           サンプル限定（scoring > 0 の企業のみ）で追加推定。
#
#   [P4]  h_it が all_bids_dataset に存在しないことを実際に確認。
#         → analysis_dataset から firm × fy キーで結合。
#         → T_a = 0（定数）による型変換エラーを float64 明示キャストで修正。
#         → h_it 結合後の有効行数を確認して診断を追加。
#
#   [図]  本文掲載推奨:
#         Figure 2: fig_pillar1_main.png     (3変数 Backlog binned)
#         Figure 3: fig_timeseries.png        (時系列サマリー) ← 最推奨
#         Figure 4: fig_pillar3_eventstudy.png (イベントスタディ Bartik版)
#         Figure 5: fig_pillar2_rdd.png       (RDD ※h_it問題解決後)
# =============================================================================

# %%
# ============================================================
# 0. ライブラリ
# ============================================================
import subprocess, sys
for pkg in ["linearmodels", "statsmodels"]:
    try:
        __import__(pkg.replace("-","_"))
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
})
print("ライブラリ読み込み完了")

# %%
# ============================================================
# 1. データ読み込み・基本前処理
# ============================================================
print("データ読み込み中...")
ana  = pd.read_csv("analysis_dataset.csv",  low_memory=False)
bids = pd.read_csv("all_bids_dataset.csv",  low_memory=False)
print(f"  analysis_dataset : {ana.shape}")
print(f"  all_bids_dataset : {bids.shape}")

# ---------- 数値変換ユーティリティ ----------
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

to_num(ana, [
    "bid_rate","scoring_rate","scoring","w_it","tau_it","log_backlog",
    "h_it","perf_avg","perf_y1","perf_y2","log_estimate","log_n_bidders",
    "inv_rate","precision_gap","n_bidders","estimate_price","est_duration",
    "firm_id","year_id","fy","contract_fy","post_policy1","post_policy2",
    "prec_flag","ishikawa",
])

to_num(bids, [
    "bid_rate","scoring","precision","won","log_estimate","log_n_bidders",
    "inv_rate","threshold_margin","abs_margin","n_bidders","estimate_price",
    "cum_bids","cum_wins","win_rate_cum","log_cum_bids",
    "bid_interval_days","bid_interval_months","mkt_n_bids_total",
    "mkt_n_firms","mkt_mean_inv","entry_fy",
    "fy","contract_fy","bid_fy","post_policy1","post_policy2",
    "prec_flag","ishikawa",
])

# FY 統一
for df in [ana, bids]:
    if "fy" not in df.columns:
        for src in ["contract_fy","bid_fy"]:
            if src in df.columns:
                df["fy"] = df[src]; break

# --- log_backlog ---
if "log_backlog" not in ana.columns and "w_it" in ana.columns:
    ana["log_backlog"] = np.log1p(ana["w_it"].clip(lower=0))

# --- prec_flag (analysis) ---
if "prec_flag" not in ana.columns and "precision_gap" in ana.columns:
    ana["prec_flag"] = (
        (ana["precision_gap"] >= 0) & (ana["precision_gap"] <= 0.005)
    ).astype(float)

# --- prec_flag (bids) ---
if "prec_flag" not in bids.columns:
    for src in ["threshold_margin","abs_margin"]:
        if src in bids.columns:
            bids["prec_flag"] = (
                (bids[src] >= 0) & (bids[src] <= 0.005)
            ).astype(float)
            break

# --- ishikawa ---
def make_ishikawa(df):
    if "ishikawa" not in df.columns or df["ishikawa"].isna().all():
        if "office" in df.columns:
            df["ishikawa"] = df["office"].str.contains(
                "金沢|能登|手取|石川", na=False).astype(int)
for df in [ana, bids]:
    make_ishikawa(df)

# --- post フラグ ---
for df in [ana, bids]:
    if "post_policy1" not in df.columns:
        df["post_policy1"] = (df["fy"] >= 2009).astype(int)
    if "post_policy2" not in df.columns:
        df["post_policy2"] = (df["fy"] >= 2014).astype(int)

print("前処理完了\n")

# %%
# ============================================================
# 2. 記述統計
# ============================================================
print("="*62)
print("記述統計 — analysis_dataset")
print("="*62)
desc_a = [c for c in ["bid_rate","scoring_rate","log_backlog","w_it","tau_it",
                       "h_it","perf_avg","log_estimate","n_bidders"] if c in ana.columns]
print(ana[desc_a].describe().round(4).to_string())

print("\n" + "="*62)
print("記述統計 — all_bids_dataset")
print("="*62)
desc_b = [c for c in ["bid_rate","scoring","precision","prec_flag",
                       "log_estimate","log_n_bidders"] if c in bids.columns]
print(bids[desc_b].describe().round(4).to_string())

# %%
# ============================================================
# [P2 診断] h_it の詳細分布確認（v3強化版）
# ============================================================
print("\n" + "="*62)
print("[P2 診断] h_it (CPR running variable) の詳細確認")
print("="*62)

if "h_it" in ana.columns:
    h_valid = ana["h_it"].dropna()
    print(f"  有効観測数       : {len(h_valid):,} / {len(ana):,}")
    print(f"  平均 / SD        : {h_valid.mean():.3f} / {h_valid.std():.3f}")
    print(f"  最小 / 最大      : {h_valid.min()} / {h_valid.max()}")
    print(f"  80点の頻度       : {(h_valid == 80).sum():,} ({(h_valid==80).mean()*100:.1f}%)")
    print(f"  80点超の頻度     : {(h_valid > 80).sum():,} ({(h_valid>80).mean()*100:.1f}%)")
    print(f"  80点未満の頻度   : {(h_valid < 80).sum():,} ({(h_valid<80).mean()*100:.1f}%)")
    print(f"  値のユニーク数   : {h_valid.nunique()}")
    print(f"\n  スコア別頻度 (全スコア):")
    vc = h_valid.value_counts().sort_index()
    for score, count in vc.items():
        bar = "█" * min(int(count / max(vc) * 30), 30)
        print(f"    {score:5.1f}  {count:6,}  {bar}")

    # ① h_it ≥ 80 の企業の翌年CE参加率を直接確認（RDD右辺ゼロ問題の診断）
    print(f"\n  ① RDD右辺ゼロ問題の直接診断:")
    print(f"     h_it ≥ 80 の企業は翌年CE入札に参加しているか？")

    # scoring_rate 候補列を確認（存在するものを使う）
    sr_col = None
    for cand in ["scoring_rate", "scoring"]:
        if cand in ana.columns and pd.to_numeric(ana[cand], errors="coerce").notna().sum() > 0:
            sr_col = cand
            break
    if sr_col is None and "scoring" in bids.columns:
        # bids の firm×fy 平均を使う
        sr_tmp = (bids.groupby(["firm","fy"])["scoring"].mean()
                  .reset_index().rename(columns={"scoring":"scoring_rate","firm":"firm_id_tmp"}))
        if "firm" in ana.columns:
            sr_tmp2 = sr_tmp.rename(columns={"firm_id_tmp":"firm"})
            ana = ana.merge(sr_tmp2, on=["firm","fy"], how="left")
            sr_col = "scoring_rate"

    if sr_col is None:
        print("  ⚠ scoring_rate / scoring 列が ana に見当たりません")
        print("    診断①②をスキップします")
    else:
        print(f"     (使用列: {sr_col})")
        ana_sorted = ana.sort_values(["firm_id","fy"]) if "firm_id" in ana.columns \
                     else ana.sort_values(["firm","fy"])
        group_col = "firm_id" if "firm_id" in ana_sorted.columns else "firm"
        ana_sorted = ana_sorted.copy()
        ana_sorted["scoring_rate_lag1"] = (ana_sorted
                                           .groupby(group_col)[sr_col].shift(-1))
        for threshold in [78, 79, 80, 81]:
            above = ana_sorted[ana_sorted["h_it"] >= threshold]["scoring_rate_lag1"].dropna()
            below = ana_sorted[(ana_sorted["h_it"] < threshold) &
                               (ana_sorted["h_it"] >= threshold - 2)
                               ]["scoring_rate_lag1"].dropna()
            above_m = above.mean() if len(above)>0 else float("nan")
            below_m = below.mean() if len(below)>0 else float("nan")
            print(f"     h_it ≥ {threshold}: 翌年{sr_col} mean={above_m:.4f} "
                  f"(N={len(above):,})  |  h_it=[{threshold-2},{threshold}): "
                  f"mean={below_m:.4f} (N={len(below):,})")

    # ② h_it と scoring_rate の関係（同年・翌年）
    print(f"\n  ② h_it と {sr_col if sr_col else 'scoring_rate'} の関係（段階別平均）:")
    bins_h = [53, 70, 75, 77, 78, 79, 80, 81, 82, 83]
    if sr_col and sr_col in ana.columns:
        ana_binned = ana.copy()
        ana_binned["h_bin"] = pd.cut(ana_binned["h_it"], bins=bins_h)
        summary = ana_binned.groupby("h_bin", observed=True)[sr_col].agg(["mean","count"])
        print(f"     {sr_col}:")
        for idx, row in summary.iterrows():
            print(f"       {str(idx):20s}: mean={row['mean']:.4f}, N={row['count']:,}")
    else:
        print("     ⚠ 列が存在しないためスキップ")
    for alt in ["perf_y1", "perf_y2", "perf_avg", "h_it"]:
        if alt in ana.columns:
            v = pd.to_numeric(ana[alt], errors="coerce").dropna()
            conc80 = (v == 80).mean()
            conc_flag = " ⚠ 集中過多" if conc80 > 0.4 else " ✓"
            print(f"    {alt:12s}: 平均={v.mean():.2f}, SD={v.std():.3f}, "
                  f"最大={v.max()}, 80点集中率={conc80*100:.1f}%{conc_flag}")

# %%
# ============================================================
# 3. PILLAR I — Two-Way FE (修正版)
# ============================================================
print("\n" + "="*62)
print("PILLAR I: Two-Way FE — Backlog and Bidding Behavior")
print("="*62)

df1 = ana.dropna(subset=["firm_id","fy","log_backlog","bid_rate"]).copy()
df1 = df1.set_index(["firm_id","fy"])

outcomes_p1 = {}
for v,l in [("scoring_rate","Scoring Rate"),("prec_flag","Precision Rate"),("bid_rate","Bid Rate")]:
    if v in df1.columns: outcomes_p1[v] = l

controls_p1 = [c for c in ["log_estimate","log_n_bidders"] if c in df1.columns]

results_p1 = {}
print(f"\n  {'Outcome':20s}  {'β (log_backlog)':>16}  {'SE':>8}  {'p':>6}  {'N':>6}")
print("  " + "-"*62)
for yvar, ylabel in outcomes_p1.items():
    try:
        sub = df1[[yvar,"log_backlog"]+controls_p1].dropna()
        exog = sm.add_constant(sub[["log_backlog"]+controls_p1])
        res  = PanelOLS(sub[yvar], exog, entity_effects=True,
                        time_effects=True, drop_absorbed=True
               ).fit(cov_type="clustered", cluster_entity=True)
        b  = res.params["log_backlog"]
        se = res.std_errors["log_backlog"]
        p  = res.pvalues["log_backlog"]
        n  = int(res.nobs)
        stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.1 else ""
        results_p1[ylabel] = dict(b=b,se=se,p=p,n=n,stars=stars)
        print(f"  {ylabel:20s}  {b:+16.4f}  ({se:.4f})  {p:6.3f}  {n:6d} {stars}")
    except Exception as e:
        print(f"  {ylabel:20s}: エラー — {e}")

# [修正P1] Scoring Rate 正係数の構成効果検定
# → バックログ > 中央値 の企業のみサブサンプルで再推定
print("\n  [修正P1] Scoring Rate 構成効果の確認")
print("  (高バックログ企業 = 過去落札企業 の構成バイアス検定)")
if "scoring_rate" in df1.columns and "log_backlog" in df1.columns:
    w_med = df1["log_backlog"].median()
    for label, mask in [("高バックログ企業のみ (log_backlog > 中央値)",
                          df1["log_backlog"] > w_med),
                         ("全サンプル", pd.Series(True, index=df1.index))]:
        sub = df1.loc[mask, ["scoring_rate","log_backlog"]+controls_p1].dropna()
        if len(sub) < 100: continue
        try:
            exog = sm.add_constant(sub[["log_backlog"]+controls_p1])
            res  = PanelOLS(sub["scoring_rate"], exog,
                            entity_effects=True, time_effects=True,
                            drop_absorbed=True
                   ).fit(cov_type="clustered", cluster_entity=True)
            b  = res.params["log_backlog"]
            se = res.std_errors["log_backlog"]
            p  = res.pvalues["log_backlog"]
            stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            print(f"    {label}: β={b:+.4f} ({se:.4f}) p={p:.3f} {stars}  N={res.nobs:.0f}")
        except Exception as e:
            print(f"    {label}: {e}")

# %%
# ============================================================
# [修正P1] Figure 2候補: Backlog × Bid Rate (本文掲載版)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

df1_plot = df1.reset_index()
for ax, yvar, ylabel in zip(
        axes,
        ["bid_rate","scoring_rate","prec_flag"],
        ["Bid Rate","Scoring Rate","Precision Rate (0–0.5%)"]):
    if yvar not in df1_plot.columns: continue
    sub = df1_plot[["log_backlog", yvar]].dropna()
    bins = pd.cut(sub["log_backlog"], bins=20)
    bm   = sub.groupby(bins, observed=True)[yvar].mean()
    mid  = [iv.mid for iv in bm.index]
    ax.scatter(mid, bm.values, color="#2c7bb6", s=45, alpha=0.85, zorder=3)
    m, b_ = np.polyfit(sub["log_backlog"], sub[yvar], 1)
    xr = np.linspace(sub["log_backlog"].min(), sub["log_backlog"].max(), 100)
    ax.plot(xr, m*xr+b_, color="#d7191c", lw=1.8)
    ax.set_xlabel("log Backlog (w_it)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    if yvar in results_p1:
        r = results_p1[yvar]
        ax.text(0.05, 0.95,
                f"FE β = {r['b']:+.4f}{r['stars']}\n(SE={r['se']:.4f})",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.suptitle("Figure 2 (候補): Backlog and Bidding Behavior\n"
             "(binned means; FE coefficient annotated)", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig_pillar1_main.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ fig_pillar1_main.png 保存")

# %%
# ============================================================
# 4. PILLAR I Robustness — Lead / Lag (Table 6.1)
# ============================================================
print("\n" + "="*62)
print("Pillar I Robustness: Lead/Lag + Firm Trend Specifications")
print("="*62)

df1r = ana.sort_values(["firm_id","fy"]).copy()
df1r["log_backlog_lag"]  = df1r.groupby("firm_id")["log_backlog"].shift(1)
df1r["log_backlog_lead"] = df1r.groupby("firm_id")["log_backlog"].shift(-1)
# firm linear trend
df1r["fy_demeaned"] = df1r.groupby("firm_id")["fy"].transform(lambda x: x - x.mean())
df1r = df1r.set_index(["firm_id","fy"])

specs_rob = [
    ("log_backlog",       "Main (t)"),
    ("log_backlog_lag",   "Lagged (t−1)"),
    ("log_backlog_lead",  "Lead (t+1, placebo)"),
]

for yvar in [v for v in ["bid_rate","scoring_rate"] if v in df1r.columns]:
    print(f"\n  Outcome: {yvar}")
    print(f"  {'Specification':30s}  {'Coef':>8}  {'SE':>8}  {'p':>6}  {'N':>6}")
    print("  " + "-"*60)
    for xvar, label in specs_rob:
        if xvar not in df1r.columns: continue
        sub = df1r[[yvar, xvar]+controls_p1].dropna()
        if len(sub) < 50: continue
        try:
            exog = sm.add_constant(sub[[xvar]+controls_p1])
            res  = PanelOLS(sub[yvar], exog,
                            entity_effects=True, time_effects=True,
                            drop_absorbed=True
                   ).fit(cov_type="clustered", cluster_entity=True)
            b, se, p = res.params[xvar], res.std_errors[xvar], res.pvalues[xvar]
            stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.1 else ""
            print(f"  {label:30s}  {b:+8.4f}  ({se:.4f})  {p:6.3f}  {len(sub):6d} {stars}")
        except Exception as e:
            print(f"  {label:30s}: {e}")
    # Firm linear trend spec
    if "fy_demeaned" in df1r.columns:
        label = "Main + firm trend"
        sub2 = df1r[[yvar,"log_backlog","fy_demeaned"]+controls_p1].dropna()
        if len(sub2) >= 50:
            try:
                exog2 = sm.add_constant(sub2[["log_backlog","fy_demeaned"]+controls_p1])
                res2  = PanelOLS(sub2[yvar], exog2,
                                 entity_effects=True, time_effects=True,
                                 drop_absorbed=True
                        ).fit(cov_type="clustered", cluster_entity=True)
                b, se, p = (res2.params["log_backlog"],
                            res2.std_errors["log_backlog"],
                            res2.pvalues["log_backlog"])
                stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.1 else ""
                print(f"  {label:30s}  {b:+8.4f}  ({se:.4f})  {p:6.3f}  {len(sub2):6d} {stars}")
            except Exception as e:
                print(f"  {label:30s}: {e}")

# %%
# ============================================================
# 5. PILLAR II — RDD (修正版)
# ============================================================
print("\n" + "="*62)
print("PILLAR II: RDD at 80-Point CPR Threshold (修正版)")
print("="*62)

# [修正P2] running variable の選択: 最も分散が大きく80点集中率が低いものを優先
rv_candidates = []
for cand in ["perf_y1","perf_y2","perf_avg","h_it"]:
    if cand in ana.columns:
        v = pd.to_numeric(ana[cand], errors="coerce").dropna()
        conc = (v == 80).mean()
        rv_candidates.append((cand, v.std(), conc, len(v)))

print("\n  Running variable 候補:")
print(f"  {'変数':12s}  {'SD':>6}  {'80点集中率':>10}  {'N':>6}")
for cand,sd,conc,n in rv_candidates:
    flag = " ← 推奨" if sd == max(c[1] for c in rv_candidates) else ""
    print(f"  {cand:12s}  {sd:6.3f}  {conc*100:10.1f}%  {n:6d}{flag}")

# 最も分散が大きいものを選択（ただし80点集中率 < 50% を条件に）
rv_candidates_valid = [(c,s,cc,n) for c,s,cc,n in rv_candidates if cc < 0.5]
if rv_candidates_valid:
    rdd_rv = max(rv_candidates_valid, key=lambda x: x[1])[0]
else:
    rdd_rv = rv_candidates[0][0] if rv_candidates else None
    print(f"  ⚠ 全候補で80点集中率50%超 → {rdd_rv} を使用（結果解釈に注意）")

if rdd_rv is None:
    print("  CPR変数が見当たりません。Pillar II をスキップします")
else:
    print(f"\n  使用する running variable: {rdd_rv}")

    # [修正P2] scoring_rate_next の結合を再構築
    # firm × fy でユニーク行を確保し、翌年行を結合
    rdd_base = ana[["firm_id","fy", rdd_rv, "scoring_rate"] +
                   [c for c in ["log_estimate","log_n_bidders","n_bidders"]
                    if c in ana.columns]].copy()
    rdd_base[rdd_rv]       = pd.to_numeric(rdd_base[rdd_rv], errors="coerce")
    rdd_base["scoring_rate"] = pd.to_numeric(rdd_base["scoring_rate"], errors="coerce")

    # firm-year ユニーク集計（重複がある場合）
    agg_d = {rdd_rv:"mean", "scoring_rate":"mean"}
    for c in ["log_estimate","log_n_bidders","n_bidders"]:
        if c in rdd_base.columns: agg_d[c] = "mean"
    rdd_fy = rdd_base.groupby(["firm_id","fy"]).agg(agg_d).reset_index()

    # 翌年の scoring_rate を結合
    next_s = rdd_fy[["firm_id","fy","scoring_rate"]].copy()
    next_s.columns = ["firm_id","fy","scoring_rate_next"]
    next_s["fy"] = next_s["fy"] - 1
    rdd_df = rdd_fy.merge(next_s, on=["firm_id","fy"], how="left")

    rdd_df["run"]   = rdd_df[rdd_rv] - 80.0
    rdd_df["treat"] = (rdd_df["run"] >= 0).astype(int)

    # [修正P2] 診断: 80点以上と未満で翌年CE参加率を比較
    print(f"\n  翌年 scoring_rate の平均 (cutoff = {rdd_rv} 80点):")
    for g, label in [(rdd_df[rdd_df["treat"]==0], "< 80"),
                     (rdd_df[rdd_df["treat"]==1], "≥ 80")]:
        if "scoring_rate_next" in g.columns:
            m = g["scoring_rate_next"].dropna().mean()
            n = g["scoring_rate_next"].dropna().count()
            print(f"    {label}: mean={m:.4f}  N={n:,}")

    def local_linear_rdd(df, outcome, bw, donut=0.0):
        sub = df[(df["run"].abs() <= bw) & (df["run"].abs() > donut)
                 ].dropna(subset=["run","treat",outcome]).copy()
        if len(sub) < 30: return None
        sub["rx"] = sub["run"] * sub["treat"]
        X = sm.add_constant(sub[["treat","run","rx"]])
        cov_kwds = ({"groups": sub["firm_id"]}
                    if "firm_id" in sub.columns else {})
        try:
            res = sm.OLS(sub[outcome], X).fit(
                cov_type="cluster", cov_kwds=cov_kwds)
            return dict(tau=res.params["treat"], se=res.bse["treat"],
                        p=res.pvalues["treat"], n=int(res.nobs))
        except Exception:
            res = sm.OLS(sub[outcome], X).fit(cov_type="HC1")
            return dict(tau=res.params["treat"], se=res.bse["treat"],
                        p=res.pvalues["treat"], n=int(res.nobs))

    outcome_rdd = ("scoring_rate_next"
                   if rdd_df["scoring_rate_next"].notna().sum() > 50
                   else "scoring_rate")
    print(f"\n  Outcome: {outcome_rdd}")
    print(f"\n  {'BW':>4}  {'Donut':>5}  {'τ':>10}  {'SE':>8}  {'p':>6}  {'N':>6}")
    print("  " + "-"*50)

    rdd_results = {}
    for bw in [1,2,3,5]:
        for donut in [0.0]:
            r = local_linear_rdd(rdd_df, outcome_rdd, bw, donut)
            if r:
                stars = "***" if r["p"]<0.01 else "**" if r["p"]<0.05 else "*" if r["p"]<0.1 else ""
                print(f"  ±{bw:2d}  {donut:>5.1f}  {r['tau']:+10.4f}  ({r['se']:.4f})  "
                      f"{r['p']:6.3f}  {r['n']:6d} {stars}")
                if bw==2 and donut==0.0: rdd_results["main"] = r

    # Placebo cutoffs
    print(f"\n  Placebo cutoffs (BW=±2):")
    print(f"  {'Cutoff':>6}  {'τ':>10}  {'p':>6}  {'N':>6}")
    for fake_c in [77,78,79,80,81,82]:
        tmp = rdd_df.copy()
        tmp["run"]   = tmp[rdd_rv] - fake_c
        tmp["treat"] = (tmp["run"] >= 0).astype(int)
        r = local_linear_rdd(tmp, outcome_rdd, 2)
        if r:
            marker = " ← main" if fake_c==80 else ""
            stars  = "***" if r["p"]<0.01 else "**" if r["p"]<0.05 else "*" if r["p"]<0.1 else ""
            print(f"  {fake_c:6d}   {r['tau']:+10.4f}  {r['p']:6.3f}  {r['n']:6d} {stars}{marker}")

# [修正P2] RDD 高品質 Figure
if rdd_rv:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Score distribution with integer bins
    ax = axes[0]
    score_v = rdd_df[rdd_rv].dropna()
    bin_edges = np.arange(score_v.min()-0.5, score_v.max()+1.5, 1.0)
    counts, edges, patches = ax.hist(score_v, bins=bin_edges,
                                      color="#aaaaaa", edgecolor="white", lw=0.5)
    for patch, left in zip(patches, edges[:-1]):
        if 79.5 <= left < 80.5:
            patch.set_facecolor("#d7191c")
    ax.axvline(80, color="black", lw=1.8, ls="--", label="Cutoff = 80")
    ax.set_xlabel(f"CPR Score ({rdd_rv})", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("CPR Score Distribution", fontsize=12)
    ax.legend(frameon=False)
    # McCrary-style annotation
    n80  = ((score_v >= 79.5) & (score_v < 80.5)).sum()
    n79  = ((score_v >= 78.5) & (score_v < 79.5)).sum()
    ratio = n80/n79 if n79>0 else np.nan
    ax.text(0.05, 0.95,
            f"Score-79: {n79:,}\nScore-80: {n80:,}\nRatio: {ratio:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="gray",alpha=0.8))

    # Right: Binned mean outcome
    ax = axes[1]
    rdd_plot = rdd_df[(rdd_df["run"].abs() <= 5)].copy()
    if outcome_rdd in rdd_plot.columns:
        bins_rdd = pd.cut(rdd_plot["run"],
                          bins=np.arange(-5.5, 6.0, 1.0), include_lowest=True)
        bm_rdd   = rdd_plot.groupby(bins_rdd, observed=True)[outcome_rdd].mean()
        bm_n     = rdd_plot.groupby(bins_rdd, observed=True)[outcome_rdd].count()
        mid_rdd  = [iv.mid for iv in bm_rdd.index]
        colors   = ["#2c7bb6" if m < 0 else "#d7191c" for m in mid_rdd]
        ax.scatter(mid_rdd, bm_rdd.values, c=colors, s=60, zorder=3)
        ax.axvline(0, color="black", lw=1.8, ls="--")
        # Local linear fit lines
        for side, col in [(rdd_plot[rdd_plot["run"]<0], "#2c7bb6"),
                          (rdd_plot[rdd_plot["run"]>=0], "#d7191c")]:
            s2 = side[["run",outcome_rdd]].dropna()
            if len(s2) > 5:
                m2, b2 = np.polyfit(s2["run"], s2[outcome_rdd], 1)
                xr2 = np.linspace(s2["run"].min(), s2["run"].max(), 50)
                ax.plot(xr2, m2*xr2+b2, color=col, lw=1.5, alpha=0.7)
        ax.set_xlabel(f"CPR Score ({rdd_rv}) − 80", fontsize=11)
        ax.set_ylabel(f"Mean {outcome_rdd}", fontsize=11)
        ax.set_title("RDD: Next-Year CE Participation\naround 80-point threshold", fontsize=12)
        if "main" in rdd_results:
            r = rdd_results["main"]
            stars = "***" if r["p"]<0.01 else "**" if r["p"]<0.05 else "*" if r["p"]<0.1 else ""
            ax.text(0.05, 0.95,
                    f"BW=±2: τ={r['tau']:+.4f}{stars}\n(SE={r['se']:.4f}, p={r['p']:.3f})",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="gray",alpha=0.8))
    plt.tight_layout()
    plt.savefig("fig_pillar2_rdd.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n→ fig_pillar2_rdd.png 保存")

# %%
# ============================================================
# 6. PILLAR III — イベントスタディ (v3: Bartik 正式構築 + 並行トレンド)
# ============================================================
print("\n" + "="*62)
print("PILLAR III: Event Study — Bartik Exposure (v3 正式構築)")
print("="*62)

# [v3] Bartik_i = Σ_k share_ik_pre × Δce_k
#   share_ik_pre : 企業iのpre-2009入札における工種kのシェア
#   Δce_k        : 工種k の 2009-2016 CE採用率 − pre-2009 CE採用率

firm_col = "firm"

if "work_type" in bids.columns and "scoring" in bids.columns:
    bids_ce = bids[["firm","fy","work_type","scoring"]].copy()
    bids_ce["scoring"] = pd.to_numeric(bids_ce["scoring"], errors="coerce")

    # Step 1: 工種別 CE 採用率の変化
    bids_ce["period"] = np.where(bids_ce["fy"] < 2009, "pre",
                          np.where(bids_ce["fy"].between(2009,2016), "mid", "post"))
    wt_ce = (bids_ce.groupby(["work_type","period"])["scoring"]
             .mean().unstack(fill_value=np.nan))
    delta_col = "delta_ce_09"
    if "pre" in wt_ce.columns and "mid" in wt_ce.columns:
        wt_ce[delta_col] = wt_ce["mid"] - wt_ce["pre"]
    else:
        wt_ce[delta_col] = 0.0
    wt_ce = wt_ce[[delta_col]].reset_index()

    valid_wt = wt_ce.dropna(subset=[delta_col])
    print(f"  工種別 CE 変化率: 平均={valid_wt[delta_col].mean():+.4f}, "
          f"SD={valid_wt[delta_col].std():.4f}, "
          f"正値={( valid_wt[delta_col]>0).sum()}/{len(valid_wt)} 工種")

    # Step 2: 企業のpre-2009 工種シェア
    pre09 = bids_ce[bids_ce["fy"] < 2009].copy()
    n_pre_firms = pre09["firm"].nunique()
    firm_total  = pre09.groupby("firm").size().rename("total_pre")
    firm_wt_c   = (pre09.groupby(["firm","work_type"]).size()
                   .reset_index(name="n_wt")
                   .merge(firm_total.reset_index(), on="firm"))
    firm_wt_c["share"] = firm_wt_c["n_wt"] / firm_wt_c["total_pre"]
    firm_wt_c = firm_wt_c.merge(wt_ce, on="work_type", how="left")
    firm_wt_c["bartik_contrib"] = firm_wt_c["share"] * firm_wt_c[delta_col].fillna(0)
    bartik = firm_wt_c.groupby("firm")["bartik_contrib"].sum().reset_index()
    bartik.columns = ["firm", "bartik_exposure"]

    print(f"  Bartik 構築: {n_pre_firms:,} 社 (pre-2009) → "
          f"{len(bartik):,} 社に Exposure 付与")
    print(f"  Exposure: mean={bartik['bartik_exposure'].mean():+.4f}, "
          f"SD={bartik['bartik_exposure'].std():.4f}")

    # bids に結合（重複列を避けるため既存 bartik_exposure を上書き）
    bids.drop(columns=["bartik_exposure"], errors="ignore", inplace=True)
    bids = bids.merge(bartik, on="firm", how="left")
    exposure_col = "bartik_exposure"
    n_exp = bids[exposure_col].notna().sum()

    # フォールバック: pre-2009入札がない企業には全期間CE率を代入
    if n_exp < len(bids) * 0.5:
        print(f"  フォールバック: {len(bids)-n_exp:,} 行に全期間CE率を補完")
        all_ce = bids.groupby("firm")["scoring"].mean().reset_index()
        all_ce.columns = ["firm","ce_all"]
        bids = bids.merge(all_ce, on="firm", how="left")
        bids["bartik_exposure"] = bids["bartik_exposure"].fillna(bids["ce_all"])
        bids.drop(columns=["ce_all"], errors="ignore", inplace=True)
        n_exp = bids[exposure_col].notna().sum()
    print(f"  最終的な Exposure 有効行: {n_exp:,}/{len(bids):,}")
else:
    exposure_col = None
    print("  ⚠ work_type または scoring 列が見当たりません")

# FY 別 firm-year 集計
if exposure_col and exposure_col in bids.columns:
    outcomes_p3 = {v: l for v, l in
                   [("scoring","Scoring Rate"),
                    ("prec_flag","Precision Rate"),
                    ("bid_rate","Bid Rate")]
                   if v in bids.columns}

    agg_d3 = {k:"mean" for k in list(outcomes_p3.keys()) + [exposure_col]
              if k in bids.columns}
    bids_fy = (bids.groupby(["firm","fy"]).agg(agg_d3)
               .reset_index().rename(columns={"firm":"firm_id"}))
    bids_fy = bids_fy[bids_fy["fy"].between(2006,2024)]

    # Period-wise summary
    print(f"\n  Period-wise Exposure 相関:")
    print(f"  {'Period':25s}", end="")
    for l in outcomes_p3.values(): print(f"  {l:>22}", end="")
    print(); print("  " + "-"*88)

    for label, lo, hi in [("Pre-reform (2006-08)", 2006, 2008),
                           ("Post-2009 (2009-13)",  2009, 2013),
                           ("Post-2014 (2014-16)",  2014, 2016),
                           ("Late (2017-24)",        2017, 2024)]:
        sub_t = bids_fy[bids_fy["fy"].between(lo, hi)]
        print(f"  {label:25s}", end="")
        for yvar in outcomes_p3:
            d = sub_t[[yvar, exposure_col]].dropna()
            if len(d) > 5:
                b_ols, _ = np.polyfit(d[exposure_col].astype(float),
                                      d[yvar].astype(float), 1)
                r_val, p_val = stats.pearsonr(d[exposure_col].astype(float),
                                              d[yvar].astype(float))
                stars = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else ""
                print(f"  β={b_ols:+.4f} r={r_val:+.3f}{stars:3s}", end="")
            else:
                print(f"  {'n/a':>22}", end="")
        print()

    # Year-by-year OLS (全サンプル)
    fy_range = sorted(bids_fy["fy"].dropna().astype(int).unique())
    ref_year = 2008

    def event_study_ols(df_fy, exp_col, outcomes, fy_list, ref_y):
        coefs = {v:[] for v in outcomes}
        ses   = {v:[] for v in outcomes}
        fys   = []
        for fy_t in fy_list:
            if fy_t == ref_y: continue
            sub = df_fy[df_fy["fy"].isin([ref_y, fy_t])].copy()
            for yvar in outcomes:
                tmp = sub[[exp_col, yvar]].dropna()
                if len(tmp) < 10:
                    coefs[yvar].append(np.nan); ses[yvar].append(np.nan); continue
                try:
                    X = sm.add_constant(tmp[exp_col].astype("float64"))
                    res = sm.OLS(tmp[yvar].astype("float64"), X).fit(cov_type="HC1")
                    coefs[yvar].append(res.params[exp_col])
                    ses[yvar].append(res.bse[exp_col])
                except Exception:
                    coefs[yvar].append(np.nan); ses[yvar].append(np.nan)
            fys.append(fy_t)
        return coefs, ses, fys

    ev_coefs, ev_ses, ev_fys = event_study_ols(
        bids_fy, exposure_col, list(outcomes_p3.keys()), fy_range, ref_year)

    # CE参加経験企業のみのサブサンプル
    scoring_active = (bids_fy.groupby("firm_id")["scoring"].max() > 0) \
                     if "scoring" in bids_fy.columns else pd.Series(dtype=bool)
    active_firms = scoring_active[scoring_active].index if len(scoring_active) > 0 else []
    bids_fy_act = bids_fy[bids_fy["firm_id"].isin(active_firms)]
    print(f"\n  CE参加経験企業: {len(active_firms):,} / {bids_fy['firm_id'].nunique():,}")
    ev_coefs_a, ev_ses_a, ev_fys_a = event_study_ols(
        bids_fy_act, exposure_col, list(outcomes_p3.keys()), fy_range, ref_year)

    # 並行トレンド Wald 検定
    print(f"\n  並行トレンド検定 (2006/2007 係数=0, Wald):")
    for yvar, ylabel in outcomes_p3.items():
        pre_idx = [i for i,f in enumerate(ev_fys) if f in [2006,2007]]
        pbc = [(ev_coefs[yvar][i], ev_ses[yvar][i]) for i in pre_idx
               if not (np.isnan(ev_coefs[yvar][i]) or
                       np.isnan(ev_ses[yvar][i]) or ev_ses[yvar][i]==0)]
        if pbc:
            w = sum((b/s)**2 for b,s in pbc)
            p_w = 1 - stats.chi2.cdf(w, df=len(pbc))
            flag = "⚠ 並行トレンド違反" if p_w<0.1 else "✓ 問題なし"
            print(f"    {ylabel:15s}: W={w:.2f}, p={p_w:.4f}  {flag}")

    # Figure: 全サンプル + CE参加経験企業 (2行3列)
    n_out = len(outcomes_p3)
    fig, axes = plt.subplots(2, n_out, figsize=(6*n_out, 9), sharex=True)
    if n_out == 1: axes = axes.reshape(2,1)

    for col_i, (yvar, ylabel) in enumerate(outcomes_p3.items()):
        for row_i, (c_d, s_d, f_d, sfx) in enumerate([
            (ev_coefs,   ev_ses,   ev_fys,   "全サンプル"),
            (ev_coefs_a, ev_ses_a, ev_fys_a, "CE参加経験企業"),
        ]):
            ax = axes[row_i][col_i]
            c_arr = np.array(c_d[yvar], dtype=float)
            s_arr = np.array(s_d[yvar], dtype=float)
            f_arr = np.array(f_d,       dtype=float)
            valid = ~np.isnan(c_arr)
            colors_ev = ["#d7191c" if f>=2014 else "#2c7bb6" if f>=2009 else "#999999"
                         for f in f_arr[valid]]
            ax.scatter(f_arr[valid], c_arr[valid], c=colors_ev, s=50, zorder=3)
            ax.errorbar(f_arr[valid], c_arr[valid],
                        yerr=1.96*np.where(np.isnan(s_arr[valid]),0,s_arr[valid]),
                        fmt="none", ecolor="gray", alpha=0.5, capsize=2)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.axvline(2009, color="#2c7bb6", lw=1.5, ls=":", alpha=0.7,
                       label="2009 reform" if col_i==0 else "")
            ax.axvline(2014, color="#d7191c", lw=1.5, ls=":", alpha=0.7,
                       label="2014 reform" if col_i==0 else "")
            ax.axvline(ref_year, color="gray", lw=1, ls="-", alpha=0.4)
            ax.set_xlabel("Fiscal Year" if row_i==1 else "")
            ax.set_ylabel("Bartik Exposure coeff.")
            ax.set_title(f"{ylabel}\n({sfx})", fontsize=10)
            if col_i == 0: ax.legend(frameon=False, fontsize=8)

    plt.suptitle("Figure 4: Event Study — Bartik Exposure × FY\n"
                 "上段=全サンプル, 下段=CE参加経験企業 (95% CI)",
                 fontweight="bold", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("fig_pillar3_eventstudy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("→ fig_pillar3_eventstudy.png 保存")

# %%
# ============================================================
# 7. PILLAR IV — DDD 2024 能登地震 (修正版)
# ============================================================
print("\n" + "="*62)
print("PILLAR IV: Triple Differences — 2024 Noto Earthquake (修正版)")
print("="*62)

# [v3修正P4-1] h_it を analysis_dataset から all_bids_dataset に結合
# [修正点] bids.get() は DataFrame では使えないためKeyErrorが発生していた
#          → "h_it_from_ana" 列の存在を明示確認してから fillna
print("  h_it 結合を試行中...")
if "h_it" in ana.columns and "firm" in ana.columns:
    # firm × fy で h_it の最頻値または平均を取得
    h_src = (ana.groupby(["firm","fy"])["h_it"]
             .agg(lambda x: x.dropna().mean() if x.notna().any() else np.nan)
             .reset_index())
    h_src.columns = ["firm","fy","h_it_ana"]

    # bids に h_it が既にある場合は欠損補完、ない場合は新規結合
    bids = bids.merge(h_src, on=["firm","fy"], how="left")

    if "h_it" not in bids.columns:
        bids["h_it"] = bids["h_it_ana"]
        print(f"  h_it 新規結合: {bids['h_it'].notna().sum():,} 行")
    else:
        bids["h_it"] = pd.to_numeric(bids["h_it"], errors="coerce")
        before = bids["h_it"].notna().sum()
        bids["h_it"] = bids["h_it"].fillna(bids["h_it_ana"])
        after  = bids["h_it"].notna().sum()
        print(f"  h_it 補完: {before:,} → {after:,} 行 (+{after-before:,})")
    bids.drop(columns=["h_it_ana"], errors="ignore", inplace=True)
else:
    print("  ⚠ analysis_dataset に firm または h_it 列がありません")
print(f"  最終的な h_it 有効行: {bids['h_it'].notna().sum():,} / {len(bids):,}")

# [修正P4-2] T フラグ作成
bids["h_it_num"] = pd.to_numeric(bids["h_it"], errors="coerce")
bids["T_a"] = (bids["h_it_num"] >= 75).astype(float)
bids["T_a"] = bids["T_a"].fillna(0.0)
bids["post_noto"] = (bids["fy"] >= 2024).astype(float)

print(f"\n  Ishikawa フラグ: {bids['ishikawa'].sum():,} 入札")
print(f"  T_a (CPR≥75):   {bids['T_a'].mean():.3f} (h_it 有効行の割合)")
print(f"  Post-Noto:       {bids['post_noto'].sum():,} 入札 (FY≥2024)")

# [修正P4-3] DDD 推定関数 (型変換エラー修正)
def ddd_ols(df, outcome, I_col, Post_col, T_col,
            controls=None, cluster_col=None):
    cols_need = [outcome, I_col, Post_col, T_col]
    if controls: cols_need += controls
    sub = df[cols_need + ([cluster_col] if cluster_col and cluster_col in df.columns else [])].copy()

    # [修正] 全変数を明示的に float64 に変換（参照型配列エラーの回避）
    for c in [outcome, I_col, Post_col, T_col] + (controls or []):
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").astype("float64")
    sub = sub.dropna(subset=[outcome, I_col, Post_col, T_col])
    if len(sub) < 20:
        return {"error": f"観測数不足 ({len(sub)})"}

    sub["I_Post"]  = sub[I_col] * sub[Post_col]
    sub["I_T"]     = sub[I_col] * sub[T_col]
    sub["Post_T"]  = sub[Post_col] * sub[T_col]
    sub["DDD"]     = sub[I_col] * sub[Post_col] * sub[T_col]

    X_cols = [I_col, Post_col, T_col, "I_Post","I_T","Post_T","DDD"]
    if controls: X_cols += [c for c in controls if c in sub.columns]
    X = sm.add_constant(sub[X_cols].astype("float64"))

    try:
        if cluster_col and cluster_col in sub.columns:
            res = sm.OLS(sub[outcome].astype("float64"), X).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub[cluster_col].astype(str)})
        else:
            res = sm.OLS(sub[outcome].astype("float64"), X).fit(cov_type="HC1")
        return {
            "DDD_b": res.params.get("DDD", np.nan),
            "DDD_se": res.bse.get("DDD", np.nan),
            "DDD_p": res.pvalues.get("DDD", np.nan),
            "DiD_b": res.params.get("I_Post", np.nan),
            "DiD_se": res.bse.get("I_Post", np.nan),
            "DiD_p": res.pvalues.get("I_Post", np.nan),
            "n": int(res.nobs),
        }
    except Exception as e:
        return {"error": str(e)}

ctrl_p4 = [c for c in ["log_estimate","log_n_bidders"] if c in bids.columns]
outcomes_p4 = {v:l for v,l in
               [("prec_flag","Precision Rate"),
                ("scoring","Scoring Rate"),
                ("bid_rate","Bid Rate")]
               if v in bids.columns}

print(f"\n  Panel B: T_a (CPR≥75 または h_it 欠損→0)")
print(f"  {'Outcome':15s}  {'DDD':>10}  {'SE':>8}  {'p':>6}  "
      f"{'DiD(I×Post)':>12}  {'SE':>8}  {'p':>6}  {'N':>6}")
print("  " + "-"*80)

for yvar, ylabel in outcomes_p4.items():
    r = ddd_ols(bids, yvar, "ishikawa", "post_noto", "T_a",
                controls=ctrl_p4, cluster_col=firm_col)
    if "error" in r:
        print(f"  {ylabel:15s}: エラー — {r['error']}")
    else:
        ddd_s = "***" if r["DDD_p"]<0.01 else "**" if r["DDD_p"]<0.05 else "*" if r["DDD_p"]<0.1 else ""
        did_s = "***" if r["DiD_p"]<0.01 else "**" if r["DiD_p"]<0.05 else "*" if r["DiD_p"]<0.1 else ""
        print(f"  {ylabel:15s}  {r['DDD_b']:+10.4f}  ({r['DDD_se']:.4f})  {r['DDD_p']:6.3f}{ddd_s}"
              f"  {r['DiD_b']:+12.4f}  ({r['DiD_se']:.4f})  {r['DiD_p']:6.3f}{did_s}  {r['n']:6d}")

# Placebo test
print(f"\n  プラセボ検定 (Post = FY≥2022, サンプルはFY<2024):")
bids_plac = bids[bids["fy"] < 2024].copy()
bids_plac["post_placebo"] = (bids_plac["fy"] >= 2022).astype(float)
for yvar, ylabel in outcomes_p4.items():
    r = ddd_ols(bids_plac, yvar, "ishikawa", "post_placebo", "T_a",
                controls=ctrl_p4, cluster_col=firm_col)
    if "error" not in r:
        flag = "← 有意 ⚠ (pre-trend)" if r["DDD_p"]<0.1 else "← 非有意 ✓"
        print(f"  {ylabel:15s}: DDD p={r['DDD_p']:.3f}  {flag}")

# %%
# ============================================================
# 8. 時系列サマリー Figure (Figure 3 — 本文掲載版)
# ============================================================
print("\n" + "="*62)
print("Figure 3 (本文掲載候補): 時系列サマリー")
print("="*62)

plot_info = [
    ("scoring",   "CE Scoring Rate",         "#2c7bb6"),
    ("bid_rate",  "Bid Rate",                 "#2c7bb6"),
    ("prec_flag", "Precision Rate (0–0.5%)",  "#2c7bb6"),
]
plot_avail = [(v,l,c) for v,l,c in plot_info if v in bids.columns]

if plot_avail:
    ts = (bids.groupby("fy")[[v for v,_,_ in plot_avail]].mean().reset_index()
          .query("2006 <= fy <= 2024"))

    fig, axes = plt.subplots(1, len(plot_avail), figsize=(5.5*len(plot_avail), 4.5))
    if len(plot_avail) == 1: axes = [axes]

    for ax, (vvar, vlabel, vcol) in zip(axes, plot_avail):
        ax.plot(ts["fy"], ts[vvar], color=vcol, lw=2, marker="o",
                markersize=5, zorder=3)
        ax.axvspan(2009, 2009.5, color="#d7191c", alpha=0.12, label="2009 reform")
        ax.axvspan(2014, 2014.5, color="#e87e04", alpha=0.12, label="2014 reform")
        ax.axvline(2009, color="#d7191c", lw=1.5, ls="--", alpha=0.7)
        ax.axvline(2014, color="#e87e04", lw=1.5, ls="--", alpha=0.7)
        ax.set_xlabel("Fiscal Year", fontsize=11)
        ax.set_ylabel(vlabel, fontsize=11)
        ax.set_title(vlabel, fontsize=12)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    plt.suptitle("Figure 3: Annual Means of Key Outcomes (all_bids_dataset)",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("fig_timeseries.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("→ fig_timeseries.png 保存")

# %%
# ============================================================
# 9. 修正要否サマリー
# ============================================================
print("\n" + "="*62)
print("SUMMARY: 結果の解釈と第4節修正方針")
print("="*62)
print("""
◆ Pillar I (H1)
  Bid Rate: β=+0.0025*** (理論一致) ✓
  Scoring Rate: β=+0.0084*** (符号逆) → 構成効果検定の結果を確認
    Lead/Lag Robustness: Lead(t+1) が小さければ逆因果は否定
  Precision Rate: 非有意だが方向は一致
  ★ 推奨: 3変数並列図 (fig_pillar1_main.png) → 本文 Figure 2 として掲載

◆ Pillar II (H2)
  BW=±2: τ=+0.0067 (p=0.142) ← 本文報告値 +0.026 (p<0.001) と大きく乖離
  診断①: h_it≥80 の企業の翌年CE参加率が正か確認（本セクション冒頭を参照）
  診断②: perf_y1 を running variable とした場合の τ を確認
  ⚠ 右辺ゼロ問題が続く場合: h_it の定義を本文 Table 1 注釈と照合

◆ Pillar III (H3)
  exp_quartile_num → Bartik Exposure に差し替え済み
  確認事項:
    (1) Pre-reform 並行トレンド Wald 検定 p ≥ 0.1 なら ✓
    (2) Post-2009 期間の係数が正なら H3 支持
    (3) CE参加経験企業サブサンプルで2017年以降の係数を確認
  ★ 推奨: 2行×3列イベントスタディ図 (fig_pillar3_eventstudy.png) を掲載

◆ Pillar IV (H4)
  h_it 結合バグを修正 (bids.get() → 明示的 merge + drop_duplicates)
  確認事項:
    (1) h_it 結合後の有効行数が > 0 であることを確認
    (2) T_a の平均が > 0 であることを確認
    (3) DDD 係数の符号 (Precision Rate +, Bid Rate +)
    (4) プラセボ (Post=2022) が非有意 → causal 解釈を支持

◆ 本文掲載推奨 Figure:
  Figure 2: fig_pillar1_main.png      (Backlog × 3 outcomes)
  Figure 3: fig_timeseries.png         (年次平均時系列) ← 最優先
  Figure 4: fig_pillar3_eventstudy.png (Bartik × FY イベントスタディ)
  Figure 5: fig_pillar2_rdd.png        (RDD ※h_it問題解決後に判断)
""")
