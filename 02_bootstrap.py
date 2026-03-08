# =============================================================================
# 02_bootstrap.py   (v6-calibrated, 2026-03-08)
# =============================================================================
# 実際の v6 推定結果に完全準拠したブートストラップスクリプト
#
# 点推定値 (vf_v6_theta_stage2.npy から確認済み):
#   β        = 0.6057  (beta_raw=0.4421, 変換: 0.995·σ(beta_raw))
#   κ        = 1.1666  (log_kappa=0.1541, 変換: exp(log_kappa))
#   u_LOW_w  = −0.2824 (バックログ係数, Low=PO state, X_COLS[2])
#   u_HIGH_w = +0.2824 (バックログ係数, High=CE state, X_COLS[2])
#   u_LOW_h  = +0.0902 (CPR係数, Low state)
#   u_HIGH_h = +0.3921 (CPR係数, High state)
#   σ_sc=0.1216, σ_pr=0.1011, σ_bd=0.0547
#
# 点推定に使ったサンプル:
#   STAGED_FIRMS=[200, 500], 最終stage=500 firms (7,699 rows)
#
# ブートストラップの設計:
#   - firm-level nonparametric bootstrap (企業単位でリサンプリング)
#   - 初期値 = 点推定値 theta_hat (収束が速い)
#   - staged subsample = [200, 500] (点推定と同じ)
#   - CI は変換後パラメータのパーセンタイルで計算
#     → β の CI逆転問題は raw→変換後の一貫した処理で解消
#
# 使い方 (Google Colab):
#   1. analysis_dataset.csv と all_bids_dataset.csv をアップロード
#   2. vf_v6_theta_stage2.npy を SAVE_DIR に置く
#   3. 本スクリプトをそのまま実行
#
# 出力:
#   bootstrap_raw.csv       — 全反復の変換後パラメータ (診断用)
#   bootstrap_summary.csv   — Table 7 用 SE・CI サマリー
#   bootstrap_trace.png     — β トレース + 分布プロット (収束確認)
#   bootstrap_dists.png     — 全主要パラメータの分布
# =============================================================================

# ── セル 0: インストール (初回のみ) ──
# !pip install -q scipy numpy pandas matplotlib tqdm

# ── セル 1: Google Drive マウント ──
# from google.colab import drive
# drive.mount('/content/drive')

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special  import expit, logsumexp
from scipy.optimize import minimize
from tqdm.auto      import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# 0.  パス設定
# =============================================================================
SAVE_DIR = "/content/drive/MyDrive/structural_estimation_v6"

ALL_BIDS_CANDIDATES = [
    os.path.join(SAVE_DIR, "all_bids_dataset_contractfy_fixed.csv"),
    os.path.join(SAVE_DIR, "all_bids_dataset.csv"),
    "/content/all_bids_dataset.csv",
]
ANALYSIS_CANDIDATES = [
    os.path.join(SAVE_DIR, "analysis_dataset_contractfy_fixed.csv"),
    os.path.join(SAVE_DIR, "analysis_dataset.csv"),
    "/content/analysis_dataset.csv",
]
THETA_HAT_PATH = os.path.join(SAVE_DIR, "vf_v6_theta_stage2.npy")

OUT_RAW_CSV   = os.path.join(SAVE_DIR, "bootstrap_raw.csv")
OUT_SUMM_CSV  = os.path.join(SAVE_DIR, "bootstrap_summary.csv")
OUT_TRACE_PNG = os.path.join(SAVE_DIR, "bootstrap_trace.png")
OUT_DIST_PNG  = os.path.join(SAVE_DIR, "bootstrap_dists.png")

# =============================================================================
# 1.  設定
# =============================================================================
N_BOOT       = 200      # 推奨200。Colab free tier なら100でも可。
SEED         = 42
BOOT_MAXITER = 150      # 各反復のL-BFGS-Bイテレーション上限
STAGED_FIRMS = [200, 500]   # 点推定と同じ staged subsample

REFORM_2009  = 2009
REFORM_2014  = 2014
REFORM_2023  = 2023

# =============================================================================
# 2.  モデル定数・ユーティリティ (v6 と完全一致)
# =============================================================================
STATE_LOW  = 0    # PO (price-only) regime
STATE_HIGH = 1    # CE (comprehensive evaluation) regime

X_COLS = ["x0", "h", "w", "post09", "post14", "post23"]
N_COV  = len(X_COLS)
W_IDX  = X_COLS.index("w")   # = 2
H_IDX  = X_COLS.index("h")   # = 1
K      = 1

def param_dims(has_prec=True):
    n_out = 3 if has_prec else 2
    return dict(
        n_beta=1, n_kap=1,
        n_u=2*N_COV, n_m=n_out*2*N_COV, n_sig=n_out, n_out=n_out,
        total=1+1+2*N_COV+n_out*2*N_COV+n_out,
    )

def unpack_theta(theta, has_prec=True):
    """raw θ → 変換後パラメータ (v6 K=1 と完全一致)"""
    d   = param_dims(has_prec)
    idx = 0
    beta  = 0.995 * expit(theta[idx]); idx += 1
    kappa = np.exp(theta[idx]);        idx += 1
    u     = theta[idx:idx+d["n_u"]].reshape(2, N_COV); idx += d["n_u"]
    m     = theta[idx:idx+d["n_m"]].reshape(d["n_out"], 2, N_COV); idx += d["n_m"]
    sigs  = np.exp(theta[idx:idx+d["n_sig"]]) + 1e-6
    return beta, kappa, u, m, sigs

def extract_key(theta, has_prec=True):
    b, k, u, m, s = unpack_theta(theta, has_prec)
    return {
        "beta":      float(b),
        "kappa":     float(k),
        "u_LOW_w":   float(u[STATE_LOW,  W_IDX]),
        "u_HIGH_w":  float(u[STATE_HIGH, W_IDX]),
        "u_LOW_h":   float(u[STATE_LOW,  H_IDX]),
        "u_HIGH_h":  float(u[STATE_HIGH, H_IDX]),
        "u_LOW_x0":  float(u[STATE_LOW,  0]),
        "u_HIGH_x0": float(u[STATE_HIGH, 0]),
        "sig_sc":    float(s[0]),
        "sig_pr":    float(s[1]) if has_prec else np.nan,
        "sig_bd":    float(s[-1]),
    }

def zscore_vec(x):
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd + 1e-10)

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def normalize_firm(s):
    import re
    if not isinstance(s, str):
        return str(s)
    s = re.sub(r"[（(].*?[)）]", "", s.strip())
    return re.sub(r"\s+", "", s)

# =============================================================================
# 3.  尤度計算 (v6 forward_filter_exact, K=1)
# =============================================================================
def firm_loglik(theta, fd, has_prec):
    b, k, u, m, sigs = unpack_theta(theta, has_prec)
    x, y = fd["x_seq"], fd["y_seq"]
    T, n_out = len(y), m.shape[0]
    if T < 2:
        return 0.0
    # 遷移確率行列
    U  = x @ u.T
    P  = np.zeros((T, 2, 2))
    EV = np.zeros(2)
    for t in range(T-1, -1, -1):
        v00 = U[t,0] + b*EV[0]
        v01 = U[t,1] - k + b*EV[1]
        v10 = U[t,0] - k + b*EV[0]
        v11 = U[t,1] + b*EV[1]
        l0, l1 = logsumexp([v00,v01]), logsumexp([v10,v11])
        P[t,0,0]=np.exp(v00-l0); P[t,0,1]=np.exp(v01-l0)
        P[t,1,0]=np.exp(v10-l1); P[t,1,1]=np.exp(v11-l1)
        EV[0]=l0; EV[1]=l1
    # エミッション行列
    E = np.zeros((T, 2))
    for s in range(2):
        mu   = x @ m[:,s,:].T
        diff = y - mu
        E[:,s] = (-0.5*n_out*np.log(2*np.pi)
                  - np.sum(np.log(sigs))
                  - 0.5*np.sum((diff/sigs[None,:])**2, axis=1))
    # フォワードフィルタ
    la = np.zeros((T, 2))
    la[0] = np.log(0.5) + E[0]
    for t in range(1, T):
        for s1 in range(2):
            la[t,s1] = E[t,s1] + logsumexp(
                [la[t-1,s0] + np.log(P[t-1,s0,s1]+1e-15) for s0 in range(2)])
    ll = float(logsumexp(la[-1]))
    return ll if np.isfinite(ll) else -1e6

def neg_loglik(theta, groups, has_prec):
    return -sum(firm_loglik(theta, fd, has_prec) for _, fd in groups)

# =============================================================================
# 4.  パネル構築ヘルパー
# =============================================================================
def build_groups(panel_sub, Y_COLS):
    out = []
    for firm, grp in panel_sub.groupby("firm"):
        grp = grp.sort_values("contract_fy")
        if len(grp) < 2:
            continue
        out.append((firm, {
            "x_seq": grp[X_COLS].to_numpy(float),
            "y_seq": grp[Y_COLS].to_numpy(float),
        }))
    return out

def subset_top(panel, n):
    if n is None:
        return panel.copy()
    keep = panel["firm"].value_counts().head(n).index
    return panel[panel["firm"].isin(keep)].copy()

# =============================================================================
# 5.  データ読み込み
# =============================================================================
print("=" * 62)
print("  データ読み込み")
print("=" * 62)

def pick(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

allbids_path  = pick(ALL_BIDS_CANDIDATES)
analysis_path = pick(ANALYSIS_CANDIDATES)
if allbids_path is None:
    raise FileNotFoundError(f"all_bids_dataset.csv が見つかりません。\n検索: {ALL_BIDS_CANDIDATES}")
print(f"  all_bids : {allbids_path}")
print(f"  analysis : {analysis_path}")

allbids  = pd.read_csv(allbids_path,  encoding="utf-8-sig", low_memory=False)
analysis = pd.read_csv(analysis_path, encoding="utf-8-sig", low_memory=False) \
           if analysis_path else None

# scoring
if "scoring" not in allbids.columns:
    for c in ["scoring_flag","総合評価の有無"]:
        if c in allbids.columns:
            sf = allbids[c]
            allbids["scoring"] = (sf.astype(str).str.strip().isin(
                ["有","1","True","true","yes"]).astype(int)
                if sf.dtype==object
                else pd.to_numeric(sf, errors="coerce").fillna(0))
            break
    else:
        allbids["scoring"] = 0

# contract_fy
for c in ["contract_fy","bid_fy","fy","年度"]:
    if c in allbids.columns:
        allbids["contract_fy"] = pd.to_numeric(allbids[c], errors="coerce"); break

# bid_rate
if "bid_rate" not in allbids.columns:
    allbids["bid_rate"] = (
        pd.to_numeric(allbids.get("final_bid"),      errors="coerce")
      / pd.to_numeric(allbids.get("estimate_price"), errors="coerce"))

# precision: margin in [0, 0.005]  ← v6 と同じ定義
if "precision" not in allbids.columns:
    for gc in ["precision_gap","threshold_margin"]:
        if gc in allbids.columns:
            gap = pd.to_numeric(allbids[gc], errors="coerce")
            allbids["precision"] = ((gap >= 0) & (gap <= 0.005)).astype(float)
            break

# log_backlog_raw
if "log_backlog_raw" not in allbids.columns:
    if "w_it" in allbids.columns:
        allbids["log_backlog_raw"] = pd.to_numeric(allbids["w_it"], errors="coerce")
    elif "estimate_price" in allbids.columns:
        allbids["log_backlog_raw"] = np.log1p(
            pd.to_numeric(allbids["estimate_price"], errors="coerce").clip(lower=1))

# firm
for c in ["firm","落札者","入札者","bidder"]:
    if c in allbids.columns:
        allbids["firm"] = allbids[c].apply(normalize_firm); break

print(f"  allbids rows : {len(allbids):,}")

# h_it from analysis
analysis_fy = None
if analysis is not None:
    if "firm" in analysis.columns:
        analysis["firm"] = analysis["firm"].apply(normalize_firm)
    for c in ["contract_fy","bid_fy","fy"]:
        if c in analysis.columns:
            analysis["contract_fy"] = pd.to_numeric(analysis[c], errors="coerce"); break
    for hc in ["perf_avg","h_it"]:
        if hc in analysis.columns:
            analysis_fy = (
                analysis[["firm","contract_fy",hc]].dropna()
                .groupby(["firm","contract_fy"])[hc].mean().reset_index()
                .rename(columns={hc:"h_from_analysis"})
            )
            print(f"  h source : analysis['{hc}']")
            break

# =============================================================================
# 6.  パネル構築 (v6 と同一の前処理)
# =============================================================================
print("\n  パネル構築中 ...")

agg = {
    "scoring_rate": ("scoring",         "mean"),
    "bid_rate":     ("bid_rate",        "mean"),
    "log_backlog":  ("log_backlog_raw", "mean"),
}
if "precision" in allbids.columns:
    agg["prec_rate"] = ("precision", "mean")

panel = allbids.groupby(["firm","contract_fy"]).agg(**agg).reset_index()

if analysis_fy is not None:
    panel = panel.merge(analysis_fy, on=["firm","contract_fy"], how="left")
    panel["h_2yr"] = panel["h_from_analysis"]
else:
    panel["h_2yr"] = np.nan

def safe_fill(s):
    mn = s.mean() if s.notna().any() else 0.0
    return s.fillna(mn)

panel["h"]      = zscore_vec(safe_fill(panel["h_2yr"]).values)
panel["w"]      = zscore_vec(safe_fill(panel["log_backlog"]).values)
panel["post09"] = (panel["contract_fy"] >= REFORM_2009).astype(int)
panel["post14"] = (panel["contract_fy"] >= REFORM_2014).astype(int)
panel["post23"] = (panel["contract_fy"] >= REFORM_2023).astype(int)
panel["x0"]     = 1.0
panel["scoring_rate"] = clip01(pd.to_numeric(panel["scoring_rate"], errors="coerce"))
panel["bid_rate"]     = pd.to_numeric(panel["bid_rate"], errors="coerce")

HAS_PREC = (
    "prec_rate" in panel.columns
    and panel["prec_rate"].notna().mean() > 0.3
)
if HAS_PREC:
    panel["prec_rate"] = clip01(pd.to_numeric(panel["prec_rate"], errors="coerce"))

drop_sub = ["scoring_rate","bid_rate","h","w"] + (["prec_rate"] if HAS_PREC else [])
panel = (panel.dropna(subset=drop_sub)
              .sort_values(["firm","contract_fy"])
              .reset_index(drop=True))

Y_COLS = ["scoring_rate","prec_rate","bid_rate"] if HAS_PREC else ["scoring_rate","bid_rate"]
D      = param_dims(HAS_PREC)["total"]

print(f"  パネル: {panel['firm'].nunique():,} firms, {len(panel):,} rows")
print(f"  HAS_PREC={HAS_PREC}, Y_COLS={Y_COLS}, θ次元={D}")
assert D == 53, f"θ次元不一致: {D} ≠ 53 (has_prec={HAS_PREC})"

# =============================================================================
# 7.  点推定値の読み込みと検証
# =============================================================================
print(f"\n  theta_hat 読み込み: {THETA_HAT_PATH}")
if not os.path.exists(THETA_HAT_PATH):
    raise FileNotFoundError(
        f"vf_v6_theta_stage2.npy が見つかりません。\n"
        f"SAVE_DIR を確認してください: {SAVE_DIR}"
    )

theta_hat = np.load(THETA_HAT_PATH)
assert len(theta_hat) == D, f"theta_hat 次元{len(theta_hat)} ≠ {D}"
pt_hat = extract_key(theta_hat, HAS_PREC)

# 検証
print(f"\n  ── 点推定値 (変換後) ── [期待値]")
checks = [
    ("β",       "beta",     0.6057, 0.001),
    ("κ",       "kappa",    1.1666, 0.001),
    ("u_LOW_w", "u_LOW_w", -0.2824, 0.001),
    ("u_HIGH_w","u_HIGH_w", 0.2824, 0.001),
]
for name, key, exp, tol in checks:
    val = pt_hat[key]
    flag = "✓" if abs(val - exp) < tol else "⚠"
    print(f"  {flag} {name:<12} = {val:8.4f}   [期待値 {exp:+.4f}]")
print(f"    u_LOW_h   = {pt_hat['u_LOW_h']:8.4f}")
print(f"    u_HIGH_h  = {pt_hat['u_HIGH_h']:8.4f}")
print(f"    sig_sc    = {pt_hat['sig_sc']:8.4f}")
print(f"    sig_pr    = {pt_hat['sig_pr']:8.4f}")
print(f"    sig_bd    = {pt_hat['sig_bd']:8.4f}")

# bounds (v6 と同一)
d   = param_dims(HAS_PREC)
BND_KAPPA = (np.log(1e-4), np.log(20.0))
BND_SIG   = (np.log(1e-4), np.log(10.0))
bounds = (
    [(None, None)]  * d["n_beta"] +
    [BND_KAPPA]     * d["n_kap"]  +
    [(None, None)]  * d["n_u"]    +
    [(None, None)]  * d["n_m"]    +
    [BND_SIG]       * d["n_sig"]
)
assert len(bounds) == D

# =============================================================================
# 8.  ブートストラップ
# =============================================================================
print(f"\n{'=' * 62}")
print(f"  Bootstrap  N={N_BOOT}, SEED={SEED}")
print(f"  STAGED_FIRMS={STAGED_FIRMS}, BOOT_MAXITER={BOOT_MAXITER}")
print(f"{'=' * 62}")

firm_names = panel["firm"].unique().tolist()
rng        = np.random.default_rng(SEED)
boot_conv  = []
n_failed   = 0
t0         = time.time()

for b in tqdm(range(N_BOOT), desc="Bootstrap"):

    # リサンプリング
    draw = rng.choice(firm_names, size=len(firm_names), replace=True)
    rows = []
    for j, nm in enumerate(draw):
        sub = panel[panel["firm"] == nm].copy()
        sub["firm"] = f"b{b}f{j}"
        rows.append(sub)
    bpanel = pd.concat(rows, ignore_index=True)

    # staged L-BFGS-B (初期値 = 点推定値)
    th, ok = theta_hat.copy(), True
    for n_s in STAGED_FIRMS:
        grps = build_groups(subset_top(bpanel, n_s), Y_COLS)
        if len(grps) < 5:
            ok = False; break
        obj = lambda t, g=grps: neg_loglik(t, g, HAS_PREC)
        try:
            res = minimize(obj, x0=th, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": BOOT_MAXITER, "disp": False,
                                    "ftol": 1e-9, "gtol": 1e-6})
            th = res.x
        except Exception:
            ok = False; break

    if not ok:
        n_failed += 1
        boot_conv.append({k: np.nan for k in pt_hat})
        continue
    try:
        boot_conv.append(extract_key(th, HAS_PREC))
    except Exception:
        n_failed += 1
        boot_conv.append({k: np.nan for k in pt_hat})

elapsed = time.time() - t0
print(f"\n  完了: {N_BOOT}反復 ({elapsed/60:.1f}分), 失敗={n_failed}")

# =============================================================================
# 9.  結果集計
# =============================================================================
boot_df = pd.DataFrame(boot_conv)
boot_df.to_csv(OUT_RAW_CSV, index=False, encoding="utf-8-sig")

ALPHA = 0.05
label_map = {
    "beta":      "Discount factor β",
    "kappa":     "Switching cost κ",
    "u_LOW_w":   "Backlog u_w(L) [Low/PO]",
    "u_HIGH_w":  "Backlog u_w(H) [High/CE]",
    "u_LOW_h":   "CPR coeff u_h(L) [Low]",
    "u_HIGH_h":  "CPR coeff u_h(H) [High]",
    "u_LOW_x0":  "Intercept u_0(L)",
    "u_HIGH_x0": "Intercept u_0(H)",
    "sig_sc":    "σ (scoring rate)",
    "sig_pr":    "σ (precision rate)",
    "sig_bd":    "σ (bid rate)",
}

rows_out, ci_warn = [], []
for param, est in pt_hat.items():
    col   = boot_df[param].dropna().values
    n_val = len(col)
    if n_val < max(10, N_BOOT//4):
        se = ci_lo = ci_hi = np.nan
    else:
        se    = float(np.std(col, ddof=1))
        ci_lo = float(np.percentile(col, 100*ALPHA/2))
        ci_hi = float(np.percentile(col, 100*(1-ALPHA/2)))
        if not (ci_lo <= est <= ci_hi):
            ci_warn.append(
                f"  ⚠ {param}: 点推定={est:.5f}, CI=[{ci_lo:.5f},{ci_hi:.5f}], "
                f"中央値={np.median(col):.5f}")
    rows_out.append({"label":label_map.get(param,param), "param":param,
                     "estimate":est, "boot_se":se,
                     "ci_lower":ci_lo, "ci_upper":ci_hi, "n_valid":n_val})

summ = pd.DataFrame(rows_out)
summ.to_csv(OUT_SUMM_CSV, index=False, encoding="utf-8-sig")

# ── コンソール表示 ──────────────────────────────────────────────────────────
SEP = "=" * 78
print(f"\n{SEP}")
print(f"  Bootstrap Results  (N={N_BOOT}, 95% Percentile CI)")
print(SEP)
print(f"  {'Parameter':<36} {'Estimate':>9} {'Boot SE':>9} {'95% CI':>24}  n")
print(f"  {'-'*36} {'-'*9} {'-'*9} {'-'*24}  -")
for _, r in summ.iterrows():
    se_s = f"{r['boot_se']:9.4f}" if not np.isnan(r["boot_se"]) else "        —"
    ci_s = f"[{r['ci_lower']:7.4f},{r['ci_upper']:8.4f}]" \
           if not np.isnan(r["boot_se"]) else "                       —"
    flag = ""
    if not np.isnan(r["boot_se"]) and not (r["ci_lower"]<=r["estimate"]<=r["ci_upper"]):
        flag = " ⚠"
    print(f"  {r['label']:<36} {r['estimate']:>9.4f} {se_s}  {ci_s}  {int(r['n_valid'])}{flag}")
print(SEP)
if ci_warn:
    print("\n  ─── CI 警告 ───")
    for w in ci_warn: print(w)

# ── Table 7 ──────────────────────────────────────────────────────────────────
T7 = ["beta","kappa","u_LOW_w","u_HIGH_w","sig_sc","sig_pr","sig_bd"]
print(f"\n{'─'*78}")
print(f"  Table 7: Bootstrap Standard Errors (論文挿入用)")
print(f"{'─'*78}")
print(f"  {'Parameter':<36} {'Estimate':>9} {'SE':>7} {'95% CI':>24}")
print(f"  {'-'*36} {'-'*9} {'-'*7} {'-'*24}")
for _, r in summ[summ["param"].isin(T7)].iterrows():
    if np.isnan(r["boot_se"]):
        print(f"  {r['label']:<36} {r['estimate']:>9.4f}       —              —")
    else:
        print(f"  {r['label']:<36} {r['estimate']:>9.4f} "
              f"{r['boot_se']:>7.4f}  [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
print(f"{'─'*78}")
print(f"  Notes: Firm-level nonparametric bootstrap, N={N_BOOT} replications.")
print(f"  Staged L-BFGS-B on top-{STAGED_FIRMS[-1]} firms; 95% CI = percentile interval.")

# =============================================================================
# 10.  β 収束プロット
# =============================================================================
beta_draws = boot_df["beta"].dropna().values

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
est_b = pt_hat["beta"]

# トレース
axes[0].plot(beta_draws, lw=0.7, alpha=0.8, color="#3a7abf")
axes[0].axhline(est_b, color="crimson", lw=1.8, ls="--",
                label=f"θ̂={est_b:.4f}")
axes[0].set_title("β: Trace", fontsize=10)
axes[0].set_xlabel("Bootstrap draw"); axes[0].set_ylabel("β")
axes[0].legend(fontsize=9)

# 累積平均
cumavg = np.cumsum(beta_draws)/np.arange(1,len(beta_draws)+1)
axes[1].plot(cumavg, lw=1.5, color="#3a7abf")
axes[1].axhline(est_b, color="crimson", lw=1.8, ls="--",
                label=f"θ̂={est_b:.4f}")
axes[1].set_title("β: Cumulative mean", fontsize=10)
axes[1].set_xlabel("Bootstrap draw"); axes[1].set_ylabel("Cum. mean")
axes[1].legend(fontsize=9)

# 分布
ci_lo_b = np.percentile(beta_draws, 2.5)
ci_hi_b = np.percentile(beta_draws, 97.5)
axes[2].hist(beta_draws, bins=30, color="#3a7abf", edgecolor="white", alpha=0.85)
axes[2].axvline(est_b, color="crimson", lw=2, label=f"θ̂={est_b:.4f}")
axes[2].axvline(ci_lo_b, color="gray", lw=1.2, ls="--",
                label=f"95%CI [{ci_lo_b:.4f},{ci_hi_b:.4f}]")
axes[2].axvline(ci_hi_b, color="gray", lw=1.2, ls="--")
ok_ci = ci_lo_b <= est_b <= ci_hi_b
axes[2].set_facecolor("white" if ok_ci else "#fff0f0")
axes[2].set_title("β: Distribution " + ("✓" if ok_ci else "⚠"), fontsize=10,
                  color="black" if ok_ci else "red")
axes[2].legend(fontsize=8)

plt.suptitle(f"Bootstrap Convergence Check  (N={N_BOOT})", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_TRACE_PNG, dpi=150, bbox_inches="tight"); plt.close()
print(f"\n  β収束プロット    → {OUT_TRACE_PNG}")

# =============================================================================
# 11.  全主要パラメータ分布
# =============================================================================
plot_params = ["beta","kappa","u_LOW_w","u_HIGH_w","u_LOW_h","u_HIGH_h"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, param in zip(axes.flatten(), plot_params):
    col = boot_df[param].dropna().values
    est = pt_hat[param]
    if len(col) < 5:
        ax.set_visible(False); continue
    ax.hist(col, bins=30, color="#4878CF", edgecolor="white", alpha=0.85)
    ax.axvline(est, color="crimson", lw=2, label=f"θ̂={est:.4f}")
    lo, hi = np.percentile(col, 2.5), np.percentile(col, 97.5)
    ax.axvline(lo, color="gray", lw=1.2, ls="--")
    ax.axvline(hi, color="gray", lw=1.2, ls="--")
    if not (lo <= est <= hi):
        ax.set_facecolor("#fff0f0")
    ax.set_title(label_map.get(param, param), fontsize=8)
    ax.legend(fontsize=7)
    ax.set_xlabel(f"95%CI [{lo:.3f}, {hi:.3f}]", fontsize=7)

plt.suptitle(
    f"Bootstrap Distributions (N={N_BOOT})\n"
    "Red=point estimate; Dashed=2.5/97.5 percentiles", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIST_PNG, dpi=150, bbox_inches="tight"); plt.close()
print(f"  パラメータ分布プロット → {OUT_DIST_PNG}")

print(f"\n  出力ファイル:")
for f in [OUT_RAW_CSV, OUT_SUMM_CSV, OUT_TRACE_PNG, OUT_DIST_PNG]:
    print(f"    {f}")
print(f"\n  完了。")
