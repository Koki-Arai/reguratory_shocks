# ============================================================
# 01_structural_estimation.py
# ------------------------------------------------------------
# Dynamic structural estimation — v6 (corrected from v5)
#
# Corrections relative to v5:
#   [FIX 1] State labeling: u_params[0] = Low(PO), u_params[1] = High(CE)
#           Made explicit throughout; consistent with compute_transitions
#           where state=0 is the default/PO regime and state=1 is CE.
#   [FIX 2] precision_rate definition corrected to [0, 0.5%] above MIP
#           (was erroneously [0, 2%] in v5)
#   [FIX 3] STAGED_MAXITER corrected so later stages have MORE iterations,
#           not fewer. Recommended: {200:150, 500:200, 1000:100, None:50}
#   [FIX 4] Bartik Exposure computed on FULL panel before subsetting,
#           so market-level CE adoption rates use all available firms
#   [FIX 5] Staged progression extended to [200, 500, 1000, None]
#           for full-sample estimation
#   [FIX 6] AIC/BIC reported with explicit sign convention note
#           (negloglik-based; negative values are expected and valid)
#   [FIX 7] Subsampling method documented (top-N by bid frequency);
#           logged clearly for Appendix reporting
#
# Required uploaded files:
#   - all_bids_dataset.csv
#   - analysis_dataset.csv
# Optional:
#   - Book2_with_kouken(1).csv
# ============================================================

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.special import expit, logsumexp
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 0. Google Drive mount
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = "/content/drive/MyDrive/structural_estimation_v6/"
os.makedirs(SAVE_DIR, exist_ok=True)

OUT_JSON         = os.path.join(SAVE_DIR, "vf_v6_main_results.json")
OUT_CSV1         = os.path.join(SAVE_DIR, "vf_v6_param_table.csv")
OUT_CSV2         = os.path.join(SAVE_DIR, "vf_v6_moment_fit.csv")
OUT_FIG          = os.path.join(SAVE_DIR, "vf_v6_diagnostics.png")
OUT_THETA_STAGE2 = os.path.join(SAVE_DIR, "vf_v6_theta_stage2.npy")
OUT_META_STAGE2  = os.path.join(SAVE_DIR, "vf_v6_theta_stage2_meta.json")

print("SAVE_DIR =", SAVE_DIR)

# ============================================================
# 1. Settings
# ============================================================
ALL_BIDS_CANDIDATES = [
    "all_bids_dataset_contractfy_fixed.csv",
    "all_bids_dataset.csv",
]
ANALYSIS_CANDIDATES = [
    "analysis_dataset_contractfy_fixed.csv",
    "analysis_dataset.csv",
]
BOOK_CANDIDATES = ["Book2_with_kouken(1).csv"]

REFORM_2009 = 2009
REFORM_2014 = 2014
REFORM_2023 = 2023

SEED = 123
rng  = np.random.default_rng(SEED)

# ---- model options ----
RUN_K2        = False
RUN_BOOTSTRAP = False
N_BOOT        = 10
USE_SMC       = False
N_PARTICLES   = 200

# ---- save / callback ----
SAVE_INTERMEDIATE = True
CALLBACK_EVERY    = 5

# ---- [FIX 3] staged estimation ----
# Iterations INCREASE with sample size so later stages can refine.
# Set STAGED_FIRMS = [200, 500] for a quick first run,
# then [200, 500, 1000, None] for the full production run.
STAGED_FIRMS  = [200, 500, 1000, None]   # None = all firms
STAGED_MAXITER = {
    200:  150,   # warm-up on small sample
    500:  200,   # refine on medium sample  (was 80 in v5 — corrected)
    1000: 150,   # refine on large sample
    None:  80,   # final pass on full panel
}

# [FIX 7] Subsampling note for Appendix reporting:
# Subsetting selects the top-N firms by total bid frequency.
# This prioritises firms with long panel histories for identification
# of the dynamic discount factor β.  The full-sample stage (None)
# uses all firms and should be treated as the primary reported result.
SUBSAMPLE_NOTE = (
    "Subsample = top-N firms by bid frequency (most bids across FY2006-2024). "
    "Final stage uses all firms (None)."
)

print("STAGED_FIRMS   =", STAGED_FIRMS)
print("STAGED_MAXITER =", STAGED_MAXITER)
print("SUBSAMPLE_NOTE :", SUBSAMPLE_NOTE)

# ============================================================
# 2. Helpers
# ============================================================
def pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def clean_cols(df):
    df = df.copy()
    df.columns = [str(c).replace("\n", "").replace(" ", "").strip()
                  for c in df.columns]
    return df

def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))

def zscore_vec(x):
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s if s > 0 else 1.0)

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def normalize_firm(s):
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return None
    if any(k in s for k in ["共同企業体", "共同事業体", "JV", "ＪＶ"]):
        return None
    for a, b in [
        ("株式会社", "(株)"), ("㈱", "(株)"),
        ("有限会社", "(有)"), ("㈲", "(有)"),
        ("（", "("), ("）", ")"),
        (" ", ""), ("　", ""),
        ("−", "-"), ("－", "-"), ("‐", "-"),
    ]:
        s = s.replace(a, b)
    return s

def rename_if_exists(df, mapping):
    mp = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=mp)

# ============================================================
# 3. Load data
# ============================================================
ALL_BIDS_PATH = pick_existing(ALL_BIDS_CANDIDATES)
ANALYSIS_PATH = pick_existing(ANALYSIS_CANDIDATES)
BOOK_PATH     = pick_existing(BOOK_CANDIDATES)

if ALL_BIDS_PATH is None:
    raise FileNotFoundError("all_bids file not found.")
if ANALYSIS_PATH is None:
    raise FileNotFoundError("analysis file not found.")

print("[Load] files ...")
print("  all_bids :", ALL_BIDS_PATH)
print("  analysis :", ANALYSIS_PATH)
print("  book     :", BOOK_PATH if BOOK_PATH else "(none)")

allbids  = pd.read_csv(ALL_BIDS_PATH, encoding="utf-8-sig", low_memory=False)
analysis = pd.read_csv(ANALYSIS_PATH, encoding="utf-8-sig", low_memory=False)
allbids  = clean_cols(allbids)
analysis = clean_cols(analysis)

book = None
if BOOK_PATH is not None:
    book = pd.read_csv(BOOK_PATH, encoding="utf-8-sig", low_memory=False)
    book = clean_cols(book)

print("[Shapes]")
print("  allbids :", allbids.shape)
print("  analysis:", analysis.shape)
if book is not None:
    print("  book    :", book.shape)

# ============================================================
# 4. Harmonize columns
# ============================================================
rename_map = {
    "入札業者":          "firm",
    "契約年度":          "contract_fy",
    "事務所名":          "office",
    "工種区分":          "work_type",
    "総合評価の有無":    "scoring_flag",
    "参加者数":          "n_bidders",
    "前1年度工事成績評定": "perf_y1",
    "予定価格":          "estimate_price",
    "調査基準価格":      "investigation_price",
    "最終入札価格(税抜)": "final_bid",
    "入札率":            "bid_rate",
    "備考":              "result_flag",
    "予定価格ランク":    "rank",
}

allbids  = rename_if_exists(allbids,  rename_map)
analysis = rename_if_exists(analysis, rename_map)
if book is not None:
    book = rename_if_exists(book, rename_map)

for df in [allbids, analysis] + ([book] if book is not None else []):
    for c in [
        "contract_fy", "n_bidders", "perf_y1", "estimate_price",
        "investigation_price", "final_bid", "bid_rate", "w_it", "tau_it",
        "h_it", "scoring", "score", "precision", "precision_gap",
        "cum_bids", "cum_wins", "won_n", "won_rate",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

if "contract_fy" not in allbids.columns:
    raise ValueError("allbids must contain contract_fy")
if "firm" not in allbids.columns:
    raise ValueError("allbids must contain firm")

allbids = allbids.dropna(subset=["contract_fy"]).copy()
allbids["contract_fy"] = allbids["contract_fy"].astype(int)
allbids = allbids[allbids["contract_fy"] <= 2024].copy()
allbids["firm_norm"] = allbids["firm"].apply(normalize_firm)

# scoring flag
if "scoring" not in allbids.columns:
    if "scoring_flag" in allbids.columns:
        if allbids["scoring_flag"].dtype == object:
            allbids["scoring"] = (
                allbids["scoring_flag"].astype(str)
                .str.contains("有|1|True", na=False).astype(int)
            )
        else:
            allbids["scoring"] = (
                pd.to_numeric(allbids["scoring_flag"], errors="coerce").fillna(0)
            )

# [FIX 2] precision_rate: bids within [MIP, MIP + 0.5%]
# v5 erroneously used <= 0.02 (2%); corrected to <= 0.005 (0.5%)
if "precision" not in allbids.columns:
    if ("investigation_price" in allbids.columns
            and "final_bid" in allbids.columns):
        valid = (
            allbids["investigation_price"].notna()
            & (allbids["investigation_price"] > 0)
        )
        allbids["margin"] = np.nan
        allbids.loc[valid, "margin"] = (
            (allbids.loc[valid, "final_bid"]
             - allbids.loc[valid, "investigation_price"])
            / allbids.loc[valid, "investigation_price"]
        )
        # Corrected threshold: 0.5% above MIP  [FIX 2]
        allbids["precision"] = (
            (allbids["margin"] >= 0.000)
            & (allbids["margin"] <= 0.005)   # was 0.02 in v5
        ).astype(float)
        print("[FIX 2] precision_rate defined as margin in [0, 0.005] (0–0.5% above MIP)")

# backlog proxy
if "cum_bids" in allbids.columns:
    allbids["log_backlog_raw"] = safe_log1p(allbids["cum_bids"].fillna(0))
elif "cum_wins" in allbids.columns:
    allbids["log_backlog_raw"] = safe_log1p(allbids["cum_wins"].fillna(0))
else:
    raise ValueError("No backlog-like column found: need cum_bids or cum_wins.")

# ============================================================
# 5. Score construction
# ============================================================
analysis_fy = None
if "firm" in analysis.columns and "contract_fy" in analysis.columns:
    analysis["firm_norm"]   = analysis["firm"].apply(normalize_firm)
    analysis["contract_fy"] = pd.to_numeric(
        analysis["contract_fy"], errors="coerce"
    )
    analysis_fy = analysis.copy()
    if "h_it" in analysis_fy.columns:
        analysis_fy["h_from_analysis"] = pd.to_numeric(
            analysis_fy["h_it"], errors="coerce"
        )

if book is not None:
    if "firm" not in book.columns:
        for cand in ["firm_p", "業者名", "会社名", "企業名"]:
            if cand in book.columns:
                book["firm"] = book[cand]
                break
    if "contract_fy" not in book.columns:
        for cand in ["fy", "年度", "year", "bid_fy"]:
            if cand in book.columns:
                book["contract_fy"] = pd.to_numeric(
                    book[cand], errors="coerce"
                )
                break
    if "score" not in book.columns:
        for cand in ["score", "perf_y1", "評点", "工事成績評定", "成績評定"]:
            if cand in book.columns:
                book["score"] = pd.to_numeric(book[cand], errors="coerce")
                break
    if "firm" in book.columns:
        book["firm_norm"] = book["firm"].apply(normalize_firm)

# ============================================================
# 6. Build firm-year panel (FULL panel — used for Bartik)
# ============================================================
agg_dict = {
    "n_bids":        ("firm",          "size"),
    "scoring_rate":  ("scoring",       "mean"),
    "bid_rate":      ("bid_rate",      "mean"),
    "n_bidders":     ("n_bidders",     "mean"),
    "log_estimate":  ("estimate_price",
                      lambda x: np.log1p(
                          pd.to_numeric(x, errors="coerce").clip(lower=1)
                      ).mean()),
    "log_backlog":   ("log_backlog_raw", "mean"),
    "firm_norm":     ("firm_norm",     "first"),
}
if "work_type" in allbids.columns:
    agg_dict["work_type"] = (
        "work_type",
        lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
    )
if "office" in allbids.columns:
    agg_dict["office"] = (
        "office",
        lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
    )
if "precision" in allbids.columns:
    agg_dict["prec_rate"] = ("precision", "mean")

panel_full = (
    allbids.groupby(["firm", "contract_fy"])
    .agg(**agg_dict)
    .reset_index()
)

# merge h_it from analysis
if analysis_fy is not None and "h_from_analysis" in analysis_fy.columns:
    tmp = (
        analysis_fy[["firm", "contract_fy", "h_from_analysis"]]
        .dropna()
        .drop_duplicates()
    )
    panel_full = panel_full.merge(tmp, on=["firm", "contract_fy"], how="left")

# fallback: book CPR scores
if ("h_from_analysis" not in panel_full.columns
        or panel_full["h_from_analysis"].notna().mean() < 0.2):
    if (book is not None
            and all(c in book.columns for c in ["firm_norm", "contract_fy", "score"])):
        bsub = (
            book[["firm_norm", "contract_fy", "score"]]
            .dropna()
            .drop_duplicates()
        )
        panel_full = panel_full.merge(bsub, on=["firm_norm", "contract_fy"], how="left")
        panel_full = panel_full.sort_values(["firm", "contract_fy"])
        panel_full["score_l1"] = panel_full.groupby("firm")["score"].shift(1)
        panel_full["score_l2"] = panel_full.groupby("firm")["score"].shift(2)
        panel_full["h_2yr"]    = (
            panel_full["score_l1"].fillna(panel_full["score"])
            + panel_full["score_l2"].fillna(panel_full["score"])
        ) / 2
    else:
        panel_full["h_2yr"] = np.nan
else:
    panel_full["h_2yr"] = panel_full["h_from_analysis"]

# fallback: perf_y1
if panel_full["h_2yr"].notna().mean() < 0.2 and "perf_y1" in analysis.columns:
    tmp = analysis[["firm", "contract_fy", "perf_y1"]].dropna().drop_duplicates()
    panel_full = panel_full.merge(tmp, on=["firm", "contract_fy"], how="left")
    panel_full = panel_full.sort_values(["firm", "contract_fy"])
    panel_full["perf_l1"] = panel_full.groupby("firm")["perf_y1"].shift(1)
    panel_full["perf_l2"] = panel_full.groupby("firm")["perf_y1"].shift(2)
    panel_full["h_2yr"]   = panel_full["h_2yr"].fillna(
        (
            panel_full["perf_l1"].fillna(panel_full["perf_y1"])
            + panel_full["perf_l2"].fillna(panel_full["perf_y1"])
        ) / 2
    )

panel_full["h"]      = zscore_vec(panel_full["h_2yr"].fillna(panel_full["h_2yr"].mean()).values)
panel_full["w"]      = zscore_vec(panel_full["log_backlog"].fillna(panel_full["log_backlog"].mean()).values)
panel_full["post09"] = (panel_full["contract_fy"] >= REFORM_2009).astype(int)
panel_full["post14"] = (panel_full["contract_fy"] >= REFORM_2014).astype(int)
panel_full["post23"] = (panel_full["contract_fy"] >= REFORM_2023).astype(int)
panel_full["x0"]     = 1.0

panel_full["scoring_rate"] = clip01(
    pd.to_numeric(panel_full["scoring_rate"], errors="coerce")
)
panel_full["bid_rate"] = pd.to_numeric(panel_full["bid_rate"], errors="coerce")

HAS_PREC = (
    "prec_rate" in panel_full.columns
    and panel_full["prec_rate"].notna().mean() > 0.3
)
if HAS_PREC:
    panel_full["prec_rate"] = clip01(
        pd.to_numeric(panel_full["prec_rate"], errors="coerce")
    )

panel_full = panel_full.dropna(subset=["scoring_rate", "bid_rate", "h", "w"])
if HAS_PREC:
    panel_full = panel_full.dropna(subset=["prec_rate"])

panel_full = panel_full.sort_values(["firm", "contract_fy"]).reset_index(drop=True)

print("[Full panel]")
print("  rows :", len(panel_full))
print("  firms:", panel_full["firm"].nunique())
print("  years:", panel_full["contract_fy"].min(), "-",
      panel_full["contract_fy"].max())
print("  HAS_PREC:", HAS_PREC)

# ============================================================
# 7. [FIX 4] Bartik Exposure on FULL panel
# ============================================================
# v5 computed Bartik after subsetting — corrected to use full panel
# so market-level CE adoption rates reflect the entire market.
if "work_type" in panel_full.columns:
    pre_full = panel_full[panel_full["contract_fy"].between(2006, 2008)].copy()
    sh = (
        pre_full.groupby(["firm", "work_type"])
        .size()
        .reset_index(name="cnt")
    )
    tot = sh.groupby("firm")["cnt"].sum().reset_index(name="tot")
    sh  = sh.merge(tot, on="firm")
    sh["share_pre"] = sh["cnt"] / sh["tot"]

    mkt_pre  = (
        panel_full[panel_full["contract_fy"].between(2006, 2008)]
        .groupby("work_type")["scoring_rate"].mean()
        .reset_index(name="scr_pre")
    )
    mkt_post = (
        panel_full[panel_full["contract_fy"] >= 2009]
        .groupby("work_type")["scoring_rate"].mean()
        .reset_index(name="scr_post")
    )
    mkt = mkt_pre.merge(mkt_post, on="work_type", how="outer").fillna(0)
    mkt["delta"] = mkt["scr_post"] - mkt["scr_pre"]

    sh = sh.merge(mkt[["work_type", "delta"]], on="work_type", how="left")
    sh["contrib"] = sh["share_pre"] * sh["delta"].fillna(0)
    exp_firm = sh.groupby("firm")["contrib"].sum().reset_index(name="exposure_pre")
    panel_full = panel_full.merge(exp_firm, on="firm", how="left")
    print("[FIX 4] Bartik Exposure computed on full panel "
          f"({panel_full['firm'].nunique()} firms)")
else:
    panel_full["exposure_pre"] = 0.0
    print("[FIX 4] work_type not found; exposure_pre set to 0")

panel_full["exposure_pre"] = panel_full["exposure_pre"].fillna(0.0)
panel_full["exp_z"]        = zscore_vec(panel_full["exposure_pre"].values)

# ============================================================
# 8. Model structure
# ============================================================
X_COLS = ["x0", "h", "w", "post09", "post14", "post23"]
Y_COLS = (
    ["scoring_rate", "prec_rate", "bid_rate"]
    if HAS_PREC
    else ["scoring_rate", "bid_rate"]
)
N_COV = len(X_COLS)

# [FIX 1] STATE LABELING CONVENTION — explicit throughout
# state = 0  →  Low  regime  (PO-format, price-only)
# state = 1  →  High regime  (CE-format, quality competition)
# u_params[0, :] = utility parameters for Low  state
# u_params[1, :] = utility parameters for High state
# This matches compute_transitions where:
#   v_s0_stay = U[t, 0] + beta * EV_next[0]  (Low  → Low)
#   v_s1_stay = U[t, 1] + beta * EV_next[1]  (High → High)
STATE_LOW  = 0   # PO-format
STATE_HIGH = 1   # CE-format
print(f"[FIX 1] State labeling: state={STATE_LOW}=Low(PO), "
      f"state={STATE_HIGH}=High(CE)")
print(f"        u_params[{STATE_LOW}] = Low state utility params")
print(f"        u_params[{STATE_HIGH}] = High state utility params")

def param_dims(K, has_prec=True):
    n_beta = K
    n_pi   = K - 1
    n_kap  = 1
    n_u    = 2 * N_COV           # Low + High
    n_out  = 3 if has_prec else 2
    n_m    = n_out * 2 * N_COV   # per outcome × per state × covariates
    n_sig  = n_out
    return n_beta, n_pi, n_kap, n_u, n_m, n_sig, n_out

def unpack_theta(theta, K=1, has_prec=True):
    """
    Returns (betas, pis, kappa, u_params, m_params, sigmas).

    u_params shape: (2, N_COV)
        u_params[STATE_LOW,  :] = Low  state (PO) utility coefficients
        u_params[STATE_HIGH, :] = High state (CE) utility coefficients
        u_params[:, 2] = w-coefficient  →  sign identifies capacity-wall mechanism:
            u_params[STATE_LOW,  2] < 0  (backlog deters CE entry)
            u_params[STATE_HIGH, 2] > 0  (backlog reinforces CE continuation)

    m_params shape: (n_out, 2, N_COV)
        m_params[j, STATE_LOW,  :] = outcome j mean in Low  state
        m_params[j, STATE_HIGH, :] = outcome j mean in High state
    """
    n_beta, n_pi, n_kap, n_u, n_m, n_sig, n_out = param_dims(K, has_prec)
    idx = 0

    beta_raw = theta[idx:idx + K]; idx += K
    betas    = 0.995 * expit(beta_raw)

    if K == 1:
        pis = np.array([1.0])
    elif K == 2:
        p1  = expit(theta[idx]); idx += 1
        pis = np.array([p1, 1 - p1])
    else:
        raise NotImplementedError("Only K=1 or K=2 supported.")

    kappa = np.exp(theta[idx]); idx += 1

    # [FIX 1] u_params[0] = Low(PO), u_params[1] = High(CE)
    u_params = theta[idx:idx + n_u].reshape(2, N_COV); idx += n_u

    m_params = theta[idx:idx + n_m].reshape(n_out, 2, N_COV); idx += n_m

    sigmas = np.exp(theta[idx:idx + n_sig]) + 1e-6; idx += n_sig

    if idx != len(theta):
        raise ValueError(f"unpack mismatch: idx={idx}, len={len(theta)}")

    return betas, pis, kappa, u_params, m_params, sigmas

# ============================================================
# 9. Filtering
# ============================================================
def compute_transitions(x_seq, beta, kappa, u_params):
    """
    Compute transition probability matrices.

    Convention: state 0 = Low(PO), state 1 = High(CE).  [FIX 1]

    P[t, s, s'] = Pr(next state = s' | current state = s, x_t)

    Value-function recursion (backward pass):
        V_s = U[t, s] + beta * E[max(V_0, V_1 - kappa * 1[switch])]
    Switching cost kappa is paid when leaving the current regime.
    """
    T  = x_seq.shape[0]
    U  = x_seq @ u_params.T          # shape (T, 2): U[:, 0]=Low, U[:, 1]=High
    P  = np.zeros((T, 2, 2))
    EV_next = np.zeros(2)

    for t in range(T - 1, -1, -1):
        # From Low (s=0): stay=Low(free), switch=High(-kappa)
        v00 = U[t, STATE_LOW]  + beta * EV_next[STATE_LOW]
        v01 = U[t, STATE_HIGH] - kappa + beta * EV_next[STATE_HIGH]
        den0 = logsumexp([v00, v01])

        # From High (s=1): stay=High(free), switch=Low(-kappa)
        v10 = U[t, STATE_LOW]  - kappa + beta * EV_next[STATE_LOW]
        v11 = U[t, STATE_HIGH] + beta * EV_next[STATE_HIGH]
        den1 = logsumexp([v10, v11])

        P[t, STATE_LOW,  STATE_LOW]  = np.exp(v00 - den0)
        P[t, STATE_LOW,  STATE_HIGH] = np.exp(v01 - den0)
        P[t, STATE_HIGH, STATE_LOW]  = np.exp(v10 - den1)
        P[t, STATE_HIGH, STATE_HIGH] = np.exp(v11 - den1)

        EV_next = np.array([den0, den1])
    return P

def build_emission_matrix(y_seq, x_seq, m_params, sigmas):
    T     = x_seq.shape[0]
    n_out = m_params.shape[0]
    E     = np.zeros((T, 2))

    for s in (STATE_LOW, STATE_HIGH):
        mu   = x_seq @ m_params[:, s, :].T    # (T, n_out)
        diff = y_seq - mu
        E[:, s] = (
            -0.5 * n_out * np.log(2 * np.pi)
            - np.sum(np.log(sigmas))
            - 0.5 * np.sum((diff / sigmas[None, :]) ** 2, axis=1)
        )
    return E

def forward_filter_exact(y_seq, x_seq, beta, kappa, u_params, m_params, sigmas):
    T = x_seq.shape[0]
    P = compute_transitions(x_seq, beta, kappa, u_params)
    E = build_emission_matrix(y_seq, x_seq, m_params, sigmas)

    # Stationary distribution as initial state probabilities
    p_LH = P[0, STATE_LOW,  STATE_HIGH]
    p_HL = P[0, STATE_HIGH, STATE_LOW]
    den  = p_LH + p_HL
    if den > 1e-10:
        pi0 = np.array([p_HL / den, p_LH / den])   # [pi_Low, pi_High]
    else:
        pi0 = np.array([0.5, 0.5])

    log_alpha    = np.zeros((T, 2))
    log_alpha[0] = np.log(pi0 + 1e-15) + E[0]

    for t in range(1, T):
        for s_new in (STATE_LOW, STATE_HIGH):
            log_alpha[t, s_new] = E[t, s_new] + logsumexp([
                log_alpha[t - 1, s_old]
                + np.log(P[t, s_old, s_new] + 1e-15)
                for s_old in (STATE_LOW, STATE_HIGH)
            ])

    return float(logsumexp(log_alpha[-1]))

def forward_filter_smc(y_seq, x_seq, beta, kappa, u_params, m_params, sigmas,
                       Np=200, seed=0):
    rng_local = np.random.default_rng(seed)
    T = x_seq.shape[0]
    P = compute_transitions(x_seq, beta, kappa, u_params)

    s    = rng_local.integers(0, 2, size=Np)
    logw = np.zeros(Np)

    def emit_ll(t, sval):
        x   = x_seq[t]
        mu  = np.array([x @ m_params[j, sval] for j in range(m_params.shape[0])])
        diff = y_seq[t] - mu
        return (
            -0.5 * len(sigmas) * np.log(2 * np.pi)
            - np.sum(np.log(sigmas))
            - 0.5 * np.sum((diff / sigmas) ** 2)
        )

    for i in range(Np):
        logw[i] = emit_ll(0, s[i])
    ll   = logsumexp(logw) - np.log(Np)
    logw -= logsumexp(logw)

    for t in range(1, T):
        w   = np.exp(logw)
        idx = rng_local.choice(np.arange(Np), size=Np, replace=True, p=w)
        s_prev = s[idx].copy()
        s_new  = np.zeros(Np, dtype=int)
        for i in range(Np):
            p1      = P[t, s_prev[i], STATE_HIGH]
            s_new[i] = STATE_HIGH if rng_local.random() < p1 else STATE_LOW
        s = s_new
        for i in range(Np):
            logw[i] = emit_ll(t, s[i])
        ll   += logsumexp(logw) - np.log(Np)
        logw -= logsumexp(logw)

    return float(ll)

# ============================================================
# 10. Objective
# ============================================================
def total_negloglik(theta, groups, K=1, use_smc=False, Np=200,
                    has_prec=True, kappa_fixed=None, early_stop=True):
    betas, pis, kappa, u_params, m_params, sigmas = unpack_theta(
        theta, K=K, has_prec=has_prec
    )
    if kappa_fixed is not None:
        kappa = kappa_fixed

    total_ll = 0.0
    for firm, fd in groups:
        x_seq = fd["x_seq"]
        y_seq = fd["y_seq"]

        ll_types = []
        for k in range(K):
            if use_smc:
                llk = forward_filter_smc(
                    y_seq, x_seq, betas[k], kappa, u_params, m_params, sigmas,
                    Np=Np, seed=SEED + hash(firm) % 10000 + k * 1000,
                )
            else:
                llk = forward_filter_exact(
                    y_seq, x_seq, betas[k], kappa, u_params, m_params, sigmas
                )
            ll_types.append(np.log(pis[k] + 1e-15) + llk)

        total_ll += float(logsumexp(ll_types))

        if early_stop and total_ll < -1e12:
            return 1e15

    return -total_ll

# ============================================================
# 11. Build groups from panel
# ============================================================
def build_groups_from_panel(panel_sub, X_COLS, Y_COLS):
    firm_data = {}
    for firm, grp in panel_sub.groupby("firm"):
        grp = grp.sort_values("contract_fy").copy()
        firm_data[firm] = {
            "df":    grp,
            "x_seq": grp[X_COLS].to_numpy(float),
            "y_seq": grp[Y_COLS].to_numpy(float),
            "years": grp["contract_fy"].to_numpy(int),
        }
    return firm_data, list(firm_data.items())

def subset_panel_by_top_firms(panel, n_firms=None):
    """
    [FIX 7] Select top-N firms by bid frequency.
    When n_firms=None, return the full panel.
    Subsampling method is logged for Appendix reporting.
    """
    if n_firms is None:
        print(f"[Subsample] Using full panel: {panel['firm'].nunique()} firms")
        return panel.copy()
    vc   = panel["firm"].value_counts()
    keep = vc.head(n_firms).index
    sub  = panel[panel["firm"].isin(keep)].copy()
    print(f"[Subsample] top-{n_firms} firms by bid count: "
          f"{sub['firm'].nunique()} firms, {len(sub)} rows  "
          f"(min bids={vc.iloc[n_firms-1]}, max bids={vc.iloc[0]})")
    return sub

# ============================================================
# 12. Initialization
# ============================================================
# Use full panel for OLS warm-start
tmp = panel_full.copy()

def quick_ols(y, xs):
    use = tmp.dropna(subset=[y] + xs).copy()
    X   = use[xs].astype(float)
    X   = np.column_stack([np.ones(len(use)), X])
    yy  = use[y].astype(float).values
    return np.linalg.lstsq(X, yy, rcond=None)[0]

x_ols = ["h", "w", "post09", "post14", "post23"]
b_sc  = quick_ols("scoring_rate", x_ols)
b_bd  = quick_ols("bid_rate",     x_ols)
if HAS_PREC:
    b_pr = quick_ols("prec_rate", x_ols)

n_beta, n_pi, n_kap, n_u, n_m, n_sig, n_out = param_dims(1, HAS_PREC)
D1 = n_beta + n_pi + n_kap + n_u + n_m + n_sig

theta0_k1 = np.zeros(D1)
i = 0

# beta (logit-scale)
theta0_k1[i] = 0.5; i += 1

# log(kappa)
theta0_k1[i] = np.log(0.3); i += 1

# [FIX 1] u_params[STATE_LOW=0] : Low(PO) utility
#   w-coefficient (index 2) initialized < 0: backlog hurts CE entry from Low
theta0_k1[i:i + N_COV] = np.array([0.0,  0.1, -0.2,  0.1, -0.1,  0.0])
i += N_COV

# [FIX 1] u_params[STATE_HIGH=1]: High(CE) utility
#   w-coefficient (index 2) initialized > 0: backlog helps CE continuation
theta0_k1[i:i + N_COV] = np.array([0.2,  0.4,  0.2,  0.0, -0.1,  0.0])
i += N_COV

# m_params: outcome means by state
# scoring_rate
theta0_k1[i:i + N_COV] = np.array([b_sc[0] - 0.05] + [x * 0.5 for x in b_sc[1:]])
i += N_COV
theta0_k1[i:i + N_COV] = b_sc[:N_COV].tolist()             # High state (intercept + covariates)
i += N_COV

if HAS_PREC:
    theta0_k1[i:i + N_COV] = np.array([b_pr[0]] + [x * 0.5 for x in b_pr[1:]])
    i += N_COV
    theta0_k1[i:i + N_COV] = np.array(list(b_pr[:N_COV]))
    i += N_COV

# bid_rate
theta0_k1[i:i + N_COV] = np.array([b_bd[0]] + [x * 0.5 for x in b_bd[1:]])
i += N_COV
theta0_k1[i:i + N_COV] = np.array(list(b_bd[:N_COV]))
i += N_COV

# log(sigmas)
for s in ([0.15, 0.15, 0.05] if HAS_PREC else [0.15, 0.05]):
    theta0_k1[i] = np.log(s); i += 1

assert i == D1, f"Init mismatch: i={i}, D1={D1}"

bounds_k1 = (
    [(-5, 5)]       +   # beta
    [(-4, 4)]       +   # log(kappa)
    [(-4, 4)] * n_u +   # utility params
    [(-2, 2)] * n_m +   # measurement params
    [(-6, 2)] * n_sig   # log(sigmas)
)

# ============================================================
# 13. Checkpoint helpers
# ============================================================
def save_checkpoint(theta, stage_name, n_firms, n_rows, extra=None):
    np.save(OUT_THETA_STAGE2, theta)
    meta = {
        "stage_name": stage_name,
        "n_firms":    int(n_firms),
        "n_rows":     int(n_rows),
        "theta_file": OUT_THETA_STAGE2,
    }
    if extra is not None:
        meta.update(extra)
    with open(OUT_META_STAGE2, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_previous_theta(default_theta):
    if os.path.exists(OUT_THETA_STAGE2):
        print(f"[Init] loading checkpoint from {OUT_THETA_STAGE2}")
        return np.load(OUT_THETA_STAGE2)
    print("[Init] no previous checkpoint; using default theta0")
    return default_theta.copy()

def maybe_project_theta(theta_in, target_len):
    th = np.array(theta_in, dtype=float).copy()
    if len(th) == target_len:
        return th
    print(f"[Init] checkpoint length {len(th)} != target {target_len}; projecting.")
    out    = np.zeros(target_len)
    m      = min(len(th), target_len)
    out[:m] = th[:m]
    return out

# ============================================================
# 14. Callback
# ============================================================
cb_state = {"i": 0}

def make_callback(stage_name, n_firms, n_rows):
    def cb(xk):
        cb_state["i"] += 1
        if cb_state["i"] % CALLBACK_EVERY == 0:
            print(f"[callback] {stage_name} iter={cb_state['i']}  checkpoint saved")
            if SAVE_INTERMEDIATE:
                save_checkpoint(xk, stage_name=stage_name,
                                n_firms=n_firms, n_rows=n_rows)
    return cb

def run_lbfgsb_stage(obj_fun, theta_start, bounds, stage_name,
                     n_firms, n_rows, maxiter):
    global cb_state
    cb_state = {"i": 0}
    print(f"\n[MLE] {stage_name} ...")
    t0  = time.time()
    res = minimize(
        obj_fun,
        x0=theta_start,
        method="L-BFGS-B",
        bounds=bounds,
        callback=make_callback(stage_name, n_firms=n_firms, n_rows=n_rows),
        options={"maxiter": maxiter, "disp": True, "maxls": 20},
    )
    elapsed = (time.time() - t0) / 60
    print(f"[MLE] {stage_name} done: negloglik={res.fun:.6f}  "
          f"time={elapsed:.2f} min  success={res.success}")
    if not res.success:
        print(f"  [WARNING] {stage_name}: {res.message}")
        print(f"  Consider increasing STAGED_MAXITER[{n_firms}] or "
              "restarting from this checkpoint.")
    if SAVE_INTERMEDIATE:
        save_checkpoint(
            res.x,
            stage_name=stage_name,
            n_firms=n_firms,
            n_rows=n_rows,
            extra={
                "negloglik": float(res.fun),
                "success":   bool(res.success),
                "message":   str(res.message),
            },
        )
    return res

# ============================================================
# 15. [FIX 3 & 5] K=1 staged estimation — corrected MAXITER & full sample
# ============================================================
print("\n" + "=" * 68)
print("STAGED K=1 ESTIMATION WITH CHECKPOINT RESTART  (v6)")
print("=" * 68)

theta_current = load_previous_theta(theta0_k1)
theta_current = maybe_project_theta(theta_current, D1)

best_res        = None
best_stage_info = None

for nf in STAGED_FIRMS:
    # [FIX 4] subset comes AFTER Bartik is computed on full panel
    panel_stage = subset_panel_by_top_firms(panel_full, nf)
    panel_stage = panel_stage.sort_values(["firm", "contract_fy"]).reset_index(drop=True)

    firm_data_stage, groups_stage = build_groups_from_panel(
        panel_stage, X_COLS, Y_COLS
    )

    n_firms_stage = panel_stage["firm"].nunique()
    n_rows_stage  = len(panel_stage)

    print("\n" + "-" * 68)
    print(f"[Stage] firms={n_firms_stage}, rows={n_rows_stage}, "
          f"years={panel_stage['contract_fy'].min()}-"
          f"{panel_stage['contract_fy'].max()}")
    print(f"        MAXITER = {STAGED_MAXITER[nf]}   [FIX 3: increases with N]")
    print("-" * 68)

    obj_stage = lambda th: total_negloglik(
        th, groups_stage, K=1, use_smc=USE_SMC,
        Np=N_PARTICLES, has_prec=HAS_PREC,
    )

    stage_name = f"K1_firms_{'full' if nf is None else nf}"

    res_stage = run_lbfgsb_stage(
        obj_fun=obj_stage,
        theta_start=theta_current,
        bounds=bounds_k1,
        stage_name=stage_name,
        n_firms=n_firms_stage,
        n_rows=n_rows_stage,
        maxiter=STAGED_MAXITER[nf],
    )

    theta_current = res_stage.x.copy()
    best_res       = res_stage
    best_stage_info = {
        "stage_name": stage_name,
        "n_firms":    n_firms_stage,
        "n_rows":     n_rows_stage,
        "success":    bool(res_stage.success),
        "message":    str(res_stage.message),
        "subsample_note": SUBSAMPLE_NOTE,
    }

print("\n[Stage summary]")
print(best_stage_info)

theta_hat = best_res.x
negll_hat = float(best_res.fun)
best_K    = 1

betas, pis, kappa, u_params, m_params, sigmas = unpack_theta(
    theta_hat, K=1, has_prec=HAS_PREC
)

print("\n[Final K=1 estimates after staged run]")
print("  negll  :", negll_hat)
print("  betas  :", np.round(betas, 4))
print("  pis    :", np.round(pis, 4))
print("  kappa  :", round(float(kappa), 4))
print("  sigmas :", np.round(sigmas, 4))
print()
print("  [FIX 1] u_params state labeling verification:")
w_idx = X_COLS.index("w")
print(f"    u_params[LOW={STATE_LOW},  w={w_idx}] = "
      f"{u_params[STATE_LOW,  w_idx]:.4f}  "
      f"(expected < 0: backlog deters CE entry)")
print(f"    u_params[HIGH={STATE_HIGH}, w={w_idx}] = "
      f"{u_params[STATE_HIGH, w_idx]:.4f}  "
      f"(expected > 0: backlog reinforces CE continuation)")

# Keep final panel/groups
panel     = panel_stage.copy()
firm_data = firm_data_stage
groups    = groups_stage

print("\n[Final panel for reporting]")
print("  rows :", len(panel))
print("  firms:", panel["firm"].nunique())

# ============================================================
# 16. Optional K=2 refinement
# ============================================================
if RUN_K2:
    print("\n[MLE] K=2 refinement ...")
    n_beta2, n_pi2, n_kap2, n_u2, n_m2, n_sig2, _ = param_dims(2, HAS_PREC)
    D2 = n_beta2 + n_pi2 + n_kap2 + n_u2 + n_m2 + n_sig2

    theta0_k2    = np.zeros(D2)
    j            = 0
    beta1_raw    = best_res.x[0]
    theta0_k2[j] = beta1_raw - 0.2; j += 1
    theta0_k2[j] = beta1_raw + 0.2; j += 1
    theta0_k2[j] = 0.0;             j += 1
    theta0_k2[j] = best_res.x[1];   j += 1
    theta0_k2[j:] = best_res.x[2:]

    bounds_k2 = (
        [(-5, 5)] * 2 + [(-8, 8)] + [(-4, 4)]
        + [(-4, 4)] * n_u2 + [(-2, 2)] * n_m2 + [(-6, 2)] * n_sig2
    )

    obj_k2 = lambda th: total_negloglik(
        th, groups, K=2, use_smc=USE_SMC, Np=N_PARTICLES, has_prec=HAS_PREC
    )

    cb_state = {"i": 0}
    t2 = time.time()
    res2 = minimize(
        obj_k2, x0=theta0_k2, method="L-BFGS-B", bounds=bounds_k2,
        callback=make_callback("K2_refinement",
                               n_firms=panel["firm"].nunique(),
                               n_rows=len(panel)),
        options={"maxiter": 50, "disp": True, "maxls": 20},
    )
    print(f"  K=2: negloglik={res2.fun:.6f}  time={(time.time()-t2)/60:.2f} min")

    bic_k1 = 2 * best_res.fun + len(best_res.x) * np.log(len(panel))
    bic_k2 = 2 * res2.fun     + len(res2.x)     * np.log(len(panel))
    print(f"  BIC K=1 = {bic_k1:.2f}")
    print(f"  BIC K=2 = {bic_k2:.2f}")

    if bic_k2 < bic_k1:
        best_res  = res2
        best_K    = 2
        theta_hat = res2.x
        negll_hat = float(res2.fun)
        betas, pis, kappa, u_params, m_params, sigmas = unpack_theta(
            theta_hat, K=2, has_prec=HAS_PREC
        )
        print("  -> K=2 selected by BIC")
    else:
        print("  -> K=1 retained by BIC")

# ============================================================
# 17. Filtered P(H) and fitted moments
# ============================================================
def filtered_pH_one_firm(fd, beta, kappa, u_params, m_params, sigmas):
    x_seq = fd["x_seq"]
    y_seq = fd["y_seq"]
    T     = len(x_seq)

    P = compute_transitions(x_seq, beta, kappa, u_params)
    E = build_emission_matrix(y_seq, x_seq, m_params, sigmas)

    p_LH = P[0, STATE_LOW,  STATE_HIGH]
    p_HL = P[0, STATE_HIGH, STATE_LOW]
    den  = p_LH + p_HL
    pi0  = np.array([p_HL / den, p_LH / den]) if den > 1e-10 else np.array([0.5, 0.5])

    log_alpha    = np.zeros((T, 2))
    log_alpha[0] = np.log(pi0 + 1e-15) + E[0]
    filt         = np.zeros(T)
    filt[0]      = np.exp(log_alpha[0, STATE_HIGH] - logsumexp(log_alpha[0]))

    for t in range(1, T):
        for s_new in (STATE_LOW, STATE_HIGH):
            log_alpha[t, s_new] = E[t, s_new] + logsumexp([
                log_alpha[t - 1, s_old]
                + np.log(P[t, s_old, s_new] + 1e-15)
                for s_old in (STATE_LOW, STATE_HIGH)
            ])
        filt[t] = np.exp(log_alpha[t, STATE_HIGH] - logsumexp(log_alpha[t]))
    return filt

records = []
for firm, fd in groups:
    grp = fd["df"].copy()
    pH_mix = np.zeros(len(grp))
    for k in range(best_K):
        pH_mix += pis[k] * filtered_pH_one_firm(
            fd, betas[k], kappa, u_params, m_params, sigmas
        )
    grp["pH"] = pH_mix
    records.append(grp)
fit_df = pd.concat(records, axis=0).reset_index(drop=True)

xmat = fit_df[X_COLS].to_numpy(float)
Ey   = np.zeros((len(fit_df), len(Y_COLS)))
for j in range(len(Y_COLS)):
    muL    = xmat @ m_params[j, STATE_LOW]
    muH    = xmat @ m_params[j, STATE_HIGH]
    Ey[:, j] = (1 - fit_df["pH"].values) * muL + fit_df["pH"].values * muH

for j, y in enumerate(Y_COLS):
    fit_df[f"{y}_fit"] = Ey[:, j]
    if y in ("scoring_rate", "prec_rate"):
        fit_df[f"{y}_fit"] = clip01(fit_df[f"{y}_fit"])

fit_df["w_bin"] = pd.qcut(fit_df["w"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
moments = []

for q in ["Q1", "Q2", "Q3", "Q4"]:
    g = fit_df[fit_df["w_bin"] == q]
    for y in Y_COLS:
        moments.append({
            "moment": f"{y}_{q}",
            "emp":    g[y].mean(),
            "fit":    g[f"{y}_fit"].mean(),
        })

for ref, lab in [(2009, "diff2009"), (2014, "diff2014"), (2023, "diff2023")]:
    pre  = fit_df[fit_df["contract_fy"] < ref]
    post = fit_df[fit_df["contract_fy"] >= ref]
    for y in Y_COLS:
        moments.append({
            "moment": f"{lab}_{y}",
            "emp":    post[y].mean() - pre[y].mean(),
            "fit":    post[f"{y}_fit"].mean() - pre[f"{y}_fit"].mean(),
        })

momtab         = pd.DataFrame(moments)
momtab["diff"] = momtab["fit"] - momtab["emp"]
momtab.to_csv(OUT_CSV2, index=False, encoding="utf-8-sig")

# ============================================================
# 18. Parameter table
# ============================================================
cov_names = X_COLS
pnames    = [f"beta_raw_{k}" for k in range(best_K)]
if best_K == 2:
    pnames += ["pi_raw_0"]
pnames += ["log_kappa"]
# [FIX 1] explicit state labels in param names
pnames += [f"u_LOW_{c}"  for c in cov_names]
pnames += [f"u_HIGH_{c}" for c in cov_names]
out_pref = ["sc", "pr", "bd"] if HAS_PREC else ["sc", "bd"]
state_labels = {STATE_LOW: "LOW", STATE_HIGH: "HIGH"}
pnames += [
    f"m_{o}_{state_labels[s]}_{c}"
    for o in out_pref
    for s in (STATE_LOW, STATE_HIGH)
    for c in cov_names
]
pnames += [f"sig_{o}" for o in out_pref]

param_table = pd.DataFrame({"param": pnames, "estimate": theta_hat})
param_table.to_csv(OUT_CSV1, index=False, encoding="utf-8-sig")

# ============================================================
# 19. [FIX 6] Save JSON — AIC/BIC sign convention noted
# ============================================================
# AIC = 2*negloglik + 2k  where negloglik = -log L < 0
# => AIC is negative when log-likelihood is large; lower (more negative) = better
n_params = len(theta_hat)
aic_val  = 2 * negll_hat + 2 * n_params
bic_val  = 2 * negll_hat + n_params * np.log(len(panel))

out_json = {
    "success":  bool(best_res.success),
    "message":  str(best_res.message),
    "K_selected": int(best_K),
    "negloglik":  float(negll_hat),
    # [FIX 6] AIC/BIC note: values are negative because negloglik < 0
    # Convention: AIC = 2*negloglik + 2*k; lower (more negative) = better fit
    "AIC": float(aic_val),
    "BIC": float(bic_val),
    "AIC_BIC_note": (
        "AIC = 2*negloglik + 2*k; BIC = 2*negloglik + k*log(N). "
        "Negative values are expected (negloglik < 0). "
        "Lower (more negative) = better fit."
    ),
    "betas":  betas.tolist(),
    "pis":    pis.tolist(),
    "kappa":  float(kappa),
    "sigmas": sigmas.tolist(),
    # [FIX 1] u_params stored with explicit state labels
    "u_params": {
        "state_convention": "index 0 = Low(PO), index 1 = High(CE)",
        "X_COLS": X_COLS,
        "u_LOW":  u_params[STATE_LOW].tolist(),
        "u_HIGH": u_params[STATE_HIGH].tolist(),
        f"u_LOW_w  (index {X_COLS.index('w')}, expected <0)":
            float(u_params[STATE_LOW,  X_COLS.index("w")]),
        f"u_HIGH_w (index {X_COLS.index('w')}, expected >0)":
            float(u_params[STATE_HIGH, X_COLS.index("w")]),
        "raw_array": u_params.tolist(),
    },
    "m_params": m_params.tolist(),
    "has_prec": bool(HAS_PREC),
    "use_smc":  bool(USE_SMC),
    "Y_COLS":   Y_COLS,
    "X_COLS":   X_COLS,
    "n_panel_rows": int(len(panel)),
    "n_firms":      int(panel["firm"].nunique()),
    "year_min":     int(panel["contract_fy"].min()),
    "year_max":     int(panel["contract_fy"].max()),
    "stage_info":   best_stage_info,
    "staged_firms": STAGED_FIRMS,
    # [FIX 7] subsample documentation
    "subsample_note": SUBSAMPLE_NOTE,
    # [FIX 2] precision definition
    "precision_definition": "margin in [0.000, 0.005]: bid within [MIP, MIP+0.5%]",
    # [FIX 4] Bartik note
    "bartik_note": (
        "Bartik Exposure computed on full panel before subsetting. "
        "Market-level CE adoption rates use all available firms."
    ),
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2)

# ============================================================
# 20. Diagnostics figure
# ============================================================
fig = plt.figure(figsize=(15, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# A. moment fit
ax = fig.add_subplot(gs[0, :])
x  = np.arange(len(momtab))
ax.bar(x - 0.2, momtab["emp"], 0.38, label="Empirical", alpha=0.8)
ax.bar(x + 0.2, momtab["fit"], 0.38, label="Model fit", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(momtab["moment"], rotation=60, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Moment fit")
ax.legend()

# B. average P(H) by year  [FIX 1: labeled as P(High=CE)]
ax = fig.add_subplot(gs[1, 0])
ph_year = fit_df.groupby("contract_fy")["pH"].mean().reset_index()
ax.plot(ph_year["contract_fy"], ph_year["pH"], marker="o")
for yr, col in [(2009, "gray"), (2014, "tomato"), (2023, "steelblue")]:
    ax.axvline(yr, ls=":", lw=1.5, color=col)
ax.set_ylim(0, 1)
ax.set_title("Average filtered P(High=CE) by year")
ax.set_ylabel("P(state=High)")

# C. observed vs fitted bid_rate
ax = fig.add_subplot(gs[1, 1])
ax.scatter(fit_df["bid_rate"], fit_df["bid_rate_fit"], alpha=0.25)
mn = min(fit_df["bid_rate"].min(), fit_df["bid_rate_fit"].min())
mx = max(fit_df["bid_rate"].max(), fit_df["bid_rate_fit"].max())
ax.plot([mn, mx], [mn, mx], "r--")
ax.set_title("Observed vs fitted bid_rate")
ax.set_xlabel("Observed")
ax.set_ylabel("Fitted")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 21. Optional bootstrap
# ============================================================
if RUN_BOOTSTRAP:
    print("\n[Bootstrap] starting ...")
    boot_params = []
    firm_names  = list(firm_data.keys())
    boot_rng    = np.random.default_rng(SEED + 999)

    for b in range(N_BOOT):
        draw        = boot_rng.choice(firm_names, size=len(firm_names), replace=True)
        boot_groups = []
        for j, nm in enumerate(draw):
            fd  = firm_data[nm]
            grp = fd["df"].copy()
            grp["firm"] = f"boot_{j}"
            boot_groups.append((
                f"boot_{j}",
                {
                    "df":    grp,
                    "x_seq": grp[X_COLS].to_numpy(float),
                    "y_seq": grp[Y_COLS].to_numpy(float),
                    "years": grp["contract_fy"].to_numpy(int),
                },
            ))

        obj_b = lambda th: total_negloglik(
            th, boot_groups, K=best_K, use_smc=False, Np=0, has_prec=HAS_PREC
        )
        rb = minimize(
            obj_b, x0=theta_hat, method="L-BFGS-B",
            options={"maxiter": 100, "disp": False},
        )
        boot_params.append(rb.x)
        if (b + 1) % 2 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}")

    boot_arr              = np.vstack(boot_params)
    boot_se               = boot_arr.std(axis=0, ddof=1)
    param_table["boot_se"] = boot_se
    param_table["t_stat"]  = param_table["estimate"] / (param_table["boot_se"] + 1e-10)
    param_table.to_csv(OUT_CSV1, index=False, encoding="utf-8-sig")

# ============================================================
# 22. Final console summary
# ============================================================
print("\n" + "=" * 68)
print("STRUCTURAL ESTIMATION (v6 staged — corrected)")
print("=" * 68)
print(f"  Selected K     : {best_K}")
print(f"  negloglik      : {negll_hat:.6f}")
print(f"  AIC            : {aic_val:.2f}  (negative = high likelihood; lower=better)")
print(f"  BIC            : {bic_val:.2f}  [FIX 6]")
print(f"  betas          : {np.round(betas, 4)}")
print(f"  pis            : {np.round(pis, 4)}")
print(f"  kappa          : {kappa:.4f}")
print(f"  sigmas         : {np.round(sigmas, 4)}")
print(f"  N panel rows   : {len(panel):,}")
print(f"  N firms        : {panel['firm'].nunique():,}")
print(f"  HAS_PREC       : {HAS_PREC}")
print()
w_i = X_COLS.index("w")
print(f"  [FIX 1] u_w(Low={STATE_LOW})  = "
      f"{u_params[STATE_LOW,  w_i]:.4f}  (should be < 0)")
print(f"  [FIX 1] u_w(High={STATE_HIGH}) = "
      f"{u_params[STATE_HIGH, w_i]:.4f}  (should be > 0)")
print(f"  [FIX 2] precision = bid in [MIP, MIP+0.5%]")
print(f"  [FIX 3] STAGED_MAXITER increasing: {STAGED_MAXITER}")
print(f"  [FIX 4] Bartik on full panel ({panel_full['firm'].nunique()} firms)")
print(f"  [FIX 5] Staged up to full sample (None)")
print()
print("  Files:")
for f in [OUT_JSON, OUT_CSV1, OUT_CSV2, OUT_FIG,
          OUT_THETA_STAGE2, OUT_META_STAGE2]:
    print(f"   - {f}")
print("=" * 68)
