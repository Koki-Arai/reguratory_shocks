# Replication Code: "Regulatory Shocks and Market Equilibria"

Replication materials for:

> Arai, [co-authors]. "Regulatory Shocks and Market Equilibria: Evidence from Japanese Public Procurement." *Journal of Law and Economics* (submitted).

All scripts are designed to run on **Google Colab** with Google Drive mounted.  
Data files (`all_bids_dataset.csv`, `analysis_dataset.csv`) must be placed in the `SAVE_DIR` folder on Google Drive before running.

---

## Repository structure

```
.
├── 01_structural_estimation.py   # Dynamic structural model (HMM, VFI, MLE)
├── 02_bootstrap.py               # Bootstrap standard errors for structural estimates
├── 03_reduced_form.py            # Reduced-form estimation: Pillars I–IV
├── 04_robustness.py              # Robustness checks (Sections 5–6)
├── data/
│   ├── README.md                 # Variable descriptions
│   ├── all_bids_dataset.csv      # Bid-level panel (81,934 obs)
│   └── analysis_dataset.csv      # Winning-bid panel (15,627 obs)
├── requirements.txt
└── README.md
```

---

## Data

See [`data/README.md`](data/README.md) for full variable descriptions.

| File | Obs | Unit | Coverage |
|------|-----|------|----------|
| `all_bids_dataset.csv` | 81,934 | Bidder × lot | Hokuriku RDB, FY2006–2024 |
| `analysis_dataset.csv` | 15,627 | Awarded contract | Hokuriku RDB, FY2006–2024 |

---

## Scripts

### `01_structural_estimation.py` — Dynamic structural model

Estimates a two-state Hidden Markov Model (HMM) with value function iteration (VFI).

**Model:**
- States: Low/PO (price-only auctions) and High/CE (comprehensive evaluation auctions)
- State variable: h_it = two-year average CPR score; w_it = log backlog (capacity proxy)
- Discount factor β identified from forward-looking switching dynamics
- Switching cost κ estimated jointly

**Algorithm:**
- Staged L-BFGS-B on subsamples of increasing size: top-200 → top-500 → top-1000 → all firms
- Forward filter (exact, K=1) for likelihood evaluation
- Checkpoint restart: saves `vf_v6_theta_stage2.npy` every 30 minutes

**Key outputs** (saved to `SAVE_DIR`):

| File | Contents |
|------|----------|
| `vf_v6_theta_stage2.npy` | Estimated parameter vector θ (dim=53) |
| `vf_v6_main_results.json` | Point estimates, AIC/BIC, model metadata |
| `vf_v6_param_table.csv` | Parameter table (raw and transformed) |
| `vf_v6_moment_fit.csv` | Empirical vs. model-fitted moments |
| `vf_v6_diagnostics.png` | Moment fit + filtered P(High=CE) by year |

**Runtime:** approximately 20–30 hours on Google Colab (A100 GPU not used; CPU only).

**Settings to change before running:**
```python
SAVE_DIR      = "/content/drive/MyDrive/structural_estimation_v6/"  # your Drive path
STAGED_FIRMS  = [200, 500, 1000, None]   # None = all 2,580 firms
N_BOOT        = 0    # set to 0 here; use 02_bootstrap.py instead
```

---

### `02_bootstrap.py` — Bootstrap standard errors

Firm-level nonparametric bootstrap for the structural model (N = 200 replications recommended).

**Design:**
- Resamples firms with replacement (cluster bootstrap at the firm level)
- Restarts from the point estimate `vf_v6_theta_stage2.npy` for each draw
- CI constructed from percentile of the **transformed** parameter distribution
  (ensures CI ∋ point estimate by construction; see note below)
- Staged subsampling `[200, 500]` per draw for speed

**Note on β CI:** The raw optimizer works in the unconstrained space  
`beta_raw ∈ ℝ`, with β = 0.995·σ(beta_raw). CIs are computed on the  
transformed scale to guarantee `CI_lower ≤ β̂ ≤ CI_upper`.

**Key outputs:**

| File | Contents |
|------|----------|
| `bootstrap_raw.csv` | All bootstrap draws (transformed parameters) |
| `bootstrap_summary.csv` | Point estimates, SEs, 95% CIs (Table 7) |
| `bootstrap_trace.png` | β trace plot + cumulative mean (convergence check) |
| `bootstrap_dists.png` | Distribution plots for all key parameters |

**Runtime:** approximately 12–18 hours for N = 200 on Google Colab.

**Settings:**
```python
SAVE_DIR       = "/content/drive/MyDrive/structural_estimation_v6/"
THETA_HAT_PATH = os.path.join(SAVE_DIR, "vf_v6_theta_stage2.npy")
N_BOOT         = 200
STAGED_FIRMS   = [200, 500]
```

---

### `03_reduced_form.py` — Reduced-form estimation (Pillars I–IV)

Replicates the four-pillar reduced-form evidence in Section 4.

| Pillar | Method | Outcome |
|--------|--------|---------|
| I | TWFE panel OLS (15,627 obs) | Bid rate, scoring rate, precision rate |
| II | RDD at h = 80 (CPR threshold) | CE participation rate (next year) |
| III | Bartik IV (work-type exposure) | Scoring rate, bid rate |
| IV | DDD (Noto Peninsula earthquake 2024) | Scoring rate, precision rate |

**Key outputs:** `fig_pillar1_main.png`, `fig_pillar2_rdd.png`,  
`fig_pillar3_eventstudy.png`, `fig_timeseries.png`

---

### `04_robustness.py` — Robustness checks

Replicates robustness tables and figures in Sections 5–6.

- Alternative bandwidth / kernel for RDD (Pillar II)
- Alternative backlog measures (Pillar I)
- Work-type heterogeneity
- Placebo tests

**Key output:** `fig_section6_robustness_v2.png`

---

## Execution order

```
1. 01_structural_estimation.py   (~20–30 hrs; saves checkpoint every 30 min)
2. 02_bootstrap.py               (~12–18 hrs; requires checkpoint from step 1)
3. 03_reduced_form.py            (~10–20 min)
4. 04_robustness.py              (~10–20 min)
```

Steps 3 and 4 are independent of steps 1–2 and can be run at any time.

---

## Google Colab setup

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install additional packages
!pip install linearmodels tqdm statsmodels

# 3. Upload data files to SAVE_DIR on Drive, then run scripts
```

---

## Parameter table (Table 5 in paper)

Estimated from the full panel (2,580 firms, 11,963 firm-years), final checkpoint at top-500 firms:

| Parameter | Estimate | Description |
|-----------|----------|-------------|
| β | 0.6057 | Discount factor |
| κ | 1.1666 | Switching cost |
| u_w(L) | −0.2824 | Backlog coeff, Low/PO state |
| u_w(H) | +0.2824 | Backlog coeff, High/CE state |
| u_h(L) | +0.0902 | CPR coeff, Low/PO state |
| u_h(H) | +0.3921 | CPR coeff, High/CE state |
| σ_sc | 0.1216 | Noise, scoring rate |
| σ_pr | 0.1011 | Noise, precision rate |
| σ_bd | 0.0547 | Noise, bid rate |

*Note: full-sample estimates (STAGED_FIRMS including None) will update these values.*

---

## Software

- Python 3.10+
- See `requirements.txt` for package versions
- All scripts run on Google Colab (CPU runtime); no GPU required

---

## Funding

This research is supported by JSPS KAKENHI Grant Number 23K01404.

---

## License

Code: MIT License  
Data: for academic replication purposes only. Please contact the authors before redistribution.
