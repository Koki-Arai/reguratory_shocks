# Data

## Files required

| File | Rows | Description |
|------|------|-------------|
| `all_bids_dataset.csv` | 81,934 | All bid-level observations, Hokuriku Regional Development Bureau, FY2006–2024. One row per bidder per lot. |
| `analysis_dataset.csv` | 15,627 | Winning-bid panel for reduced-form estimation (Pillars I–IV). One row per awarded contract. |

## Key variables — `all_bids_dataset.csv`

| Variable | Type | Description |
|----------|------|-------------|
| `lot_id` | int | Unique auction identifier |
| `firm` | str | Bidder name (normalized) |
| `contract_fy` | int | Fiscal year of contract award |
| `scoring_flag` | str | "有" = comprehensive evaluation (CE) format |
| `bid_rate` | float | Bid / engineer's estimate |
| `final_bid` | int | Final bid amount (JPY) |
| `estimate_price` | int | Engineer's estimate (JPY) |
| `investigation_price` | int | Minimum investigation price (MIP, JPY) |
| `precision_gap` | float | (bid − MIP) / estimate |
| `w_it` | float | Log backlog proxy (firm-level capacity measure) |
| `perf_y1` | float | CPR score, year t−1 (0–100) |
| `perf_y2` | float | CPR score, year t−2 |
| `n_bidders` | int | Number of bidders |
| `work_type` | str | Construction category |
| `office` | str | Procuring office |

## Key variables — `analysis_dataset.csv`

| Variable | Type | Description |
|----------|------|-------------|
| `h_it` | float | Two-year average CPR score = (perf_y1 + perf_y2) / 2 |
| `perf_avg` | float | Alias for h_it |
| `tau_it` | float | Backlog index |
| `scoring` | int | 1 = CE-format auction |
| `precision` | int | 1 = bid ∈ [MIP, MIP + 0.5%] |

## Notes

- `precision` is defined as `precision_gap ∈ [0, 0.005]`, consistent with the structural model.
- CPR threshold for CE eligibility: **h ≥ 80** (MLIT 2009 reform).
- Coverage: Hokuriku Regional Development Bureau, FY2006–2024.
