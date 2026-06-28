# Project AegisSilicon — Hardware Telemetry EDA & Predictive Modeling

Exploratory data analysis and machine learning on simulated silicon/hardware node telemetry, aimed at detecting and predicting node degradation before failure.

## Overview

AegisSilicon nodes continuously run compute jobs and report back an `expected_result` vs an `actual_result`. Drift between the two (`deviation`, `z_score`) is the core signal used to flag a node as `HEALTHY` or `DEGRADED`, alongside hardware telemetry like temperature, power draw, memory/compute utilization, and error counters.

This repo contains:
- A raw telemetry dataset (`aegis_silicon_telemetry_200k.csv`)
- A full EDA + modeling notebook (`AegisSilicon_EDA_Rewritten__1_.ipynb`) covering data quality checks, statistical analysis, regression, classification, anomaly detection, and cross-validation

## Dataset

**`aegis_silicon_telemetry_200k.csv`** — 200,000 rows × 21 columns, second-by-second telemetry across **20 nodes** (`Node_01`–`Node_20`), spanning **2026-06-01 to 2026-06-03**.

| Column | Description |
|---|---|
| `timestamp` | Reading timestamp (1-second resolution) |
| `node_id` | Node identifier |
| `batch_id` | Compute batch/job index |
| `expected_result` | Expected output value for the batch |
| `actual_result` | Observed output value |
| `deviation` | `actual_result − expected_result` |
| `pct_deviation` | Deviation as a percentage |
| `z_score` | Standard deviations from expected value |
| `psi_value` | Population Stability Index (post-hoc, leakage column) |
| `anomaly_score` | Model-derived anomaly score (post-hoc, leakage column) |
| `is_corrupted` | Corruption flag (post-hoc, leakage column) |
| `temperature_celsius` | Node temperature |
| `power_draw_watts` | Power draw |
| `memory_util_pct` | Memory utilization % |
| `compute_util_pct` | Compute utilization % |
| `ops_per_sec` | Operations per second |
| `tensor_core_errors` | Tensor core error count |
| `mem_ecc_errors` | Memory ECC error count |
| `network_latency_ms` | Network latency |
| `mtbf_hours` | Mean time between failures (hours) |
| `node_status` | Target label — `HEALTHY` or `DEGRADED` |

**Class balance:** 194,017 `HEALTHY` rows vs 5,983 `DEGRADED` rows (~3% positive class — significant imbalance).

> ⚠️ **Leakage warning:** `anomaly_score`, `is_corrupted`, and `psi_value` are computed *after* the fact and are dropped before any modeling step in the notebook — they should not be used as model inputs.

## Notebook Walkthrough

`AegisSilicon_EDA_Rewritten__1_.ipynb` is organized into two parts:

### Part 1 — EDA
1. **Load Dataset & Drop Leakage Columns**
2. **Data Quality Check** — nulls, duplicates, date range
3. **Class Balance** — HEALTHY vs DEGRADED counts
4. **Descriptive Statistics** — summary stats, skewness, kurtosis
5. **Z-Score Distribution** — HEALTHY vs DEGRADED comparison
6. **Deviation Analysis** — KDE plots and per-node average deviation
7. **Feature Boxplots** — HEALTHY vs DEGRADED separation per feature
8. **Correlation Heatmap** — pairwise feature relationships
9. **Z-Score Over Time** — 10-minute resampled trends for sample nodes

### Part 2 — Model Training
- **Feature Preparation** — scaling for distance-based models
- **Part A — Regression** (target: `deviation`)
  - OLS regression with statistical significance
  - Model comparison (R², RMSE) across linear and tree-based models
  - Actual vs predicted, residual diagnostics (Residuals vs Fitted, Q-Q plot, Scale-Location)
- **Part B — Classification** (target: `node_status`)
  - Model comparison via ROC-AUC, F1, Accuracy
  - ROC curves
  - Confusion matrix and classification report on the best model
  - Random Forest feature importance
- **Part C — Anomaly Detection** (unsupervised, no labels)
  - Isolation Forest deep dive: per-node flag rates and anomaly score distribution
- **Part D — Cross-Validation & Final Leaderboard**
  - 5-fold cross-validation for model stability comparison

## Getting Started

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
jupyter notebook AegisSilicon_EDA_Rewritten__1_.ipynb
```

## Repo Structure

```
.
├── AegisSilicon_EDA_Rewritten__1_.ipynb   # EDA + modeling notebook
├── aegis_silicon_telemetry_200k.csv       # Raw telemetry dataset (200k rows)
└── README.md
```

## Notes

- Because `DEGRADED` is a minority class (~3%), classification results are evaluated primarily on **F1** and **ROC-AUC** rather than raw accuracy.
- In a hardware-monitoring context, **false negatives** (missed DEGRADED nodes) are treated as the most costly error type.
