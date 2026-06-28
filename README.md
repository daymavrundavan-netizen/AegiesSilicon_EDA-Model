# Project AegisSilicon

**AI-Powered Silicon Telemetry Intelligence Platform**  
Node Health Prediction · Deviation Forecasting · Unsupervised Anomaly Detection

---

## Overview

Project AegisSilicon is an end-to-end machine learning pipeline built for real-time health monitoring of silicon compute nodes. Using 200,000 telemetry records captured across 20 nodes over a 55-hour operational window, the project delivers three predictive capabilities:

- **Regression** — forecast the continuous deviation signal (actual minus expected compute output) to quantify node misbehaviour magnitude
- **Classification** — classify each telemetry reading as `HEALTHY` or `DEGRADED` using 14 sensor-derived features
- **Anomaly Detection** — surface structurally unusual node behaviour in fully unsupervised mode, without relying on labelled data

All three pipelines achieved near-perfect or perfect predictive performance on held-out test data.

---

## Results at a Glance

| Task | Champion Model | Key Metric |
|---|---|---|
| Regression | Linear Regression | R² = 1.0000 · RMSE = 0.000000 |
| Classification | Logistic Regression | ROC-AUC = 1.000 · F1 = 1.000 |
| Anomaly Detection | Isolation Forest | 10,000 flags (5%) · Top nodes: 91.6% flag rate |

---

## Dataset

| Attribute | Detail |
|---|---|
| File | `aegis_silicon_telemetry_200k_17C.csv` |
| Records | 200,000 |
| Features used in modelling | 14 numeric sensor signals |
| Regression target | `deviation` (continuous, float64) |
| Classification target | `node_status` (HEALTHY / DEGRADED) |
| Observation window | 2026-06-01 00:00 → 2026-06-03 07:33 |
| Nodes monitored | 20 |
| Class distribution | 194,017 HEALTHY (97%) · 5,983 DEGRADED (3%) |
| Missing values | None |

---

## Project Structure

```
aegissilicon/
├── AegisSilicon_EDA_Rewritten.ipynb   # Main notebook — EDA, modelling, evaluation
├── aegis_silicon_telemetry_200k_17C.csv  # Telemetry dataset (not tracked in git)
└── README.md
```

---

## Methodology

### Preprocessing
- StandardScaler applied to all 14 numeric features (zero mean, unit variance)
- `node_id` and `timestamp` excluded from the feature matrix
- 80/20 stratified train/test split — 160,000 training rows, 40,000 test rows

### Regression Models Evaluated

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **Linear Regression** ★ | 1.0000 | 0.000000 | 0.000000 |
| Gradient Boosting | 1.0000 | 0.000002 | 0.000068 |
| Decision Tree (depth 6) | 0.9999 | 0.000116 | 0.004794 |
| Random Forest | 0.9996 | 0.000160 | 0.011409 |
| Ridge (α=1.0) | 0.9993 | 0.000325 | 0.014755 |
| Lasso (α=0.001) | 0.9975 | 0.000274 | 0.028704 |
| LightGBM | 0.8037 | 0.005193 | 0.253750 |
| XGBoost | 0.6750 | 0.003204 | 0.326503 |

### Classification Models Evaluated

| Model | ROC-AUC | F1 Score | Accuracy |
|---|---|---|---|
| **Logistic Regression** ★ | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 0.9999 | 0.9999 |
| XGBoost | 0.9999 | 1.0000 | 1.0000 |
| Decision Tree (d=5) | 0.9996 | 1.0000 | 1.0000 |
| Gradient Boosting | 0.9996 | 1.0000 | 1.0000 |
| AdaBoost | 0.9996 | 1.0000 | 1.0000 |
| LightGBM | 0.9996 | 1.0000 | 1.0000 |

> **Class imbalance handling:** `class_weight='balanced'` was applied where supported. ROC-AUC and F1 Score were used as primary evaluation metrics given the 97:3 class ratio.

### Anomaly Detection — Isolation Forest

- **Configuration:** 200 estimators · contamination = 0.05 · no labels used
- **Flagged records:** 10,000 (5.00% of 200,000)
- All five highest-flagged nodes carried a `DEGRADED` ground-truth label, confirming strong alignment between the unsupervised signal and actual node health status

| Node | Status | Flags | Flag Rate |
|---|---|---|---|
| Node_20 | DEGRADED | 283 | 91.59% |
| Node_18 | DEGRADED | 275 | 86.21% |
| Node_02 | DEGRADED | 281 | 83.88% |
| Node_13 | DEGRADED | 246 | 80.13% |
| Node_07 | DEGRADED | 245 | 79.55% |

### Cross-Validation
5-fold cross-validation was applied to top regression and classification candidates. Linear Regression achieved Mean R² = 1.0000 (Std = 0.0000) across all folds, confirming zero variance and perfect generalisation.

---

## Key EDA Findings

1. **Severe class imbalance (97:3)** — necessitated ROC-AUC and F1 as primary classification metrics over raw accuracy
2. **Dominant predictor** — `pct_deviation` holds a Pearson correlation of +0.9989 with the regression target; all other features show |r| < 0.09
3. **Z-score separability** — clear distributional separation between HEALTHY and DEGRADED nodes at the 2.0 standard deviation threshold
4. **Extreme error skewness** — `tensor_core_errors` skewness = 84.6, kurtosis = 9,874.6; heavy-tail distribution indicating rare but extreme hardware error spikes
5. **Node-level deviation consistency** — DEGRADED nodes exhibit systematically larger absolute deviations across all sensor features

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Statistical analysis | `statsmodels`, `scipy` |
| Machine learning | `scikit-learn` |
| Gradient boosting | `xgboost`, `lightgbm` |
| Environment | Python 3 · Google Colab |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/aegissilicon.git
cd aegissilicon

# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels scipy scikit-learn xgboost lightgbm

# Launch the notebook
jupyter notebook AegisSilicon_EDA_Rewritten.ipynb
