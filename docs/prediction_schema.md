# Prediction Output Schema

The inference workflow writes scored account predictions as parquet output from the SageMaker Processing job.

## Output Location

```text
/opt/ml/processing/output/predictions.parquet
```

In production, outputs are written to a date-partitioned S3 prefix similar to:

```text
s3://<bucket>/inference/output/snapshot_run_date=YYYYMMDD/
```

---

# Dataset Semantics

The output dataset contains one row per scored account per the most recent inference run on the initial churn risk report table.

A second table, churn_risk_historic is maintained that cumulatively stores past runs to track risk profile over time.

Predictions are generated from the latest available feature snapshot for each account.

Rows are ordered by descending churn probability.

---
## Downstream Tables

The inference output is materialized into two downstream reporting tables.

| Table | Grain | Purpose |
|---|---|---|
| `churn_risk_report` | One row per account from the latest inference run | Current-state reporting and dashboard consumption |
| `churn_risk_historic` | One row per account per inference run | Historical tracking of account risk over time |

`churn_risk_report` is the latest snapshot table. It is overwritten or refreshed each inference cycle and is intended for operational reporting.

`churn_risk_historic` is append-oriented. Each inference run is retained so account-level risk movement can be analyzed over time.

This split supports both current prioritization and longitudinal monitoring without forcing downstream dashboards to filter historical partitions manually.

# Columns

| Column | Type | Description |
|---|---|---|
| `account_id` | string | Unique account identifier used for inference and downstream joins. |
| `data_snapshot_date` | date | Snapshot date of the feature frame used for scoring. |
| `inference_run_date` | date | Date the inference workflow executed. |
| `churn_score_raw` | double | Raw XGBoost margin output before probability calibration. |
| `churn_proba` | double | Final churn probability after Platt scaling calibration when available. Falls back to native XGBoost probability if no calibration artifact exists. |
| `prob_rank` | bigint | Descending risk rank derived from `churn_proba`. Rank `1` represents the highest-risk account. |

---

# Example Output

| account_id | data_snapshot_date | inference_run_date | churn_score_raw | churn_proba | prob_rank |
|---|---|---|---:|---:|---:|
| `acct_1001` | `2026-05-25` | `2026-05-28` | `2.8137` | `0.8214` | `1` |
| `acct_1044` | `2026-05-25` | `2026-05-28` | `1.4471` | `0.6118` | `2` |
| `acct_2218` | `2026-05-25` | `2026-05-28` | `-0.7288` | `0.1836` | `3` |

---

# Inference Requirements

The inference feature frame must contain:

| Required Column | Description |
|---|---|
| `account_id` | Account identifier used for downstream joins and ranking. |
| `data_snapshot_date` | Snapshot date associated with the feature row. |

All model feature columns are dynamically reconstructed from the saved model artifact bundle.

---

# Model Artifact Dependencies

The inference workflow expects the following artifacts:

| Artifact | Purpose |
|---|---|
| `xgb_model.json` | Serialized XGBoost booster model. |
| `feature_names.txt` | Ordered feature list used to reconstruct the inference matrix. |
| `feature_medians.json` | Default imputation values for missing features. |
| `platt_scaler.pkl` | Optional probability calibration model. |

---

# Notes


- Missing feature columns are automatically reconstructed using saved feature medians.
- Numeric coercion is applied defensively during inference.
- Duplicate accounts are resolved by keeping the most recent snapshot row.
- Probabilities are clipped to `[0, 1]`.
- The model is optimized primarily for ranking and prioritization workflows rather than static binary classification thresholds.
- Output datasets are intended for downstream Athena, BI, or operational risk-review workflows.
