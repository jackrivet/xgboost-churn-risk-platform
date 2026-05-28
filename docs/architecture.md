## Overview

Batch churn prediction platform built on:


- Amazon Athena for feature generation
- SageMaker Processing for training and inference
- S3 for dataset and artifact storage
- SageMaker Unified Studio DAGs for orchestration
- QuickSight for downstream reporting

---

## Training Workflow

Schedule: Monthly

Flow:

1. Athena generates training snapshot
2. Snapshot exported to S3
3. SageMaker Processing job trains model
4. Model artifacts written to S3
5. Metrics and calibration outputs persisted


Artifacts:

- xgb_model.json
- feature_names.txt
- feature_medians.json
- platt_scaler.pkl
- metrics.json

---

## Inference Workflow

Schedule: Weekly

Flow:

1. Athena generates latest account feature snapshot
2. Snapshot exported to S3
3. SageMaker Processing batch scores accounts
4. Predictions written to S3 as parquet


Outputs feed:

- operational reporting
- prioritization dashboards
- historical risk tracking

---

## Reporting Tables

### churn_risk_report

Latest prediction per account.

Used for operational consumption.

### churn_risk_historic

Append-only prediction history.

Used for:

- risk trajectory analysis
- longitudinal monitoring
- retrospective validation


---

## Storage Layout

```text
training/raw/
training/output/

inference/current/raw/
inference/current/output/

models/
metrics/
```
