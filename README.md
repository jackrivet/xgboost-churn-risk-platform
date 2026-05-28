# Churn Risk Prediction Platform

Production churn prediction system built on AWS using Athena, SageMaker Processing, XGBoost, and SageMaker Unified Studio workflows.

The platform generates time-indexed account-level feature snapshots from operational datasets, trains a leakage-aware XGBoost classifier, and executes scheduled batch inference through Airflow-style DAG orchestration.

The system is designed around:

- time-aware model validation
- reproducible batch inference
- feature parity between training and inference
- containerized execution
- scheduled retraining and scoring workflows


---

# System Architecture

## High-Level Flow

```text
                   ┌──────────────────┐
                   │   Athena Views   │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │  S3 Batch Export │
                   └────────┬─────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
     ┌──────────────────┐   ┌──────────────────┐
     │ Monthly Training │   │ Weekly Inference │
     │       DAG        │   │       DAG        │
     └────────┬─────────┘   └────────┬─────────┘
              │                      │
              ▼                      ▼
     ┌──────────────────┐   ┌──────────────────┐
     │ SageMaker Proc.  │   │ SageMaker Proc.  │
     │ XGBoost Training │   │ Batch Scoring    │
     └────────┬─────────┘   └────────┬─────────┘
              │                      │
              ▼                      ▼
     ┌──────────────────┐   ┌──────────────────┐
     │  Model Artifacts │   │ Prediction Output│
     │       in S3      │   │    Parquet/S3    │
     └──────────────────┘   └──────────────────┘
```
## Training and Inference DAGs
<img width="407" height="660" alt="Screenshot 2026-05-28 at 9 14 21 AM" src="https://github.com/user-attachments/assets/ddd43832-c7d5-4aaa-8524-694d1edc2418" />
<img width="361" height="456" alt="Screenshot 2026-05-28 at 9 14 48 AM" src="https://github.com/user-attachments/assets/746ff180-d9b0-48a7-9606-5f8e8ddb3706" />

---

# Core Stack

| Component | Technology |
|---|---|
| Query Layer | Amazon Athena |
| Object Storage | Amazon S3 |
| ML Execution | SageMaker Processing |
| Model | XGBoost |
| Workflow Orchestration | SageMaker Unified Studio |
| Workflow Runtime | MWAA / Airflow-style DAGs |
| Container Runtime | Docker + Amazon ECR |
| Reporting Layer | Amazon QuickSight |
| Infrastructure | AWS CDK / CloudFormation |

---

# Repository Structure

```text
churn-risk-platform/
│
├── dags/
│   ├── training/
│   └── inference/
│
├── processing/
│   ├── train_xgb_processing.py
│   ├── churn_xgb_inference.py
│   ├── preprocessing/
│   └── utils/
│
├── sql/
│   ├── training/
│   ├── inference/
│   ├── features/
│   └── labels/
│
├── infrastructure/
│   ├── cdk/
│   ├── iam/
│   └── workflow_configs/
│
├── docker/
│   └── processing/
│
├── docs/
│   ├── architecture.md
│   ├── data_model.md
│   └── ml_notes.md
│
└── README.md
```

---

# Workflow Orchestration

The platform is orchestrated through two independent SageMaker Unified Studio workflows.

---

# Monthly Training Workflow

Schedule:

- first day of each month


Workflow structure:

```text
Purge-Old-Runs
        ↓
Retrieve-Training-Data
        ↓
Clear-Artifacts-Folder
        ↓
XGBoost-Preprocessing
        ↓
Register-Model
```

The workflow launches a containerized SageMaker Processing job that:

- loads temporal training snapshots from S3
- validates snapshot integrity
- applies censor-aware filtering
- constructs time-aware train/holdout splits
- trains and calibrates an XGBoost classifier
- packages model artifacts
- registers the resulting model bundle


---

# Weekly Inference Workflow

Schedule:

- weekly


Workflow structure:

```text
Unload-Previous-Run
        ↓
Purge-Old-Run
        ↓
Fetch-New-Batch
        ↓
Execute-Inference
```

The workflow:

- retrieves the latest inference snapshot from Athena
- loads the most recent model bundle
- reconstructs the feature matrix
- generates calibrated churn probabilities
- writes ranked predictions to S3


---

# Training Pipeline

Training data is exported from Athena as Parquet snapshots and mounted into the processing container at:

```text
/opt/ml/processing/input/raw
```

Primary fields:

| Column | Description |
|---|---|
| account_id | Account identifier |
| as_of_date | Snapshot timestamp |
| label_churn_365d | 365-day churn target |

---

# Time-Aware Validation

The training workflow uses strictly chronological validation logic.

Validation characteristics:

- adaptive last-N-month holdout selection
- chronological calibration/test separation
- duplicate snapshot protection
- leakage-aware splitting
- right-censor filtering


Random train/test splits are intentionally avoided.

---

# Right-Censor Handling

Recent negative examples are excluded when the full churn observation horizon has not elapsed.

Filtering logic:

```text
Drop:
    label == 0
    AND
    as_of_date > (today - horizon_days)
```

This prevents incorrectly labeling unresolved future churn outcomes as negatives.

---

# Feature Engineering

Features are generated as rolling temporal aggregates at the account level.

Examples:

| Feature | Description |
|---|---|
| calls_30d | Support call volume over 30 days |
| calls_90d | Support call volume over 90 days |
| avg_handle_time_90d | Average support handle time |
| avg_eval_score_90d | QA evaluation average |
| fcr_rate_90d | First-contact resolution rate |
| survey_score_90d | Survey average |
| annual_saas_revenue | Annual SaaS revenue |
| total_enrollment | Enrollment-based size proxy |

---

# Model Training

Model:

- XGBoost binary classifier


Training characteristics:

- class imbalance weighting via `scale_pos_weight`
- early stopping
- histogram tree method
- probability calibration using Platt scaling
- persisted feature medians for inference parity


The processing job outputs:

```text
xgb_model.json
platt_scaler.pkl
feature_names.txt
feature_medians.json
metrics.json
model.tar.gz
```

---

# Inference Pipeline

Inference executes through a containerized SageMaker Processing job.

Mounted inputs:

```text
/opt/ml/processing/input/raw
/opt/ml/processing/code
/opt/ml/processing/model
```

The inference pipeline:

- loads persisted feature metadata
- reconstructs training feature ordering
- imputes missing values using persisted medians
- computes raw XGBoost margins
- applies Platt calibration when available
- ranks accounts by churn probability


---

# Inference Output

Predictions are written to:

```text
/opt/ml/processing/output/predictions.parquet
```

Output schema:

| Column | Description |
|---|---|
| account_id | Account identifier |
| data_snapshot_date | Snapshot timestamp |
| inference_run_date | Inference execution timestamp |
| churn_score_raw | Raw XGBoost margin |
| churn_proba | Calibrated churn probability |
| prob_rank | Probability rank |

---

# Processing Configuration

Inference jobs are launched dynamically through workflow templates.

Example configuration:

```yaml
ProcessingJobName: churn-risk-xgb-inference-{{ ds_nodash }}

AppSpecification:
  ImageUri: <xgboost-ecr-image>

  ContainerEntrypoint:

python
/opt/ml/processing/code/churn_xgb_inference.py


ProcessingResources:
  ClusterConfig:
    InstanceType: ml.m5.xlarge
    InstanceCount: 1
```

---

# Data Sources

The platform aggregates operational/account-level data from systems including:

- Amazon Connect
- Salesforce Service Cloud Voice
- QA evaluation datasets
- Survey / CSAT systems
- Revenue metadata
- Product lifecycle datasets


---

# Local Development

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

# Planned Enhancements

Potential future additions:

- drift monitoring
- SHAP-based explainability
- feature store integration
- automated hyperparameter tuning
- online inference endpoints
- multi-model evaluation workflows


---

# Notes

This repository contains generalized implementation patterns only.

Internal datasets, credentials, proprietary business logic, and sensitive operational details are excluded.
