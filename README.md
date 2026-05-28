# Churn Risk Prediction Platform

Time-aware churn prediction system built on AWS using Athena, SageMaker Processing, XGBoost, and SageMaker Unified Studio workflows.

The system trains and deploys an account-level churn model using temporal feature snapshots derived from operational support data. Training and inference are orchestrated through scheduled DAGs that launch containerized SageMaker Processing jobs for feature preparation, model training, calibration, and batch scoring.

The implementation emphasizes:

- temporal correctness
- reproducible batch inference
- strict training/inference feature consistency
- leakage-aware validation
- scheduled retraining workflows


---

# Architecture

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
     │     Workflow     │   │     Workflow     │
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

---

# Stack

| Component | Technology |
|---|---|
| Query Engine | Amazon Athena |
| Storage | Amazon S3 |
| ML Runtime | SageMaker Processing |
| Model | XGBoost |
| Workflow Orchestration | SageMaker Unified Studio |
| Workflow Runtime | MWAA / Airflow |
| Container Registry | Amazon ECR |
| Reporting | Amazon QuickSight |
| Infrastructure | AWS CDK / CloudFormation |

---

# Repository Structure

```text
churn-risk-platform/
│
├── workflows/
│   ├── training/
│   │   ├── training_processing_job.yaml
│   │   └── training_dag.png
│   │
│   └── inference/
│       ├── inference_processing_job.yaml
│       └── inference_dag.png
│
├── processing/
│   ├── train_xgb_processing.py
│   └── churn_xgb_inference.py
│
├── sql/
│   ├── training.sql
│   └── inference.sql
│
├── docker/
│   └── processing/
│
├── examples/
│   ├── metrics_example.json
│   └── predictions_schema.md
│
├── docs/
│   ├── architecture.md
│   ├── data_model.md
│   └── ml_notes.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Training Workflow

The training workflow executes monthly through a SageMaker Unified Studio DAG.

Workflow stages:

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

The workflow:

- exports training snapshots from Athena to S3
- launches a containerized SageMaker Processing job
- applies temporal validation logic
- trains and calibrates the model
- packages model artifacts
- registers the resulting model bundle


---

# Inference Workflow

The inference workflow executes weekly.

Workflow stages:

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

- retrieves the latest inference snapshot
- loads the current model artifacts
- reconstructs the training feature space
- performs batch scoring
- writes ranked predictions back to S3


---

# Training Pipeline

Training snapshots are mounted into the processing container at:

```text
/opt/ml/processing/input/raw
```

Primary fields:

| Column | Description |
|---|---|
| account_id | Account identifier |
| as_of_date | Snapshot timestamp |
| label_churn_365d | 365-day churn target |

The training pipeline uses strictly chronological validation logic.

Validation characteristics:

- adaptive last-N-month holdout selection
- chronological calibration/test separation
- right-censor filtering
- duplicate snapshot handling


Random train/test splitting is intentionally avoided.

---

# Right-Censor Filtering

Recent negative observations are excluded when the full churn horizon has not elapsed.

Filtering logic:

```text
Drop:
    label == 0
    AND
    as_of_date > (today - horizon_days)
```

This prevents unresolved future churn outcomes from being treated as observed negatives.

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
| annual_saas_revenue | SaaS revenue |
| total_enrollment | Enrollment-based size proxy |

---

# Model Training

The current implementation uses an XGBoost binary classifier.

Training characteristics:

- class imbalance weighting via `scale_pos_weight`
- early stopping
- histogram tree method
- Platt probability calibration
- persisted feature medians for inference parity


Training artifacts:

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

Inference runs through a containerized SageMaker Processing job.

Mounted inputs:

```text
/opt/ml/processing/input/raw
/opt/ml/processing/code
/opt/ml/processing/model
```

The inference pipeline:

- loads persisted feature metadata
- reconstructs feature ordering
- imputes missing values using persisted medians
- computes raw XGBoost margins
- applies calibration when available
- ranks accounts by predicted churn probability


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

# Example Workflow Configuration

```yaml
ProcessingJobName: churn-risk-xgb-inference-{{ ds_nodash }}

AppSpecification:
  ImageUri: <ecr-image-uri>

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

The platform aggregates operational and account-level data from:

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

# Notes

This repository contains generalized implementation patterns only.

Internal datasets, credentials, proprietary business logic, and sensitive operational details are excluded.
