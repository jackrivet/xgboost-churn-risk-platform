## Model Selection

XGBoost was selected because the problem is:

- tabular
- sparse
- heavily imbalanced
- feature-engineered

The model produced materially stronger ranking performance than simpler linear baselines.

---

## Optimization Objective

The system is optimized for ranking quality rather than binary classification accuracy.

Primary operational goal:

- identify the highest-risk accounts
- maximize lift within constrained intervention capacity

Most important metrics:

- PR-AUC
- precision@k
- lift@k

---

## Calibration

Probabilities are calibrated using Platt scaling on a temporally separated calibration split.

This improves probability stability for downstream reporting and prioritization.

---

## Leakage Controls

The pipeline includes:

- chronological train/test separation
- right-censor filtering
- forward-looking label generation
- pre-snapshot feature enforcement

---

## Inference Design

Inference is batch-oriented rather than real-time.

Weekly scoring was selected because:

- account risk evolves slowly
- features are aggregation-heavy
- predictions support operational review workflows
