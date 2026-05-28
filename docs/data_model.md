## Training Grain

One row per:

```text
(account_id, as_of_date)
```

Each row represents the state of an account at a historical snapshot date.

---

## Labeling Strategy

Targets are generated using forward-looking churn windows.

Example:

```text
label_churn_365d = 1
```

indicates the account churned within 365 days after the snapshot date.

---

## Temporal Feature Windows

Features are generated using rolling lookback windows.

Examples:

- 30d support activity
- 90d QA metrics
- 180d survey history

Only data available prior to the snapshot date is included.

---

## Feature Categories

### Support Features

Examples:

- calls_30d
- avg_handle_time_90d
- low_eval_count_90d


### Survey Features

Examples:

- surveys_30d
- avg_quality_score_90d
- fcr_rate_30d


### Case Features

Examples:

- cases_30d
- days_since_last_case


### Static Account Features

Examples:

- enrollment
- site count
- district complexity


---

## Validation Strategy

Training and evaluation are separated chronologically.

Example:

```text
Train: 2024-01 → 2025-03
Test:  2025-04 → 2025-12
```

This prevents temporal leakage.
