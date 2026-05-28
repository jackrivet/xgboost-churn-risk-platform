"""
SageMaker Processing entrypoint for time-aware XGBoost churn model training.

This public version is sanitized:
- no internal table names
- no account IDs
- no production S3 paths
- no proprietary feature logic

Expected input:
    /opt/ml/processing/input/raw/*.parquet

Expected columns:
    account_id
    as_of_date
    label_churn_365d
    numeric feature columns

Outputs:
    /opt/ml/processing/output/model/xgb_model.json
    /opt/ml/processing/output/model/platt_scaler.pkl
    /opt/ml/processing/output/feature_names.txt
    /opt/ml/processing/output/feature_medians.json
    /opt/ml/processing/output/metrics.json
    /opt/ml/processing/output/model.tar.gz
"""

import json
import logging
import os
import pickle
import tarfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

RAW_DIR = "/opt/ml/processing/input/raw"
OUTPUT_DIR = "/opt/ml/processing/output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")

LABEL_COL = "label_churn_365d"
ACCOUNT_COL = "account_id"
ASOF_COL = "as_of_date"

HORIZON_DAYS = 365

INITIAL_HOLDOUT_MONTHS = 4
MAX_HOLDOUT_MONTHS = 18
MIN_HOLDOUT_POS = 10
MIN_HOLDOUT_NEG = 50

CAL_FRACTION = 0.5
RANK_FRACS = [0.005, 0.01, 0.02, 0.05]
RANDOM_STATE = 42

METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_names.txt")
MEDIANS_PATH = os.path.join(OUTPUT_DIR, "feature_medians.json")
MODEL_JSON_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
PLATT_SCALER_PATH = os.path.join(MODEL_DIR, "platt_scaler.pkl")
MODEL_TAR_PATH = os.path.join(OUTPUT_DIR, "model.tar.gz")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_STATE)


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    hold_df: pd.DataFrame
    cutoff_asof: pd.Timestamp
    train_months: List[str]
    hold_months: List[str]
    debug: Dict[str, Any]


def find_raw_files(raw_dir: str) -> List[str]:
    files: List[str] = []

    for root, _, filenames in os.walk(raw_dir):
        for name in filenames:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                files.append(path)

    logger.info("Found %d candidate files under %s", len(files), raw_dir)
    return files


def load_data() -> pd.DataFrame:
    files = find_raw_files(RAW_DIR)

    if not files:
        raise RuntimeError(f"No files found in {RAW_DIR}")

    dfs: List[pd.DataFrame] = []

    for path in files:
        try:
            part = pd.read_parquet(path)
            if len(part):
                dfs.append(part)
                logger.info("Loaded %d rows from %s", len(part), path)
        except Exception as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)

    if not dfs:
        raise RuntimeError(f"No readable parquet files found in {RAW_DIR}")

    df = pd.concat(dfs, ignore_index=True)

    required = [ACCOUNT_COL, ASOF_COL, LABEL_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df[ASOF_COL] = pd.to_datetime(df[ASOF_COL], errors="coerce")

    bad_dates = int(df[ASOF_COL].isna().sum())
    if bad_dates:
        raise RuntimeError(f"{ASOF_COL} contains {bad_dates} unparsable values")

    return df


def validate_asof_coverage(
    df: pd.DataFrame,
    min_unique_months: int = 8,
    max_span_days: int = 4000,
) -> Dict[str, Any]:
    asof = df[ASOF_COL].dropna()

    min_asof = asof.min()
    max_asof = asof.max()
    span_days = int((max_asof - min_asof).days)

    months = asof.dt.to_period("M")
    month_counts = months.value_counts().sort_index()

    info = {
        "min_asof": str(min_asof),
        "max_asof": str(max_asof),
        "span_days": span_days,
        "unique_days": int(asof.dt.normalize().nunique()),
        "unique_months": int(month_counts.shape[0]),
        "month_counts": {str(k): int(v) for k, v in month_counts.items()},
    }

    if span_days <= 0 or span_days > max_span_days:
        raise RuntimeError(f"Invalid as_of_date span: {info}")

    if info["unique_months"] < min_unique_months:
        raise RuntimeError(f"Insufficient temporal coverage: {info}")

    logger.info("as_of_date coverage:\n%s", json.dumps(info, indent=2))
    return info


def apply_right_censor_filter(
    df: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop recent negative labels when the full prediction horizon has not elapsed.

    Positive labels are retained because the event has already been observed.
    Recent negatives are censored because they may still churn within the horizon.
    """
    df = df.copy()

    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=horizon_days)

    drop_mask = (df[ASOF_COL] > cutoff) & (df[LABEL_COL] == 0)

    debug = {
        "today": str(today),
        "censor_cutoff_asof": str(cutoff),
        "rows_before": int(len(df)),
        "dropped_right_censored_negatives": int(drop_mask.sum()),
        "kept_post_cutoff_positives": int(((df[ASOF_COL] > cutoff) & (df[LABEL_COL] == 1)).sum()),
    }

    df = df.loc[~drop_mask].copy()
    debug["rows_after"] = int(len(df))

    logger.info("Right-censor filter:\n%s", json.dumps(debug, indent=2))
    return df, debug


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = df.copy()

    df = df[df[LABEL_COL].notna()]
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    df = df.sort_values([ACCOUNT_COL, ASOF_COL])

    before = len(df)
    df = df.drop_duplicates(subset=[ACCOUNT_COL, ASOF_COL], keep="last")
    dropped = before - len(df)

    if dropped:
        logger.warning("Dropped %d duplicate account/snapshot rows", dropped)

    drop_cols = {ACCOUNT_COL, ASOF_COL, LABEL_COL}
    feature_candidates = [c for c in df.columns if c not in drop_cols]

    for col in feature_candidates:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_cols = df[feature_candidates].select_dtypes(include=[np.number]).columns.tolist()

    dropped_non_numeric = sorted(set(feature_candidates) - set(feature_cols))
    if dropped_non_numeric:
        logger.warning("Dropping non-numeric feature columns: %s", dropped_non_numeric)

    if not feature_cols:
        raise RuntimeError("No numeric feature columns available after preprocessing")

    y = df[LABEL_COL].to_numpy(dtype=np.int32)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(f"Need both classes. Found n_pos={n_pos}, n_neg={n_neg}")

    logger.info("Using %d numeric features", len(feature_cols))
    return df, y, feature_cols


def split_last_n_months_adaptive(
    df: pd.DataFrame,
    initial_holdout_months: int = INITIAL_HOLDOUT_MONTHS,
    max_holdout_months: int = MAX_HOLDOUT_MONTHS,
    min_hold_pos: int = MIN_HOLDOUT_POS,
    min_hold_neg: int = MIN_HOLDOUT_NEG,
) -> SplitResult:
    df = df.copy()
    df["asof_month"] = df[ASOF_COL].dt.to_period("M").astype(str)

    months = sorted(df["asof_month"].unique())

    if len(months) < initial_holdout_months + 1:
        raise RuntimeError(f"Not enough months to split. Found {len(months)} months.")

    for n in range(initial_holdout_months, max_holdout_months + 1):
        if len(months) < n + 1:
            break

        hold_months = months[-n:]
        train_months = months[:-n]

        train_df = df[df["asof_month"].isin(train_months)].copy()
        hold_df = df[df["asof_month"].isin(hold_months)].copy()

        hold_pos = int((hold_df[LABEL_COL] == 1).sum())
        hold_neg = int((hold_df[LABEL_COL] == 0).sum())

        logger.info(
            "Trying holdout=%d months: rows=%d pos=%d neg=%d",
            n,
            len(hold_df),
            hold_pos,
            hold_neg,
        )

        if hold_pos >= min_hold_pos and hold_neg >= min_hold_neg:
            cutoff = pd.to_datetime(hold_months[0] + "-01")

            debug = {
                "split_type": "last_n_months_adaptive",
                "holdout_months": n,
                "train_month_min": train_months[0],
                "train_month_max": train_months[-1],
                "hold_month_min": hold_months[0],
                "hold_month_max": hold_months[-1],
                "n_train_rows": int(len(train_df)),
                "n_hold_rows": int(len(hold_df)),
                "train_pos_rows": int((train_df[LABEL_COL] == 1).sum()),
                "hold_pos_rows": hold_pos,
                "train_pos_rate": float(train_df[LABEL_COL].mean()),
                "hold_pos_rate": float(hold_df[LABEL_COL].mean()),
                "cutoff_asof": str(cutoff),
            }

            return SplitResult(
                train_df=train_df.drop(columns=["asof_month"]),
                hold_df=hold_df.drop(columns=["asof_month"]),
                cutoff_asof=cutoff,
                train_months=train_months,
                hold_months=hold_months,
                debug=debug,
            )

    raise RuntimeError(
        f"Could not find holdout window with at least "
        f"{min_hold_pos} positives and {min_hold_neg} negatives"
    )


def split_holdout_cal_test(
    hold_df: pd.DataFrame,
    cal_fraction: float = CAL_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    hold_df = hold_df.sort_values(ASOF_COL).copy()

    n = len(hold_df)
    cut = int(round(n * cal_fraction))
    cut = min(max(cut, 1), n - 1)

    cal_df = hold_df.iloc[:cut].copy()
    test_df = hold_df.iloc[cut:].copy()

    debug = {
        "split_type": "holdout_cal_test",
        "cal_fraction": cal_fraction,
        "n_cal_rows": int(len(cal_df)),
        "n_test_rows": int(len(test_df)),
        "cal_pos_rows": int((cal_df[LABEL_COL] == 1).sum()),
        "test_pos_rows": int((test_df[LABEL_COL] == 1).sum()),
        "cal_pos_rate": float(cal_df[LABEL_COL].mean()),
        "test_pos_rate": float(test_df[LABEL_COL].mean()),
    }

    if len(np.unique(cal_df[LABEL_COL])) == 2 and len(np.unique(test_df[LABEL_COL])) == 2:
        debug["fallback_used"] = False
        return cal_df, test_df, debug

    logger.warning("Chronological cal/test split produced a single-class split; using stratified fallback")

    rng = np.random.RandomState(RANDOM_STATE)
    y = hold_df[LABEL_COL].to_numpy(dtype=int)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_cal = cut
    pos_rate = len(pos_idx) / n
    n_pos_cal = min(len(pos_idx), max(1, int(round(n_cal * pos_rate))))
    n_neg_cal = min(len(neg_idx), max(1, n_cal - n_pos_cal))

    cal_idx = np.concatenate(
        [
            rng.choice(pos_idx, size=n_pos_cal, replace=False),
            rng.choice(neg_idx, size=n_neg_cal, replace=False),
        ]
    )

    cal_idx = np.unique(cal_idx)
    test_idx = np.setdiff1d(np.arange(n), cal_idx)

    cal_df = hold_df.iloc[cal_idx].copy()
    test_df = hold_df.iloc[test_idx].copy()

    debug.update(
        {
            "fallback_used": True,
            "fallback_type": "stratified_row_split",
            "n_cal_rows": int(len(cal_df)),
            "n_test_rows": int(len(test_df)),
            "cal_pos_rows": int((cal_df[LABEL_COL] == 1).sum()),
            "test_pos_rows": int((test_df[LABEL_COL] == 1).sum()),
            "cal_pos_rate": float(cal_df[LABEL_COL].mean()),
            "test_pos_rate": float(test_df[LABEL_COL].mean()),
        }
    )

    return cal_df, test_df, debug


def impute_with_train_medians(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Optional[float]]]:
    medians = train_df[feature_cols].median(numeric_only=True)

    medians_dict = {
        col: None if pd.isna(value) else float(value)
        for col, value in medians.to_dict().items()
    }

    train_X = train_df[feature_cols].fillna(medians).to_numpy(dtype=np.float32)
    other_X = other_df[feature_cols].fillna(medians).to_numpy(dtype=np.float32)

    return train_X, other_X, medians_dict


def train_xgb_classifier(
    train_X: np.ndarray,
    train_y: np.ndarray,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    pos = float(np.sum(train_y == 1))
    neg = float(np.sum(train_y == 0))

    if pos == 0 or neg == 0:
        raise RuntimeError(f"Training set is single-class: pos={pos}, neg={neg}")

    scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=4000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        gamma=0.0,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model.fit(
        train_X,
        train_y,
        eval_set=[(eval_X, eval_y)],
        verbose=False,
        early_stopping_rounds=200,
    )

    best_iteration = getattr(model, "best_iteration", None)

    return model, {
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": None if best_iteration is None else int(best_iteration),
    }


def platt_calibrate_if_possible(
    model: xgb.XGBClassifier,
    cal_X: np.ndarray,
    cal_y: np.ndarray,
    test_X: np.ndarray,
) -> Tuple[np.ndarray, Optional[LogisticRegression], Dict[str, Any]]:
    booster = model.get_booster()

    cal_margin = booster.predict(xgb.DMatrix(cal_X), output_margin=True)
    test_margin = booster.predict(xgb.DMatrix(test_X), output_margin=True)

    raw_test_proba = 1.0 / (1.0 + np.exp(-test_margin))

    unique_labels = np.unique(cal_y)
    debug = {"calibration_labels": [int(x) for x in unique_labels.tolist()]}

    if len(unique_labels) < 2:
        logger.warning("Skipping Platt calibration because calibration split is single-class")
        debug["platt_used"] = False
        return raw_test_proba, None, debug

    platt = LogisticRegression(solver="lbfgs", C=1e3, max_iter=2000)
    platt.fit(cal_margin.reshape(-1, 1), cal_y.astype(int))

    calibrated_proba = platt.predict_proba(test_margin.reshape(-1, 1))[:, 1]

    debug["platt_used"] = True
    return calibrated_proba, platt, debug


def compute_ranking_metrics(y: np.ndarray, proba: np.ndarray) -> List[Dict[str, Any]]:
    n = len(y)
    base_rate = float(np.mean(y)) if n else float("nan")
    order = np.argsort(-proba)
    total_pos = float(np.sum(y == 1))

    rows: List[Dict[str, Any]] = []

    for frac in RANK_FRACS:
        k = max(1, int(round(n * frac)))
        top = order[:k]

        pos_in_top = float(np.sum(y[top] == 1))
        precision_at_k = float(pos_in_top / k)
        recall_at_k = float(pos_in_top / total_pos) if total_pos else float("nan")
        lift_at_k = float(precision_at_k / base_rate) if base_rate else float("nan")

        rows.append(
            {
                "k_frac": float(frac),
                "k": int(k),
                "base_rate": base_rate,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "lift_at_k": lift_at_k,
                "pos_in_top": pos_in_top,
                "pos_total": total_pos,
            }
        )

    return rows


def compute_metrics(y: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
    y = y.astype(np.int32)
    unique = np.unique(y)
    has_both_classes = len(unique) == 2

    precision, recall, thresholds = precision_recall_curve(y, proba)
    thresholds = np.append(thresholds, 1.0)

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1))

    return {
        "num_rows": int(len(y)),
        "num_pos": int(np.sum(y == 1)),
        "num_neg": int(np.sum(y == 0)),
        "roc_auc": float(roc_auc_score(y, proba)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if has_both_classes else float("nan"),
        "logloss": float(log_loss(y, proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, proba)),
        "best_f1": float(f1[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
        "ranking": compute_ranking_metrics(y, proba),
        "debug": {
            "mean_proba": float(np.mean(proba)),
            "unique_labels": [int(x) for x in unique.tolist()],
        },
    }


def save_artifacts(
    model: xgb.XGBClassifier,
    platt: Optional[LogisticRegression],
    feature_cols: List[str],
    medians: Dict[str, Optional[float]],
    metrics: Dict[str, Any],
) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.get_booster().save_model(MODEL_JSON_PATH)

    if platt is not None:
        with open(PLATT_SCALER_PATH, "wb") as f:
            pickle.dump(platt, f)

    with open(FEATURES_PATH, "w") as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    with open(MEDIANS_PATH, "w") as f:
        json.dump(medians, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    with tarfile.open(MODEL_TAR_PATH, "w:gz") as tar:
        tar.add(MODEL_JSON_PATH, arcname="model/xgb_model.json")

        if platt is not None and os.path.exists(PLATT_SCALER_PATH):
            tar.add(PLATT_SCALER_PATH, arcname="model/platt_scaler.pkl")

        tar.add(FEATURES_PATH, arcname="feature_names.txt")
        tar.add(MEDIANS_PATH, arcname="feature_medians.json")
        tar.add(METRICS_PATH, arcname="metrics.json")

    logger.info("Saved model archive to %s", MODEL_TAR_PATH)


def main() -> None:
    logger.info("Starting churn-risk XGBoost training job")

    df = load_data()

    asof_debug = validate_asof_coverage(df)

    df, censor_debug = apply_right_censor_filter(df)

    df, _, feature_cols = preprocess(df)

    split = split_last_n_months_adaptive(df)

    cal_df, test_df, cal_test_debug = split_holdout_cal_test(split.hold_df)

    train_X, cal_X, medians = impute_with_train_medians(split.train_df, cal_df, feature_cols)
    _, test_X, _ = impute_with_train_medians(split.train_df, test_df, feature_cols)

    train_y = split.train_df[LABEL_COL].to_numpy(dtype=np.int32)
    cal_y = cal_df[LABEL_COL].to_numpy(dtype=np.int32)
    test_y = test_df[LABEL_COL].to_numpy(dtype=np.int32)

    model, train_debug = train_xgb_classifier(train_X, train_y, cal_X, cal_y)

    test_proba, platt, calibration_debug = platt_calibrate_if_possible(
        model=model,
        cal_X=cal_X,
        cal_y=cal_y,
        test_X=test_X,
    )

    metrics = compute_metrics(test_y, test_proba)

    metrics["split_debug"] = {
        **split.debug,
        **cal_test_debug,
        **train_debug,
        "asof_coverage": asof_debug,
        "censor_filter": censor_debug,
    }

    metrics["calibration_debug"] = calibration_debug

    logger.info("Final metrics:\n%s", json.dumps(metrics, indent=2))

    save_artifacts(
        model=model,
        platt=platt,
        feature_cols=feature_cols,
        medians=medians,
        metrics=metrics,
    )

    logger.info("Training job completed successfully")


if __name__ == "__main__":
    main()
