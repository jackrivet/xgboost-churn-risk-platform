import os
import sys
import json
import glob
import logging
import datetime as dt
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

RAW_DIR = "/opt/ml/processing/input/raw"
OUTPUT_DIR = "/opt/ml/processing/output"
MODEL_BASE_DIR = "/opt/ml/processing/model"

ACCOUNT_COL = "account_id"
DATA_SNAPSHOT_COL = "data_snapshot_date"
INFERENCE_RUN_COL = "inference_run_date"

logger = logging.getLogger("churn_xgb_inference")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
logger.handlers = [handler]



def _list_candidate_files(root: str) -> List[str]:
    patterns = [
        os.path.join(root, "**", "*.parquet"),
        os.path.join(root, "**", "*.snappy.parquet"),
        os.path.join(root, "**", "*"),
    ]

    out: List[str] = []
    seen = set()

    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if os.path.isfile(p) and p not in seen:
                seen.add(p)
                out.append(p)

    return out



def load_raw_df(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    files = _list_candidate_files(raw_dir)

    if not files:
        raise RuntimeError(f"No input files found under {raw_dir}")

    dfs: List[pd.DataFrame] = []
    skipped = 0

    for path in files:
        try:
            df_part = pd.read_parquet(path)

            if len(df_part) == 0:
                skipped += 1
                continue

            dfs.append(df_part)

        except Exception:
            skipped += 1

    if not dfs:
        raise RuntimeError(
            f"No readable parquet files found under {raw_dir} "
            f"(checked {len(files)} files)"
        )

    df = pd.concat(dfs, ignore_index=True)

    logger.info(
        "Loaded inference raw df shape=%s (skipped_files=%d, total_candidates=%d)",
        df.shape,
        skipped,
        len(files),
    )

    for col in [DATA_SNAPSHOT_COL, INFERENCE_RUN_COL]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    today = dt.date.today()

    if INFERENCE_RUN_COL not in df.columns:
        df[INFERENCE_RUN_COL] = today
    else:
        df[INFERENCE_RUN_COL] = df[INFERENCE_RUN_COL].fillna(today)

    if ACCOUNT_COL in df.columns:
        if DATA_SNAPSHOT_COL in df.columns:
            df = (
                df.sort_values([ACCOUNT_COL, DATA_SNAPSHOT_COL])
                .drop_duplicates(subset=[ACCOUNT_COL], keep="last")
            )
        else:
            df = df.drop_duplicates(subset=[ACCOUNT_COL], keep="first")

    logger.info(
        "After defensive dedupe: shape=%s (unique_accounts=%d)",
        df.shape,
        df[ACCOUNT_COL].nunique() if ACCOUNT_COL in df.columns else -1,
    )

    return df



def _find_file_anywhere(base_dir: str, filename: str) -> Optional[str]:
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)

    return None



def load_model_bundle(
    model_base_dir: str = MODEL_BASE_DIR,
) -> Tuple[List[str], Dict[str, float], xgb.Booster, Optional[object]]:
    fn_path = _find_file_anywhere(model_base_dir, "feature_names.txt")
    fm_path = _find_file_anywhere(model_base_dir, "feature_medians.json")
    model_path = _find_file_anywhere(model_base_dir, "xgb_model.json")
    platt_path = _find_file_anywhere(model_base_dir, "platt_scaler.pkl")

    if not fn_path or not fm_path or not model_path:
        logger.error("Could not locate required artifacts under %s", model_base_dir)

        for root, _, files in os.walk(model_base_dir):
            logger.info("DIR %s FILES %s", root, files)

        missing = [
            name
            for name, path in [
                ("feature_names.txt", fn_path),
                ("feature_medians.json", fm_path),
                ("xgb_model.json", model_path),
            ]
            if path is None
        ]

        raise FileNotFoundError(
            f"Missing required artifacts under {model_base_dir}: {missing}"
        )

    with open(fn_path, "r") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    with open(fm_path, "r") as f:
        feature_medians_raw = json.load(f)

    def _safe_float(v, default: float = 0.0) -> float:
        if v is None:
            return default

        try:
            value = float(v)
            if np.isnan(value):
                return default
            return value

        except Exception:
            return default

    feature_medians: Dict[str, float] = {
        k: _safe_float(v, 0.0) for k, v in feature_medians_raw.items()
    }

    n_bad = sum(
        (v is None) or (isinstance(v, (float, int)) and np.isnan(float(v)))
        for v in feature_medians_raw.values()
    )

    if n_bad:
        logger.warning(
            "feature_medians.json had %d null/NaN values; defaulted to 0.0",
            n_bad,
        )

    booster = xgb.Booster()
    booster.load_model(model_path)

    platt_scaler = None

    if platt_path and os.path.exists(platt_path):
        platt_scaler = joblib.load(platt_path)
        logger.info("Loaded Platt scaler at %s", platt_path)
    else:
        logger.warning(
            "No platt_scaler.pkl found; will fall back to XGB logistic probability."
        )

    logger.info(
        "Loaded model bundle: features=%d, medians=%d, model=%s",
        len(feature_names),
        len(feature_medians),
        model_path,
    )

    return feature_names, feature_medians, booster, platt_scaler



def build_X(
    df: pd.DataFrame,
    feature_names: List[str],
    feature_medians: Dict[str, float],
) -> pd.DataFrame:
    n = len(df)

    X = np.empty((n, len(feature_names)), dtype=np.float32)

    for j, feat in enumerate(feature_names):
        if feat in df.columns:
            col = pd.to_numeric(df[feat], errors="coerce")
            med = float(
                feature_medians.get(
                    feat,
                    col.median(skipna=True) if col.notna().any() else 0.0,
                )
            )
            X[:, j] = col.fillna(med).astype(np.float32).to_numpy()
        else:
            med = float(feature_medians.get(feat, 0.0))
            X[:, j] = np.full(n, med, dtype=np.float32)

    X_df = pd.DataFrame(X, columns=feature_names)

    logger.info("Built X matrix shape=%s", X_df.shape)

    return X_df



def predict_proba(
    booster: xgb.Booster,
    platt_scaler,
    X_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    dmat = xgb.DMatrix(X_df.values, feature_names=feature_names)

    raw_margin = booster.predict(dmat, output_margin=True)

    if platt_scaler is not None:
        proba = platt_scaler.predict_proba(raw_margin.reshape(-1, 1))[:, 1]
        return raw_margin.astype(np.float64), proba.astype(np.float64)

    native_proba = booster.predict(dmat)
    return raw_margin.astype(np.float64), native_proba.astype(np.float64)



def make_output(
    df_raw: pd.DataFrame,
    raw_margin: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            ACCOUNT_COL: df_raw[ACCOUNT_COL].astype(str).values,
            "data_snapshot_date": (
                df_raw[DATA_SNAPSHOT_COL].astype("object").values
                if DATA_SNAPSHOT_COL in df_raw.columns
                else None
            ),
            "inference_run_date": (
                df_raw[INFERENCE_RUN_COL].astype("object").values
                if INFERENCE_RUN_COL in df_raw.columns
                else dt.date.today()
            ),
            "churn_score_raw": raw_margin,
            "churn_proba": np.clip(proba, 0.0, 1.0),
        }
    )

    out["data_snapshot_date"] = pd.to_datetime(
        out["data_snapshot_date"],
        errors="coerce",
    ).dt.date

    out["inference_run_date"] = pd.to_datetime(
        out["inference_run_date"],
        errors="coerce",
    ).dt.date

    out = out.sort_values("churn_proba", ascending=False)
    out["prob_rank"] = np.arange(1, len(out) + 1, dtype=np.int64)

    return out



def write_parquet(df_out: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "predictions.parquet")

    df_out.to_parquet(path, index=False)

    logger.info("Wrote output parquet: %s (rows=%d)", path, len(df_out))

    return path



def main() -> None:
    logger.info("Starting churn-risk inference processing job.")

    df_raw = load_raw_df()

    missing = [
        col
        for col in [ACCOUNT_COL, DATA_SNAPSHOT_COL]
        if col not in df_raw.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns from inference input: {missing}. "
            f"The inference dataset must include account_id and data_snapshot_date."
        )

    feature_names, feature_medians, booster, platt_scaler = load_model_bundle()

    logger.info(
        "Model feature count = %d (first 20=%s)",
        len(feature_names),
        feature_names[:20],
    )

    missing_feats = [f for f in feature_names if f not in df_raw.columns]
    present_feats = [f for f in feature_names if f in df_raw.columns]

    logger.info(
        "Inference input cols=%d. Present model feats=%d. "
        "Missing model feats=%d (first 30 missing=%s)",
        df_raw.shape[1],
        len(present_feats),
        len(missing_feats),
        missing_feats[:30],
    )

    X_df = build_X(df_raw, feature_names, feature_medians)

    uniq_rows = X_df.drop_duplicates().shape[0]
    logger.info("X_df shape=%s unique_rows=%d", X_df.shape, uniq_rows)

    nunique_by_col = X_df.nunique(dropna=False).sort_values()

    logger.info("Bottom 20 nunique cols:\n%s", nunique_by_col.head(20).to_string())
    logger.info("Top 20 nunique cols:\n%s", nunique_by_col.tail(20).to_string())

    raw_margin, proba = predict_proba(
        booster,
        platt_scaler,
        X_df,
        feature_names,
    )

    df_out = make_output(df_raw, raw_margin, proba)

    logger.info(
        "Scored %d accounts. proba summary: "
        "min=%.6f p50=%.6f p90=%.6f p99=%.6f max=%.6f mean=%.6f",
        len(df_out),
        float(df_out["churn_proba"].min()),
        float(df_out["churn_proba"].median()),
        float(df_out["churn_proba"].quantile(0.90)),
        float(df_out["churn_proba"].quantile(0.99)),
        float(df_out["churn_proba"].max()),
        float(df_out["churn_proba"].mean()),
    )

    write_parquet(df_out)

    logger.info("Inference job completed successfully.")



if __name__ == "__main__":
    main()
