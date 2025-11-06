#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py — Load a trained model, rebuild features, and generate predictions.

Basic usage:
    python predict.py \
        --model path/model.pkl \
        --input data/input.csv \
        --output preds.csv \
        --id-cols date station_id \
        --features-json training_features.json
"""

from __future__ import annotations

# ---------------------------- Imports ----------------------------
import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Sequence
import re
import glob
try:
    import xgboost as xgb   # opcional, solo si tienes Booster .json
except Exception:
    xgb = None
from datetime import timezone
from datetime import datetime, date, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # fallback handled below

import numpy as np
import pandas as pd

# Prefer joblib; fall back to pickle if unavailable
try:
    import joblib  # type: ignore
    _HAVE_JOBLIB = True
except Exception:
    import pickle
    _HAVE_JOBLIB = False

# ---------------------------- Constants ----------------------------
WINDOWS  = (7, 14, 30, 60)
MIN_FRAC = 0.7

METEO_COLS = [
    "wind_speed_2m","wind_avg","precipitation","air_temperature","temp_max","temp_min",
    "air_pressure","Watt_per_m2","PAR","water_temperature","water_level"
]
OPTIC_COLS = [
    "CHL_OC4ME","TSM_NN","ADG443_NN","KD490_M07","T865","A865","rho_665","rho_681","rho_709",
    "CIcyano","Oa01_reflectance","Oa02_reflectance","Oa03_reflectance","Oa04_reflectance",
    "Oa05_reflectance","Oa06_reflectance","Oa07_reflectance","Oa08_reflectance","Oa09_reflectance",
    "Oa10_reflectance","Oa11_reflectance","Oa12_reflectance","Oa16_reflectance","Oa17_reflectance",
    "Oa18_reflectance","Oa21_reflectance"
]

# ---------------------------- Logging ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict")

# ---------------------------- Model I/O ----------------------------
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    log.info(f"Loading model from: {path}")
    ext = os.path.splitext(path)[1].lower()

    # XGBoost Booster saved as JSON
    if ext == ".json":
        if xgb is None:
            raise RuntimeError("xgboost is not installed; cannot load Booster JSON.")
        bst = xgb.Booster()
        bst.load_model(path)
        return bst

    # sklearn/joblib/pickle
    if _HAVE_JOBLIB:
        return joblib.load(path)  # may return estimator or dict-bundle
    with open(path, "rb") as f:
        return pickle.load(f)

def unwrap_model_bundle(obj):
    """
    Accepts a bare estimator/pipeline, an xgboost Booster, or a dict bundle.
    Returns (estimator_like, preproc_or_None).
    Recognized bundle keys: 'pipeline','model','estimator','clf','regressor','scaler'.
    """
    # Booster goes through as-is (preproc None)
    if "xgboost.core.Booster" in str(type(obj)):
        return obj, None

    if isinstance(obj, dict):
        est = None
        for k in ("pipeline", "model", "estimator", "clf", "regressor"):
            if k in obj:
                est = obj[k]
                break
        preproc = obj.get("scaler", None)
        if est is None:
            raise TypeError(f"Model bundle dict missing estimator; keys={list(obj.keys())}")
        return est, preproc

    return obj, None  # plain estimator/pipeline


def predict_any(est, X_df: pd.DataFrame, preproc=None):
    """
    Predict with sklearn-like estimator/pipeline or xgboost Booster.
    Applies 'preproc.transform' if provided and estimator is not a Pipeline.
    Returns (yhat, proba_or_None).
    """
    X_in = X_df

    # apply scaler if provided and estimator isn't already a Pipeline
    if preproc is not None and hasattr(preproc, "transform") and not hasattr(est, "steps"):
        try:
            X_tr = preproc.transform(X_df)
            # keep DataFrame for Booster to preserve feature_names
            X_in = pd.DataFrame(X_tr, index=X_df.index, columns=X_df.columns)
            log.info("Applied bundled scaler to features.")
        except Exception as e:
            log.warning(f"Scaler.transform failed; using raw features. Error: {e}")
            X_in = X_df

    # Booster path
    if "xgboost.core.Booster" in str(type(est)):
        if xgb is None:
            raise RuntimeError("xgboost not available for Booster prediction.")
        dmx = xgb.DMatrix(X_in, feature_names=list(X_in.columns))
        yhat = est.predict(dmx)
        return yhat, None

    # sklearn-like
    proba = None
    if hasattr(est, "predict_proba"):
        try:
            proba = est.predict_proba(X_in)
        except Exception:
            proba = None
    yhat = est.predict(X_in)
    return yhat, proba

# ---------------------------- Time / Date utils ----------------------------
def _local_yesterday(tz_str: str) -> date:
    now = datetime.now(ZoneInfo(tz_str)) if ZoneInfo else datetime.now()
    return (now.date() - timedelta(days=1))

def _to_local_date(datestr: str | None, tz_str: str) -> date:
    if datestr is None:
        return _local_yesterday(tz_str)
    d = pd.to_datetime(datestr, errors="coerce")
    if pd.isna(d):
        raise ValueError("Invalid --predict-date. Use YYYY-MM-DD.")
    return d.date()

def _coerce_date_col(df: pd.DataFrame, date_col: str | None):
    df = df.copy()
    chosen = date_col or ("date" if "date" in df.columns else "datetime" if "datetime" in df.columns else None)
    if chosen is None:
        df["date"] = pd.to_datetime(df.index, errors="coerce").floor("D")
        return df, "date"
    if not pd.api.types.is_datetime64_any_dtype(df[chosen]):
        df[chosen] = pd.to_datetime(df[chosen], errors="coerce")
    df[chosen] = df[chosen].dt.floor("D")
    if chosen != "date":
        df["date"] = df[chosen]
        chosen = "date"
    return df, chosen

def _select_rows_for_exact_date(df: pd.DataFrame, date_col: str, target_date: date):
    mask = df[date_col].dt.date == target_date
    return df[mask].index

def _extract_h_from_dirname(name: str) -> Optional[int]:
    """
    Parse horizons like '7d', '15d', 'h7', 'H15', etc. Returns int or None.
    """
    m = re.search(r"(\d+)\s*d", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"[hH]\s*(\d+)", name)
    return int(m.group(1)) if m else None

def _label_from_thresholds(y: float, q1: float, q2: float) -> str:
    if y < q1:
        return "low"
    if y <= q2:
        return "medium"
    return "high"

def find_model_file(model_dir: str) -> str:
    """
    Pick a model file inside model_dir with sensible priority.
    Accepts: model.pkl / model.joblib / *.joblib / *.pkl / *.sav / *.json (XGBoost Booster).
    """
    # priority names
    for name in ("model.pkl", "model.joblib", "model.sav"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    # generic patterns
    for pat in ("*.joblib", "*.pkl", "*.sav", "*.json"):
        files = sorted(glob.glob(os.path.join(model_dir, pat)))
        if files:
            return files[0]
    raise FileNotFoundError(f"No model file found in: {model_dir}")

# ---------------------------- Yesterday filler (simple) ----------------------------
def ensure_yday_with_last_values(
    df: pd.DataFrame,
    date_col: str = "date",
    tz: str = "America/Puerto_Rico",
) -> pd.DataFrame:
    """
    Ensure there's a row for local 'yesterday'; for each column, copy the last
    non-NaN value observed on or before yesterday. If the row doesn't exist, create it.
    """
    df = df.copy()

    # normalize date column
    if date_col not in df.columns and "datetime" in df.columns:
        df[date_col] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df[date_col] = pd.to_datetime(df.get(date_col, pd.NaT), errors="coerce")
    df[date_col] = df[date_col].dt.floor("D")

    yday = _local_yesterday(tz)

    # columns to fill = all except the date column
    fill_cols = [c for c in df.columns if c != date_col]

    def last_valid_le_yday(series: pd.Series) -> object:
        s = series[df[date_col].dt.date <= yday].dropna()
        return s.iloc[-1] if len(s) else pd.NA

    df = df.sort_values(date_col)
    mask_y = (df[date_col].dt.date == yday)

    if not mask_y.any():
        # create a new row for yesterday
        new_row = {date_col: pd.Timestamp(yday)}
        for c in fill_cols:
            new_row[c] = last_valid_le_yday(df[c])
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # update the last 'yesterday' row with last non-NaN values per column
        idx = df.index[mask_y][-1]
        for c in fill_cols:
            df.loc[idx, c] = last_valid_le_yday(df[c])

    df = df.sort_values(date_col).reset_index(drop=True)
    return df

# ---------------------------- Feature helpers ----------------------------
def _add_prefixed(out, df_stat, pfx, name):
    if isinstance(df_stat, pd.Series):
        df_stat = df_stat.to_frame()
    return pd.concat([out, df_stat.add_prefix(f"{pfx}{name}_")], axis=1)

def _rolling_feats_shifted(df, cols, windows=WINDOWS, min_frac=MIN_FRAC, prefix=""):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)

    out   = pd.DataFrame(index=df.index)
    X     = df[cols].astype(float)
    X_lag = X.shift(1)  # avoid leakage

    for w in windows:
        minp = max(1, int(w * min_frac))
        r    = X_lag.rolling(window=w, min_periods=minp)
        pfx  = f"{prefix}{w}d_"

        out = _add_prefixed(out, r.mean(),                           pfx, "mean")
        out = _add_prefixed(out, r.std(ddof=0),                      pfx, "std")
        out = _add_prefixed(out, r.median(),                         pfx, "median")
        out = _add_prefixed(out, r.quantile(0.95),                   pfx, "p95")
        out = _add_prefixed(out, r.quantile(0.75) - r.quantile(0.25), pfx, "iqr")

        if w >= 28:
            mu  = r.mean()
            xc  = X_lag - mu
            rxc = xc.rolling(w, min_periods=minp)
            mu2 = rxc.apply(lambda s: np.mean(s**2), raw=False)
            mu3 = rxc.apply(lambda s: np.mean(s**3), raw=False)
            mu4 = rxc.apply(lambda s: np.mean(s**4), raw=False)
            out = _add_prefixed(out, mu3/(mu2**1.5), pfx, "skew")
            out = _add_prefixed(out, mu4/(mu2**2),   pfx, "kurt")

        if "wind_speed_2m" in X_lag.columns:
            ev3 = X_lag["wind_speed_2m"].rolling(w, min_periods=minp)\
                  .apply(lambda s: np.mean(s**3), raw=False)
            out[f"{pfx}E_v3_wind_speed_2m"] = ev3

    return out

# ---------------------------- Build features ----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Safe date parsing for 'datetime'/'date'.
    2) Builds weekly-smoothed CHL_NN_total -> CHL_W without leakage (ffill + shift(1)).
    3) Rolling stats for meteorological & optical groups on X.shift(1).
    4) y lags (1, 7, 14) from CHL_W.
    5) y rolling stats (shifted).
    6) Basic calendar splits.
    """
    df = df.copy()

    # 1) Dates
    for dcol in ("datetime", "date"):
        if dcol in df.columns and not pd.api.types.is_datetime64_any_dtype(df[dcol]):
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    if "date" not in df.columns and "datetime" in df.columns:
        df["date"] = df["datetime"].dt.floor("D")
    idx = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index, errors="coerce")

    # 2) Weekly smoothing without leakage -> CHL_W
    if "CHL_NN_total" in df.columns:
        df_tmp = df.copy()
        df_tmp.index = idx
        weekly = df_tmp["CHL_NN_total"].resample("W-SUN").mean()
        CHL_W = weekly.reindex(idx, method="ffill").shift(1)
        df["CHL_W"] = CHL_W.values

    # 3) Rolling feats (shifted)
    feats_meteo = _rolling_feats_shifted(df, METEO_COLS, windows=WINDOWS, min_frac=MIN_FRAC, prefix="met_")
    feats_optic = _rolling_feats_shifted(df, OPTIC_COLS, windows=WINDOWS, min_frac=MIN_FRAC, prefix="opt_")

    # 4) y lags from target CHL_W
    target = "CHL_W"
    ylags = pd.DataFrame(index=df.index)
    if target in df.columns:
        ylags["y_lag1"]  = df[target].shift(1)
        ylags["y_lag7"]  = df[target].shift(7)
        ylags["y_lag14"] = df[target].shift(14)

    # 5) y rolling stats (shifted)
    yroll = pd.DataFrame(index=df.index)
    if target in df.columns:
        yroll = _rolling_feats_shifted(df[[target]], [target], windows=WINDOWS, min_frac=MIN_FRAC, prefix="y_")

    # 6) Concat & calendar
    X = pd.concat([feats_meteo, feats_optic, ylags, yroll], axis=1)
    X = X.drop(columns=[c for c in X.columns if c.startswith("y_7d_mean_")], errors="ignore")

    if "date" in df.columns:
        X["dow"]   = df["date"].dt.dayofweek
        X["month"] = df["date"].dt.month
        X["day"]   = df["date"].dt.day

    return X

# ---------------------------- Feature alignment ----------------------------
def load_feature_list(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"--features-json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "features" in data:
        data = data["features"]
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("features JSON must be a list of column names or {'features': [...]} ")
    return data  # type: ignore[return-value]

def align_columns(X: pd.DataFrame, expected: Sequence[str]) -> pd.DataFrame:
    X = X.copy()
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    return X.loc[:, list(expected)]

def infer_expected_features_from_model(model) -> Optional[List[str]]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    steps = getattr(model, "steps", None)
    if steps and len(steps) > 0:
        final_est = steps[-1][1]
        names = getattr(final_est, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None

# ---------------------------- Prediction ----------------------------

def run_all_models(
    models_root: str,
    input_csv: str,
    results_dir: str,
    features_json: Optional[str],
    tz: str,
    predict_date: Optional[str],
    q1: float,
    q2: float,
) -> None:
    """
    Iterate subfolders in models_root (e.g., models/xgboost/7d, 15d),
    run one prediction per model, and write a compact CSV per model in results_dir.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Execution date (when script runs) in local tz
    run_date = datetime.now(ZoneInfo(tz) if ZoneInfo else None).date()

    # Parent algo name for nicer filenames (e.g., 'xgboost')
    algo_name = os.path.basename(os.path.normpath(models_root))

    subdirs = [
        d for d in sorted(os.listdir(models_root))
        if os.path.isdir(os.path.join(models_root, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No model subfolders found in: {models_root}")

    for d in subdirs:
        model_dir = os.path.join(models_root, d)
        model_pkl = os.path.join(model_dir, "model.pkl")
        if not os.path.exists(model_pkl):
            log.warning(f"Skipping '{model_dir}' (missing model.pkl)")
            continue

        H = _extract_h_from_dirname(d)  # may be None, we handle it
        try:
            # scratch output for the underlying single-model run
            scratch_out = os.path.join(results_dir, f"_{algo_name}_{d}_scratch.csv")

            # run single-model prediction (returns float if exactly one row)
            y_value = run_prediction(
                model_path=model_pkl,
                input_csv=input_csv,
                output_csv=scratch_out,
                id_cols=["date"],
                features_json=features_json,
                prediction_col="y_pred",
                nrows=None,
                predict_date=predict_date,
                predict_strict_one=True,
                date_col="date",
                tz=tz,
            )

            # Assemble compact result row
            as_of_date = _to_local_date(predict_date, tz)
            predicted_date = (as_of_date + timedelta(days=H)) if H is not None else None
            label = _label_from_thresholds(float(y_value), q1, q2)

            row = {
                "run_date": run_date.isoformat(),
                "as_of_date": as_of_date.isoformat(),
                "horizon_days": H,
                "predicted_date": predicted_date.isoformat() if predicted_date else "",
                "y_pred": float(y_value),
                "label": label,
                "model_dir": model_dir,
            }

            # Write one-row CSV per model
            out_path = os.path.join(
                results_dir,
                f"{algo_name}_{d}_{run_date.isoformat()}.csv"
            )
            pd.DataFrame([row]).to_csv(out_path, index=False)
            log.info(f"Wrote: {out_path}")

        except Exception as e:
            log.exception(f"Failed running model at {model_dir}: {e}")


def run_prediction(
    model_path: str,
    input_csv: str,
    output_csv: str,
    id_cols: Optional[List[str]] = None,
    features_json: Optional[str] = None,
    prediction_col: str = "y_pred",
    save_x_used: Optional[str] = None,
    nrows: Optional[int] = None,
    # date controls
    predict_date: Optional[str] = None,      # "YYYY-MM-DD" or None => local yesterday
    predict_strict_one: bool = False,
    date_col: Optional[str] = None,
    tz: str = "America/Puerto_Rico",
):
    # 1) Load
    log.info(f"Reading CSV: {input_csv}")
    df_raw = pd.read_csv(input_csv, nrows=nrows)
    log.info(f"Raw shape: {df_raw.shape}")

    # 1.1) Ensure 'yesterday' row (simple per-column last non-NaN)
    df_raw = ensure_yday_with_last_values(
        df_raw,
        date_col=(date_col or "date"),
        tz=tz,
    )

    # 2) Build features
    X = build_features(df_raw)
    log.info(f"Shape after feature build: {X.shape}")

    # 3) Load model
    model = load_model(model_path)
    estimator, preproc = unwrap_model_bundle(model)
    log.info(f"Estimator type: {type(estimator)}; Preproc: {type(preproc) if preproc is not None else None}")

    # 4) Align to training columns
    expected = load_feature_list(features_json) or infer_expected_features_from_model(model)
    X_aligned = align_columns(X, expected) if expected is not None else X
    log.info(f"Shape used for prediction: {X_aligned.shape}")

    # 4.1) Select exact date rows
    df_ids, chosen_date_col = _coerce_date_col(df_raw, date_col)
    target_date = _to_local_date(predict_date, tz)  # defaults to local yesterday
    idx_for_date = _select_rows_for_exact_date(df_ids, chosen_date_col, target_date)

    if len(idx_for_date) == 0:
        raise ValueError(f"No data rows found for predict-date = {target_date} (timezone {tz}).")
    if predict_strict_one and len(idx_for_date) != 1:
        raise ValueError(f"Expected exactly 1 row for {target_date}, but found {len(idx_for_date)}.")

    # Subset
    X_pred  = X_aligned.loc[idx_for_date]
    ids_pred = df_raw.loc[idx_for_date]
    log.info(f"Predicting for date={target_date} rows={len(X_pred)}")

    # 5) Predict
    yhat, proba = predict_any(estimator, X_pred, preproc=preproc)

    # 6) Output frame
    out = pd.DataFrame(index=X_pred.index)
    if id_cols:
        missing = [c for c in id_cols if c not in ids_pred.columns]
        if missing:
            raise KeyError(f"Missing ID columns in input: {missing}")
        out[id_cols] = ids_pred[id_cols]
    out[prediction_col] = yhat

    if proba is not None and hasattr(proba, "shape"):
        n_classes = proba.shape[1]
        for k in range(n_classes):
            out[f"{prediction_col}_proba_{k}"] = proba[:, k]

    # 7) Write outputs
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    log.info(f"Predictions saved to: {output_csv} (rows={len(out)})")

    if save_x_used:
        ext = os.path.splitext(save_x_used)[1].lower()
        to_save = X_pred  # features actually used
        os.makedirs(os.path.dirname(save_x_used) or ".", exist_ok=True)
        if ext == ".parquet":
            to_save.to_parquet(save_x_used, index=False)
        else:
            to_save.to_csv(save_x_used, index=False)
        log.info(f"Saved features used to: {save_x_used}")

    # 8) Return
    if len(yhat) == 1:
        y_value = float(yhat[0])
        print(f"{prediction_col}({target_date}) = {y_value}")
        return y_value
    else:
        result = {int(i): float(v) for i, v in zip(out.index, yhat)}
        print(f"{prediction_col}({target_date}) = {result}")
        return result

# ---------------------------- CLI ----------------------------
# --- in parse_args() ---
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load a model, rebuild features, and predict on a CSV.")

    # Single-run (still optional; batch is default)
    p.add_argument("--model", default=None, help="Path to .pkl/.joblib model (single-run).")

    # Defaults you asked for:
    p.add_argument("--batch-models-root", default="models/xgboost",
                   help="Run all models under this folder (default: models/xgboost).")
    p.add_argument("--input", default="src/pipeline/final_data/SJL_daily_df.csv",
                   help="Input CSV (default: src/pipeline/final_data/SJL_daily_df.csv).")
    p.add_argument("--features-json", default="src/features/training_features.json",
                   help="Training feature list JSON (default: src/features/training_features.json).")
    p.add_argument("--results-dir", default="results",
                   help="Where to write per-model CSVs (default: results/).")

    # Common options
    p.add_argument("--output", default=None,
                   help="Output CSV for single-run mode (ignored in batch).")
    p.add_argument("--id-cols", nargs="*", default=None, help="Columns to keep in the output.")
    p.add_argument("--prediction-col", default="y_pred", help="Name of the prediction column.")
    p.add_argument("--save-x-used", default=None, help="Save features actually used (CSV/Parquet).")
    p.add_argument("--nrows", type=int, default=None, help="Read only first N rows (debug).")

    # Thresholds (your defaults)
    p.add_argument("--q1", type=float, default=10.148629867177084,
                   help="Lower threshold for labeling (default: 10.148629867177084).")
    p.add_argument("--q2", type=float, default=15.377913418040292,
                   help="Upper threshold for labeling (default: 15.377913418040292).")

    # Date / TZ
    p.add_argument("--date-col", default=None, help="Name of the date column (defaults to 'date' or 'datetime').")
    p.add_argument("--tz", default="America/Puerto_Rico", help="IANA TZ for 'yesterday'.")
    p.add_argument("--predict-date", default=None, help="YYYY-MM-DD for target date (default: yesterday).")

    # Default TRUE using BooleanOptionalAction so user can disable with --no-predict-strict-one
    p.add_argument("--predict-strict-one", action=argparse.BooleanOptionalAction, default=True,
                   help="Require exactly one row for the prediction date (default: True).")

    return p.parse_args(argv)


# --- in main() ---
def main():
    args = parse_args()

    # Batch is the default path (uses defaults above). User can still pass --model for single run.
    if args.batch_models_root:
        run_all_models(
            models_root=args.batch_models_root,
            input_csv=args.input,
            results_dir=args.results_dir,
            features_json=args.features_json,
            tz=args.tz,
            predict_date=args.predict_date,
            q1=args.q1,
            q2=args.q2,
        )
        return

    # Single-model fallback (only if --model is provided)
    if not args.model:
        raise SystemExit("No --model given and no --batch-models-root set. (Batch is default; did you remove it?)")

    if not args.output:
        raise SystemExit("For single-run mode, please provide --output.")

    run_prediction(
        model_path=args.model,
        input_csv=args.input,
        output_csv=args.output,
        id_cols=args.id_cols,
        features_json=args.features_json,
        prediction_col=args.prediction_col,
        save_x_used=args.save_x_used,
        nrows=args.nrows,
        predict_date=args.predict_date,
        predict_strict_one=args.predict_strict_one,
        date_col=args.date_col,
        tz=args.tz,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(e)
        sys.exit(1)
