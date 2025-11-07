# gold_merge.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
from datetime import date, timedelta
from typing import Optional, Tuple, List, Set

import pandas as pd

# --- bootstrap import path to repo root ---
import sys
from pathlib import Path
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        break
# -----------------------------------------
from utils.paths import norm, REPO_ROOT  # requires utils/paths.py in the repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gold")

DEFAULT_FINAL_DIR  = "src/pipeline/final_data"
DEFAULT_FINAL_CSV  = os.path.join(DEFAULT_FINAL_DIR, "SJL_daily_df.csv")
DEFAULT_STATE_PATH = os.path.join(DEFAULT_FINAL_DIR, "gold_state.json")


# ---------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------
def _to_date_only(dtlike) -> pd.Timestamp:
    ts = pd.to_datetime(dtlike, errors="coerce")
    if pd.isna(ts):
        return ts
    return ts.normalize()

def _yesterday_str() -> str:
    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

def _parse_user_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        return pd.to_datetime(s).normalize()
    except Exception:
        raise ValueError(f"Invalid date: {s}")

def _max_date_in_csv(path: str) -> Optional[pd.Timestamp]:
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, usecols=["date"])
        if df.empty:
            return None
        d = pd.to_datetime(df["date"], errors="coerce").dropna()
        if d.empty:
            return None
        return d.max().normalize()
    except Exception as e:
        logger.warning(f"Could not read last date from {path}: {e}")
        return None

def _normalize_goes_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # limpia espacios y normaliza
    df = df.rename(columns=lambda c: c.strip())
    # mapea variantes a un nombre canónico
    lower_map = {c.lower(): c for c in df.columns}
    if "magnitude" in lower_map and "Watt_per_m2" not in df.columns:
        df = df.rename(columns={lower_map["magnitude"]: "Watt_per_m2"})
    # si por alguna razón existen ambas, consolida y elimina 'Magnitude'
    if "Magnitude" in df.columns and "Watt_per_m2" in df.columns:
        df["Watt_per_m2"] = df["Watt_per_m2"].fillna(df["Magnitude"])
        df = df.drop(columns=["Magnitude"])
    return df


def _min_date_in_inputs(paths: List[str]) -> Optional[pd.Timestamp]:
    mins = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=["date"])
            if df.empty:
                continue
            d = pd.to_datetime(df["date"], errors="coerce").dropna()
            if not d.empty:
                mins.append(d.min().normalize())
        except Exception as e:
            logger.warning(f"Could not read min date from {p}: {e}")
    if not mins:
        return None
    return min(mins)

def resolve_window(
    final_csv: str,
    user_start: Optional[str],
    user_end: Optional[str],
    input_paths: List[str]
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    end: user_end or YESTERDAY.
    start:
      - if user_start -> start=user_start
      - elif OUT exists -> start = max(OUT)+1 day
      - else -> start = min(date) across inputs (or end if nothing)
    """
    end_ts = _parse_user_date(user_end) or pd.to_datetime(_yesterday_str()).normalize()

    if user_start:
        start_ts = _parse_user_date(user_start)
        if start_ts is None:
            raise ValueError("Invalid --start date.")
    else:
        last_in_final = _max_date_in_csv(final_csv)
        if last_in_final is not None:
            start_ts = (last_in_final + pd.Timedelta(days=1)).normalize()
        else:
            min_in_inputs = _min_date_in_inputs(input_paths)
            start_ts = min_in_inputs or end_ts

    if start_ts > end_ts:
        logger.info(f"Nothing to update (start {start_ts.date()} > end {end_ts.date()}).")
    else:
        logger.info(f"Effective window: {start_ts.date()} .. {end_ts.date()}")
    return start_ts, end_ts


# ---------------------------------------------------------------------
# IO and preprocessing
# ---------------------------------------------------------------------
def _read_daily_csv(path: Optional[str], label: str) -> pd.DataFrame:
    """
    Reads a CSV with 'date', normalizes 'date', dedups per day (keep last), sorts by date.
    """
    if not path or not os.path.exists(path):
        logger.info(f"{label}: not found -> skip")
        return pd.DataFrame(columns=["date"])
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning(f"{label}: could not read {path}: {e}")
        return pd.DataFrame(columns=["date"])

    if "date" not in df.columns:
        logger.warning(f"{label}: missing 'date' column in {path}")
        return pd.DataFrame(columns=["date"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return df

def _subset_window(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    return df.loc[mask].copy()

def _prepare_source_no_prefix(df: pd.DataFrame,
                              start_ts: pd.Timestamp,
                              end_ts: pd.Timestamp,
                              renames: Optional[dict] = None) -> pd.DataFrame:
    """
    - Cuts to [start..end], normalizes date, dedups per day (last).
    - Applies renames (if provided).
    - Returns DF indexed by 'date' (without 'date' column), no prefixes.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "date" not in out.columns:
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    mask = (out["date"] >= start_ts) & (out["date"] <= end_ts)
    out = out.loc[mask].copy()
    if out.empty:
        return pd.DataFrame()

    if renames:
        out = out.rename(columns=renames)

    out = out.set_index("date").sort_index()
    out.index = pd.to_datetime(out.index, errors="coerce")  # keep DatetimeIndex dtype
    return out


# ---------------------------------------------------------------------
# Overlay / upsert
# ---------------------------------------------------------------------
def _overlay_upsert_by_source(base: pd.DataFrame,
                              src: pd.DataFrame,
                              src_cols: List[str],
                              start_ts: pd.Timestamp,
                              end_ts: pd.Timestamp,
                              force: bool) -> pd.DataFrame:
    """
    - Unions base ∪ src index (creates new rows if needed).
    - If force=True: clears ONLY these src_cols in [start..end].
    - base.update(src) writes only where src is non-NaN (won’t clobber with NaN).
    """
    if src is None or src.empty:
        return base

    # Ensure columns exist
    for c in src.columns:
        if c not in base.columns:
            base[c] = pd.NA

    # Union of dates (keep datetime dtype)
    base.index = pd.to_datetime(base.index, errors="coerce")
    src.index  = pd.to_datetime(src.index,  errors="coerce")
    base = base.reindex(base.index.union(src.index)).sort_index()

    # Selective clear if --force
    if force and src_cols:
        mask = (base.index >= start_ts) & (base.index <= end_ts)
        base.loc[mask, src_cols] = pd.NA

    # Overlay (does not overwrite with NaN)
    base.update(src, overwrite=True)
    return base


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _max_date_in_df(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    return None if d.empty else d.max().normalize()

def _min_date_in_df(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    return None if d.empty else d.min().normalize()

def _last_date_with_source_data(base_idxed: pd.DataFrame, source_cols: List[str]) -> Optional[pd.Timestamp]:
    """
    Returns the last date where ANY of the given columns has data in OUT.
    """
    if base_idxed.empty or not source_cols:
        return None
    cols = [c for c in source_cols if c in base_idxed.columns]
    if not cols:
        return None
    mask = base_idxed[cols].notna().any(axis=1)
    if not mask.any():
        return None
    # mask.index already aligns with base_idxed.index (dates)
    return pd.to_datetime(mask.index[mask]).max().normalize()

def _drop_colliding_columns(src_cols: Set[str], already_used: Set[str], source_name: str) -> List[str]:
    """
    If any column in src overlaps with previously assigned columns, drop them and warn.
    """
    collisions = sorted(src_cols & already_used)
    if collisions:
        logger.warning(f"{source_name}: dropping colliding columns -> {collisions}")
    return sorted(list(src_cols - already_used))

def apply_chl_quality_gate(chl_df: pd.DataFrame, coverage_col="coverage_percent", min_cov=10.0) -> pd.DataFrame:
    """
    If coverage < min_cov, set non-key chl columns to NA.
    Keys kept: 'date', 'coverage_percent', 'CI_index'.
    """
    if chl_df.empty or coverage_col not in chl_df.columns:
        return chl_df
    out = chl_df.copy()
    keep_cols = {"date", coverage_col, "CI_index"}
    target_cols = [c for c in out.columns if c not in keep_cols]
    bad = out[coverage_col].lt(min_cov)
    if bad.any() and target_cols:
        out.loc[bad, target_cols] = pd.NA
    return out

def _reorder_columns_on_create_no_prefix(df: pd.DataFrame,
                                         groups: dict) -> pd.DataFrame:
    """
    Standard initial order when creating OUT:
      date | (chl group) | (ncei group) | (tides group) | (goes group) | other
    `groups` is a dict {'chl': set([...]), 'ncei': set([...]), 'tides': set([...]), 'goes': set([...])}
    """
    if df.empty:
        return df

    cols = list(df.columns)
    if "date" in cols:
        cols.remove("date")

    group_names = ["chl", "ncei", "tides", "goes"]
    ordered = ["date"]
    seen = set(["date"])

    for g in group_names:
        gset = groups.get(g, set())
        gcols = [c for c in cols if c in gset and c not in seen]
        ordered += sorted(gcols)
        seen.update(gcols)

    others = [c for c in cols if c not in seen]
    ordered += sorted(others)
    return df.reindex(columns=ordered)


# ---------------------------------------------------------------------
# Core engine (NO PREFIX)
# ---------------------------------------------------------------------
def merge_gold(
    tides_path: Optional[str],
    ncei_path: Optional[str],
    goes_daily_path: Optional[str],
    chl_daily_path: Optional[str],
    final_csv: str,
    start: Optional[str],
    end: Optional[str],
    force: bool,
    write_state: bool = True,
    state_path: Optional[str] = None,
    chl_min_cov: float = 10.0,
    backfill: bool = True,   # True: overlay full source history each run (fills holes)
) -> str:

    # 1) Global window
    inputs = [p for p in [tides_path, ncei_path, goes_daily_path, chl_daily_path] if p]
    start_ts, end_ts = resolve_window(final_csv, start, end, inputs)
    os.makedirs(os.path.dirname(final_csv) or ".", exist_ok=True)

    # 2) Load OUT (base)
    created_new = not os.path.exists(final_csv)
    if os.path.exists(final_csv):
        base = pd.read_csv(final_csv)
        if not base.empty and "date" in base.columns:
            base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
            base = base.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        else:
            base = pd.DataFrame(columns=["date"])
    else:
        base = pd.DataFrame(columns=["date"])

    base = base.set_index("date").sort_index()
    base.index = pd.to_datetime(base.index, errors="coerce")

    # 3) Read sources (raw)
    tides = _read_daily_csv(tides_path, "TIDES")
    ncei  = _read_daily_csv(ncei_path,  "NCEI")
    goes  = _read_daily_csv(goes_daily_path, "GOES")
    goes  = _normalize_goes_columns(goes)
    chl   = _read_daily_csv(chl_daily_path,  "CHL")

    # 4) Per-source normalization / renames (NO prefixes)
    # NCEI renames
    ncei_ren = {
        "AWND": "wind_avg",        # average wind speed
        "PRCP": "precipitation",   # precipitation
        "TMAX": "temp_max",        # max temperature
        "TMIN": "temp_min",        # min temperature
        "WSF2": "wind_speed_2m",   # 2-min avg wind (or gust at 2m, depends on feed)
    }
    if "station" in ncei.columns:
        ncei = ncei.drop(columns=["station"])
    ncei = ncei.rename(columns=ncei_ren)

    # GOES rename
    goes_ren = {"Magnitude": "Watt_per_m2"}
    goes = goes.rename(columns=goes_ren)

    # CHL renames and quality gate
    chl_ren = {}
    if "coverage_pct" in chl.columns:
        chl_ren["coverage_pct"] = "coverage_percent"
    if "CIcyano" in chl.columns and "CI_index" not in chl.columns:
        chl_ren["CIcyano"] = "CI_index"
    chl = chl.rename(columns=chl_ren)
    if "coverage_percent" in chl.columns:
        chl = apply_chl_quality_gate(chl, coverage_col="coverage_percent", min_cov=chl_min_cov)

    # 5) Column sets per source (AFTER renames)
    tides_cols_all = set([c for c in tides.columns if c != "date"])
    ncei_cols_all  = set([c for c in ncei.columns  if c != "date"])
    goes_cols_all  = set([c for c in goes.columns  if c != "date"])
    chl_cols_all   = set([c for c in chl.columns   if c != "date"])

    # For first creation, remember groups to order columns nicely
    groups = {
        "tides": set(tides_cols_all),
        "ncei":  set(ncei_cols_all),
        "goes":  set(goes_cols_all),
        "chl":   set(chl_cols_all),
    }

    # Track columns already assigned (avoid collisions in this run)
    already_used = set([c for c in base.columns if c != "date"])

    # Helper to compute source-specific window:
    def _source_window(src_df: pd.DataFrame, src_name: str, src_cols: List[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if src_df.empty:
            return None, None
        input_min = _min_date_in_df(src_df)
        input_max = _max_date_in_df(src_df)
        if input_min is None or input_max is None:
            return None, None

        # If backfill=True, start from earliest available date in the source to fill holes
        if backfill:
            start_src = max(input_min, _to_date_only("1900-01-01"))  # clamp
        else:
            # incremental mode: continue from last written date + 1, or from input_min
            last_out = _last_date_with_source_data(base, src_cols)
            if last_out is not None:
                start_src = max(last_out + pd.Timedelta(days=1), input_min)
            else:
                start_src = input_min

        end_src = min(end_ts, input_max)
        if start_src > end_src:
            logger.info(f"{src_name}: nothing to do (start {start_src.date()} > end {end_src.date()}).")
            return None, None
        return start_src, end_src

    # 6) Prepare and overlay each source (NO prefixes)
    # ----- TIDES -----
    tides_cols = _drop_colliding_columns(tides_cols_all, already_used, "TIDES")
    start_tides, end_tides = _source_window(tides, "TIDES", tides_cols)
    if start_tides and end_tides:
        tides_src = _prepare_source_no_prefix(_subset_window(tides, start_tides, end_tides),
                                              start_tides, end_tides, renames=None)
        # keep only selected (non-colliding) columns
        if not tides_src.empty and tides_cols:
            tides_src = tides_src[[c for c in tides_cols if c in tides_src.columns]]
            base = _overlay_upsert_by_source(base, tides_src, list(tides_cols), start_tides, end_tides, force)
            already_used.update(tides_cols)

    # ----- NCEI -----
    ncei_cols = _drop_colliding_columns(ncei_cols_all, already_used, "NCEI")
    start_ncei, end_ncei = _source_window(ncei, "NCEI", ncei_cols)
    if start_ncei and end_ncei:
        ncei_src = _prepare_source_no_prefix(_subset_window(ncei, start_ncei, end_ncei),
                                             start_ncei, end_ncei, renames=None)
        if not ncei_src.empty and ncei_cols:
            ncei_src = ncei_src[[c for c in ncei_cols if c in ncei_src.columns]]
            base = _overlay_upsert_by_source(base, ncei_src, list(ncei_cols), start_ncei, end_ncei, force)
            already_used.update(ncei_cols)

    # ----- GOES -----
    goes_cols = _drop_colliding_columns(goes_cols_all, already_used, "GOES")
    start_goes, end_goes = _source_window(goes, "GOES", goes_cols)
    if start_goes and end_goes:
        goes_src = _prepare_source_no_prefix(_subset_window(goes, start_goes, end_goes),
                                             start_goes, end_goes, renames=None)
        if not goes_src.empty and goes_cols:
            goes_src = goes_src[[c for c in goes_cols if c in goes_src.columns]]
            base = _overlay_upsert_by_source(base, goes_src, list(goes_cols), start_goes, end_goes, force)
            already_used.update(goes_cols)

    # ----- CHL -----
    chl_cols = _drop_colliding_columns(chl_cols_all, already_used, "CHL")
    start_chl, end_chl = _source_window(chl, "CHL", chl_cols)
    if start_chl and end_chl:
        chl_src = _prepare_source_no_prefix(_subset_window(chl, start_chl, end_chl),
                                            start_chl, end_chl, renames=None)
        if not chl_src.empty and chl_cols:
            chl_src = chl_src[[c for c in chl_cols if c in chl_src.columns]]
            base = _overlay_upsert_by_source(base, chl_src, list(chl_cols), start_chl, end_chl, force)
            already_used.update(chl_cols)

    # 7) Save OUT
    out = base.sort_index().reset_index()
    # Initial nice ordering on first creation
    if created_new:
        out = _reorder_columns_on_create_no_prefix(out, groups)

    out.to_csv(final_csv, index=False)
    logger.info(f"Final updated -> {final_csv} (rows={len(out)})")

    # 8) Save state (last written date per source)
    if write_state:
        state_path = state_path or DEFAULT_STATE_PATH
        state = {
            "tides": str(_last_date_with_source_data(base, list(tides_cols_all)).date())
                     if _last_date_with_source_data(base, list(tides_cols_all)) else None,
            "ncei":  str(_last_date_with_source_data(base, list(ncei_cols_all)).date())
                     if _last_date_with_source_data(base, list(ncei_cols_all)) else None,
            "goes":  str(_last_date_with_source_data(base, list(goes_cols_all)).date())
                     if _last_date_with_source_data(base, list(goes_cols_all)) else None,
            "chl":   str(_last_date_with_source_data(base, list(chl_cols_all)).date())
                     if _last_date_with_source_data(base, list(chl_cols_all)) else None,
        }
        os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
        tmp = state_path + ".part"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, state_path)
        logger.info(f"State updated -> {state_path}: {state}")

    return final_csv


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GOLD unifier (SJL_daily_df) with auto window and per-source upsert (NO PREFIX).")
    ap.add_argument("--tides", default="src/pipeline/data/tide_data/tide_data.csv",
                    help="Daily tides CSV (e.g., src/pipeline/data/tide_data/tide_data.csv)")
    ap.add_argument("--ncei",  default="src/pipeline/data/ncei_data/ncei_data.csv",
                    help="Daily NCEI CSV (e.g., src/pipeline/data/ncei_data/ncei_data.csv)")
    ap.add_argument("--goes",  default="src/pipeline/data/goes_data/averaged_radiation.csv",
                    help="Daily GOES CSV (e.g., src/pipeline/data/goes_data/averaged_radiation.csv)")
    ap.add_argument("--chl",   default="src/pipeline/data/chl_total/chl_daily.csv",
                    help="Daily chlorophyll CSV (e.g., src/pipeline/data/chl_total/chl_daily.csv)")

    ap.add_argument("--out", default=DEFAULT_FINAL_CSV,
                    help="Path to final unified CSV (default: final_data/SJL_daily_df.csv)")
    ap.add_argument("--start", help="YYYY-MM-DD. If omitted and OUT exists -> max(OUT)+1; if OUT not exists -> min(date) across inputs.")
    ap.add_argument("--end",   help="YYYY-MM-DD. If omitted -> YESTERDAY.")
    ap.add_argument("--force", action="store_true",
                    help="Rewrite ONLY the affected source columns in [start..end] before overlaying (safe re-sync).")

    ap.add_argument("--chl-min-cov", type=float, default=10.0,
                    help="Minimum coverage_percent to keep CHL values (default 10%%).")
    ap.add_argument("--no-state", dest="write_state", action="store_false", default=True,
                    help="Do not write gold_state.json")
    ap.add_argument("--state-path", default=DEFAULT_STATE_PATH,
                    help="Path for gold_state.json")
    ap.add_argument("--no-backfill", dest="backfill", action="store_false", default=True,
                    help="Disable historical backfill (only continue after last written date).")

    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])

    args = ap.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    merge_gold(
        tides_path=norm(args.tides),
        ncei_path=norm(args.ncei),
        goes_daily_path=norm(args.goes),
        chl_daily_path=norm(args.chl),
        final_csv=norm(args.out),
        start=args.start,
        end=args.end,
        force=args.force,
        write_state=args.write_state,
        state_path=norm(args.state_path),
        chl_min_cov=args.chl_min_cov,
        backfill=args.backfill,
    )

if __name__ == "__main__":
    main()
