"""
NOAA Tides & Currents downloader / updater

Goals:
- Respect user-provided [--start .. --end].
- Skip already-complete product-months using a persisted month log; optionally also require on-disk verification.
- Download only missing product-months.
- Build daily averages per product and merge to a single final CSV (date + products).
- Fill missing product values with NaN.
- If --force is set, re-download all product-months in range and rewrite those dates in final CSV.

Outputs:
- Per-product intermediates: data/tide_data/<product>_<station>_full.csv
- Month log: logs/tides_processed_log.csv (status per (year, month, product, station), with logged_at)
- Final CSV (upserted): data/tide_data/tide_data.csv
"""

import os
import csv
import argparse
import logging
import calendar
import time
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from glob import glob

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil.rrule import rrule, MONTHLY

# ------------------------- CONFIG / CONSTANTS -------------------------

BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

DEFAULT_STATION = "9755371"  # NOAA CO-OPS: San Juan Harbor
DEFAULT_PRODUCTS = ["air_temperature", "water_temperature", "air_pressure", "water_level"]

DEFAULT_INTERVAL = "h"       # applies only where valid (e.g., water_level)
DEFAULT_UNITS = "metric"
DEFAULT_TZ = "gmt"

OUT_DIR = "data/tide_data"                      # per-product CSVs
FINAL_CSV = "data/tide_data/tide_data.csv"      # merged daily CSV (date + products)
LOG_DIR = "logs"
MONTH_LOG = "logs/tides_processed_log.csv"      # month-level status log

# Products that accept 'datum' and/or 'interval'
PRODUCTS_NEED_DATUM = {"water_level"}
PRODUCTS_ACCEPT_INTERVAL = {"water_level"}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tides")

# ------------------------- TIME/LOG HELPERS -------------------------
def yesterday_utc() -> date:
    return (datetime.now(timezone.utc).date() - timedelta(days=1))

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_log_has_logged_at_column(path: str) -> None:
    """
    If the processed-log CSV exists but lacks a 'logged_at' column,
    rewrite it in-place adding that column (empty for old rows).
    Safe for reasonably sized logs.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header = rows[0]
        if "logged_at" in header:
            return
        header2 = header + ["logged_at"]
        rows2 = [header2] + [r + [""] for r in rows[1:]]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows2)
        logger.info(f"Upgraded processed-log schema with 'logged_at': {path}")
    except Exception as e:
        logger.warning(f"Could not upgrade processed-log schema: {e}")

# ------------------------- HTTP SESSION -------------------------

def make_session(user_agent: str = "tides-pipeline/1.0") -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET", "HEAD"},
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()

# ------------------------- UTILITIES -------------------------

def clamp_month_span(y: int, m: int, start: date, end: date) -> Tuple[str, str]:
    first = date(y, m, 1)
    last = date(y, m, calendar.monthrange(y, m)[1])
    b = max(first, start)
    e = min(last, end)
    return b.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")

def load_month_log(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["year", "month", "product", "station", "status", "rows", "filename", "logged_at"])

def append_month_log(log_csv_path: str,
                     year: int,
                     month: int,
                     product: str,
                     station: str,
                     status: str,
                     rows: int,
                     filename: str = "",
                     logged_at: Optional[str] = None) -> None:
    """
    Append one processed row and record the UTC timestamp in 'logged_at'.
    Auto-upgrades existing logs to add the column if missing.
    """
    os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
    ensure_log_has_logged_at_column(log_csv_path)

    exists = os.path.exists(log_csv_path)
    with open(log_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["year", "month", "product", "station", "status", "rows", "filename", "logged_at"])
        w.writerow([year, month, product, station, status, rows, filename, logged_at or now_utc_iso()])

    logger.info(
        f"Processed-log updated -> {log_csv_path} | "
        f"{year}-{month:02d} {product} station={station} status={status} rows={rows}"
    )

def product_csv_path(output_dir: str, product: str, station_id: str) -> str:
    return os.path.join(output_dir, f"{product}_{station_id}_full.csv")

def product_file_has_rows_in_range(path: str, begin: str, end: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, usecols=["date"])
        if df.empty:
            return False
        d = pd.to_datetime(df["date"], errors="coerce")
        if d.isna().all():
            return False
        b = pd.to_datetime(begin)
        e = pd.to_datetime(end) + timedelta(days=1) - timedelta(seconds=1)
        return ((d >= b) & (d <= e)).any()
    except Exception:
        return False

def overwrite_month_slice(path: str, df_month: pd.DataFrame, begin: str, end: str) -> int:
    """
    Persist per-product file by replacing the [begin..end] slice with df_month (dedup on 'date').
    Returns number of rows written from df_month.
    """
    if os.path.exists(path):
        old = pd.read_csv(path)
        if not old.empty and "date" in old.columns:
            old["date"] = pd.to_datetime(old["date"], errors="coerce")
            mask = (old["date"] >= pd.to_datetime(begin)) & (old["date"] <= pd.to_datetime(end))
            old = old.loc[~mask]
        else:
            old = pd.DataFrame(columns=df_month.columns)
        new_all = pd.concat([old, df_month], ignore_index=True)
        new_all = new_all.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        new_all = df_month.sort_values("date")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    new_all.to_csv(path, index=False)
    return len(df_month)

# ------------------------- DOWNLOADER -------------------------

def download_tide_data(station_id: str,
                       begin_date: str,
                       end_date: str,
                       product: str,
                       interval: Optional[str] = DEFAULT_INTERVAL,
                       units: str = DEFAULT_UNITS,
                       time_zone: str = DEFAULT_TZ,
                       max_retries_local: int = 3,
                       throttle_s: float = 0.15) -> Optional[pd.DataFrame]:
    """
    Returns DataFrame ['date', product] with hourly rows, or None if no data.
    """
    params = {
        "station": station_id,
        "begin_date": begin_date,
        "end_date": end_date,
        "product": product,
        "units": units,
        "time_zone": time_zone,
        "format": "json"
    }
    if product in PRODUCTS_NEED_DATUM:
        params["datum"] = "MLLW"
    if interval and (product in PRODUCTS_ACCEPT_INTERVAL):
        params["interval"] = interval

    logger.info(f"[GET] {product} {begin_date}..{end_date} @ station {station_id}")

    for attempt in range(max_retries_local):
        try:
            r = SESSION.get(BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                logger.warning(f"API error {product} {begin_date}..{end_date}: {data['error'].get('message')}")
                return None

            key = "data" if "data" in data else "predictions" if "predictions" in data else None
            if key is None or not data.get(key):
                return None

            df = pd.DataFrame(data[key])
            if "t" in df.columns:
                df["date"] = pd.to_datetime(df["t"]); df.drop(columns=["t"], inplace=True)
            elif "time" in df.columns:
                df["date"] = pd.to_datetime(df["time"]); df.drop(columns=["time"], inplace=True)
            else:
                return None

            val_col = "v" if "v" in df.columns else "value" if "value" in df.columns else None
            if val_col is None:
                return None

            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            df = df.rename(columns={val_col: product})
            df = df[["date", product]].dropna(subset=[product])
            if df.empty:
                return None
            time.sleep(throttle_s)
            return df
        except Exception as e:
            if attempt < max_retries_local - 1:
                wait = 2 ** attempt
                logger.warning(f"Error {product} try {attempt+1}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Failed {product} after {max_retries_local} attempts. {e}")
                return None

# ------------------------- CLEANUP -------------------------

def cleanup_per_product_csvs(output_dir: str, station_id: str) -> int:
    """
    Delete '*_<station>_full.csv' files. Returns the count removed.
    """
    files = glob(os.path.join(output_dir, f"*_{station_id}_full.csv"))
    cnt = 0
    for f in files:
        try:
            os.remove(f)
            cnt += 1
        except Exception as e:
            logger.warning(f"Could not remove {f}: {e}")
    if cnt:
        logger.info(f"Removed {cnt} per-product files from {output_dir}.")
    return cnt

# ------------------------- MAIN WORKERS -------------------------

def run_months_download(start: str,
                        end: str,
                        station_id: str,
                        products: List[str],
                        output_dir: str,
                        month_log_path: str,
                        force: bool = False,
                        skip_on_log_ok: bool = True) -> None:
    """
    For each month in [start..end] and each product:
      - If not force:
          * If skip_on_log_ok=True: skip when month_log says status='ok' (ignores file presence).
          * If skip_on_log_ok=False: skip only when log=='ok' AND the per-product CSV has rows in that month.
        Months with status='no_data' are always retried.
      - Else (force): download anyway.
      - When downloaded, overwrite that month's slice in the per-product CSV (dedup by 'date').
      - Append to the month log (with logged_at).
    """
    os.makedirs(output_dir, exist_ok=True)
    log_df = load_month_log(month_log_path)

    # Build index of ok entries for quick skip
    ok_set = set()
    if not log_df.empty:
        for _, r in log_df.iterrows():
            try:
                if str(r.get("status")) == "ok":
                    ok_set.add((int(r["year"]), int(r["month"]), str(r["product"]), str(r["station"])))
            except Exception:
                continue

    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()

    for dt in rrule(MONTHLY, dtstart=start_d, until=end_d):
        y, m = dt.year, dt.month
        b, e = clamp_month_span(y, m, start_d, end_d)

        for product in products:
            p_path = product_csv_path(output_dir, product, station_id)

            if (not force) and ((y, m, product, station_id) in ok_set):
                if skip_on_log_ok:
                    logger.info(f"Skip {product} {y}-{m:02d}: log=ok (ignoring file presence).")
                    continue
                elif product_file_has_rows_in_range(p_path, b, e):
                    logger.info(f"Skip {product} {y}-{m:02d}: log=ok & file present.")
                    continue
                else:
                    logger.debug(f"Re-download {product} {y}-{m:02d}: log=ok but file missing/empty for {b}..{e}.")

            df = download_tide_data(station_id, b, e, product)
            if df is None or df.empty:
                append_month_log(
                    month_log_path, y, m, product, station_id,
                    status="no_data", rows=0,
                    filename=os.path.basename(p_path)
                )
                continue

            # Overwrite this month slice in per-product CSV (dedup by date)
            written = overwrite_month_slice(p_path, df, b, e)
            append_month_log(
                month_log_path, y, m, product, station_id,
                status="ok", rows=written,
                filename=os.path.basename(p_path)
            )

def build_daily_matrix(output_dir: str,
                       station_id: str,
                       products: List[str],
                       start: str,
                       end: str) -> pd.DataFrame:
    """
    Reads per-product CSVs, computes daily means, outer-joins on 'date', and
    returns a DataFrame with all dates present in [start..end] (missing -> NaN).
    """
    daily_frames = []
    for product in products:
        p_path = product_csv_path(output_dir, product, station_id)
        if not os.path.exists(p_path):
            continue
        df = pd.read_csv(p_path)
        if df.empty or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        grp = df.groupby(df["date"].dt.date)[product].mean().reset_index()
        grp.rename(columns={"date": "date"}, inplace=True)
        grp["date"] = pd.to_datetime(grp["date"])
        daily_frames.append(grp)

    if not daily_frames:
        return pd.DataFrame(columns=["date"] + products)

    # Outer join all products on date
    merged = daily_frames[0]
    for f in daily_frames[1:]:
        merged = pd.merge(merged, f, on="date", how="outer")

    # Reindex to full date span so missing days appear with NaN
    b = pd.to_datetime(start)
    e = pd.to_datetime(end)
    full_index = pd.DataFrame({"date": pd.date_range(b, e, freq="D")})
    merged = pd.merge(full_index, merged, on="date", how="left").sort_values("date")

    # Ensure all expected product columns exist
    for p in products:
        if p not in merged.columns:
            merged[p] = pd.NA

    return merged

def upsert_final_csv(final_csv: str,
                     new_daily: pd.DataFrame,
                     start: str,
                     end: str,
                     force: bool) -> str:
    """
    Upsert robusto para el CSV final:
      - Preserva TODAS las fechas fuera de new_daily.
      - Para fechas presentes en new_daily:
          * (modo normal) solo actualiza columnas con valores no-NaN; no borra datos existentes.
          * (force=True) primero limpia todas las columnas en el rango [start..end] a NaN y luego sobreescribe.
    """
    os.makedirs(os.path.dirname(final_csv) or ".", exist_ok=True)

    # --- Normaliza new_daily
    nd = new_daily.copy()
    if "date" not in nd.columns:
        raise ValueError("new_daily must have a 'date' column")
    nd["date"] = pd.to_datetime(nd["date"], errors="coerce").dt.date

    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    nd = nd[(nd["date"] >= s) & (nd["date"] <= e)]
    # Quita filas donde TODAS las columnas de datos están NaN (evita que 'borremos' sin querer)
    data_cols = [c for c in nd.columns if c != "date"]
    if data_cols:
        nd = nd.dropna(how="all", subset=data_cols)
    nd = nd.drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()

    # --- Carga base
    if os.path.exists(final_csv):
        base = pd.read_csv(final_csv)
        if "date" not in base.columns:
            base = pd.DataFrame()
        else:
            base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.date
            base = base.drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
    else:
        base = pd.DataFrame().set_index(pd.Index([], name="date"))

    # --- Unión de columnas para no perder nada
    all_cols = sorted(set(base.columns) | set(nd.columns))
    base = base.reindex(columns=all_cols)
    nd   = nd.reindex(columns=all_cols)

    # --- Índice unión (fechas de base y de nd)
    union_idx = base.index.union(nd.index)
    out = base.reindex(index=union_idx, columns=all_cols)

    if force:
        # Limpia TODO el rango [s..e] (todas las columnas a NaN)
        rng_mask = (out.index >= s) & (out.index <= e)
        if rng_mask.any():
            out.loc[rng_mask, all_cols] = pd.NA

    # Overlay por columnas: nd tiene prioridad solo donde NO es NaN
    # combine_first: toma nd cuando hay valor, usa base cuando nd es NaN
    for col in all_cols:
        if col in nd.columns:
            out[col] = nd[col].combine_first(out[col])

    out = out.sort_index()
    out.reset_index().to_csv(final_csv, index=False)
    logger.info(f"Final CSV updated: {final_csv} (rows={len(out)})")
    return final_csv


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="NOAA Tides downloader with month-log skipping and final CSV upsert.")
    ap.add_argument("--start", default="2016-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end",   default=None, help="End date (YYYY-MM-DD)")
    ap.add_argument("--station", default=DEFAULT_STATION, help="NOAA CO-OPS station (e.g., 9755371)")
    ap.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS, help="Products to fetch")
    ap.add_argument("--out", default=OUT_DIR, help="Output folder for per-product CSVs")
    ap.add_argument("--final", default=FINAL_CSV, help="Final merged daily CSV (date + products)")
    ap.add_argument("--month-log", default=MONTH_LOG, help="Path to month-level status log CSV")

    # Behavior toggles
    ap.add_argument("--force", action="store_true",
                    help="Re-download all months in range and rewrite those dates in the final CSV")
    ap.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,  # provides --cleanup / --no-cleanup
        default=True,
        help="Delete per-product CSVs after updating the final CSV (default: True)"
    )
    ap.add_argument(
        "--skip-on-log-ok",
        action=argparse.BooleanOptionalAction,  # provides --skip-on-log-ok / --no-skip-on-log-ok
        default=True,
        help="Skip months marked ok in the log even if per-product files were cleaned (default: True)"
    )

    args = ap.parse_args()
    if args.end is None:
        args.end = yesterday_utc().strftime("%Y-%m-%d")

    # 1) Download (skip product-months already OK, depending on --skip-on-log-ok; unless --force)
    run_months_download(
        start=args.start,
        end=args.end,
        station_id=args.station,
        products=args.products,
        output_dir=args.out,
        month_log_path=args.month_log,
        force=args.force,
        skip_on_log_ok=args.skip_on_log_ok
    )

    # 2) Build daily matrix (date x products) for the range; NaN for missing
    daily = build_daily_matrix(
        output_dir=args.out,
        station_id=args.station,
        products=args.products,
        start=args.start,
        end=args.end
    )

    if daily.empty:
        logger.warning("No daily data built for the requested range.")
        return

    # 3) Upsert to final CSV (rewrite range if --force)
    upsert_final_csv(
        final_csv=args.final,
        new_daily=daily,
        start=args.start,
        end=args.end,
        force=args.force
    )

    # 4) Optional cleanup (default: True; use --no-cleanup to skip)
    if args.cleanup:
        cleanup_per_product_csvs(args.out, args.station)

if __name__ == "__main__":
    main()
