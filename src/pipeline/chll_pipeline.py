#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from shapely import geometry

import eumdac

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def add_file_logger(log_dir: str, filename: str = "chlorophyll_pipeline.log", level: int = logging.INFO) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    return path

# ---------------------------------------------------------------------
# Defaults / Constants
# ---------------------------------------------------------------------
DEFAULT_OUT_DIR = "src/pipeline/data/chl_data"
DEFAULT_LOG_DIR = "src/pipeline/logs"
DEFAULT_LOG_CSV = "chl_processed_log.csv"
DEFAULT_POINTS_CSV = "src/pipeline/data/chl_data/chl_points.csv"
DEFAULT_DAILY_CSV = "src/pipeline/data/chl_data/chl_daily.csv"

DEFAULT_PRODUCT_PATTERNS = [
    "chl_nn", "oa08_reflectance", "oa11_reflectance", "oa17_reflectance",
    "iwv", "oa04_reflectance", "oa21_reflectance", "oa07_reflectance",
    "chl_oc4me", "t865", "a865", "tsm_nn", "oa12_reflectance", "oa03_reflectance",
    "oa16_reflectance", "oa01_reflectance", "oa06_reflectance", "par",
    "adg443_nn", "kd490_m07", "oa02_reflectance", "oa09_reflectance",
    "oa05_reflectance", "oa10_reflectance", "oa18_reflectance",
    # required by the pipeline
    "geo_coordinates", "wqsf",
]

# Never download entries that contain these tokens (lowercase, substring match)
EXCLUDED_PATTERNS = {"tie_geo_coordinates"}

MAX_RETRIES_DOWNLOAD = 3
RETRY_DELAY_START = 5  # seconds
RETRY_DELAY_MAX = 60   # seconds

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def yesterday_utc() -> date:
    return (datetime.now(timezone.utc).date() - timedelta(days=1))

def now_utc_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def km_to_degrees(km: float) -> float:
    return km / 111.32  # ~111.32 km per degree

def ensure_dir(path: str) -> None:
    os.makedirs(path or ".", exist_ok=True)

def normalize_tokens(csv_like: str) -> List[str]:
    if not csv_like:
        return []
    return [t.strip().lower() for t in csv_like.split(",") if t.strip()]

# ---------------------------------------------------------------------
# Processed log (idempotency)
# ---------------------------------------------------------------------
def _upgrade_log_schema_with_logged_at(path: str) -> None:
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

def load_processed_log(log_csv_path: str) -> pd.DataFrame:
    if os.path.exists(log_csv_path):
        _upgrade_log_schema_with_logged_at(log_csv_path)
        return pd.read_csv(log_csv_path)
    ensure_dir(os.path.dirname(log_csv_path))
    return pd.DataFrame(columns=["product_id", "entry_name", "collection_id", "status", "rows", "csv_file", "logged_at"])

def save_processed_entry(log_csv_path: str,
                         product_id: str,
                         entry_name: str,
                         collection_id: str,
                         status: str,
                         rows: int,
                         csv_file: str = "") -> None:
    ensure_dir(os.path.dirname(log_csv_path))
    _upgrade_log_schema_with_logged_at(log_csv_path)
    exists = os.path.exists(log_csv_path)
    with open(log_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["product_id", "entry_name", "collection_id", "status", "rows", "csv_file", "logged_at"])
        w.writerow([product_id, entry_name, collection_id, status, rows, csv_file, now_utc_iso()])
    logger.info(
        f"Processed-log updated -> {log_csv_path} | entry={entry_name} status={status} rows={rows} file={csv_file}"
    )

# ---------------------------------------------------------------------
# Timeliness helpers — (prefer NT > STC > NR)
# ---------------------------------------------------------------------
def extract_entry_name_from_pid(pid: str) -> Optional[str]:
    parts = str(pid).split("_")
    return parts[7] if len(parts) > 7 else None

def timeliness_rank(pid: str) -> int:
    p = pid.upper()
    if "_O_NT_" in p:
        return 3
    if "_O_STC_" in p or "_O_ST_" in p:
        return 2
    if "_O_NR_" in p or "_O_NRT_" in p:
        return 1
    return 0

def pick_best_timeliness_per_entry(products) -> list:
    best = {}  # entry_name -> (rank, product)
    for prod in products:
        pid = getattr(prod, "_id", str(prod))
        entry = extract_entry_name_from_pid(pid)
        if not entry:
            continue
        r = timeliness_rank(pid)
        prev = best.get(entry)
        if (prev is None) or (r > prev[0]):
            best[entry] = (r, prod)
    return [bp[1] for entry, bp in sorted(best.items(), key=lambda kv: kv[0])]

# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def build_roi_polygon(lon_center: float, lat_center: float, half_size_deg: float) -> Tuple[List[Tuple[float, float]], str]:
    roi_coords = [
        (lon_center + half_size_deg, lat_center + half_size_deg),
        (lon_center - half_size_deg, lat_center + half_size_deg),
        (lon_center - half_size_deg, lat_center - half_size_deg),
        (lon_center + half_size_deg, lat_center - half_size_deg),
        (lon_center + half_size_deg, lat_center + half_size_deg),
    ]
    roi_wkt = "POLYGON((" + ", ".join([f"{lon} {lat}" for lon, lat in roi_coords]) + "))"
    return roi_coords, roi_wkt

def shutil_copyfileobj(src, dst, length: int = 1024 * 1024) -> None:
    while True:
        buf = src.read(length)
        if not buf:
            break
        dst.write(buf)

def safe_download_entry(product, entry: str, dest_dir: str) -> Optional[str]:
    filename = os.path.basename(entry)
    final_path = os.path.join(dest_dir, filename)
    tmp_path = final_path + ".part"
    if os.path.exists(final_path):
        return final_path
    delay = RETRY_DELAY_START
    for attempt in range(1, MAX_RETRIES_DOWNLOAD + 1):
        try:
            with product.open(entry=entry) as fsrc, open(tmp_path, "wb") as fdst:
                logger.info(f"Downloading {filename} (attempt {attempt})")
                shutil_copyfileobj(fsrc, fdst)
            os.replace(tmp_path, final_path)
            return final_path
        except Exception as e:
            logger.warning(f"Error downloading {filename}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt < MAX_RETRIES_DOWNLOAD:
                time.sleep(min(delay, RETRY_DELAY_MAX))
                delay *= 2
            else:
                logger.error(f"Persistent error downloading {filename}. Giving up.")
                return None

def find_any(file_map: dict, *candidates: str) -> Optional[str]:
    if not file_map:
        return None
    for c in candidates:
        if c in file_map:
            return file_map[c]
    lower_map = {k.lower(): v for k, v in file_map.items()}
    for c in candidates:
        v = lower_map.get(c.lower())
        if v:
            return v
    return None

# ---------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------
def process_chlorophyll_data(datastore,
                             lon: float,
                             lat: float,
                             radius_km: float,
                             start_date: str,
                             end_date: str,
                             collection_ids: List[str],
                             output_dir: str,
                             selected_products: List[str],
                             log_csv_path: str,
                            #  points_csv_path: str,
                            #  daily_csv_path: str,
                            #  daily_stats: List[str],
                            #  cleanup: bool,
                             force: bool = False,
                             throttle_s: float = 0.2) -> None:
    ensure_dir(output_dir)

    # Time log (per scene)
    time_log_file = os.path.join(output_dir, "time_spent.csv")
    ensure_dir(os.path.dirname(time_log_file))
    with open(time_log_file, "w", newline="") as tf:
        csv.writer(tf).writerow(["entry_name", "seconds"])

    # Processed product ids (idempotency)
    log_df = load_processed_log(log_csv_path)
    processed_ids = set(log_df["product_id"]) if not log_df.empty else set()

    # Wanted tokens
    extra_patterns = [p.lower() for p in (selected_products or [])]
    selected_patterns = list(dict.fromkeys(DEFAULT_PRODUCT_PATTERNS + extra_patterns))
    selected_patterns = [p for p in selected_patterns if p not in EXCLUDED_PATTERNS]

    # ROI
    half_size_deg = km_to_degrees(radius_km)
    roi_coords, roi_wkt = build_roi_polygon(lon, lat, half_size_deg)
    polygon = geometry.Polygon(roi_coords)  # retained for clarity, but we’ll use a fast bbox mask
    min_lon, max_lon = min(x for x, _ in roi_coords), max(x for x, _ in roi_coords)
    min_lat, max_lat = min(y for _, y in roi_coords), max(y for _, y in roi_coords)

    total_time = 0.0

    for collection_id in collection_ids:
        coll = datastore.get_collection(collection_id)
        logger.info(f"Searching products in {collection_id} for {start_date}..{end_date}")
        try:
            products = coll.search(geo=roi_wkt, dtstart=start_date, dtend=end_date)
        except Exception as e:
            logger.error(f"Error searching products in {collection_id}: {e}")
            continue

        products = pick_best_timeliness_per_entry(products)

        for product in products:
            product_id = getattr(product, "_id", None) or str(product)
            if (not force) and (product_id in processed_ids):
                logger.debug(f"Already processed: {product_id}. Skipping.")
                continue

            entry_name = extract_entry_name_from_pid(product_id) or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            logger.info(f"Processing product {product_id} (entry_name={entry_name})")
            start_t = time.time()
            downloaded_files: List[str] = []

            # 1) Download required entries
            try:
                for entry in product.entries:
                    entry_base_l = os.path.basename(entry).lower()
                    if any(ex in entry_base_l for ex in EXCLUDED_PATTERNS):
                        continue
                    if not any(pat in entry_base_l for pat in selected_patterns):
                        continue
                    path = safe_download_entry(product, entry, output_dir)
                    if path:
                        downloaded_files.append(path)
                    if throttle_s > 0:
                        time.sleep(throttle_s)
            except Exception as e:
                logger.error(f"Error while downloading entries for {product_id}: {e}")
                _cleanup_files(downloaded_files)
                save_processed_entry(log_csv_path, product_id, entry_name, collection_id, "download_error", 0, "")
                continue

            # 2) Map essential files
            file_map = {os.path.basename(f): f for f in downloaded_files}
            geo_path  = find_any(file_map, "geo_coordinates.nc")
            flag_path = find_any(file_map, "wqsf.nc")
            chl_nn_path    = find_any(file_map, "chl_nn.nc")
            chl_oc4me_path = find_any(file_map, "chl_oc4me.nc")
            chl_paths = [p for p in (chl_nn_path, chl_oc4me_path) if p]

            if (geo_path is None) or (not chl_paths):
                logger.warning("Missing required files (geo_coordinates or chlorophyll). Skipping scene.")
                _cleanup_files(downloaded_files)
                save_processed_entry(log_csv_path, product_id, entry_name, collection_id, "missing_files", 0, "")
                continue

            # 3) Process scene
            scene_csv = os.path.join(output_dir, f"{entry_name}.csv")
            rows_written = 0
            try:
                # Geolocation
                with xr.open_dataset(geo_path) as geo_ds:
                    lat_arr = geo_ds["latitude"].data
                    lon_arr = geo_ds["longitude"].data

                # Fast vectorized ROI (axis-aligned square)
                mask = (lon_arr >= min_lon) & (lon_arr <= max_lon) & (lat_arr >= min_lat) & (lat_arr <= max_lat)

                # Scene timestamp
                try:
                    ts = datetime.strptime(entry_name, "%Y%m%dT%H%M%S")
                except Exception:
                    ts = datetime.utcnow()

                df = pd.DataFrame({
                    "latitude":  lat_arr[mask],
                    "longitude": lon_arr[mask],
                    "datetime":  ts
                })

                # Optional flags
                if flag_path and os.path.exists(flag_path):
                    try:
                        with xr.open_dataset(flag_path) as flag_ds:
                            wqsf = flag_ds["WQSF"].data
                        if wqsf.shape == lat_arr.shape:
                            wqsf_masked = wqsf[mask]
                            df["INVALID"] = (wqsf_masked & (1 << 0)) > 0
                            df["WATER"]   = (wqsf_masked & (1 << 1)) > 0
                            df["CLOUD"]   = (wqsf_masked & (1 << 2)) > 0
                            df["LAND"]    = (wqsf_masked & (1 << 3)) > 0
                    except Exception as e:
                        logger.warning(f"WQSF read error for {entry_name}: {e}")

                # Track var names
                var_list_file = os.path.join(output_dir, "var_names.txt")
                seen_vars = set()
                if os.path.exists(var_list_file):
                    with open(var_list_file, "r") as f:
                        seen_vars = {line.strip() for line in f if line.strip()}

                # Add variables that match geo grid
                for fpath in downloaded_files:
                    base = os.path.basename(fpath).lower()
                    if fpath.endswith(".nc") and base not in {"geo_coordinates.nc", "wqsf.nc"}:
                        try:
                            with xr.open_dataset(fpath) as ds:
                                for var_name, da in ds.data_vars.items():
                                    v = da.data
                                    if hasattr(v, "shape") and v.shape == lat_arr.shape:
                                        df[var_name] = v[mask]
                                        seen_vars.add(var_name)
                        except Exception as e:
                            logger.warning(f"Error reading {fpath}: {e}")

                with open(var_list_file, "w") as f:
                    for v in sorted(seen_vars):
                        f.write(v + "\n")

                # Save per-scene CSV
                df.to_csv(scene_csv, index=False)
                rows_written = len(df)
                logger.info(f"Wrote scene CSV: {scene_csv} ({rows_written} rows)")

                # Also append to combined points CSV (append-only)
                # appended = append_points(df, points_csv_path)
                # logger.info(f"Appended {appended} rows to {points_csv_path}")

                save_processed_entry(
                    log_csv_path, product_id, entry_name, collection_id, "ok", rows_written, os.path.basename(scene_csv)
                )

            except Exception as e:
                logger.error(f"Processing error for {entry_name}: {e}")
                save_processed_entry(log_csv_path, product_id, entry_name, collection_id, "process_error", 0, "")
            finally:
                _cleanup_files(downloaded_files)

            # Timing
            spent = time.time() - start_t
            total_time += spent
            _append_time(time_log_file, entry_name, spent)
            logger.info(f"Processing time for {entry_name}: {spent:.2f}s")


def _cleanup_files(paths: List[str]) -> None:
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def _append_time(path: str, entry_name: str, seconds: float) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([entry_name, f"{seconds:.2f}"])

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chlorophyll downloader/processor (idempotent, NT>STC>NR) with combined & daily outputs.")
    parser.add_argument("--start", default="2016-11-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD")

    parser.add_argument("--lon", type=float, default=-66.025, help="ROI center longitude")
    parser.add_argument("--lat", type=float, default=18.425,   help="ROI center latitude")
    parser.add_argument("--radius-km", type=float, default=5.0,  help="ROI half-size in kilometers (square ROI)")

    parser.add_argument("--collections", nargs="+", default=["EO:EUM:DAT:0407", "EO:EUM:DAT:0556"],
                        help="EUMETSAT collection IDs to search")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory for CSVs and helper files")

    parser.add_argument("--products", type=str, default="",
                        help="Extra comma-separated entry tokens to include (added to defaults). Case-insensitive.")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory for logs")
    parser.add_argument("--log-csv", default=DEFAULT_LOG_CSV, help="Processed-log CSV filename (inside --log-dir)")
    parser.add_argument("--file-log", action="store_true", help="Also write a runtime .log file in --log-dir")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Console log level")

    parser.add_argument("--force", action="store_true", help="Reprocess even if product_id is already in the processed log")
    parser.add_argument("--throttle", type=float, default=0.2, help="Pause between entry downloads (seconds)")

    args = parser.parse_args()
    if args.end is None:
        args.end = yesterday_utc().strftime("%Y-%m-%d")
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Optional runtime file logger
    ensure_dir(args.log_dir)
    log_csv_path = os.path.join(args.log_dir, args.log_csv)
    if args.file_log:
        fpath = add_file_logger(args.log_dir, level=getattr(logging, args.log_level))
        logger.info(f"Runtime log file: {fpath}")

    # EUMDAC credentials (~/.eumdac/credentials: 'user,pass')
    cred_file = Path.home() / ".eumdac" / "credentials"
    try:
        user, pwd = cred_file.read_text().split(",")
        token = eumdac.AccessToken((user.strip(), pwd.strip()))
        logger.info(f"Token obtained. Expires: {token.expiration}")
    except Exception as e:
        logger.error(f"Error loading EUMDAC credentials: {e}")
        return

    datastore = eumdac.DataStore(token)

    extra = normalize_tokens(args.products)

    process_chlorophyll_data(
        datastore=datastore,
        lon=args.lon,
        lat=args.lat,
        radius_km=args.radius_km,
        start_date=args.start,
        end_date=args.end,
        collection_ids=args.collections,
        output_dir=args.out,
        selected_products=extra,
        log_csv_path=log_csv_path,
        force=args.force,
        throttle_s=args.throttle
    )

if __name__ == "__main__":
    main()