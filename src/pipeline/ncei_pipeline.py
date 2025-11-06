import os
import csv
import time
import json
import argparse
import logging
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional, Tuple

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ncei")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
DEFAULT_DATASET = "GHCND"
DEFAULT_STATION = "GHCND:RQW00011641"  # SJU TJSJ
DEFAULT_DTYPES = ["TMAX", "TMIN", "PRCP", "AWND", "WSF2"]

DEFAULT_OUTPUT_DIR = "data/ncei_data"
DEFAULT_FINAL_CSV = "data/ncei_data/ncei_data.csv"  # <- final “safe upsert” CSV

# Processed log
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_CSV = "ncei_processed_log.csv"
DEFAULT_RUNLOG_NAME = "ncei_downloader.log"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def add_file_logger(log_dir: str,
                    filename: str = DEFAULT_RUNLOG_NAME,
                    level: int = logging.INFO) -> str:
    os.makedirs(log_dir, exist_ok=True)
    p = os.path.join(log_dir, filename)
    fh = logging.FileHandler(p)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    return p

def parse_iso_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def yesterday_utc() -> date:
    return (datetime.now(timezone.utc).date() - timedelta(days=1))

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def env_or_file_token(env_name: str = "NCEI_TOKEN",
                      token_file: str = ".ncei_token") -> str:
    tok = os.getenv(env_name)
    if tok:
        return tok.strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, token_file)
    if not os.path.exists(path):
        raise RuntimeError(
            f"API token not found. Set {env_name} or create a '{token_file}' file next to this script."
        )
    with open(path, "r") as f:
        return f.read().strip()

# ---------------------------------------------------------------------
# HTTP Session
# ---------------------------------------------------------------------
def make_session(user_agent: str = "etl-ncei/1.0") -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"HEAD", "GET"},
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()

# ---------------------------------------------------------------------
# Processed log (idempotency)
# ---------------------------------------------------------------------
def ensure_log_has_logged_at_column(path: str) -> None:
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
            w = csv.writer(f); w.writerows(rows2)
        logger.info(f"Upgraded processed-log schema with 'logged_at': {path}")
    except Exception as e:
        logger.warning(f"Could not upgrade processed-log schema: {e}")

def load_processed_log(log_csv_path: str) -> pd.DataFrame:
    if os.path.exists(log_csv_path):
        return pd.read_csv(log_csv_path)
    return pd.DataFrame(columns=["chunk_start", "chunk_end", "station", "status", "rows", "filename", "logged_at"])

def save_processed_entry(log_csv_path: str,
                         chunk_start: str,
                         chunk_end: str,
                         station: str,
                         status: str,
                         rows: int,
                         filename: str = "",
                         logged_at: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
    ensure_log_has_logged_at_column(log_csv_path)
    exists = os.path.exists(log_csv_path)
    with open(log_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["chunk_start","chunk_end","station","status","rows","filename","logged_at"])
        w.writerow([chunk_start, chunk_end, station, status, rows, filename, logged_at or now_utc_iso()])

    logger.info(
        f"Processed-log updated -> {log_csv_path} | "
        f"{chunk_start}..{chunk_end} station={station} status={status} rows={rows}"
    )

# ---------------------------------------------------------------------
# Download one chunk (with pagination via offset)
# ---------------------------------------------------------------------
def fetch_ncei_chunk(api_token: str,
                     start_date: str,
                     end_date: str,
                     station_id: str,
                     datasetid: str,
                     datatypeids: List[str],
                     throttle_s: float = 0.3,
                     per_page_limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Download a date range from NCEI using pagination and return a pivoted DataFrame:
      columns = ['date'] + datatypes
      date is UTC date (daily)
    """
    headers = {"token": api_token}
    offset = 1
    all_rows = []

    while True:
        params = {
            "datasetid": datasetid,
            "datatypeid": datatypeids,
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "limit": per_page_limit,
            "units": "metric",
            "includemetadata": "false",
            "offset": offset,
        }
        try:
            resp = SESSION.get(BASE_URL, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for chunk {start_date}..{end_date}: {e}")
            raise
        except Exception as e:
            logger.error(f"Network error: {e}")
            raise

        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.error("Invalid JSON response from NCEI.")
            raise

        results = data.get("results", [])
        if not results:
            break

        all_rows.extend(results)
        offset += per_page_limit
        time.sleep(throttle_s)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    # Normalize
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df_p = (df.pivot_table(index="date", columns="datatype", values="value", aggfunc="mean")
              .reset_index()
              .sort_values("date"))
    return df_p

# ---------------------------------------------------------------------
# Safe overlay upsert (does NOT erase old data)
# ---------------------------------------------------------------------
def overlay_upsert_ncei(final_csv: str,
                        df: pd.DataFrame,
                        station_id: str,
                        start_date: str,
                        end_date: str,
                        force: bool) -> str:
    """
    Merge df into final_csv by (date, station):
      - Never replace existing values with NaN.
      - New values overwrite base values (cell-wise).
      - If force=True, clear [start..end] range for this station before overlay.
    df must have 'date' (datetime.date), 'station', and one or more data columns.
    """
    os.makedirs(os.path.dirname(final_csv) or ".", exist_ok=True)

    if "date" not in df.columns:
        raise ValueError("df must include 'date'")
    if "station" not in df.columns:
        df = df.copy()
        df.insert(1, "station", station_id)

    # Normalize incoming
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # Remove rows entirely NaN (except date/station)
    value_cols = [c for c in df.columns if c not in ("date", "station")]
    if value_cols:
        df = df.dropna(how="all", subset=value_cols)

    df = df.drop_duplicates(subset=["date", "station"], keep="last")
    df = df.set_index(["date", "station"]).sort_index()

    # Load base
    if os.path.exists(final_csv):
        base = pd.read_csv(final_csv)
        if "date" in base.columns:
            base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.date
        else:
            base = pd.DataFrame(columns=["date", "station"])
        if "station" not in base.columns:
            base.insert(1, "station", station_id)
        base = base.drop_duplicates(subset=["date", "station"], keep="last")
        base = base.set_index(["date", "station"]).sort_index()
    else:
        base = pd.DataFrame().set_index(pd.MultiIndex.from_tuples([], names=["date","station"]))

    # Union columns
    all_cols = sorted(set(base.columns) | set(df.columns))
    base = base.reindex(columns=all_cols)
    df   = df.reindex(columns=all_cols)

    # Optional force clear within [start..end] for this station
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    if force and not base.empty:
        # Mask rows for this station and date range
        idx = base.index
        mask = ((idx.get_level_values("station") == station_id) &
                (idx.get_level_values("date") >= s) &
                (idx.get_level_values("date") <= e))
        if mask.any():
            base.loc[mask, all_cols] = pd.NA

    # Union index, then overlay cell-wise: df wins where it has a value
    out_idx = base.index.union(df.index)
    out = base.reindex(index=out_idx, columns=all_cols)
    for col in all_cols:
        out[col] = df[col].combine_first(out[col])

    out = out.sort_index()
    out.reset_index().to_csv(final_csv, index=False)
    logger.info(f"Final CSV updated: {final_csv} (rows={len(out)})")
    return final_csv

# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def run_downloader(start_date: str,
                   end_date: str,
                   station_id: str,
                   datasetid: str,
                   datatypeids: List[str],
                   output_dir: str,
                   final_csv: str,
                   log_csv_path: str,
                   token_env: str,
                   token_file: str,
                   per_year: bool,
                   force: bool,
                   throttle_s: float,
                   allow_future: bool,
                   skip_on_log_ok: bool) -> Optional[str]:

    api_token = env_or_file_token(token_env, token_file)
    os.makedirs(output_dir, exist_ok=True)

    # Clamp end to yesterday UTC unless allowed
    end_req = parse_iso_date(end_date).date()
    if not allow_future:
        end_final = min(end_req, yesterday_utc())
    else:
        end_final = end_req
    end_date = end_final.strftime("%Y-%m-%d")

    # Determine chunks
    y0 = parse_iso_date(start_date).year
    y1 = parse_iso_date(end_date).year
    chunks: List[Tuple[str, str]] = []
    if per_year:
        for y in range(y0, y1 + 1):
            s = f"{y}-01-01" if y > y0 else start_date
            e = f"{y}-12-31" if y < y1 else end_date
            chunks.append((s, e))
    else:
        chunks.append((start_date, end_date))

    # Load processed log
    log_df = load_processed_log(log_csv_path)

    # Skip set: only 'ok' chunks
    ok_set = set()
    if not log_df.empty:
        for _, r in log_df.iterrows():
            try:
                if str(r.get("status")) == "ok":
                    ok_set.add((str(r["chunk_start"]), str(r["chunk_end"]), str(r["station"])))
            except Exception:
                continue

    # Process chunks
    any_written = False
    for i, (s, e) in enumerate(chunks, 1):
        logger.info(f"[{i}/{len(chunks)}] Downloading {s}..{e} for {station_id}")

        if (not force) and ((s, e, station_id) in ok_set) and skip_on_log_ok:
            logger.info(f"Skip chunk {s}..{e}: log=ok (ignoring file presence).")
            continue

        try:
            df = fetch_ncei_chunk(
                api_token=api_token,
                start_date=s,
                end_date=e,
                station_id=station_id,
                datasetid=datasetid,
                datatypeids=datatypeids,
                throttle_s=throttle_s,
            )
        except Exception as ex:
            logger.error(f"Failed to download {s}..{e}: {ex}")
            # do not log as processed; retry on next run
            continue

        if df is None or df.empty:
            save_processed_entry(log_csv_path, s, e, station_id, status="no_data", rows=0, filename="")
            # retry next time (not added to ok_set)
            continue

        # Prepare for overlay upsert (add station)
        df_out = df.copy()
        df_out.insert(1, "station", station_id)

        # Safe overlay upsert into final CSV
        overlay_upsert_ncei(final_csv, df_out, station_id, s, e, force=force)
        any_written = True

        # Mark as ok
        chunk_filename = f"ncei_{station_id.replace(':','_')}_{s}_to_{e}.csv"
        save_processed_entry(log_csv_path, s, e, station_id, status="ok", rows=len(df_out), filename=chunk_filename)

        time.sleep(throttle_s)

    return final_csv if any_written or os.path.exists(final_csv) else None

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download daily NCEI data (GHCND) with safe upsert and idempotent logging.")
    p.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default=None, help="End date YYYY-MM-DD")
    p.add_argument("--station", default=DEFAULT_STATION, help="NCEI station ID (e.g., GHCND:USW00022521)")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset ID (e.g., GHCND)")
    p.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES, help="Datatype IDs (e.g., TMAX TMIN PRCP AWND WSF2)")
    p.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="Output folder (unused for final, but kept for symmetry)")
    p.add_argument("--final", default=DEFAULT_FINAL_CSV, help="Path to the final daily CSV (safe upsert)")

    # Logs
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Folder to store processed log and runtime .log")
    p.add_argument("--log-csv", default=DEFAULT_LOG_CSV, help="Processed-chunks log CSV filename (inside log-dir)")
    p.add_argument("--file-log", action="store_true", help="Also write a runtime .log file in log-dir")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Console log level")

    # Behavior
    p.add_argument("--per-year", dest="per_year", action="store_true", default=True,
                   help="Process in yearly chunks (default ON)")
    p.add_argument("--one-chunk", dest="per_year", action="store_false",
                   help="Process the whole range as a single chunk")
    p.add_argument("--force", default=True,action="store_true",
                   help="Re-download chunks even if log says ok, and rewrite [start..end] in final CSV for this station")
    p.add_argument("--throttle", type=float, default=0.3, help="Pause between requests/chunks (seconds)")
    p.add_argument("--allow-future", action="store_true",
                   help="Allow end date beyond yesterday UTC (may produce 'no data')")
    p.add_argument("--skip-on-log-ok",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Skip chunks marked ok in the log even if files were cleaned (default: True)")

    args = p.parse_args()
    if args.end is None:
        args.end = yesterday_utc().strftime("%Y-%m-%d")

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # processed log path
    os.makedirs(args.log_dir, exist_ok=True)
    processed_log_csv_path = os.path.join(args.log_dir, args.log_csv)

    if args.file_log:
        path = add_file_logger(args.log_dir, level=getattr(logging, args.log_level))
        logger.info(f"Runtime log file: {path}")

    logger.info(f"Processed-log CSV: {processed_log_csv_path}")

    run_downloader(
        start_date=args.start,
        end_date=args.end,
        station_id=args.station,
        datasetid=args.dataset,
        datatypeids=args.dtypes,
        output_dir=args.out,
        final_csv=args.final,
        log_csv_path=processed_log_csv_path,
        token_env="NCEI_TOKEN",
        token_file=".ncei_token",
        per_year=args.per_year,
        force=args.force,
        throttle_s=args.throttle,
        allow_future=args.allow_future,
        skip_on_log_ok=args.skip_on_log_ok,
    )
