from __future__ import annotations

import argparse
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import requests
import xarray as xr
from xarray.conventions import SerializationWarning


# Silence xarray QC decode warnings (non-fatal)
warnings.filterwarnings("ignore", category=SerializationWarning)


# -----------------------------
# THREDDS PR2 Meteorology (merged deployments)
# -----------------------------
DEFAULT_CATALOG_XML = "https://dm1.caricoos.org/thredds/catalog/UMAINE_buoys_newFormat/PR2/Meteorology/catalog.xml"
DEFAULT_CATALOG_HTML = "https://dm1.caricoos.org/thredds/catalog/UMAINE_buoys_newFormat/PR2/Meteorology/catalog.html"
DEFAULT_OPENDAP_BASE = "https://dm1.caricoos.org/thredds/dodsC/"

# Default behavior requested: 2016–2025 in one CSV
DEFAULT_FROM_YEAR = 2016
DEFAULT_TO_YEAR = 2025
DEFAULT_OUT = "src/pipeline/data/caricoos_pr2/PR2_met.csv"


@dataclass(frozen=True)
class DatasetRef:
    name: str
    url_path: str
    opendap_url: str


# -----------------------------
# Time helpers
# Strategy: keep EVERYTHING tz-naive but interpreted as UTC.
# -----------------------------
def _to_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse date/time as UTC, then drop tz to keep tz-naive timestamps."""
    if not s:
        return None
    return pd.to_datetime(s, utc=True).tz_convert(None)


def year_bounds_utc_naive(year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0, second=0)
    end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
    return start, end


def year_chunks(start: pd.Timestamp, end: pd.Timestamp, years_per_chunk: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    years_per_chunk = max(1, int(years_per_chunk))
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur <= end:
        nxt = cur + pd.DateOffset(years=years_per_chunk)
        chunk_end = min(end, nxt - pd.Timedelta(seconds=1))
        chunks.append((cur, chunk_end))
        cur = nxt
    return chunks


# -----------------------------
# Catalog discovery
# -----------------------------
def _http_get_text(url: str, timeout_s: int = 30) -> str:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def _parse_catalog_xml_for_met_files(xml_text: str, opendap_base: str) -> List[DatasetRef]:
    root = ET.fromstring(xml_text)

    m = re.match(r"\{(.*)\}", root.tag)
    ns_uri = m.group(1) if m else ""
    ns = {"t": ns_uri} if ns_uri else {}

    def iter_datasets():
        if ns:
            return root.findall(".//t:dataset", ns)
        return root.findall(".//dataset")

    out: List[DatasetRef] = []
    for ds in iter_datasets():
        url_path = ds.attrib.get("urlPath", "") or ""
        name = ds.attrib.get("name", "") or ""
        if not url_path.endswith(".met.merged.nc"):
            continue
        if "/PR2/Meteorology/" not in url_path.replace("\\", "/"):
            continue

        if not name:
            name = Path(url_path).name

        opendap_url = opendap_base.rstrip("/") + "/" + url_path.lstrip("/")
        out.append(DatasetRef(name=name, url_path=url_path, opendap_url=opendap_url))

    return out


def _fallback_parse_catalog_html_for_met_files(html_text: str, opendap_base: str) -> List[DatasetRef]:
    names = sorted(set(re.findall(r"(PR2\d{2}\.met\.merged\.nc)", html_text)))
    out: List[DatasetRef] = []
    for name in names:
        url_path = f"UMAINE_buoys_newFormat/PR2/Meteorology/{name}"
        opendap_url = opendap_base.rstrip("/") + "/" + url_path
        out.append(DatasetRef(name=name, url_path=url_path, opendap_url=opendap_url))
    return out


def list_pr2_met_datasets(
    catalog_xml_url: str = DEFAULT_CATALOG_XML,
    catalog_html_url: str = DEFAULT_CATALOG_HTML,
    opendap_base: str = DEFAULT_OPENDAP_BASE,
) -> List[DatasetRef]:
    try:
        xml_text = _http_get_text(catalog_xml_url)
        ds = _parse_catalog_xml_for_met_files(xml_text, opendap_base=opendap_base)
        if ds:
            return ds
    except Exception:
        pass

    html_text = _http_get_text(catalog_html_url)
    ds = _fallback_parse_catalog_html_for_met_files(html_text, opendap_base=opendap_base)
    if not ds:
        raise RuntimeError("Could not discover PR2 met merged datasets from THREDDS catalog.")
    return ds


# -----------------------------
# Variable selection
# -----------------------------
def parse_vars_keep(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [v.strip() for v in s.split(",") if v.strip()]


def default_vars(ds: xr.Dataset, include_qc: bool) -> List[str]:
    vars_ = list(ds.data_vars)
    if include_qc:
        return vars_

    # Common QC suffix patterns to exclude
    drop_patterns = (
        r"_qc_tests$",
        r"_qc_results$",
        r"_qc$",
        r"qc_tests$",
        r"qc_results$",
    )

    out = []
    for v in vars_:
        if any(re.search(p, v) for p in drop_patterns):
            continue
        out.append(v)
    return out


# -----------------------------
# Fast xarray Dataset -> DataFrame
# (avoids ds.to_dataframe() which can be extremely slow/large for remote datasets)
# -----------------------------
def ds_to_wide_df_fast(ds_sel: xr.Dataset, keep: List[str]) -> pd.DataFrame:
    """
    Efficiently convert a selected Dataset to a "wide" DataFrame:

    - Requires a 'time' coordinate.
    - Keeps variables that can be reduced to 1D over time.
    - Squeezes singleton dims (e.g., station=1).
    - Loads the selection once (important for OPeNDAP).
    """
    if "time" not in ds_sel.coords:
        raise ValueError("Dataset selection has no 'time' coordinate.")

    ds_sel = ds_sel.load()  # materialize remote selection once

    time_vals = pd.to_datetime(ds_sel["time"].values, errors="coerce")
    out = {"time": time_vals}

    # Add scalar coords as constant columns (optional but useful)
    for cname, c in ds_sel.coords.items():
        if cname == "time":
            continue
        if c.ndim == 0:
            out[cname] = [c.item()] * len(time_vals)

    for v in keep:
        if v not in ds_sel.data_vars:
            continue

        da = ds_sel[v]
        if "time" not in da.dims:
            continue

        da2 = da

        # Squeeze non-time singleton dims
        squeeze_dims = [d for d in da2.dims if d != "time" and da2.sizes.get(d, 1) == 1]
        if squeeze_dims:
            da2 = da2.squeeze(squeeze_dims, drop=True)

        # Keep only pure 1D time series
        if da2.ndim == 1 and da2.dims == ("time",):
            out[v] = da2.values

    return pd.DataFrame(out)


# -----------------------------
# Dataset coverage + CSV writing
# -----------------------------
def dataset_time_coverage(ds: xr.Dataset) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    t0 = _to_ts(ds.attrs.get("time_coverage_start"))
    t1 = _to_ts(ds.attrs.get("time_coverage_end"))

    # Fallback to time coordinate if needed
    if (t0 is None or t1 is None) and "time" in ds:
        try:
            tvals = ds["time"].values
            if len(tvals) > 0:
                if t0 is None:
                    t0 = pd.to_datetime(tvals[0], utc=True).tz_convert(None)
                if t1 is None:
                    t1 = pd.to_datetime(tvals[-1], utc=True).tz_convert(None)
        except Exception:
            pass

    return t0, t1


def append_df_to_csv(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    df.to_csv(out_csv, index=False, mode="a", header=write_header)


# -----------------------------
# Export driver
# -----------------------------
def export_pr2_met_to_csv(
    out_csv: str,
    year: Optional[int] = None,
    from_year: Optional[int] = DEFAULT_FROM_YEAR,
    to_year: Optional[int] = DEFAULT_TO_YEAR,
    start: Optional[str] = None,
    end: Optional[str] = None,
    chunk_years: int = 1,
    vars_keep: Optional[List[str]] = None,
    include_qc: bool = False,
    drop_all_nan_rows: bool = False,
    split_csv_by_chunk: bool = False,
) -> Path:
    out_base = Path(out_csv)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # Resolve requested window
    if year is not None:
        start_ts, end_ts = year_bounds_utc_naive(year)
    elif start is not None or end is not None:
        start_ts = _to_ts(start) if start else None
        end_ts = _to_ts(end) if end else None
    else:
        # default: 2016–2025
        if from_year is None or to_year is None:
            raise ValueError("from_year/to_year must be set when year/start/end are not provided.")
        start_ts, _ = year_bounds_utc_naive(from_year)
        _, end_ts = year_bounds_utc_naive(to_year)

    # Discover datasets
    datasets = list_pr2_met_datasets()

    # Read metadata for ordering + overlap checks
    meta: List[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], DatasetRef]] = []
    for d in datasets:
        try:
            with xr.open_dataset(d.opendap_url, decode_times=True) as ds:
                t0, t1 = dataset_time_coverage(ds)
                meta.append((t0, t1, d))
        except Exception as e:
            print(f"[WARN] Could not open metadata for {d.name}: {e}", file=sys.stderr)

    meta = [m for m in meta if m[0] is not None or m[1] is not None]
    if not meta:
        raise RuntimeError("Could not read any dataset metadata (time coverage).")

    known_starts = [m[0] for m in meta if m[0] is not None]
    known_ends = [m[1] for m in meta if m[1] is not None]

    overall_min = min(known_starts) if known_starts else None
    overall_max = max(known_ends) if known_ends else None

    if start_ts is None:
        if overall_min is None:
            raise RuntimeError("No start time available and no --start/--year provided.")
        start_ts = overall_min
    if end_ts is None:
        if overall_max is None:
            raise RuntimeError("No end time available and no --end/--year provided.")
        end_ts = overall_max

    if end_ts < start_ts:
        raise ValueError("end < start. Check your arguments.")

    # Sort datasets by coverage start (None-safe)
    meta.sort(key=lambda x: (pd.Timestamp.max if x[0] is None else x[0]))

    chunks = year_chunks(start_ts, end_ts, years_per_chunk=chunk_years)

    def chunk_out_path(cs: pd.Timestamp, ce: pd.Timestamp) -> Path:
        if not split_csv_by_chunk:
            return out_base
        tag = f"{cs.strftime('%Y%m%d')}_{ce.strftime('%Y%m%d')}"
        return out_base.with_name(out_base.stem + f"_{tag}" + out_base.suffix)

    # If writing a single CSV, recreate it from scratch each run
    if not split_csv_by_chunk and out_base.exists():
        out_base.unlink()

    # Track last written time to avoid duplicates across deployments
    last_time_written: Optional[pd.Timestamp] = None
    wrote_any = False

    for t0, t1, d in meta:
        # Fast skip by dataset coverage vs requested window
        if t0 is not None and t0 > end_ts:
            continue
        if t1 is not None and t1 < start_ts:
            continue

        print(f"[INFO] Opening {d.name}")

        with xr.open_dataset(d.opendap_url, decode_times=True) as ds:
            keep = vars_keep if vars_keep is not None else default_vars(ds, include_qc=include_qc)
            keep = [v for v in keep if v in ds.data_vars]
            if not keep:
                print(f"[WARN] No exportable variables for {d.name}. Skipping.")
                continue

            # Add deployment column so you can audit provenance if needed
            deployment_name = d.name

            for cs, ce in chunks:
                # Skip chunk if no overlap with dataset coverage
                if t0 is not None and ce < t0:
                    continue
                if t1 is not None and cs > t1:
                    continue

                # If we already wrote up to some time, do not re-export earlier timestamps
                effective_start = cs
                if last_time_written is not None:
                    effective_start = max(effective_start, last_time_written + pd.Timedelta(seconds=1))
                if effective_start > ce:
                    continue

                # Slice using datetime64 to avoid timezone issues inside xarray
                ds_sel = ds[keep].sel(time=slice(effective_start.to_datetime64(), ce.to_datetime64()))

                # Fast conversion
                df = ds_to_wide_df_fast(ds_sel, keep)

                if df.empty:
                    continue

                # Ensure time and ordering
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                df = df.dropna(subset=["time"]).sort_values("time")

                # Optional: drop rows where all exported variables are NaN
                if drop_all_nan_rows:
                    present_vars = [v for v in keep if v in df.columns]
                    if present_vars:
                        df = df.dropna(subset=present_vars, how="all")

                if df.empty:
                    continue

                # Attach deployment tag
                df.insert(1, "deployment", deployment_name)

                # Update monotonic last time
                last_time_written = df["time"].iloc[-1]

                outp = chunk_out_path(cs, ce)
                append_df_to_csv(df, outp)
                wrote_any = True
                print(f"[INFO]   Wrote {len(df):,} rows -> {outp}")

    if not wrote_any:
        raise RuntimeError("No rows written. Your window may be outside available coverage.")

    return out_base


# -----------------------------
# MAIN (default 2016–2025 single CSV)
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CariCOOS PR2 Meteorology -> CSV (OPeNDAP subset + chunked export).")

    # Defaults requested
    p.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path (default: PR2_met.csv).")
    p.add_argument("--years", nargs=2, type=int, default=[DEFAULT_FROM_YEAR, DEFAULT_TO_YEAR],
                   metavar=("FROM_YEAR", "TO_YEAR"),
                   help=f"Year range (default: {DEFAULT_FROM_YEAR} {DEFAULT_TO_YEAR}).")

    # Optional overrides
    p.add_argument("--year", type=int, default=None, help="Export a single calendar year (UTC).")
    p.add_argument("--range", nargs=2, metavar=("START", "END"), default=None,
                   help="Start/End date or datetime (interpreted as UTC). Example: 2016-01-01 2025-12-31")

    p.add_argument("--chunk-years", type=int, default=1, help="Process in N-year chunks (default 1).")

    # Variables / output controls
    p.add_argument("--vars", default=None,
                   help="Comma-separated variables to export. Default: all (excluding QC unless --include-qc).")
    p.add_argument("--include-qc", action="store_true", help="Include QC variables in export.")
    p.add_argument("--drop-all-nan-rows", action="store_true", help="Drop rows where ALL exported vars are NaN.")
    p.add_argument("--split-csv-by-chunk", action="store_true", help="Write one CSV per chunk (default: single CSV).")

    args = p.parse_args()

    # Default window from --years
    start = end = None
    from_year, to_year = args.years[0], args.years[1]
    year = None

    # Override precedence: --year > --range > --years(default)
    if args.year is not None:
        year = args.year
        from_year = to_year = None
    elif args.range is not None:
        start, end = args.range[0], args.range[1]
        from_year = to_year = None

    out = export_pr2_met_to_csv(
        out_csv=args.out,
        year=year,
        from_year=from_year,
        to_year=to_year,
        start=start,
        end=end,
        chunk_years=args.chunk_years,
        vars_keep=parse_vars_keep(args.vars),
        include_qc=args.include_qc,
        drop_all_nan_rows=args.drop_all_nan_rows,
        split_csv_by_chunk=args.split_csv_by_chunk,
    )

    print(f"[DONE] Export finished. Output: {out}")
