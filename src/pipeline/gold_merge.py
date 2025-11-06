# gold_merge.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
from datetime import date, timedelta
from typing import Optional, Tuple, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gold")

DEFAULT_FINAL_DIR  = "final_data"
DEFAULT_FINAL_CSV  = os.path.join(DEFAULT_FINAL_DIR, "SJL_daily_df.csv")
DEFAULT_STATE_PATH = os.path.join(DEFAULT_FINAL_DIR, "gold_state.json")

# ---------------------------------------------------------------------
# Utilidades de estado (opcional)
# ---------------------------------------------------------------------
def _load_state(path: str) -> dict:
    return json.load(open(path)) if os.path.exists(path) else {}

def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

# ---------------------------------------------------------------------
# Utilidades de fechas y ventanas
# ---------------------------------------------------------------------
def _to_date_only(dtlike) -> pd.Timestamp:
    ts = pd.to_datetime(dtlike, errors="coerce")
    if pd.isna(ts):
        return ts
    return ts.normalize()

def _yesterday() -> str:
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
    end: user_end o AYER.
    start:
      - si user_start -> start=user_start
      - si existe OUT -> start = max(OUT)+1 día
      - si no existe OUT -> start = min(fecha) entre inputs (o end si vacío)
    """
    end_ts = _parse_user_date(user_end) or pd.to_datetime(_yesterday()).normalize()

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
# Lectura y preprocesamiento de fuentes
# ---------------------------------------------------------------------
def _read_daily_csv(path: Optional[str], label: str) -> pd.DataFrame:
    """
    Lee un CSV con 'date', normaliza 'date', dedup por fecha (último), ordena.
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

def _prefix_cols(df: pd.DataFrame, prefix: str, keep: Optional[List[str]] = None, renames: Optional[dict] = None) -> pd.DataFrame:
    """
    Prefija columnas con `prefix` excepto las listadas en `keep`.
    Aplica `renames` primero si se provee.
    """
    if df.empty:
        return df
    keep = keep or ["date"]
    renames = renames or {}
    out = df.copy().rename(columns=renames)
    newcols = {}
    for c in out.columns:
        if c in keep:
            continue
        newcols[c] = f"{prefix}{c}"
    out = out.rename(columns=newcols)
    return out

def _prepare_source(df: pd.DataFrame,
                    label: str,
                    start_ts: pd.Timestamp,
                    end_ts: pd.Timestamp,
                    prefix: bool,
                    renames: Optional[dict] = None,
                    keep_unprefixed: Optional[List[str]] = None) -> pd.DataFrame:
    """
    - Recorta a [start..end], normaliza date y dedup por fecha (último).
    - Renombra columnas (opcional).
    - Prefija columnas (opcional) excepto 'keep_unprefixed'.
    - Devuelve DF indexado por 'date' (sin columna 'date').
    """
    if df is None or df.empty:
        return pd.DataFrame()

    keep_unprefixed = keep_unprefixed or ["date"]
    renames = renames or {}

    out = df.copy()
    if "date" not in out.columns:
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    mask = (out["date"] >= start_ts) & (out["date"] <= end_ts)
    out = out.loc[mask].copy()
    if out.empty:
        return pd.DataFrame()

    out = out.rename(columns=renames)

    if prefix:
        newcols = {}
        for c in out.columns:
            if c in keep_unprefixed:
                continue
            if label:
                newcols[c] = f"{label}{c}"
        out = out.rename(columns=newcols)

    out = out.set_index("date").sort_index()
    return out

# ---------------------------------------------------------------------
# Overlay/Upsert por fuente (columnar)
# ---------------------------------------------------------------------
def _overlay_upsert_by_source(base: pd.DataFrame,
                              src: pd.DataFrame,
                              cols_to_clear: List[str],
                              start_ts: pd.Timestamp,
                              end_ts: pd.Timestamp,
                              force: bool) -> pd.DataFrame:
    """
    - Unión de índices (fechas) base ∪ src (crea filas nuevas si hace falta).
    - Si force=True: limpia SOLO esas columnas en [start..end].
    - base.update(src) escribe solo donde src tiene NO-NaN (no pisa con NaN).
    """
    if src is None or src.empty:
        return base

    # Asegura columnas de la fuente en base
    for c in src.columns:
        if c not in base.columns:
            base[c] = pd.NA

    # Unión de fechas
    base = base.reindex(base.index.union(src.index)).sort_index()

    # Limpieza selectiva si force
    if force and cols_to_clear:
        mask = (base.index >= start_ts) & (base.index <= end_ts)
        base.loc[mask, cols_to_clear] = pd.NA

    # Overlay (no sobreescribe con NaN)
    base.update(src, overwrite=True)
    return base

# ---------------------------------------------------------------------
# Ayudas varias
# ---------------------------------------------------------------------
def _max_date_in_df(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    return None if d.empty else d.max().normalize()

def _last_date_with_prefix_data(base_idxed: pd.DataFrame, prefix: str) -> Optional[pd.Timestamp]:
    """
    base_idxed = OUT con index 'date'. Devuelve la última fecha donde
    cualquier columna que empiece por `prefix` tiene dato.
    """
    cols = [c for c in base_idxed.columns if c.startswith(prefix)]
    if not cols:
        return None
    mask = base_idxed[cols].notna().any(axis=1)
    if not mask.any():
        return None
    return pd.to_datetime(mask.index[mask]).max().normalize()

def apply_chl_quality_gate(chl_df: pd.DataFrame, coverage_col="chl_coverage_percent", min_cov=10.0) -> pd.DataFrame:
    """
    Si coverage < min_cov, anula (pone NA) las columnas 'chl_*' excepto coverage.
    """
    if chl_df.empty or coverage_col not in chl_df.columns:
        return chl_df
    out = chl_df.copy()
    chl_cols = [c for c in out.columns if c.startswith("chl_") and c != coverage_col]
    bad = out[coverage_col].lt(min_cov)
    out.loc[bad, chl_cols] = pd.NA
    return out

def _reorder_columns_on_create(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orden estándar al crear OUT por primera vez:
      date | chl_* | ncei_* | tide_* | goes_* | otras
    """
    if df.empty:
        return df
    cols = list(df.columns)
    cols.remove("date")
    groups = {"chl_": [], "ncei_": [], "tide_": [], "goes_": [], "other": []}
    for c in cols:
        if c.startswith("chl_"):
            groups["chl_"].append(c)
        elif c.startswith("ncei_"):
            groups["ncei_"].append(c)
        elif c.startswith("tide_"):
            groups["tide_"].append(c)
        elif c.startswith("goes_"):
            groups["goes_"].append(c)
        else:
            groups["other"].append(c)
    ordered = (["date"]
               + sorted(groups["chl_"])
               + sorted(groups["ncei_"])
               + sorted(groups["tide_"])
               + sorted(groups["goes_"])
               + sorted(groups["other"]))
    return df.reindex(columns=ordered)

# ---------------------------------------------------------------------
# Motor principal
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
    prefix: bool,
    write_state: bool = True,
    state_path: Optional[str] = None,
    chl_min_cov: float = 10.0,
) -> str:

    # 1) Ventana efectiva
    inputs = [p for p in [tides_path, ncei_path, goes_daily_path, chl_daily_path] if p]
    start_ts, end_ts = resolve_window(final_csv, start, end, inputs)
    os.makedirs(os.path.dirname(final_csv) or ".", exist_ok=True)

    # 2) Cargar base existente (si hay)
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

    # 3) Leer fuentes crudas
    tides = _read_daily_csv(tides_path, "TIDES")
    ncei  = _read_daily_csv(ncei_path,  "NCEI")
    goes  = _read_daily_csv(goes_daily_path, "GOES")
    chl   = _read_daily_csv(chl_daily_path,  "CHL")

    # 4) Normalización específica
    ncei_ren = {
        "AWND": "wind_avg",        # velocidad media del viento
        "PRCP": "precipitation",   # precipitación
        "TMAX": "temp_max",        # temp máx
        "TMIN": "temp_min",        # temp mín
        "WSF2": "wind_speed_2m",   # ráfaga/vel. 2-min
    }
    # Quita 'station' si existe y aplica renombres
    if "station" in ncei.columns:
        ncei = ncei.drop(columns=["station"])
    ncei = ncei.rename(columns=ncei_ren)

    # --- GOES: renombre de irradiancia ---
    goes_ren = {
        "irradiance_Wm2": "Watt_per_m2"
    }
    goes = goes.rename(columns=goes_ren)

    chl_ren = {}

    if "coverage_pct" in chl.columns:
        chl_ren["coverage_pct"] = "coverage_percent"
    if "CIcyano" in chl.columns and "CI_index" not in chl.columns:
        chl_ren["CIcyano"] = "CI_index"

    # 5) Preparar cada fuente (prefijos, renombres, ventanas por fuente)
    # ---------- TIDES ----------
    input_max_tide = _max_date_in_df(tides)
    last_out_tide  = _last_date_with_prefix_data(base, "tide_" if prefix else "")
    if (not force) and (input_max_tide is not None) and (last_out_tide is not None) and (input_max_tide <= last_out_tide):
        logger.info(f"Tides sin novedades (input_max={input_max_tide.date()} <= last_out={last_out_tide.date()}); skip.")
        tides_p, start_tide, end_tide = pd.DataFrame(), None, None
    else:
        start_tide = max(start_ts, (last_out_tide + pd.Timedelta(days=1)) if last_out_tide is not None else start_ts)
        end_tide   = min(end_ts, input_max_tide) if input_max_tide is not None else end_ts
        if end_tide >= start_tide:
            tides_p = _prepare_source(
                _subset_window(tides, start_tide, end_tide),
                "tide_" if prefix else "",
                start_tide, end_tide, prefix
            )
        else:
            tides_p = pd.DataFrame()

    # ---------- NCEI ----------
    input_max_ncei = _max_date_in_df(ncei)
    last_out_ncei  = _last_date_with_prefix_data(base, "ncei_" if prefix else "")
    if (not force) and (input_max_ncei is not None) and (last_out_ncei is not None) and (input_max_ncei <= last_out_ncei):
        logger.info(f"NCEI sin novedades (input_max={input_max_ncei.date()} <= last_out={last_out_ncei.date()}); skip.")
        ncei_p, start_ncei, end_ncei = pd.DataFrame(), None, None
    else:
        start_ncei = max(start_ts, (last_out_ncei + pd.Timedelta(days=1)) if last_out_ncei is not None else start_ts)
        end_ncei   = min(end_ts, input_max_ncei) if input_max_ncei is not None else end_ts
        if end_ncei >= start_ncei:
            ncei_p = _prepare_source(
                _subset_window(ncei, start_ncei, end_ncei),
                "ncei_" if prefix else "",
                start_ncei, end_ncei, prefix,
                renames=ncei_ren
            )
        else:
            ncei_p = pd.DataFrame()

    # ---------- GOES ----------
    input_max_goes = _max_date_in_df(goes)
    last_out_goes  = _last_date_with_prefix_data(base, "goes_" if prefix else "")
    if (not force) and (input_max_goes is not None) and (last_out_goes is not None) and (input_max_goes <= last_out_goes):
        logger.info(f"GOES sin novedades (input_max={input_max_goes.date()} <= last_out={last_out_goes.date()}); skip.")
        goes_p, start_goes, end_goes = pd.DataFrame(), None, None
    else:
        start_goes = max(start_ts, (last_out_goes + pd.Timedelta(days=1)) if last_out_goes is not None else start_ts)
        end_goes   = min(end_ts, input_max_goes) if input_max_goes is not None else end_ts
        if end_goes >= start_goes:
            goes_p = _prepare_source(
                _subset_window(goes, start_goes, end_goes),
                "goes_" if prefix else "",
                start_goes, end_goes, prefix,
                renames=goes_ren
            )
        else:
            goes_p = pd.DataFrame()

    # ---------- CHL ----------
    chl = chl.rename(columns=chl_ren)
    # Gate de calidad por cobertura
    if "coverage_percent" in chl.columns:
        bad = chl["coverage_percent"].lt(chl_min_cov)
        # Anular TODO lo que no sea fecha/cobertura/CI (por seguridad)
        keep_cols = {"date", "coverage_percent", "CI_index"}
        chl_cols_to_null = [c for c in chl.columns if c not in keep_cols]
        if bad.any() and chl_cols_to_null:
            chl.loc[bad, chl_cols_to_null] = pd.NA


    input_max_chl = _max_date_in_df(chl)
    last_out_chl  = _last_date_with_prefix_data(base, "chl_" if prefix else "")
    if (not force) and (input_max_chl is not None) and (last_out_chl is not None) and (input_max_chl <= last_out_chl):
        logger.info(f"CHL sin novedades (input_max={input_max_chl.date()} <= last_out={last_out_chl.date()}); skip.")
        chl_p, start_chl, end_chl = pd.DataFrame(), None, None
    else:
        start_chl = max(start_ts, (last_out_chl + pd.Timedelta(days=1)) if last_out_chl is not None else start_ts)
        end_chl   = min(end_ts, input_max_chl) if input_max_chl is not None else end_ts
        if end_chl >= start_chl:
            chl_p = _prepare_source(
                _subset_window(chl, start_chl, end_chl),
                "chl_" if prefix else "",
                start_chl, end_chl, prefix,
                renames=chl_ren,
                keep_unprefixed=(["date", "coverage_percent", "CI_index"] if prefix else ["date"])
            )

        else:
            chl_p = pd.DataFrame()

    # 6) Overlay por fuente (cada una solo sus columnas)
    if not tides_p.empty:
        base = _overlay_upsert_by_source(
            base, tides_p, list(tides_p.columns), start_tide, end_tide, force
        )
    if not ncei_p.empty:
        base = _overlay_upsert_by_source(
            base, ncei_p, list(ncei_p.columns), start_ncei, end_ncei, force
        )
    if not goes_p.empty:
        base = _overlay_upsert_by_source(
            base, goes_p, list(goes_p.columns), start_goes, end_goes, force
        )
    if not chl_p.empty:
        base = _overlay_upsert_by_source(
            base, chl_p, list(chl_p.columns), start_chl, end_chl, force
        )

    # 7) Guardar
    out = base.sort_index().reset_index()

    # Si estamos creando OUT por primera vez, imponer orden estándar
    if created_new:
        out = _reorder_columns_on_create(out)

    out.to_csv(final_csv, index=False)
    logger.info(f"Final updated -> {final_csv} (rows={len(out)})")

    # 8) (Opcional) guardar estado con última fecha por prefijo
    if write_state:
        state_path = state_path or DEFAULT_STATE_PATH
        state = {
            "tide_": str(_last_date_with_prefix_data(base, "tide_").date()) if _last_date_with_prefix_data(base, "tide_") else None,
            "ncei_": str(_last_date_with_prefix_data(base, "ncei_").date()) if _last_date_with_prefix_data(base, "ncei_") else None,
            "goes_": str(_last_date_with_prefix_data(base, "goes_").date()) if _last_date_with_prefix_data(base, "goes_") else None,
            "chl_":  str(_last_date_with_prefix_data(base, "chl_").date())  if _last_date_with_prefix_data(base, "chl_")  else None,
        }
        _save_state(state_path, state)
        logger.info(f"State updated -> {state_path}: {state}")

    return final_csv

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Unificador GOLD (SJL_daily_df) con ventana automática y upsert por fuente.")
    ap.add_argument("--tides", default="data/tide_data/tide_data.csv", help="CSV diario de Tides (p.ej., data/tide_data/tide_data.csv)")
    ap.add_argument("--ncei",  default="data/ncei_data/ncei_data.csv", help="CSV daily de NCEI (p.ej., data/ncei_data/ncei_data.csv)")
    ap.add_argument("--goes",  default="data/goes_data/INSOLRICO_daily_mean.csv", help="CSV diario de GOES (p.ej., data/goes_data/goes_daily_avg.csv)")
    ap.add_argument("--chl",   default="data/chl_total/chl_daily.csv", help="CSV diario de clorofila (p.ej., data/chl_total/chl_daily.csv)")

    ap.add_argument("--out", default=DEFAULT_FINAL_CSV, help="Ruta del totalizador final (default: final_data/SJL_daily_df.csv)")
    ap.add_argument("--start", help="YYYY-MM-DD. Si se omite y existe OUT -> max(OUT)+1; si OUT no existe -> min(fecha) de inputs.")
    ap.add_argument("--end",   help="YYYY-MM-DD. Si se omite, usa AYER.")
    ap.add_argument("--force", action="store_true", help="Re-escribe el rango [start..end] SOLO para las columnas de cada fuente con datos nuevos")
    ap.add_argument("--no-prefix", dest="prefix", action="store_false", default=False, help="No prefijar columnas por dataset")
    ap.add_argument("--chl-min-cov", type=float, default=10.0, help="Cobertura mínima para aceptar CHL (default 10%)")
    ap.add_argument("--no-state", dest="write_state", action="store_false", default=True, help="No escribir gold_state.json")
    ap.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="Ruta para gold_state.json")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])

    args = ap.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    merge_gold(
        tides_path=args.tides,
        ncei_path=args.ncei,
        goes_daily_path=args.goes,
        chl_daily_path=args.chl,
        final_csv=args.out,
        start=args.start,
        end=args.end,
        force=args.force,
        prefix=args.prefix,
        write_state=args.write_state,
        state_path=args.state_path,
        chl_min_cov=args.chl_min_cov,
    )

if __name__ == "__main__":
    main()