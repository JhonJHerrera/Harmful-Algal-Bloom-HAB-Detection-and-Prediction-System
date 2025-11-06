#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
from typing import Dict, Tuple, Optional
import csv
import re
from pathlib import Path
from datetime import datetime, date as Date
import numpy as np
import pandas as pd


from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.vectorized import contains
import xml.etree.ElementTree as ET



logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chl-daily-coverage")

# ---------------------------------------------------------------------
# Defaults: your three reference points (lat, lon)
# ---------------------------------------------------------------------
# I/O
DEFAULT_IN_DIR  = "data/chl_data"
DEFAULT_POLYGON = "polygon/LSJmasking.kml"

DEFAULT_OUT_DIR = "data/chl_total"
DEFAULT_OUT_FILE = os.path.join(DEFAULT_OUT_DIR, "chl_daily.csv")

DEFAULT_LOGS_DIR = os.path.join("logs")
FILES_LOG = os.path.join(DEFAULT_LOGS_DIR, "coverage_files_log.csv")
DAILY_LOG = os.path.join(DEFAULT_LOGS_DIR, "coverage_daily_log.csv")



DEFAULT_COORDS_R: Dict[str, Tuple[float, float]] = {
    "CHL_NN_R1": (18.44050857142857, -66.03748285714286),
    "CHL_NN_R2": (18.42804866666667, -66.02746733333333),
    "CHL_NN_R3": (18.42005797101449, -66.01991884057972),
}

DEFAULT_RADIUS_M = 500.0
DEFAULT_THRESHOLD = 1.0  # percent
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols_lower.get(cand.lower())
        if c:
            return c
    return None

_TS_RE = re.compile(r"(\d{8}T\d{6})")  # p.ej., 20210805T152233

def parse_dt_from_filename(path: Path) -> Optional[pd.Timestamp]:
    m = _TS_RE.search(path.stem)
    if not m:
        return None
    # Devuelve pandas Timestamp (naive)
    return pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S", errors="coerce")


def load_points(points_csv: str, parse_dates=True) -> pd.DataFrame:
    if not os.path.exists(points_csv):
        raise FileNotFoundError(points_csv)
    df = pd.read_csv(points_csv)

    # Normalize required columns (case-insensitive)
    lat_col  = find_col(df, ["latitude", "lat"])
    lon_col  = find_col(df, ["longitude", "lon", "lng"])
    dt_col   = find_col(df, ["datetime", "time", "timestamp"])
    chl_col  = find_col(df, ["CHL_NN", "chl_nn"])
    r665_col = find_col(df, ["Oa08_reflectance", "oa08_reflectance"])
    r681_col = find_col(df, ["Oa10_reflectance", "oa10_reflectance"])
    r709_col = find_col(df, ["Oa11_reflectance", "oa11_reflectance"])

    needed = [lat_col, lon_col, dt_col, chl_col, r665_col, r681_col, r709_col]
    if any(c is None for c in needed):
        missing = ["latitude","longitude","datetime","CHL_NN/chl_nn","Oa08_reflectance","Oa10_reflectance","Oa11_reflectance"]
        raise ValueError(f"Missing expected columns (case-insensitive). Needed ~ {missing}. Found: {list(df.columns)}")

    if parse_dates:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col])

    # Rename to canonical internal names
    df = df.rename(columns={
        lat_col: "latitude",
        lon_col: "longitude",
        dt_col: "datetime",
        chl_col: "CHL_NN",
        r665_col: "Oa08_reflectance",
        r681_col: "Oa10_reflectance",
        r709_col: "Oa11_reflectance",
    })
    return df

def add_ci_cyano(df: pd.DataFrame) -> pd.DataFrame:
    # rho = reflectance * pi
    df["rho_665"] = df["Oa08_reflectance"] * np.pi
    df["rho_681"] = df["Oa10_reflectance"] * np.pi
    df["rho_709"] = df["Oa11_reflectance"] * np.pi

    # CI-cyano (Wynne et al.): CI = -(rho681 - rho665 + ((rho665 - rho709)/(709-665))*(681-665))
    df["CIcyano"] = -(
        df["rho_681"] -
        df["rho_665"] +
        ((df["rho_665"] - df["rho_709"]) / (709.0 - 665.0)) * (681.0 - 665.0)
    )
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance (meters) between arrays (lat1, lon1) and a single point (lat2, lon2).
    lat1/lon1 can be pandas Series or numpy arrays.
    """
    R = 6371000.0  # meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def nunique_latlon_pairs(sub: pd.DataFrame) -> int:
    return len(set(zip(sub["latitude"].to_numpy(), sub["longitude"].to_numpy())))

def load_coords_from_json(path: str) -> Dict[str, Tuple[float, float]]:
    """
    JSON should map names -> [lat, lon] or { "lat": ..., "lon": ... }.
    Example:
    {
      "CHL_NN_R1": [18.44, -66.037],
      "CHL_NN_R2": {"lat": 18.428, "lon": -66.027}
    }
    """
    with open(path, "r") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = (float(v["lat"]), float(v["lon"]))
        else:
            out[k] = (float(v[0]), float(v[1]))
    return out

def _append_dict_row(csv_path: str, fieldnames, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

def _drop_unc_err(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.lower().endswith(("_unc", "_err"))]
    return df.drop(columns=drop_cols, errors="ignore")

def _delogify_chl(df: pd.DataFrame,
                  cols=("CHL_NN", "CHL_OC4ME")) -> pd.DataFrame:
    """
    Convierte columnas en log10(mg/m^3) a mg/m^3 in-place.
    Si una columna no existe, la ignora. NaN se mantienen.
    """
    for col in cols:
        if col in df.columns:
            # Asegura float y aplica 10**x solo en valores no nulos
            mask = df[col].notna()
            df.loc[mask, col] = np.power(10.0, df.loc[mask, col].astype(float))
    return df

def _load_polygon_from_kml(kml_path: str, placemark_name: Optional[str] = None):
    """
    Carga el primer polígono (o el que tenga 'name=placemark_name') de un KML.
    Devuelve un shapely Polygon/MultiPolygon. Soporta MultiGeometry con varios polígonos.
    """
    with open(kml_path, "r", encoding="utf-8") as f:
        kml = f.read()

    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    root = ET.fromstring(kml)

    def _coords_to_poly(coords_text: str):
        pts = []
        for token in coords_text.strip().split():
            lon, lat, *_ = map(float, token.split(","))
            pts.append((lon, lat))
        if len(pts) < 3:
            return None
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)  # arregla self-intersections
        return poly

    polys = []

    # Si se especifica un Placemark por nombre, búscalo
    if placemark_name:
        for pm in root.findall(".//kml:Placemark", ns):
            nm = pm.find("kml:name", ns)
            if nm is not None and (nm.text or "").strip() == placemark_name:
                # puede haber Polygon o MultiGeometry
                for coords_el in pm.findall(".//kml:Polygon//kml:coordinates", ns):
                    poly = _coords_to_poly(coords_el.text)
                    if poly is not None:
                        polys.append(poly)
                break
    else:
        # toma el primer Polygon que encuentres
        for coords_el in root.findall(".//kml:Polygon//kml:coordinates", ns):
            poly = _coords_to_poly(coords_el.text)
            if poly is not None:
                polys.append(poly)
            # si no especificas nombre, con uno basta
            if polys:
                break

    if not polys:
        raise ValueError("No se encontró ningún Polygon en el KML (o no coincide el Placemark).")

    if len(polys) == 1:
        return polys[0]
    return unary_union(polys)  # MultiPolygon si hay varios

def _filter_df_by_polygon(df: pd.DataFrame,
                          polygon,
                          lat_col: str = "latitude",
                          lon_col: str = "longitude") -> pd.DataFrame:
    """
    Filtra filas cuyo (lon,lat) cae dentro del polígono (vectorizado).
    Ignora filas con NaN en coords. Devuelve copia filtrada.
    """
    sub = df.dropna(subset=[lat_col, lon_col])
    if sub.empty:
        return sub
    mask = contains(polygon, sub[lon_col].to_numpy(), sub[lat_col].to_numpy())
    return sub.loc[mask].copy()

# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------

def load_points_dir(points_dir: str, last_date: Optional[Date]) -> pd.DataFrame:
    """
    Lee múltiples CSV por-escena de un directorio. Cada archivo debe llamarse
    <YYYYMMDDTHHMMSS>.csv para extraer la fecha si no hay columna datetime.
    Aplica la misma normalización de columnas que load_points().
    Filtra por last_date si se provee (solo escenas con fecha > last_date).
    """
    pdir = Path(points_dir)
    if not pdir.exists() or not pdir.is_dir():
        raise FileNotFoundError(f"Directorio no encontrado: {points_dir}")

    frames = []
    skipped = 0

    # Excluir CSV que no son escenas (logs, agregados, etc.)
    EXCLUDE = {"time_spent.csv", "chl_daily.csv", "chl_points.csv", "chl_processed_log.csv"}

    for csv_path in sorted(pdir.glob("*.csv")):
        if csv_path.name in EXCLUDE:
            continue
        # Solo escena si el nombre contiene timestamp
        scene_ts = parse_dt_from_filename(csv_path)
        if scene_ts is None:
            # si no cumple patrón, lo saltamos silenciosamente
            skipped += 1
            continue
        if (last_date is not None) and (scene_ts.date() <= last_date):
            # escena ya agregada históricamente
            continue

        try:
            df = pd.read_csv(csv_path)

            # Detectar columnas como en load_points()
            lat_col  = find_col(df, ["latitude", "lat"])
            lon_col  = find_col(df, ["longitude", "lon", "lng"])
            dt_col   = find_col(df, ["datetime", "time", "timestamp", "date"])
            chl_col  = find_col(df, ["CHL_NN", "chl_nn"])
            r665_col = find_col(df, ["Oa08_reflectance", "oa08_reflectance"])
            r681_col = find_col(df, ["Oa10_reflectance", "oa10_reflectance"])
            r709_col = find_col(df, ["Oa11_reflectance", "oa11_reflectance"])

            # Si no trae datetime en columnas, lo rellenamos con el timestamp del filename
            if dt_col is None:
                df["datetime"] = scene_ts
                dt_col = "datetime"

            needed = [lat_col, lon_col, dt_col, chl_col, r665_col, r681_col, r709_col]
            if any(c is None for c in needed):
                log.warning(f"Saltando {csv_path.name}: faltan columnas requeridas.")
                continue

            # Parsear fechas y normalizar nombres
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            df = df.dropna(subset=[dt_col])

            df = df.rename(columns={
                lat_col: "latitude",
                lon_col: "longitude",
                dt_col: "datetime",
                chl_col: "CHL_NN",
                r665_col: "Oa08_reflectance",
                r681_col: "Oa10_reflectance",
                r709_col: "Oa11_reflectance",
            })

            # Log de archivo incluido
            _append_dict_row(
                FILES_LOG,
                ["processed_at", "file", "scene_ts", "rows"],
                {
                    "processed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "file": csv_path.name,
                    "scene_ts": (scene_ts.to_pydatetime() if hasattr(scene_ts, "to_pydatetime") else scene_ts).strftime("%Y-%m-%dT%H:%M:%S"),
                    "rows": int(len(df)),
                },
            )

            frames.append(df)

        except Exception as e:
            log.warning(f"No se pudo leer {csv_path.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=[
            "latitude","longitude","datetime","CHL_NN",
            "Oa08_reflectance","Oa10_reflectance","Oa11_reflectance"
        ])

    big = pd.concat(frames, ignore_index=True)

    # Elimina duplicados exactos de (datetime, lat, lon) si hay solapes entre escenas
    big = big.drop_duplicates(subset=["datetime","latitude","longitude"], keep="last").reset_index(drop=True)
    return big


def compute_daily(points_csv: str,
                  out_csv: str,
                  coverage_threshold: float = DEFAULT_THRESHOLD,
                  positive_chl_only: bool = True,
                  coords_r: Dict[str, Tuple[float, float]] = DEFAULT_COORDS_R,
                  radius_m: float = DEFAULT_RADIUS_M,
                  kml_polygon=None) -> str:
    """
    - Coverage per day = (#unique (lat,lon) with valid CHL_NN) / (#unique (lat,lon) that day) * 100
      * valid CHL_NN: notna (& >0 if positive_chl_only=True)
    - Keep only days with coverage >= threshold
    - Daily means for numeric columns (CHL_NN, reflectances, rho_*, CI_cyano, etc.)
    - Add columns per region key in coords_r: mean CHL_NN within radius_m (meters)
    - Adds coverage columns: coverage_percent, total_pixels, valid_pixels
    """
  
    # 0) Cargar histórico primero para saber last_date
    if os.path.isfile(out_csv):
        prev = pd.read_csv(out_csv, parse_dates=["date"])
        prev["date"] = prev["date"].dt.date
        last_date = max(prev["date"]) if not prev.empty else None
    else:
        prev = None
        last_date = None

    # 1) Cargar datos nuevos:
    if os.path.isdir(points_csv):
        df = load_points_dir(points_csv, last_date=last_date)
    else:
        # caso archivo único (comportamiento anterior)
        df = load_points(points_csv)
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        if last_date is not None:
            df = df[df["date"] > last_date]
        # quitamos 'date' para recalcular luego igual que en branch de directorio
        if "date" in df.columns:
            df = df.drop(columns=["date"])

    if df.empty:
        log.info("No hay fechas nuevas. Nada que hacer.")
        return out_csv

    # --- Filtro espacial por ROI (KML) ---
    if kml_polygon is not None:
        before = len(df)
        df = _filter_df_by_polygon(df, kml_polygon, lat_col="latitude", lon_col="longitude")
        log.info(f"Filtro ROI KML: {before} -> {len(df)} filas dentro del polígono.")
        if df.empty:
            log.info("Tras aplicar ROI no quedaron datos. Nada que hacer.")
            return out_csv


    # 2) Enriquecer y derivar fecha
    df = add_ci_cyano(df)
    df["date"] = df["datetime"].dt.date
    
    # --- CONVERSIÓN a mg/m^3: CHL_NN y CHL_OC4ME dejan de estar en log10 ---
    _delogify_chl(df, cols=("CHL_NN", "CHL_OC4ME"))

    # 3) Cobertura por día
    df_pairs = (
        df[["date", "latitude", "longitude", "CHL_NN"]]
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )

    # 2) base = pares únicos por día (evita doble conteo si hay varias escenas el mismo día)
    base = df_pairs.drop_duplicates(subset=["date", "latitude", "longitude"])

    # 3) válidos = CHL_NN > 0 (o notna si se permite no-positivos)
    if positive_chl_only:
        valid = base[ base["CHL_NN"].gt(0) ]
    else:
        valid = base[ base["CHL_NN"].notna() ]

    # (opcional) integrar flags de calidad si existen:
    # for col, want in [("INVALID", 0), ("CLOUD", 0), ("LAND", 0), ("WATER", 1)]:
    #     if col in base.columns and col in df.columns:
    #         valid = valid[ valid[col] == want ]

    # 4) conteos y porcentaje
    coverage = (
        base.groupby("date").size().rename("total_pixels").to_frame()
        .join(valid.groupby("date").size().rename("valid_pixels"))
        .fillna(0)
        .astype({"total_pixels":"int64", "valid_pixels":"int64"})
    )
    coverage["coverage_percent"] = np.where(
        coverage["total_pixels"] > 0,
        100.0 * coverage["valid_pixels"] / coverage["total_pixels"],
        0.0
    )

    # 5) filtrar fechas por umbral
    keep_dates = coverage.index[ coverage["coverage_percent"] >= coverage_threshold ]
    df_keep = df[df["date"].isin(keep_dates)]
    if df_keep.empty:
        log.info("Fechas nuevas pero ninguna pasa el umbral. No se agrega nada.")
        return out_csv

    # 4) Medias diarias
    excluded = {"latitude", "longitude", "INVALID", "CLOUD", "LAND", "WATER"}
    num_cols = [
        c for c in df_keep.select_dtypes(include=[np.number]).columns
        if c not in excluded and not c.lower().endswith(("_unc", "_err"))
    ]

    daily_means = df_keep.groupby("date")[num_cols].mean().reset_index()

    # 5) Medias regionales alrededor de puntos de referencia
    region_rows = []
    for date_key, g in df_keep.groupby("date"):
        row = {"date": date_key}
        for name, (ref_lat, ref_lon) in coords_r.items():
            dists = haversine_distance(g["latitude"].to_numpy(), g["longitude"].to_numpy(), ref_lat, ref_lon)
            nearby = g.loc[dists <= radius_m]
            row[name] = nearby["CHL_NN"].mean() if not nearby.empty else np.nan
        region_rows.append(row)
    region_df = pd.DataFrame(region_rows)

    cov_reset = coverage.loc[keep_dates].reset_index()
    daily_new = (
        daily_means
        .merge(cov_reset, on="date", how="left")
        .merge(region_df, on="date", how="left")
    )

    daily_new = daily_new.sort_values("date")

    # Renombrar CHL_NN -> CHLL_TOTAL en la salida diaria
    if "CHL_NN" in daily_new.columns:
        daily_new = daily_new.rename(columns={"CHL_NN": "CHLL_NN_TOTAL"})
    
    # Log de "vacíos" en columnas regionales
    region_cols = [k for k in coords_r.keys() if k in daily_new.columns]  # p.ej., ["CHL_NN_R1","CHL_NN_R2","CHL_NN_R3"]
    daily_log_fields = ["processed_at", "date", "coverage_percent"] + [f"{c}_empty" for c in region_cols]

    for _, r in daily_new.iterrows():
        empties = {f"{c}_empty": (pd.isna(r[c]) if c in r else True) for c in region_cols}
        _append_dict_row(
            DAILY_LOG,
            daily_log_fields,
            {
                "processed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "date": r["date"].isoformat() if hasattr(r["date"], "isoformat") else str(r["date"]),
                "coverage_percent": float(r.get("coverage_percent", float("nan"))),
                **empties,
            },
        )

    # 6) Unir con histórico y guardar
    if prev is not None and not prev.empty:
        daily_final = pd.concat([prev, daily_new], ignore_index=True).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        daily_final = daily_new
    
    daily_final = _drop_unc_err(daily_final)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    daily_final.to_csv(out_csv, index=False)
    return out_csv


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build daily chlorophyll averages with coverage gating, CI_cyano, and regional CHL windows.")
    ap.add_argument("--kml-roi", default=DEFAULT_POLYGON,
                help="Ruta a un KML con el polígono del ROI.")
    ap.add_argument("--kml-placemark", default=None,
                help="Nombre del Placemark a usar dentro del KML (opcional).")
    ap.add_argument("--points", default=DEFAULT_IN_DIR, required=False, help="Path to per-point CSV (merged scenes in ROI)")
    ap.add_argument("--out", default=DEFAULT_OUT_FILE, help="Output daily CSV")
    ap.add_argument("--coverage-threshold", type=float, default=DEFAULT_THRESHOLD, help="Min coverage %% to keep a day (default 50)")
    ap.add_argument("--allow-nonpositive-chl", action="store_true",
                    help="Count CHL_NN <= 0 as valid for coverage (default = only CHL_NN > 0 counts)")
    ap.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M, help="Haversine radius in meters for regional means (default 500)")
    ap.add_argument("--coords-json", default=None,
                    help="Optional JSON file mapping names -> [lat, lon] or {\"lat\":..,\"lon\":..}. Defaults to 3 preset points.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    polygon = None
    if args.kml_roi:
        try:
            polygon = _load_polygon_from_kml(args.kml_roi, args.kml_placemark)
            log.info("ROI KML cargado correctamente.")
        except Exception as e:
            log.error(f"No se pudo cargar el KML: {e}")
            return

    coords = DEFAULT_COORDS_R
    if args.coords_json:
        try:
            coords = load_coords_from_json(args.coords_json)
            if not coords:
                raise ValueError("Empty coords JSON.")
            log.info(f"Loaded {len(coords)} reference points from {args.coords_json}")
        except Exception as e:
            log.error(f"Failed to load --coords-json: {e}")
            return

    compute_daily(
        points_csv=args.points,
        out_csv=args.out,
        coverage_threshold=args.coverage_threshold,
        positive_chl_only=(not args.allow_nonpositive_chl),
        coords_r=coords,
        radius_m=args.radius_m,
        kml_polygon=polygon
    )

if __name__ == "__main__":
    main()
