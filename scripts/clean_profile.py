from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "conzo actualizado.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "guardados.parquet"
REPORT_PATH = ROOT_DIR / "reports" / "profiling_report.html"

NUMERIC_KEYWORDS = (
    "almacenamiento",
    "peso",
    "produccion",
    "total",
    "guardado",
    "planta",
)

COLUMN_RENAME_MAP = {
    "almacenamiento": "almacenamiento_tejar",
    "almacenamiento_1": "almacenamiento_cm",
    "almacenamiento_3": "almacenamiento_pa",
    "almacenamiento_4": "almacenamiento_total",
}

COLUMNS_REQUIRED_PA = [
    "almacenamiento_dia_anterior_pa",
    "peso_sugerido_pa",
    "total_pa",
    "almacenamiento_pa",
]

COLUMNS_TO_DROP = {
    "almacenamiento_dia_anterior_fl",
    "almacenamiento_dia_anterior_rpa",
    "almacenamiento_2",
    "almacenamiento_rpa",
    "sugerido_produccion_en_crudo",
    "peso_sugerido_fl",
    "peso_sugerido_pa_produc_anticipada",
    "peso_sugerido_tejar_reserva",
    "total_fl",
}

COLUMNS_TO_KEEP = [
    "fecha",
    "producto",
    "almacenamiento_dia_anterior_tejar",
    "almacenamiento_dia_anterior_cm",
    "almacenamiento_dia_anterior_pa",
    "almacenamiento_dia_anterior_total",
    "peso_sugerido",
    "peso_sugerido_cm",
    "peso_sugerido_pa",
    "peso_sugerido_total",
    "sugerido_produccion_en_cocido",
    "produccion_cocido_real",
    "produccion_cocido_total",
    "total_tejar",
    "total_cm",
    "total_pa",
    "total",
    "almacenamiento_tejar",
    "almacenamiento_cm",
    "almacenamiento_pa",
    "almacenamiento_total",
    "dia_semana",
    "es_fin_semana",
    "es_quincena",
    "es_pre_quincena",
]

VALID_START_DATE = pd.Timestamp("2018-01-01")

NEGATIVE_COLUMNS = [
    "peso_sugerido",
    "peso_sugerido_cm",
    "peso_sugerido_pa",
    "peso_sugerido_total",
    "sugerido_produccion_en_cocido",
    "produccion_cocido_real",
    "produccion_cocido_total",
    "total_tejar",
    "total_cm",
    "total_pa",
    "total",
    "almacenamiento_tejar",
    "almacenamiento_cm",
    "almacenamiento_pa",
    "almacenamiento_total",
]

PERCENTILE_CAPS = {
    "peso_sugerido_total": 0.99,
    "peso_sugerido": 0.995,
    "peso_sugerido_cm": 0.995,
    "peso_sugerido_pa": 0.995,
}

DAY_NAME_MAP = {
    "Monday": "lunes",
    "Tuesday": "martes",
    "Wednesday": "miercoles",
    "Thursday": "jueves",
    "Friday": "viernes",
    "Saturday": "sabado",
    "Sunday": "domingo",
}


def normalize_column_name(raw_name: str | float | int | None) -> str:
    """Convert raw headers into snake_case ASCII column names."""
    if raw_name is None or (isinstance(raw_name, float) and np.isnan(raw_name)):
        raw_name = ""
    name = str(raw_name).strip().lower()
    name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "col"


def deduplicate_columns(columns: list[str]) -> list[str]:
    """Ensure column names are unique by appending suffixes."""
    counts: dict[str, int] = {}
    unique_cols: List[str] = []
    for col in columns:
        base = col
        if counts.get(base):
            col = f"{base}_{counts[base]}"
        counts[base] = counts.get(base, 0) + 1
        unique_cols.append(col)
    return unique_cols


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Standardize numeric values replacing commas, % and stray characters."""
    as_str = (
        series.astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
        .replace("", np.nan)
    )
    return pd.to_numeric(as_str, errors="coerce")


def detect_outliers(
    df: pd.DataFrame, numeric_cols: list[str] | None = None
) -> pd.DataFrame:
    """Flag rows with z-score > 3 in any numeric column."""
    if numeric_cols is None:
        numeric_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if df[col].dtype != bool
        ]
    else:
        numeric_cols = [
            col
            for col in numeric_cols
            if col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and df[col].dtype != bool
        ]

    outlier_columns = defaultdict(list)
    flag_series = pd.Series(False, index=df.index)

    for col in numeric_cols:
        if col not in df.columns:
            continue
        col_values = df[col]
        valid_count = col_values.notna().sum()
        if valid_count < 5:
            continue
        std = col_values.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        z_scores = (col_values - col_values.mean()) / std
        mask = z_scores.abs() > 3
        if not mask.any():
            continue
        flag_series = flag_series | mask.fillna(False)
        for idx in df.index[mask.fillna(False)]:
            outlier_columns[idx].append(col)

    df["flag_outlier"] = flag_series
    df["outlier_columns"] = df.index.map(
        lambda idx: ",".join(outlier_columns[idx]) if outlier_columns[idx] else np.nan
    )
    return df


def filter_valid_date_range(
    df: pd.DataFrame, start_date: pd.Timestamp = VALID_START_DATE
) -> pd.DataFrame:
    """Remove rows outside the configured date range window."""
    end_date = pd.Timestamp.today().normalize() + pd.DateOffset(years=1)
    mask = df["fecha"].between(start_date, end_date)
    dropped = (~mask).sum()
    if dropped:
        logging.warning(
            "Se eliminaron %s filas fuera del rango de fechas (%s - %s).",
            dropped,
            start_date.date(),
            end_date.date(),
        )
    return df.loc[mask].copy()


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the curated list of columns for model-ready dataset."""
    available_columns = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    missing = sorted(set(COLUMNS_TO_KEEP) - set(available_columns))
    if missing:
        logging.warning(
            "Las siguientes columnas esperadas no están presentes y se omiten: %s",
            ", ".join(missing),
        )
    extra_cols = sorted(set(df.columns) - set(available_columns))
    if extra_cols:
        logging.info(
            "Columnas adicionales descartadas del dataset limpio: %s",
            ", ".join(extra_cols),
        )
    filtered_df = df.loc[:, available_columns].copy()
    filtered_df.sort_values(["fecha", "producto"], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def winsorize_numeric_columns(
    df: pd.DataFrame,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Cap extreme values to mitigate the influence of outliers."""
    if columns is not None:
        numeric_cols = [
            col
            for col in columns
            if col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and df[col].dtype != bool
        ]
    else:
        numeric_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if df[col].dtype != bool
        ]
    for col in numeric_cols:
        lower, upper = df[col].quantile([lower_quantile, upper_quantile])
        if pd.isna(lower) or pd.isna(upper) or lower == upper:
            continue
        df[col] = df[col].clip(lower, upper)
    return df


def drop_rows_missing_columns(
    df: pd.DataFrame, required_columns: list[str], label: str
) -> pd.DataFrame:
    """Drop rows that have missing values in the required columns."""
    existing = [col for col in required_columns if col in df.columns]
    if not existing:
        logging.warning(
            "No se encontraron las columnas requeridas para %s: %s",
            label,
            ", ".join(required_columns),
        )
        return df

    mask = df[existing].notna().all(axis=1)
    dropped = (~mask).sum()
    if dropped:
        logging.info(
            "Se descartaron %s filas sin datos completos para %s.",
            dropped,
            label,
        )
    return df.loc[mask].copy()


def drop_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing negative values in critical numeric columns."""
    columns = [col for col in NEGATIVE_COLUMNS if col in df.columns]
    if not columns:
        return df
    mask = (df[columns] >= 0).all(axis=1)
    dropped = (~mask).sum()
    if dropped:
        logging.info(
            "Se descartaron %s filas con valores negativos en columnas clave.",
            dropped,
        )
    return df.loc[mask].copy()


def cap_extreme_values(df: pd.DataFrame) -> pd.DataFrame:
    """Cap selected columns using predefined percentile thresholds."""
    cols_to_cap = [col for col in PERCENTILE_CAPS if col in df.columns]
    for col in cols_to_cap:
        cap = df[col].quantile(PERCENTILE_CAPS[col])
        df[col] = df[col].clip(upper=cap)
    return df


def remove_flagged_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows flagged as outliers after detection."""
    if "flag_outlier" not in df.columns:
        return df
    mask = ~df["flag_outlier"].fillna(False)
    dropped = (~mask).sum()
    if dropped:
        logging.info("Se eliminaron %s filas marcadas como outliers.", dropped)
    return df.loc[mask].copy()


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    """Load the raw CSV, clean headers, parse datatypes and enrich features."""
    logging.info("Cargando archivo bruto desde %s", raw_path)
    for encoding in ("utf-8", "latin-1", "windows-1252"):
        try:
            df_raw = pd.read_csv(
                raw_path,
                sep=";",
                header=None,
                dtype=str,
                na_values=["", "NA", "N/A", "null"],
                keep_default_na=True,
                engine="python",
                encoding=encoding,
            )
            logging.info("Archivo cargado usando codificación %s.", encoding)
            break
        except UnicodeDecodeError:
            logging.warning("Falló la lectura con codificación %s, probando siguiente.", encoding)
    else:
        raise ValueError(
            "No se pudo decodificar el archivo con las codificaciones utf-8, latin-1 ni windows-1252."
        )

    header_mask = (
        df_raw.iloc[:, 0]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("FECHA")
    )
    if not header_mask.any():
        raise ValueError("No se encontró fila de encabezado que inicie con 'FECHA'.")

    header_idx = header_mask[header_mask].index[0]
    logging.info("Fila de encabezado encontrada en índice %s", header_idx)

    raw_headers = df_raw.iloc[header_idx].fillna("").tolist()
    normalized_headers = deduplicate_columns(
        [normalize_column_name(c) for c in raw_headers]
    )

    df = df_raw.iloc[header_idx + 1 :].reset_index(drop=True)
    df.columns = normalized_headers
    df = df.dropna(how="all")

    if "fecha" not in df.columns:
        raise ValueError("La columna 'fecha' no está presente luego de limpiar los datos.")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    df = df[df["fecha"].notna()].copy()
    if "producto" in df.columns:
        df["producto"] = df["producto"].fillna("").astype(str).str.strip()
        df = df[df["producto"] != ""]
    df.sort_values("fecha", inplace=True)
    removed_cols = sorted(col for col in COLUMNS_TO_DROP if col in df.columns)
    if removed_cols:
        logging.info(
            "Eliminando columnas no relevantes: %s",
            ", ".join(removed_cols),
        )
    df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors="ignore")

    numeric_cols = [
        col for col in df.columns if any(keyword in col for keyword in NUMERIC_KEYWORDS)
    ]
    for col in numeric_cols:
        df[col] = clean_numeric_series(df[col])

    df["dia_semana"] = df["fecha"].dt.day_name().map(DAY_NAME_MAP)
    df["es_fin_semana"] = df["fecha"].dt.weekday >= 5
    df["es_quincena"] = df["fecha"].dt.day.isin([15, 30])
    df["es_pre_quincena"] = df["fecha"].dt.day.isin([14, 29])

    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    df = drop_rows_missing_columns(
        df, COLUMNS_REQUIRED_PA, label="Plaza de las Américas"
    )
    df = drop_negative_values(df)
    df = cap_extreme_values(df)

    df = filter_valid_date_range(df)
    relevant_numeric_cols = [
        col
        for col in COLUMNS_TO_KEEP
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    df = winsorize_numeric_columns(df, columns=relevant_numeric_cols)
    df = detect_outliers(df, numeric_cols=relevant_numeric_cols)
    df = remove_flagged_outliers(df)
    df = select_relevant_columns(df)
    return df


def build_profile(df: pd.DataFrame, output_path: Path) -> None:
    """Generate an HTML profiling report."""
    logging.info("Generando reporte de perfilado en %s", output_path)
    profile = ProfileReport(
        df,
        title="Reporte de Perfilado - Guardados por sede",
        minimal=True,
        explorative=True,
        correlations={"pearson": {"calculate": False}},
    )
    profile.to_file(output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logging.info("Iniciando proceso de limpieza y perfilado.")
    df_clean = load_and_clean(RAW_DATA_PATH)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Guardando dataset limpio en %s", PROCESSED_DATA_PATH)
    df_clean.to_parquet(PROCESSED_DATA_PATH, index=False)

    build_profile(df_clean, REPORT_PATH)
    logging.info("Proceso finalizado correctamente.")


if __name__ == "__main__":
    main()

