from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "guardados.parquet"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Transform wide dataset into long format per (fecha, producto, sede)."""
    value_map = {
        "Tejar": {
            "sugerido": "peso_sugerido",
            "total": "total_tejar",
            "almacen": "almacenamiento_tejar",
        },
        "Ciudad Montes": {
            "sugerido": "peso_sugerido_cm",
            "total": "total_cm",
            "almacen": "almacenamiento_cm",
        },
        "Plaza Américas": {
            "sugerido": "peso_sugerido_pa",
            "total": "total_pa",
            "almacen": "almacenamiento_pa",
        },
    }

    registros: list[pd.DataFrame] = []
    base_cols = [
        "fecha",
        "producto",
        "peso_sugerido_total",
        "produccion_cocido_real",
        "almacenamiento_dia_anterior_total",
        "almacenamiento_total",
        "dia_semana",
        "es_fin_semana",
        "es_quincena",
        "es_pre_quincena",
    ]

    for sede, mapping in value_map.items():
        subset = df[base_cols + [mapping["sugerido"], mapping["total"], mapping["almacen"]]].copy()
        subset = subset.rename(
            columns={
                mapping["sugerido"]: "peso_sugerido_sede",
                mapping["total"]: "total_sede",
                mapping["almacen"]: "almacenamiento_sede",
            }
        )
        subset["sede"] = sede
        registros.append(subset)

    return pd.concat(registros, ignore_index=True)


def add_rolling_features(data: pd.DataFrame, windows: tuple[int, ...] = (3, 7, 14)) -> pd.DataFrame:
    enriched = data.sort_values(["producto", "sede", "fecha"]).copy()
    for window in windows:
        enriched[f"sugerido_media_{window}d"] = (
            enriched.groupby(["producto", "sede"])["peso_sugerido_sede"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        enriched[f"total_media_{window}d"] = (
            enriched.groupby(["producto", "sede"])["total_sede"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return enriched


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logging.info("Cargando dataset %s", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    logging.info("Construyendo dataset largo por sede...")
    df_long = build_training_frame(df)
    df_long = add_rolling_features(df_long)

    split_date = df_long["fecha"].quantile(0.8)
    train_df = df_long[df_long["fecha"] <= split_date].reset_index(drop=True).copy()
    val_df = df_long[df_long["fecha"] > split_date].reset_index(drop=True).copy()

    logging.info("Fecha de corte: %s", split_date.date())
    logging.info("Tamaño entrenamiento: %s", train_df.shape)
    logging.info("Tamaño validación: %s", val_df.shape)

    target = "peso_sugerido_sede"
    categorical_columns = ["producto", "sede", "dia_semana"]
    for col in categorical_columns:
        train_df[col] = train_df[col].astype("category")
        val_df[col] = val_df[col].astype("category")

    excluded_features = {
        target,
        "fecha",
        "sugerido_produccion_en_cocido",
        "produccion_cocido_total",
    }
    feature_cols = [col for col in train_df.columns if col not in excluded_features]
    logging.info("Número de features: %s", len(feature_cols))
    logging.info("Features: %s", feature_cols)

    modelos: dict[str, LGBMRegressor] = {}
    metricas: list[dict[str, float | str | int]] = []

    for sede_actual in train_df["sede"].cat.categories:
        logging.info("Entrenando modelo para sede %s", sede_actual)
        train_sede = train_df[train_df["sede"] == sede_actual]
        val_sede = val_df[val_df["sede"] == sede_actual]

        X_train = train_sede[feature_cols]
        y_train = train_sede[target]
        X_val = val_sede[feature_cols]
        y_val = val_sede[target]

        modelo = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        modelo.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=categorical_columns,
            callbacks=[],
        )

        y_pred = modelo.predict(X_val)
        mae = float(np.mean(np.abs(y_val - y_pred)))
        mape = float(np.mean(np.abs((y_val - y_pred) / y_val.replace(0, np.nan))))

        metricas.append(
            {
                "sede": sede_actual,
                "MAE": mae,
                "MAPE": mape,
                "muestras_val": len(y_val),
            }
        )

        modelos[sede_actual] = modelo

    metricas_df = pd.DataFrame(metricas)
    logging.info("Métricas obtenidas:\n%s", metricas_df)

    MODEL_DIR.mkdir(exist_ok=True)
    for sede_actual, modelo in modelos.items():
        path = MODEL_DIR / f"modelo_lightgbm_{sede_actual.replace(' ', '_')}.pkl"
        joblib.dump(modelo, path, compress=3)
        logging.info(
            "Modelo guardado %s (features: %s)",
            path,
            getattr(modelo, "n_features_in_", None),
        )


if __name__ == "__main__":
    main()

