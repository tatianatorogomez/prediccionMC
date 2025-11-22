from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
AVAILABLE_SEDES = ["Ciudad Montes", "Plaza Américas", "Tejar"]
PRODUCT_MAP = {
    1: "Ropa Vieja",
    2: "Pollo",
    3: "Chorizos Alemanes",
    4: "Salsa Chorizos",
    5: "Maiz Texas",
    6: "Champiñones",
    7: "Carne Molida Mexicana",
    8: "Manchamantel",
    9: "Salsa ManchaMantel",
    10: "Salami",
    11: "Salsa Italiana",
    12: "Lechuga",
    13: "Salchichas",
    14: "Jamón",
    15: "Queso",
    16: "Chantilly",
    17: "Frijol",
    18: "Masa",
    19: "Fresa Picada",
    20: "Fresa Entera",
    21: "Banano",
    22: "Limones",
    23: "Piña en Cubos",
    24: "Cerezas",
    25: "Salsa Piña",
    26: "BBQ",
    27: "Mostaneza",
    28: "Cerezada",
    29: "Salsa de Ajo",
    30: "Carne Hamburguesa",
    31: "Salsa de Tocineta",
    32: "Lechuga Crespa",
    33: "Cebolla",
    34: "Tomate",
}
PRODUCT_IDS = set(PRODUCT_MAP.keys())
CATEGORICAL_COLUMNS = ["producto", "sede", "dia_semana"]
FEATURE_COLUMNS = [
    "producto",
    "sede",
    "dia_semana",
    "es_fin_semana",
    "es_quincena",
    "es_pre_quincena",
    "peso_sugerido_total",
    "produccion_cocido_real",
    "almacenamiento_dia_anterior_total",
    "almacenamiento_total",
    "total_sede",
    "almacenamiento_sede",
    "sugerido_media_3d",
    "sugerido_media_7d",
    "sugerido_media_14d",
    "total_media_3d",
    "total_media_7d",
    "total_media_14d",
]


class PredictionRequest(BaseModel):
    sede: Literal["Ciudad Montes", "Plaza Américas", "Tejar"] = Field(
        ..., description="Sede para la cual se solicita la predicción."
    )
    product_id: int = Field(..., description="Identificador numérico del producto.")
    dia_semana: str = Field(..., description="Día de la semana en minúsculas (ej. 'lunes').")
    es_fin_semana: bool
    es_quincena: bool
    es_pre_quincena: bool
    peso_sugerido_total: float
    produccion_cocido_real: float
    almacenamiento_dia_anterior_total: float
    almacenamiento_total: float
    total_sede: float
    almacenamiento_sede: float
    sugerido_media_3d: float
    sugerido_media_7d: float
    sugerido_media_14d: float
    total_media_3d: float
    total_media_7d: float
    total_media_14d: float


class PredictionResponse(BaseModel):
    sede: str
    product_id: int
    producto: str
    prediccion_kg: float
    features_usadas: List[str]


def load_models() -> Dict[str, object]:
    models: Dict[str, object] = {}
    for sede in AVAILABLE_SEDES:
        model_name = f"modelo_lightgbm_{sede.replace(' ', '_')}.pkl"
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo guardado para '{sede}' en {model_path}."
            )
        models[sede] = joblib.load(model_path)
    return models


MODELOS = load_models()
app = FastAPI(title="Servicio de sugeridos por sede", version="1.0.0")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["predicciones"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    if payload.sede not in MODELOS:
        raise HTTPException(status_code=400, detail=f"Sede no soportada: {payload.sede}")

    if payload.product_id not in PRODUCT_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"product_id '{payload.product_id}' no está en el catálogo soportado.",
        )

    data_dict = payload.dict()
    product_name = PRODUCT_MAP[payload.product_id]
    data_dict["producto"] = product_name
    data_dict.pop("product_id", None)

    data = pd.DataFrame([data_dict])

    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas requeridas en el payload: {missing}",
        )

    for col in CATEGORICAL_COLUMNS:
        data[col] = data[col].astype("category")

    modelo = MODELOS[payload.sede]
    prediccion = float(modelo.predict(data[FEATURE_COLUMNS])[0])

    return PredictionResponse(
        sede=payload.sede,
        product_id=payload.product_id,
        producto=product_name,
        prediccion_kg=prediccion,
        features_usadas=FEATURE_COLUMNS,
    )

