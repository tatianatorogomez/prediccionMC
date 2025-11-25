# Resumen del modelo de sugeridos

Este documento describe el pipeline actual para estimar el `peso_sugerido_sede` utilizado en la plataforma de Retool, así como los resultados obtenidos y cómo reutilizar los artefactos entrenados.

## Pipeline de datos

1. **Ingesta y limpieza (`scripts/clean_profile.py`):**
   - Lee `conzo actualizado.csv`, normaliza encabezados y tipos numéricos.
   - Elimina columnas irrelevantes (FL, RPA, reservas especiales) y filas con fechas fuera de 2018–2026.
   - Descarta registros con valores negativos o incompletos para Plaza de las Américas y realiza un *winsorize* al 99.º percentil para sugeridos grandes.
   - Marca outliers vía z-score y los elimina antes de guardar el dataset depurado en `data/processed/guardados.parquet` (12 942 filas, 25 columnas).

2. **Feature engineering (`analysis/eda_guardados.ipynb`):**
   - Se transforma el dataset a formato largo por `(fecha, sede, producto)`, renombrando columnas específicas de cada sede (`peso_sugerido_sede`, `total_sede`, `almacenamiento_sede`).
   - Se generan medias móviles de 3, 7 y 14 días para `peso_sugerido_sede` y `total_sede`.
   - Se añaden variables temporales (`dia_semana`, `es_fin_semana`, `es_quincena`, `es_pre_quincena`).
   - Se divide el histórico en entrenamiento (80 %) y validación (20 %) utilizando `fecha` como eje temporal.

## Entrenamiento

- Modelo: `LGBMRegressor` (LightGBM) independiente por sede.
- Objetivo (`target`): `peso_sugerido_sede`.
- Features utilizadas:
- Columnas numéricas: `peso_sugerido_total`, `produccion_cocido_real`, `almacenamiento_dia_anterior_total`, `almacenamiento_total`, `total_sede`, `almacenamiento_sede`, `sugerido_media_{3,7,14}d`, `total_media_{3,7,14}d`.
  - Categóricas: `producto`, `sede`, `dia_semana`.

### Métricas de validación

| Sede             | MAE (kg) | MAPE | Muestras val. |
|------------------|---------:|-----:|---------------:|
| Ciudad Montes    | 73.12    | 0.05 | 2 573 |
| Plaza Américas   | 67.40    | 0.05 | 2 573 |
| Tejar            | 62.42    | 0.05 | 2 573 |

Los tres modelos mantienen un error relativo ≈5 %, adecuado para planificación operativa.

> **Nota:** si se reentrena el modelo sin las columnas `sugerido_produccion_en_cocido` y `produccion_cocido_total`, recuerda rehacer el notebook `analysis/eda_guardados.ipynb` con la versión actualizada del listado de features y volver a guardar los `.pkl`.

## Artefactos guardados

Los modelos entrenados se guardan con `joblib` en `models/`:

- `modelo_lightgbm_Ciudad_Montes.pkl`
- `modelo_lightgbm_Plaza_Américas.pkl`
- `modelo_lightgbm_Tejar.pkl`

Para cargar cualquier modelo:

```python
import joblib
model = joblib.load("models/modelo_lightgbm_Tejar.pkl")
pred = model.predict(nuevos_datos_df[FEATURE_COLUMNS])
```

Asegúrate de que el DataFrame de inferencia contenga exactamente las columnas de `FEATURE_COLUMNS` con los mismos tipos (categóricas en `producto`, `sede`, `dia_semana`).

## Endpoint de inferencia

El servicio FastAPI propuesto en `api/app.py`:

- Carga los modelos de `models/`.
- Recibe un JSON con las features necesarias (incluidas las medias móviles) y el `product_id` en lugar del nombre literal.
- Devuelve el `peso_sugerido_sede` predicho y eco de las variables clave.

Ejemplo de payload:

```json
{
  "sede": "Tejar",
  "product_id": 7,
  "dia_semana": "lunes",
  "es_fin_semana": false,
  "es_quincena": false,
  "es_pre_quincena": false,
  "peso_sugerido_total": 4200,
  "produccion_cocido_real": 3600,
  "almacenamiento_dia_anterior_total": 800,
  "almacenamiento_total": 900,
  "total_sede": 1400,
  "almacenamiento_sede": 300,
  "sugerido_media_3d": 1350,
  "sugerido_media_7d": 1325,
  "sugerido_media_14d": 1290,
  "total_media_3d": 1380,
  "total_media_7d": 1350,
  "total_media_14d": 1310
}
```

El campo `product_id` debe corresponder a uno de los productos incluidos en el catálogo interno:

| id | Producto              | id | Producto           |
|----|----------------------|----|--------------------|
| 1  | Ropa Vieja           | 18 | Masa               |
| 2  | Pollo                | 19 | Fresa Picada       |
| 3  | Chorizos Alemanes    | 20 | Fresa Entera       |
| 4  | Salsa Chorizos       | 21 | Banano             |
| 5  | Maiz Texas           | 22 | Limones            |
| 6  | Champiñones          | 23 | Piña en Cubos      |
| 7  | Carne Molida Mexicana| 24 | Cerezas            |
| 8  | Manchamantel         | 25 | Salsa Piña         |
| 9  | Salsa ManchaMantel   | 26 | BBQ                |
| 10 | Salami               | 27 | Mostaneza          |
| 11 | Salsa Italiana       | 28 | Cerezada           |
| 12 | Lechuga              | 29 | Salsa de Ajo       |
| 13 | Salchichas           | 30 | Carne Hamburguesa  |
| 14 | Jamón                | 31 | Salsa de Tocineta  |
| 15 | Queso                | 32 | Lechuga Crespa     |
| 16 | Chantilly            | 33 | Cebolla            |
| 17 | Frijol               | 34 | Tomate             |

`Retool` puede calcular las medias móviles en el frontend o consultando un endpoint auxiliar que devuelva los últimos valores.

### Descripción de los campos de entrada

- **sede**: sede objetivo de la predicción (`"Ciudad Montes"`, `"Plaza Américas"`, `"Tejar"`).
- **product_id**: identificador numérico del producto según la tabla anterior.
- **dia_semana**: día de la semana en minúsculas (`"lunes"`, `"martes"`, …).
- **es_fin_semana**: `true` si la fecha es sábado o domingo; `false` en caso contrario.
- **es_quincena**: `true` si la fecha es 15 o 30 del mes (días de pago); `false` en otro caso.
- **es_pre_quincena**: `true` si la fecha es 14 o 29 (previo a pago); `false` resto de días.
- **peso_sugerido_total**: sugerido total (todas las sedes) planificado para el día.
- **produccion_cocido_real**: producción cocida real del día anterior (total).
- **almacenamiento_dia_anterior_total**: inventario total disponible al iniciar el día.
- **almacenamiento_total**: inventario total esperado tras producción/distribución.
- **total_sede**: ventas/consumo de la sede en el día anterior.
- **almacenamiento_sede**: inventario de la sede al inicio del día.
- **sugerido_media_3d / 7d / 14d**: medias móviles de `peso_sugerido_sede` en los últimos 3, 7 y 14 días.
- **total_media_3d / 7d / 14d**: medias móviles de `total_sede` (ventas) en los últimos 3, 7 y 14 días.

## Próximos pasos sugeridos

1. Generar reportes de residuos y comparar contra un baseline simple (media móvil) para documentar la mejora.
2. Incorporar nuevas features (festivos externos, eventos puntuales).
3. Automatizar el pipeline (cron/CI) para reentrenar y desplegar modelos con datos actualizados.
4. Integrar el servicio en la plataforma de Retool asegurando autenticación por token/API key.

## Despliegue en Vercel

1. **Dependencias**: `requirements.txt` incluye FastAPI, LightGBM, pandas, numpy, etc. (instaladas automáticamente por `@vercel/python`).
2. **Entry point**: `api/index.py` expone la instancia `app` importando `api/app.py`.
3. **Configuración**: el archivo `vercel.json` indica a Vercel que use `@vercel/python` y enrute todas las peticiones hacia `api/index.py`.
4. **Modelos**: la carpeta `models/` (con los `.pkl`) debe formar parte del despliegue; se cargan al iniciar la función.
5. **Comandos**:
   ```bash
   npm install -g vercel      # una vez
   vercel                     # despliegue preview
   vercel --prod              # despliegue producción
   ```
6. **Endpoints**:
   - `GET /health` → chequeo de estado.
   - `POST /predict` → predicción enviando el payload descrito arriba.

