"""
Fase 3 (Avanzada) y Fase 5: Limpieza para Series Temporales, Regularización de Grilla,
Tratamiento de Outliers, Ingeniería de Características (Lags y Geolocalización) y
Partición Cronológica (Time-Based Splitting) para el Modelado Supervisado (2014-2019).
"""
import os
import time
import numpy as np
import pandas as pd
import holidays
from src.estaciones_geo import get_estaciones_metadata_df


def prepare_modeling_data():
    t0 = time.time()
    print("=" * 72, flush=True)
    print("FASE 3 Y 5: PREPARACIÓN Y LIMPIEZA AVANZADA PARA MODELADO (2014-2019)", flush=True)
    print("=" * 72, flush=True)

    input_path = os.path.join('data_processed', 'subte_consolidado_15min.parquet')
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró {input_path}. Ejecute primero preprocesamiento.py")

    df = pd.read_parquet(input_path)
    raw_consolidated_rows = len(df)
    print(f"[1/6] Dataset consolidado cargado: {raw_consolidated_rows:,} registros (2014-2019).", flush=True)

    # -------------------------------------------------------------------------
    # 1. FILTRADO DE HORARIO OPERATIVO COMERCIAL (05:00 a 23:45 -> 76 ventanas)
    # -------------------------------------------------------------------------
    ventanas_operativas = sorted([
        f"{h:02d}:{m:02d}" for h in range(5, 24) for m in (0, 15, 30, 45)
    ])
    ventana_to_idx = {v: i for i, v in enumerate(ventanas_operativas)}

    mask_operativo = df['ventana_15min'].isin(ventana_to_idx)
    night_rows_removed = int((~mask_operativo).sum())
    df = df[mask_operativo].copy()
    print(f"[2/6] Filtro horario comercial (05:00-23:45): {night_rows_removed:,} registros nocturnos excluidos.", flush=True)

    # -------------------------------------------------------------------------
    # 2. DETECCIÓN DE NULOS ESTRUCTURALES (CIERRES TOTALES DE ESTACIÓN / PAROS)
    # -------------------------------------------------------------------------
    # Si una estación registra menos de 20 ingresos en toda una jornada completa,
    # corresponde a un cierre estructural (remodelación de estación o paro total).
    UMBRAL_CIERRE_DIARIO = 20
    flujo_diario_est = df.groupby(['linea', 'estacion', 'fecha'], as_index=False)['ingresos'].sum()
    flujo_diario_est.rename(columns={'ingresos': 'ingresos_dia_estacion'}, inplace=True)

    dias_totales_observados = len(flujo_diario_est)
    dias_cierre = flujo_diario_est[flujo_diario_est['ingresos_dia_estacion'] < UMBRAL_CIERRE_DIARIO]
    n_dias_cierre = len(dias_cierre)

    dias_activos = flujo_diario_est[flujo_diario_est['ingresos_dia_estacion'] >= UMBRAL_CIERRE_DIARIO][
        ['linea', 'estacion', 'fecha']
    ].copy()
    print(f"[3/6] Detección de cierres estructurales: {n_dias_cierre:,} jornadas-estación con cierre total excluidas "
          f"(de {dias_totales_observados:,} jornadas-estación observadas).", flush=True)

    # -------------------------------------------------------------------------
    # 3. REGULARIZACIÓN DE LA GRILLA TEMPORAL EQUIDISTANTE (76 VENTANAS / DÍA)
    # -------------------------------------------------------------------------
    # Construimos la grilla cartesiana exacta (jornadas activas x 76 ventanas de 15 min)
    # para que los desplazamientos de lags (.shift) sean estrictamente equidistantes.
    dias_activos['key'] = 1
    df_ventanas = pd.DataFrame({
        'ventana_15min': ventanas_operativas,
        'ventana_idx': np.arange(len(ventanas_operativas), dtype=np.int16),
        'key': 1
    })
    grid = pd.merge(dias_activos, df_ventanas, on='key').drop(columns=['key'])

    df_merged = pd.merge(
        grid,
        df[['linea', 'estacion', 'fecha', 'ventana_15min', 'ingresos']],
        on=['linea', 'estacion', 'fecha', 'ventana_15min'],
        how='left'
    )
    del grid, df

    missing_windows_filled = int(df_merged['ingresos'].isna().sum())
    df_merged['ingresos'] = df_merged['ingresos'].fillna(0.0).astype(np.float32)
    print(f"[4/6] Grilla equidistante regularizada: {len(df_merged):,} registros totales "
          f"({missing_windows_filled:,} intervalos operativos sin pasadas imputados con 0).", flush=True)

    # -------------------------------------------------------------------------
    # 4. CALENDARIO Y TRATAMIENTO DE OUTLIERS DE TELEMETRÍA POR ESTACIÓN
    # -------------------------------------------------------------------------
    ar_holidays = set(holidays.Argentina(years=range(2014, 2020)).keys())
    df_merged['anio'] = df_merged['fecha'].dt.year.astype(np.int16)
    df_merged['mes'] = df_merged['fecha'].dt.month.astype(np.int8)
    df_merged['dia_semana'] = df_merged['fecha'].dt.dayofweek.astype(np.int8)
    df_merged['es_feriado'] = df_merged['fecha'].dt.date.isin(ar_holidays).astype(np.int8)
    df_merged['es_sabado'] = (df_merged['dia_semana'] == 5).astype(np.int8)
    df_merged['es_domingo'] = (df_merged['dia_semana'] == 6).astype(np.int8)
    df_merged['es_no_laborable'] = (
        (df_merged['dia_semana'].isin([5, 6])) | (df_merged['es_feriado'] == 1)
    ).astype(np.int8)
    df_merged['tipo_dia'] = np.where(df_merged['es_no_laborable'] == 1, 'No Laborable', 'Hábil')

    # Detección de outliers de telemetría (descargas acumuladas de molinete)
    # Calculamos el umbral por (estacion, es_no_laborable) EXCLUSIVAMENTE sobre el período de Train (2014-2017)
    # para evitar cualquier data leakage estadístico hacia Validación/Test.
    train_sub = df_merged[df_merged['anio'] <= 2017]
    stats_umbral = train_sub.groupby(['estacion', 'es_no_laborable'])['ingresos'].agg(
        q75=lambda x: np.percentile(x, 75),
        q25=lambda x: np.percentile(x, 25),
        q999=lambda x: np.percentile(x, 99.9)
    ).reset_index()
    stats_umbral['iqr'] = stats_umbral['q75'] - stats_umbral['q25']
    stats_umbral['umbral_outlier'] = np.maximum(
        stats_umbral['q999'],
        stats_umbral['q75'] + 4.5 * stats_umbral['iqr']
    ).astype(np.float32)

    df_merged = pd.merge(
        df_merged,
        stats_umbral[['estacion', 'es_no_laborable', 'umbral_outlier']],
        on=['estacion', 'es_no_laborable'],
        how='left'
    )

    mask_outlier = df_merged['ingresos'] > df_merged['umbral_outlier']
    n_outliers_capped = int(mask_outlier.sum())
    max_before_cap = float(df_merged['ingresos'].max())
    df_merged['ingresos'] = np.where(
        mask_outlier,
        df_merged['umbral_outlier'],
        df_merged['ingresos']
    ).astype(np.float32)
    max_after_cap = float(df_merged['ingresos'].max())
    df_merged.drop(columns=['umbral_outlier'], inplace=True)
    print(f"[5/6] Limpieza de outliers de telemetría: {n_outliers_capped:,} picos anómalos acotados "
          f"(máximo bruto {max_before_cap:,.0f} -> máximo limpio {max_after_cap:,.0f} pax/15min).", flush=True)

    # -------------------------------------------------------------------------
    # 5. INGENIERÍA DE CARACTERÍSTICAS: REZAGOS (LAGS), CICLOS Y GEOLOCALIZACIÓN
    # -------------------------------------------------------------------------
    print("[6/6] Construyendo rezagos autoregresivos, variables cíclicas y espaciales...", flush=True)
    df_merged.sort_values(by=['estacion', 'fecha', 'ventana_idx'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # Hora decimal y codificación armónica (seno / coseno)
    df_merged['hora'] = (5 + (df_merged['ventana_idx'] // 4)).astype(np.int8)
    df_merged['hora_decimal'] = (5.0 + df_merged['ventana_idx'] * 0.25).astype(np.float32)

    df_merged['sin_hora'] = np.sin(2.0 * np.pi * df_merged['hora_decimal'] / 24.0).astype(np.float32)
    df_merged['cos_hora'] = np.cos(2.0 * np.pi * df_merged['hora_decimal'] / 24.0).astype(np.float32)
    df_merged['sin_dia_semana'] = np.sin(2.0 * np.pi * df_merged['dia_semana'] / 7.0).astype(np.float32)
    df_merged['cos_dia_semana'] = np.cos(2.0 * np.pi * df_merged['dia_semana'] / 7.0).astype(np.float32)
    df_merged['sin_mes'] = np.sin(2.0 * np.pi * (df_merged['mes'] - 1) / 12.0).astype(np.float32)
    df_merged['cos_mes'] = np.cos(2.0 * np.pi * (df_merged['mes'] - 1) / 12.0).astype(np.float32)

    # Rezagos por estación (76 ventanas = 1 jornada operativa completa)
    grp = df_merged.groupby('estacion')['ingresos']
    df_merged['lag_1'] = grp.shift(1).astype(np.float32)      # t - 15 min
    df_merged['lag_2'] = grp.shift(2).astype(np.float32)      # t - 30 min
    df_merged['lag_3'] = grp.shift(3).astype(np.float32)      # t - 45 min
    df_merged['lag_4'] = grp.shift(4).astype(np.float32)      # t - 60 min (1 hora antes)
    df_merged['lag_1d'] = grp.shift(76).astype(np.float32)    # Misma ventana de 15m hace 1 día
    df_merged['lag_7d'] = grp.shift(76 * 7).astype(np.float32) # Misma ventana de 15m hace 7 días (mismo día semana)

    # Media móvil de la última hora (exclusivamente con valores pasados t-1 a t-4)
    df_merged['media_movil_1h'] = (
        (df_merged['lag_1'] + df_merged['lag_2'] + df_merged['lag_3'] + df_merged['lag_4']) / 4.0
    ).astype(np.float32)
    df_merged.drop(columns=['lag_3'], inplace=True)

    # Horizontes futuros para evaluación multi-step (t+1 = +15m, t+2 = +30m, t+4 = +60m)
    df_merged['target_t2'] = grp.shift(-1).astype(np.float32) # +30 min (1 paso adelante adicional)
    df_merged['target_t4'] = grp.shift(-3).astype(np.float32) # +60 min (3 pasos adelante adicionales)

    # Incorporar metadatos geolocalizados de las 44 estaciones
    df_geo = get_estaciones_metadata_df()
    df_merged = pd.merge(df_merged, df_geo, on='estacion', how='left')

    # Eliminar las primeras 7 jornadas de calentamiento de lags por estación (donde lag_7d es NaN)
    rows_before_lag_drop = len(df_merged)
    df_merged.dropna(subset=['lag_1', 'lag_2', 'lag_4', 'lag_1d', 'lag_7d', 'media_movil_1h'], inplace=True)
    lag_warmup_dropped = rows_before_lag_drop - len(df_merged)

    # -------------------------------------------------------------------------
    # 6. PARTICIÓN CRONOLÓGICA (TIME-BASED SPLITTING: TRAIN / VAL / TEST)
    # -------------------------------------------------------------------------
    # Ordenamos estrictamente por fecha y ventana horaria
    df_merged.sort_values(by=['fecha', 'ventana_idx', 'linea', 'estacion'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    cond_train = df_merged['anio'] <= 2017
    cond_val = df_merged['anio'] == 2018
    cond_test = df_merged['anio'] == 2019

    df_merged['particion'] = np.select(
        [cond_train, cond_val, cond_test],
        ['train', 'val', 'test'],
        default='train'
    )

    n_train = int(cond_train.sum())
    n_val = int(cond_val.sum())
    n_test = int(cond_test.sum())
    n_total_final = len(df_merged)

    output_model_parquet = os.path.join('data_processed', 'subte_modelado_15min.parquet')
    df_merged.to_parquet(output_model_parquet, index=False)

    # -------------------------------------------------------------------------
    # GENERAR REPORTE MARKDOWN DE AUDITORÍA DE LIMPIEZA Y PARTICIÓN
    # -------------------------------------------------------------------------
    os.makedirs(os.path.join('output', 'tablas'), exist_ok=True)
    md_path = os.path.join('output', 'tablas', '07_auditoria_limpieza_modelado.md')

    md = "# Auditoría de Limpieza Avanzada, Ingeniería de Características y Partición Temporal\n\n"
    md += "### 1. Trazabilidad del Pipeline de Preparación para Series Temporales (2014–2019)\n\n"
    md += "| Etapa del Pipeline | Descripción Metodológica | Registros Afectados / Resultantes |\n"
    md += "| :--- | :--- | :---: |\n"
    md += f"| **1. Consolidado Inicial (15 min)** | Registros agrupados por estación y ventana en `subte_consolidado_15min.parquet` | {raw_consolidated_rows:,} |\n"
    md += f"| **2. Recorte de Horario Comercial** | Exclusión de validaciones residuales nocturnas fuera de servicio (`00:00` a `04:45`) | -{night_rows_removed:,} |\n"
    md += f"| **3. Cierres Estructurales (Fase 3)** | Exclusión de jornadas-estación con cierre total por obras o paros (< {UMBRAL_CIERRE_DIARIO} pax/día) | {n_dias_cierre:,} días-estación |\n"
    md += f"| **4. Regularización Equidistante** | Reindexación en grilla continua de 76 ventanas/día (`05:00` a `23:45`) e imputación de 0 en ventanas activas sin pasadas | +{missing_windows_filled:,} ventanas completadas |\n"
    md += f"| **5. Tratamiento de Outliers** | Acotamiento (*winsorización*) de descargas acumuladas de telemetría ($> \\max(Q_{{99.9}}, Q_{{75}}+4.5\\cdot\\text{{IQR}})$ en Train) | {n_outliers_capped:,} intervalos corregidos |\n"
    md += f"| **6. Calentamiento de Rezagos (Lags)** | Recorte de las primeras 7 jornadas operativas por estación para inicializar `lag_7d` sin `NaN` | -{lag_warmup_dropped:,} |\n"
    md += f"| **7. Matriz Final de Modelado** | Dataset equidistante libre de nulos y fugas guardado en `subte_modelado_15min.parquet` | **{n_total_final:,}** |\n\n"

    md += "### 2. Esquema de Partición Cronológica (*Time-Based Splitting* - Fase 7)\n\n"
    md += "Para respetar la dependencia autoregresiva y evitar la fuga de información futura (*data leakage*), el conjunto de datos se dividió respetando estrictamente el orden cronológico en años calendarios completos (preservando la estacionalidad anual en cada subconjunto):\n\n"
    md += "| Subconjunto | Período Temporal | Registros (Estación × 15 min) | Proporción (%) | Propósito en la Metodología |\n"
    md += "| :--- | :---: | :---: | :---: | :--- |\n"
    md += f"| **Entrenamiento (`train`)** | `2014-01-09` a `2017-12-31` | {n_train:,} | {(n_train/n_total_final)*100:.2f}% | Ajuste de parámetros de Regresión Lineal, escaladores y árboles de Gradient Boosting |\n"
    md += f"| **Validación (`val`)** | `2018-01-01` a `2018-12-31` | {n_val:,} | {(n_val/n_total_final)*100:.2f}% | Selección de hiperparámetros, regularización y control de sobreajuste |\n"
    md += f"| **Prueba (`test`)** | `2019-01-01` a `2019-12-31` | {n_test:,} | {(n_test/n_total_final)*100:.2f}% | Evaluación final independiente (MSE, RMSE, MAE) y cálculo de métricas operativas (Fase 8) |\n\n"

    md += "### 3. Diccionario de Características Construidas (Fase 5)\n\n"
    md += "- **Rezagos Autoregresivos (*Lags*):** `lag_1` ($t-15\\text{m}$), `lag_2` ($t-30\\text{m}$), `lag_4` ($t-60\\text{m}$), `lag_1d` (misma ventana hace 1 día), `lag_7d` (misma ventana hace 7 días) y `media_movil_1h` (promedio móvil de la última hora).\n"
    md += "- **Variables Temporales y Armónicas:** `hora_decimal` (5.00 a 23.75), `ventana_idx` (0 a 75), `sin_hora`, `cos_hora`, `dia_semana`, `sin_dia_semana`, `cos_dia_semana`, `mes`, `sin_mes`, `cos_mes`, `es_feriado`, `es_sabado`, `es_domingo`, `es_no_laborable`.\n"
    md += "- **Variables Espaciales y Geolocalizadas:** `latitud`, `longitud`, `distancia_centro_km` (distancia Haversine al Obelisco), `orden_linea`, `es_cabecera`, `es_combinacion` (nodos de transbordo).\n"

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)

    elapsed = time.time() - t0
    print(f"Dataset de modelado guardado en '{output_model_parquet}' ({n_total_final:,} filas) en {elapsed:.1f}s.", flush=True)
    print(f"Reporte guardado en '{md_path}'.", flush=True)
    return df_merged


if __name__ == '__main__':
    prepare_modeling_data()
