# Auditoría de Limpieza Avanzada, Ingeniería de Características y Partición Temporal

### 1. Trazabilidad del Pipeline de Preparación para Series Temporales (2014–2019)

| Etapa del Pipeline | Descripción Metodológica | Registros Afectados / Resultantes |
| :--- | :--- | :---: |
| **1. Consolidado Inicial (15 min)** | Registros agrupados por estación y ventana en `subte_consolidado_15min.parquet` | 6,485,625 |
| **2. Recorte de Horario Comercial** | Exclusión de validaciones residuales nocturnas fuera de servicio (`00:00` a `04:45`) | -11,365 |
| **3. Cierres Estructurales (Fase 3)** | Exclusión de jornadas-estación con cierre total por obras o paros (< 20 pax/día) | 627 días-estación |
| **4. Regularización Equidistante** | Reindexación en grilla continua de 76 ventanas/día (`05:00` a `23:45`) e imputación de 0 en ventanas activas sin pasadas | +711,396 ventanas completadas |
| **5. Tratamiento de Outliers** | Acotamiento (*winsorización*) de descargas acumuladas de telemetría ($> \max(Q_{99.9}, Q_{75}+4.5\cdot\text{IQR})$ en Train) | 4,928 intervalos corregidos |
| **6. Calentamiento de Rezagos (Lags)** | Recorte de las primeras 7 jornadas operativas por estación para inicializar `lag_7d` sin `NaN` | -23,408 |
| **7. Matriz Final de Modelado** | Dataset equidistante libre de nulos y fugas guardado en `subte_modelado_15min.parquet` | **7,145,292** |

### 2. Esquema de Partición Cronológica (*Time-Based Splitting* - Fase 7)

Para respetar la dependencia autoregresiva y evitar la fuga de información futura (*data leakage*), el conjunto de datos se dividió respetando estrictamente el orden cronológico en años calendarios completos (preservando la estacionalidad anual en cada subconjunto):

| Subconjunto | Período Temporal | Registros (Estación × 15 min) | Proporción (%) | Propósito en la Metodología |
| :--- | :---: | :---: | :---: | :--- |
| **Entrenamiento (`train`)** | `2014-01-09` a `2017-12-31` | 4,743,084 | 66.38% | Ajuste de parámetros de Regresión Lineal, escaladores y árboles de Gradient Boosting |
| **Validación (`val`)** | `2018-01-01` a `2018-12-31` | 1,195,860 | 16.74% | Selección de hiperparámetros, regularización y control de sobreajuste |
| **Prueba (`test`)** | `2019-01-01` a `2019-12-31` | 1,206,348 | 16.88% | Evaluación final independiente (MSE, RMSE, MAE) y cálculo de métricas operativas (Fase 8) |

### 3. Diccionario de Características Construidas (Fase 5)

- **Rezagos Autoregresivos (*Lags*):** `lag_1` ($t-15\text{m}$), `lag_2` ($t-30\text{m}$), `lag_4` ($t-60\text{m}$), `lag_1d` (misma ventana hace 1 día), `lag_7d` (misma ventana hace 7 días) y `media_movil_1h` (promedio móvil de la última hora).
- **Variables Temporales y Armónicas:** `hora_decimal` (5.00 a 23.75), `ventana_idx` (0 a 75), `sin_hora`, `cos_hora`, `dia_semana`, `sin_dia_semana`, `cos_dia_semana`, `mes`, `sin_mes`, `cos_mes`, `es_feriado`, `es_sabado`, `es_domingo`, `es_no_laborable`.
- **Variables Espaciales y Geolocalizadas:** `latitud`, `longitud`, `distancia_centro_km` (distancia Haversine al Obelisco), `orden_linea`, `es_cabecera`, `es_combinacion` (nodos de transbordo).
