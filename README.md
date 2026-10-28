# Variación y Modelado Predictivo de la Demanda en el Subte de CABA (Líneas A, B y C)
### Preprocesamiento Masivo, EDA, Validación Estadística, Ingeniería de Series Temporales y Aprendizaje Supervisado (2014–2019)

Este repositorio contiene el pipeline analítico y predictivo completo para el estudio empírico y modelado de la demanda de pasajeros en la Red de Subterráneos de la Ciudad Autónoma de Buenos Aires (Líneas A, B y C, 44 estaciones), evaluando la heterogeneidad entre **días hábiles y no laborables** (sábados, domingos y feriados nacionales) bajo la metodología **CRISP-DM adaptada a series temporales** (Fases 1 a 8).

---

## 📊 Principales Hallazgos y Resultados (EDA + Modelado)

1. **Volumen Procesado y Grilla Temporal:** Más de 70 millones de transacciones de molinetes consolidadas en **44 estaciones** (2014–2019) y regularizadas en una matriz equidistante de **7.145.292 registros** (76 ventanas comerciales de 15 min por día, `05:00` a `23:45`).
2. **Heterogeneidad Estadística (Kolmogorov-Smirnov):** Se rechazó $H_0$ en el **100% de las 44 estaciones** ($p < 10^{-15}$), validando que las distribuciones de demanda en días no laborables difieren estructuralmente del régimen laboral.
3. **Modelado Supervisado (Regresión Lineal vs. Gradient Boosting / XGBoost):**
   - Bajo partición estrictamente cronológica (*Time-Based Splitting*: `Train` 2014–2017, `Val` 2018, `Test` 2019), **XGBoost** redujo el **RMSE en un 23,25%** (`55,89` vs `72,82` pax/15min) y el **MAE en un 32,34%** (`22,20` vs `32,80` pax/15min) respecto a la Regresión Lineal en el año de prueba (2019), logrando **$R^2 = 0,9504$**.
4. **Métricas Operativas Derivadas (Fase 8):**
   - El modelo predice el Ratio de Retención por estación con un error medio de solo **0,58 puntos porcentuales**, captura el desplazamiento horario de los picos (*peak shift* hacia la franja vespertina) y clasifica las 44 estaciones en **4 Categorías Operativas según Presión de Andén**.

---

## 📁 Estructura del Repositorio

```text
UGR-TP-3ra-entrega/
├── data_processed/                     # Datasets procesados en formato Parquet
│   ├── subte_consolidado_15min.parquet # Demanda agregada cada 15 min (2014-2019)
│   ├── subte_diario_estaciones.parquet # Demanda diaria consolidada por estación
│   └── subte_modelado_15min.parquet    # Matriz equidistante con lags y geolocalización (7.14M filas)
├── output/
│   ├── graficos/                       # Figuras analíticas y de modelado en alta resolución (300 DPI)
│   │   ├── 01_curva_demanda_dia_semana.png
│   │   ├── 02_curva_demanda_estacional_mes.png
│   │   ├── 03_perfil_horario_por_linea.png
│   │   ├── 04_comparativa_habil_nolab_estaciones_clave.png
│   │   ├── 05_kde_habil_vs_nolaborable.png
│   │   ├── 06_presion_anden_ventanas_criticas.png
│   │   ├── 07_comparativa_prediccion_real_vs_modelos.png
│   │   ├── 08_evaluacion_errores_e_importancia_variables.png
│   │   └── 09_categorizacion_estaciones_presion_predicha.png
│   ├── tablas/                         # Tablas tabuladas en Markdown (Fases 3 a 8)
│   │   ├── 01_resumen_nulos_limpieza.md
│   │   ├── 02_estadisticas_descriptivas.md
│   │   ├── 03_ratios_retencion_linea.md
│   │   ├── 04_ratios_retencion_top_bottom_estaciones.md
│   │   ├── 05_resultados_kolmogorov_smirnov_44_estaciones.md
│   │   ├── 06_presion_anden_analisis.md
│   │   ├── 07_auditoria_limpieza_modelado.md
│   │   ├── 08_comparativa_modelos_metricas.md
│   │   └── 09_metricas_operativas_predichas.md
│   └── resumen_metodologia.md          # Síntesis interpretativa para el informe
├── src/                                # Módulos Python del pipeline
│   ├── preprocesamiento.py             # Ingesta por bloques, sanitización y normalización
│   ├── eda_visualizaciones.py          # Generador de gráficos EDA (Figuras 01 a 06)
│   ├── validacion_estadistica.py       # Test K-S, ratios de retención y presión de andén histórica
│   ├── estaciones_geo.py               # Coordenadas geolocalizadas y metadatos espaciales (44 estaciones)
│   ├── preparacion_modelado.py         # Grilla equidistante, limpieza de cierres/outliers, lags y split temporal
│   ├── modelado_evaluacion.py          # Regresión Lineal vs XGBoost, evaluación MSE/RMSE/MAE y Fase 8
│   └── generar_notebook.py             # Generador del Jupyter Notebook reproducible
├── notebook_eda_subte.ipynb            # Jupyter Notebook completo (EDA + Modelado + Evaluación)
├── run_analysis.py                     # Script orquestador end-to-end (Pasos 1 a 5)
└── requirements.txt                    # Dependencias del proyecto
```

---

## 🚀 Ejecución del Pipeline

```bash
# Ejecutar todo el pipeline (EDA + Preparación Modelado + Entrenamiento y Evaluación):
python run_analysis.py

# Ejecutar únicamente la preparación para modelado y entrenamiento (Pasos 4 y 5):
python run_analysis.py --only-modeling
```
