import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & intro
cells.append(nbf.v4.new_markdown_cell("""# Análisis y Modelado Predictivo de la Demanda del Subte de CABA (Líneas A, B y C)
### Preprocesamiento Masivo, EDA, Validación Estadística, Ingeniería de Series Temporales y Aprendizaje Supervisado (2014 - 2019)
**Grupo 34 — Licenciatura en Ciencia de Datos (UGR)**  
**Proyecto:** Variación de la Demanda en el Subte de Buenos Aires en Días Hábiles y No Laborables

---
## 1. Introducción y Objetivos Metodológicos
El presente cuaderno reproducible desarrolla de principio a fin las **Fases 1 a 8 de la metodología CRISP-DM adaptada a series temporales**, evaluando y modelando la demanda de pasajeros en la Red de Subterráneos de la Ciudad Autónoma de Buenos Aires (Líneas A, B y C y sus 44 estaciones) entre **días hábiles** y **no laborables** (sábados, domingos y feriados nacionales) durante el período **2014–2019**:

1. **Preprocesamiento y Sanitización (Fases 1 a 3):** Ingesta de más de 70 millones de registros de molinete, normalización de esquemas y unificación de las 44 estaciones.
2. **Análisis Exploratorio de Datos y Comparación de Distribuciones (Fase 4):** Curvas de demanda, perfiles horarios, KDE y prueba no paramétrica de **Kolmogorov-Smirnov (`ks_2samp`)** estación por estación.
3. **Limpieza para Series Temporales e Ingeniería de Características (Fases 3 y 5):** Filtrado de horario comercial (`05:00`–`23:45`), detección de cierres estructurales de estación, regularización de grilla equidistante de 15 min, acotamiento de outliers de telemetría, construcción de rezagos autoregresivos (`lag_1`, `lag_2`, `lag_4`, `lag_1d`, `lag_7d`, `media_movil_1h`), variables armónicas y geolocalización de estaciones.
4. **Modelado y Evaluación Temporal (Fases 6 y 7):** Partición cronológica (*Time-Based Splitting*: `Train` 2014–2017, `Val` 2018, `Test` 2019) y comparación entre **Regresión Lineal** y **Gradient Boosting (`XGBoost`)** mediante **MSE, RMSE y MAE** en horizontes *One-Step* (+15 min) y *Multi-Step* (+30 min y +60 min).
5. **Métricas Operativas Derivadas (Fase 8):** Cálculo de Ratio de Retención Predicho, Desplazamiento de Horas Pico y Categorización Operativa de las 44 estaciones por Presión de Andén."""))

# Imports
cells.append(nbf.v4.new_code_cell("""# Librerías esenciales
import os
import glob
import unicodedata
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 150
pd.set_option('display.max_columns', 25)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print("Entorno configurado correctamente.")"""))

# Data loading / inspection
cells.append(nbf.v4.new_markdown_cell("""## 2. Carga y Consolidación de Datos (2014–2019)
Para optimizar el uso de memoria RAM frente a los 6.7 GB de archivos CSV brutos, se recurre al dataset consolidado a nivel de estación e intervalo de 15 minutos (`data_processed/subte_consolidado_15min.parquet`) y a la agregación diaria (`data_processed/subte_diario_estaciones.parquet`)."""))

cells.append(nbf.v4.new_code_cell("""# Carga de datasets optimizados
df_15min_path = os.path.join('data_processed', 'subte_consolidado_15min.parquet')
df_diario_path = os.path.join('data_processed', 'subte_diario_estaciones.parquet')

df_15min = pd.read_parquet(df_15min_path)
df_diario = pd.read_parquet(df_diario_path)

# Conversión a escala continua horaria (05.00 a 23.75)
parts = df_15min['ventana_15min'].str.split(':', expand=True)
df_15min['hora_decimal'] = parts[0].astype(float) + parts[1].astype(float) / 60.0
df_service = df_15min[(df_15min['hora_decimal'] >= 5.0) & (df_15min['hora_decimal'] <= 23.75)].copy()

print(f"Registros en dataset 15 min: {len(df_15min):,}")
print(f"Registros en dataset diario: {len(df_diario):,}")
print(f"Estaciones únicas analizadas: {df_15min['estacion'].nunique()}")
display(df_15min.head())"""))

# Descriptive statistics
cells.append(nbf.v4.new_markdown_cell("""## 3. Estadísticas Descriptivas de la Demanda Diaria
Analizamos las medidas de posición (media, mediana) y dispersión (desvío estándar, rango intercuartílico IQR) desagregadas por línea y tipo de día."""))

cells.append(nbf.v4.new_code_cell("""desc_diario = df_diario.groupby(['linea', 'tipo_dia'])['ingresos'].agg(
    Media='mean',
    Desv_Std='std',
    Mediana='median',
    Q25=lambda x: np.percentile(x, 25),
    Q75=lambda x: np.percentile(x, 75),
    Total_Pax='sum'
).reset_index()
desc_diario['IQR'] = desc_diario['Q75'] - desc_diario['Q25']
display(desc_diario)"""))

# EDA 1: Perfil horario por día de la semana
cells.append(nbf.v4.new_markdown_cell("""## 4. Análisis Exploratorio de Datos (EDA) — Fase 4
### 4.1. Curva de Demanda por Día de la Semana y Detección de Horas Pico
Visualización en 2 paneles: Panel A para días hábiles (Lunes a Viernes) con resaltado preciso de horas pico matutina y vespertina; Panel B para fines de semana (Sábado y Domingo) contrastados contra la referencia hábil."""))

cells.append(nbf.v4.new_code_cell("""dias_habiles = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
hourly_day = df_service.groupby(['dia_semana_nombre', 'hora_decimal'])['ingresos'].mean().reset_index()
hourly_habil_mean = df_service[df_service['tipo_dia'] == 'Hábil'].groupby('hora_decimal')['ingresos'].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

# Panel 1: Hábiles
ax1.axvspan(7.5, 9.5, color='#e74c3c', alpha=0.18, label='Pico Matutino (07:30 - 09:30)')
ax1.axvspan(17.0, 19.5, color='#e67e22', alpha=0.18, label='Pico Vespertino (17:00 - 19:30)')

habiles_colors = {'Lunes': '#2980b9', 'Martes': '#27ae60', 'Miércoles': '#8e44ad', 'Jueves': '#16a085', 'Viernes': '#d35400'}
for d in dias_habiles:
    sub = hourly_day[hourly_day['dia_semana_nombre'] == d].sort_values(by='hora_decimal')
    ax1.plot(sub['hora_decimal'], sub['ingresos'], label=d, color=habiles_colors[d], linewidth=1.5, alpha=0.85)

ax1.plot(hourly_habil_mean['hora_decimal'], hourly_habil_mean['ingresos'], label='Promedio Hábil Global', color='#1a252f', linewidth=2.5)
ax1.set_title('A. Días Hábiles: Régimen Bimodal Pendular', fontsize=12, fontweight='bold')
ax1.set_xlabel('Hora del Día')
ax1.set_ylabel('Ingresos Promedio por Estación (pax / intervalo 15 min)')
ax1.set_xlim(5.0, 23.75)
hours_ticks = np.arange(5, 24, 1)
ax1.set_xticks(hours_ticks)
ax1.set_xticklabels([f'{int(h):02d}:00' for h in hours_ticks], rotation=45, ha='right')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right', fontsize=8.5)

# Panel 2: Fines de semana
ax2.plot(hourly_habil_mean['hora_decimal'], hourly_habil_mean['ingresos'], label='Referencia Hábil', color='#95a5a6', linewidth=2.0, linestyle=':')
sub_sab = hourly_day[hourly_day['dia_semana_nombre'] == 'Sábado'].sort_values(by='hora_decimal')
sub_dom = hourly_day[hourly_day['dia_semana_nombre'] == 'Domingo'].sort_values(by='hora_decimal')
ax2.plot(sub_sab['hora_decimal'], sub_sab['ingresos'], label='Sábado', color='#e67e22', linewidth=2.2)
ax2.plot(sub_dom['hora_decimal'], sub_dom['ingresos'], label='Domingo', color='#9b59b6', linewidth=2.2)
ax2.axvspan(13.0, 19.0, color='#9b59b6', alpha=0.12, label='Meseta No Laboral (13:00 - 19:00)')

ax2.set_title('B. Fines de Semana: Régimen Unimodal / Meseta', fontsize=12, fontweight='bold')
ax2.set_xlabel('Hora del Día')
ax2.set_xlim(5.0, 23.75)
ax2.set_xticks(hours_ticks)
ax2.set_xticklabels([f'{int(h):02d}:00' for h in hours_ticks], rotation=45, ha='right')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right', fontsize=8.5)

plt.suptitle('Perfil Horario de Demanda por Día de la Semana (Líneas A, B y C - 2014-2019)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# EDA 2: Estacionalidad mensual
cells.append(nbf.v4.new_markdown_cell("""### 4.2. Curvas de Demanda por Mes (Estacionalidad Anual)
Comportamiento de la demanda agregada a lo largo de los doce meses del año."""))

cells.append(nbf.v4.new_code_cell("""meses_orden = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
monthly_line = df_diario.groupby(['mes', 'mes_nombre', 'linea'])['ingresos'].mean().reset_index().sort_values(by=['linea', 'mes'])
monthly_line['mes_idx'] = monthly_line['mes'] - 1
line_colors = {'Línea A': '#009ee0', 'Línea B': '#e31b23', 'Línea C': '#005baa'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
for l in sorted(monthly_line['linea'].unique()):
    sub = monthly_line[monthly_line['linea'] == l]
    ax1.plot(sub['mes_idx'], sub['ingresos'], marker='o', linewidth=2.5, label=l, color=line_colors.get(l))

ax1.set_title('Promedio Diario por Estación según Mes')
ax1.set_xlabel('Mes')
ax1.set_ylabel('Ingresos Promedio Diarios (pax / día)')
ax1.set_xticks(range(12))
ax1.set_xticklabels(meses_orden, rotation=45, ha='right')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

monthly_total = df_diario.groupby(['mes', 'mes_nombre'])['ingresos'].mean().reset_index().sort_values(by='mes')
monthly_total['mes_idx'] = monthly_total['mes'] - 1
ax2.bar(monthly_total['mes_idx'], monthly_total['ingresos'], color='#2b5c8f', alpha=0.85, edgecolor='black')
mean_val = monthly_total['ingresos'].mean()
ax2.axhline(mean_val, color='red', linestyle='--', label=f'Media Anual ({mean_val:,.0f} pax/día)')
ax2.set_title('Estacionalidad Mensual Red Consolidada (A, B, C)')
ax2.set_xlabel('Mes')
ax2.set_ylabel('Ingresos Promedio Diarios (pax / día)')
ax2.set_xticks(range(12))
ax2.set_xticklabels(meses_orden, rotation=45, ha='right')
ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
ax2.legend()
plt.tight_layout()
plt.show()"""))

# EDA 3: Perfil por línea y estaciones clave
cells.append(nbf.v4.new_markdown_cell("""### 4.3. Comparativa Hábil vs No Laborable en Estaciones Representativas
Evaluación del contraste en cuatro nodos neurálgicos: Constitución, Retiro, Carlos Pellegrini y San Pedrito."""))

cells.append(nbf.v4.new_code_cell("""estaciones_clave = ['Constitución', 'Retiro', 'Carlos Pellegrini', 'San Pedrito']
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
axes = axes.flatten()

for i, est in enumerate(estaciones_clave):
    ax = axes[i]
    sub = df_service[df_service['estacion'] == est]
    st_agg = sub.groupby(['tipo_dia', 'hora_decimal'])['ingresos'].mean().reset_index()
    
    habil = st_agg[st_agg['tipo_dia'] == 'Hábil'].sort_values(by='hora_decimal')
    nolab = st_agg[st_agg['tipo_dia'] == 'No Laborable'].sort_values(by='hora_decimal')
    
    ax.plot(habil['hora_decimal'], habil['ingresos'], label='Hábil', color='#2ca02c', linewidth=2.5)
    ax.plot(nolab['hora_decimal'], nolab['ingresos'], label='No Laborable', color='#d62728', linewidth=2.5, linestyle='--')
    ax.fill_between(habil['hora_decimal'], habil['ingresos'], alpha=0.15, color='#2ca02c')
    ax.fill_between(nolab['hora_decimal'], nolab['ingresos'], alpha=0.15, color='#d62728')
    
    ax.set_title(f'Estación {est} ({sub["linea"].iloc[0]})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Promedio pax / 15 min')
    ax.set_xlim(5.0, 23.75)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    if i >= 2:
        ax.set_xlabel('Hora del Día')
        step_ticks = np.arange(6, 24, 2)
        ax.set_xticks(step_ticks)
        ax.set_xticklabels([f'{int(h):02d}:00' for h in step_ticks], rotation=45, ha='right')

plt.suptitle('Demanda Horaria en Estaciones Clave (Hábil vs No Laborable)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# Retention metrics
cells.append(nbf.v4.new_markdown_cell("""## 5. Ratios de Retención de Demanda Histórica
El Ratio de Retención mide la proporción de demanda habitual que persiste en días no laborables:
$$\\text{Ratio de Retención (\\%)} = \\frac{\\overline{\\text{Ingresos Diarios}}_{\\text{No Laborable}}}{\\overline{\\text{Ingresos Diarios}}_{\\text{Hábil}}} \\times 100$$"""))

cells.append(nbf.v4.new_code_cell("""# Retención por línea
ret_linea = df_diario.groupby(['linea', 'tipo_dia'])['ingresos'].mean().unstack()
ret_linea['Ratio_Retencion_%'] = (ret_linea['No Laborable'] / ret_linea['Hábil']) * 100
ret_linea['Caida_Demanda_%'] = 100 - ret_linea['Ratio_Retencion_%']
display(ret_linea)"""))

cells.append(nbf.v4.new_code_cell("""# Retención por estación: Top 5 vs Bottom 5
ret_est = df_diario.groupby(['linea', 'estacion', 'tipo_dia'])['ingresos'].mean().unstack().reset_index()
ret_est['Ratio_Retencion_%'] = (ret_est['No Laborable'] / ret_est['Hábil']) * 100
ret_est['Caida_Demanda_%'] = 100 - ret_est['Ratio_Retencion_%']
ret_est.sort_values(by='Ratio_Retencion_%', ascending=False, inplace=True)

print("=== TOP 5 ESTACIONES CON MAYOR RETENCIÓN ===")
display(ret_est.head(5))

print("\\n=== TOP 5 ESTACIONES CON MAYOR CAÍDA DE DEMANDA ===")
display(ret_est.tail(5).iloc[::-1])"""))

# KDE
cells.append(nbf.v4.new_markdown_cell("""## 6. Validación Estadística de Heterogeneidad (Sección 2.5 y Fase 4)
### 6.1. Estimación de Densidad por Kernel (KDE)
Contraste de formas de distribución en estaciones representativas."""))

cells.append(nbf.v4.new_code_cell("""kde_estaciones = ['Constitución', 'Carlos Pellegrini', 'Carlos Gardel', 'Retiro']
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.flatten()

for i, est in enumerate(kde_estaciones):
    ax = axes[i]
    sub = df_service[df_service['estacion'] == est]
    habil_data = sub[sub['tipo_dia'] == 'Hábil']['ingresos']
    nolab_data = sub[sub['tipo_dia'] == 'No Laborable']['ingresos']
    
    sns.kdeplot(habil_data, ax=ax, label='Hábil', color='#1f77b4', fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(nolab_data, ax=ax, label='No Laborable', color='#ff7f0e', fill=True, alpha=0.35, linewidth=2)
    
    ax.set_title(f'KDE Ingresos: {est}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Ingresos (pasajeros / 15 min)')
    ax.set_ylabel('Densidad')
    ax.set_xlim(left=0, right=np.percentile(habil_data, 99.5))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

plt.suptitle('Estimación de Densidad por Kernel (KDE): Días Hábiles vs No Laborables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# Kolmogorov-Smirnov Test
cells.append(nbf.v4.new_markdown_cell("""### 6.2. Prueba de Kolmogorov-Smirnov (K-S 2 Muestras) para las 44 Estaciones
- **$H_0$:** La distribución empírica de demanda por 15 min es idéntica entre Días Hábiles y No Laborables.
- **$H_1$:** Las distribuciones son distintas (heterogeneidad).
- **$\\alpha = 0.05$**."""))

cells.append(nbf.v4.new_code_cell("""ks_records = []
estaciones_unicas = df_15min[['linea', 'estacion']].drop_duplicates().sort_values(by=['linea', 'estacion'])

for _, row in estaciones_unicas.iterrows():
    l, st = row['linea'], row['estacion']
    s_habil = df_15min[(df_15min['estacion'] == st) & (df_15min['tipo_dia'] == 'Hábil')]['ingresos'].values
    s_nolab = df_15min[(df_15min['estacion'] == st) & (df_15min['tipo_dia'] == 'No Laborable')]['ingresos'].values
    
    res = stats.ks_2samp(s_habil, s_nolab)
    ks_records.append({
        'Línea': l,
        'Estación': st,
        'Estadístico_D': res.statistic,
        'p_valor': res.pvalue,
        'Rechaza_H0': res.pvalue < 0.05
    })

df_ks_res = pd.DataFrame(ks_records)
rechazos = df_ks_res['Rechaza_H0'].sum()
print(f"Total estaciones analizadas: {len(df_ks_res)}")
print(f"Estaciones que rechazan H0: {rechazos} ({(rechazos/len(df_ks_res))*100:.1f}%)")
display(df_ks_res.head(10))"""))

# Fase 3 avanzada y Fase 5: Preparación para el Modelado
cells.append(nbf.v4.new_markdown_cell("""## 7. Limpieza Avanzada para Series Temporales e Ingeniería de Características (Fases 3 y 5)
Para entrenar los modelos supervisados de regresión (**Regresión Lineal** y **Gradient Boosting / XGBoost**) sin inducir sesgos ni fuga de datos (*data leakage*), se ejecuta el módulo `src.preparacion_modelado`:
1. **Recorte de Horario Comercial (`05:00` a `23:45`):** Exclusión de registros nocturnos residuales fuera de servicio (76 ventanas equidistantes por jornada).
2. **Nulos Estructurales vs. Operativos:** Exclusión de jornadas con cierre total de estación por obras o paros ($<20\\text{ pax/día}$) y regularización de la grilla equidistante de 15 minutos completando con `0` las ventanas operativas activas sin validaciones.
3. **Tratamiento de Outliers de Telemetría:** Acotamiento (*winsorización*) de descargas acumuladas de molinete usando umbrales robustos ($\\max(Q_{99.9}, Q_{75} + 4.5\\cdot\\text{IQR})$) estimados exclusivamente sobre el set de entrenamiento.
4. **Ingeniería de Características:** Rezagos autoregresivos (`lag_1`, `lag_2`, `lag_4`, `lag_1d`, `lag_7d`, `media_movil_1h`), variables armónicas (`sin_hora`, `cos_hora`, `sin_dia_semana`, `cos_dia_semana`) y variables espaciales/geolocalizadas (`latitud`, `longitud`, `distancia_centro_km`, `orden_linea`, `es_cabecera`, `es_combinacion`).
5. **Partición Cronológica (*Time-Based Splitting*):** `Train` (2014–2017), `Validación` (2018) y `Test` (2019)."""))

cells.append(nbf.v4.new_code_cell("""from src.preparacion_modelado import prepare_modeling_data

df_modelado_path = os.path.join('data_processed', 'subte_modelado_15min.parquet')
if not os.path.exists(df_modelado_path):
    df_modelado = prepare_modeling_data()
else:
    df_modelado = pd.read_parquet(df_modelado_path)

print(f"Dimensiones de la matriz de modelado: {df_modelado.shape}")
display(df_modelado['particion'].value_counts())
display(df_modelado[['fecha', 'linea', 'estacion', 'ventana_15min', 'ingresos', 'lag_1', 'lag_1d', 'lag_7d', 'distancia_centro_km', 'particion']].head(10))"""))

# Fases 6, 7 y 8: Modelado, Evaluación y Métricas Operativas
cells.append(nbf.v4.new_markdown_cell("""## 8. Entrenamiento, Evaluación Temporal y Métricas Operativas Derivadas (Fases 6, 7 y 8)
Se entrenan y comparan **Regresión Lineal** (con `StandardScaler` ajustado únicamente en `train`) y **Gradient Boosting (`XGBRegressor`)**, evaluando **MSE**, **RMSE** y **MAE** sobre los conjuntos de Validación (2018) y Prueba (2019), tanto en horizonte *One-Step* (+15 min) como *Multi-Step* (+30 min y +60 min). Finalmente, sobre las predicciones en el año de prueba (2019), se calculan el **Ratio de Retención Predicho**, el **Desplazamiento de Horas Pico** y la **Categorización de las 44 Estaciones por Presión de Andén**."""))

cells.append(nbf.v4.new_code_cell("""from src.modelado_evaluacion import run_modeling_and_evaluation

# Ejecuta el entrenamiento/evaluación y exporta las tablas 08 y 09 y los gráficos 07, 08 y 09
run_modeling_and_evaluation()"""))

# Conclusions
cells.append(nbf.v4.new_markdown_cell("""## 9. Conclusiones y Resumen para Metodología
1. **Heterogeneidad Demostrada al 100%:** La prueba de Kolmogorov-Smirnov rechaza $H_0$ en la totalidad de las 44 estaciones ($p < 10^{-15}$), confirmando que las curvas y distribuciones de demanda no laboral difieren estructuralmente del régimen laboral.
2. **Asimetría Territorial y Ratios de Retención:** La retención global del 42,03% enmascara brechas extremas entre el Microcentro (caídas superiores al 70% en Diagonal Norte y Plaza de Mayo) y polos recreativos o de transbordo (Carlos Gardel retiene 70,43%; Constitución retiene casi 35.000 pax/día).
3. **Superioridad del Ensamble No Lineal (XGBoost vs. Regresión Lineal):** El modelo de Gradient Boosting captura las interacciones no lineales entre estación, hora y tipo de día, reduciendo el RMSE en un **23,25%** (55,89 vs. 72,82 pax/15min) y el MAE en un **32,34%** (22,20 vs. 32,80 pax/15min) en el año de prueba 2019, alcanzando un $R^2 = 0,9504$.
4. **Optimización de Frecuencias Basada en Predicciones (Fase 8):** La categorización de las 44 estaciones mediante la presión de andén predicha identifica 8 nodos críticos de saturación en días no laborables y fundamenta la adopción de diagramaciones operativas dinámicas por corredor y franja horaria."""))

nb.cells = cells

with open('notebook_eda_subte.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("notebook_eda_subte.ipynb regenerado con éxito.")
