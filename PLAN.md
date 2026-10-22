
### 📋 Prompt Actualizado para el Agente (con Preprocesamiento y EDA extendido):

```text
Actúa como un Senior Data Scientist y Analista Operativo de Transporte Público. Tu objetivo es ejecutar el Preprocesamiento Básico, un Análisis Exploratorio de Datos (EDA) completo y la Validación Estadística sobre los archivos CSV adjuntos, correspondientes a los registros de ingresos por molinete de la Red de Subterráneos de la CABA (Líneas A, B y C).

Este análisis forma parte de un proyecto de investigación que evalúa la "Variación de la Demanda en el Subte de CABA en Días Hábiles y No Laborables" para demostrar analíticamente la heterogeneidad de la demanda y fundamentar la optimización de los cronogramas de servicio.

---

### 1. PREPROCESAMIENTO BÁSICO Y SANITIZACIÓN DE DATOS

1. **Estandarización de Columnas:**
   - Renombra y estandariza las columnas a formato snake_case en español: `fecha`, `estacion`, `linea`, `ventana_15min`, `ingresos`, etc.
2. **Estandarización de Valores y Nombres:**
   - Verifica y unifica la codificación de nombres de estaciones y líneas (evita duplicados por tildes, mayúsculas/minúsculas o espacios extra, ej. 'Plaza de Mayo', 'Linea A').
   - Normaliza nombres de lugares y nodos de transbordo.
3. **Tratamiento de Nulos y Limpieza:**
   - Analiza y documenta la presencia de valores nulos, registros faltantes o intervalos sin validaciones.
   - Aplica una estrategia explicada de imputación (ej. imputación por media/mediana según franja horaria y tipo de día) o eliminación justificada de registros erróneos/duplicados.
4. **Ingeniería de Variables Temporales:**
   - Genera las variables: `dia_semana` (Nombre y número 0-6), `mes` (Nombre y número 1-12), `hora`, `ventana_15min` (0 a 95 o HH:MM), `es_feriado` (0 o 1) y `tipo_dia` ('Hábil' vs 'No Laborable').

---

### 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)

#### A. Curvas y Perfiles de Demanda
Genera visualizaciones claras y estilizadas (Seaborn/Matplotlib) para los siguientes ejes:
1. **Curvas de Demanda por Día de la Semana:**
   - Grafica el volumen medio de ingresos para cada día (Lunes a Domingo).
   - **Visualización de Horas Pico:** Resalta claramente las franjas de hora pico matutina y vespertina en días hábiles frente a los momentos de mayor afluencia en fines de semana.
2. **Curvas de Demanda por Mes:**
   - Muestra la evolución y estacionalidad mensual de los ingresos agregados para identificar variaciones a lo largo del año.
3. **Curvas de Demanda Desagregadas por Línea y por Estación:**
   - Perfil horario promedio (15 min) desglosado de forma independiente para cada línea (Línea A, B y C).
   - Curvas comparativas de demanda 'Hábil' vs 'No Laborable' para estaciones representativas y de alta transferencia (ej. Constitución, Retiro, Carlos Pellegrini, San Pedrito).

#### B. Métrica de Retención de Demanda
1. **Ratio de Retención (%):**
   - Calcula el Ratio = (Promedio Ingresos Diarios No Laborable / Promedio Ingresos Diarios Hábil) * 100.
2. **Desagregación:**
   - Presenta una tabla con el Ratio de Retención global por **Línea**.
   - Presenta la tabla completa del Ratio de Retención por **Estación** (Top 5 estaciones con mayor retención vs Top 5 con mayor caída de demanda).

#### C. Validación Estadística e Indicadores Operativos
1. **Comparación de Distribuciones (KDE):**
   - Grafica la Estimación de Densidad por Kernel (KDE) comparando 'Hábil' vs 'No Laborable' en estaciones clave.
2. **Prueba Analítica de Kolmogorov-Smirnov (K-S de 2 Muestras):**
   - Ejecuta `scipy.stats.ks_2samp` independientemente para cada una de las 44 estaciones.
   - H0: La distribución de demanda es idéntica entre tipo de día en esa estación.
   - Reporta la tabla de resultados (`Línea`, `Estación`, `Estadístico_D`, `p_valor`, `Rechaza_H0` a alpha=0.05) y el porcentaje total de estaciones heterogéneas.
3. **Presión de Andén:**
   - Identifica ventanas horarias de 15 min donde los ingresos por molinete superen la capacidad nominal de despacho de formaciones en días no laborables.

---

### 3. ENTREGABLES REQUERIDOS

1. **Código Python ejecutable** modular y comentado para Google Colab / Jupyter.
2. **Tablas tabuladas en Markdown** con las estadísticas descriptivas, tratamiento de nulos, ratios de retención y resultados del test K-S.
3. **Gráficos en alta resolución (PNG)** etiquetados correctamente.
4. **Resumen Interpretativo (3-4 párrafos)** integrando los hallazgos para la sección de Metodología de la entrega escrita.
```

