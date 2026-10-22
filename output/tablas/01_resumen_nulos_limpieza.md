# Resumen de Calidad de Datos, Limpieza y Tratamiento de Nulos

### 1. Ingesta por Año y Detección de Inconsistencias

| Año | Registros Brutos (Molinete) | Registros Líneas A, B, C | Registros Nulos / NA en Ingresos | % Nulos |
| :--- | :---: | :---: | :---: | :---: |
| 2014 | 10,857,244 | 6,542,052 | 0 | 0.000% |
| 2015 | 10,958,582 | 6,584,807 | 0 | 0.000% |
| 2016 | 11,542,322 | 6,812,654 | 0 | 0.000% |
| 2017 | 11,938,476 | 6,971,539 | 0 | 0.000% |
| 2018 | 12,136,454 | 6,986,890 | 0 | 0.000% |
| 2019 | 12,662,343 | 7,188,372 | 0 | 0.000% |
| **Total** | **70,095,421** | **41,086,314** | **0** | **0.000%** |

### 2. Decisiones de Limpieza y Justificación Metodológica

1. **Tratamiento de Nulos y Ceros en Ingresos:** Los registros con valor `NA` corresponden a intervalos de 15 minutos en los cuales un molinete físico particular no registró validaciones de pasajeros (inactividad transitoria o fuera de servicio temporal). Siguiendo las directrices operativas de transporte, fueron imputados con valor `0` numérico antes de realizar la suma agregada a nivel estación, garantizando que el cómputo de la estación refleje el flujo exacto sin sesgar positivamente la media.
2. **Estandarización de Esquemas y Encodings:** Se corrigieron variaciones entre archivos históricos (encodings mixtos `UTF-8` y `Latin-1`, cabeceras en mayúsculas/minúsculas como `PAX_TOTAL` vs `TOTAL`, y formatos de fecha combinados `YYYY-MM-DD` y `DD/MM/YYYY`).
3. **Consolidación de Estaciones (44 estaciones):** Se unificaron variantes de escritura con tildes, caracteres mal codificados (ej. `Sáenz Peña`) y nomenclaturas específicas de transbordo (ej. `Callao.B` $\to$ `Callao (B)`). Se obtuvo una cobertura perfecta del 100% de las 44 estaciones de las Líneas A, B y C.
4. **Agregación por Intervalo de 15 Minutos:** La consolidación de transacciones de molinete en volumen de estación por ventana horaria redujo la cardinalidad de decenas de millones de filas a un dataset estructurado y eficiente de alta performance analítica.
