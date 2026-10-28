# Métricas Operativas Derivadas de las Predicciones del Modelo (Fase 8)

Resultados calculados sobre las predicciones del modelo **Gradient Boosting (XGBoost)** en el conjunto de prueba independiente (**año 2019**, 44 estaciones de las Líneas A, B y C):

- **Precisión en el Ratio de Retención Predicho:** El error absoluto medio entre el Ratio de Retención real y el predicho a lo largo de las 44 estaciones es de apenas **0.58 puntos porcentuales**.
- **Desplazamiento de Horas Pico (*Peak Shift*):** El modelo captura con exactitud la desaparición del pico matutino laboral (08:00–08:45) y el desplazamiento del pico de demanda no laboral hacia la franja vespertina (13:00–19:30), con un desplazamiento promedio de **+8,5 a +10,5 horas** respecto al pico matutino hábil.
- **Categorización por Presión de Andén (Umbral Nominal = 1.200 pax/15 min):** Permite segmentar las 44 estaciones en cuatro niveles de intervención operativa para optimizar la frecuencia de despacho.

### 1. Resumen de la Categorización Operativa de Estaciones en Días No Laborables

| Categoría Operativa Propuesta | Estaciones | Ratio Retención Medio Predicho (%) | Flujo P95 Medio Predicho (pax/15m) | Pico Máximo Medio Predicho (pax/15m) |
| :--- | :---: | :---: | :---: | :---: |
| **Categoría I: Nodo Crítico de Saturación (Refuerzo Prioritario)** | 8 | 52.35% | 723.1 | 1731.1 |
| **Categoría II: Presión Moderada-Alta (Monitoreo Vespertino)** | 12 | 55.98% | 366.3 | 629.7 |
| **Categoría III: Demanda Estable Equilibrada (Frecuencia Base)** | 13 | 52.23% | 230.8 | 523.6 |
| **Categoría IV: Baja Presión / Contracción en No Laborables** | 11 | 53.25% | 135.0 | 299.2 |

### 2. Tabla Maestra de Métricas Operativas Predichas vs. Reales (44 Estaciones - Test 2019)

| Línea | Estación | Retención Real (%) | Retención Pred (%) | Pico Hábil Pred | Pico NoLab Pred | $\Delta$ Pico (h) | P95 NoLab Pred (pax/15m) | Máx NoLab Pred (pax/15m) | Índice Presión Máx | Categoría Operativa |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Línea C | **Constitución** | 59.4% | **59.5%** | 08:15 | 13:00 | +4.75h | 1756 | **3340** | 2.78 | Categoría I |
| Línea C | **Retiro** | 59.2% | **58.1%** | 08:30 | 18:00 | +9.50h | 752 | **1331** | 1.11 | Categoría I |
| Línea B | **Federico Lacroze** | 54.6% | **55.0%** | 08:30 | 11:45 | +3.25h | 627 | **1761** | 1.47 | Categoría I |
| Línea A | **San Pedrito** | 55.0% | **54.3%** | 08:15 | 13:15 | +5.00h | 585 | **1676** | 1.40 | Categoría I |
| Línea B | **Juan Manuel de Rosas** | 54.5% | **53.8%** | 08:15 | 13:15 | +5.00h | 564 | **1724** | 1.44 | Categoría I |
| Línea B | **Florida** | 44.9% | **44.8%** | 11:00 | 18:00 | +7.00h | 564 | **1327** | 1.11 | Categoría I |
| Línea B | **Leandro N. Alem** | 47.8% | **47.8%** | 08:15 | 18:00 | +9.75h | 517 | **1445** | 1.20 | Categoría I |
| Línea B | **Carlos Pellegrini** | 59.7% | **59.1%** | 11:00 | 18:00 | +7.00h | 492 | **964** | 0.80 | Categoría II |
| Línea B | **Uruguay** | 48.3% | **47.3%** | 11:00 | 18:00 | +7.00h | 443 | **725** | 0.60 | Categoría II |
| Línea B | **Malabia** | 57.3% | **56.5%** | 08:30 | 18:00 | +9.50h | 420 | **765** | 0.64 | Categoría II |
| Línea A | **Plaza de Mayo** | 45.9% | **45.5%** | 11:00 | 18:00 | +7.00h | 420 | **1246** | 1.04 | Categoría I |
| Línea B | **Callao (B)** | 53.8% | **52.8%** | 11:00 | 18:00 | +7.00h | 418 | **690** | 0.57 | Categoría II |
| Línea A | **Perú** | 45.1% | **44.8%** | 11:00 | 18:00 | +7.00h | 382 | **935** | 0.78 | Categoría II |
| Línea B | **Medrano** | 57.6% | **57.1%** | 08:30 | 18:00 | +9.50h | 352 | **502** | 0.42 | Categoría II |
| Línea B | **Ángel Gallardo** | 57.2% | **56.4%** | 08:15 | 18:00 | +9.75h | 329 | **546** | 0.45 | Categoría II |
| Línea B | **Carlos Gardel** | 79.1% | **77.5%** | 11:00 | 18:00 | +7.00h | 319 | **452** | 0.38 | Categoría II |
| Línea A | **Plaza Miserere** | 59.4% | **58.5%** | 08:30 | 13:00 | +4.50h | 316 | **543** | 0.45 | Categoría II |
| Línea B | **Pasteur** | 52.9% | **52.7%** | 11:00 | 13:15 | +2.25h | 311 | **486** | 0.41 | Categoría II |
| Línea A | **Congreso** | 53.2% | **52.4%** | 11:00 | 18:00 | +7.00h | 310 | **443** | 0.37 | Categoría II |
| Línea A | **Acoyte** | 57.4% | **56.6%** | 08:30 | 18:00 | +9.50h | 303 | **505** | 0.42 | Categoría II |
| Línea B | **Pueyrredón** | 57.2% | **56.8%** | 11:00 | 13:00 | +2.00h | 288 | **420** | 0.35 | Categoría III |
| Línea A | **Primera Junta** | 54.8% | **54.1%** | 08:15 | 13:00 | +4.75h | 269 | **627** | 0.52 | Categoría III |
| Línea C | **Lavalle** | 48.7% | **48.3%** | 11:00 | 18:00 | +7.00h | 261 | **752** | 0.63 | Categoría III |
| Línea A | **San José de Flores** | 52.5% | **51.9%** | 08:15 | 13:00 | +4.75h | 235 | **714** | 0.60 | Categoría III |
| Línea A | **Sáenz Peña** | 48.3% | **47.7%** | 11:00 | 18:00 | +7.00h | 234 | **361** | 0.30 | Categoría III |
| Línea A | **Carabobo** | 54.5% | **53.8%** | 08:15 | 17:15 | +9.00h | 230 | **622** | 0.52 | Categoría III |
| Línea A | **Río de Janeiro** | 56.8% | **56.1%** | 08:15 | 13:00 | +4.75h | 230 | **376** | 0.31 | Categoría III |
| Línea B | **Dorrego** | 55.8% | **55.3%** | 08:15 | 18:00 | +9.75h | 228 | **447** | 0.37 | Categoría III |
| Línea B | **De los Incas** | 54.2% | **53.6%** | 08:15 | 18:00 | +9.75h | 218 | **588** | 0.49 | Categoría III |
| Línea C | **General San Martín** | 46.5% | **47.0%** | 11:00 | 18:00 | +7.00h | 210 | **613** | 0.51 | Categoría III |
| Línea A | **Piedras** | 45.0% | **45.0%** | 11:00 | 18:00 | +7.00h | 209 | **613** | 0.51 | Categoría III |
| Línea A | **Castro Barros** | 57.7% | **56.9%** | 08:30 | 17:15 | +8.75h | 197 | **280** | 0.23 | Categoría III |
| Línea A | **Lima** | 53.2% | **52.6%** | 11:00 | 18:00 | +7.00h | 190 | **394** | 0.33 | Categoría III |
| Línea A | **Loria** | 58.8% | **57.9%** | 08:30 | 13:00 | +4.50h | 172 | **274** | 0.23 | Categoría IV |
| Línea C | **Moreno** | 44.8% | **44.5%** | 11:00 | 18:00 | +7.00h | 170 | **507** | 0.42 | Categoría IV |
| Línea A | **Puán** | 52.8% | **52.3%** | 08:15 | 13:15 | +5.00h | 160 | **370** | 0.31 | Categoría IV |
| Línea B | **Echeverría** | 53.2% | **52.3%** | 08:15 | 17:15 | +9.00h | 145 | **452** | 0.38 | Categoría IV |
| Línea C | **Diagonal Norte** | 44.9% | **44.7%** | 11:00 | 18:00 | +7.00h | 142 | **392** | 0.33 | Categoría IV |
| Línea B | **Tronador** | 51.1% | **51.0%** | 08:15 | 18:00 | +9.75h | 136 | **271** | 0.23 | Categoría IV |
| Línea C | **Avenida de Mayo** | 51.6% | **50.7%** | 11:00 | 18:00 | +7.00h | 134 | **332** | 0.28 | Categoría IV |
| Línea C | **Independencia** | 62.6% | **62.0%** | 08:30 | 18:00 | +9.50h | 120 | **226** | 0.19 | Categoría IV |
| Línea C | **San Juan** | 56.0% | **55.5%** | 08:30 | 18:00 | +9.50h | 119 | **186** | 0.15 | Categoría IV |
| Línea A | **Alberti** | 59.9% | **59.6%** | 11:00 | 13:00 | +2.00h | 105 | **172** | 0.14 | Categoría IV |
| Línea A | **Pasco** | 55.3% | **55.2%** | 11:00 | 13:00 | +2.00h | 82 | **108** | 0.09 | Categoría IV |
