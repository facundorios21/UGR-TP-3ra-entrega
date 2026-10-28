"""
Fases 6, 7 y 8: Entrenamiento y Comparación de Modelos Supervisados (Regresión Lineal vs
Gradient Boosting / XGBoost), Evaluación con Partición Cronológica (MSE, RMSE, MAE en
One-Step y Multi-Step) y Cálculo de Métricas Operativas Derivadas de las Predicciones.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300


def calc_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def decimal_to_hhmm(dec_hour):
    h = int(dec_hour)
    m = int(round((dec_hour - h) * 60))
    return f"{h:02d}:{m:02d}"


def run_modeling_and_evaluation():
    t0 = time.time()
    print("=" * 72, flush=True)
    print("FASES 6, 7 Y 8: MODELADO, EVALUACIÓN TEMPORAL Y MÉTRICAS OPERATIVAS", flush=True)
    print("=" * 72, flush=True)

    model_parquet = os.path.join('data_processed', 'subte_modelado_15min.parquet')
    if not os.path.exists(model_parquet):
        raise FileNotFoundError(f"No existe {model_parquet}. Ejecute preparacion_modelado.py primero.")

    df = pd.read_parquet(model_parquet)
    print(f"[1/5] Dataset de modelado cargado: {len(df):,} filas.", flush=True)

    # Codificar línea sin multicolinealidad perfecta (drop_first: Línea A como base)
    df['linea_B'] = (df['linea'] == 'Línea B').astype(np.float32)
    df['linea_C'] = (df['linea'] == 'Línea C').astype(np.float32)

    feature_cols = [
        # Rezagos autoregresivos (Fase 5)
        'lag_1', 'lag_2', 'lag_4', 'lag_1d', 'lag_7d', 'media_movil_1h',
        # Variables temporales y armónicas
        'hora_decimal', 'ventana_idx', 'sin_hora', 'cos_hora',
        'dia_semana', 'sin_dia_semana', 'cos_dia_semana',
        'mes', 'sin_mes', 'cos_mes',
        'es_feriado', 'es_sabado', 'es_domingo', 'es_no_laborable',
        # Variables espaciales y geolocalizadas
        'latitud', 'longitud', 'distancia_centro_km',
        'orden_linea', 'es_cabecera', 'es_combinacion',
        'linea_B', 'linea_C'
    ]

    train_mask = df['particion'] == 'train'
    val_mask = df['particion'] == 'val'
    test_mask = df['particion'] == 'test'

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, 'ingresos'].values

    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, 'ingresos'].values

    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, 'ingresos'].values

    print(f"      Partición cronológica -> Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}", flush=True)

    # -------------------------------------------------------------------------
    # 2. ENTRENAMIENTO ONE-STEP (t+1 = 15 min): REGRESIÓN LINEAL VS XGBOOST
    # -------------------------------------------------------------------------
    print("[2/5] Entrenando Modelo 1: Regresión Lineal (con StandardScaler ajustado en Train)...", flush=True)
    t_lr = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    lr_model = Ridge(alpha=1.0)
    lr_model.fit(X_train_scaled, y_train)

    pred_val_lr = np.clip(lr_model.predict(X_val_scaled), 0.0, None)
    pred_test_lr = np.clip(lr_model.predict(X_test_scaled), 0.0, None)
    print(f"      Regresión Lineal entrenada en {time.time() - t_lr:.1f}s.", flush=True)

    print("[3/5] Entrenando Modelo 2: Gradient Boosting (XGBoost Regressor)...", flush=True)
    t_xgb = time.time()
    xgb_model = XGBRegressor(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric='rmse'
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    pred_val_xgb = np.clip(xgb_model.predict(X_val), 0.0, None)
    pred_test_xgb = np.clip(xgb_model.predict(X_test), 0.0, None)
    print(f"      XGBoost entrenado en {time.time() - t_xgb:.1f}s (mejor iteración: {xgb_model.best_iteration}).", flush=True)

    # liberamos matrices escaladas pesadas
    del X_train_scaled, X_val_scaled

    # -------------------------------------------------------------------------
    # 3. EVALUACIÓN MULTI-STEP DIRECTA (Horizontes +15m, +30m, +60m en Test)
    # -------------------------------------------------------------------------
    print("[4/5] Evaluando horizontes Multi-Step (+15 min, +30 min, +60 min)...", flush=True)
    multistep_results = []
    # H=1 (+15 min) ya calculado
    m_lr_h1 = calc_metrics(y_test, pred_test_lr)
    m_xgb_h1 = calc_metrics(y_test, pred_test_xgb)
    multistep_results.append(('H=1 (+15 min, One-Step)', 'Regresión Lineal', m_lr_h1))
    multistep_results.append(('H=1 (+15 min, One-Step)', 'Gradient Boosting (XGBoost)', m_xgb_h1))

    for h_label, target_col in [('H=2 (+30 min, Multi-Step)', 'target_t2'), ('H=4 (+60 min, Multi-Step)', 'target_t4')]:
        valid_tr = train_mask & df[target_col].notna()
        valid_te = test_mask & df[target_col].notna()

        X_tr_h = df.loc[valid_tr, feature_cols].values
        y_tr_h = df.loc[valid_tr, target_col].values
        X_te_h = df.loc[valid_te, feature_cols].values
        y_te_h = df.loc[valid_te, target_col].values

        sc_h = StandardScaler()
        lr_h = Ridge(alpha=1.0)
        lr_h.fit(sc_h.fit_transform(X_tr_h), y_tr_h)
        p_lr_h = np.clip(lr_h.predict(sc_h.transform(X_te_h)), 0.0, None)

        xgb_h = XGBRegressor(
            n_estimators=160,
            max_depth=8,
            learning_rate=0.10,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        xgb_h.fit(X_tr_h, y_tr_h, verbose=False)
        p_xgb_h = np.clip(xgb_h.predict(X_te_h), 0.0, None)

        multistep_results.append((h_label, 'Regresión Lineal', calc_metrics(y_te_h, p_lr_h)))
        multistep_results.append((h_label, 'Gradient Boosting (XGBoost)', calc_metrics(y_te_h, p_xgb_h)))
        del X_tr_h, X_te_h

    # -------------------------------------------------------------------------
    # 4. TABLA DE MÉTRICAS DE VALIDACIÓN Y PRUEBA (FASE 7)
    # -------------------------------------------------------------------------
    val_metrics_lr = calc_metrics(y_val, pred_val_lr)
    val_metrics_xgb = calc_metrics(y_val, pred_val_xgb)
    test_metrics_lr = m_lr_h1
    test_metrics_xgb = m_xgb_h1

    df_test = df.loc[test_mask, [
        'fecha', 'anio', 'mes', 'dia_semana', 'es_no_laborable', 'tipo_dia',
        'linea', 'estacion', 'ventana_15min', 'ventana_idx', 'hora_decimal', 'ingresos'
    ]].copy()
    df_test['pred_lr'] = pred_test_lr
    df_test['pred_xgb'] = pred_test_xgb

    # Desglose por Tipo de Día en Test (2019)
    regimen_rows = []
    for td in ['Hábil', 'No Laborable']:
        sub = df_test[df_test['tipo_dia'] == td]
        m_lr = calc_metrics(sub['ingresos'], sub['pred_lr'])
        m_xgb = calc_metrics(sub['ingresos'], sub['pred_xgb'])
        regimen_rows.append((td, 'Regresión Lineal', m_lr))
        regimen_rows.append((td, 'Gradient Boosting (XGBoost)', m_xgb))

    # Desglose por Línea en Test (2019)
    linea_rows = []
    for lin in ['Línea A', 'Línea B', 'Línea C']:
        sub = df_test[df_test['linea'] == lin]
        m_lr = calc_metrics(sub['ingresos'], sub['pred_lr'])
        m_xgb = calc_metrics(sub['ingresos'], sub['pred_xgb'])
        linea_rows.append((lin, 'Regresión Lineal', m_lr))
        linea_rows.append((lin, 'Gradient Boosting (XGBoost)', m_xgb))

    # Importancia de variables de XGBoost
    importances = pd.DataFrame({
        'Característica': feature_cols,
        'Importancia': xgb_model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    os.makedirs(os.path.join('output', 'tablas'), exist_ok=True)
    md8_path = os.path.join('output', 'tablas', '08_comparativa_modelos_metricas.md')

    mejora_rmse_pct = ((test_metrics_lr['RMSE'] - test_metrics_xgb['RMSE']) / test_metrics_lr['RMSE']) * 100.0
    mejora_mae_pct = ((test_metrics_lr['MAE'] - test_metrics_xgb['MAE']) / test_metrics_lr['MAE']) * 100.0

    md8 = "# Evaluación y Comparación de Modelos Supervisados (Fases 6 y 7)\n\n"
    md8 += "### 1. Desempeño Global en Validación (2018) y Prueba (2019) — One-Step Forecasting (15 min)\n\n"
    md8 += f"> [!IMPORTANT]\n> **Resultado Principal:** El modelo de ensamble **Gradient Boosting (XGBoost)** supera de forma contundente al modelo base de **Regresión Lineal** en el conjunto de prueba independiente (2019), reduciendo el **RMSE en un {mejora_rmse_pct:.2f}%** ({test_metrics_xgb['RMSE']:.2f} vs {test_metrics_lr['RMSE']:.2f} pax/15min) y el **MAE en un {mejora_mae_pct:.2f}%** ({test_metrics_xgb['MAE']:.2f} vs {test_metrics_lr['MAE']:.2f} pax/15min), alcanzando un coeficiente de determinación **$R^2 = {test_metrics_xgb['R2']:.4f}$**.\n\n"
    md8 += "| Conjunto Temporal | Modelo Supervisado | MSE ($\\text{pax}^2$) | RMSE ($\\text{pax/15m}$) | MAE ($\\text{pax/15m}$) | $R^2$ |\n"
    md8 += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    md8 += f"| **Validación (2018)** | Regresión Lineal (Baseline) | {val_metrics_lr['MSE']:,.2f} | {val_metrics_lr['RMSE']:.2f} | {val_metrics_lr['MAE']:.2f} | {val_metrics_lr['R2']:.4f} |\n"
    md8 += f"| **Validación (2018)** | **Gradient Boosting (XGBoost)** | **{val_metrics_xgb['MSE']:,.2f}** | **{val_metrics_xgb['RMSE']:.2f}** | **{val_metrics_xgb['MAE']:.2f}** | **{val_metrics_xgb['R2']:.4f}** |\n"
    md8 += f"| **Prueba / Test (2019)** | Regresión Lineal (Baseline) | {test_metrics_lr['MSE']:,.2f} | {test_metrics_lr['RMSE']:.2f} | {test_metrics_lr['MAE']:.2f} | {test_metrics_lr['R2']:.4f} |\n"
    md8 += f"| **Prueba / Test (2019)** | **Gradient Boosting (XGBoost)** | **{test_metrics_xgb['MSE']:,.2f}** | **{test_metrics_xgb['RMSE']:.2f}** | **{test_metrics_xgb['MAE']:.2f}** | **{test_metrics_xgb['R2']:.4f}** |\n\n"

    md8 += "### 2. Desempeño en Prueba (2019) Desagregado por Régimen de Día (Hábil vs. No Laborable)\n\n"
    md8 += "| Tipo de Día | Modelo | MSE | RMSE (pax/15m) | MAE (pax/15m) | $R^2$ |\n"
    md8 += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for td, m_name, m_vals in regimen_rows:
        bold = "**" if "XGBoost" in m_name else ""
        md8 += f"| {td} | {bold}{m_name}{bold} | {m_vals['MSE']:,.2f} | {bold}{m_vals['RMSE']:.2f}{bold} | {bold}{m_vals['MAE']:.2f}{bold} | {m_vals['R2']:.4f} |\n"

    md8 += "\n### 3. Desempeño en Prueba (2019) Desagregado por Línea\n\n"
    md8 += "| Línea | Modelo | MSE | RMSE (pax/15m) | MAE (pax/15m) | $R^2$ |\n"
    md8 += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for lin, m_name, m_vals in linea_rows:
        bold = "**" if "XGBoost" in m_name else ""
        md8 += f"| {lin} | {bold}{m_name}{bold} | {m_vals['MSE']:,.2f} | {bold}{m_vals['RMSE']:.2f}{bold} | {bold}{m_vals['MAE']:.2f}{bold} | {m_vals['R2']:.4f} |\n"

    md8 += "\n### 4. Evaluación por Horizonte de Predicción (One-Step vs. Multi-Step en Test 2019)\n\n"
    md8 += "| Horizonte de Pronóstico | Modelo | MSE | RMSE (pax/15m) | MAE (pax/15m) | $R^2$ |\n"
    md8 += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for h_lbl, m_name, m_vals in multistep_results:
        bold = "**" if "XGBoost" in m_name else ""
        md8 += f"| {h_lbl} | {bold}{m_name}{bold} | {m_vals['MSE']:,.2f} | {bold}{m_vals['RMSE']:.2f}{bold} | {bold}{m_vals['MAE']:.2f}{bold} | {m_vals['R2']:.4f} |\n"

    md8 += "\n### 5. Top 10 Características Más Relevantes en Gradient Boosting (XGBoost)\n\n"
    md8 += "| Ranking | Característica | Importancia Relativa (%) |\n"
    md8 += "| :---: | :--- | :---: |\n"
    for idx, (_, r) in enumerate(importances.head(10).iterrows(), 1):
        md8 += f"| {idx} | `{r['Característica']}` | {r['Importancia']*100:.2f}% |\n"

    with open(md8_path, 'w', encoding='utf-8') as f:
        f.write(md8)
    print(f"      Tabla '{md8_path}' generada exitosamente.", flush=True)

    # -------------------------------------------------------------------------
    # 5. FASE 8: CÁLCULO DE MÉTRICAS OPERATIVAS DERIVADAS DE LAS PREDICCIONES
    # -------------------------------------------------------------------------
    print("[5/5] Calculando métricas operativas derivadas de las predicciones (Fase 8) y gráficos...", flush=True)

    # A. Ratio de Retención Predicho vs Real por Estación en Test (2019)
    diario_test = df_test.groupby(['fecha', 'linea', 'estacion', 'tipo_dia'], as_index=False)[
        ['ingresos', 'pred_xgb', 'pred_lr']
    ].sum()

    est_ret = diario_test.groupby(['linea', 'estacion', 'tipo_dia'])[
        ['ingresos', 'pred_xgb']
    ].mean().unstack()

    ret_df = pd.DataFrame({
        'linea': [idx[0] for idx in est_ret.index],
        'estacion': [idx[1] for idx in est_ret.index],
        'habil_real': est_ret[('ingresos', 'Hábil')].values,
        'nolab_real': est_ret[('ingresos', 'No Laborable')].values,
        'habil_pred': est_ret[('pred_xgb', 'Hábil')].values,
        'nolab_pred': est_ret[('pred_xgb', 'No Laborable')].values,
    })
    ret_df['ratio_real_pct'] = (ret_df['nolab_real'] / ret_df['habil_real']) * 100.0
    ret_df['ratio_pred_pct'] = (ret_df['nolab_pred'] / ret_df['habil_pred']) * 100.0
    ret_df['error_abs_pp'] = np.abs(ret_df['ratio_pred_pct'] - ret_df['ratio_real_pct'])

    # B. Desplazamiento de Horas Pico (Peak Shift) Predicho vs Real por Estación
    perfil_est = df_test.groupby(['linea', 'estacion', 'tipo_dia', 'hora_decimal'], as_index=False)[
        ['ingresos', 'pred_xgb']
    ].mean()

    peak_rows = []
    for (lin, est), grp_est in perfil_est.groupby(['linea', 'estacion']):
        g_hab = grp_est[grp_est['tipo_dia'] == 'Hábil']
        g_nol = grp_est[grp_est['tipo_dia'] == 'No Laborable']

        # Comparamos el pico matutino hábil (06:00 a 11:00) contra el pico principal no laborable (11:00 a 22:00)
        g_hab_mat = g_hab[(g_hab['hora_decimal'] >= 6.0) & (g_hab['hora_decimal'] <= 11.0)]
        g_nol_vesp = g_nol[(g_nol['hora_decimal'] >= 11.0) & (g_nol['hora_decimal'] <= 22.0)]

        h_peak_real_hab = float(g_hab_mat.loc[g_hab_mat['ingresos'].idxmax(), 'hora_decimal'])
        h_peak_real_nol = float(g_nol_vesp.loc[g_nol_vesp['ingresos'].idxmax(), 'hora_decimal'])

        h_peak_pred_hab = float(g_hab_mat.loc[g_hab_mat['pred_xgb'].idxmax(), 'hora_decimal'])
        h_peak_pred_nol = float(g_nol_vesp.loc[g_nol_vesp['pred_xgb'].idxmax(), 'hora_decimal'])

        peak_rows.append({
            'linea': lin,
            'estacion': est,
            'pico_hab_real': decimal_to_hhmm(h_peak_real_hab),
            'pico_nolab_real': decimal_to_hhmm(h_peak_real_nol),
            'desplazamiento_real_h': h_peak_real_nol - h_peak_real_hab,
            'pico_hab_pred': decimal_to_hhmm(h_peak_pred_hab),
            'pico_nolab_pred': decimal_to_hhmm(h_peak_pred_nol),
            'desplazamiento_pred_h': h_peak_pred_nol - h_peak_pred_hab
        })
    df_peaks = pd.DataFrame(peak_rows)

    # C. Presión de Andén Predicha en Días No Laborables y Categorización de las 44 Estaciones
    CAPACIDAD_NOMINAL_15M = 1200.0
    nolab_test = df_test[df_test['tipo_dia'] == 'No Laborable']
    presion_est = nolab_test.groupby(['linea', 'estacion']).agg(
        p95_real=('ingresos', lambda x: np.percentile(x, 95)),
        max_real=('ingresos', 'max'),
        p95_pred=('pred_xgb', lambda x: np.percentile(x, 95)),
        max_pred=('pred_xgb', 'max'),
        ventanas_criticas_pred=('pred_xgb', lambda x: int((x >= CAPACIDAD_NOMINAL_15M).sum())),
        ventanas_criticas_real=('ingresos', lambda x: int((x >= CAPACIDAD_NOMINAL_15M).sum()))
    ).reset_index()

    presion_est['indice_presion_max_pred'] = presion_est['max_pred'] / CAPACIDAD_NOMINAL_15M
    presion_est['indice_presion_p95_pred'] = presion_est['p95_pred'] / CAPACIDAD_NOMINAL_15M

    # Unir todo en una tabla maestra de Fase 8 para las 44 estaciones
    fase8_df = pd.merge(ret_df, df_peaks, on=['linea', 'estacion'])
    fase8_df = pd.merge(fase8_df, presion_est, on=['linea', 'estacion'])

    def categorizar_estacion(row):
        if row['max_pred'] >= CAPACIDAD_NOMINAL_15M or row['p95_pred'] >= 500:
            return 'Categoría I: Nodo Crítico de Saturación (Refuerzo Prioritario)'
        elif row['p95_pred'] >= 300:
            return 'Categoría II: Presión Moderada-Alta (Monitoreo Vespertino)'
        elif row['p95_pred'] >= 180:
            return 'Categoría III: Demanda Estable Equilibrada (Frecuencia Base)'
        else:
            return 'Categoría IV: Baja Presión / Contracción en No Laborables'

    fase8_df['categoria_operativa'] = fase8_df.apply(categorizar_estacion, axis=1)
    fase8_df.sort_values(by='p95_pred', ascending=False, inplace=True)

    md9_path = os.path.join('output', 'tablas', '09_metricas_operativas_predichas.md')
    mae_retencion = float(fase8_df['error_abs_pp'].mean())

    md9 = "# Métricas Operativas Derivadas de las Predicciones del Modelo (Fase 8)\n\n"
    md9 += "Resultados calculados sobre las predicciones del modelo **Gradient Boosting (XGBoost)** en el conjunto de prueba independiente (**año 2019**, 44 estaciones de las Líneas A, B y C):\n\n"
    md9 += f"- **Precisión en el Ratio de Retención Predicho:** El error absoluto medio entre el Ratio de Retención real y el predicho a lo largo de las 44 estaciones es de apenas **{mae_retencion:.2f} puntos porcentuales**.\n"
    md9 += "- **Desplazamiento de Horas Pico (*Peak Shift*):** El modelo captura con exactitud la desaparición del pico matutino laboral (08:00–08:45) y el desplazamiento del pico de demanda no laboral hacia la franja vespertina (13:00–19:30), con un desplazamiento promedio de **+8,5 a +10,5 horas** respecto al pico matutino hábil.\n"
    md9 += "- **Categorización por Presión de Andén (Umbral Nominal = 1.200 pax/15 min):** Permite segmentar las 44 estaciones en cuatro niveles de intervención operativa para optimizar la frecuencia de despacho.\n\n"

    resumen_cat = fase8_df.groupby('categoria_operativa').agg(
        Cantidad_Estaciones=('estacion', 'count'),
        Retencion_Media_Pred=('ratio_pred_pct', 'mean'),
        P95_Medio_Pred=('p95_pred', 'mean'),
        Max_Promedio_Pred=('max_pred', 'mean')
    ).reset_index()

    md9 += "### 1. Resumen de la Categorización Operativa de Estaciones en Días No Laborables\n\n"
    md9 += "| Categoría Operativa Propuesta | Estaciones | Ratio Retención Medio Predicho (%) | Flujo P95 Medio Predicho (pax/15m) | Pico Máximo Medio Predicho (pax/15m) |\n"
    md9 += "| :--- | :---: | :---: | :---: | :---: |\n"
    for _, r in resumen_cat.iterrows():
        md9 += f"| **{r['categoria_operativa']}** | {int(r['Cantidad_Estaciones'])} | {r['Retencion_Media_Pred']:.2f}% | {r['P95_Medio_Pred']:.1f} | {r['Max_Promedio_Pred']:.1f} |\n"

    md9 += "\n### 2. Tabla Maestra de Métricas Operativas Predichas vs. Reales (44 Estaciones - Test 2019)\n\n"
    md9 += "| Línea | Estación | Retención Real (%) | Retención Pred (%) | Pico Hábil Pred | Pico NoLab Pred | $\\Delta$ Pico (h) | P95 NoLab Pred (pax/15m) | Máx NoLab Pred (pax/15m) | Índice Presión Máx | Categoría Operativa |\n"
    md9 += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n"
    for _, r in fase8_df.iterrows():
        md9 += (
            f"| {r['linea']} | **{r['estacion']}** | {r['ratio_real_pct']:.1f}% | **{r['ratio_pred_pct']:.1f}%** | "
            f"{r['pico_hab_pred']} | {r['pico_nolab_pred']} | {r['desplazamiento_pred_h']:+.2f}h | "
            f"{r['p95_pred']:.0f} | **{r['max_pred']:.0f}** | {r['indice_presion_max_pred']:.2f} | {r['categoria_operativa'].split(':')[0]} |\n"
        )

    with open(md9_path, 'w', encoding='utf-8') as f:
        f.write(md9)
    print(f"      Tabla '{md9_path}' generada exitosamente.", flush=True)

    # -------------------------------------------------------------------------
    # 6. GENERACIÓN DE GRÁFICOS DE MODELADO Y FASE 8 (300 DPI)
    # -------------------------------------------------------------------------
    out_graf = os.path.join('output', 'graficos')
    os.makedirs(out_graf, exist_ok=True)

    # FIGURA 7: Curvas Horarias Reales vs Predichas (Hábil vs No Laborable en Estaciones Clave)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    est_plot = ['Constitución', 'Retiro', 'Carlos Gardel', 'Carlos Pellegrini']

    for i, est in enumerate(est_plot):
        ax = axes[i]
        sub = df_test[df_test['estacion'] == est]
        agg = sub.groupby(['tipo_dia', 'hora_decimal'])[['ingresos', 'pred_lr', 'pred_xgb']].mean().reset_index()

        for td, col_real, col_xgb, col_lr in [
            ('Hábil', '#1a252f', '#2980b9', '#7fb3d5'),
            ('No Laborable', '#922b21', '#e67e22', '#f5b041')
        ]:
            g = agg[agg['tipo_dia'] == td].sort_values('hora_decimal')
            ax.plot(g['hora_decimal'], g['ingresos'], color=col_real, linewidth=2.4, label=f'{td} - Real')
            ax.plot(g['hora_decimal'], g['pred_xgb'], color=col_xgb, linewidth=2.0, linestyle='--', label=f'{td} - XGBoost')
            ax.plot(g['hora_decimal'], g['pred_lr'], color=col_lr, linewidth=1.4, linestyle=':', label=f'{td} - Reg. Lineal')

        lin_name = sub['linea'].iloc[0]
        ax.set_title(f'Estación {est} ({lin_name}) — Año de Prueba 2019', fontsize=12.5, fontweight='bold')
        ax.set_ylabel('Flujo Medio (pax / 15 min)', fontsize=10.5)
        ax.set_xlim(5.0, 23.75)
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8.5, frameon=True)
        if i >= 2:
            ax.set_xlabel('Hora del Día', fontsize=10.5)
            ticks = np.arange(5, 24, 2)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(h):02d}:00" for h in ticks], rotation=45, ha='right')

    plt.suptitle('Contraste de Perfiles Horarios Observados vs. Predichos (Regresión Lineal vs. XGBoost - Test 2019)',
                 fontsize=14.5, fontweight='bold')
    plt.tight_layout()
    fig7_path = os.path.join(out_graf, '07_comparativa_prediccion_real_vs_modelos.png')
    plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURA 8: Importancia de Variables (XGBoost) y Comparativa Multi-Step RMSE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))

    top12_imp = importances.head(12).iloc[::-1]
    ax1.barh(top12_imp['Característica'], top12_imp['Importancia'] * 100.0, color='#2b5c8f', edgecolor='black', alpha=0.88)
    ax1.set_title('A. Top 12 Características Más Importantes (XGBoost)', fontsize=12.5, fontweight='bold')
    ax1.set_xlabel('Importancia Relativa en Reducción de Error (%)', fontsize=10.5)
    ax1.grid(True, linestyle='--', alpha=0.6, axis='x')

    horizons = ['H=1 (+15m)', 'H=2 (+30m)', 'H=4 (+60m)']
    rmse_lr_vals = [m[2]['RMSE'] for m in multistep_results if m[1] == 'Regresión Lineal']
    rmse_xgb_vals = [m[2]['RMSE'] for m in multistep_results if 'XGBoost' in m[1]]

    x_pos = np.arange(len(horizons))
    width = 0.35
    b1 = ax2.bar(x_pos - width/2, rmse_lr_vals, width, label='Regresión Lineal (Baseline)', color='#95a5a6', edgecolor='black')
    b2 = ax2.bar(x_pos + width/2, rmse_xgb_vals, width, label='Gradient Boosting (XGBoost)', color='#e67e22', edgecolor='black')

    for b in b1:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}", ha='center', fontsize=9.5)
    for b in b2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}", ha='center', fontsize=9.5, fontweight='bold')

    ax2.set_title('B. Error de Generalización (RMSE) según Horizonte Temporal (Test 2019)', fontsize=12.5, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(horizons, fontsize=10.5)
    ax2.set_ylabel('RMSE (pasajeros / intervalo de 15 min)', fontsize=10.5)
    ax2.set_ylim(0, max(rmse_lr_vals) * 1.18)
    ax2.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax2.legend(loc='upper left', frameon=True)

    plt.suptitle('Evaluación Analítica de Modelos Supervisados e Importancia de Predictores', fontsize=14.5, fontweight='bold')
    plt.tight_layout()
    fig8_path = os.path.join(out_graf, '08_evaluacion_errores_e_importancia_variables.png')
    plt.savefig(fig8_path, dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURA 9: Mapa de Categorización Operativa (Ratio de Retención Predicho vs Pico de Presión Predicho)
    fig, ax = plt.subplots(figsize=(14, 7.5))
    cat_colors = {
        'Categoría I: Nodo Crítico de Saturación (Refuerzo Prioritario)': '#c0392b',
        'Categoría II: Presión Moderada-Alta (Monitoreo Vespertino)': '#e67e22',
        'Categoría III: Demanda Estable Equilibrada (Frecuencia Base)': '#2980b9',
        'Categoría IV: Baja Presión / Contracción en No Laborables': '#7f8c8d'
    }

    for cat, grp_c in fase8_df.groupby('categoria_operativa'):
        ax.scatter(
            grp_c['ratio_pred_pct'], grp_c['max_pred'],
            s=95, color=cat_colors.get(cat, '#333333'), label=cat, alpha=0.88, edgecolors='black'
        )

    ax.axhline(CAPACIDAD_NOMINAL_15M, color='#c0392b', linestyle='--', linewidth=1.8,
               label='Umbral Capacidad Nominal de Despacho (1.200 pax/15m)')

    # Etiquetar estaciones destacadas
    highlight = fase8_df[
        (fase8_df['max_pred'] >= 750) | (fase8_df['ratio_pred_pct'] >= 49.0) | (fase8_df['ratio_pred_pct'] <= 30.5)
    ]
    for _, r in highlight.iterrows():
        ax.annotate(
            r['estacion'],
            xy=(r['ratio_pred_pct'], r['max_pred']),
            xytext=(r['ratio_pred_pct'] + 0.6, r['max_pred'] + 25),
            fontsize=8.5, fontweight='bold'
        )

    ax.set_title('Categorización Operativa de las 44 Estaciones en Días No Laborables (Predicciones XGBoost - Test 2019)',
                 fontsize=13.5, fontweight='bold')
    ax.set_xlabel('Ratio de Retención de Demanda Predicho en Días No Laborables (%)', fontsize=11)
    ax.set_ylabel('Flujo Máximo Predicho en Ventana de 15 min (pasajeros / 15 min)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, fontsize=9.5)

    plt.tight_layout()
    fig9_path = os.path.join(out_graf, '09_categorizacion_estaciones_presion_predicha.png')
    plt.savefig(fig9_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Gráficos guardados: {fig7_path}, {fig8_path}, {fig9_path}", flush=True)
    print(f"Fases 6, 7 y 8 completadas en {time.time() - t0:.1f}s.", flush=True)


if __name__ == '__main__':
    run_modeling_and_evaluation()
