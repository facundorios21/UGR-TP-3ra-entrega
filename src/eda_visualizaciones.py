import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = os.path.join('output', 'graficos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_time_to_decimal_hour(time_series):
    """Convierte serie de strings 'HH:MM' a float continuo 'hora.decimal' (ej. '08:30' -> 8.5)"""
    parts = time_series.str.split(':', expand=True)
    h = parts[0].astype(float)
    m = parts[1].astype(float)
    return h + m / 60.0

def generate_visualizations():
    print("Iniciando generación de visualizaciones corregidas y optimizadas (300 DPI)...", flush=True)
    
    df_15min_path = os.path.join('data_processed', 'subte_consolidado_15min.parquet')
    df_diario_path = os.path.join('data_processed', 'subte_diario_estaciones.parquet')
    
    df_15min = pd.read_parquet(df_15min_path)
    df_diario = pd.read_parquet(df_diario_path)
    
    # Crear columna temporal numérica continua (hora decimal)
    df_15min['hora_decimal'] = parse_time_to_decimal_hour(df_15min['ventana_15min'])
    
    # Filtrar horario operativo del servicio comercial (05:00 a 23:75)
    df_service = df_15min[(df_15min['hora_decimal'] >= 5.0) & (df_15min['hora_decimal'] <= 23.75)].copy()

    # -------------------------------------------------------------------------
    # FIGURA 1: Curva de Demanda por Día de la Semana y Horas Pico (2 Paneles)
    # -------------------------------------------------------------------------
    print("Generando Figura 1 corregida: Curvas por día de la semana en 2 paneles (Hábiles vs Fin de Semana)...", flush=True)
    
    dias_habiles = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
    dias_fin_semana = ['Sábado', 'Domingo']
    
    # Agrupación por día de la semana y hora continua
    hourly_day = df_service.groupby(['dia_semana_nombre', 'hora_decimal'])['ingresos'].mean().reset_index()
    # Promedio hábil global para referencia
    hourly_habil_mean = df_service[df_service['tipo_dia'] == 'Hábil'].groupby('hora_decimal')['ingresos'].mean().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5), sharey=True)
    
    # --- PANEL 1: DÍAS HÁBILES (Lunes a Viernes) ---
    # Sombreado de horas pico con límites numéricos continuos
    ax1.axvspan(7.5, 9.5, color='#e74c3c', alpha=0.18, label='Pico Matutino (07:30 - 09:30)')
    ax1.axvspan(17.0, 19.5, color='#e67e22', alpha=0.18, label='Pico Vespertino (17:00 - 19:30)')
    
    habiles_colors = {
        'Lunes': '#2980b9',
        'Martes': '#27ae60',
        'Miércoles': '#8e44ad',
        'Jueves': '#16a085',
        'Viernes': '#d35400'
    }
    
    for d in dias_habiles:
        sub = hourly_day[hourly_day['dia_semana_nombre'] == d].sort_values(by='hora_decimal')
        ax1.plot(sub['hora_decimal'], sub['ingresos'], label=d, color=habiles_colors[d], linewidth=1.8, alpha=0.85)
        
    # Curva promedio hábil destacada
    ax1.plot(hourly_habil_mean['hora_decimal'], hourly_habil_mean['ingresos'], 
             label='Promedio Hábil Global', color='#1a252f', linewidth=3.0, linestyle='-')
    
    ax1.set_title('A. Días Hábiles (Lunes a Viernes): Patrón Bimodal Pendular', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Hora del Día', fontsize=11, labelpad=8)
    ax1.set_ylabel('Ingresos Promedio por Estación (pasajeros / intervalo de 15 min)', fontsize=11, labelpad=8)
    ax1.set_xlim(5.0, 23.75)
    
    hours_ticks = np.arange(5, 24, 1)
    hours_labels = [f'{int(h):02d}:00' for h in hours_ticks]
    ax1.set_xticks(hours_ticks)
    ax1.set_xticklabels(hours_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fontsize=9.5)
    
    # Anotaciones de picos
    peak_mat = hourly_habil_mean[(hourly_habil_mean['hora_decimal'] >= 7.5) & (hourly_habil_mean['hora_decimal'] <= 9.5)].sort_values(by='ingresos', ascending=False).iloc[0]
    peak_vesp = hourly_habil_mean[(hourly_habil_mean['hora_decimal'] >= 17.0) & (hourly_habil_mean['hora_decimal'] <= 19.5)].sort_values(by='ingresos', ascending=False).iloc[0]
    ax1.annotate(f'Pico Mañana\n({peak_mat["ingresos"]:.0f} pax/15m)', 
                 xy=(peak_mat['hora_decimal'], peak_mat['ingresos']), 
                 xytext=(peak_mat['hora_decimal'] - 2.2, peak_mat['ingresos'] + 15),
                 arrowprops=dict(facecolor='#c0392b', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#c0392b')
    ax1.annotate(f'Pico Tarde\n({peak_vesp["ingresos"]:.0f} pax/15m)', 
                 xy=(peak_vesp['hora_decimal'], peak_vesp['ingresos']), 
                 xytext=(peak_vesp['hora_decimal'] - 2.5, peak_vesp['ingresos'] + 15),
                 arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#d35400')
    
    # --- PANEL 2: FINES DE SEMANA (Sábado y Domingo) VS HÁBIL ---
    # Línea de referencia del día hábil en gris
    ax2.plot(hourly_habil_mean['hora_decimal'], hourly_habil_mean['ingresos'], 
             label='Referencia Promedio Hábil', color='#95a5a6', linewidth=2.0, linestyle=':')
    
    sub_sab = hourly_day[hourly_day['dia_semana_nombre'] == 'Sábado'].sort_values(by='hora_decimal')
    sub_dom = hourly_day[hourly_day['dia_semana_nombre'] == 'Domingo'].sort_values(by='hora_decimal')
    
    ax2.plot(sub_sab['hora_decimal'], sub_sab['ingresos'], label='Sábado', color='#e67e22', linewidth=2.5)
    ax2.plot(sub_dom['hora_decimal'], sub_dom['ingresos'], label='Domingo', color='#9b59b6', linewidth=2.5)
    
    # Sombreado de la meseta no laboral
    ax2.axvspan(13.0, 19.0, color='#9b59b6', alpha=0.12, label='Meseta de Actividad No Laboral (13:00 - 19:00)')
    
    ax2.set_title('B. Fines de Semana (Sábado y Domingo): Desaparición de Picos Laborales', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Hora del Día', fontsize=11, labelpad=8)
    ax2.set_xlim(5.0, 23.75)
    ax2.set_xticks(hours_ticks)
    ax2.set_xticklabels(hours_labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True, fontsize=9.5)
    
    plt.suptitle('Perfil Horario de Ingresos por Día de la Semana - Subte de CABA (Líneas A, B y C | 2014-2019)', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, '01_curva_demanda_dia_semana.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado exitoso: {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGURA 2: Curvas de Demanda por Mes (Estacionalidad Anual)
    # -------------------------------------------------------------------------
    print("Generando Figura 2: Estacionalidad mensual de la demanda...", flush=True)
    
    meses_orden = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    monthly_line = df_diario.groupby(['mes', 'mes_nombre', 'linea'])['ingresos'].mean().reset_index()
    monthly_line['mes_idx'] = monthly_line['mes'] - 1
    
    line_colors = {'Línea A': '#009ee0', 'Línea B': '#e31b23', 'Línea C': '#005baa'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: por línea
    for l in sorted(monthly_line['linea'].unique()):
        sub = monthly_line[monthly_line['linea'] == l].sort_values(by='mes')
        ax1.plot(sub['mes_idx'], sub['ingresos'], marker='o', linewidth=2.5, label=l, color=line_colors.get(l, None))
        
    ax1.set_title('Promedio Diario de Ingresos por Estación según Mes', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Mes del Año', fontsize=11)
    ax1.set_ylabel('Ingresos Promedio Diarios por Estación (pax / día)', fontsize=11)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(meses_orden, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', frameon=True)
    
    # Right: total red agregada
    monthly_total = df_diario.groupby(['mes', 'mes_nombre'])['ingresos'].mean().reset_index().sort_values(by='mes')
    monthly_total['mes_idx'] = monthly_total['mes'] - 1
    ax2.bar(monthly_total['mes_idx'], monthly_total['ingresos'], color='#2b5c8f', alpha=0.85, edgecolor='black')
    ax2.set_title('Estacionalidad Mensual Red Consolidada (Líneas A, B y C)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Mes del Año', fontsize=11)
    ax2.set_ylabel('Ingresos Promedio Diarios por Estación (pax / día)', fontsize=11)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(meses_orden, rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    mean_val = monthly_total['ingresos'].mean()
    ax2.axhline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Media Anual ({mean_val:,.0f} pax/día)')
    ax2.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, '02_curva_demanda_estacional_mes.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Guardado: {fig2_path}")

    # -------------------------------------------------------------------------
    # FIGURA 3: Perfil Horario Desagregado por Línea (Hábil vs No Laborable)
    # -------------------------------------------------------------------------
    print("Generando Figura 3: Perfil horario por línea continuo...", flush=True)
    
    hourly_line_type = df_service.groupby(['linea', 'tipo_dia', 'hora_decimal'])['ingresos'].mean().reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    lines = ['Línea A', 'Línea B', 'Línea C']
    
    for i, l in enumerate(lines):
        ax = axes[i]
        sub_habil = hourly_line_type[(hourly_line_type['linea'] == l) & (hourly_line_type['tipo_dia'] == 'Hábil')].sort_values(by='hora_decimal')
        sub_nolab = hourly_line_type[(hourly_line_type['linea'] == l) & (hourly_line_type['tipo_dia'] == 'No Laborable')].sort_values(by='hora_decimal')
        
        ax.plot(sub_habil['hora_decimal'], sub_habil['ingresos'], label='Día Hábil', color='#1f77b4', linewidth=2.5)
        ax.plot(sub_nolab['hora_decimal'], sub_nolab['ingresos'], label='Día No Laborable', color='#ff7f0e', linewidth=2.5, linestyle='--')
        
        ax.fill_between(sub_habil['hora_decimal'], sub_habil['ingresos'], alpha=0.15, color='#1f77b4')
        ax.fill_between(sub_nolab['hora_decimal'], sub_nolab['ingresos'], alpha=0.15, color='#ff7f0e')
        
        ax.set_title(f'{l}', fontsize=14, fontweight='bold', color=line_colors.get(l, 'black'))
        ax.set_xlabel('Hora del Día', fontsize=11)
        if i == 0:
            ax.set_ylabel('Ingresos Promedio por Estación (pax / 15 min)', fontsize=11)
            
        ax.set_xlim(5.0, 23.75)
        step_ticks = np.arange(6, 24, 2)
        ax.set_xticks(step_ticks)
        ax.set_xticklabels([f'{int(h):02d}:00' for h in step_ticks], rotation=45, ha='right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True)
        
    plt.suptitle('Perfil Horario de Demanda (15 min): Días Hábiles vs No Laborables por Línea', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, '03_perfil_horario_por_linea.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {fig3_path}")

    # -------------------------------------------------------------------------
    # FIGURA 4: Comparativa Hábil vs No Laborable en Estaciones Clave
    # -------------------------------------------------------------------------
    print("Generando Figura 4: Comparativa en estaciones representativas...", flush=True)
    
    estaciones_clave = ['Constitución', 'Retiro', 'Carlos Pellegrini', 'San Pedrito']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
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
        
        linea_est = sub['linea'].iloc[0]
        ax.set_title(f'Estación {est} ({linea_est})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Ingresos Promedio (pax / 15 min)', fontsize=11)
        ax.set_xlim(5.0, 23.75)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True)
        
        if i >= 2:
            ax.set_xlabel('Hora del Día', fontsize=11)
            step_ticks = np.arange(6, 24, 2)
            ax.set_xticks(step_ticks)
            ax.set_xticklabels([f'{int(h):02d}:00' for h in step_ticks], rotation=45, ha='right', fontsize=9)
            
    plt.suptitle('Demanda Horaria en Estaciones Clave y Nodos de Transferencia (Hábil vs No Laborable)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, '04_comparativa_habil_nolab_estaciones_clave.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Guardado: {fig4_path}")

    # -------------------------------------------------------------------------
    # FIGURA 5: Estimación de Densidad por Kernel (KDE)
    # -------------------------------------------------------------------------
    print("Generando Figura 5: Gráficos de KDE comparando distribuciones...", flush=True)
    
    kde_estaciones = ['Constitución', 'Carlos Pellegrini', 'Carlos Gardel', 'Retiro']
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    
    for i, est in enumerate(kde_estaciones):
        ax = axes[i]
        sub = df_service[df_service['estacion'] == est]
        habil_data = sub[sub['tipo_dia'] == 'Hábil']['ingresos']
        nolab_data = sub[sub['tipo_dia'] == 'No Laborable']['ingresos']
        
        sns.kdeplot(habil_data, ax=ax, label='Hábil', color='#1f77b4', fill=True, alpha=0.35, linewidth=2)
        sns.kdeplot(nolab_data, ax=ax, label='No Laborable', color='#ff7f0e', fill=True, alpha=0.35, linewidth=2)
        
        ax.set_title(f'Distribución de Ingresos (KDE): {est}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Ingresos por Intervalo (pasajeros / 15 min)', fontsize=11)
        ax.set_ylabel('Densidad Estimada', fontsize=11)
        ax.set_xlim(left=0, right=np.percentile(habil_data, 99.5))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True)
        
    plt.suptitle('Estimación de Densidad por Kernel (KDE): Contraste de Formas de Distribución', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig5_path = os.path.join(OUTPUT_DIR, '05_kde_habil_vs_nolaborable.png')
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Guardado: {fig5_path}")

    # -------------------------------------------------------------------------
    # FIGURA 6: Presión de Andén y Ventanas Críticas
    # -------------------------------------------------------------------------
    print("Generando Figura 6: Análisis de presión de andén continuo...", flush=True)
    
    congested_stations = ['Constitución', 'Retiro', 'Federico Lacroze', 'Carlos Gardel', 'Carlos Pellegrini']
    sub_presion = df_service[(df_service['tipo_dia'] == 'No Laborable') & (df_service['estacion'].isin(congested_stations))]
    
    hourly_presion = sub_presion.groupby(['estacion', 'hora_decimal'])['ingresos'].agg(
        P95=lambda x: np.percentile(x, 95),
        Maximo='max'
    ).reset_index()
    hourly_presion = hourly_presion[(hourly_presion['hora_decimal'] >= 7.0) & (hourly_presion['hora_decimal'] <= 23.0)]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    UMBRAL = 1200
    ax.axhline(UMBRAL, color='red', linestyle='--', linewidth=2, label=f'Capacidad Nominal de Despacho ({UMBRAL} pax / 15 min)')
    
    station_palette = sns.color_palette("Set1", len(congested_stations))
    for i, est in enumerate(congested_stations):
        sub_st = hourly_presion[hourly_presion['estacion'] == est].sort_values(by='hora_decimal')
        ax.plot(sub_st['hora_decimal'], sub_st['P95'], marker='o', markersize=3, label=f'{est} (Percentil 95)', color=station_palette[i], linewidth=2)
        
    ax.set_title('Presión de Andén en Días No Laborables: Demanda P95 vs Capacidad Nominal de Formaciones', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Hora del Día', fontsize=12, labelpad=10)
    ax.set_ylabel('Afluencia de Pasajeros (pasajeros / 15 min)', fontsize=12, labelpad=10)
    ax.set_xlim(7.0, 23.0)
    
    step_ticks = np.arange(7, 24, 1)
    ax.set_xticks(step_ticks)
    ax.set_xticklabels([f'{int(h):02d}:00' for h in step_ticks], rotation=45, ha='right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    fig6_path = os.path.join(OUTPUT_DIR, '06_presion_anden_ventanas_criticas.png')
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Guardado: {fig6_path}")
    print("Todas las visualizaciones regeneradas con éxito.")

if __name__ == '__main__':
    generate_visualizations()
