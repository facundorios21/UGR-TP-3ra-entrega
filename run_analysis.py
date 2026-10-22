"""
Pipeline Orquestador End-to-End: Análisis de Demanda del Subte de CABA
Ejecuta la ingesta, validación estadística y visualizaciones en secuencia.
"""
import sys
import time
from src.preprocesamiento import preprocess_subte
from src.validacion_estadistica import compute_statistics
from src.eda_visualizaciones import generate_visualizations

def run_all():
    start_time = time.time()
    print("=" * 70)
    print("INICIANDO PIPELINE ANALÍTICO COMPLETO - SUBTE CABA (LÍNEAS A, B, C)")
    print("=" * 70)
    
    # 1. Preprocesamiento (si no existe ya el parquet procesado)
    import os
    if not os.path.exists(os.path.join('data_processed', 'subte_consolidado_15min.parquet')):
        print("\nPASO 1/3: Preprocesamiento y consolidación de datos...")
        preprocess_subte()
    else:
        print("\nPASO 1/3: Datasets preprocesados encontrados en 'data_processed/'. Omitiendo re-ingesta.")
        
    # 2. Validación Estadística
    print("\nPASO 2/3: Ejecutando validación estadística y generación de tablas...")
    compute_statistics()
    
    # 3. Visualizaciones
    print("\nPASO 3/3: Generando visualizaciones en alta resolución (300 DPI)...")
    generate_visualizations()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETADO EXITOSAMENTE en {elapsed:.1f} segundos.")
    print("Todos los entregables están disponibles en 'output/' y 'notebook_eda_subte.ipynb'.")
    print("=" * 70)

if __name__ == '__main__':
    run_all()
