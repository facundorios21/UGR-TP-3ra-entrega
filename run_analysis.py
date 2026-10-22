"""
Pipeline Orquestador End-to-End: Análisis de Demanda del Subte de CABA
Ejecuta la ingesta, validación estadística y generación de visualizaciones.
"""
import os
import sys
import time
import argparse
from src.preprocesamiento import preprocess_subte
from src.validacion_estadistica import compute_statistics
from src.eda_visualizaciones import generate_visualizations

def run_all(force_reprocess=False):
    start_time = time.time()
    print("=" * 70)
    print("INICIANDO PIPELINE ANALÍTICO COMPLETO - SUBTE CABA (LÍNEAS A, B, C)")
    print("=" * 70)
    
    parquet_path = os.path.join('data_processed', 'subte_consolidado_15min.parquet')
    
    # 1. Preprocesamiento de datos brutos
    if force_reprocess or not os.path.exists(parquet_path):
        if force_reprocess:
            print("\nPASO 1/3: Flag '--force-reprocess' activado. Re-procesando CSVs originales...")
        else:
            print("\nPASO 1/3: No se encontraron datos procesados. Iniciando ingesta de 'csv/'...")
        preprocess_subte()
    else:
        print("\nPASO 1/3: Datasets preprocesados encontrados en 'data_processed/'.")
        print("          (Usa 'python run_analysis.py --force-reprocess' si deseas re-procesar los CSVs originales).")
        
    # 2. Validación Estadística y Tablas Markdown
    print("\nPASO 2/3: Ejecutando validación estadística y regenerando tablas Markdown...")
    compute_statistics()
    
    # 3. Visualizaciones en Alta Resolución (300 DPI)
    print("\nPASO 3/3: Regenerando todos los gráficos en alta resolución en 'output/graficos/'...")
    generate_visualizations()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETADO EXITOSAMENTE en {elapsed:.1f} segundos.")
    print("Todos los gráficos y tablas han sido regenerados en la carpeta 'output/'.")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ejecutar análisis de demanda de Subte CABA")
    parser.add_argument(
        '--force-reprocess', 
        action='store_true', 
        help="Fuerza el preprocesamiento desde los archivos CSV originales en csv/ incluso si data_processed ya existe."
    )
    args = parser.parse_args()
    run_all(force_reprocess=args.force_reprocess)
