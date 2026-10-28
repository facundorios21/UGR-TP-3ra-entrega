"""
Pipeline Orquestador End-to-End: Análisis y Modelado de Demanda del Subte de CABA
Ejecuta la ingesta, validación estadística, visualizaciones EDA, limpieza avanzada
para series temporales, entrenamiento/evaluación (Regresión Lineal vs XGBoost) y
cálculo de métricas operativas derivadas (Fases 1 a 8 de CRISP-DM).
"""
import os
import sys
import time
import argparse
from src.preprocesamiento import preprocess_subte
from src.validacion_estadistica import compute_statistics
from src.eda_visualizaciones import generate_visualizations
from src.preparacion_modelado import prepare_modeling_data
from src.modelado_evaluacion import run_modeling_and_evaluation


def run_all(force_reprocess=False, only_modeling=False):
    start_time = time.time()
    print("=" * 72)
    print("INICIANDO PIPELINE COMPLETO - SUBTE CABA (LÍNEAS A, B, C | 2014-2019)")
    print("=" * 72)

    parquet_path = os.path.join('data_processed', 'subte_consolidado_15min.parquet')

    if not only_modeling:
        # 1. Preprocesamiento de datos brutos
        if force_reprocess or not os.path.exists(parquet_path):
            if force_reprocess:
                print("\nPASO 1/5: Flag '--force-reprocess' activado. Re-procesando CSVs originales...")
            else:
                print("\nPASO 1/5: No se encontraron datos procesados. Iniciando ingesta de 'csv/'...")
            preprocess_subte()
        else:
            print("\nPASO 1/5: Datasets preprocesados encontrados en 'data_processed/'.")

        # 2. Validación Estadística y Tablas Markdown (EDA)
        print("\nPASO 2/5: Ejecutando validación estadística y regenerando tablas Markdown...")
        compute_statistics()

        # 3. Visualizaciones EDA en Alta Resolución (300 DPI)
        print("\nPASO 3/5: Regenerando gráficos de EDA en 'output/graficos/'...")
        generate_visualizations()

    # 4. Limpieza Avanzada para Series Temporales e Ingeniería de Características (Fases 3 y 5)
    print("\nPASO 4/5: Ejecutando limpieza avanzada para series temporales, grilla equidistante y lags...")
    prepare_modeling_data()

    # 5. Modelado Supervisado, Evaluación Temporal y Métricas Operativas Derivadas (Fases 6, 7 y 8)
    print("\nPASO 5/5: Entrenando y evaluando Regresión Lineal vs Gradient Boosting (XGBoost)...")
    run_modeling_and_evaluation()

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print(f"PIPELINE COMPLETADO EXITOSAMENTE en {elapsed:.1f} segundos.")
    print("Todos los datasets, modelos, gráficos y tablas están actualizados en 'output/'.")
    print("=" * 72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ejecutar pipeline completo de Subte CABA (EDA + Modelado)")
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help="Fuerza el preprocesamiento desde los archivos CSV originales en csv/."
    )
    parser.add_argument(
        '--only-modeling',
        action='store_true',
        help="Ejecuta directamente la preparación para modelado y el entrenamiento/evaluación (Pasos 4 y 5)."
    )
    args = parser.parse_args()
    run_all(force_reprocess=args.force_reprocess, only_modeling=args.only_modeling)
