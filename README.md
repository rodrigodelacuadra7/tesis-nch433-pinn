# Tesis: metamodelo PINN para la respuesta sísmica según NCh 433

Metamodelo neuronal informado por la física (PINN) que predice la respuesta sísmica de edificios de muros de hormigón armado de una familia paramétrica (Familia B), con etiquetas obtenidas de OpenSeesPy y del análisis modal espectral de la NCh 433.

## Estructura

| Carpeta | Contenido |
|---|---|
| `1_FAMILIA_B/` | Definición de la familia paramétrica de edificios y análisis modal |
| `2_DATASET/` | Generación del dataset con OpenSeesPy y corrección normativa |
| `3_METAMODELO/` | Metamodelo PINN modal y optimización de hiperparámetros (Optuna) |
| `4_VALIDACION/` | Validación, ablación y verificación de cumplimiento (FP/FN) |
| `5_PAPER/` | Paquete de replicación: modelo final, scalers, figuras y trazabilidad |
| `6_INTERFAZ_DEMO/` | Interfaz de demostración y réplica del pipeline completo |

## Tecnologías

Python · PyTorch · OpenSeesPy · Optuna · NumPy / pandas · Jupyter

## Datos

Los datasets (HDF5/PKL, ~600 MB) no se incluyen por tamaño.
