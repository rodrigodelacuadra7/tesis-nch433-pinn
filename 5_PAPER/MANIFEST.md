# Manifiesto de trazabilidad — Metamodelo PINN Familia B (paper)
Generado: 2026-06-21 19:23

Este documento traza, de punta a punta, los artefactos del modelo definitivo reportado.
**Regla de oro:** el modelo, los scalers y el dataset son una TRIPLETA. Solo reproducen las
metricas reportadas si se usan JUNTOS. No mezclar con versiones de mayo.

> **Esta carpeta es un paquete de replicación autocontenido.** Empieza por
> `README_REPLICACION.md`. El dataset (141 MB) ahora vive AQUÍ; `reproducir_metricas.py`
> recalcula las métricas reportadas con un solo comando.

## 1. Cadena de trazabilidad (artefactos FINALES — los del paper)

| # | Artefacto | Rol | sha256 (12) |
|---|-----------|-----|-------------|
| 1 | `dataset_familyB_OPENSEES_v6_DEFINITIVO.h5` | Ground truth corregido (10.000 casos) | `5bc750f13652` |
| 2 | `Notebook_5_CORREGIDO.ipynb` | Entrenamiento Trial 16 sobre (1) | — |
| 3 | `model_fase2_definitivo.pt` | Metamodelo final (PINNModal_v4b) | `3561ea2d0d9c` |
| 4 | `scalers_trial16_DEFINITIVO.pkl` | Normalizacion (calculada sobre pool train zona 1+3) | `0e531b829ccd` |
| 5 | `resultado_fase2_definitivo.json` | Metricas reportadas | — |
| 6 | `history_fase2_definitivo.csv` | Curva de entrenamiento | — |

### Paquete de replicación (codigo, agregado 2026-06-22)
| Archivo | Rol |
|---------|-----|
| `README_REPLICACION.md` | Guia maestra: usar / reproducir / reentrenar |
| `pinn_familiaB.py` | Modulo autocontenido: arquitectura + carga + prediccion + veredicto NCh433 + split Trial 16 |
| `reproducir_metricas.py` | Script autoverificable (reproduce drift/T1/Vb/MAC sin reentrenar) |
| `requirements.txt` | Dependencias (CPU-friendly) |

Ruta original del dataset: `C:/Users/rodri/Documents/NB\dataset_familyB_OPENSEES_v6_DEFINITIVO.h5`
Ruta original del modelo:   `C:/Users/rodri/Documents/PINN/b2_proyecto/models/trials_31/V8/DEFINITIVO_CORREGIDO\model_fase2_definitivo.pt`

### sha256 completos
- dataset : `5bc750f13652f1e8c2d3c257724578a98a92ca841545f0f16a4454f7d5837edd`
- modelo  : `3561ea2d0d9cb9b7b273fca8b5fd756f3f0fb1a0a3f30495e8502136bf749e8f`
- scalers : `0e531b829ccd8dd978a2bd5ac2489385a3854afc8a56f12124daaaf2cde66ef4`

## 2. Configuracion y metricas reportadas (de resultado_fase2_definitivo.json)
- Trial: 16 — zona_1_3_vs_zona_2  (entrena zonas 1 y 3, evalua OOD en zona 2)
- Arquitectura: PINNModal_v4b  |  Loss: compute_loss_v16  |  Curriculum: v20_staged_training
- Dataset usado: `dataset_familyB_OPENSEES_v6_DEFINITIVO.h5`
- n_train = 6623  |  n_test (OOD zona 2) = 2977  (de 3377 totales; 400 a validacion)
- Epocas: 600  |  best_epoch: 333  |  tiempo: 26.0 min

**Metricas OOD (fase 2):**
- MAC_avg   = 0.9979
- T1_MAPE   = 7.51 %
- Vb_MAPE   = 11.45 %   (Vbx 12.93 | Vby 9.98)
- drift_MAPE= 13.64 %   <-- 13.64 % de la tesis

## 3. Como reproducir las metricas
**Forma rapida (1 comando):** `python reproducir_metricas.py`
Reproducido y verificado el 2026-06-22:
- n_test = 2977 | drift_MAPE = 13.6392 | T1_MAPE = 7.5068 | Vb_MAPE = 11.4539 | MAC_avg = 0.9979
(coinciden con resultado_fase2_definitivo.json: 13.64 / 7.51 / 11.45 / 0.9979).

**Forma manual:**
1. Cargar dataset (1), construir X de 21 columnas one-hot (ver Notebook_5_CORREGIDO celda 3 o `pinn_familiaB.construir_X`).
2. Split: train = zonas 1 y 3 (6623); OOD zona 2 (3377) -> shuffle con rng_b -> val[:400], test[400:] (2977).
   (`pinn_familiaB.split_trial16`; rng = default_rng(2026 + 16*200)).
3. Cargar modelo (3) y scalers (4).
4. Normalizar con (4): X z-score; periodos log-z; Ux/Uy/drift/Vb en log-z (medias/std del pool train).
5. Evaluar sobre test OOD -> drift_MAPE = 13.64 %, etc.
> Los scalers (4) se calculan SOLO con el pool de entrenamiento (zonas 1+3 = determinista),
> por eso se pudieron reconstruir exactos sin re-entrenar.

## 3b. Generación del dataset — subcarpeta `generacion_dataset/`
Artefacto INTERMEDIO del pipeline (dominio → casos → OpenSees → .h5).
- `dataset_familyB_cases_20260523_0758.pkl` (471 MB, sha256(12) `0a1a90a06652`):
  los 10.000 casos de la Familia B (Notebook 3). 10.000 casos, calza en N con el `.h5`.
- `Notebook_3_FamiliaB_PKL.ipynb`: notebook que lo genera.

> ⚠️ NO es el ground truth. Verificado caso a caso vs `DEFINITIVO.h5`:
> `params`/`geom`/`floor_masses`/`M` coinciden; `K` difiere (en el `.h5` es
> reconstrucción modal `K=MΦΩ²ΦᵀM`, en el `.pkl` es ensamblado crudo);
> `modal`/`response` (drift, Vb, Sa) están **OBSOLETOS** (espectro pre-corrección
> 18-jun, ej. caso 0 drift_x 4.78e-05 vs .h5 7.16e-06). Sirve como input geométrico
> para re-correr OpenSees; NO sustituye al `.h5`. Ver `generacion_dataset/README.md`.

## 4. Archivos OBSOLETOS — NO usar para el paper (causan el 20.1 %)
Estos son de un run anterior (dataset de mayo, sin corregir, y un bug en el scaler .max):
- `models/trials_31/V8/DEFINITIVO/` (carpeta vieja, modelo 16-jun sobre dataset de mayo)
- `models/trials_31/V8/DEFINITIVO/scalers_trial16.pkl` (27-may, scaler con bug .max)
- `dataset_familyB_OPENSEES_v6_20260523_1226.h5` (mayo, espectro sin corregir)
- `dataset_familyB_OPENSEES_v6_CORREGIDO.h5` (intermedio)
> Mezclar el modelo nuevo con estos scalers viejos da drift 20.1 % en vez de 13.64 %.

## 5. Analisis de cumplimiento (falsos positivos / negativos)
Notebook: `Analisis_FP_FN_cumplimiento.ipynb` — carga la tripleta (modelo+scalers+dataset) y
evalua el veredicto de cumplimiento NCh433 (deriva <= 0.002) sobre el conjunto OOD (zona 2).

Resultado (con la tripleta correcta — scalers_trial16_DEFINITIVO.pkl):
- n (OOD zona 2) = 3377  |  exactitud = 96.95 %
- TP=2627  FP=36 (inseguros, 1.1%)  FN=67 (conservadores, 2.0%)  TN=647
- Tendencia CONSERVADORA (FN > FP): tiende a descartar antes que a aprobar de mas.
- Margen de seguridad: con umbral de decision tau=0.00154 los FP bajan a 0, marcando
  1084 edificios (32.1%) para verificacion FEM -> tamiz de cumplimiento SEGURO.

NOTA: una version anterior daba acc 93.07% (FP=69, FN=165) porque usaba los scalers viejos
(seccion 4). Los numeros validos para el paper son los de arriba (96.95%).
