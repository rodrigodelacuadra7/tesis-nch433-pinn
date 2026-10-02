# Generación del dataset — artefacto intermedio (Familia B)

> ⚠️ **LEER ANTES DE USAR.** Esto NO es el ground truth del paper.
> El GT corregido es **únicamente** `../dataset_familyB_OPENSEES_v6_DEFINITIVO.h5`.

## Qué hay aquí

| Archivo | Qué es |
|---|---|
| `dataset_familyB_cases_20260523_0758.pkl` | Los **10.000 casos** de la Familia B generados por Notebook 3 (geometría + parámetros + matrices). 471 MB. sha256(12): `0a1a90a06652`. |
| `Notebook_3_FamiliaB_PKL.ipynb` | El notebook que genera ese `.pkl` (dominio paramétrico → casos). |

## Lugar en el pipeline

```
dominio paramétrico (Familia B)
        │   Notebook 3  ───────────────►  cases.pkl  (ESTE artefacto)
        ▼
   OpenSees + espectro NCh433
        │   (corrección α(T0,p), R*, Cmin/Cmax — 18-jun-2026)
        ▼
   dataset_familyB_OPENSEES_v6_DEFINITIVO.h5   ◄── GT del paper
        │   Notebook 5 (entrenamiento)
        ▼
   model_fase2_definitivo.pt + scalers
```

## Qué es válido y qué NO (verificado caso por caso)

Comparado contra `DEFINITIVO.h5`:

| Campo del `.pkl` | Estado | Detalle |
|---|---|---|
| `params`, `geom`, `floor_masses` | ✅ **Válido** | Definición del edificio (no depende del espectro). |
| `M` (masa) | ✅ **Coincide** con el `.h5` (bloque nativo). | |
| `K` (rigidez) | ⚠️ **Difiere** del `.h5` | En el `.pkl` es el ensamblado crudo de OpenSees; en el `.h5` es la **reconstrucción modal** `K = MΦΩ²ΦᵀM`. Difieren *por construcción*, no es error. |
| `modal`, `response` (T_r, drift, Vb, Sa…) | ❌ **OBSOLETO** | Calculado con el **espectro VIEJO**, antes de la corrección del 18-jun. Ej. caso 0: `drift_x` pkl `4.78e-05` vs `.h5` `7.16e-06`. **No usar como GT.** |

## Para qué sirve (y para qué NO)

- ✅ **Sirve** como entrada geométrica/paramétrica si alguien quiere **re-correr OpenSees**
  con el espectro corregido y regenerar el `.h5`.
- ✅ **Sirve** para auditar el dominio paramétrico de la Familia B.
- ❌ **NO sirve** para reproducir las métricas del modelo (usar `../reproducir_metricas.py`).
- ❌ **NO sustituye** al `.h5`: su respuesta sísmica es pre-corrección y reabriría
  la discrepancia 20.1 % vs 13.64 %.

> Para regenerar edificios desde el dominio paramétrico de forma autónoma (a escala
> reducida) existe además `../../Pipeline_PINN_Replicacion_FamiliaB.ipynb`.
