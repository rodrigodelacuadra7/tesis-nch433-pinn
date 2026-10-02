# Nota de reconciliación — métricas del modelo final (Tabla 9 / abstract / conclusiones)

**Fecha:** 2026-06-22
**Estado:** PENDIENTE de aplicar al manuscrito

## El problema
El paper reporta el desempeño del modelo retenido (Trial 16) como:
- T1 MAPE 9.7 % · RMSE 0.092 s · r 0.986
- Vb,x MAPE 10.5 % · RMSE 816.1 kN · r 0.979
- drift MAPE 13.3 % · RMSE 3.57e-4 · r 0.979

Estos valores son **idénticos, dígito por dígito, a los del primer borrador (pre-corrección normativa)**.
El modelo final CORREGIDO (este folder, `model_fase2_definitivo.pt` + `scalers_trial16_DEFINITIVO.pkl` + `DEFINITIVO.h5`) da, re-evaluado sobre el holdout zona 2:
- T1 MAPE 7.5 % · RMSE 0.088 s · r 0.996
- Vb,x MAPE 12.9 % · RMSE 1055 kN · r 0.993
- drift MAPE 13.6 % · RMSE 2.15e-4 · r 0.991
- MAC: Y1 0.996 · torsional 0.998 · X1 0.999 (avg 0.998)

Fuente oficial: `resultado_fase2_definitivo.json` (metricas_f2: T1 7.51, Vbx 12.93, drift 13.64, MAC_avg 0.998).

## Evidencia de que la Tabla 9 está vieja
- RMSE byte-idénticos al primer PDF.
- El RMSE de Vb,x (816 kN) es ~30 % menor que el del modelo corregido (1055 kN); ese ~30 % es justo el efecto de la "Response-spectrum correction" que la propia Tabla 3 documenta. => Tabla 9 trae números pre-corrección bajo el texto que dice "all results on the corrected audited reference". Contradicción interna.

## Severidad
- NO es fraude ni invalida conclusiones (mismo orden de magnitud; T1 incluso mejora).
- ES una inconsistencia interna (Tabla 9 vs Tabla 3 +30 %) que mina el discurso de "referencia auditada/corregida". Arreglo acotado.

## Qué corregir (8 puntos, solo el bloque Trial-16/zona-2)
TIER 1 (modelo retenido): abstract; §3.5 (MAC torsional 0.995→0.998 + prosa RMSE); Tabla 9 (3 filas); párrafo "near 0.98"→"at or above 0.99"; conclusión 6.
TIER 2 (fila zona-2 del barrido, para que §3.4 no contradiga la Tabla 9): §3.4; Tabla 8 fila "Zone stress test" (valor puntual 0.998/7.5/11.5/13.6 + nota al pie); Tabla A.11 fila "Zone 1/3 vs 2".
Aparte: regenerar Figura 7.

## DUDA ABIERTA QUE HAY QUE CERRAR
¿El barrido de 31 splits (Tablas 8, A.11, A.12) se corrió sobre el dataset CORREGIDO o sobre uno viejo?
- Si corregido → solo aplica el arreglo de arriba.
- Si viejo → la frase "all results on corrected reference" abarca 31 tablas pre-corrección; habría que re-correr el barrido o reformular. (No existe corrida multi-semilla corregida del barrido, así que no se pueden generar las ±std corregidas.)

## Fuente de verdad
Regenerar Tabla 9 y Figura 7 desde el script de evaluación oficial que produjo `resultado_fase2_definitivo.json`, no desde una re-corrida externa.
