# Auditoría técnica — Tesis PINNs NCh433
Revisor: ingeniería estructural + código. Registro vivo de hallazgos.
Severidad: 🔴 crítico (afecta resultados/validez) · 🟡 medio (rigor/consistencia) · 🔵 menor (typo/forma)

Estado: EN CURSO. Cubierto: Cap.1 (texto) + núcleo de código (espectro). Pendiente: Cap.3, 4, 5, 6.

---

## CÓDIGO — Núcleo físico/normativo

| # | Sev | Ubicación | Hallazgo | Fix sugerido |
|---|-----|-----------|----------|--------------|
| C1 | 🔴 | `b2core/nch433.py:56` | α con denominador `(1+x)³` en vez de `1+x³`. Hunde α de ~2.75 a ~0.69 (3–4× bajo). | Cambiar a `1.0 + x**3`. Verificar que ningún resultado de la tesis pasó por este archivo (lo importan `spectral_analysis.py`, `__init__.py`, `generator_VF.py` = pipeline viejo). |
| C2 | 🟡 | `opensees_functions_v6.py:94-97` y `nch433_torch.py` (`alpha_v6_numpy`) | α usa `T'` y `n` donde NCh433 Ec.(6-9) usa `T0` y `p`. GT y PINN consistentes entre sí, pero ambos desviados de la norma (y de la Ec.1.23 del propio texto). Impacto: ~0% en casos topados, hasta ~20% sin tope; mayor en modos altos. | Opción A: usar `T0,p`, regenerar GT, reentrenar. Opción B: documentar como decisión y cuantificar. |
| C3 | 🟡 | `opensees_functions_v6.py:90` | R* usa `(Ro-1)=10` en vez de `Ro=11` (Ec.6-10). Conservador (~8% más Sa en T largos), pero no es la fórmula literal. | Usar `Ro` salvo justificación. |

## TEXTO — Capítulo 1 (marco normativo)

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| T1 | 🔴 | Ec.(1.23) vs código | Texto escribe α canónica (`T0,p`) — CORRECTA — pero el GT se generó con `T',n` (ver C2). Inconsistencia texto–código en la ecuación normativa central. | Reconciliar texto y código (ver C2). |
| T2 | 🔵 | Ec.(1.22) | Denominador aparece `R*/T`; debe ser `R*/I`. (Fig.1.3 lo muestra bien con I → typo.) | Corregir `T`→`I`. |
| T3 | 🔵 | Tabla 1.3 (A0) | Tercera fila dice "Zona **2** → 0.40g"; debe ser "Zona **3**". | Corregir a Zona 3. |
| T4 | 🔵 | Tabla 1.4 (suelos) | Encabezado dice "Zona sísmica"; debe ser "Tipo de suelo". Falta el suelo **E** (salta D→F). | Corregir encabezado; agregar fila E. |
| T5 | 🟡 | Ec.(1.25) | `C = S·A0·I/R` incompleta como coef. sísmico estático NCh433 (falta `2.75·(T'/T*)^n`). | Completar o etiquetar como conceptual. |
| T6 | 🔵 | Ec.(1.26) | `wh/Σwh` es la distribución simplificada; NCh433 usa la fórmula Ak con raíces. | Etiquetar como aproximación. |
| T7 | 🟡 | §1.7.5 vs GT | Presenta torsión accidental como exigencia, pero el GT (masa en centro geométrico, sin ±5% ni momentos 0.10b·Zk/H) NO la incluye. | Declarar como simplificación del GT (confirmar en Cap.4). |
| T8 | 🔵 | Ec.(1.9) | Llama a `fn=1/Tn` "frecuencia cíclica circular"; es cíclica (Hz). La circular es ωn. | "cíclica (Hz)". |

---

## TEXTO — Capítulo 3 (modelación) + código de respuesta

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| T9 | 🟡 | §3.3 amortiguamiento | Dice que se usa matriz **Rayleigh 5%**. En análisis modal espectral el amortiguamiento entra como **modal (5% por modo) vía CQC**, no como matriz Rayleigh; el eigen solo usa M,K y la CQC usa ζ=0.05. Rayleigh no se usa. | Reemplazar por "amortiguamiento modal 5% en CQC"; quitar Rayleigh. |
| T10 | ✅ | §3.3 / §1.7.5 vs `compute_spectral_response:557` | Torsión accidental: el texto dice "excentricidad 5%" y §1.7.5 describe correr CM ±5% o momentos `0.10b·Zk/H`. El **código no hace ninguno**: amplifica el desplazamiento traslacional por `(1+e_acc/r_giro)` (heurístico). Para el 2100 eso infla Uy ~17%. Además ref. "art. 6.1.2" probablemente errónea (torsión accidental modal ≈ 6.3.4). | **RESUELTO — ver sección de resolución.** Derivación + divulgación (§4.5), validación honesta contra método riguroso (§4.7). |
| T11 | 🟡 | Tabla 3.1 vs código | `t_borde` se rotula "muros de borde **corredor (B4)**"; en realidad `t_borde`=**B2** (fachada) y `t_mid`=**B3 y B4**. El texto p.69 dice bien "t_borde = B2" → Tabla 3.1 se contradice con el texto y el código. | Corregir Tabla 3.1: t_borde→B2; aclarar t_mid→B3+B4. |
| T12 | 🟡 | Tabla 3.1 | Lista `lw_borde` y `lw_mid` (2,0–4,0 m, paso 0,2) como variables muestreadas, pero **no existen en el dataset** (X tiene 18 features, sin `lw_*`). Los largos de muro son **derivados de la geometría modular** (B2≈Ly, B3≈prof_depto, B4≈45 m), no 2–4 m. Entradas obsoletas de una versión vieja. | Eliminar `lw_borde`/`lw_mid` de Tabla 3.1 o marcarlas como derivadas. |
| T13 | 🔵 | §3.4 p.67 vs Tabla 3.1 | Texto dice `n_unid_lado` "entre 2 y **9**"; Tabla 3.1 y dataset dicen **2–6**. | Corregir a 2–6. |
| C4 | 🟡 | `opensees_functions_v6.py:113-115` | Topes Cmín/Cmáx (`Sa_min`, `Sa_max`) se aplican **por ordenada espectral de cada modo**, no al **cortante basal combinado** como exige NCh433. Enmascara la desviación de α (C2) y altera modos altos. | Revisar: aplicar Cmín/Cmáx a V combinado. |
| C5 | 🟡 | `compute_spectral_response:575` | "cumple" aplica **un solo límite** a la deriva ya amplificada por torsión; NCh433 separa 0.002h (CM) y 0.001h adicional (perímetro−CM). | Separar los dos chequeos si se reporta cumplimiento. |
| C6 | 🔵 | `compute_spectral_response:537` | Γ usa masa generalizada **solo traslacional** (omite I_rot·φθ²). Aproximación válida para modos traslacionales; pequeño sesgo en modos con rotación. | Nota/aclaración; impacto menor. |

## ✅ Verificado correcto en Cap.3
Masa concentrada `gk+ψ·qk`; diafragma rígido 3 GDL; `elasticBeamColumn`; E=4700√fc (ACI); Ie=0.35Ig; ζ=5%; ψ_piso=0.25, ψ_techo=0; qk=2.0; R0=11; I=1.0; rangos de Tabla 3.1 (salvo T11–T13) coinciden con el dataset; Vb modal = CQC(mass_part·M·Sa) correcto; Sd=Sa/ω² correcto.

## TEXTO — Capítulo 4 (dataset y QA)

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| T14 | 🟡 | Tabla 4.3 vs archivo real | La historia de versiones llega a **v5**, pero el dataset que usa la PINN es **v6.1** (`..._v6_20260523_1226.h5`), que agrega la **reconstrucción de K en float32** (consistencia K·Φ=ω²M·Φ) — paso relevante para la pérdida física y NO documentado en la tesis. | Agregar v6/v6.1 a Tabla 4.3 o aclarar que "v5"=archivo v6. |
| T15 | 🔵 | Tabla 4.3 | Tanto v4 como v5 se rotulan "Dataset definitivo" (doble). La celda de v5 describe el *problema* (3067 casos) en vez de la *corrección*. | Un solo "definitivo"; reescribir celda v5 como la solución. |
| T16 | 🔵 | Tabla 4.4 vs código | Tabla dice modos extraídos 18/24/30; el código usa `N_MODOS_POR_PISOS = 3·N` (hasta 54). El resultado igual cumple ≥90%, pero la tabla no refleja el código. | Alinear Tabla 4.4 con el 3N real (o explicar el tope). |
| T17 | 🔵 | p.108 | "La **Figura 35** ilustra…" — referencia rota; debería ser Fig. 4.11. | Corregir cross-reference. |
| T10b | 🟡 | §4.5 | Reaparece "Art. **6.1.2**" para torsión accidental (igual que §1.7.5/§3.3). La torsión accidental modal no es el Art. 6.1.2. Verificar número de artículo NCh433. | Corregir referencia de artículo. |

## ✅ Verificado correcto en Cap.4 (contra el HDF5 real)
- **Cobertura modal ≥90%**: 100% de los 10.000 casos cumple en X y en Y al slot 18 (mín. 0.959). La afirmación de p.110 se sostiene. ✅
- Evolución v1→v5 honesta y autocrítica (masa rotacional, ARPACK→fullGenLapack, torsión accidental, 90% masa X).
- Depuración 37→18 features (elimina derivadas/post-análisis para evitar shortcut learning). ✅
- Estructura HDF5 (inputs/modal/response/matrices/scalars) coherente con el archivo. ✅
- Normalización modal M-ponderada = 1.0. ✅

## TEXTO — Capítulo 5 (PINN)

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| T18 | 🔵 | Ec.(5.1) | Escrita `L_total = λ_data + λ_phys·L_phys` — falta el factor `L_data` (debe ser `λ_data·L_data + λ_phys·L_phys`). | Corregir ecuación. |
| T19 | 🔵 | p.128 | Glitch de render: "PINN_Modal_v4 puede representarse como la combinación" aparece pegado sin espacios (problema LaTeX). | Recompilar/espaciar. |
| T20 | 🟡 | Tabla 5.1 vs Tabla 3.1 | Tabla 5.1 rotula bien `t_mid`=B3/B4 y `t_borde`=B2; pero Tabla 3.1 (Cap.3) los rotula mal. **Inconsistencia interna** → corregir Tabla 3.1 (ver T11). | Alinear Tabla 3.1 con Tabla 5.1. |

## ✅ Verificado correcto en Cap.5
- 21 inputs (one-hot suelo/zona) coinciden con `COLS_BASE` del código. ✅
- Exclusión de masa sísmica del input (anti-shortcut). ✅
- Tres pérdidas físicas (eig normalizado, MAC, M-ortogonalidad) **correctas físicamente**. ✅
- L_data_resp excluye drift_x (se reconstruye vía ratio drift_x/drift_y) — coherente. ✅
- Arquitectura (encoder T1 + compartido, multi-head, ResBlocks, SiLU) coincide con `PINNModal_v4`. ✅
- Masking para pisos/slots inexistentes. ✅
- Pendiente verificar en Cap.6: que λ_φ, λ_eig, λ_ortho sean **no nulos** y que la ablación muestre que la física **aporta** (si no, "physics-informed" sería nominal).

## TEXTO — Cap.5.4 (entrenamiento) + Cap.6.1 (métricas)

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| T21 | 🔵 | p.133 | Glitch "del metamodelo.En particular" (sin espacio tras punto). | Espaciar. |
| T22 | 🔵 | p.142 | Dice que T1 "varía entre 0.2 y 1.5 s"; en el dataset el fundamental llega a **~2.7 s** (edificios de 18 pisos). Subestima el rango superior. | Corregir a ~0.05–2.7 s. |

## ✅ FORTALEZAS verificadas (Cap.5.4–6.1)
- **Detección y corrección de fuga de datos**: reconocen que la partición aleatoria daba métricas infladas (vecinos casi duplicados) → migran a **31 trials OOD** con separación por variable clave, verificada con **d de Cohen >0.8** y chequeo de confounding con N (Tabla 6.3). Metodología sólida. ✅✅
- Staged training (5 etapas), 600 épocas, Adam, Cosine Annealing WR, gradient clipping, checkpoints solo en etapas con física activa. ✅
- Métricas todas correctas: MAC (6.1), MAC_avg (6.2), MAPE (6.3), MSE/RMSE/MAE/R²/Pearson (6.4–6.8). ✅
- Umbrales de aceptación razonables (Tabla 6.1). ✅

## TEXTO — Capítulo 6 (validación)

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| T23 | 🔴 | §6.6 / Tabla 6.8 | **El "Caso 2100" de la validación Robot NO coincide con el índice 2100 del dataset entregado (v6).** Tesis: 11 pisos, T1_Y=0.5468, masa 6829.5 t, fc=35. HDF5 idx 2100: **16 pisos, T1=1.2686, 17.966 t**. El caso de la tesis es de una versión vieja del dataset; no es localizable con ese índice en v6. **Compromete la reproducibilidad de la validación externa (argumento estrella).** | Re-identificar el caso real en v6 y relabelar, o declarar la versión usada. Idealmente rehacer Robot sobre el 2100 actual. |
| T24 | 🟡 | §6.6 interpretación | El metamodelo "le gana" a Robot en T1_Y (−1.8% vs −17.7%) **porque fue entrenado con el GT de OpenSees** (columna ancha); reproduce su idealización. No demuestra superioridad física sobre Robot, sino fidelidad a OpenSees. El texto lo matiza, pero la conclusión "más cerca que Robot" puede leerse como superioridad. | Reformular: "reproduce fielmente la referencia validada cross-tool", no "más preciso que Robot". |
| T25 | 🟡 | Conclusiones (p.169) vs Tabla 6.7 | Cifras estrella "9.7% T1, 10.5% Vb, 13.3% deriva" ≈ Trial 16 (9.71/9.95/13.33), pero el modelo definitivo reentrenado da 6.91/10.10/15.01 (Tabla 6.7). Vb=10.5% no coincide con ninguna. Mezcla de fuentes. | Unificar qué modelo/cifras son las oficiales. |

## ✅ FORTALEZAS verificadas en Cap.6
- **Ablación (Tabla 6.6) demuestra que la física APORTA** (no es nominal): MAC 0.10→0.29 (L_eig) →0.996 (L_MAC), L_ortho estabiliza escalares. Cada componente cumple un rol. ✅✅
- SHAP físicamente coherente (N_pisos→T1/deriva; n_unid_lado+N+zona+suelo→Vb). El modelo aprendió física, no atajos. ✅
- Métrica MAC correcta para modos (invariante a signo/escala). ✅
- Honestidad sobre la limitación en modo X de orden superior (−32% meta). ✅

## TEXTO — Capítulo 2 (teoría PINN / estado del arte)

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| T26 | 🟡 | Tabla 2.1 | Lista **"Proyecto PINNs Sísmico (Anónimo)", 2025** como referencia del estado del arte, pero describe **el propio trabajo de esta tesis** (respuesta sísmica NCh433 en ms vs 45s OpenSees, autograd PyTorch, curriculum). Citarse a sí mismo como "Anónimo" en la revisión es problemático. | Eliminar la fila o etiquetarla como "el presente trabajo". |
| T27 | 🔵 | p.41 | Cita rota "[?]" (referencia LaTeX no definida). | Corregir referencia. |
| T28 | 🔵 | p.40, p.42 | "hiper parámetros" (con espacio) → "hiperparámetros". | Corregir. |

## ✅ Verificado correcto en Cap.2
Ecs. (2.1)–(2.5) de Raissi (residuo físico, MSE_u, MSE_f) correctas; Ec.(2.3) `L=L_datos+w·L_física` bien escrita (confirma que Ec.5.1 tiene typo). Estado del arte razonable.

---

## CÓDIGO/GT — Clasificación modal (hallazgo nuevo, vía comparación shell)

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| C7 | 🟡 | clasificación modal del GT / `seleccionar_18_slots` | El "T1" almacenado (slot1 = primer `trans_Y`) **no es el periodo fundamental** en **6.8% de los casos (680)**, donde el modo de periodo más largo se clasifica como **torsional** (edificios acoplados). **26 casos severos** (T1 guardado <50% del fundamental real), **todos `activar_B2=0`** — ej. **4051** (guarda 0.084 s; real ≈0.528 s) y **6182** (0.046 s; real ≈0.29 s). La PINN aprende un "periodo fundamental" físicamente incorrecto en estos casos. La *respuesta* (Vb, deriva) NO se afecta porque usa `T_star=T[argmax(mass_part)]` (el periodo dominante correcto). | Redefinir "T1" como el modo de periodo más largo con masa significativa en su dirección, o excluir/marcar los casos acoplados. En la comparación Robot, NO usar el T1_Y guardado de 4051/6182. |

## TEXTO — Conclusiones / Bibliografía / Anexo

| # | Sev | Ubicación | Hallazgo | Fix |
|---|-----|-----------|----------|-----|
| T29 | 🟡 | Anexo A (p.179) | El Anexo está **vacío** ("Aquí se incorpora información complementaria"), pero el Cap.6 remite a contenido "en el anexo" (SHAP de Vb_y y de drift_x). **Referencias colgantes** a material inexistente. | Completar el anexo con las figuras citadas, o quitar las remisiones. |
| T30 | 🟡 | [INN, 2026] vs código | El texto cita **NCh433:2026** como norma rectora, pero el código implementa los parámetros espectrales de **DS61 (2011)** (S, T0, T', n, p de la tabla DS61). NCh433:2026 trae clasificación de suelo por Vs30 y factores distintos. Posible mezcla de versiones. | Aclarar qué espectro se implementó realmente (DS61) y separar la cita normativa. |
| T26b | 🟡 | Tabla 2.1 / Bibliografía | "Proyecto PINNs Sísmico (Anónimo)" **no tiene entrada en la bibliografía** (todos los demás autores de la Tabla 2.1 sí). Confirma que es auto-referencia/placeholder. | Eliminar fila o citar como "el presente trabajo". |

## ✅ Conclusiones (p.169-174): sólidas y honestas
Reconocen: deriva = mayor error/dispersión; dominio de validez = edificios regulares; recomiendan uso en prediseño (no reemplazo). El **trabajo futuro ya menciona** modelar los vanos de puerta y validar contra modelos tipo lámina (shell) — justo lo que estamos haciendo. Coherente.

# AUDITORÍA COMPLETA — 100% de la tesis + códigos núcleo cubiertos.
TOTAL: 🔴 ×4 · 🟡 ×19 · 🔵 ×14.

---

## ESTADO DE RESOLUCIÓN (actualizado)

- **C1 ✅ RESUELTO:** verificado que el Vb del HDF5 coincide con la fórmula `opensees_v6` (correcta), NO con la buggy → el `(1+x)³` de `nch433.py` nunca tocó el dataset. Acción restante: borrar/corregir el archivo por limpieza (cosmético).
- **C2 / T1 ✅ RESUELTO (en lo esencial):** bug confirmado (3 fuentes); generado `dataset_..._CORREGIDO.h5` (solo `response/`, α canónica T0,p + R* con Ro); modelo final reentrenado. Impacto: mediana 1.4%, las métricas del modelo final cambian ~1pp; **ninguna conclusión cualitativa cambia**. Acción restante: (a) actualizar números del modelo definitivo (Tabla 6.7 + conclusiones), (b) 1 frase de trazabilidad en Cap.4.
- **T10 / T7 / T10b ✅ RESUELTO:** torsión accidental. Confirmado que el código usa amplificación heurística `U=U_trasl·(1+e_acc/r_giro)`, no los métodos rigurosos NCh433. Derivada de primeros principios (diafragma rígido acoplado, hipótesis Ω=ω_θ/ω_t=1; excentricidad normalizada e/r — Chopra). Validada contra el método riguroso (CM±5%, eig acoplado, CQC, esquina, misma α): el riguroso da deriva +3.5% mediana, hasta +30% en edificios alargados → el método simplificado **subestima levemente** (NO conservador). Solo **3/10.000 casos (0.03%)** cambian veredicto. Scripts: `_torsion_rig.py`, `_torsion_flip.py`, `_torsion_clean.py`. Acciones de redacción: (a) §1.7.5 se queda — fix cita `[Chopra]→[NCh433]` en excepción del 20% y verificar `0,10b·Zk/H`; (b) §4.5 reemplazar párrafo con divulgación + fórmula `(1+e_acc/r)` + derivación; (c) §4.7 nuevo párrafo de validación honesta (mediana 3.5%, máx 30%, 3/10.000); (d) corregir "Art. 6.1.2". **NO se usa el argumento del 20%** (métricas distintas; el error supera 20% en la cola).
- **T30 ✅ RESUELTO (con la NCh433:2026 real en mano):** verificado punto por punto que el espectro del código = NCh433:2026 EXACTO — Tabla 8 (S,T0,T',n,p suelos A-D idénticos), Tabla 7 (A0), Ec.(8) Sa, **Ec.(9) α usa T0,p**, **Ec.(10) R* usa R0**, Tabla 9 (Cmáx=0.35·S·A0), §6.3.6.1 (Cmín), §6.3.5.2 (CQC ζ=0.05), §6.3.2 (masa ≥90%), §5.9 (deriva 0.002h CM + 0.001h perímetro), Tabla 6 (I). **CRÍTICO:** las Ec.(9)/(10) de la 2026 = la fórmula CORREGIDA (T0,p,R0); el dataset viejo (T',n,R0-1) NO cumple la 2026. Producción usa el corregido → cumple. Acciones de redacción hechas: (a) `[INN,2026]` válido como rectora + `[MINVU,2011]` antecedente; (b) bullet NCh433:2026 agregado a §1.7.1 (aclara que Sa/α/derivas no cambiaron); (c) frase de alcance clasificación suelo (Vs30/Tg fuera de alcance); (d) eliminada "excepción del 20%" de §1.7.5 (NO existe en la 2026; contradecía Cap.4); (e) artículo torsión 6.1.2→**6.3.3**; (f) diafragma rígido 6.1.1→reescrito como idealización consistente con §5.5.2/§6.1.1 (no "exigida"). El `0.10·b·Zk/H` confirmado correcto (Ec.7). Archivo: `NORMA CHILENA 433 2016.pdf` (es la 2026 real, Quinta edición).
- **T9 ✅ RESUELTO:** §3.3 reescrito — amortiguamiento es modal 5% del crítico vía CQC (§6.3.5.2), eigenproblema solo con M y K. Eliminada la mención a matriz Rayleigh y "por defecto en OpenSees".
- **C7 ✅ RESUELTO (documentado, sin rehacer):** cuantificado sobre dataset corregido (`_c7_analisis.py`): el "T1" guardado (slot1=primer trans_Y) NO es el fundamental en **628 casos (6,3%)**, **TODOS B2=0** (sin muros de fachada exterior). El modo de periodo más largo es **torsional en el 100%** de esos casos (edificios torsionalmente flexibles: sin B2 cae K_θ). 26 severos (ratio<0.5: 870,2155,...,4051,6182,8594). **La respuesta NO se afecta** (usa T*=T[argmax(mass_part)], correcto; slot1 es el de mayor masa Y en 95.9%). Resolución: (a) §4.8 párrafo nuevo documentando el 6,3% como extremo torsionalmente flexible del dominio (terminología B2=fachada, cita chopra2014); (b) §5.1.2 nomenclatura: T1=primer modo traslacional en Y (no "fundamental"), coincide con fundamental en 93,7%; (c) en validación Robot NO usar T1 guardado de los 26 severos. NO se regenera ni reentrena.
- **T26 / T26b ✅ RESUELTO:** usuario eliminó la fila "Proyecto PINNs Sísmico (Anónimo)" de Tabla 2.1 y el estudio asociado.
- **C4 ✅ RESUELTO (HALLAZGO MAYOR):** el código capaba el espectro `Sa=clip(Sa/R*,smin,smax)` POR ORDENADA (interpretación estática) y lo usaba para Vb Y desplazamientos. NCh433:2026 §6.3.6: Cmín/Cmáx van sobre el corte basal COMBINADO, y Cmáx NO toca desplazamientos. Verificado 4 formas: (1) código real `opensees_functions_v6.py:115,529,554`; (2) reconstrucción = HDF5 al 0.000%; (3) texto norma (usuario confirmó §6.3.4.1/§6.3.6.2/§6.2.3); (4) Vb/estático pasó de 0.66 (mal) a 0.92 (correcto). Impacto: Vb +40%, deriva +21% (rango -14% a +50%), ~700 etiquetas cumple cambian. **Generado `dataset_..._DEFINITIVO.h5`** (α canónica + R* Ro + cap sobre V combinado). Reentrenado modelo final: MAC 0.998, T1 7.5%, Vb 11.5%, drift 13.6% (drift MEJORÓ vs 15.5%). Scripts: `_c4_analisis.py`, `_c4_verif.py`, `_generar_definitivo.py`, `_verif_definitivo.py`. NO se rehacen 31 trials ni ablation (modal intacto).
- **T11/T12/T20 ✅ RESUELTO:** Tabla 3.1 corregida (verificado contra PKL producción + código `familia_B_core.py`): t_borde→**B2 fachada** (no B4), t_mid→**B3+B4**; `lw_borde`/`lw_mid` reemplazadas por largos derivados reales (B2:15.5-17.7, B3:6.98-7.92, B4:18.95-47.23 m).
- **T14/T15 ✅ RESUELTO:** Tabla 4.3 con fila v6 (corrección espectral); narrativa §4.7 actualizada (v1→v6); quitado doble "definitivo".
- **§4.5 (A1) ✅:** topes Cmín/Cmáx descritos sobre corte combinado, no ordenada.
- **T9, T30, C7, T10, T26 ✅** (ver arriba).
- **T23 🔧 EN PROGRESO:** rehaciendo la validación externa Robot con el set de 30 casos diversos (cobertura completa N_pisos + H/Lx).
Total: 🔴 ×4 · 🟡 ×16 · 🔵 ×14.  Detalle arriba. Fortalezas verificadas: cobertura modal ≥90%, validación OOD (sin fuga), ablación (la física aporta), SHAP coherente, pérdidas físicas correctas.
- Cap.5 (arquitectura PINN, función de pérdida física, normalización, partición/fugas)
- Cap.6 (métricas, ground truth, ablación, SHAP, validación Robot)
