# -*- coding: utf-8 -*-
"""Construye el notebook unico y factorizado del pipeline PINN (nbformat v4 a mano)."""
import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.splitlines(keepends=True)})

# ---------------------------------------------------------------- PORTADA
md(r"""# Pipeline metodológico del metamodelo PINN
### Versión secuencial y factorizada — desde el modelo FEM hasta el entrenamiento

**Tesis:** Desarrollo de un metamodelo híbrido basado en PINNs para la aproximación del análisis dinámico de edificios tipo bajo normativa chilena NCh433.

---

## Descripción general

Este notebook reúne, en un solo lugar y de forma lineal, todo el procedimiento que en
producción está repartido entre varios notebooks y *workers* paralelos. La idea es que
se pueda **leer el método de punta a punta**: se toman los parámetros de un edificio de
la Familia B y se recorre el flujo completo, modelo de elementos finitos → análisis modal
→ clasificación de modos → respuesta espectral NCh433 → ensamblaje del caso → metamodelo
PINN → entrenamiento.

Para privilegiar la legibilidad se omiten tres cosas que sí existen en producción y que no
aportan al método: la paralelización en 12 *workers*, el manejo de archivos y el control de
errores. Cada bloque queda como una función independiente, de manera que el notebook sirve
a la vez como referencia y como anexo de código.

El procedimiento de análisis modal espectral corresponde a la versión **definitiva (v6.1)**,
conforme a NCh433:2026:

- $\alpha$ evaluado con $(T_0, p)$ del espectro de diseño (Ec. 9),
- factor de reducción $R^\ast$ construido con $R_0$ (Ec. 10),
- límites $C_{\min}/C_{\max}$ aplicados al **corte basal combinado** (no por ordenada),
  y $C_{\max}$ **sin afectar los desplazamientos**.

> **Sobre el dataset.** La base de datos de 10.000 casos **no se regenera aquí**: su
> construcción tomó del orden de horas de cómputo OpenSees en paralelo. En la Sección 10
> simplemente se **carga** el archivo HDF5 definitivo y se inspecciona su estructura, que es
> exactamente lo que consume el metamodelo. Las funciones de las Secciones 5 a 9 son las que
> *generaron* ese archivo, y se incluyen para dejar trazable el origen de cada campo.

### Orden del notebook
| # | Sección | Rol |
|---|---------|-----|
| 0 | Instalación de dependencias | autónomo (Colab) |
| 1 | *Imports* | librerías base |
| 2 | Constantes NCh433:2026 | parámetros normativos |
| 3 | Espectro de diseño | $\alpha$, $R^\ast$, $S_a$, CQC |
| 4 | Modelo FEM (OpenSees) | edificio de muros, diafragma rígido |
| 5 | Análisis modal adaptativo | número de modos según altura |
| 6 | Clasificación en 18 slots | patrón Y–T–X |
| 7 | Respuesta espectral + torsión | desplazamientos, deriva, corte basal |
| 8 | Ensamblaje del caso | *padding* a 18 pisos + máscara |
| 8b | **Demo FEM en vivo** | un edificio de punta a punta (OpenSees) |
| 9 | El dataset | estructura y carga (no se regenera) |
| 10 | Metamodelo `PINNModal_v4` | arquitectura |
| 11 | Pérdida híbrida | datos + física |
| 12 | Currículo de pesos $\lambda$ | activación progresiva |
| 13 | Entrenamiento | Adam + *warm restarts* + dos velocidades |
| 14 | **Demo del metamodelo en vivo** | *forward* + pérdida + currículo |
| 15 | Esquema de producción | orquestación completa |
""")

# ---------------------------------------------------------------- 0 DEPS
md(r"""## 0. Instalación de dependencias

Este notebook es **autónomo**: ejecutándolo de principio a fin (*Entorno de ejecución →
Ejecutar todo* en Colab) se corre el pipeline completo. En Colab solo falta `openseespy`;
`numpy`, `pandas`, `h5py` y `torch` ya vienen preinstalados (por eso **no** se reinstala
`torch`: forzarlo reemplazaría la versión con CUDA y se perdería la GPU).

La única celda que puede no ejecutarse es la carga del dataset de 10.000 casos (Sección 9),
porque ese archivo no se distribuye con el notebook; en su lugar, la demostración del
metamodelo (Sección 14) corre sobre datos sintéticos con las dimensiones reales.""")
code(r"""# Colab: solo falta openseespy (el resto ya viene preinstalado).
%pip install -q openseespy
# Entorno local desde cero: descomentar la linea siguiente.
# %pip install -q numpy pandas h5py torch openseespy""")

# ---------------------------------------------------------------- 1 IMPORTS
md(r"""## 1. *Imports*

El núcleo del metamodelo solo necesita NumPy y PyTorch. `openseespy` y `pandas` se importan
**dentro** de las funciones del FEM, de modo que el resto del notebook (red, pérdida,
entrenamiento) corra aunque OpenSees no esté disponible en el entorno.""")
code(r"""import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F""")

# ---------------------------------------------------------------- 2 CONSTANTES
md(r"""## 2. Constantes y parámetros NCh433:2026

Parámetros normativos y de discretización usados en todo el pipeline. Los arreglos por
piso tienen tamaño fijo de 18 (`N_MAX_PISOS`); los edificios más bajos se rellenan con
ceros y se enmascaran (Sección 8). El esquema de modos es **adaptativo**: se extraen
$3N$ modos según la altura del edificio, de los cuales se almacenan 18 *slots*.""")
code(r"""G            = 9.80665     # gravedad [m/s2]
I_IMP        = 1.00        # factor de importancia
RO           = 11.0        # factor de reduccion R0 (muros)
ZETA         = 0.05        # amortiguamiento (CQC)
DRIFT_LIMIT  = 0.002       # limite de deriva de entrepiso (NCh433 5.9)
ETA_CRACKED  = 0.35        # rigidez agrietada DS61 (Ie = 0.35 Ig)
N_MAX_PISOS  = 18          # tamano fijo de los arreglos por piso
N_MODOS      = 18          # numero de slots modales almacenados
N_MODOS_PHYS = 18          # modos usados en las perdidas fisicas

A0_ZONA = {1: 0.20, 2: 0.30, 3: 0.40}                       # zona sismica -> A0 [g]
SUELO   = {  # (S, T0, p) canonicos del espectro de diseno NCh433
    'A': (0.90, 0.15, 2.00), 'B': (1.00, 0.30, 1.50),
    'C': (1.05, 0.40, 1.60), 'D': (1.20, 0.75, 1.00),
}
# numero de modos extraidos segun la altura (esquema adaptativo)
N_MODOS_POR_PISOS = {n: 3 * n for n in range(6, 19)}        # N=6->18, ..., N=18->54""")

# ---------------------------------------------------------------- 3 ESPECTRO
md(r"""## 3. Espectro de diseño NCh433:2026

La ordenada espectral elástica reducida se calcula como

$$S_a(T) = \frac{I\, S\, A_0\, \alpha(T)}{R^\ast(T^\ast)}\, g,
\qquad
\alpha(T) = \frac{1 + 4.5\,(T/T_0)^{p}}{1 + (T/T_0)^{3}},
\qquad
R^\ast = 1 + \frac{T^\ast}{0.10\,T_0 + T^\ast/R_0}.$$

Dos detalles de la versión definitiva:

1. $\alpha$ se evalúa con los $(T_0, p)$ **del suelo** (no con el período de cada modo), y
   $R^\ast$ usa $R_0$. Esta fue la corrección de *alpha* respecto de la versión previa.
2. $S_a$ **no se capa por ordenada**. Los límites $C_{\min}/C_{\max}$ se aplican más adelante
   sobre el corte basal combinado (Sección 7).

La combinación modal usa CQC con $\zeta = 5\%$.""")
code(r'''def alpha_nch433(T, T0, p):
    x = np.asarray(T, dtype=float) / T0
    return (1.0 + 4.5 * x**p) / (1.0 + x**3)               # factor de amplificacion (Ec. 9)


def Rstar_nch433(T_star, T0, Ro=RO):
    return 1.0 + T_star / (0.10 * T0 + T_star / Ro)         # factor de reduccion (Ec. 10)


def Sa_diseno(T, T_star, zona, suelo):
    """Ordenada espectral elastica reducida [m/s2], SIN tope (el tope va al corte)."""
    S, T0, p = SUELO[suelo]; A0 = A0_ZONA[zona]
    Sa_g = S * A0 * alpha_nch433(T, T0, p) * I_IMP / Rstar_nch433(T_star, T0)
    return Sa_g * G


def cqc_rho(wi, wj, zeta=ZETA):
    if wi <= 0 or wj <= 0:
        return 0.0
    b   = wj / wi
    num = 8 * zeta**2 * (1 + b) * b**1.5
    den = (1 - b**2)**2 + 4 * zeta**2 * b * (1 + b)**2
    return num / den if den > 0 else 0.0


def cqc_combine(values, omegas, zeta=ZETA):
    """Combinacion cuadratica completa (CQC) de respuestas modales (NCh433 6.3.5)."""
    values = np.asarray(values, dtype=float); omegas = np.asarray(omegas, dtype=float)
    s = 0.0
    for i in range(len(values)):
        for j in range(len(values)):
            s += cqc_rho(omegas[i], omegas[j], zeta) * values[i] * values[j]
    return float(np.sqrt(max(s, 0.0)))''')

# ---------------------------------------------------------------- 4 FEM
md(r"""## 4. Modelo FEM en OpenSees

Cada edificio se modela con muros como `elasticBeamColumn` de sección agrietada
($I_e = 0.35\,I_g$, DS61), un nodo maestro por piso y diafragma rígido. Quedan 3 GDL
libres por piso ($U_x$, $U_y$, $R_z$). La masa traslacional incluye la sobrecarga con
factor $\psi$; la masa rotacional usa el radio de giro del piso.""")
code(r"""def construir_modelo(params, wall_df):
    import openseespy.opensees as ops
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    N  = int(params['N_pisos']); h = float(params['h_story_m'])
    E  = float(params['E_MPa']) * 1e6; nu = 0.20; Gm = E / (2 * (1 + nu))
    A  = float(params['A_geom_m2']); gk = float(params['gk_kN_m2'])
    qk = float(params.get('qk_kN_m2', 2.0)); psi_f = float(params.get('psi_floor', 0.25))

    # masa traslacional y rotacional por piso
    m_floor = A * (gk + psi_f * qk) * 1000.0 / G
    floor_masses = np.full(N, m_floor); floor_masses[-1] = A * gk * 1000.0 / G
    Lx = float(params['Lx_m']); Ly = float(params['Ly_m'])
    xCM, yCM = Lx / 2.0, Ly / 2.0

    master = {}; tag = 1
    for piso in range(1, N + 1):
        ops.node(tag, xCM, yCM, piso * h); master[piso] = tag
        m = float(floor_masses[piso - 1]); I_rot = m * (Lx**2 + Ly**2) / 12.0
        ops.mass(tag, m, m, 0.0, 0.0, 0.0, I_rot); tag += 1

    base = tag; ops.node(base, xCM, yCM, 0.0)
    ops.fix(base, 1, 1, 1, 1, 1, 1); tag += 1

    elem = 1; wall_nodes = {}
    for _, wall in wall_df.iterrows():
        cx, cy = float(wall['cx']), float(wall['cy'])
        w, hw  = float(wall['w']), float(wall['h'])
        if w >= hw: orient, L_w, t_w = 'X', w, hw          # muro en X
        else:       orient, L_w, t_w = 'Y', hw, w          # muro en Y
        A_sec = L_w * t_w; I_fuerte = t_w * L_w**3 / 12; I_debil = L_w * t_w**3 / 12
        J = (t_w * L_w**3 / 3) * 0.01

        nb = tag; ops.node(nb, cx, cy, 0.0); ops.fix(nb, 1, 1, 1, 1, 1, 1); tag += 1
        n_prev = nb
        for piso in range(1, N + 1):
            n_curr = tag; ops.node(n_curr, cx, cy, piso * h); tag += 1
            wall_nodes[(wall['label'], piso)] = (n_prev, n_curr)
            tr = elem * 10
            ops.geomTransf('Linear', tr, *((1, 0, 0) if orient == 'Y' else (0, 1, 0)))
            ops.element('elasticBeamColumn', elem, n_prev, n_curr,
                        A_sec, ETA_CRACKED * E, ETA_CRACKED * Gm, J,
                        I_debil, I_fuerte, tr)
            elem += 1; n_prev = n_curr

    # diafragma rigido: nodo maestro por piso esclaviza los nodos de muro
    for piso in range(1, N + 1):
        slaves = [n_sup for (_, p), (_, n_sup) in wall_nodes.items() if p == piso]
        if slaves: ops.rigidDiaphragm(3, master[piso], *slaves)
    for n in master.values():
        ops.fix(n, 0, 0, 1, 1, 1, 0)                       # libera Ux, Uy, Rz

    return dict(master=master, floor_masses=floor_masses, N=N, h=h)""")

# ---------------------------------------------------------------- 5 MODAL
md(r"""## 5. Análisis modal adaptativo

Se resuelve el problema de valores propios con `-fullGenLapack`, extrayendo un número de
modos proporcional a la altura ($3N$, limitado a $3N$ GDL). Se arma la matriz de formas
modales $\Phi$ con $[U_x, U_y, R_z]$ por piso y se calcula la **masa modal efectiva
participativa** por dirección. El caso satisface el requisito normativo si acumula
$\geq 90\%$ de masa participativa en ambas direcciones.""")
code(r"""def analisis_modal(model_info, params, n_modes=None):
    import openseespy.opensees as ops
    N = model_info['N']; master = model_info['master']
    fm = model_info['floor_masses']; M_total = float(fm.sum())
    Lx = float(params['Lx_m']); Ly = float(params['Ly_m'])
    I_rot = fm * (Lx**2 + Ly**2) / 12.0

    if n_modes is None:
        n_modes = N_MODOS_POR_PISOS.get(N, 30)
    n_modes = min(n_modes, 3 * N)

    lamda = np.array(ops.eigen('-fullGenLapack', n_modes), dtype=float)
    if np.any(lamda <= 0):
        raise RuntimeError('eigenvalores no positivos (modelo mal condicionado)')
    omegas = np.sqrt(lamda); T = 2 * np.pi / omegas

    # ensamblaje de formas modales: [Ux, Uy, Rz] por piso
    Phi = np.zeros((3 * N, n_modes))
    for r in range(n_modes):
        for piso in range(1, N + 1):
            n = master[piso]; idx = 3 * (piso - 1)
            Phi[idx,     r] = ops.nodeEigenvector(n, r + 1, 1)   # Ux
            Phi[idx + 1, r] = ops.nodeEigenvector(n, r + 1, 2)   # Uy
            Phi[idx + 2, r] = ops.nodeEigenvector(n, r + 1, 6)   # Rz

    # masa modal efectiva participativa por direccion
    mpx = np.zeros(n_modes); mpy = np.zeros(n_modes)
    for r in range(n_modes):
        Lx_r = sum(fm[p-1] * Phi[3*(p-1),   r] for p in range(1, N+1))
        Ly_r = sum(fm[p-1] * Phi[3*(p-1)+1, r] for p in range(1, N+1))
        Mn   = sum(fm[p-1]*(Phi[3*(p-1),r]**2 + Phi[3*(p-1)+1,r]**2)
                   + I_rot[p-1]*Phi[3*(p-1)+2,r]**2 for p in range(1, N+1))
        Mn   = max(Mn, 1e-30)
        mpx[r] = Lx_r**2 / (Mn * M_total); mpy[r] = Ly_r**2 / (Mn * M_total)
    mpx = np.clip(mpx, 0, 1); mpy = np.clip(mpy, 0, 1)

    return dict(T=T, omegas=omegas, Phi=Phi, mass_part_x=mpx, mass_part_y=mpy,
                I_rot_floors=I_rot, floor_masses=fm, M_total=M_total,
                cumple_90=(mpx.sum() >= 0.90 and mpy.sum() >= 0.90))""")

# ---------------------------------------------------------------- 6 CLASIFICACION
md(r"""## 6. Clasificación y ordenamiento en 18 *slots*

Cada modo se clasifica por energía: traslacional en X, traslacional en Y o torsional
(si la energía rotacional supera el 40% del total). Los modos se ordenan en 18 *slots*
con el patrón repetido **Y–T–X**, que fija una convención estable para el metamodelo. El
caso es válido si los *slots* 1 y 4 ($T_1$ y $T_2$) quedan ocupados.""")
code(r"""def clasificar_modos(modal, N, n_out=N_MODOS):
    Phi = modal['Phi']; fm = modal['floor_masses']; I_rot = modal['I_rot_floors']
    n_modes = len(modal['T']); tipos = []
    for r in range(n_modes):
        ex = sum(fm[p-1]    * Phi[3*(p-1),   r]**2 for p in range(1, N+1))
        ey = sum(fm[p-1]    * Phi[3*(p-1)+1, r]**2 for p in range(1, N+1))
        et = sum(I_rot[p-1] * Phi[3*(p-1)+2, r]**2 for p in range(1, N+1))
        total = ex + ey + et + 1e-30
        tipos.append('torsional' if et/total > 0.40 else ('trans_X' if ex >= ey else 'trans_Y'))

    idx_y = [r for r, t in enumerate(tipos) if t == 'trans_Y']
    idx_t = [r for r, t in enumerate(tipos) if t == 'torsional']
    idx_x = [r for r, t in enumerate(tipos) if t == 'trans_X']

    slots = []
    for k in range(n_out // 3):                             # patron Y, T, X repetido
        slots += [idx_y[k] if k < len(idx_y) else None,
                  idx_t[k] if k < len(idx_t) else None,
                  idx_x[k] if k < len(idx_x) else None]
    valido = (slots[1] is not None) and (slots[4] is not None)  # T1 y T2 presentes

    T18  = np.zeros(n_out); om18 = np.zeros(n_out)
    mx18 = np.zeros(n_out); my18 = np.zeros(n_out)
    Px = np.zeros((N_MAX_PISOS, n_out)); Py = np.zeros((N_MAX_PISOS, n_out))
    Pt = np.zeros((N_MAX_PISOS, n_out))
    for s, src in enumerate(slots):
        if src is None:
            continue
        T18[s], om18[s] = modal['T'][src], modal['omegas'][src]
        mx18[s], my18[s] = modal['mass_part_x'][src], modal['mass_part_y'][src]
        for piso in range(N):
            Px[piso, s] = Phi[3*piso,     src]
            Py[piso, s] = Phi[3*piso + 1, src]
            Pt[piso, s] = Phi[3*piso + 2, src]
    return dict(T_18=T18, omega_18=om18, Phi_x=Px, Phi_y=Py, Phi_t=Pt,
                mpart_x=mx18, mpart_y=my18, valido=valido)""")

# ---------------------------------------------------------------- 7 RESPUESTA
md(r"""## 7. Respuesta espectral + torsión accidental

Procedimiento definitivo (v6). Los desplazamientos de piso se obtienen por superposición
modal y CQC. El corte basal se combina sin capar y luego se acotan con los límites
$C_{\min}/C_{\max}$ sobre el **corte combinado**:

$$C_{\min} = \frac{S\,A_0}{6}, \qquad C_{\max} = 0.35\,S\,A_0, \qquad P = M_\text{tot}\,g.$$

$C_{\min}$ escala los desplazamientos si el corte queda por debajo del mínimo; $C_{\max}$
**no toca** los desplazamientos. Se aplica torsión accidental amplificando en el vértice
($e_\text{acc} = 0.05\,L$ perpendicular al sismo) y se calculan las derivas de entrepiso,
que se comparan contra el límite normativo.""")
code(r"""def respuesta_espectral(params, modal):
    N = int(params['N_pisos']); h = float(params['h_story_m'])
    Lx = float(params['Lx_m']); Ly = float(params['Ly_m'])
    T = modal['T']; omegas = modal['omegas']; Phi = modal['Phi']
    fm = modal['floor_masses']; M_total = modal['M_total']; K = len(T)
    S, T0, p = SUELO[params['suelo']]; A0 = A0_ZONA[params['zona']]

    T_star_x = T[np.argmax(modal['mass_part_x'])]
    T_star_y = T[np.argmax(modal['mass_part_y'])]
    Sa_x = Sa_diseno(T, T_star_x, params['zona'], params['suelo'])
    Sa_y = Sa_diseno(T, T_star_y, params['zona'], params['suelo'])
    Sd_x = Sa_x / np.maximum(omegas**2, 1e-6)
    Sd_y = Sa_y / np.maximum(omegas**2, 1e-6)

    # desplazamientos de piso por superposicion modal + CQC (Mn traslacional)
    Ux = np.zeros(N); Uy = np.zeros(N)
    for piso in range(1, N + 1):
        Mx, My = [], []
        for r in range(K):
            Mn = sum(fm[q-1]*(Phi[3*(q-1),r]**2 + Phi[3*(q-1)+1,r]**2) for q in range(1, N+1))
            Mn = max(Mn, 1e-30)
            Gx = sum(fm[q-1]*Phi[3*(q-1),   r] for q in range(1, N+1)) / Mn
            Gy = sum(fm[q-1]*Phi[3*(q-1)+1, r] for q in range(1, N+1)) / Mn
            Mx.append(Gx * Phi[3*(piso-1),   r] * Sd_x[r])
            My.append(Gy * Phi[3*(piso-1)+1, r] * Sd_y[r])
        Ux[piso-1] = cqc_combine(Mx, omegas); Uy[piso-1] = cqc_combine(My, omegas)

    # corte basal sin capar (CQC) y limites Cmin/Cmax sobre el corte COMBINADO
    Vb_unc_x = cqc_combine(modal['mass_part_x'] * M_total * Sa_x, omegas)
    Vb_unc_y = cqc_combine(modal['mass_part_y'] * M_total * Sa_y, omegas)
    Cmin = S * A0 / 6.0; Cmax = 0.35 * S * A0; P = M_total * G
    Vmin = Cmin * I_IMP * P; Vmax = Cmax * I_IMP * P
    Vb_x = min(max(Vb_unc_x, Vmin), Vmax); Vb_y = min(max(Vb_unc_y, Vmin), Vmax)

    # Cmin escala los desplazamientos si el corte queda por debajo; Cmax NO los toca
    Ux *= max(1.0, Vmin / max(Vb_unc_x, 1e-9))
    Uy *= max(1.0, Vmin / max(Vb_unc_y, 1e-9))

    # torsion accidental: amplificacion en el vertice (e_acc perpendicular al sismo)
    r_giro = np.sqrt((Lx**2 + Ly**2) / 12.0)
    Ux_t = np.abs(Ux) * (1 + 0.05 * Ly / r_giro)
    Uy_t = np.abs(Uy) * (1 + 0.05 * Lx / r_giro)

    # derivas de entrepiso
    dx = np.empty(N); dy = np.empty(N); dx[0] = Ux_t[0] / h; dy[0] = Uy_t[0] / h
    dx[1:] = np.abs(np.diff(Ux_t)) / h; dy[1:] = np.abs(np.diff(Uy_t)) / h
    cumple = bool(dx.max() <= DRIFT_LIMIT and dy.max() <= DRIFT_LIMIT and modal['cumple_90'])

    return dict(Ux_m=Ux_t, Uy_m=Uy_t, drift_x=dx, drift_y=dy,
                Vb_x_kN=Vb_x/1000.0, Vb_y_kN=Vb_y/1000.0, cumple=cumple)""")

# ---------------------------------------------------------------- 8 ENSAMBLAJE
md(r"""## 8. Ensamblaje del caso: *padding* y máscara

Todos los arreglos por piso se rellenan a 18 posiciones y se acompaña una **máscara
binaria** (`mask_floors`): 1 en los pisos reales, 0 en el relleno. Esta máscara es la que
luego anula el aporte de los pisos inexistentes en la red y en la pérdida. La suma de la
máscara es exactamente `N_pisos`.""")
code(r"""def ensamblar_caso(inputs_18, modal18, respuesta, N):
    def pad(arr):
        out = np.zeros(N_MAX_PISOS); out[:N] = np.asarray(arr)[:N]; return out
    mask = np.zeros(N_MAX_PISOS); mask[:N] = 1.0                # 1 = piso real, 0 = relleno
    return dict(
        X=np.asarray(inputs_18, dtype=np.float32),             # 18 variables de diseno
        mask_floors=mask.astype(np.float32), N_pisos=N,
        T_r=modal18['T_18'], Phi_x=modal18['Phi_x'],
        Phi_y=modal18['Phi_y'], Phi_theta=modal18['Phi_t'],
        Ux_m=pad(respuesta['Ux_m']), Uy_m=pad(respuesta['Uy_m']),
        drift_x=pad(respuesta['drift_x']), drift_y=pad(respuesta['drift_y']),
        Vb_x_kN=respuesta['Vb_x_kN'], Vb_y_kN=respuesta['Vb_y_kN'],
        cumple=int(respuesta['cumple']),
    )""")

# ---------------------------------------------------------------- 8b DEMO FEM
md(r"""## 8b. Demostración: un edificio de punta a punta (OpenSees en vivo)

Las Secciones 4 a 8 son las que *generan* cada caso. Aquí se ejecutan sobre **un edificio
de ejemplo** de la Familia B, con una disposición de muros sintética (`muros_demo`). Si
`openseespy` no está disponible en el entorno, la celda lo informa y continúa: el metamodelo
de las secciones siguientes no depende de OpenSees.""")
code(r'''def muros_demo(params):
    """Disposicion de muros simetrica de ejemplo (en X e Y) para un edificio Familia B."""
    import pandas as pd
    Lx, Ly, t = params['Lx_m'], params['Ly_m'], 0.25
    filas = []
    for j, cy in enumerate([0.0, Ly]):                      # muros largos en X
        filas.append(dict(label=f'MX{j}', cx=Lx/2, cy=cy, w=0.6*Lx, h=t))
    for i, cx in enumerate([0.0, Lx]):                      # muros largos en Y
        filas.append(dict(label=f'MY{i}', cx=cx, cy=Ly/2, w=t, h=0.6*Ly))
    filas.append(dict(label='NX', cx=Lx/2, cy=Ly/2, w=0.2*Lx, h=t))   # nucleo
    filas.append(dict(label='NY', cx=Lx/2, cy=Ly/2, w=t, h=0.2*Ly))
    return pd.DataFrame(filas)


# parametros de un edificio de ejemplo (reutilizados en el driver final)
params = dict(N_pisos=12, h_story_m=2.7, E_MPa=23500.0, A_geom_m2=320.0,
              gk_kN_m2=8.0, qk_kN_m2=2.0, Lx_m=36.0, Ly_m=12.0, zona=3, suelo='C')

try:
    walls   = muros_demo(params)
    modelo  = construir_modelo(params, walls)
    modal   = analisis_modal(modelo, params)
    modal18 = clasificar_modos(modal, params['N_pisos'])
    resp    = respuesta_espectral(params, modal)
    caso    = ensamblar_caso(np.zeros(18, np.float32), modal18, resp, params['N_pisos'])
    dmax = max(resp['drift_x'].max(), resp['drift_y'].max())
    print(f"Edificio demo: N={params['N_pisos']}  zona={params['zona']}  suelo={params['suelo']}")
    print(f"  T1 = {modal['T'][0]:.3f} s")
    print(f"  masa participativa acumulada:  X={modal['mass_part_x'].sum():.2f}  Y={modal['mass_part_y'].sum():.2f}")
    print(f"  Vb_x = {resp['Vb_x_kN']:.1f} kN   Vb_y = {resp['Vb_y_kN']:.1f} kN")
    print(f"  deriva maxima = {dmax:.5f}  (limite {DRIFT_LIMIT})   ->  cumple = {resp['cumple']}")
    print(f"  caso ensamblado: X{caso['X'].shape}, mask.sum={int(caso['mask_floors'].sum())} pisos reales")
except Exception as e:
    print('[info] OpenSees no disponible en este entorno; se omite la generacion en vivo.')
    print('      ', type(e).__name__, str(e)[:90])
    print('       El metamodelo (Secciones 10 en adelante) no depende de OpenSees y si se ejecuta.')''')

# ---------------------------------------------------------------- 9 DATASET
md(r"""## 9. El dataset: estructura y carga

Repetir las Secciones 4 a 8 sobre los 10.000 edificios de la Familia B produce el archivo
HDF5 que consume el metamodelo. Ese proceso es costoso (OpenSees en paralelo), por lo que
**aquí no se regenera**: se carga el archivo definitivo y se inspecciona su estructura.

El archivo `dataset_familyB_OPENSEES_v6_DEFINITIVO.h5` agrupa, para cada caso:

- `inputs/` — variables de diseño `X` (10000×18), `mask_floors`, masas de piso;
- `modal/` — períodos `T_r`, formas `Phi_x/Phi_y/Phi_theta` (10000×18×18), masas participativas, tipo de modo;
- `response/` — desplazamientos `Ux_m/Uy_m`, derivas `drift_x/drift_y`, corte basal `Vb_x_kN/Vb_y_kN`;
- `scalars/` — `N_pisos`, `cumple` (etiqueta normativa);
- `stiffness/` — `K_global`, `M_global` (10000×54×54) usadas por las pérdidas físicas.

> $K$ se almacena **reconstruida modalmente** ($K = M\Phi\,\Omega^2\,\Phi^\top M$) en `float32`,
> para garantizar la consistencia numérica $K\Phi = \omega^2 M\Phi$ que evalúa la pérdida del
> eigenproblema. $M$ es diagonal (masa concentrada por piso).""")
code(r'''import h5py

H5 = 'dataset_familyB_OPENSEES_v6_DEFINITIVO.h5'   # <-- ruta al dataset (si se dispone)

if os.path.exists(H5):
    with h5py.File(H5, 'r') as f:
        print(f"Casos: {f['inputs/X'].shape[0]}   |   cumplen NCh433: {int(f['scalars/cumple'][:].sum())}")
        print('-' * 60)
        def _show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f'{name:28s} {str(obj.shape):16s} {obj.dtype}')
        f.visititems(_show)
else:
    print(f'[info] No se encontro "{H5}" en este entorno.')
    print('       El dataset de 10.000 casos no se distribuye con el notebook (tamano/costo).')
    print('       La Seccion 14 entrena una demostracion sobre datos sinteticos de shape real.')''')

md(r"""Carga de un *batch* en tensores, en el formato exacto que espera el metamodelo. Las
respuestas (períodos, desplazamientos, corte) se entrenan en escala logarítmica, así que el
`DataLoader` de producción aplica esa transformación y la normalización con los *scalers*
ajustados sobre el conjunto de entrenamiento. Aquí solo se muestra la lectura cruda.""")
code(r"""def cargar_batch(h5_path=H5, idx=slice(0, 64), device='cpu'):
    with h5py.File(h5_path, 'r') as f:
        t = lambda k: torch.from_numpy(f[k][idx].astype('float32')).to(device)
        return {
            'X':        t('inputs/X'),         'mask':     t('inputs/mask_floors'),
            'T_r':      t('modal/T_r'),        'logT':     torch.log(t('modal/T_r').clamp(min=1e-6)),
            'Phi_x':    t('modal/Phi_x'),      'Phi_y':    t('modal/Phi_y'),
            'Phi_theta':t('modal/Phi_theta'),
            'Ux':       t('response/Ux_m'),    'Uy':       t('response/Uy_m'),
            'Dy':       t('response/drift_y'),
            'Vbx':      t('response/Vb_x_kN'), 'Vby':      t('response/Vb_y_kN'),
            'ratio':    t('response/drift_x') / t('response/drift_y').clamp(min=1e-9),
            'K_global': t('stiffness/K_global'), 'M_global': t('stiffness/M_global'),
        }

# batch = cargar_batch()      # descomentar para inspeccionar shapes""")

# ---------------------------------------------------------------- 10 RED
md(r"""## 10. Metamodelo `PINNModal_v4`

La red tiene dos *encoders* y cuatro cabezales especializados:

- **`encoder_T1`** (oculto 128) dedicado solo al período fundamental $T_1$, la variable más
  sensible;
- **`encoder`** compartido (oculto 256, cuatro `ResBlock`) que alimenta los cabezales de
  períodos restantes, formas modales, respuestas y corte basal.

Los períodos se predicen de forma **monótona decreciente**: $T_1$ explícito y luego
$T_{r+1} = T_r - \Delta_r$ con $\Delta_r = \text{softplus}(\cdot) \geq 0$. Las formas modales
se enmascaran por piso y se normalizan a norma unitaria. La deriva se parametriza como
$d_x = \text{ratio}\cdot d_y$ con $\text{ratio} \geq 0$.""")
code(r"""class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.SiLU(),
                                 nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))


class PINNModal_v4(nn.Module):
    def __init__(self, n_in=18, n_modos=18, n_pisos=18, hidden=256, hidden_T1=128):
        super().__init__()
        self.n_modos, self.n_pisos = n_modos, n_pisos
        self.encoder_T1 = nn.Sequential(nn.Linear(n_in, hidden_T1), nn.LayerNorm(hidden_T1),
                                        nn.SiLU(), ResBlock(hidden_T1), ResBlock(hidden_T1))
        self.head_T1 = nn.Sequential(nn.Linear(hidden_T1, 32), nn.SiLU(), nn.Linear(32, 1))
        self.encoder = nn.Sequential(nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                     ResBlock(hidden), ResBlock(hidden),
                                     ResBlock(hidden), ResBlock(hidden))
        self.head_T_rest = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(),
                                         nn.Linear(128, n_modos - 1))
        self.head_Phi  = nn.Sequential(nn.Linear(hidden, 512), nn.SiLU(),
                                       nn.Linear(512, n_modos * n_pisos * 3))
        self.head_resp = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(),
                                       nn.Linear(256, n_pisos * 4))
        self.head_Vb   = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, X, mask):
        B = X.shape[0]
        T1 = self.head_T1(self.encoder_T1(X))                  # encoder dedicado a T1
        h  = self.encoder(X)                                   # encoder compartido
        Td = F.softplus(self.head_T_rest(h))                   # periodos decrecientes T2..T18
        parts = [T1]
        for r in range(self.n_modos - 1):
            parts.append(parts[-1] - Td[:, r:r+1])
        logT = torch.cat(parts, dim=1)
        Phi = self.head_Phi(h).view(B, self.n_modos, self.n_pisos, 3)
        Phi = Phi * mask.unsqueeze(1).unsqueeze(-1)            # anula pisos de relleno
        Phi = Phi / Phi.norm(dim=2, keepdim=True).clamp(min=1e-8)
        Phi = Phi.permute(0, 2, 1, 3)
        resp = self.head_resp(h).view(B, self.n_pisos, 4)
        Ux = resp[:, :, 0] * mask; Uy = resp[:, :, 1] * mask
        ratio = F.softplus(resp[:, :, 2]) * mask; dy = resp[:, :, 3] * mask
        Vb = self.head_Vb(h)
        return logT, Phi[:, :, :, 0], Phi[:, :, :, 1], Phi[:, :, :, 2], Ux, Uy, ratio, dy, Vb""")

# ---------------------------------------------------------------- 11 PERDIDA
md(r"""## 11. Función de pérdida híbrida

La pérdida combina términos de **datos** (MSE enmascarado sobre desplazamientos, deriva,
corte basal y períodos) con tres términos de **física**, cada uno enmascarado por piso y
por modo válido:

- **$L_\phi$** — alineamiento de formas modales: $1 - \cos^2$ entre la forma predicha y la
  de referencia, promediado sobre modos válidos (equivale a $1 - \text{MAC}$);
- **$L_\text{eig}$** — consistencia del eigenproblema $K\Phi = \omega^2 M\Phi$, como residuo
  relativo;
- **$L_\text{ortho}$** — M-ortonormalidad de los modos, $\Phi^\top M \Phi = I$.

Los términos físicos se aplican **por muestra** y solo sobre los modos con $T_r > 10^{-6}$;
$K$ y $M$ se enmascaran por piso (3 GDL) y se normalizan por su escala, y $\Phi$ se
normaliza por modo. Esto evita que el relleno o las diferencias de magnitud dominen el
gradiente.""")
code(r'''def perdida_hibrida(batch, out, lambdas, stage, device='cuda'):
    logT_p, Phx_p, Phy_p, Pht_p, Ux_p, Uy_p, ratio_p, dy_p, Vb_p = out
    mask = batch['mask'].to(device)
    ld = {}

    # --- datos: desplazamientos, deriva, cortante (MSE enmascarado) ---
    ld['data_resp'] = (F.mse_loss(Ux_p * mask, batch['Ux'].to(device) * mask) +
                       F.mse_loss(Uy_p * mask, batch['Uy'].to(device) * mask) +
                       F.mse_loss(dy_p * mask, batch['Dy'].to(device) * mask) +
                       F.mse_loss(Vb_p[:, 0], batch['Vbx'].to(device)) +
                       F.mse_loss(Vb_p[:, 1], batch['Vby'].to(device)))
    n_act = mask.sum().clamp(min=1.0)
    ld['ratio'] = ((ratio_p - batch['ratio'].to(device)).pow(2) * mask).sum() / n_act

    # --- datos: periodos (T1 explicito + T2..T18 con mascara de modo valido) ---
    mask_T  = (batch['T_r'].to(device)[:, 1:] > 1e-6).float()
    n_valid = mask_T.sum().clamp(min=1.0)
    ld['data_T1']     = F.mse_loss(logT_p[:, 0], batch['logT'].to(device)[:, 0])
    ld['data_T_rest'] = ((logT_p[:, 1:] - batch['logT'].to(device)[:, 1:]).pow(2)
                         * mask_T).sum() / n_valid

    # --- fisica 1: alineamiento de formas modales (1 - coseno^2), por modo valido ---
    L_phi = torch.tensor(0.0, device=device)
    if stage.get('phi'):
        for r in range(N_MODOS):
            comp = [(Phy_p, batch['Phi_y']), (Pht_p, batch['Phi_theta']),
                    (Phx_p, batch['Phi_x'])][r % 3]
            pp = comp[0][:, :, r] * mask; pt = comp[1].to(device)[:, :, r] * mask
            num = (pp * pt).sum(1) ** 2
            den = ((pp**2).sum(1) * (pt**2).sum(1)).clamp(min=1e-12)
            val = (pt.pow(2).sum(1) > 1e-10).float()
            L_phi = L_phi + (val * (1 - num/den)).sum() / val.sum().clamp(min=1.0)
        L_phi = L_phi / N_MODOS
    ld['phi'] = L_phi

    # --- fisica 2: consistencia del eigenproblema  K*Phi = omega^2 * M*Phi ---
    L_eig = torch.tensor(0.0, device=device)
    if stage.get('eig'):
        K_b, M_b = _ensamblar_KM(batch, mask, device)
        Phi_g = _ensamblar_Phi_global(Phx_p, Phy_p, Pht_p, device)
        T_pred = torch.exp(logT_p[:, :N_MODOS_PHYS]).clamp(min=1e-6)   # (desnormalizado)
        w2 = (2 * math.pi / T_pred) ** 2
        KPhi = torch.bmm(K_b, Phi_g); MPhi = torch.bmm(M_b, Phi_g)
        res = KPhi - w2.unsqueeze(1) * MPhi
        den = KPhi.pow(2).sum(1) + (w2.unsqueeze(1) * MPhi).pow(2).sum(1) + 1e-12
        L_eig = (res.pow(2).sum(1) / den).mean()
    ld['eig'] = L_eig

    # --- fisica 3: M-ortogonalidad de los modos  Phi^T M Phi = I ---
    L_ortho = torch.tensor(0.0, device=device)
    if stage.get('ortho'):
        _, M_b = _ensamblar_KM(batch, mask, device)
        Phi_g = _ensamblar_Phi_global(Phx_p, Phy_p, Pht_p, device)
        Gmat = torch.bmm(Phi_g.transpose(1, 2), torch.bmm(M_b, Phi_g))
        I_m = torch.eye(Phi_g.shape[2], device=device).unsqueeze(0)
        L_ortho = ((Gmat - I_m) ** 2).mean()
    ld['ortho'] = L_ortho

    total = sum(lambdas.get(k, 0.0) * ld[k] for k in ld)
    return total, ld


def _ensamblar_KM(batch, mask, device):
    """Enmascara los pisos inexistentes en K y M (3 GDL/piso) y normaliza por escala."""
    B = mask.shape[0]
    mf = mask.unsqueeze(-1).expand(-1, -1, 3).reshape(B, 54)
    m2 = mf.unsqueeze(2) * mf.unsqueeze(1)
    K_b = batch['K_global'].to(device) * m2; M_b = batch['M_global'].to(device) * m2
    s = K_b.abs().amax(dim=(-2, -1)).clamp(min=1.0).view(B, 1, 1)
    return K_b / s, M_b / s


def _ensamblar_Phi_global(Phx, Phy, Pht, device):
    """Arma Phi global (B,54,R) intercalando [Ux,Uy,Rz] por piso; normaliza por modo."""
    B, _, R = Phx.shape; Phi = torch.zeros(B, 54, R, device=device)
    Phi[:, 0::3, :] = Phx; Phi[:, 1::3, :] = Phy; Phi[:, 2::3, :] = Pht
    return Phi / Phi.norm(dim=1, keepdim=True).clamp(min=1e-8)''')

# ---------------------------------------------------------------- 12 CURRICULO
md(r"""## 12. Currículo de pesos $\lambda$

Los términos físicos no se activan desde el inicio: primero la red aprende a reproducir los
datos y recién después se introduce la física, de forma escalonada. El alineamiento de
formas entra en la época 150, el eigenproblema en la 200 y la ortogonalidad en la 350. Esto
estabiliza el entrenamiento y evita que el residuo físico domine antes de que las
predicciones tengan sentido.""")
code(r"""def curriculo_lambda(epoch):
    lam = {'data_resp': 2.0, 'data_T_rest': 1.0, 'ratio': 1.0,
           'data_T1': 0.0, 'phi': 0.0, 'T1': 0.0, 'eig': 0.0, 'ortho': 0.0}
    stage = {'phi': False, 'eig': False, 'ortho': False}
    if epoch >= 150: lam['data_T1'] = 1.0; lam['phi'] = 0.1; stage['phi'] = True
    if epoch >= 200: lam['T1'] = 1.0; lam['eig'] = 0.1; stage['eig'] = True
    if epoch >= 350: lam['ortho'] = 1e-5; stage['ortho'] = True
    return lam, stage""")

# ---------------------------------------------------------------- 13 ENTRENAMIENTO
md(r"""## 13. Entrenamiento

Optimizador Adam ($lr = 10^{-3}$) con *scheduler* `CosineAnnealingWarmRestarts`
($T_0 = 200$, $T_\text{mult} = 2$). Esquema de **dos velocidades**: el `encoder_T1` se
mantiene congelado las primeras 150 épocas, de modo que el período fundamental se ajuste
una vez que el resto de la red ya capturó la tendencia gruesa. Se recorta la norma del
gradiente a 1.0 en cada paso. El entrenamiento de producción corrió 600 épocas
(~26 min en una RTX 3070 Ti).""")
code(r"""def entrenar(model, loader, epochs=600, device='cuda'):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=200, T_mult=2, eta_min=1e-6)
    for ep in range(epochs):
        lam, stage = curriculo_lambda(ep)
        # dos velocidades: encoder_T1 congelado las primeras 150 epocas
        for prm in model.encoder_T1.parameters():
            prm.requires_grad = (ep >= 150)
        for batch in loader:
            opt.zero_grad()
            out = model(batch['X'].to(device), batch['mask'].to(device))
            loss, _ = perdida_hibrida(batch, out, lam, stage, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model""")

# ---------------------------------------------------------------- 14 DEMO ENTRENAMIENTO
md(r"""## 14. Demostración del metamodelo en vivo

Como el dataset real no se distribuye con el notebook, esta sección entrena el metamodelo
sobre un *batch* **sintético con las dimensiones reales** (18 modos, 18 pisos, $K/M$ de
$54\times54$). El objetivo no es ajustar un modelo útil, sino **comprobar que toda la
maquinaria corre de punta a punta**: el paso *forward*, la pérdida híbrida y el currículo de
activación de los términos físicos. Se recorren las cuatro fases del currículo y se imprime
cómo entran en juego $L_\phi$ (época 150), $L_\text{eig}$ (200) y $L_\text{ortho}$ (350).

El entrenamiento de producción usa exactamente estas mismas funciones, pero sobre los 10.000
casos del HDF5, 600 épocas y GPU.""")
code(r'''def batch_demo(B=32, device='cpu'):
    """Batch sintetico con las shapes reales que consume el metamodelo."""
    torch.manual_seed(0)
    mask = torch.zeros(B, 18)
    npis = torch.randint(6, 19, (B,))
    for i in range(B):
        mask[i, :npis[i]] = 1.0
    base = torch.linspace(1.2, 0.05, 18).unsqueeze(0)
    T_r  = (base * (0.8 + 0.4 * torch.rand(B, 1))).clamp(min=1e-3)
    Md = torch.zeros(B, 54, 54); d = torch.arange(54)
    Md[:, d, d] = torch.rand(B, 54) + 0.5                  # M diagonal positiva
    A = torch.randn(B, 54, 54); K = A @ A.transpose(1, 2) + torch.eye(54)   # K simetrica
    t = lambda x: x.to(device)
    return {
        'X': t(torch.randn(B, 18)), 'mask': t(mask),
        'T_r': t(T_r), 'logT': t(torch.log(T_r)),
        'Phi_x': t(torch.randn(B, 18, 18)), 'Phi_y': t(torch.randn(B, 18, 18)),
        'Phi_theta': t(torch.randn(B, 18, 18)),
        'Ux': t(torch.randn(B, 18)), 'Uy': t(torch.randn(B, 18)),
        'Dy': t(torch.randn(B, 18)), 'ratio': t(torch.rand(B, 18)),
        'Vbx': t(torch.randn(B)), 'Vby': t(torch.randn(B)),
        'K_global': t(K), 'M_global': t(Md),
    }


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = PINNModal_v4().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
batch  = batch_demo(32, device)

print(f'PINNModal_v4: {sum(p.numel() for p in model.parameters()):,} parametros  |  device={device}')
print('-' * 70)
print(f"{'epoca':>6} {'fase':>22} {'loss':>10} {'L_phi':>8} {'L_eig':>8} {'L_ortho':>9}")
for ep in (0, 160, 210, 360):
    lam, stage = curriculo_lambda(ep)
    out = model(batch['X'], batch['mask'])
    loss, ld = perdida_hibrida(batch, out, lam, stage, device)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    fase = {0: 'solo datos', 160: '+ L_phi', 210: '+ L_eig', 360: '+ L_ortho'}[ep]
    print(f"{ep:>6} {fase:>22} {float(loss):>10.4f} "
          f"{float(ld['phi']):>8.3f} {float(ld['eig']):>8.3f} {float(ld['ortho']):>9.4f}")''')

# ---------------------------------------------------------------- 15 DRIVER
md(r"""## 15. Esquema de producción

El entrenamiento real reemplaza el *batch* sintético por un `DataLoader` sobre el HDF5 de la
Sección 9 y corre 600 épocas con el currículo y el esquema de dos velocidades de la Sección
13. La estructura completa es la misma que la demostrada arriba:

```
for caso in 10.000 edificios:        # Secciones 4-8 (OpenSees, una vez -> HDF5)
    modelo  = construir_modelo(params, muros(caso))
    modal   = analisis_modal(modelo, params)
    modal18 = clasificar_modos(modal, N)
    resp    = respuesta_espectral(params, modal)
    guardar(ensamblar_caso(X_18, modal18, resp, N))

loader = DataLoader(HDF5)             # Seccion 9
model  = entrenar(PINNModal_v4(), loader, epochs=600, device='cuda')   # Secciones 10-13
```""")
code(r"""# Entrenamiento de produccion (descomentar con el dataset real disponible):
# from torch.utils.data import DataLoader
# loader = DataLoader(MiDatasetHDF5(H5), batch_size=64, shuffle=True)
# model  = entrenar(PINNModal_v4().to(device), loader, epochs=600, device=device)
print('Pipeline de referencia cargado de punta a punta.')""")

# ---------------------------------------------------------------- ESCRIBIR
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = r'c:\Users\rodri\Documents\NB\Pipeline_PINN_Metamodelo.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('OK ->', out, '|', len(cells), 'celdas')
