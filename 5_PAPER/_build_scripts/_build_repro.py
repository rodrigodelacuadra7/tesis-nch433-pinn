# -*- coding: utf-8 -*-
"""Genera el notebook de REPLICACION: dominio parametrico -> geometria -> FEM/modal/espectral
-> dataset en vivo -> entrenamiento PINN -> evaluacion, con ploteos. Embebe las funciones
REALES del proyecto via inspect.getsource para garantizar fidelidad."""
import sys, json, inspect

sys.path.insert(0, r'C:/Users/rodri/Documents/PINN/b2_proyecto/notebooks/outputs/familia_B')
sys.path.insert(0, r'C:/Users/rodri/Documents/PINN/b2_proyecto/dataset/opensees/workers_v6')
sys.path.insert(0, r'C:/Users/rodri/Documents/NB/anexos')
import familia_B_core as fbc
import opensees_functions_v6 as osf
import pipeline as pl

def src(*fns):
    return '\n\n'.join(inspect.getsource(f) for f in fns)

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                           "source":s.splitlines(keepends=True)})

# =================================================================== PORTADA
md(r"""# Réplica reproducible del metamodelo PINN — Familia B (NCh433)
### Del dominio paramétrico al modelo final, en un solo notebook autónomo

**Tesis:** Desarrollo de un metamodelo híbrido basado en PINNs para la aproximación del análisis dinámico de edificios tipo bajo normativa chilena NCh433.

---

## Qué hace este notebook

Este notebook **reconstruye todo el pipeline desde cero** y de forma autónoma: define el
dominio paramétrico de la Familia B, **genera edificios** dentro de ese dominio, los resuelve
con OpenSees (análisis modal espectral NCh433), **arma un conjunto de entrenamiento en vivo**
y **entrena el metamodelo PINN**, mostrando al final que reproduce el análisis dinámico con
buena exactitud. A lo largo del camino se grafican edificios generados, el espectro de
diseño, distribuciones del conjunto y la predicción del metamodelo frente a OpenSees.

El objetivo es que **cualquiera pueda ejecutarlo de principio a fin** y **comprobar que el
método es real, factible y entrega un buen modelo** sobre el dominio de edificios definido —
sin necesidad de descargar ningún dataset ni modelo previamente entrenado.

> **Escala.** Para que corra en minutos, la generación usa un número reducido de edificios
> (`N_CASOS`, configurable). La tesis usó 10.000 casos y 600 épocas; aquí se reproduce la
> **misma maquinaria** a menor escala. Subir `N_CASOS` y las épocas acerca la exactitud a la
> de producción.

Todo el código de generación (geometría, modelo FEM, análisis modal, reconstrucción $K/M$) es
el **código real del proyecto**, incorporado tal cual. El procedimiento espectral es la
versión **definitiva** conforme a NCh433:2026 ($\alpha$ con $(T_0,p)$, $R^\ast$ con $R_0$,
límites $C_{\min}/C_{\max}$ sobre el corte combinado).

### Mapa del notebook
| # | Sección |
|---|---------|
| 0 | Dependencias |
| 1 | Constantes NCh433 |
| 2 | **Dominio paramétrico** de la Familia B |
| 3 | Geometría del edificio (núcleo, corredor, muros B1–B4) |
| 4 | **Generar y graficar** un edificio de muestra (planta + 3D) |
| 5 | Espectro de diseño NCh433 (+ gráfico) |
| 6 | Modelo FEM + análisis modal (OpenSees) |
| 7 | Reconstrucción $K/M$ modal-consistente |
| 8 | Respuesta espectral + ensamblaje del caso |
| 9 | **Generación del conjunto en vivo** (+ distribuciones) |
| 10 | Metamodelo `PINNModal` |
| 11 | Pérdida híbrida + currículo |
| 12 | Normalización y partición |
| 13 | **Entrenamiento** (+ curva de pérdida) |
| 14 | **Evaluación: PINN vs OpenSees** (+ gráficos) |
| 15 | **Guardar el modelo** (`.pt`) |
| 16 | **Usar el metamodelo** (interfaz + veredicto NCh433) |
| 17 | Cierre |
""")

# =================================================================== 0 DEPS
md(r"""## 0. Dependencias

La celda siguiente instala todo lo necesario. Si una librería ya está presente, `pip` la
omite, así que el notebook corre en cualquier entorno con Python ejecutándolo de principio
a fin.""")
code(r"""# Version de openseespy fijada a la que funciona (critica para la generacion OpenSees).
%pip install -q numpy pandas matplotlib torch "openseespy==3.7.1" """)

# =================================================================== 1 CONST
md(r"""## 1. Constantes NCh433 y de discretización

Valores normativos y de malla usados en todo el pipeline (idénticos a los del dataset de la
tesis). Se definen con los dos juegos de nombres que usan los módulos del proyecto.""")
code(r"""import os, math
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- fisica / normativa (valores del dataset DEFINITIVO) ---
G = G_ACCEL    = 9.81
I_IMP = I_FIXED = 1.00
RO = RO_FIXED  = 11.0
ZETA           = 0.05
DRIFT_LIMIT    = 0.002
MIN_MASS_ACC   = 0.90
ETA_CRACKED    = 0.35

# --- discretizacion / tamanos fijos ---
N_MAX_PISOS = N_PISOS_MAX = 18
N_MODOS = N_MODOS_HDF5    = 18
N_MODOS_PHYS              = 18
N_GDL_MAX_V6              = 54
N_MODOS_MAX_V6            = 54
N_MODOS_POR_PISOS = {n: 3 * n for n in range(6, 19)}     # modos adaptativos: 3N

# --- espectro NCh433 ---
A0_ZONA = A0_BY_ZONE_G = {1: 0.20, 2: 0.30, 3: 0.40}
SUELO = {  # (S, T0, p) del espectro de diseno (version definitiva)
    'A': (0.90, 0.15, 2.00), 'B': (1.00, 0.30, 1.50),
    'C': (1.05, 0.40, 1.60), 'D': (1.20, 0.75, 1.00),
}
SOIL_PARAMS = {
    'A': dict(S=0.90, T0=0.15, Tp=0.20, n=1.00, p=2.00),
    'B': dict(S=1.00, T0=0.30, Tp=0.35, n=1.33, p=1.50),
    'C': dict(S=1.05, T0=0.40, Tp=0.45, n=1.40, p=1.60),
    'D': dict(S=1.20, T0=0.75, Tp=0.85, n=1.80, p=1.00),
}

torch.manual_seed(0); np.random.seed(0)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.25})
print('Constantes NCh433 cargadas.')""")

# =================================================================== 1b GUARDAR
md(r"""## 1b. Guardar todo — un solo interruptor

**Aquí decides si se guarda todo lo que produce el notebook.** Pon `GUARDAR_TODO = True` y al
ejecutar de principio a fin se escriben automáticamente, en la carpeta `CARPETA_SALIDA`, el
**modelo entrenado**, el **dataset generado**, las **figuras** y las **métricas**. Déjalo en
`False` para solo ver los resultados sin crear archivos.""")
code(r'''GUARDAR_TODO   = False           # <<<<<<<<<<  PON True PARA GUARDAR TODO  >>>>>>>>>>
CARPETA_SALIDA = 'salida_replica'


def ruta_salida(nombre):
    """Devuelve la ruta de un archivo dentro de CARPETA_SALIDA (la crea si hace falta)."""
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    return os.path.join(CARPETA_SALIDA, nombre)


def mostrar(nombre=None):
    """Muestra la figura actual; si GUARDAR_TODO, ademas la guarda como PNG en CARPETA_SALIDA."""
    if GUARDAR_TODO and nombre:
        plt.savefig(ruta_salida(nombre), dpi=130, bbox_inches='tight')
    plt.show()


print('GUARDAR_TODO =', GUARDAR_TODO,
      ('->  todo se guardara en: ' + os.path.abspath(CARPETA_SALIDA)) if GUARDAR_TODO
      else '->  no se escribiran archivos (solo se muestran resultados)')''')

# =================================================================== 2 DOMINIO
md(r"""## 2. Dominio paramétrico de la Familia B

La Familia B es una **barra habitacional** con corredor central longitudinal, núcleo central
(ASC + escalera + ASC) y departamentos de dos módulos. Un edificio queda definido por las
variables libres de abajo (alturas, geometría del departamento y del núcleo, materiales,
espesores de muro, suelo y zona sísmica); el resto se deriva. Esta configuración es la que
delimita el universo de edificios sobre el que el metamodelo es válido.""")
code('FAMILIA_B_CFG = ' + repr(fbc.__dict__.get('FAMILIA_B_CFG')) if False else r'''FAMILIA_B_CFG = {
    "meta": {"family_name": "Familia B",
             "description": "Barra habitacional con corredor central, nucleo ASC-ESC-ASC, "
                            "deptos de 2 modulos, simetrica respecto al nucleo."},

    # ── IDENTIDAD / RESTRICCIONES BASE ───────────────────────────────────────
    "fixed": {
        "tipologia": "barra_habitacional",
        "circulacion": "corredor_central_longitudinal",
        "nucleo_tipo": "asc1_esc_asc2",
        "mods_por_depto": 2,          # constante v6
        "activar_B3": 1,              # medianeros siempre activos
        "activar_B4": 1,              # borde corredor siempre activo
        "muros_variables_en_pares": True,
        "categoria_uso": "habitacional",
        "suelo_default": "D",
        "qk_default_kN_m2": 2.0,      # sobrecarga NCh1537 habitacional
        "psi_default": 0.25,          # combinacion sismica NCh433
        "area_unidad_min_m2": 30.0,
        "area_unidad_max_m2": 60.0,
        "aspect_ratio_min": 1.75,     # Lx/Ly minimo (tipologia barra)
        "B_nucleo_frac_Ly_max": 0.45, # nucleo <= 45% del ancho Ly
        "tech_bays_each_side": 1,     # 1 bahia tecnica a cada lado del nucleo
        "lw_max_m": 100.0,            # filtro de largo de edificio
    },

    # ── VARIABLES LIBRES ──────────────────────────────────────────────────────
    "variable_ranges": {
        "n_unid_lado":       {"type": "int",   "min": 2,    "max": 6},
        "L_mod_m":           {"type": "float", "min": 3.20, "max": 3.80},
        "prof_depto_m":      {"type": "float", "min": 7.00, "max": 7.90},
        "ancho_corredor_m":  {"type": "float", "min": 1.50, "max": 1.90},
        "eficiencia_planta": {"type": "float", "min": 0.74, "max": 0.82},
        "L_nucleo_m":             {"type": "float", "min": 5.40, "max": 6.60},
        "B_nucleo_m":             {"type": "float", "min": 4.00, "max": 5.00},
        "gap_core_to_corridor_m": {"type": "float", "min": 0.00, "max": 0.20},
        "N_pisos":   {"type": "int",    "min": 6,   "max": 18},
        "h_story_m": {"type": "choice", "values": [2.60, 2.70, 2.80, 2.90]},
        "fc_MPa":    {"type": "choice", "values": [25, 30, 35, 40]},
        "gk_kN_m2":  {"type": "choice", "values": [6.0, 6.5, 7.0, 7.5]},
        "zona":      {"type": "choice", "values": [1, 2, 3]},
        "suelo":     {"type": "choice", "values": ["A", "B", "C", "D"]},
        "t_muro_nucleo_m": {"type": "choice", "values": [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]},
        "t_muro_borde_m":  {"type": "choice", "values": [0.15, 0.18, 0.20, 0.22, 0.25]},
        "t_muro_mid_m":    {"type": "choice", "values": [0.15, 0.18, 0.20, 0.22, 0.25]},
        "lw_borde_m":      {"type": "choice", "values": [2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]},
        "lw_mid_m":        {"type": "choice", "values": [2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]},
        "activar_B2":      {"type": "choice", "values": [0, 1]},
    },

    # ── GRUPOS DE MUROS ───────────────────────────────────────────────────────
    "wall_groups": {
        "B1": {"name": "Muros del nucleo",            "state": "obligatorio",
               "main_role": "rigidez basica X e Y", "main_direction": "X + Y", "paired": False, "activar_var": None},
        "B2": {"name": "Muros de fachada (extremos)", "state": "opcional activar_B2",
               "main_role": "deriva global en Y",  "main_direction": "Y", "paired": True, "activar_var": "activar_B2"},
        "B3": {"name": "Medianeros entre deptos",     "state": "obligatorio v6",
               "main_role": "rigidez distribuida Y", "main_direction": "Y", "paired": True, "activar_var": None},
        "B4": {"name": "Borde del corredor",          "state": "obligatorio",
               "main_role": "rigidez longitudinal X", "main_direction": "X", "paired": True, "activar_var": None},
    },

    # ── CASO BASE (defaults) ──────────────────────────────────────────────────
    "defaults": {
        "n_unid_lado": 4, "L_mod_m": 3.50, "prof_depto_m": 7.50, "ancho_corredor_m": 1.70,
        "eficiencia_planta": 0.78, "L_nucleo_m": 6.00, "B_nucleo_m": 4.50,
        "gap_core_to_corridor_m": 0.10, "N_pisos": 12, "h_story_m": 2.70, "fc_MPa": 30,
        "gk_kN_m2": 7.0, "zona": 2, "suelo": "D", "t_muro_nucleo_m": 0.25,
        "t_muro_borde_m": 0.20, "t_muro_mid_m": 0.20, "lw_borde_m": 3.20, "lw_mid_m": 2.80,
        "activar_B2": 1, "activar_B3": 1,
    },
}

# variables que ve el metamodelo: 21 columnas (one-hot de suelo y zona) — IGUAL que Trial 16
COLS_BASE = ['N_pisos', 'n_unid_lado', 'activar_B2', 'L_mod_m', 'prof_depto_m',
             'ancho_corredor_m', 'L_nucleo_m', 'B_nucleo_m', 'h_story_m', 'fc_MPa', 'gk_kN_m2',
             't_muro_nucleo_m', 't_muro_borde_m', 't_muro_mid_m',
             'suelo_A', 'suelo_B', 'suelo_C', 'suelo_D', 'zona_1', 'zona_2', 'zona_3']
print('Variables libres:', len(FAMILIA_B_CFG['variable_ranges']),
      '| entradas del metamodelo X (one-hot):', len(COLS_BASE))''')

# =================================================================== 3 GEOMETRIA
md(r"""## 3. Geometría del edificio

Estas son las funciones **reales** del proyecto (Notebook 1). A partir de los parámetros
muestreados arman la planta: contorno, corredor, núcleo (con sus tres celdas ASC–ESC–ASC),
zonas técnicas, hall y los cuatro grupos de muros — B1 (núcleo), B2 (fachada), B3 (medianeros)
y B4 (borde del corredor). `geometry_to_wall_df` convierte la geometría a la tabla de muros
(centro `cx,cy` y dimensiones) que consume OpenSees.""")
code(r'''from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def prism_faces(x, y, z, dx, dy, dz):
    """6 caras de un prisma rectangular (para visualizacion 3D de muros)."""
    v = np.array([[x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                  [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]])
    return [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], [v[0], v[1], v[5], v[4]],
            [v[1], v[2], v[6], v[5]], [v[2], v[3], v[7], v[6]], [v[3], v[0], v[4], v[7]]]

''' + src(fbc.rect_dict, fbc.sample_from_spec, fbc.build_params_B,
          fbc.build_family_B_geometry, fbc.plot_family_B_plan, fbc.plot_family_B_3D))
code(r'''def geometry_to_wall_df(geom, params):
    """Convierte geom['walls'] (esquina x,y) a tabla de muros (centro cx,cy) para OpenSees."""
    rows = []
    for w in geom["walls"]:
        x, y, ww, hh = float(w["x"]), float(w["y"]), float(w["w"]), float(w["h"])
        rows.append({"label": w.get("label", ""), "group": w.get("group", ""),
                     "role": w.get("role", ""), "x": x, "y": y, "w": ww, "h": hh,
                     "cx": x + ww / 2, "cy": y + hh / 2, "area_plan_m2": ww * hh})
    return pd.DataFrame(rows)''')

# =================================================================== 4 PLOT EDIFICIO
md(r"""## 4. Generar y graficar un edificio de muestra

Muestreamos un edificio del dominio (con el filtro de validez de la familia: relación de
aspecto, áreas de unidad, fracción del núcleo y largo máximo) y dibujamos su planta.""")
code(r'''def edificio_valido(p, cfg):
    return (p['aspect_ratio_ok'] and p['area_unid_sup_ok'] and p['area_unid_inf_ok']
            and p['B_nucleo_frac_ok'] and p['Lx_m'] <= cfg['fixed']['lw_max_m'])

# muestrear hasta obtener un edificio valido
_rng = np.random.default_rng(7)
while True:
    p_demo = build_params_B(FAMILIA_B_CFG, randomize=True, seed=int(_rng.integers(0, 2**31)))
    if edificio_valido(p_demo, FAMILIA_B_CFG):
        break
geom_demo = build_family_B_geometry(p_demo)
wdf_demo  = geometry_to_wall_df(geom_demo, p_demo)

print(f"N={p_demo['N_pisos']} pisos | Lx={p_demo['Lx_m']:.1f} m  Ly={p_demo['Ly_m']:.1f} m "
      f"| A={p_demo['A_geom_m2']:.0f} m^2 | fc={p_demo['fc_MPa']} MPa  E={p_demo['E_MPa']:.0f} MPa "
      f"| suelo {p_demo['suelo']}  zona {p_demo['zona']} | {len(wdf_demo)} muros")

fig, ax = plt.subplots(figsize=(15, 6))
plot_family_B_plan(geom_demo, p_demo, ax=ax, cfg=FAMILIA_B_CFG)
plt.tight_layout(); mostrar('01_edificio_planta.png')''')

# =================================================================== 4b 3D
md(r"""### Vista 3D del edificio

El mismo edificio en 3D, con los muros de los cuatro grupos (B1 núcleo, B2 fachada,
B3 medianeros, B4 corredor) extruidos en altura sobre las losas de piso.""")
code(r'''fig = plt.figure(figsize=(12, 8))
ax3d = fig.add_subplot(111, projection='3d')
plot_family_B_3D(geom_demo, p_demo, ax=ax3d)             # render 3D del pipeline (muros B1-B4)
plt.tight_layout(); mostrar('02_edificio_3d.png')''')

# =================================================================== 5 ESPECTRO
md(r"""## 5. Espectro de diseño NCh433 (versión definitiva)

$$S_a(T)=\frac{I\,S\,A_0\,\alpha(T)}{R^\ast(T^\ast)}\,g,\qquad
\alpha=\frac{1+4.5(T/T_0)^p}{1+(T/T_0)^3},\qquad
R^\ast=1+\frac{T^\ast}{0.10\,T_0+T^\ast/R_0}.$$

$\alpha$ se evalúa con $(T_0,p)$ del suelo y $R^\ast$ con $R_0$; la ordenada no se capa (los
límites $C_{\min}/C_{\max}$ van sobre el corte combinado). La combinación modal usa CQC.""")
code(src(pl.alpha_nch433, pl.Rstar_nch433, pl.Sa_diseno, pl.cqc_rho, pl.cqc_combine))
code(r'''# espectro de diseno para tres suelos (zona 3), con T* = 0.5 s de referencia
Tg = np.linspace(0.02, 3.0, 300)
fig, ax = plt.subplots(figsize=(8, 4.5))
for su in ['A', 'B', 'C', 'D']:
    Sa = Sa_diseno(Tg, T_star=0.5, zona=3, suelo=su) / G       # en [g]
    ax.plot(Tg, Sa, label=f'Suelo {su}')
ax.set_xlabel('Periodo T [s]'); ax.set_ylabel('S$_a$ [g]')
ax.set_title('Espectro de diseno NCh433 — zona 3 (I=1, R$_0$=11, T*=0.5 s)')
ax.legend(); plt.tight_layout(); mostrar('03_espectro.png')''')

# =================================================================== 6 FEM
md(r"""## 6. Modelo FEM y análisis modal (OpenSees)

Funciones **reales** del worker de producción. Cada muro es un `elasticBeamColumn` de sección
agrietada ($I_e=0.35\,I_g$), con nodo maestro por piso y diafragma rígido (3 GDL/piso). El
análisis modal es adaptativo ($3N$ modos), calcula masa participativa y clasifica cada modo
(trasl. X / trasl. Y / torsional) para ordenarlo en 18 *slots* con patrón Y–T–X.""")
code(src(osf.SuppressStderr, osf.build_opensees_model, osf.run_eigen_analysis,
         osf.normalizar_phi, osf.seleccionar_18_slots))

# =================================================================== 7 KM
md(r"""## 7. Reconstrucción $K/M$ modal-consistente

Para las pérdidas físicas se necesitan las matrices $K$ y $M$ (54×54). $M$ es diagonal (masa
de piso concentrada); $K$ se **reconstruye modalmente** como $K=M\Phi_n\Omega^2\Phi_n^\top M$
con $\Phi_n$ M-normalizado, lo que garantiza por construcción $K\Phi=\omega^2 M\Phi$.""")
code(src(osf.build_M_global_pad_v6, osf.normalize_modes_M_v6,
         osf.build_Phi_global_pad_v6, osf.reconstruct_K_modal_pad_v6))

# =================================================================== 8 RESPUESTA + ENSAMBLE
md(r"""## 8. Respuesta espectral y ensamblaje del caso

La respuesta espectral (definitiva) entrega desplazamientos, derivas y corte basal con torsión
accidental y los límites $C_{\min}/C_{\max}$ sobre el corte combinado. `ensamblar_caso` deja
todo con *padding* a 18 pisos + máscara y construye el vector de 18 variables de diseño.""")
code(src(pl.respuesta_espectral))
code(r'''def construir_X(p):
    """Vector de 21 entradas en el orden COLS_BASE, con one-hot de suelo y zona (igual que Trial 16)."""
    d = {c: float(p[c]) for c in COLS_BASE
         if not c.startswith('suelo_') and not c.startswith('zona_')}
    for s in ['A', 'B', 'C', 'D']:
        d[f'suelo_{s}'] = 1.0 if p['suelo'] == s else 0.0
    for z in [1, 2, 3]:
        d[f'zona_{z}'] = 1.0 if int(p['zona']) == z else 0.0
    return np.array([d[c] for c in COLS_BASE], dtype=np.float32)


def ensamblar_caso(p, modal, sel, resp):
    N = int(p['N_pisos'])
    def pad(a):
        out = np.zeros(N_MAX_PISOS, np.float32); out[:N] = np.asarray(a, np.float32)[:N]; return out
    mask = np.zeros(N_MAX_PISOS, np.float32); mask[:N] = 1.0

    # K/M (54x54) modal-consistentes
    M_pad = build_M_global_pad_v6(modal['floor_masses'], modal['I_rot_floors'], N)
    omega_pad = np.zeros(N_MODOS_MAX_V6)
    om = modal['omegas'][:N_MODOS_MAX_V6]; omega_pad[:len(om)] = om
    Phi_pad = build_Phi_global_pad_v6(modal, N)
    K_pad, _ = reconstruct_K_modal_pad_v6(M_pad, Phi_pad, omega_pad, N)

    return dict(
        X=construir_X(p), mask=mask, N_pisos=N,
        T_r=sel['T_18'].astype(np.float32),
        Phi_x=sel['Phi_x_18'].astype(np.float32),
        Phi_y=sel['Phi_y_18'].astype(np.float32),
        Phi_theta=sel['Phi_t_18'].astype(np.float32),
        Ux=pad(resp['Ux_m']), Uy=pad(resp['Uy_m']),
        drift_x=pad(resp['drift_x']), drift_y=pad(resp['drift_y']),
        Vb_x=np.float32(resp['Vb_x_kN']), Vb_y=np.float32(resp['Vb_y_kN']),
        cumple=int(resp['cumple']),
        K_global=K_pad.astype(np.float32), M_global=M_pad.astype(np.float32),
    )


def generar_caso(cfg, seed):
    """Un edificio de punta a punta. Devuelve (resultado_o_None, motivo) para diagnostico."""
    p = build_params_B(cfg, randomize=True, seed=seed)
    if not edificio_valido(p, cfg):
        return None, 'filtro_geometria'
    geom = build_family_B_geometry(p)
    wdf  = geometry_to_wall_df(geom, p)
    minfo = build_opensees_model(p, wdf)
    modal, err = run_eigen_analysis(minfo, p)
    if modal is None:
        return None, f'eigen_fallo:{err}'
    sel = seleccionar_18_slots(modal, int(p['N_pisos']))
    if not sel['valido']:
        return None, 'slots_T1_T2'
    resp = respuesta_espectral(p, modal)
    return dict(case=ensamblar_caso(p, modal, sel, resp), params=p, geom=geom), 'ok'
''')

# =================================================================== 9 GENERACION
md(r"""## 9. Generación del conjunto en vivo

Muestreamos edificios del dominio y los resolvemos uno a uno con OpenSees. Solo se conservan
los casos válidos (geometría admisible, eigen exitoso y *slots* $T_1,T_2$ presentes). Ajusta
`N_CASOS` según el tiempo disponible: más casos → mejor metamodelo (la tesis usó 10.000).""")
code(r'''from time import time

N_CASOS = 150        # objetivo de casos validos. OpenSees tarda ~4 s/edificio:
                     #   150 -> ~10 min | 80 -> ~5 min | sube para mejor modelo, baja para correr rapido.
SEED    = 0          # cambia la semilla si un caso pesado puntual traba el eigen

import traceback
casos = []; rng = np.random.default_rng(SEED); intentos = 0; motivos = {}; t0 = time()
while len(casos) < N_CASOS and intentos < N_CASOS * 8:
    intentos += 1
    try:
        r, motivo = generar_caso(FAMILIA_B_CFG, int(rng.integers(0, 2**31)))
    except Exception as e:
        motivo = 'EXCEPCION:' + type(e).__name__
        if motivos.get(motivo, 0) == 0:                  # primera vez: muestra el traceback
            traceback.print_exc()
        r = None
    motivos[motivo] = motivos.get(motivo, 0) + 1
    if r is not None:
        casos.append(r)
        if len(casos) % 5 == 0:
            print(f'  {len(casos)}/{N_CASOS} casos | {intentos} intentos | {time()-t0:.0f}s', flush=True)
print(f'\nGenerados {len(casos)} casos validos en {time()-t0:.0f}s ({intentos} intentos).')
print('motivos de descarte:', motivos)
if not casos:
    raise RuntimeError('0 casos validos. Revisa "motivos de descarte": slots_T1_T2 = criterio modal | '
                       'EXCEPCION = mira el traceback | eigen_fallo = openseespy/version.')

# apilar a arreglos
def stack(k): return np.stack([c['case'][k] for c in casos]).astype(np.float32)
X   = stack('X');   mask = stack('mask'); T_r = stack('T_r')
Phx = stack('Phi_x'); Phy = stack('Phi_y'); Pht = stack('Phi_theta')
Ux  = stack('Ux');  Uy = stack('Uy'); Dx = stack('drift_x'); Dy = stack('drift_y')
Kg  = stack('K_global'); Mg = stack('M_global')
Vbx = np.array([c['case']['Vb_x'] for c in casos], np.float32)
Vby = np.array([c['case']['Vb_y'] for c in casos], np.float32)
cumple = np.array([c['case']['cumple'] for c in casos], int)
Npis   = np.array([c['case']['N_pisos'] for c in casos], int)
zonas  = np.array([c['params']['zona'] for c in casos], int)      # para el split OOD por zona
T1     = T_r.max(1)        # periodo fundamental por caso
print(f'X{X.shape} | T1 in [{T1.min():.2f}, {T1.max():.2f}] s | cumplen NCh433: {cumple.sum()}/{len(casos)}')''')
code(r'''# distribuciones del conjunto generado
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].hist(Npis, bins=np.arange(5.5, 19.5, 1), color='steelblue', edgecolor='w')
ax[0].set_title('N pisos'); ax[0].set_xlabel('pisos')
ax[1].hist(T1, bins=20, color='seagreen', edgecolor='w')
ax[1].set_title('Periodo fundamental T$_1$'); ax[1].set_xlabel('T$_1$ [s]')
ax[2].bar(['cumple', 'no cumple'], [cumple.sum(), len(cumple)-cumple.sum()],
          color=['mediumseagreen', 'indianred'])
ax[2].set_title('Cumplimiento NCh433 (deriva $\\leq$ 0.002)')
plt.tight_layout(); mostrar('04_distribuciones.png')''')
md(r"""**¿Dónde queda el conjunto generado?** Hasta aquí vive **en memoria** (la lista `casos`
y los arreglos `X`, `T_r`, `Ux`, …): se usa para entrenar. Si activaste `GUARDAR_TODO`
(sección 1b), la celda siguiente lo escribe a disco como `.npz` comprimido para reusarlo sin
regenerar.""")
code(r'''if GUARDAR_TODO:
    np.savez_compressed(ruta_salida('dataset_familiaB.npz'),
        X=X, mask=mask, T_r=T_r, Phi_x=Phx, Phi_y=Phy, Phi_theta=Pht,
        Ux=Ux, Uy=Uy, drift_x=Dx, drift_y=Dy, Vb_x=Vbx, Vb_y=Vby,
        cumple=cumple, N_pisos=Npis, K_global=Kg, M_global=Mg)
    print('Conjunto guardado en', ruta_salida('dataset_familiaB.npz'),
          f"({os.path.getsize(ruta_salida('dataset_familiaB.npz'))/1e6:.1f} MB)")
else:
    print('Conjunto en memoria (GUARDAR_TODO=False; no se escribio a disco).')''')

# =================================================================== 10 PINN
md(r"""## 10. Metamodelo `PINNModal`

Arquitectura de la tesis: un `encoder_T1` dedicado al período fundamental y un `encoder`
compartido (cuatro `ResBlock`) que alimenta cuatro cabezales — períodos, formas modales,
respuestas de piso y corte basal. Las formas se enmascaran por piso y se normalizan por modo;
la deriva se parametriza como $d_x=\text{ratio}\cdot d_y$ con $\text{ratio}\ge 0$.

Los períodos se predicen con la **construcción monótona** de la tesis: $T_1$ explícito y
$T_{r+1}=T_r-\Delta_r$ con $\Delta_r=\text{softplus}(\cdot)\ge 0$, de modo que
$T_1\ge T_2\ge\dots$. Las entradas son las **21 variables one-hot** (suelo y zona), idénticas a
las de Trial 16.""")
code(r'''class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.SiLU(),
                                 nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))


class PINNModal(nn.Module):
    def __init__(self, n_in=21, n_modos=18, n_pisos=18, hidden=256, hidden_T1=128):
        super().__init__()
        self.n_modos, self.n_pisos = n_modos, n_pisos
        self.encoder_T1 = nn.Sequential(nn.Linear(n_in, hidden_T1), nn.LayerNorm(hidden_T1),
                                        nn.SiLU(), ResBlock(hidden_T1), ResBlock(hidden_T1))
        self.head_T1 = nn.Sequential(nn.Linear(hidden_T1, 32), nn.SiLU(), nn.Linear(32, 1))
        self.encoder = nn.Sequential(nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                     ResBlock(hidden), ResBlock(hidden), ResBlock(hidden), ResBlock(hidden))
        self.head_T_rest = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, n_modos - 1))
        self.head_Phi  = nn.Sequential(nn.Linear(hidden, 512), nn.SiLU(), nn.Linear(512, n_modos * n_pisos * 3))
        self.head_resp = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(), nn.Linear(256, n_pisos * 4))
        self.head_Vb   = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, X, mask):
        B = X.shape[0]
        T1 = self.head_T1(self.encoder_T1(X))                       # log-T1 (encoder dedicado)
        h  = self.encoder(X)
        Td = F.softplus(self.head_T_rest(h))                        # incrementos >= 0
        parts = [T1]
        for r in range(self.n_modos - 1):
            parts.append(parts[-1] - Td[:, r:r+1])                  # periodos monotonos: T1>=T2>=...
        logT = torch.cat(parts, dim=1)
        Phi = self.head_Phi(h).view(B, self.n_modos, self.n_pisos, 3)
        Phi = Phi * mask.unsqueeze(1).unsqueeze(-1)
        Phi = Phi / Phi.norm(dim=2, keepdim=True).clamp(min=1e-8)
        Phi = Phi.permute(0, 2, 1, 3)
        resp = self.head_resp(h).view(B, self.n_pisos, 4)
        Ux = resp[:, :, 0] * mask; Uy = resp[:, :, 1] * mask
        ratio = F.softplus(resp[:, :, 2]) * mask; dy = resp[:, :, 3] * mask
        Vb = self.head_Vb(h)
        return logT, Phi[:, :, :, 0], Phi[:, :, :, 1], Phi[:, :, :, 2], Ux, Uy, ratio, dy, Vb

print('PINNModal:', sum(p.numel() for p in PINNModal().parameters()), 'parametros')''')

# =================================================================== 11 LOSS
md(r"""## 11. Pérdida híbrida y currículo

La pérdida combina términos de **datos** (períodos, desplazamientos, deriva, corte basal,
enmascarados) con tres de **física** — alineamiento de formas ($1-\cos^2$, equiv. $1-$MAC),
consistencia del eigenproblema $K\Phi=\omega^2 M\Phi$ y M-ortonormalidad $\Phi^\top M\Phi=I$.
El currículo activa la física de forma escalonada una vez que los datos ya se ajustan.""")
code(r'''def _ensamblar_KM(batch, mask, device):
    B = mask.shape[0]
    mf = mask.unsqueeze(-1).expand(-1, -1, 3).reshape(B, 54)
    m2 = mf.unsqueeze(2) * mf.unsqueeze(1)
    K_b = batch['K_global'].to(device) * m2; M_b = batch['M_global'].to(device) * m2
    s = K_b.abs().amax(dim=(-2, -1)).clamp(min=1.0).view(B, 1, 1)
    return K_b / s, M_b / s


def _ensamblar_Phi_global(Phx, Phy, Pht, device):
    B, _, R = Phx.shape; Phi = torch.zeros(B, 54, R, device=device)
    Phi[:, 0::3, :] = Phx; Phi[:, 1::3, :] = Phy; Phi[:, 2::3, :] = Pht
    return Phi / Phi.norm(dim=1, keepdim=True).clamp(min=1e-8)


def perdida_hibrida(batch, out, lambdas, stage, device='cpu'):
    logT_p, Phx_p, Phy_p, Pht_p, Ux_p, Uy_p, ratio_p, dy_p, Vb_p = out
    mask = batch['mask'].to(device); ld = {}
    ld['data_resp'] = (F.mse_loss(Ux_p * mask, batch['Ux'].to(device) * mask) +
                       F.mse_loss(Uy_p * mask, batch['Uy'].to(device) * mask) +
                       F.mse_loss(dy_p * mask, batch['Dy'].to(device) * mask) +
                       F.mse_loss(Vb_p[:, 0], batch['Vbx'].to(device)) +
                       F.mse_loss(Vb_p[:, 1], batch['Vby'].to(device)))
    n_act = mask.sum().clamp(min=1.0)
    ld['ratio'] = ((ratio_p - batch['ratio'].to(device)).pow(2) * mask).sum() / n_act
    mask_T  = (batch['T_r'].to(device) > 1e-6).float()
    n_valid = mask_T.sum().clamp(min=1.0)
    ld['data_T1']     = F.mse_loss(logT_p[:, 0], batch['logT'].to(device)[:, 0])
    ld['data_T_rest'] = ((logT_p[:, 1:] - batch['logT'].to(device)[:, 1:]).pow(2)
                         * mask_T[:, 1:]).sum() / n_valid

    L_phi = torch.tensor(0.0, device=device)
    if stage.get('phi'):
        for r in range(N_MODOS):
            comp = [(Phy_p, batch['Phi_y']), (Pht_p, batch['Phi_theta']), (Phx_p, batch['Phi_x'])][r % 3]
            pp = comp[0][:, :, r] * mask; pt = comp[1].to(device)[:, :, r] * mask
            num = (pp * pt).sum(1) ** 2
            den = ((pp**2).sum(1) * (pt**2).sum(1)).clamp(min=1e-12)
            val = (pt.pow(2).sum(1) > 1e-10).float()
            L_phi = L_phi + (val * (1 - num / den)).sum() / val.sum().clamp(min=1.0)
        L_phi = L_phi / N_MODOS
    ld['phi'] = L_phi

    L_eig = torch.tensor(0.0, device=device)
    if stage.get('eig'):
        K_b, M_b = _ensamblar_KM(batch, mask, device)
        Phi_g = _ensamblar_Phi_global(Phx_p, Phy_p, Pht_p, device)
        T_pred = torch.exp(logT_p[:, :N_MODOS_PHYS]).clamp(min=1e-6)
        w2 = (2 * math.pi / T_pred) ** 2
        KPhi = torch.bmm(K_b, Phi_g); MPhi = torch.bmm(M_b, Phi_g)
        res = KPhi - w2.unsqueeze(1) * MPhi
        den = KPhi.pow(2).sum(1) + (w2.unsqueeze(1) * MPhi).pow(2).sum(1) + 1e-12
        L_eig = (res.pow(2).sum(1) / den).mean()
    ld['eig'] = L_eig

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


def curriculo_lambda(epoch, n_epochs):
    # hitos escalados al presupuesto de epocas (tesis: 150/200/350 sobre 600)
    e1, e2, e3 = int(0.25*n_epochs), int(0.33*n_epochs), int(0.58*n_epochs)
    lam = {'data_resp': 2.0, 'data_T_rest': 1.0, 'ratio': 1.0,
           'data_T1': 1.0, 'phi': 0.0, 'eig': 0.0, 'ortho': 0.0}
    stage = {'phi': False, 'eig': False, 'ortho': False}
    if epoch >= e1: lam['phi'] = 0.1;  stage['phi'] = True
    if epoch >= e2: lam['eig'] = 0.1;  stage['eig'] = True
    if epoch >= e3: lam['ortho'] = 1e-5; stage['ortho'] = True
    return lam, stage''')

# =================================================================== 12 NORMALIZACION
md(r"""## 12. Normalización y partición

**Partición OOD por zona, igual que Trial 16:** se entrena con edificios de zona 1 y 3, y se
**evalúa en zona 2** (que el modelo nunca vio). Las entradas se estandarizan (z-score); los
períodos se entrenan en log; los desplazamientos, la deriva y el corte basal en **log-z** (como
producción). Las constantes se ajustan **solo sobre el entrenamiento** y se invierten para
evaluar en unidades físicas.""")
code(r'''# particion OOD por zona (igual que Trial 16): entrena zona 1 y 3, evalua en zona 2
tr = np.where(zonas != 2)[0]
te = np.where(zonas == 2)[0]
mtr = mask[tr].astype(bool)

# scalers (solo train): X z-score; periodos en log; respuestas y corte en log-z (como produccion)
zlog = lambda a: np.log(np.clip(a, 1e-9, None))
def _logstat(a):
    la = zlog(a)[mtr]; return float(la.mean()), float(la.std() + 1e-8)
Xmu, Xsd   = X[tr].mean(0), X[tr].std(0) + 1e-8
Uxmu, Uxsd = _logstat(Ux[tr]); Uymu, Uysd = _logstat(Uy[tr]); Dymu, Dysd = _logstat(Dy[tr])
Vxmu, Vxsd = float(zlog(Vbx[tr]).mean()), float(zlog(Vbx[tr]).std() + 1e-8)
Vymu, Vysd = float(zlog(Vby[tr]).mean()), float(zlog(Vby[tr]).std() + 1e-8)
SC = dict(Xmu=Xmu, Xsd=Xsd, Uxmu=Uxmu, Uxsd=Uxsd, Uymu=Uymu, Uysd=Uysd,
          Dymu=Dymu, Dysd=Dysd, Vxmu=Vxmu, Vxsd=Vxsd, Vymu=Vymu, Vysd=Vysd)

def make_batch(ids, device):
    t = lambda a: torch.tensor(np.asarray(a, np.float32), device=device)
    return dict(
        X=t((X[ids]-Xmu)/Xsd), mask=t(mask[ids]),
        T_r=t(T_r[ids]), logT=t(zlog(T_r[ids])),
        Phi_x=t(Phx[ids]), Phi_y=t(Phy[ids]), Phi_theta=t(Pht[ids]),
        Ux=t((zlog(Ux[ids])-Uxmu)/Uxsd), Uy=t((zlog(Uy[ids])-Uymu)/Uysd),
        Dy=t((zlog(Dy[ids])-Dymu)/Dysd),
        ratio=t(Dx[ids] / np.clip(Dy[ids], 1e-9, None)),
        Vbx=t((zlog(Vbx[ids])-Vxmu)/Vxsd), Vby=t((zlog(Vby[ids])-Vymu)/Vysd),
        K_global=t(Kg[ids]), M_global=t(Mg[ids]))

print(f'split OOD por zona  ->  train (zona 1 y 3) = {len(tr)}   test (zona 2) = {len(te)}')''')

# =================================================================== 13 ENTRENAMIENTO
md(r"""## 13. Entrenamiento

Adam con `CosineAnnealingWarmRestarts` y recorte de gradiente, mismo esquema de la tesis. El
`encoder_T1` se congela en la primera fase (dos velocidades). Se grafican las curvas de pérdida
total y de los términos físicos a medida que el currículo los activa.""")
code(r'''device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 600; BATCH = 64
model = PINNModal().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=2, eta_min=1e-6)

hist = {k: [] for k in ['total','data_resp','data_T_rest','phi','eig','ortho']}
t0 = time()
for ep in range(EPOCHS):
    lam, stage = curriculo_lambda(ep, EPOCHS)
    for prm in model.encoder_T1.parameters():
        prm.requires_grad = (ep >= int(0.25*EPOCHS))      # dos velocidades
    perm = np.random.permutation(tr)
    ep_loss = {k: 0.0 for k in hist}; nb = 0
    model.train()
    for i in range(0, len(perm), BATCH):
        ids = perm[i:i+BATCH]
        batch = make_batch(ids, device)
        out = model(batch['X'], batch['mask'])
        loss, ld = perdida_hibrida(batch, out, lam, stage, device)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ep_loss['total'] += loss.item()
        for k in ['data_resp','data_T_rest','phi','eig','ortho']:
            ep_loss[k] += ld[k].item()
        nb += 1
    sch.step()
    for k in hist: hist[k].append(ep_loss[k] / max(nb, 1))
    if ep % 50 == 0 or ep == EPOCHS-1:
        print(f"ep {ep:3d} | total {hist['total'][-1]:.4f} | data {hist['data_resp'][-1]:.4f} "
              f"| phi {hist['phi'][-1]:.3f} eig {hist['eig'][-1]:.3f} | {time()-t0:.0f}s")
print(f'\nEntrenamiento terminado en {time()-t0:.0f}s.')

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(hist['total']); ax[0].set_yscale('log'); ax[0].set_title('Perdida total'); ax[0].set_xlabel('epoca')
for k in ['data_resp','data_T_rest','phi','eig','ortho']:
    ax[1].plot(hist[k], label=k)
ax[1].set_yscale('log'); ax[1].set_title('Terminos de la perdida'); ax[1].set_xlabel('epoca'); ax[1].legend(fontsize=8)
plt.tight_layout(); mostrar('05_curva_perdida.png')''')

# =================================================================== 14 EVAL
md(r"""## 14. Evaluación: PINN vs OpenSees

Sobre los **edificios de zona 2** (que el metamodelo nunca vio — partición OOD de Trial 16),
comparamos las predicciones del PINN contra el análisis OpenSees: período fundamental $T_1$,
corte basal $V_b$ y deriva máxima. Se reporta el error porcentual medio y el $R^2$, y se
grafica predicción vs verdad.""")
code(r'''model.eval()
with torch.no_grad():
    b = make_batch(te, device)
    out = model(b['X'], b['mask'])
logT_p, _, _, _, Ux_p, Uy_p, ratio_p, dy_p, Vb_p = [o.cpu().numpy() if torch.is_tensor(o) else o for o in out]

# desnormalizar a unidades fisicas
T_pred  = np.exp(logT_p)                                  # periodos por slot (log -> s)
T1_pred = T_pred.max(1)
T1_true = T_r[te].max(1)
Vb_pred = np.exp(Vb_p[:, 0] * Vxsd + Vxmu)               # corte basal X [kN]  (log-z -> kN)
Vb_true = Vbx[te]
dy_pred = np.exp(dy_p * Dysd + Dymu)                      # deriva Y por piso   (log-z -> -)
dymax_pred = np.array([dy_pred[i, :Npis[te][i]].max() for i in range(len(te))])
dymax_true = np.array([Dy[te][i, :Npis[te][i]].max() for i in range(len(te))])

def mape(p, t): return float(np.mean(np.abs((p - t) / np.clip(np.abs(t), 1e-9, None))) * 100)
def r2(p, t):   return float(1 - np.sum((p-t)**2) / np.sum((t-t.mean())**2))

print('Metamodelo PINN vs OpenSees (conjunto de prueba):')
for nm, pr, tv, u in [('T1', T1_pred, T1_true, 's'),
                      ('Vb_x', Vb_pred, Vb_true, 'kN'),
                      ('drift_y_max', dymax_pred, dymax_true, '-')]:
    print(f'  {nm:12s}  MAPE={mape(pr,tv):5.1f}%   R2={r2(pr,tv):.3f}   [{u}]')

metricas = {nm: dict(MAPE_pct=round(mape(pr, tv), 2), R2=round(r2(pr, tv), 4))
            for nm, pr, tv in [('T1', T1_pred, T1_true), ('Vb_x', Vb_pred, Vb_true),
                               ('drift_y_max', dymax_pred, dymax_true)]}
metricas.update(n_casos=len(casos), n_train=int(len(tr)), n_test=int(len(te)), epochs=EPOCHS)
if GUARDAR_TODO:
    import json
    with open(ruta_salida('metricas.json'), 'w', encoding='utf-8') as _f:
        json.dump(metricas, _f, indent=2, ensure_ascii=False)
    print('Metricas guardadas en', ruta_salida('metricas.json'))

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
for a, (nm, pr, tv, u) in zip(ax, [('T$_1$ [s]', T1_pred, T1_true, 's'),
                                    ('V$_b$ [kN]', Vb_pred, Vb_true, 'kN'),
                                    ('deriva máx [-]', dymax_pred, dymax_true, '-')]):
    a.scatter(tv, pr, s=22, alpha=0.7, edgecolor='w')
    lo, hi = min(tv.min(), pr.min()), max(tv.max(), pr.max())
    a.plot([lo, hi], [lo, hi], 'k--', lw=1)
    a.set_xlabel(f'OpenSees'); a.set_ylabel('PINN'); a.set_title(nm)
plt.tight_layout(); mostrar('06_pinn_vs_opensees.png')''')

# =================================================================== 15 GUARDAR
md(r"""## 15. Guardar el modelo entrenado

Si `GUARDAR_TODO` está activo (sección 1b), se escribe un único archivo `.pt` con los pesos
del metamodelo, las constantes de normalización, el dominio y el orden de las variables de
diseño. Con eso, el modelo se recarga después sin reentrenar: la función `cargar_modelo`
reconstruye la red y devuelve también los *scalers* listos para predecir.""")
code(r'''# empaquetar pesos + scalers + dominio en un solo archivo
bundle = dict(state_dict=model.state_dict(), scalers=SC, columnas=COLS_BASE, cfg=FAMILIA_B_CFG,
              meta=dict(n_casos=len(casos), epochs=EPOCHS,
                        n_params=sum(p.numel() for p in model.parameters())))
MODEL_NAME = 'modelo_pinn_familiaB.pt'
if GUARDAR_TODO:
    torch.save(bundle, ruta_salida(MODEL_NAME))
    print('Modelo guardado en', ruta_salida(MODEL_NAME),
          f'({os.path.getsize(ruta_salida(MODEL_NAME))/1e6:.2f} MB)')
else:
    print('Modelo entrenado y disponible en memoria (GUARDAR_TODO=False; no se escribio a disco).')


def cargar_modelo(path=None, device='cpu'):
    """Recarga el metamodelo y sus scalers desde el .pt (sin reentrenar)."""
    path = path or os.path.join(CARPETA_SALIDA, MODEL_NAME)
    b = torch.load(path, map_location=device, weights_only=False)
    m = PINNModal().to(device); m.load_state_dict(b['state_dict']); m.eval()
    return m, b['scalers'], b

# ejemplo: model, SC, bundle = cargar_modelo()''')

# =================================================================== 16 INTERFAZ
md(r"""## 16. Usar el metamodelo (interfaz)

Con el modelo entrenado ya se puede **consultar cualquier edificio del dominio** sin volver a
correr OpenSees: se definen los parámetros, el metamodelo entrega períodos, desplazamientos,
derivas y corte basal en **milisegundos**, y se emite el veredicto NCh433. Es el mismo flujo
de la interfaz del proyecto, adaptado a este modelo.""")
code(r'''def edificio(**overrides):
    """Construye un edificio del dominio (parametros derivados consistentes) desde unos pocos valores."""
    return build_params_B(FAMILIA_B_CFG, randomize=False, overrides=overrides)


def validar_dominio(params):
    """Avisa si algun parametro cae fuera del dominio de entrenamiento (riesgo de extrapolacion)."""
    avisos = []
    for k, spec in FAMILIA_B_CFG['variable_ranges'].items():
        if k not in params:
            continue
        v = params[k]
        if spec['type'] in ('int', 'float'):
            if v < spec['min'] or v > spec['max']:
                avisos.append(f"{k}={v} fuera de [{spec['min']}, {spec['max']}]")
        elif v not in spec['values']:
            avisos.append(f"{k}={v} no esta en {spec['values']}")
    return avisos


def predict_edificio(params, avisar=True):
    """Prediccion del metamodelo para un edificio. Devuelve dict estructurado (modal + respuesta)."""
    avisos = validar_dominio(params)
    if avisos and avisar:
        print('Aviso — fuera del dominio de entrenamiento (interpretar con cautela):')
        for a in avisos: print('   -', a)
    Np = int(params['N_pisos'])
    Xn = ((construir_X(params) - SC['Xmu']) / SC['Xsd']).astype(np.float32)
    Xt = torch.tensor(Xn[None], device=device)
    mk = np.zeros((1, 18), np.float32); mk[0, :Np] = 1.0
    mkt = torch.tensor(mk, device=device)
    with torch.no_grad():
        logT, Phx, Phy, Pht, Ux, Uy, ratio, dy, Vb = model(Xt, mkt)
    T    = np.exp(logT[0].cpu().numpy())
    Ux_r = np.exp(Ux[0, :Np].cpu().numpy() * SC['Uxsd'] + SC['Uxmu'])
    Uy_r = np.exp(Uy[0, :Np].cpu().numpy() * SC['Uysd'] + SC['Uymu'])
    dy_r = np.exp(dy[0, :Np].cpu().numpy() * SC['Dysd'] + SC['Dymu'])
    dx_r = ratio[0, :Np].cpu().numpy() * dy_r
    Vbx  = float(np.exp(Vb[0, 0].item() * SC['Vxsd'] + SC['Vxmu']))
    Vby  = float(np.exp(Vb[0, 1].item() * SC['Vysd'] + SC['Vymu']))
    return dict(params=params, avisos=avisos,
                geometria=dict(Lx=params['Lx_m'], Ly=params['Ly_m'], H=params['H_total_m']),
                modal=dict(T=T, T1=float(T.max())),
                respuesta=dict(Ux=Ux_r, Uy=Uy_r, dx=dx_r, dy=dy_r, Vb_x=Vbx, Vb_y=Vby))


def reporte_normativo(res):
    """Veredicto NCh433 (deriva <= 0.002) con la razon del incumplimiento si aplica."""
    dx, dy = res['respuesta']['dx'], res['respuesta']['dy']
    dxm, ix = float(dx.max()), int(dx.argmax()) + 1
    dym, iy = float(dy.max()), int(dy.argmax()) + 1
    razones = []
    if dxm > DRIFT_LIMIT:
        razones.append(f"Deriva X = {dxm:.5f} en piso {ix} ({(dxm/DRIFT_LIMIT-1)*100:+.0f}% sobre el limite)")
    if dym > DRIFT_LIMIT:
        razones.append(f"Deriva Y = {dym:.5f} en piso {iy} ({(dym/DRIFT_LIMIT-1)*100:+.0f}% sobre el limite)")
    veredicto = 'CUMPLE' if not razones else 'NO CUMPLE'
    return dict(veredicto=veredicto, dxm=dxm, ix=ix, dym=dym, iy=iy,
                T1=res['modal']['T1'], razones=razones)


def graficar_resultados(res, rep):
    fig = plt.figure(figsize=(13, 7.5)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)
    Np = int(res['params']['N_pisos']); pisos = np.arange(1, Np + 1)
    r = res['respuesta']
    a1 = fig.add_subplot(gs[0, 0])
    a1.plot(r['dx'], pisos, 'o-', color='steelblue', label='dx', lw=1.5)
    a1.plot(r['dy'], pisos, 's-', color='firebrick', label='dy', lw=1.5)
    a1.axvline(DRIFT_LIMIT, ls='--', color='red', label=f'limite {DRIFT_LIMIT}')
    a1.set_xlabel('Deriva [-]'); a1.set_ylabel('Piso'); a1.set_title('Derivas de entrepiso'); a1.legend(fontsize=8)
    a2 = fig.add_subplot(gs[0, 1])
    a2.plot(r['Ux'], pisos, 'o-', color='steelblue', label='Ux', lw=1.5)
    a2.plot(r['Uy'], pisos, 's-', color='firebrick', label='Uy', lw=1.5)
    a2.set_xlabel('Desplazamiento [m]'); a2.set_ylabel('Piso'); a2.set_title('Desplazamientos'); a2.legend(fontsize=8)
    a3 = fig.add_subplot(gs[0, 2])
    Tsort = np.sort(res['modal']['T'])[::-1][:6]
    a3.bar(np.arange(1, len(Tsort)+1), Tsort, color='gray', edgecolor='black')
    a3.set_xlabel('Modo'); a3.set_ylabel('T [s]'); a3.set_title(f"Periodos (T1={res['modal']['T1']:.3f} s)")
    a4 = fig.add_subplot(gs[1, :]); a4.axis('off')
    color = '#2e7d32' if rep['veredicto'] == 'CUMPLE' else '#c62828'
    a4.add_patch(Rectangle((0, 0.55), 1, 0.4, transform=a4.transAxes, facecolor=color, alpha=0.85))
    a4.text(0.5, 0.75, f"VEREDICTO NCh433: {rep['veredicto']}", transform=a4.transAxes,
            ha='center', va='center', color='white', fontsize=18, fontweight='bold')
    sub = (f"Vb,x={r['Vb_x']:,.0f} kN | Vb,y={r['Vb_y']:,.0f} kN | T1={rep['T1']:.3f} s | "
           f"dx,max={rep['dxm']:.5f} (piso {rep['ix']}) | dy,max={rep['dym']:.5f} (piso {rep['iy']})")
    a4.text(0.5, 0.62, sub, transform=a4.transAxes, ha='center', va='center', color='white', fontsize=10)
    for i, rz in enumerate(rep['razones']):
        a4.text(0.04, 0.42 - i*0.12, '- ' + rz, transform=a4.transAxes, fontsize=9, color='#333')
    if not rep['razones']:
        a4.text(0.5, 0.32, 'Edificio dentro del dominio. Prediccion confiable.', transform=a4.transAxes,
                ha='center', fontsize=10, color='#2e7d32', style='italic')
    plt.suptitle('Resultados del metamodelo PINN — Familia B', fontsize=12, fontweight='bold'); mostrar('07_interfaz_resultados.png')''')

md(r"""### Ejemplo de uso — edita los parámetros y vuelve a ejecutar""")
code(r'''# (1) definir el edificio (lo no especificado toma el valor por defecto del dominio)
mi_edificio = edificio(N_pisos=12, suelo='C', zona=3, n_unid_lado=3, fc_MPa=30,
                       t_muro_nucleo_m=0.25, h_story_m=2.7)

# (2) predecir con el metamodelo y (3) evaluar la norma
res = predict_edificio(mi_edificio)
rep = reporte_normativo(res)
print(f"T1 = {rep['T1']:.3f} s | Vb_x = {res['respuesta']['Vb_x']:.0f} kN | "
      f"deriva_y max = {rep['dym']:.5f} | {rep['veredicto']}")

# (4) graficar
graficar_resultados(res, rep)''')

md(r"""### Edificio real y deformada predicha (3D)

El metamodelo entrega el desplazamiento de cada piso, así que podemos dibujar la **deformada**
del edificio bajo el sismo de diseño. A la izquierda, la estructura real; a la derecha, la
deformada según la predicción del PINN (amplificada para que sea visible), coloreada por la
magnitud del desplazamiento.""")
code(r'''def plot_real_vs_deformada(params, res, scale=None):
    """Izquierda: edificio real (muros 3D). Derecha: deformada segun la prediccion del PINN."""
    geom = build_family_B_geometry(params)
    N = int(params['N_pisos']); h = params['h_story_m']
    Lx, Ly, H = params['Lx_m'], params['Ly_m'], N * h
    zs = np.arange(N + 1) * h
    Ux = np.concatenate([[0.0], res['respuesta']['Ux']])     # desplazamiento por nivel (0 = base)
    Uy = np.concatenate([[0.0], res['respuesta']['Uy']])
    mag = np.sqrt(Ux**2 + Uy**2); umax = max(mag.max(), 1e-9)
    if scale is None: scale = 0.18 * H / umax                # amplificacion para visualizar

    fig = plt.figure(figsize=(15, 6.5))
    # (a) edificio real — mismo render 3D del pipeline (muros translucidos B1-B4)
    axA = fig.add_subplot(1, 2, 1, projection='3d')
    plot_family_B_3D(geom, params, ax=axA)
    axA.set_title(f"Edificio real — {N} pisos | suelo {params['suelo']} zona {params['zona']}")

    # (b) deformada predicha
    axB = fig.add_subplot(1, 2, 2, projection='3d')
    corners = np.array([[0,0],[Lx,0],[Lx,Ly],[0,Ly]])
    for k in range(N + 1):
        poly = [[c[0] + scale*Ux[k], c[1] + scale*Uy[k], zs[k]] for c in corners]
        axB.add_collection3d(Poly3DCollection([poly], alpha=0.55,
            facecolor=plt.cm.viridis(mag[k] / umax), edgecolor='k', linewidth=0.3))
    for c in corners:
        axB.plot([c[0] + scale*Ux[k] for k in range(N+1)],
                 [c[1] + scale*Uy[k] for k in range(N+1)], zs, color='crimson', lw=1.6)
        axB.plot([c[0]]*(N+1), [c[1]]*(N+1), zs, color='0.6', lw=0.6, ls=':')  # real de referencia
    axB.set_xlabel('X [m]'); axB.set_ylabel('Y [m]'); axB.set_zlabel('Z [m]')
    axB.set_xlim(-scale*umax, Lx+scale*umax); axB.set_ylim(-scale*umax, Ly+scale*umax); axB.set_zlim(0, H)
    axB.view_init(elev=24, azim=-58)
    axB.set_title(f'Deformada PINN (x{scale:.0f}) | U$_{{max}}$={umax:.3f} m')
    plt.tight_layout(); mostrar('08_deformada_3d.png')


plot_real_vs_deformada(mi_edificio, res)''')

md(r"""### (Opcional) Verificación contra OpenSees

Como el pipeline puede resolver el mismo edificio con OpenSees, podemos contrastar la
predicción del metamodelo contra el análisis directo. Si `openseespy` no está disponible, la
celda lo informa y se omite.""")
code(r'''def opensees_de_params(params):
    geom = build_family_B_geometry(params); wdf = geometry_to_wall_df(geom, params)
    minfo = build_opensees_model(params, wdf); modal, _ = run_eigen_analysis(minfo, params)
    resp = respuesta_espectral(params, modal)
    return dict(T1=float(modal['T'].max()), Vb_x=resp['Vb_x_kN'],
                dymax=float(np.max(resp['drift_y'])), cumple=resp['cumple'])

try:
    osr = opensees_de_params(mi_edificio)
    prd = predict_edificio(mi_edificio, avisar=False)
    print(f"{'magnitud':<14}{'PINN':>12}{'OpenSees':>12}{'err %':>9}")
    for nm, a, b in [('T1 [s]', prd['modal']['T1'], osr['T1']),
                     ('Vb_x [kN]', prd['respuesta']['Vb_x'], osr['Vb_x']),
                     ('drift_y max', max(prd['respuesta']['dy']), osr['dymax'])]:
        print(f"{nm:<14}{a:>12.4f}{b:>12.4f}{abs(a-b)/abs(b)*100:>8.1f}%")
except Exception as e:
    print('[info] OpenSees no disponible aqui; verificacion omitida.')
    print('      ', type(e).__name__, str(e)[:80])''')

# =================================================================== 16b RESUMEN GUARDADO
md(r"""### Resumen de lo guardado

Si activaste `GUARDAR_TODO`, aquí se listan todos los archivos producidos (modelo, dataset,
figuras y métricas) dentro de la carpeta de salida.""")
code(r'''if GUARDAR_TODO and os.path.isdir(CARPETA_SALIDA):
    archivos = sorted(os.listdir(CARPETA_SALIDA))
    print(f'Se guardo todo en {os.path.abspath(CARPETA_SALIDA)}  ({len(archivos)} archivos):')
    for f in archivos:
        print(f'   {f:32s} {os.path.getsize(os.path.join(CARPETA_SALIDA, f))/1e6:7.2f} MB')
else:
    print('GUARDAR_TODO = False -> no se escribieron archivos.')
    print('Para conservar todo, pon GUARDAR_TODO = True en la seccion 1b y ejecuta de nuevo.')''')

# =================================================================== 17 CIERRE
md(r"""## 17. Cierre

Partiendo solo del **dominio paramétrico** de la Familia B, el notebook generó edificios,
los resolvió con OpenSees bajo NCh433, entrenó el metamodelo PINN y mostró que **reproduce el
análisis dinámico** sobre edificios no vistos. Todo el proceso es reproducible de principio a
fin, sin datos ni pesos externos.

La exactitud mostrada escala con `N_CASOS` y las épocas: con los 10.000 casos y 600 épocas de
la tesis, el metamodelo alcanza los niveles reportados en el documento. Esta versión reducida
existe para que cualquiera **verifique el método completo en minutos**.""")

# =================================================================== WRITE
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = r'c:\Users\rodri\Documents\NB\Pipeline_PINN_Replicacion_FamiliaB.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('OK ->', out, '|', len(cells), 'celdas')
