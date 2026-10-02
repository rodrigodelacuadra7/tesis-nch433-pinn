# -*- coding: utf-8 -*-
"""Genera las celdas OpenSees-shell para casos arbitrarios, con geometria
de muros EXACTA tomada del wall_df del PKL de origen (misma fuente que el GT)."""
import pickle, h5py, numpy as np

PKL = r"C:/Users/rodri/Documents/PINN/b2_proyecto/dataset/final/dataset_familyB_cases_20260523_0758.pkl"
H5  = r"C:/Users/rodri/Documents/NB/dataset_familyB_OPENSEES_v6_20260523_1226.h5"
D = pickle.load(open(PKL, "rb"))
f = h5py.File(H5, "r")
FM = f["inputs/floor_masses"][:]; IR = f["stiffness/I_rot_floors"][:]
MASKF = f["inputs/mask_floors"][:]
T_r = f["modal/T_r"][:]; MT = f["modal/mode_types"][:]

def gt_periods(cid):
    types = [s.decode() for s in MT[cid]]
    def first(t):
        for r, ty in enumerate(types):
            if ty == t: return float(T_r[cid][r])
        return float("nan")
    return first("trans_Y"), first("torsional"), first("trans_X")

def wall_tuple(row):
    x, y, w, h = row.x, row.y, row.w, row.h
    if w >= h:  # corre en X
        return row.label, row.group, round(x,3), round(y,3), round(w,3), round(h,3), "X"
    else:       # corre en Y
        return row.label, row.group, round(x,3), round(y,3), round(h,3), round(w,3), "Y"

def build_cell(cid):
    c = D[cid]; p = c["params"]; wdf = c["wall_df"]
    N = int(p["N_pisos"]); hs = float(p["h_story_m"]); fc = float(p["fc_MPa"])
    E_bruto = 4700.0 * np.sqrt(fc) * 1e6
    t_nuc = float(p["t_muro_nucleo_m"]); t_bor = float(p["t_muro_borde_m"]); t_mid = float(p["t_muro_mid_m"])
    Lx = float(p["Lx_m"]); Ly = float(p["Ly_m"])
    CX, CY = Lx/2.0, Ly/2.0
    act = MASKF[cid] > 0
    MASS = [round(float(m),3) for m in FM[cid][act]]
    IROT = [float(m) for m in IR[cid][act]]
    T1y, T2t, T3x = gt_periods(cid)

    # ---- agrupar muros ----
    g = {k: [] for k in ["B1","B2","B3T","B3B","B4"]}
    for row in wdf.itertuples(index=False):
        t = wall_tuple(row)
        if row.group == "B1": g["B1"].append(t)
        elif row.group == "B2": g["B2"].append(t)
        elif row.group == "B4": g["B4"].append(t)
        elif row.group == "B3":
            (g["B3T"] if row.label.startswith("B3-T") else g["B3B"]).append(t)

    def fmt_explicit(tlist, tvar):
        # tuplas (lab,grp,xo,yo,L,esp,dir) -> linea get_walls con espesor simbolico
        out = []
        for lab,grp,xo,yo,L,esp,d in tlist:
            out.append(f'    w.append(("{lab}",\'{grp}\',{xo:.3f},{yo:.3f},{L:.3f},{tvar},\'{d}\'))')
        return "\n".join(out)

    def fmt_b3(tlist, name):
        # comparten yo y largo
        xs = [f"{t[2]:.3f}" for t in tlist]
        yo = tlist[0][3]; L = tlist[0][4]
        return (f'    {name} = [{", ".join(xs)}]\n'
                f'    for k,xo in enumerate({name}):\n'
                f'        w.append((f"{tlist[0][0][:4]}{{k+1}}",\'B3\',xo,{yo:.3f},{L:.3f},T_MID,\'Y\'))')

    mass_repr = (f"[{MASS[0]:.3f}]*{N-1} + [{MASS[-1]:.3f}]"
                 if len(set(MASS[:-1]))==1 else repr(MASS))
    irot_repr = (f"[{IROT[0]:.6e}]*{N-1} + [{IROT[-1]:.6e}]"
                 if len(set(IROT[:-1]))==1 else repr([f'{v:.6e}' for v in IROT]))

    walls_code = "def get_walls():\n    w = []\n"
    walls_code += "    # B1 nucleo\n" + fmt_explicit(g["B1"], "T_NUCLEO") + "\n"
    walls_code += "    # B2 fachadas extremas (Y)\n" + fmt_explicit(g["B2"], "T_BORDE") + "\n"
    walls_code += "    # B3 medianeros superiores (Y)\n" + fmt_b3(g["B3T"], "B3T") + "\n"
    walls_code += "    # B3 medianeros inferiores (Y)\n" + fmt_b3(g["B3B"], "B3B") + "\n"
    walls_code += "    # B4 pasillo (X)\n" + fmt_explicit(g["B4"], "T_MID") + "\n"
    walls_code += "    return w"

    header = f'''# ============================================================================
#  MODELO OPENSEES - MUROS SHELL - CASO {cid} (Familia B, {N} pisos)
#  Geometria de muros EXACTA (wall_df del PKL de origen, misma fuente que el GT)
#  Metodologia identica al caso 2100: diafragma rigido + masa concentrada +
#  E reducido (eta=0.35). Validacion externa shell (equivalente a Robot).
# ============================================================================
import numpy as np
import openseespy.opensees as ops
import time

# ----------------------------------------------------------------------------
#  1. PARAMETROS DEL CASO {cid}
# ----------------------------------------------------------------------------
N_PISOS   = {N}
H_STORY   = {hs:.2f}                  # m
H_TOTAL   = N_PISOS * H_STORY     # {N*hs:.2f} m

# Hormigon fc={fc:.0f} MPa, E=4700*sqrt(fc), agrietamiento DS61 (eta=0.35)
E_BRUTO   = {E_bruto:.4e}              # Pa  (= 4700*sqrt({fc:.0f}) MPa)
ETA       = 0.35
E_FIS     = ETA * E_BRUTO
NU        = 0.20
RHO_MAT   = 0.0

# Espesores por grupo (m)
T_NUCLEO  = {t_nuc:.3f}   # B1
T_BORDE   = {t_bor:.3f}   # B2
T_MID     = {t_mid:.3f}   # B3, B4

# Masas e inercias rotacionales por piso (HDF5, exactas del GT OpenSees)
MASS = {mass_repr}          # kg
IROT = {irot_repr}           # kg*m2

# Centro de masa (centroide de la planta = Lx/2, Ly/2)
CX, CY = {CX:.3f}, {CY:.3f}            # m

MESH = 1.4
TOL  = 1e-3

# Periodos ground truth (OpenSees columna ancha, para comparar)
T_GT = {{'T1_Y': {T1y:.4f}, 'T2_Tors': {T2t:.4f}, 'T3_X': {T3x:.4f}}}

# ----------------------------------------------------------------------------
#  2. GEOMETRIA DE MUROS  (label, grupo, x_orig, y_orig, largo, espesor, dir)
#     dir='Y' -> corre en Y (espesor en X) ; dir='X' -> corre en X (espesor en Y)
# ----------------------------------------------------------------------------
'''
    return header + walls_code + "\n"

REST = open(r"C:/Users/rodri/Documents/NB/_resto_celda.txt", encoding="utf-8").read()

CASOS_20 = [3034, 3078, 3357, 3362, 3570, 4051, 4827, 5362, 5671, 5821,
            6182, 6478, 6624, 6679, 6804, 7139, 7362, 7801, 8358, 9177]
for cid in CASOS_20:
    cell = build_cell(cid) + "\n" + REST
    fn = rf"C:/Users/rodri/Documents/NB/_celda_shell_{cid}.py"
    open(fn, "w", encoding="utf-8").write(cell)
    print("escrito", fn, "len", len(cell))
