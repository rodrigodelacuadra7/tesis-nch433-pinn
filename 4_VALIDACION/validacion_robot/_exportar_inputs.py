# -*- coding: utf-8 -*-
"""
Exporta, para los 20 casos seleccionados (+ ancla 2100):
  (A) _inputs_PINN_21cols.csv  -> vector de entrada EXACTO de la PINN (COLS_BASE, 21 col)
  (B) _robot_inputs.csv        -> tabla de ingenieria lista para modelar en ROBOT
  (C) _robot_masas_piso.csv    -> masa/peso sismico por piso (largo)
Constantes del ground truth (OpenSees v6 / NCh433 DS61):
  E = 4700*sqrt(fc) MPa,  nu=0.20,  rho=2400 kg/m3,  rigidez agrietada eta=0.35
  qk=2.0 kN/m2, psi=0.25, gk variable;  I=1.0(catII), R=7, Ro=11 muros, xi=5%
"""
import h5py, numpy as np, pandas as pd

H5 = "dataset_familyB_OPENSEES_v6_20260523_1226.h5"
SEL = pd.read_csv("_seleccion_20_casos.csv")
ids = SEL["case_id"].tolist()              # incluye 2100 (ancla) al inicio
ANCHOR = 2100

f = h5py.File(H5, "r")
feat = [s.decode() for s in f["inputs/input_features"][:]]
X = f["inputs/X"][:].astype(float)
N = f["scalars/N_pisos"][:].astype(int)
cumple = f["scalars/cumple"][:].astype(int)
floor_masses = f["inputs/floor_masses"][:].astype(float)   # (10000,18)
mask_floors = f["inputs/mask_floors"][:].astype(float)
T_r = f["modal/T_r"][:].astype(float)
Vb_x = f["response/Vb_x_kN"][:].astype(float)
Vb_y = f["response/Vb_y_kN"][:].astype(float)
f.close()

def col(name): return X[:, feat.index(name)]

# ======================================================================
# (A) INPUT EXACTO DE LA PINN  -> 21 columnas COLS_BASE con one-hot
# ======================================================================
suelo_map = {0: "A", 1: "B", 2: "C", 3: "D"}
df = pd.DataFrame(X, columns=feat)
suelo_ohe = pd.get_dummies(df["suelo"].map(suelo_map), prefix="suelo").astype(int)
zona_ohe = pd.get_dummies(df["zona"].astype(int), prefix="zona").astype(int)
df = df.drop(columns=["suelo", "zona"])
df = pd.concat([df, suelo_ohe, zona_ohe], axis=1)
for c in ["suelo_A","suelo_B","suelo_C","suelo_D","zona_1","zona_2","zona_3"]:
    if c not in df.columns: df[c] = 0
COLS_BASE = [
    "N_pisos","n_unid_lado","activar_B2",
    "L_mod_m","prof_depto_m","ancho_corredor_m",
    "L_nucleo_m","B_nucleo_m","h_story_m",
    "fc_MPa","gk_kN_m2",
    "t_muro_nucleo_m","t_muro_borde_m","t_muro_mid_m",
    "suelo_A","suelo_B","suelo_C","suelo_D",
    "zona_1","zona_2","zona_3",
]
pinn = df.loc[ids, COLS_BASE].copy()
pinn.insert(0, "rol", ["ANCLA" if i == ANCHOR else "nuevo" for i in ids])
pinn.insert(0, "case_id", ids)
pinn.to_csv("_inputs_PINN_21cols.csv", index=False)

# ======================================================================
# (B) TABLA PARA ROBOT  -> geometria + material + cargas + sismo
# ======================================================================
SOIL_PARAMS = {
    "A": dict(S=0.90, T0=0.15, Tp=0.20, n=1.00, p=2.0),
    "B": dict(S=1.00, T0=0.30, Tp=0.35, n=1.33, p=1.5),
    "C": dict(S=1.05, T0=0.40, Tp=0.45, n=1.40, p=1.6),
    "D": dict(S=1.20, T0=0.75, Tp=0.85, n=1.80, p=1.0),
}
A0_ZONE = {1: 0.20, 2: 0.30, 3: 0.40}
I_CAT, R, Ro, XI, ETA, NU, RHO, QK, PSI = 1.0, 7.0, 11.0, 0.05, 0.35, 0.20, 2400.0, 2.0, 0.25
TECH_MODS = 2   # bahia tecnica cada lado (mods_por_depto=2 * 1 bahia)

rows = []
for i in ids:
    nu_lado = int(col("n_unid_lado")[i])
    Lm = col("L_mod_m")[i]; prof = col("prof_depto_m")[i]; wc = col("ancho_corredor_m")[i]
    Ln = col("L_nucleo_m")[i]; Bn = col("B_nucleo_m")[i]
    Np = int(N[i]); hs = col("h_story_m")[i]; fc = col("fc_MPa")[i]
    n_mod_total = 2 * (2 * nu_lado + TECH_MODS)        # = 4*n_unid_lado + 4
    Lx = n_mod_total * Lm + Ln
    Ly = 2 * prof + wc
    H = Np * hs
    E_MPa = 4700.0 * np.sqrt(fc)
    G_MPa = E_MPa / (2 * (1 + NU))
    s = suelo_map[int(col("suelo")[i])]; z = int(col("zona")[i])
    sp = SOIL_PARAMS[s]
    # masa / peso sismico
    m_tot = floor_masses[i][mask_floors[i] > 0].sum()       # kg
    W_tot = m_tot * 9.80665 / 1000.0                        # kN
    rows.append({
        "case_id": i,
        "rol": "ANCLA" if i == ANCHOR else "nuevo",
        "cumple_NCh433": int(cumple[i]),
        # ---- GEOMETRIA GLOBAL ----
        "N_pisos": Np, "h_story_m": round(hs,2), "H_total_m": round(H,2),
        "Lx_m": round(Lx,3), "Ly_m": round(Ly,3), "A_planta_m2": round(Lx*Ly,1),
        # ---- PROGRAMA / MODULACION ----
        "n_unid_lado": nu_lado, "n_mod_total": n_mod_total,
        "L_mod_m": round(Lm,2), "prof_depto_m": round(prof,2), "ancho_corredor_m": round(wc,2),
        "L_nucleo_m": round(Ln,2), "B_nucleo_m": round(Bn,2),
        # ---- MUROS (espesores) ----
        "t_nucleo_B1_m": round(col("t_muro_nucleo_m")[i],3),
        "t_borde_B2_m":  round(col("t_muro_borde_m")[i],3),
        "activar_B2":    int(col("activar_B2")[i]),
        "t_mid_B3_m":    round(col("t_muro_mid_m")[i],3),
        # ---- MATERIAL ----
        "fc_MPa": fc, "E_MPa": round(E_MPa,0), "G_MPa": round(G_MPa,0),
        "nu": NU, "rho_kg_m3": RHO, "eta_agrietado": ETA,
        # ---- CARGAS ----
        "gk_kN_m2": round(col("gk_kN_m2")[i],2), "qk_kN_m2": QK, "psi_sismo": PSI,
        "W_sismico_kN": round(W_tot,1), "m_sismica_ton": round(m_tot/1000.0,1),
        # ---- SISMO NCh433 ----
        "zona": z, "A0_g": A0_ZONE[z], "suelo": s,
        "S": sp["S"], "T0_s": sp["T0"], "Tp_s": sp["Tp"], "n_suelo": sp["n"], "p_suelo": sp["p"],
        "I_cat": I_CAT, "R": R, "Ro": Ro, "xi": XI,
        # ---- referencia GT ----
        "T1_GT_s": round(T_r[i,0],3), "Vbx_GT_kN": round(Vb_x[i],1), "Vby_GT_kN": round(Vb_y[i],1),
    })
robot = pd.DataFrame(rows)
robot.to_csv("_robot_inputs.csv", index=False)

# ======================================================================
# (C) MASA / PESO SISMICO POR PISO (formato largo)
# ======================================================================
mp = []
for i in ids:
    nf = int(N[i])
    for piso in range(1, nf+1):
        m = floor_masses[i][piso-1]
        mp.append({"case_id": i, "piso": piso,
                   "m_ton": round(m/1000.0,2), "W_kN": round(m*9.80665/1000.0,1)})
pd.DataFrame(mp).to_csv("_robot_masas_piso.csv", index=False)

# ---------------- impresion ----------------
pd.set_option("display.width", 240, "display.max_columns", 60)
print("######## (A) INPUT PINN (21 col COLS_BASE) ########")
print(pinn.to_string(index=False))
print("\nGuardado -> _inputs_PINN_21cols.csv\n")
print("######## (B) ROBOT - GEOMETRIA + MATERIAL + CARGAS (extracto) ########")
print(robot[["case_id","N_pisos","H_total_m","Lx_m","Ly_m","n_mod_total",
             "t_nucleo_B1_m","t_borde_B2_m","activar_B2","t_mid_B3_m",
             "fc_MPa","E_MPa","gk_kN_m2","W_sismico_kN"]].to_string(index=False))
print("\n######## (B) ROBOT - SISMO NCh433 (extracto) ########")
print(robot[["case_id","zona","A0_g","suelo","S","T0_s","Tp_s","p_suelo",
             "I_cat","R","Ro","xi","T1_GT_s","Vbx_GT_kN"]].to_string(index=False))
print("\nGuardado -> _robot_inputs.csv  (44 columnas, todo)")
print("Guardado -> _robot_masas_piso.csv  (masa y peso sismico por piso)")
