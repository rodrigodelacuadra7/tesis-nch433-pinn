# -*- coding: utf-8 -*-
"""
Selecciona 20 casos "variables entre si" del dataset familyB (OpenSees GT)
Criterio: ESTRATIFICADO DIRIGIDO
  - cobertura explicita de N_pisos (13 valores), suelo (4), zona (3) y cumple (0/1)
  - dentro de las cuotas, maximiza diversidad por farthest-point sampling (maximin)
  - ancla = caso 2100 (ya modelado en ROBOT); los 20 nuevos se eligen lejos de el
Salida: _seleccion_20_casos.csv
"""
import h5py, numpy as np, pandas as pd

SEED = 42
ANCHOR = 2100
K = 20
np.random.seed(SEED)

H5 = "dataset_familyB_OPENSEES_v6_20260523_1226.h5"
f = h5py.File(H5, "r")
feat = [s.decode() for s in f["inputs/input_features"][:]]
X = f["inputs/X"][:].astype(float)
N = f["scalars/N_pisos"][:].astype(int)
cumple = f["scalars/cumple"][:].astype(int)
T = f["modal/T_r"][:]          # periodos (para reporte)
Vb = f["response/Vb_x_kN"][:]  # corte basal (para reporte)
f.close()

# columnas que realmente varian (descarta constantes)
varying = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 1]
Xv = X[:, varying]
# estandarizacion z-score para que ninguna variable domine la distancia
mu, sd = Xv.mean(0), Xv.std(0)
sd[sd == 0] = 1.0
Xs = (Xv - mu) / sd

idx_suelo = feat.index("suelo")
idx_zona = feat.index("zona")

def min_dist_to_set(cands, chosen):
    """distancia minima de cada candidato al conjunto ya elegido (en espacio z)."""
    C = Xs[chosen]                       # (m, d)
    P = Xs[cands]                         # (n, d)
    # ||p - c||^2 broadcast
    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return d2.min(1)

chosen = [ANCHOR]          # incluye ancla (no cuenta en los 20)
result = []

# ---- Cuota de N_pisos: cada uno de los 13 valores al menos 1 vez ----
floors = sorted(np.unique(N).tolist())          # 13 valores (6..18)
floor_quota = list(floors)                       # 13 picks dirigidos
# 7 picks libres restantes -> diversidad pura
free_picks = K - len(floor_quota)

available = np.ones(len(N), bool)
available[ANCHOR] = False

# Fase 1: una pasada por cada N_pisos, eligiendo el caso mas lejano (maximin)
for t in floor_quota:
    cand = np.where(available & (N == t))[0]
    if len(cand) == 0:
        continue
    md = min_dist_to_set(cand, chosen)
    pick = cand[int(np.argmax(md))]
    result.append(pick); chosen.append(pick); available[pick] = False

# Fase 2: picks libres, farthest-point puro sobre todo el pool
for _ in range(free_picks):
    cand = np.where(available)[0]
    md = min_dist_to_set(cand, chosen)
    pick = cand[int(np.argmax(md))]
    result.append(pick); chosen.append(pick); available[pick] = False

# ---- Fase 3: REPARACION de cobertura categorica (suelo/zona/cumple) ----
def coverage(res):
    return (set(X[res, idx_suelo].astype(int)),
            set(X[res, idx_zona].astype(int)),
            set(cumple[res]))

need = {"suelo": set([0, 1, 2, 3]), "zona": set([1, 2, 3]), "cumple": set([0, 1])}

def repair(res):
    res = list(res)
    for _ in range(20):  # iteraciones de reparacion
        s_cov, z_cov, c_cov = coverage(res)
        missing = []
        for v in need["suelo"] - s_cov: missing.append(("suelo", idx_suelo, v))
        for v in need["zona"] - z_cov: missing.append(("zona", idx_zona, v))
        if 0 not in c_cov: missing.append(("cumple", None, 0))
        if 1 not in c_cov: missing.append(("cumple", None, 1))
        if not missing:
            break
        kind, col, val = missing[0]
        # candidatos que satisfacen la categoria faltante
        if kind == "cumple":
            pool = np.where(available & (cumple == val))[0]
        else:
            pool = np.where(available & (X[:, col].astype(int) == val))[0]
        if len(pool) == 0:
            break
        # reemplaza el pick mas "redundante" (no critico para otra cobertura)
        # -> el que menor distancia minima aporta al resto
        best_new = pool[int(np.argmax(min_dist_to_set(pool, [ANCHOR] + res)))]
        # elegir cual quitar: el de menor contribucion de diversidad y no unico-en-su-categoria
        removable = []
        for r in res:
            s2, z2, c2 = coverage([x for x in res if x != r])
            # no romper coberturas ya logradas
            if (need["suelo"] & ({0,1,2,3}) - s2) and not (need["suelo"] - s_cov):
                pass
            removable.append(r)
        # quitar el que esta mas cerca de los demas (mas redundante)
        dmins = min_dist_to_set(np.array(res), [ANCHOR])
        # ordenar por redundancia (menor distancia al ancla/otros)
        order = np.argsort(dmins)
        for oi in order:
            cand_remove = res[oi]
            tmp = [x for x in res if x != cand_remove]
            s2, z2, c2 = coverage(tmp)
            ok = (len(s2) >= len(s_cov - {val}) and len(z2) >= len(z_cov - {val})
                  and len(c2) >= 1)
            if ok:
                res = tmp + [best_new]
                available[best_new] = False
                available[cand_remove] = True
                break
        else:
            res = [x for x in res if x != res[0]] + [best_new]
            available[best_new] = False
    return res

result = repair(result)
result = sorted(result)

# ---------- Reporte ----------
rows = []
for r in [ANCHOR] + result:
    rows.append({
        "case_id": r,
        "rol": "ANCLA(2100)" if r == ANCHOR else "nuevo",
        "N_pisos": int(N[r]),
        "suelo": int(X[r, idx_suelo]),
        "zona": int(X[r, idx_zona]),
        "cumple": int(cumple[r]),
        "fc_MPa": X[r, feat.index("fc_MPa")],
        "h_story_m": X[r, feat.index("h_story_m")],
        "n_unid_lado": int(X[r, feat.index("n_unid_lado")]),
        "t_nucleo": X[r, feat.index("t_muro_nucleo_m")],
        "t_borde": X[r, feat.index("t_muro_borde_m")],
        "T1_s": round(float(T[r, 0]), 3),
        "Vb_x_kN": round(float(Vb[r]), 1),
    })
df = pd.DataFrame(rows)
df.to_csv("_seleccion_20_casos.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print(df.to_string(index=False))
print("\n--- COBERTURA de los 20 nuevos ---")
print("N_pisos :", sorted(set(int(N[r]) for r in result)))
print("suelo   :", sorted(set(int(X[r, idx_suelo]) for r in result)))
print("zona    :", sorted(set(int(X[r, idx_zona]) for r in result)))
print("cumple  :", {int(v): int((cumple[result]==v).sum()) for v in [0,1]})

# diversidad: distancia minima media entre los 20 (mayor = mas diversos)
dm = min_dist_to_set(np.array(result), [ANCHOR] + result)  # incluye self=0
# recomputar pairwise sin self
P = Xs[result]
D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1))
np.fill_diagonal(D, np.inf)
print(f"\nDist. minima entre pares (media/min): {D.min(1).mean():.2f} / {D.min():.2f}")
print("CSV guardado -> _seleccion_20_casos.csv")
