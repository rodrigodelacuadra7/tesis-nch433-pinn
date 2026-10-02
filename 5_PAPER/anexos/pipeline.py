# =============================================================================
# Anexo — Pipeline metodológico de referencia (versión secuencial)
#
# Esta es la versión LINEALIZADA y FACTORIZADA del pipeline completo de la tesis,
# destilada de los notebooks y workers de produccion. Toma los parametros de UN
# edificio de la Familia B y recorre todo el flujo, desde el modelo FEM hasta el
# entrenamiento del metamodelo. Se omiten la paralelizacion (12 workers),
# el manejo de archivos y el control de errores de produccion para privilegiar
# la legibilidad del METODO.
#
# El procedimiento de analisis modal espectral corresponde a la version
# DEFINITIVA (v6), conforme a NCh433:2026:
#   - alpha evaluado con (T0, p) del espectro de diseno,
#   - R* con R0,
#   - limites Cmin/Cmax aplicados al corte basal COMBINADO (no por ordenada),
#     y Cmax sin afectar los desplazamientos.
# =============================================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. CONSTANTES Y PARAMETROS NCh433:2026
# =============================================================================
G            = 9.80665     # gravedad [m/s2]
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
# numero de modos extraidos segun la altura (esquema adaptativo, NB3.2)
N_MODOS_POR_PISOS = {n: 3 * n for n in range(6, 19)}        # N=6->18, ..., N=18->54


# =============================================================================
# 2. ESPECTRO DE DISENO NCh433:2026  (Sa = I*S*A0*alpha/R*, sin tope por ordenada)
# =============================================================================
def alpha_nch433(T, T0, p):
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
    return float(np.sqrt(max(s, 0.0)))


# =============================================================================
# 3. MODELO FEM EN OPENSEES (un edificio Familia B)
#    Muros = elasticBeamColumn (seccion agrietada), nodo maestro por piso,
#    diafragma rigido, 3 GDL libres por piso (Ux, Uy, Rz).  NCh433 6.1.1.
# =============================================================================
def construir_modelo(params, wall_df):
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

    return dict(master=master, floor_masses=floor_masses, N=N, h=h)


# =============================================================================
# 4. ANALISIS MODAL ADAPTATIVO  (numero de modos segun la altura)
# =============================================================================
def analisis_modal(model_info, params, n_modes=None):
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
                cumple_90=(mpx.sum() >= 0.90 and mpy.sum() >= 0.90))


# =============================================================================
# 5. CLASIFICACION Y ORDENAMIENTO EN 18 SLOTS  (patron Y, T, X)
#    Cada modo se clasifica por energia: torsional si Et/Total > 0.40.
#    Los slots 1 y 4 (T1 y T2) deben quedar ocupados para que el caso sea valido.
# =============================================================================
def clasificar_modos(modal, N, n_out=N_MODOS):
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
                mpart_x=mx18, mpart_y=my18, valido=valido)


# =============================================================================
# 6. RESPUESTA ESPECTRAL + TORSION ACCIDENTAL  (procedimiento DEFINITIVO v6)
#    Cmin/Cmax sobre el corte COMBINADO; Cmax NO afecta los desplazamientos.
# =============================================================================
def respuesta_espectral(params, modal):
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
                Vb_x_kN=Vb_x/1000.0, Vb_y_kN=Vb_y/1000.0, cumple=cumple)


# =============================================================================
# 7. ENSAMBLAJE DEL CASO:  padding a 18 pisos + mascara binaria
# =============================================================================
def ensamblar_caso(inputs_18, modal18, respuesta, N):
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
    )


# =============================================================================
# 8. METAMODELO  PINNModal_v4
#    Encoder dedicado a T1 + encoder compartido -> 4 cabezales especializados.
# =============================================================================
class ResBlock(nn.Module):
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
        return logT, Phi[:, :, :, 0], Phi[:, :, :, 1], Phi[:, :, :, 2], Ux, Uy, ratio, dy, Vb


# =============================================================================
# 9. FUNCION DE PERDIDA HIBRIDA  (datos + fisica, enmascarada por pisos/modos)
# =============================================================================
def perdida_hibrida(batch, out, lambdas, stage, device='cuda'):
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
    return Phi / Phi.norm(dim=1, keepdim=True).clamp(min=1e-8)


# =============================================================================
# 10. CURRICULO DE PESOS lambda  (activacion progresiva de los terminos fisicos)
# =============================================================================
def curriculo_lambda(epoch):
    lam = {'data_resp': 2.0, 'data_T_rest': 1.0, 'ratio': 1.0,
           'data_T1': 0.0, 'phi': 0.0, 'T1': 0.0, 'eig': 0.0, 'ortho': 0.0}
    stage = {'phi': False, 'eig': False, 'ortho': False}
    if epoch >= 150: lam['data_T1'] = 1.0; lam['phi'] = 0.1; stage['phi'] = True
    if epoch >= 200: lam['T1'] = 1.0; lam['eig'] = 0.1; stage['eig'] = True
    if epoch >= 350: lam['ortho'] = 1e-5; stage['ortho'] = True
    return lam, stage


# =============================================================================
# 11. ENTRENAMIENTO  (Adam + Cosine warm restarts + dos velocidades)
# =============================================================================
def entrenar(model, loader, epochs=600, device='cuda'):
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
    return model


# =============================================================================
# 12. DRIVER SECUENCIAL  (un caso de punta a punta + esquema de entrenamiento)
# =============================================================================
if __name__ == '__main__':
    # --- parametros de un edificio de ejemplo (Familia B) ---
    params = dict(N_pisos=12, h_story_m=2.7, E_MPa=23500.0, A_geom_m2=320.0,
                  gk_kN_m2=8.0, Lx_m=36.0, Ly_m=12.0, zona=3, suelo='C')

    # (1) FEM -> (2) modal adaptativo -> (3) clasificacion -> (4) respuesta NCh433
    # wall_df proviene del archivo PKL del edificio (tabla de muros)
    # modelo   = construir_modelo(params, wall_df)
    # modal    = analisis_modal(modelo, params)
    # modal18  = clasificar_modos(modal, params['N_pisos'])
    # resp     = respuesta_espectral(params, modal)
    # caso     = ensamblar_caso(X_18, modal18, resp, params['N_pisos'])
    # ... repetir para los 10.000 casos -> dataset HDF5

    # (5) entrenamiento del metamodelo sobre el dataset ensamblado
    model = PINNModal_v4()
    # entrenar(model, loader)         # loader: DataLoader sobre el dataset HDF5
    print('Pipeline de referencia cargado.')
