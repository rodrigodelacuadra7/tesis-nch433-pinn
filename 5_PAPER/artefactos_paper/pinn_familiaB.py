# -*- coding: utf-8 -*-
"""
pinn_familiaB.py PTUEBA ORIGEN
================
Modulo autocontenido del metamodelo PINN (Famijjlia B / NCh433) — Trial 16.

Contiene TODO lo necesario para *usar* el modelo definitivo sin el notebook
de entrenamiento:

  - Arquitectura exacta (ResBlock + PINNModal_v4b) tal cual se entreno.
  - COLS_BASE (21 features one-hot) y construccion de X.
  - Carga de modelo + scalers.
  - Prediccion y desnormalizacion (periodos, deriva, corte basal).
  - Veredicto de cumplimiento NCh433 (deriva de entrepiso <= 0.002).

Es la pieza canonica de reutilizacion: la misma red que carga el state_dict
con 0 claves faltantes y reproduce las metricas del paper.

Tripleta indispensable (deben usarse JUNTOS):
  model_fase2_definitivo.pt  +  scalers_trial16_DEFINITIVO.pkl  +  dataset DEFINITIVO.h5
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Constantes del problema (Familia B v6)
# ----------------------------------------------------------------------------
N_MAX_PISOS = 18
N_MODOS     = 18
DRIFT_LIMIT = 0.002          # limite NCh433 de deriva de entrepiso (centro de masa)

# 21 columnas en el orden EXACTO del entrenamiento (Notebook_5_CORREGIDO, celda 3).
# 'mods_por_depto' y 'activar_B3' se excluyen por ser constantes (std=0).
COLS_BASE = [
    'N_pisos', 'n_unid_lado', 'activar_B2',
    'L_mod_m', 'prof_depto_m', 'ancho_corredor_m',
    'L_nucleo_m', 'B_nucleo_m', 'h_story_m',
    'fc_MPa', 'gk_kN_m2',
    't_muro_nucleo_m', 't_muro_borde_m', 't_muro_mid_m',
    'suelo_A', 'suelo_B', 'suelo_C', 'suelo_D',
    'zona_1', 'zona_2', 'zona_3',
]


# ----------------------------------------------------------------------------
# Arquitectura (identica al entrenamiento)
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.SiLU(),
                                 nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class PINNModal_v4(nn.Module):
    """PINNModal v4b: encoder_T1 dedicado a T1 + encoder compartido + 4 cabezales.
    Cabeza de periodos MONOTONA: T1 explicito, T_{r+1} = T_r - softplus(.) >= 0."""

    def __init__(self, n_inputs, n_modos=N_MODOS, n_max_pisos=N_MAX_PISOS,
                 hidden=256, hidden_T1=128):
        super().__init__()
        self.n_modos, self.n_max_pisos = n_modos, n_max_pisos
        self.encoder_T1 = nn.Sequential(
            nn.Linear(n_inputs, hidden_T1), nn.LayerNorm(hidden_T1),
            nn.SiLU(), ResBlock(hidden_T1), ResBlock(hidden_T1))
        self.head_T1 = nn.Sequential(nn.Linear(hidden_T1, 32), nn.SiLU(), nn.Linear(32, 1))
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            ResBlock(hidden), ResBlock(hidden), ResBlock(hidden), ResBlock(hidden))
        self.head_T_rest = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(),
                                         nn.Linear(128, n_modos - 1))
        self.head_Phi  = nn.Sequential(nn.Linear(hidden, 512), nn.SiLU(),
                                       nn.Linear(512, n_modos * n_max_pisos * 3))
        self.head_resp = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(),
                                       nn.Linear(256, n_max_pisos * 4))
        self.head_Vb   = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, X, mask):
        B = X.shape[0]
        T1 = self.head_T1(self.encoder_T1(X))
        h  = self.encoder(X)
        Td = F.softplus(self.head_T_rest(h))
        parts = [T1]
        for r in range(self.n_modos - 1):
            parts.append(parts[-1] - Td[:, r:r + 1])
        logT = torch.cat(parts, dim=1)
        Phi = self.head_Phi(h).view(B, self.n_modos, self.n_max_pisos, 3) \
            * mask.unsqueeze(1).unsqueeze(-1)
        Phi = (Phi / Phi.norm(dim=2, keepdim=True).clamp(min=1e-8)).permute(0, 2, 1, 3)
        resp = self.head_resp(h).view(B, self.n_max_pisos, 4)
        Ux = resp[:, :, 0] * mask
        Uy = resp[:, :, 1] * mask
        ratio = F.softplus(resp[:, :, 2]) * mask
        dy = resp[:, :, 3] * mask
        dx = torch.expm1(ratio.clamp(min=0)) * dy
        Vb = self.head_Vb(h)
        return logT, Phi[..., 0], Phi[..., 1], Phi[..., 2], Ux, Uy, dx, dy, Vb, ratio


# ----------------------------------------------------------------------------
# Carga de artefactos
# ----------------------------------------------------------------------------
def cargar_scalers(path='scalers_trial16_DEFINITIVO.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def cargar_modelo(path='model_fase2_definitivo.pt', n_inputs=len(COLS_BASE), device='cpu'):
    model = PINNModal_v4(n_inputs).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Construccion de X (one-hot) — igual que el entrenamiento
# ----------------------------------------------------------------------------
def construir_X(df_raw):
    """df_raw: DataFrame con las columnas crudas del HDF5 (incluye 'suelo','zona').
    Devuelve un array (N, 21) en el orden de COLS_BASE."""
    import pandas as pd
    df = df_raw.copy()
    suelo = pd.get_dummies(df['suelo'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'}),
                           prefix='suelo').astype(np.float32)
    zona = pd.get_dummies(df['zona'].astype(int), prefix='zona').astype(np.float32)
    df = pd.concat([df.drop(columns=['suelo', 'zona']), suelo, zona], axis=1)
    for c in COLS_BASE:
        if c not in df.columns:
            df[c] = 0.0
    return df[COLS_BASE].to_numpy(np.float32)


# ----------------------------------------------------------------------------
# Prediccion + desnormalizacion
# ----------------------------------------------------------------------------
@torch.no_grad()
def predecir(model, SC, X_clean, mask, device='cpu'):
    """Devuelve un dict con periodos [s], deriva maxima [-], Vbx/Vby [kN] y
    el perfil de deriva por piso. Desnormaliza EXACTAMENTE como el paper."""
    Xn = torch.tensor((X_clean - SC['X_mean']) / SC['X_std'], dtype=torch.float32, device=device)
    mk = torch.tensor(mask, dtype=torch.float32, device=device)
    logT, Phi_x, Phi_y, Phi_t, Ux, Uy, dx, dy, Vb, ratio = model(Xn, mk)

    T = np.exp(logT.cpu().numpy() * SC['logT_std'] + SC['logT_mean'])          # periodos [s]
    dy_real = np.exp(dy.cpu().numpy() * SC['Dy_std'] + SC['Dy_mean']) * mask    # deriva Y [-]
    dx_real = ratio.cpu().numpy() * dy_real                                     # deriva X [-]
    drift_perfil = np.maximum(dx_real, dy_real)
    drift_max = drift_perfil.max(axis=1)
    Vbx = np.exp(Vb[:, 0].cpu().numpy() * SC['Vbx_std'] + SC['Vbx_mean'])       # [kN]
    Vby = np.exp(Vb[:, 1].cpu().numpy() * SC['Vby_std'] + SC['Vby_mean'])       # [kN]
    return {'T': T, 'T1': T[:, 0], 'drift_max': drift_max, 'drift_perfil': drift_perfil,
            'Vbx': Vbx, 'Vby': Vby}


def veredicto_nch433(drift_max, limite=DRIFT_LIMIT):
    """True = CUMPLE (deriva <= limite)."""
    return drift_max <= limite


# ----------------------------------------------------------------------------
# Split reproducible del Trial 16 (zona 1+3 -> zona 2 OOD)
# ----------------------------------------------------------------------------
def split_trial16(zona_arr, seed=2026, trial=16, n_val=400):
    """Reconstruye el split EXACTO del run definitivo.
    Devuelve (idx_train, idx_val, idx_test)."""
    idx_pool_A = np.where((zona_arr == 1) | (zona_arr == 3))[0]   # TRAIN
    idx_pool_B = np.where(zona_arr == 2)[0]                       # OOD (zona 2)
    rng_b = np.random.default_rng(seed + trial * 200)
    idx_B_shuf = rng_b.permutation(idx_pool_B)
    return idx_pool_A, idx_B_shuf[:n_val], idx_B_shuf[n_val:]


def localizar_dataset(nombre='dataset_familyB_OPENSEES_v6_DEFINITIVO.h5'):
    """Busca el dataset en la carpeta actual y en la carpeta padre."""
    for p in (nombre, os.path.join('..', nombre)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No se encontro {nombre} (ni aqui ni en ..). '
        f'Coloca el HDF5 junto a este archivo o en la carpeta padre.')
