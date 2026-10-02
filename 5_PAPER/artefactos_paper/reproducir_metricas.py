# -*- coding: utf-8 -*-
"""
reproducir_metricas.py
======================
Reproduce, de punta a punta y SIN reentrenar, las metricas OOD reportadas
en el paper (Trial 16 — entrena zonas 1+3, evalua zona 2).

Resultado esperado (de resultado_fase2_definitivo.json):
    n_test (OOD zona 2) = 2977
    drift_MAPE = 13.64 %
    T1_MAPE    =  7.51 %
    Vb_MAPE    = 11.45 %
    MAC_avg    =  0.9979

Uso:
    python reproducir_metricas.py

Requiere (en esta carpeta o en la carpeta padre, para el .h5):
    model_fase2_definitivo.pt
    scalers_trial16_DEFINITIVO.pkl
    dataset_familyB_OPENSEES_v6_DEFINITIVO.h5
"""
import json
import numpy as np
import pandas as pd
import torch
import h5py

import pinn_familiaB as M


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # ---- 1. Cargar dataset --------------------------------------------------
    ds = M.localizar_dataset()
    print(f'dataset: {ds}')
    with h5py.File(ds, 'r') as h5:
        X_raw  = h5['inputs/X'][:].astype(np.float32)
        feats  = [s.decode() for s in h5['inputs/input_features'][:]]
        mask   = h5['inputs/mask_floors'][:].astype(np.float32)
        T_r    = h5['modal/T_r'][:].astype(np.float32)
        Phi_x  = h5['modal/Phi_x'][:].astype(np.float32)
        Phi_y  = h5['modal/Phi_y'][:].astype(np.float32)
        Phi_t  = h5['modal/Phi_theta'][:].astype(np.float32)
        driftx = h5['response/drift_x'][:].astype(np.float32)
        drifty = h5['response/drift_y'][:].astype(np.float32)
        Vb_x   = h5['response/Vb_x_kN'][:].astype(np.float32)
        Vb_y   = h5['response/Vb_y_kN'][:].astype(np.float32)

    # Normalizar Phi a norma vectorial = 1 por modo (igual que el entrenamiento)
    with np.errstate(divide='ignore', invalid='ignore'):
        for r in range(Phi_x.shape[2]):
            for A in (Phi_x, Phi_y, Phi_t):
                nrm = np.sqrt((A[:, :, r] ** 2).sum(axis=1, keepdims=True))
                A[:, :, r] = np.where(nrm > 1e-10, A[:, :, r] / nrm, 0.0)

    # ---- 2. Construir X y reconstruir el split del Trial 16 -----------------
    X_clean = M.construir_X(pd.DataFrame(X_raw, columns=feats))
    zona = X_raw[:, feats.index('zona')].astype(int)
    idx_train, idx_val, idx_test = M.split_trial16(zona)
    print(f'train (zona 1+3) = {len(idx_train)} | val = {len(idx_val)} | '
          f'test OOD (zona 2) = {len(idx_test)}')

    # ---- 3. Cargar tripleta y predecir sobre el test ------------------------
    SC = M.cargar_scalers()
    model = M.cargar_modelo(n_inputs=len(SC['COLS_BASE']), device=device)
    pred = M.predecir(model, SC, X_clean[idx_test], mask[idx_test], device=device)

    # ---- 4. Metricas (identicas a compute_metrics_ood) ----------------------
    mk = mask[idx_test]

    # drift_MAPE
    drift_true = np.maximum(driftx[idx_test] * mk, drifty[idx_test] * mk).max(axis=1)
    md = drift_true > 1e-8
    drift_MAPE = np.mean(np.abs(pred['drift_max'][md] - drift_true[md]) /
                         (drift_true[md] + 1e-8)) * 100

    # T1_MAPE
    T1_true = T_r[idx_test, 0]
    mt = T1_true > 1e-6
    T1_MAPE = np.mean(np.abs(pred['T1'][mt] - T1_true[mt]) / (T1_true[mt] + 1e-8)) * 100

    # Vb_MAPE
    Vbx_MAPE = np.mean(np.abs(pred['Vbx'] - Vb_x[idx_test]) / (Vb_x[idx_test] + 1e-8)) * 100
    Vby_MAPE = np.mean(np.abs(pred['Vby'] - Vb_y[idx_test]) / (Vb_y[idx_test] + 1e-8)) * 100
    Vb_MAPE = (Vbx_MAPE + Vby_MAPE) / 2

    # MAC (slot 0 = Phi_y, slot 1 = Phi_theta, slot 2 = Phi_x)
    Xn = torch.tensor((X_clean[idx_test] - SC['X_mean']) / SC['X_std'],
                      dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(Xn, torch.tensor(mk, dtype=torch.float32, device=device))
    Phi_pred = {'Phi_y': out[2].cpu().numpy(), 'Phi_theta': out[3].cpu().numpy(),
                'Phi_x': out[1].cpu().numpy()}
    Phi_true = {'Phi_y': Phi_y[idx_test], 'Phi_theta': Phi_t[idx_test], 'Phi_x': Phi_x[idx_test]}
    macs = []
    for slot, key in [(0, 'Phi_y'), (1, 'Phi_theta'), (2, 'Phi_x')]:
        pp = Phi_pred[key][:, :, slot] * mk
        pt = Phi_true[key][:, :, slot] * mk
        num = (pp * pt).sum(axis=1) ** 2
        den = ((pp ** 2).sum(axis=1) * (pt ** 2).sum(axis=1)).clip(min=1e-12)
        ok = (pt ** 2).sum(axis=1) > 1e-10
        macs.append(np.mean((num / den)[ok]))
    MAC_avg = float(np.mean(macs))

    # ---- 5. Reporte y comparacion contra el JSON reportado ------------------
    try:
        ref = json.load(open('resultado_fase2_definitivo.json', encoding='utf-8'))
    except Exception:
        ref = {}

    print('\n' + '=' * 60)
    print(f'{"METRICA":<14}{"REPRODUCIDO":>14}{"REPORTADO":>14}')
    print('-' * 60)
    filas = [('n_test', len(idx_test), ref.get('n_test', ref.get('n_casos_ood', 2977))),
             ('drift_MAPE', drift_MAPE, ref.get('drift_MAPE', 13.64)),
             ('T1_MAPE', T1_MAPE, ref.get('T1_MAPE', 7.51)),
             ('Vb_MAPE', Vb_MAPE, ref.get('Vb_MAPE', 11.45)),
             ('MAC_avg', MAC_avg, ref.get('MAC_avg', 0.9979))]
    for nombre, got, exp in filas:
        if nombre == 'n_test':
            print(f'{nombre:<14}{got:>14d}{exp:>14}')
        else:
            print(f'{nombre:<14}{got:>14.4f}{float(exp):>14.4f}')
    print('=' * 60)
    print('Si los valores coinciden, la tripleta (modelo+scalers+dataset) es correcta.')


if __name__ == '__main__':
    main()
