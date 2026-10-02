# -*- coding: utf-8 -*-
"""Matriz de confusion de cumplimiento normativo (deriva <= 0.002) del modelo final."""
import numpy as np, pickle, h5py, torch, torch.nn as nn, torch.nn.functional as F
dev='cpu'; LIM=0.002
MODEL=r"C:/Users/rodri/Documents/PINN/b2_proyecto/models/trials_31/V8/DEFINITIVO/model_fase2_definitivo.pt"
SCAL =r"C:/Users/rodri/Documents/PINN/b2_proyecto/models/trials_31/V8/DEFINITIVO/scalers_trial16.pkl"
H5   ="C:/Users/rodri/Documents/NB/2_DATASET/datasets_h5/dataset_familyB_OPENSEES_v6_DEFINITIVO.h5"

class ResBlock(nn.Module):
    def __init__(s,d):
        super().__init__(); s.net=nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.SiLU(),nn.Linear(d,d),nn.LayerNorm(d)); s.act=nn.SiLU()
    def forward(s,x): return s.act(x+s.net(x))
class PINNModal_v4(nn.Module):
    def __init__(s,n_inputs=21,n_modos=18,n_max_pisos=18,hidden=256,hidden_T1=128):
        super().__init__(); s.n_modos=n_modos; s.n_max_pisos=n_max_pisos
        s.encoder_T1=nn.Sequential(nn.Linear(n_inputs,hidden_T1),nn.LayerNorm(hidden_T1),nn.SiLU(),ResBlock(hidden_T1),ResBlock(hidden_T1))
        s.head_T1=nn.Sequential(nn.Linear(hidden_T1,32),nn.SiLU(),nn.Linear(32,1))
        s.encoder=nn.Sequential(nn.Linear(n_inputs,hidden),nn.LayerNorm(hidden),nn.SiLU(),ResBlock(hidden),ResBlock(hidden),ResBlock(hidden),ResBlock(hidden))
        s.head_T_rest=nn.Sequential(nn.Linear(hidden,128),nn.SiLU(),nn.Linear(128,n_modos-1))
        s.head_Phi=nn.Sequential(nn.Linear(hidden,512),nn.SiLU(),nn.Linear(512,n_modos*n_max_pisos*3))
        s.head_resp=nn.Sequential(nn.Linear(hidden,256),nn.SiLU(),nn.Linear(256,n_max_pisos*4))
        s.head_Vb=nn.Sequential(nn.Linear(hidden,64),nn.SiLU(),nn.Linear(64,2))
    def forward(s,X,mask):
        B=X.shape[0]; hT1=s.encoder_T1(X); T1=s.head_T1(hT1); h=s.encoder(X)
        Td=F.softplus(s.head_T_rest(h)); parts=[T1]
        for r in range(s.n_modos-1): parts.append(parts[-1]-Td[:,r:r+1])
        logT=torch.cat(parts,1)
        resp=s.head_resp(h).view(B,s.n_max_pisos,4)
        Ux=resp[...,0]*mask; Uy=resp[...,1]*mask; ratio=F.softplus(resp[...,2])*mask; dy=resp[...,3]*mask; dx=ratio*dy
        Vb=s.head_Vb(h)
        return logT,None,None,None,Ux,Uy,dx,dy,Vb,ratio

m=PINNModal_v4().to(dev); m.load_state_dict(torch.load(MODEL,map_location=dev)); m.eval()
SC=pickle.load(open(SCAL,'rb')); COLS=SC['COLS_BASE']

f=h5py.File(H5,'r'); feat=[t.decode() for t in f['inputs/input_features'][:]]
X=f['inputs/X'][:]; N=f['scalars/N_pisos'][:].astype(int); maskf=f['inputs/mask_floors'][:].astype(np.float32)
dxm_t=f['response/drift_x'][:]; dym_t=f['response/drift_y'][:]; f.close()
true_dx=np.array([dxm_t[i,:N[i]].max() for i in range(len(N))])
true_dy=np.array([dym_t[i,:N[i]].max() for i in range(len(N))])

import pandas as pd
df=pd.DataFrame(X,columns=feat); sm={0:'A',1:'B',2:'C',3:'D'}
soh=pd.get_dummies(df['suelo'].map(sm),prefix='suelo').astype('float32')
zoh=pd.get_dummies(df['zona'].astype(int),prefix='zona').astype('float32')
df=pd.concat([df.drop(columns=['suelo','zona']),soh,zoh],axis=1)
for c in COLS:
    if c not in df.columns: df[c]=0.0
Xn=((df[COLS].to_numpy('float32'))-SC['X_mean'])/SC['X_std']
with torch.no_grad():
    out=m(torch.from_numpy(Xn).to(dev), torch.from_numpy(maskf).to(dev))
    dx=out[6].cpu().numpy(); dy=out[7].cpu().numpy()
pred_dx=np.array([np.exp(dx[i,:N[i]]*SC['Dx_std']+SC['Dx_mean']).max() for i in range(len(N))])
pred_dy=np.array([np.exp(dy[i,:N[i]]*SC['Dy_std']+SC['Dy_mean']).max() for i in range(len(N))])

cumple_true = (true_dx<=LIM) & (true_dy<=LIM)
cumple_pred = (pred_dx<=LIM) & (pred_dy<=LIM)
TP=int(( cumple_pred &  cumple_true).sum())   # predice cumple, es cumple
TN=int((~cumple_pred & ~cumple_true).sum())   # predice no-cumple, es no-cumple
FP=int(( cumple_pred & ~cumple_true).sum())   # predice cumple, NO cumple  <- PELIGROSO
FN=int((~cumple_pred &  cumple_true).sum())   # predice no-cumple, SI cumple <- conservador
n=len(N)
print("="*60); print("MATRIZ DE CONFUSION — CUMPLIMIENTO (deriva <= 0.002)"); print("="*60)
print(f"Total casos: {n}   |   cumple real: {int(cumple_true.sum())}   no-cumple real: {int((~cumple_true).sum())}")
print(f"\n                      | REAL cumple | REAL no-cumple")
print(f"  PRED cumple         |   TP={TP:5d}  |   FP={FP:4d}  (peligroso)")
print(f"  PRED no-cumple      |   FN={FN:5d}  |   TN={TN:4d}  (conservador FN)")
print(f"\nExactitud global       : {100*(TP+TN)/n:.2f}%")
print(f"Falsos positivos (FP)  : {FP}  ({100*FP/max(int((~cumple_true).sum()),1):.2f}% de los no-cumple reales escapan como 'cumple')")
print(f"Falsos negativos (FN)  : {FN}  ({100*FN/max(int(cumple_true.sum()),1):.2f}% de los cumple reales marcados 'no-cumple')")
print(f"Recall no-cumple (detecta incumplimiento): {100*TN/max(TN+FP,1):.2f}%")
print(f"Precision cumple        : {100*TP/max(TP+FP,1):.2f}%")
# cuan cerca del umbral estan los FP/FN
fp_mask=cumple_pred & ~cumple_true; fn_mask=~cumple_pred & cumple_true
real_max=np.maximum(true_dx,true_dy)
if fp_mask.sum(): print(f"\nFP: deriva real mediana={np.median(real_max[fp_mask]):.5f}  (umbral {LIM}); P90={np.percentile(real_max[fp_mask],90):.5f}")
if fn_mask.sum(): print(f"FN: deriva real mediana={np.median(real_max[fn_mask]):.5f}  (umbral {LIM})")

# ---- Subconjunto OOD del Trial 16: zona 2 reservada para test ----
zona = X[:, feat.index('zona')].astype(int)
for etiqueta, sel in [('TEST OOD Trial16 (zona==2)', zona==2),
                      ('train zonas 1,3', np.isin(zona,[1,3]))]:
    ct=cumple_true[sel]; cp=cumple_pred[sel]
    TP=int((cp&ct).sum()); TN=int((~cp&~ct).sum())
    FP=int((cp&~ct).sum()); FN=int((~cp&ct).sum()); nn=int(sel.sum())
    print(f"\n=== {etiqueta} (n={nn}) ===")
    print(f"  cumple real={int(ct.sum())}  no-cumple real={int((~ct).sum())}")
    print(f"  TP={TP} FP={FP}(peligroso) FN={FN}(conserv) TN={TN}")
    print(f"  Exactitud={100*(TP+TN)/nn:.2f}%  FP%no-cumple={100*FP/max(int((~ct).sum()),1):.2f}  FN%cumple={100*FN/max(int(ct.sum()),1):.2f}")
    rm=np.maximum(true_dx,true_dy)[sel]; fpm=(cp&~ct); fnm=(~cp&ct)
    if fpm.sum(): print(f"  FP deriva real mediana={np.median(rm[fpm]):.5f}")
    if fnm.sum(): print(f"  FN deriva real mediana={np.median(rm[fnm]):.5f}")
