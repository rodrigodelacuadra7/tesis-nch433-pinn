# -*- coding: utf-8 -*-
"""Matriz de confusion de cumplimiento (deriva <= 0.002) sobre el conjunto OOD del
Trial 16 (zona 2), evaluado con el modelo DEFINITIVO/CORREGIDO. Agrega estadisticas
de banda marginal para la seccion 6.4 de la tesis."""
import numpy as np, pickle, h5py, torch, torch.nn as nn, torch.nn.functional as F
dev='cpu'; LIM=0.002
MODEL=r"C:/Users/rodri/Documents/PINN/b2_proyecto/models/trials_31/V8/DEFINITIVO_CORREGIDO/model_fase2_definitivo.pt"
SCAL =r"C:/Users/rodri/Documents/NB/3_METAMODELO/modelo/scalers_trial16_DEFINITIVO.pkl"
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
real_max=np.maximum(true_dx,true_dy)

# ==== Conjunto OOD Trial 16 = zona 2 (test), NO visto en entrenamiento (train = zonas 1,3) ====
zona = X[:, feat.index('zona')].astype(int)
sel = (zona==2)
ct=cumple_true[sel]; cp=cumple_pred[sel]; rm=real_max[sel]
TP=int((cp&ct).sum()); TN=int((~cp&~ct).sum())
FP=int((cp&~ct).sum()); FN=int((~cp&ct).sum()); n=int(sel.sum())
print("="*66)
print("MATRIZ DE CONFUSION — CUMPLIMIENTO (deriva <= 0.002)")
print("Conjunto OOD del Trial 16 = zona sismica 2 (test), modelo DEFINITIVO")
print("="*66)
print(f"n = {n}  |  cumple real = {int(ct.sum())}  |  no-cumple real = {int((~ct).sum())}")
print(f"\n                      | REAL cumple | REAL no-cumple")
print(f"  PRED cumple         |   TP={TP:5d}  |   FP={FP:4d}  (INSEGURO)")
print(f"  PRED no-cumple      |   FN={FN:5d}  |   TN={TN:4d}  (conservador FN)")
print(f"\nExactitud global (accuracy) : {100*(TP+TN)/n:.2f}%")
print(f"FP (inseguro) : {FP}  = {100*FP/max(int((~ct).sum()),1):.2f}% de los no-cumple reales escapan como 'cumple'")
print(f"FN (conserv.) : {FN}  = {100*FN/max(int(ct.sum()),1):.2f}% de los cumple reales marcados 'no-cumple'")
print(f"Recall no-cumple (deteccion de incumplimiento) : {100*TN/max(TN+FP,1):.2f}%")
print(f"Precision cumple : {100*TP/max(TP+FP,1):.2f}%")

# ==== Banda marginal ====
fpm=(cp&~ct); fnm=(~cp&ct)
band=LIM*1.05  # +5% del limite = 0.00210
print("\n--- Banda marginal de los errores ---")
if fpm.sum():
    v=rm[fpm]
    print(f"FP: deriva real  mediana={np.median(v):.5f}  P90={np.percentile(v,90):.5f}  max={v.max():.5f}")
    print(f"    FP con deriva real <= 0.00210 (+5% del limite): {int((v<=band).sum())}/{len(v)} = {100*(v<=band).mean():.1f}%")
if fnm.sum():
    v=rm[fnm]
    print(f"FN: deriva real  mediana={np.median(v):.5f}  P10={np.percentile(v,10):.5f}  min={v.min():.5f}")
    print(f"    FN con deriva real >= 0.00190 (-5% del limite): {int((v>=LIM*0.95).sum())}/{len(v)} = {100*(v>=LIM*0.95).mean():.1f}%")

# Cuantos veredictos INSEGUROS quedarian si se aplicara un margen de seguridad del 5%
# (clasificar 'cumple' solo si deriva predicha <= 0.95*LIM)
cp_safe = (pred_dx<=0.95*LIM) & (pred_dy<=0.95*LIM)
cp_safe_sel = cp_safe[sel]
FP_safe=int((cp_safe_sel & ~ct).sum())
print(f"\nSi se exige margen de seguridad 5% (pred <= 0.0019): FP inseguro baja de {FP} a {FP_safe}")
