# -*- coding: utf-8 -*-
"""Evalua el modelo de produccion contra el GT VIEJO vs CORREGIDO (alpha).
Decide si vale la pena reentrenar: mira cuanto se degrada el MAPE de Vb/deriva."""
import numpy as np, pickle, h5py, torch, torch.nn as nn, torch.nn.functional as F
dev='cpu'
MODEL=r"C:/Users/rodri/Documents/PINN/b2_proyecto/models/trials_31/V8/DEFINITIVO/model_fase2_definitivo.pt"
SCAL =r"C:/Users/rodri/Documents/PINN/b2_proyecto/models/trials_31/V8/DEFINITIVO/scalers_trial16.pkl"
H5_OLD="dataset_familyB_OPENSEES_v6_20260523_1226.h5"
H5_NEW="dataset_familyB_OPENSEES_v6_CORREGIDO.h5"

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

f=h5py.File(H5_OLD,'r'); feat=[t.decode() for t in f['inputs/input_features'][:]]
X=f['inputs/X'][:]; N=f['scalars/N_pisos'][:].astype(int); maskf=f['inputs/mask_floors'][:].astype(np.float32)
Vbx_old=f['response/Vb_x_kN'][:]; Vby_old=f['response/Vb_y_kN'][:]
dxm_old=f['response/drift_x'][:].max(1); dym_old=f['response/drift_y'][:].max(1); f.close()
g=h5py.File(H5_NEW,'r'); Vbx_new=g['response/Vb_x_kN'][:]; Vby_new=g['response/Vb_y_kN'][:]
dxm_new=g['response/drift_x'][:].max(1); dym_new=g['response/drift_y'][:].max(1); g.close()

# construir X_clean (21 cols, one-hot suelo/zona) en orden COLS
import pandas as pd
df=pd.DataFrame(X,columns=feat)
sm={0:'A',1:'B',2:'C',3:'D'}
soh=pd.get_dummies(df['suelo'].map(sm),prefix='suelo').astype('float32')
zoh=pd.get_dummies(df['zona'].astype(int),prefix='zona').astype('float32')
df=pd.concat([df.drop(columns=['suelo','zona']),soh,zoh],axis=1)
for c in COLS:
    if c not in df.columns: df[c]=0.0
Xc=df[COLS].to_numpy('float32')
Xn=(Xc-SC['X_mean'])/SC['X_std']

# forward batched
with torch.no_grad():
    Xt=torch.from_numpy(Xn).to(dev); mk=torch.from_numpy(maskf).to(dev)
    out=m(Xt,mk)
    Vb=out[8].cpu().numpy(); dx=out[6].cpu().numpy(); dy=out[7].cpu().numpy()
Vbx_p=np.exp(Vb[:,0]*SC['Vbx_std']+SC['Vbx_mean'])
Vby_p=np.exp(Vb[:,1]*SC['Vby_std']+SC['Vby_mean'])
dxm_p=np.array([np.exp(dx[i,:N[i]]*SC['Dx_std']+SC['Dx_mean']).max() for i in range(len(N))])
dym_p=np.array([np.exp(dy[i,:N[i]]*SC['Dy_std']+SC['Dy_mean']).max() for i in range(len(N))])

def mape(pred,gt): return 100*np.mean(np.abs((pred-gt)/gt))
# subset alto impacto alpha (|dVb_x|>10%)
hi=np.abs(100*(Vbx_new-Vbx_old)/Vbx_old)>10
print("="*64)
print("MAPE del modelo de produccion (10.000 casos)")
print("="*64)
print(f"{'metrica':<12}{'vs GT VIEJO':>14}{'vs CORREGIDO':>14}{'delta':>10}")
for nm,p,go,gn in [('Vb_x',Vbx_p,Vbx_old,Vbx_new),('Vb_y',Vby_p,Vby_old,Vby_new),
                   ('drift_x',dxm_p,dxm_old,dxm_new),('drift_y',dym_p,dym_old,dym_new)]:
    a=mape(p,go); b=mape(p,gn); print(f"{nm:<12}{a:>13.2f}%{b:>13.2f}%{b-a:>+9.2f}")
print(f"\nSubset alto impacto alpha (|dVb_x|>10%, {hi.sum()} casos):")
for nm,p,go,gn in [('Vb_x',Vbx_p,Vbx_old,Vbx_new),('drift_x',dxm_p,dxm_old,dxm_new)]:
    print(f"  {nm:<10} vs viejo={mape(p[hi],go[hi]):.1f}%   vs corregido={mape(p[hi],gn[hi]):.1f}%   delta={mape(p[hi],gn[hi])-mape(p[hi],go[hi]):+.1f}")
