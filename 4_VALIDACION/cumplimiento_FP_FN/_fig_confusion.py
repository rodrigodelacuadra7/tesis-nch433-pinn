# -*- coding: utf-8 -*-
"""Figura 2 paneles: matriz de confusion del cumplimiento (Trial 16 OOD) + banda marginal."""
import numpy as np, pickle, h5py, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
dev='cpu'; LIM=0.002
MODEL=r'C:/Users/rodri/Documents/PINN/b2_proyecto/models/trials_31/V8/DEFINITIVO_CORREGIDO/model_fase2_definitivo.pt'
SCAL =r'C:/Users/rodri/Documents/NB/3_METAMODELO/modelo/scalers_trial16_DEFINITIVO.pkl'
H5   ='C:/Users/rodri/Documents/NB/2_DATASET/datasets_h5/dataset_familyB_OPENSEES_v6_DEFINITIVO.h5'
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
        resp=s.head_resp(h).view(B,s.n_max_pisos,4)
        Ux=resp[...,0]*mask; Uy=resp[...,1]*mask; ratio=F.softplus(resp[...,2])*mask; dy=resp[...,3]*mask; dx=ratio*dy
        return None,None,None,None,Ux,Uy,dx,dy,None,ratio
m=PINNModal_v4().to(dev); m.load_state_dict(torch.load(MODEL,map_location=dev)); m.eval()
SC=pickle.load(open(SCAL,'rb')); COLS=SC['COLS_BASE']
f=h5py.File(H5,'r'); feat=[t.decode() for t in f['inputs/input_features'][:]]
X=f['inputs/X'][:]; N=f['scalars/N_pisos'][:].astype(int); maskf=f['inputs/mask_floors'][:].astype(np.float32)
dxm=f['response/drift_x'][:]; dym=f['response/drift_y'][:]; f.close()
tdx=np.array([dxm[i,:N[i]].max() for i in range(len(N))]); tdy=np.array([dym[i,:N[i]].max() for i in range(len(N))])
import pandas as pd
df=pd.DataFrame(X,columns=feat); sm={0:'A',1:'B',2:'C',3:'D'}
soh=pd.get_dummies(df['suelo'].map(sm),prefix='suelo').astype('float32'); zoh=pd.get_dummies(df['zona'].astype(int),prefix='zona').astype('float32')
df=pd.concat([df.drop(columns=['suelo','zona']),soh,zoh],axis=1)
for c in COLS:
    if c not in df.columns: df[c]=0.0
Xn=((df[COLS].to_numpy('float32'))-SC['X_mean'])/SC['X_std']
with torch.no_grad():
    out=m(torch.from_numpy(Xn),torch.from_numpy(maskf)); dx=out[6].numpy(); dy=out[7].numpy()
pdx=np.array([np.exp(dx[i,:N[i]]*SC['Dx_std']+SC['Dx_mean']).max() for i in range(len(N))])
pdy=np.array([np.exp(dy[i,:N[i]]*SC['Dy_std']+SC['Dy_mean']).max() for i in range(len(N))])
ct=(tdx<=LIM)&(tdy<=LIM); cp=(pdx<=LIM)&(pdy<=LIM); rm=np.maximum(tdx,tdy)
zona=X[:,feat.index('zona')].astype(int); sel=(zona==2)
ct,cp,rm=ct[sel],cp[sel],rm[sel]
TP=int((cp&ct).sum());TN=int((~cp&~ct).sum());FP=int((cp&~ct).sum());FN=int((~cp&ct).sum());n=len(ct)
fpv=rm[cp&~ct]; fnv=rm[~cp&ct]
plt.rcParams.update({'font.size':11})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5.0))
disp=np.array([[TP,FP],[FN,TN]],float)
ax1.imshow(disp,cmap='Blues',vmax=disp.max()*1.15)
cm=[[TP,FP],[FN,TN]]; labs=[['TP','FP'],['FN','TN']]; sub=[['(correcto)','INSEGURO'],['(conservador)','(correcto)']]
for i in range(2):
    for j in range(2):
        c='white' if disp[i,j]>disp.max()*0.5 else 'black'
        ax1.text(j,i-0.13,'%s = %d'%(labs[i][j],cm[i][j]),ha='center',va='center',fontsize=15,fontweight='bold',color=c)
        ax1.text(j,i+0.17,sub[i][j],ha='center',va='center',fontsize=9,color=c)
ax1.add_patch(Rectangle((0.5,-0.5),1,1,fill=False,edgecolor='#c0392b',lw=3))
ax1.set_xticks([0,1]); ax1.set_xticklabels(['Cumple','No cumple']); ax1.set_yticks([0,1]); ax1.set_yticklabels(['Cumple','No cumple'])
ax1.set_xlabel('Veredicto REAL (OpenSees)',fontweight='bold'); ax1.set_ylabel('Veredicto PREDICHO',fontweight='bold')
ax1.set_title('Matriz de confusión del cumplimiento (deriva ≤ 0,002·h)\nOOD Trial 16 (zona 2, n=%d)  |  Exactitud = %.1f%%'%(n,100*(TP+TN)/n),fontsize=11)
ax1.set_xticks(np.arange(-.5,2,1),minor=True); ax1.set_yticks(np.arange(-.5,2,1),minor=True)
ax1.grid(which='minor',color='w',lw=2); ax1.tick_params(which='minor',length=0)
ax2.axvspan(LIM*0.95,LIM*1.05,color='0.85',alpha=.7,label='banda ±5% del límite')
ax2.axvline(LIM,color='k',ls='--',lw=1.6,label='límite NCh433 (0,002)')
bins=np.linspace(min(fnv.min(),0.00160),max(fpv.max(),0.00235),26)
ax2.hist(fpv,bins=bins,color='#c0392b',alpha=.8,label='FP inseguro (n=%d)'%len(fpv))
ax2.hist(fnv,bins=bins,color='#2471a3',alpha=.7,label='FN conservador (n=%d)'%len(fnv))
ax2.axvline(np.median(fpv),color='#c0392b',ls=':',lw=2); ax2.axvline(np.median(fnv),color='#2471a3',ls=':',lw=2)
ax2.set_xlabel('Deriva real (ground truth) [-]'); ax2.set_ylabel('Número de casos')
ax2.set_title('Errores de clasificación concentrados en la banda marginal\nFP mediana=%.5f  |  FN mediana=%.5f'%(np.median(fpv),np.median(fnv)),fontsize=11)
ax2.legend(fontsize=9,loc='upper right'); ax2.set_xlim(bins[0],bins[-1])
plt.tight_layout(); plt.savefig('fig_confusion_trial16.png',dpi=200,bbox_inches='tight')
print('OK FP=%d FN=%d TP=%d TN=%d n=%d'%(FP,FN,TP,TN,n))
