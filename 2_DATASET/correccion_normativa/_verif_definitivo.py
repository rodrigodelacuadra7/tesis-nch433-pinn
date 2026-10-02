# -*- coding: utf-8 -*-
"""Verificacion EXHAUSTIVA del dataset DEFINITIVO."""
import h5py, numpy as np
G=9.80665; I=1.0; Ro=11.0; R_walls=7.0; ZETA=0.05; LIM=0.002
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}
def rho_mat(om):
    om=np.minimum(om,1e6); b=om[None,:]/np.maximum(om[:,None],1e-12)
    num=8*ZETA**2*(1+b)*b**1.5; den=(1-b**2)**2+4*ZETA**2*b*(1+b)**2
    R=np.where(den>0,num/den,0.0); R[om<=0,:]=0;R[:,om<=0]=0; return R
OLD="dataset_familyB_OPENSEES_v6_CORREGIDO.h5"
NEW="dataset_familyB_OPENSEES_v6_DEFINITIVO.h5"
o=h5py.File(OLD,'r'); d=h5py.File(NEW,'r')
feat=[t.decode() for t in d['inputs/input_features'][:]]; X=d['inputs/X'][:]
N=d['scalars/N_pisos'][:]; fmA=d['inputs/floor_masses'][:]
ok=True
print("="*64); print("VERIFICACION DEFINITIVO"); print("="*64)

# (1) modal/stiffness/inputs IDENTICOS
print("\n[1] Modal/stiffness/inputs intactos (solo response debe cambiar):")
for k in ['inputs/X','modal/T_r','modal/omega_r','modal/Phi_x','modal/Phi_y',
          'modal/mass_part_x','stiffness/K_global','stiffness/M_global']:
    same=np.array_equal(o[k][:],d[k][:]); ok&=same
    print(f"    {k:28s} identico={same}")

# (2) sin NaN/inf/negativos en response
print("\n[2] Sanidad numerica de response/:")
for k in ['response/Vb_x_kN','response/Vb_y_kN','response/drift_x','response/drift_y','response/Ux_m','response/Uy_m']:
    a=d[k][:]; bad=(~np.isfinite(a)).sum(); neg=(a<0).sum()
    ok&=(bad==0 and neg==0); print(f"    {k:24s} NaN/inf={bad}  negativos={neg}")

# (3) cumple == drift_ok
dx=d['response/drift_x'][:]; dy=d['response/drift_y'][:]
dmax=np.array([max(dx[i,:N[i]].max(),dy[i,:N[i]].max()) for i in range(len(N))])
cumple=d['scalars/cumple'][:]
consist=((dmax<=LIM)==(cumple==1)).all(); ok&=consist
print(f"\n[3] cumple == (deriva<=0.002) en todos: {consist}")

# (4) Vb vs corte ESTATICO NCh433 (la prueba de fondo)
rcorr=[]
for i in range(len(N)):
    nn=int(N[i]); s=smap[int(X[i,feat.index('suelo')])]; z=int(X[i,feat.index('zona')])
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    fm=fmA[i][:nn]; Tsx=d['modal/T_r'][i][np.argmax(d['modal/mass_part_x'][i])]
    Cmin=S*A0/6; Cmax=0.35*S*A0; P=fm.sum()*G
    Cst=np.clip(2.75*S*A0/R_walls*(Tp/Tsx)**n,Cmin,Cmax); Q0=Cst*I*P/1000
    rcorr.append(d['response/Vb_x_kN'][i]/max(Q0,1e-9))
rcorr=np.array(rcorr)
inrange=0.85<np.median(rcorr)<1.05; ok&=inrange
print(f"\n[4] Vb_DEFINITIVO / Q0_estatico:  mediana={np.median(rcorr):.2f}  P10={np.percentile(rcorr,10):.2f}  P90={np.percentile(rcorr,90):.2f}")
print(f"    NCh433 espera ~0.9-1.0 -> {'OK' if inrange else 'FUERA DE RANGO'}")

# (5) reproduccion: mi formula reproduce el DEFINITIVO? (deteccion de bug propio)
def rho2(om): return rho_mat(om)
errV=[]; errD=[]
np.random.seed(7); muestra=np.random.choice(len(N),40,replace=False)
for i in muestra:
    nn=int(N[i]); s=smap[int(X[i,feat.index('suelo')])]; z=int(X[i,feat.index('zona')])
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    fm=fmA[i][:nn]; Ti=d['modal/T_r'][i]; omi=np.where(d['modal/omega_r'][i]>1e-6,d['modal/omega_r'][i],1e12)
    mpx=d['modal/mass_part_x'][i]; Phx=d['modal/Phi_x'][i][:nn,:]; Phy=d['modal/Phi_y'][i][:nn,:]
    rho=rho2(omi); Tsx=Ti[np.argmax(mpx)]; Mtot=fm.sum()
    Cmin=S*A0/6; Cmax=0.35*S*A0; P=Mtot*G
    x=Ti/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Tsx/(0.10*T0+Tsx/Ro); Sa=(S*A0*a*I/R)*G
    vu=mpx*Mtot*Sa; Vbunc=np.sqrt(max(vu@rho@vu,0)); Vb=min(max(Vbunc,Cmin*I*P),Cmax*I*P)/1000
    errV.append(abs(Vb-d['response/Vb_x_kN'][i])/max(d['response/Vb_x_kN'][i],1e-9))
print(f"\n[5] Mi formula reproduce el Vb del DEFINITIVO: error max={100*max(errV):.3f}%")
ok&=(max(errV)<0.01)

# (6) cambio vs CORREGIDO
dVb=100*(d['response/Vb_x_kN'][:]-o['response/Vb_x_kN'][:])/np.maximum(o['response/Vb_x_kN'][:],1e-9)
print(f"\n[6] Cambio vs CORREGIDO:  Vb mediana={np.median(dVb):+.1f}%")
print(f"    cumple: {int((o['scalars/cumple'][:]==0).sum())} -> {int((cumple==0).sum())} no-cumple")
print(f"    deriva max global: {dmax.max():.5f}  (mediana {np.median(dmax):.5f})")

print("\n"+"="*64)
print("RESULTADO GLOBAL:", "TODO OK - DATASET DEFINITIVO VALIDO" if ok else "HAY FALLAS - REVISAR")
print("="*64)
o.close(); d.close()
