# -*- coding: utf-8 -*-
"""Genera dataset_familyB_OPENSEES_v6_CORREGIDO.h5 con la respuesta espectral
recalculada usando la alpha CANONICA NCh433 (T0,p) y R* con Ro (no Ro-1).
Solo cambia el grupo response/ ; modal/stiffness/inputs quedan identicos.
Vectorizado. Reporta impacto por caso en _impacto_alpha.csv."""
import h5py, numpy as np, shutil, time
SRC="dataset_familyB_OPENSEES_v6_20260523_1226.h5"
DST="dataset_familyB_OPENSEES_v6_CORREGIDO.h5"
G=9.80665; I=1.0; Ro=11.0; ZETA=0.05
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}

def rho_mat(om):
    b=om[None,:]/np.maximum(om[:,None],1e-12)
    num=8*ZETA**2*(1+b)*b**1.5
    den=(1-b**2)**2+4*ZETA**2*b*(1+b)**2
    R=np.where(den>0,num/den,0.0)
    R[om<=0,:]=0.0; R[:,om<=0]=0.0
    return R

def respuesta(N,h,Lx,Ly,s,z,fm,T,om,mpx,mpy,Phix,Phiy,corregir):
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    Tsx=T[np.argmax(mpx)]; Tsy=T[np.argmax(mpy)]
    smin=S*A0*I/6; smax=0.35*S*A0
    def sa(Ts):
        if corregir: x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
        else:        x=T/Tp; a=(1+4.5*x**n)/(1+x**3); R=1+Ts/(0.10*T0+Ts/(Ro-1))
        return np.clip(S*A0*a*I/R,smin,smax)*G
    Sa_x=sa(Tsx); Sa_y=sa(Tsy)
    Sd_x=Sa_x/np.maximum(om**2,1e-6); Sd_y=Sa_y/np.maximum(om**2,1e-6)
    rho=rho_mat(om)
    Mn=(fm[:,None]*(Phix**2+Phiy**2)).sum(0); Mn=np.maximum(Mn,1e-30)
    Gx=(fm[:,None]*Phix).sum(0)/Mn; Gy=(fm[:,None]*Phiy).sum(0)/Mn
    Mx=Phix*(Gx*Sd_x)[None,:]; My=Phiy*(Gy*Sd_y)[None,:]   # (N,K)
    Ux=np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',Mx,rho,Mx),0))
    Uy=np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',My,rho,My),0))
    Mtot=fm.sum()
    vx=mpx*Mtot*Sa_x; vy=mpy*Mtot*Sa_y
    Vbx=np.sqrt(max(vx@rho@vx,0))/1000.0; Vby=np.sqrt(max(vy@rho@vy,0))/1000.0
    r_g=np.sqrt((Lx**2+Ly**2)/12.0)
    Uxt=np.abs(Ux)*(1+0.05*Ly/r_g); Uyt=np.abs(Uy)*(1+0.05*Lx/r_g)
    dx=np.empty(N); dy=np.empty(N); dx[0]=Uxt[0]/h; dy[0]=Uyt[0]/h
    dx[1:]=np.abs(np.diff(Uxt))/h; dy[1:]=np.abs(np.diff(Uyt))/h
    return Vbx,Vby,Uxt,Uyt,dx,dy

f=h5py.File(SRC,'r')
feat=[t.decode() for t in f['inputs/input_features'][:]]; X=f['inputs/X'][:]
N=f['scalars/N_pisos'][:]; fmA=f['inputs/floor_masses'][:]
T=f['modal/T_r'][:]; om=f['modal/omega_r'][:]
mpx=f['modal/mass_part_x'][:]; mpy=f['modal/mass_part_y'][:]
Phix=f['modal/Phi_x'][:]; Phiy=f['modal/Phi_y'][:]
Vbx0=f['response/Vb_x_kN'][:]; Vby0=f['response/Vb_y_kN'][:]
dxm0=f['response/drift_x'][:]; dym0=f['response/drift_y'][:]
nc=len(N); ih=feat.index('h_story_m')
def geom(i):
    nu=int(X[i,feat.index('n_unid_lado')])
    return ((4*nu+4)*X[i,feat.index('L_mod_m')]+X[i,feat.index('L_nucleo_m')],
            2*X[i,feat.index('prof_depto_m')]+X[i,feat.index('ancho_corredor_m')])

Vbx_c=np.zeros(nc); Vby_c=np.zeros(nc)
Ux_c=np.zeros_like(f['response/Ux_m'][:]); Uy_c=np.zeros_like(Ux_c)
dx_c=np.zeros_like(dxm0); dy_c=np.zeros_like(dym0)
t0=time.time()
for i in range(nc):
    nn=int(N[i]); s=smap[int(X[i,feat.index('suelo')])]; z=int(X[i,feat.index('zona')])
    Lx,Ly=geom(i); fm=fmA[i][:nn]
    Vbx,Vby,Uxt,Uyt,dx,dy=respuesta(nn,X[i,ih],Lx,Ly,s,z,fm,T[i],om[i],mpx[i],mpy[i],
                                    Phix[i][:nn,:],Phiy[i][:nn,:],corregir=True)
    Vbx_c[i]=Vbx; Vby_c[i]=Vby
    Ux_c[i,:nn]=Uxt; Uy_c[i,:nn]=Uyt; dx_c[i,:nn]=dx; dy_c[i,:nn]=dy
    if i%2000==0: print(f"  {i}/{nc}  ({time.time()-t0:.0f}s)")
f.close()

# escribir HDF5 corregido
shutil.copy(SRC,DST)
g=h5py.File(DST,'r+')
g['response/Vb_x_kN'][...]=Vbx_c; g['response/Vb_y_kN'][...]=Vby_c
g['response/Ux_m'][...]=Ux_c; g['response/Uy_m'][...]=Uy_c
g['response/drift_x'][...]=dx_c; g['response/drift_y'][...]=dy_c
g['response/drift_max_x'][...] if 'drift_max_x' in g['response'] else None
g.attrs['alpha_fix']='alpha canonica NCh433 (T0,p) + R* con Ro - corregido post-v6'
g.close()

# reporte
dV=100*(Vbx_c-Vbx0)/np.maximum(Vbx0,1e-9)
dVy=100*(Vby_c-Vby0)/np.maximum(Vby0,1e-9)
dD=100*(dx_c.max(1)-dxm0.max(1))/np.maximum(dxm0.max(1),1e-9)
import csv
with open('_impacto_alpha.csv','w',newline='') as fo:
    w=csv.writer(fo); w.writerow(['case_id','N_pisos','dVb_x_%','dVb_y_%','dDrift_x_%'])
    for i in range(nc): w.writerow([i,int(N[i]),round(dV[i],2),round(dVy[i],2),round(dD[i],2)])
alld=np.concatenate([np.abs(dV),np.abs(dVy)])
print(f"\nGuardado: {DST}  +  _impacto_alpha.csv")
print(f"Impacto |dVb| (x e y): mediana={np.median(alld):.1f}%  P90={np.percentile(alld,90):.1f}%  P99={np.percentile(alld,99):.1f}%  max={alld.max():.1f}%")
print(f"  casos |dVb|<2%: {100*(alld<2).mean():.0f}%   >10%: {100*(alld>10).mean():.1f}%   >25%: {100*(alld>25).mean():.1f}%")
print(f"Impacto |dDrift_x|: mediana={np.median(np.abs(dD)):.1f}%  P90={np.percentile(np.abs(dD),90):.1f}%  max={np.abs(dD).max():.1f}%")
