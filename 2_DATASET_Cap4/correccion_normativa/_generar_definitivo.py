# -*- coding: utf-8 -*-
"""Genera dataset_familyB_OPENSEES_v6_DEFINITIVO.h5.
Parte del CORREGIDO (que ya tiene alpha canonica T0,p + R* con Ro) y recalcula
SOLO response/ con el procedimiento modal espectral NCh433:2026 CORRECTO:
  - Espectro Sa(T)=I*S*A0*alpha/R*  SIN tope por ordenada.
  - Desplazamientos desde Sa sin tope; escalado por Cmin si Vb_unc<Vmin (6.3.6.1).
  - Vb: CQC sin capar, luego cap del CORTE COMBINADO en [Vmin,Vmax] (6.3.6.1/2).
  - Cmax NO toca desplazamientos (6.3.6.2).
Modal/stiffness/inputs quedan IDENTICOS. Vectorizado."""
import h5py, numpy as np, shutil, time
SRC="dataset_familyB_OPENSEES_v6_CORREGIDO.h5"
DST="dataset_familyB_OPENSEES_v6_DEFINITIVO.h5"
G=9.80665; I=1.0; Ro=11.0; ZETA=0.05; LIM=0.002
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}

def rho_mat(om):
    om=np.minimum(om,1e6); b=om[None,:]/np.maximum(om[:,None],1e-12)
    num=8*ZETA**2*(1+b)*b**1.5; den=(1-b**2)**2+4*ZETA**2*b*(1+b)**2
    R=np.where(den>0,num/den,0.0); R[om<=0,:]=0; R[:,om<=0]=0; return R

def respuesta(N,h,Lx,Ly,s,z,fm,T,om,mpx,mpy,Phix,Phiy):
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    Tsx=T[np.argmax(mpx)]; Tsy=T[np.argmax(mpy)]
    Cmin=S*A0/6.0; Cmax=0.35*S*A0; P=fm.sum()*G
    Vmin=Cmin*I*P; Vmax=Cmax*I*P
    rho=rho_mat(om); Mtot=fm.sum()
    Mn=(fm[:,None]*(Phix**2+Phiy**2)).sum(0); Mn=np.maximum(Mn,1e-30)
    def dir_resp(Ts,mp,Phi):
        x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
        Sa=(S*A0*a*I/R)*G                          # SIN tope
        Sd=Sa/np.maximum(om**2,1e-6)
        Gam=(fm[:,None]*Phi).sum(0)/Mn
        Mmod=Phi*(Gam*Sd)[None,:]
        U=np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',Mmod,rho,Mmod),0))
        vu=mp*Mtot*Sa; Vb_unc=np.sqrt(max(vu@rho@vu,0))
        Vb=min(max(Vb_unc,Vmin),Vmax)              # cap del corte COMBINADO
        U=U*max(1.0,Vmin/max(Vb_unc,1e-9))         # Cmin escala desplaz; Cmax NO
        return U,Vb
    Ux,Vbx=dir_resp(Tsx,mpx,Phix)
    Uy,Vby=dir_resp(Tsy,mpy,Phiy)
    r_g=np.sqrt((Lx**2+Ly**2)/12.0)
    fxx=0.05*Ly/r_g; fyy=0.05*Lx/r_g
    dUx=np.abs(Ux)*fxx; dUy=np.abs(Uy)*fyy
    Uxt=np.abs(Ux)+dUx; Uyt=np.abs(Uy)+dUy
    dx=np.empty(N); dy=np.empty(N); dx[0]=Uxt[0]/h; dy[0]=Uyt[0]/h
    dx[1:]=np.abs(np.diff(Uxt))/h; dy[1:]=np.abs(np.diff(Uyt))/h
    return Vbx/1000.0,Vby/1000.0,Uxt,Uyt,dUx,dUy,dx,dy

f=h5py.File(SRC,'r')
feat=[t.decode() for t in f['inputs/input_features'][:]]; X=f['inputs/X'][:]
N=f['scalars/N_pisos'][:]; fmA=f['inputs/floor_masses'][:]
T=f['modal/T_r'][:]; om=f['modal/omega_r'][:]
mpx=f['modal/mass_part_x'][:]; mpy=f['modal/mass_part_y'][:]
Phix=f['modal/Phi_x'][:]; Phiy=f['modal/Phi_y'][:]
nc=len(N); ih=feat.index('h_story_m')
def geom(i):
    nu=int(X[i,feat.index('n_unid_lado')])
    return ((4*nu+4)*X[i,feat.index('L_mod_m')]+X[i,feat.index('L_nucleo_m')],
            2*X[i,feat.index('prof_depto_m')]+X[i,feat.index('ancho_corredor_m')])

P=18  # ancho pad
Vbx_c=np.zeros(nc); Vby_c=np.zeros(nc)
Ux_c=np.zeros((nc,P)); Uy_c=np.zeros((nc,P)); dUx_c=np.zeros((nc,P)); dUy_c=np.zeros((nc,P))
dx_c=np.zeros((nc,P)); dy_c=np.zeros((nc,P)); cumple_c=np.zeros(nc,dtype='int8')
t0=time.time()
for i in range(nc):
    nn=int(N[i]); s=smap[int(X[i,feat.index('suelo')])]; z=int(X[i,feat.index('zona')])
    Lx,Ly=geom(i); fm=fmA[i][:nn]
    vbx,vby,Uxt,Uyt,dUx,dUy,dx,dy=respuesta(nn,X[i,ih],Lx,Ly,s,z,fm,T[i],om[i],
                                            mpx[i],mpy[i],Phix[i][:nn,:],Phiy[i][:nn,:])
    Vbx_c[i]=vbx; Vby_c[i]=vby
    Ux_c[i,:nn]=Uxt; Uy_c[i,:nn]=Uyt; dUx_c[i,:nn]=dUx; dUy_c[i,:nn]=dUy
    dx_c[i,:nn]=dx; dy_c[i,:nn]=dy
    cumple_c[i]=1 if (dx.max()<=LIM and dy.max()<=LIM) else 0
    if i%2000==0: print(f"  {i}/{nc} ({time.time()-t0:.0f}s)")
f.close()

shutil.copy(SRC,DST)
g=h5py.File(DST,'r+')
g['response/Vb_x_kN'][...]=Vbx_c; g['response/Vb_y_kN'][...]=Vby_c
g['response/Ux_m'][...]=Ux_c; g['response/Uy_m'][...]=Uy_c
g['response/drift_x'][...]=dx_c; g['response/drift_y'][...]=dy_c
g['response/delta_Ux_tacc'][...]=dUx_c; g['response/delta_Uy_tacc'][...]=dUy_c
g['scalars/cumple'][...]=cumple_c
g.attrs['cap_fix']='espectro sin tope por ordenada; Cmax sobre corte combinado; Cmax no toca desplazamientos (NCh433:2026 6.3.6). Sobre _CORREGIDO (alpha T0,p + R* Ro).'
g.close()
print(f"\nGuardado: {DST}")
print(f"cumple=1: {int((cumple_c==1).sum())}  cumple=0: {int((cumple_c==0).sum())}")
