# -*- coding: utf-8 -*-
"""PREVIEW de la correccion del tope (NO escribe HDF5).
Implementa el procedimiento modal espectral NCh433:2026 CORRECTO:
  - Espectro Sa(T)=I*S*A0*alpha/R*  SIN tope por ordenada (alpha canonica T0,p).
  - Desplazamientos: desde Sa sin tope; escalado por Cmin si Vb<Vmin (6.3.6.1).
  - Vb: CQC sin capar, luego topar el CORTE COMBINADO en [Vmin, Vmax] (6.3.6).
Compara contra el GT actual (capeado por ordenada) y contra el corte estatico."""
import h5py, numpy as np
G=9.80665; I=1.0; Ro=11.0; R_walls=7.0; ZETA=0.05; LIM=0.002
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}
def rho_mat(om):
    om=np.minimum(om,1e6); b=om[None,:]/np.maximum(om[:,None],1e-12)
    num=8*ZETA**2*(1+b)*b**1.5; den=(1-b**2)**2+4*ZETA**2*b*(1+b)**2
    R=np.where(den>0,num/den,0.0); R[om<=0,:]=0;R[:,om<=0]=0; return R

def respuesta_correcta(N,h,Lx,Ly,s,z,fm,T,om,mpx,mpy,Phix,Phiy):
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    Tsx=T[np.argmax(mpx)]; Tsy=T[np.argmax(mpy)]
    Cmin=S*A0/6.0; Cmax=0.35*S*A0
    P=fm.sum()*G                              # peso sismico [N]
    Vmin=Cmin*I*P; Vmax=Cmax*I*P
    def sa_unc(Ts):                           # SIN tope (alpha canonica)
        x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
        return (S*A0*a*I/R)*G
    rho=rho_mat(om)
    out={}
    for lbl,Ts,mp,Phi,arm_perp in [('x',Tsx,mpx,Phix,Ly),('y',Tsy,mpy,Phiy,Lx)]:
        Sa=sa_unc(Ts); Sd=Sa/np.maximum(om**2,1e-6)
        Mn=(fm[:,None]*(Phix**2+Phiy**2)).sum(0); Mn=np.maximum(Mn,1e-30)
        Gam=(fm[:,None]*Phi).sum(0)/Mn
        Mmod=Phi*(Gam*Sd)[None,:]
        U=np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',Mmod,rho,Mmod),0))
        Vb_unc=np.sqrt(max((mp*fm.sum()*Sa)@rho@(mp*fm.sum()*Sa),0))
        # Vb: topar corte combinado
        Vb=min(max(Vb_unc,Vmin),Vmax)
        # desplazamientos: solo escalar por Cmin si Vb_unc<Vmin
        f_disp=max(1.0,Vmin/max(Vb_unc,1e-9))
        U=U*f_disp
        out['U'+lbl]=U; out['Vb'+lbl]=Vb/1000.0; out['Vbunc'+lbl]=Vb_unc/1000.0
    r_g=np.sqrt((Lx**2+Ly**2)/12.0)
    Uxt=np.abs(out['Ux'])*(1+0.05*Ly/r_g); Uyt=np.abs(out['Uy'])*(1+0.05*Lx/r_g)
    dx=np.empty(N); dy=np.empty(N); dx[0]=Uxt[0]/h; dy[0]=Uyt[0]/h
    dx[1:]=np.abs(np.diff(Uxt))/h; dy[1:]=np.abs(np.diff(Uyt))/h
    return out['Vbx'],out['Vby'],max(dx.max(),dy.max())

f=h5py.File('dataset_familyB_OPENSEES_v6_CORREGIDO.h5','r')
feat=[t.decode() for t in f['inputs/input_features'][:]]; X=f['inputs/X'][:]
N=f['scalars/N_pisos'][:]; fmA=f['inputs/floor_masses'][:]
T=f['modal/T_r'][:]; om=f['modal/omega_r'][:]
mpx=f['modal/mass_part_x'][:]; mpy=f['modal/mass_part_y'][:]
Phix=f['modal/Phi_x'][:]; Phiy=f['modal/Phi_y'][:]
VbxGT=f['response/Vb_x_kN'][:]; VbyGT=f['response/Vb_y_kN'][:]
dxs=f['response/drift_x'][:]; dys=f['response/drift_y'][:]
dGT=np.array([max(dxs[i,:N[i]].max(),dys[i,:N[i]].max()) for i in range(len(N))])
cumpleGT=f['scalars/cumple'][:]
ih=feat.index('h_story_m')
def geom(i):
    nu=int(X[i,feat.index('n_unid_lado')])
    return ((4*nu+4)*X[i,feat.index('L_mod_m')]+X[i,feat.index('L_nucleo_m')],
            2*X[i,feat.index('prof_depto_m')]+X[i,feat.index('ancho_corredor_m')])
Vbx_c=np.zeros(len(N)); dC=np.zeros(len(N))
for i in range(len(N)):
    nn=int(N[i]); s=smap[int(X[i,feat.index('suelo')])]; z=int(X[i,feat.index('zona')])
    Lx,Ly=geom(i); fm=fmA[i][:nn]
    vbx,vby,dmax=respuesta_correcta(nn,X[i,ih],Lx,Ly,s,z,fm,T[i],om[i],mpx[i],mpy[i],
                                    Phix[i][:nn,:],Phiy[i][:nn,:])
    Vbx_c[i]=vbx; dC[i]=dmax
f.close()
dVb=100*(Vbx_c-VbxGT)/np.maximum(VbxGT,1e-9)
dD=100*(dC-dGT)/np.maximum(dGT,1e-9)
print("="*60); print("IMPACTO DE LA CORRECCION (preview, sin escribir HDF5)")
print("="*60)
print(f"Vb_x:    cambio mediana={np.median(dVb):+.1f}%  P10={np.percentile(dVb,10):+.1f}%  P90={np.percentile(dVb,90):+.1f}%")
print(f"deriva:  cambio mediana={np.median(dD):+.1f}%  P10={np.percentile(dD,10):+.1f}%  P90={np.percentile(dD,90):+.1f}%")
# cumple
viol_c=dC>LIM
print(f"\ncumple: GT actual no-cumple={int((cumpleGT==0).sum())}  |  corregido no-cumple(solo deriva)={int(viol_c.sum())}")
print(f"  casos que pasan de CUMPLE a NO cumple: {int(((cumpleGT==1)&viol_c).sum())}")
