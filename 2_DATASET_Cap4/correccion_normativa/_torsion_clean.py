# -*- coding: utf-8 -*-
"""Aisla el ERROR de torsion (metodo simplificado vs riguroso) con la MISMA
alpha canonica en ambos lados, para no contaminar con el bug de alpha.
Compara, por caso:
  - heuristico: deriva con CM centrado, desplazamiento*(1+e/r)
  - riguroso  : deriva en esquina con CM +-5% (eig acoplado)
Ambos con alpha canonica (T0,p). Reporta cociente riguroso/heuristico."""
import h5py, numpy as np
G=9.80665; I=1.0; Ro=11.0; ZETA=0.05; LIM=0.002
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}

def gen_eigh(Ks,Ms):
    L=np.linalg.cholesky(Ms); Li=np.linalg.inv(L)
    A=Li@Ks@Li.T; A=0.5*(A+A.T); w,Psi=np.linalg.eigh(A)
    return np.maximum(w,1e-9),Li.T@Psi

def cqc_rho(om):
    b=om[None,:]/np.maximum(om[:,None],1e-12)
    num=8*ZETA**2*(1+b)*b**1.5; den=(1-b**2)**2+4*ZETA**2*b*(1+b)**2
    R=np.where(den>0,num/den,0.0); R[om<=0,:]=0; R[:,om<=0]=0; return R

def sa_spectrum(T,Ts,s,z):
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
    return np.clip(S*A0*a*I/R,S*A0*I/6,0.35*S*A0)*G

def solve(K,M,N,ecc,s,z,gdir,arm,h):
    """Devuelve (drift_cm_trasl, drift_corner). ecc=0 -> sin acoplar."""
    Mc=M.copy()
    if ecc!=0.0:
        for j in range(N):
            m=M[3*j,3*j]
            if gdir==1: Mc[3*j+1,3*j+2]+=m*ecc; Mc[3*j+2,3*j+1]+=m*ecc; Mc[3*j+2,3*j+2]+=m*ecc**2
            else:       Mc[3*j,3*j+2]+=-m*ecc;  Mc[3*j+2,3*j]+=-m*ecc;  Mc[3*j+2,3*j+2]+=m*ecc**2
    Ks,Ms=K[:3*N,:3*N],Mc[:3*N,:3*N]
    w2,Phi=gen_eigh(Ks,Ms); om=np.sqrt(w2)
    iota=np.zeros(3*N); iota[gdir::3]=1.0
    Mphi=Ms@Phi; Mn=np.einsum('ir,ir->r',Phi,Mphi)
    Gam=(Phi.T@(Ms@iota))/np.maximum(Mn,1e-30)
    part=Gam**2*Mn/M[gdir::3,gdir::3].diagonal().sum()
    Ts=2*np.pi/om[np.argmax(part)]
    Sd=sa_spectrum(2*np.pi/om,Ts,s,z)/np.maximum(om**2,1e-9)
    rho=cqc_rho(om); C=Phi*(Gam*Sd)[None,:]
    trans=C[gdir::3,:]; theta=C[2::3,:]; edge=trans+theta*arm
    def cqc(modal): return np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',modal,rho,modal),0))
    def drift(u):
        dd=np.empty(N); dd[0]=u[0]/h; dd[1:]=np.abs(np.diff(u))/h; return dd.max()
    return cqc(trans),cqc(edge)

f=h5py.File('dataset_familyB_OPENSEES_v6_20260523_1226.h5','r')
feat=[t.decode() for t in f['inputs/input_features'][:]]
Nall=f['scalars/N_pisos'][:]; cumple=f['scalars/cumple'][:]
dx,dy=f['response/drift_x'][:],f['response/drift_y'][:]
dmax_stored=np.array([max(dx[i,:Nall[i]].max(),dy[i,:Nall[i]].max()) for i in range(len(Nall))])
cand=np.nonzero((cumple==1)&(dmax_stored>0.80*LIM))[0]

ratios=[]; dh_all=[]; dr_all=[]; flips=0
for i in cand:
    d=dict(zip(feat,f['inputs/X'][i])); N=int(Nall[i])
    s=smap[int(d['suelo'])]; z=int(d['zona']); h=float(d['h_story_m'])
    nu=int(d['n_unid_lado']); Lx=(4*nu+4)*d['L_mod_m']+d['L_nucleo_m']; Ly=2*d['prof_depto_m']+d['ancho_corredor_m']
    rg=np.sqrt((Lx**2+Ly**2)/12)
    K=f['stiffness/K_global'][i].astype(float); M=f['stiffness/M_global'][i].astype(float)
    dheur=0.0; drig=0.0
    for gdir,Lperp,arm in [(0,Ly,Ly/2),(1,Lx,Lx/2)]:
        e=0.05*Lperp
        # heuristico: solve centrado (ecc=0), amplificar desplazamiento traslacional por (1+e/rg)
        u_cm,_=solve(K,M,N,0.0,s,z,gdir,arm,h)
        u_h=u_cm*(1+e/rg)
        dd=np.empty(N); dd[0]=u_h[0]/h; dd[1:]=np.abs(np.diff(u_h))/h
        dheur=max(dheur,dd.max())
        # riguroso: esquina con CM +-ecc
        for sgn in (+1,-1):
            _,u_edge=solve(K,M,N,sgn*e,s,z,gdir,arm,h)
            de=np.empty(N); de[0]=u_edge[0]/h; de[1:]=np.abs(np.diff(u_edge))/h
            drig=max(drig,de.max())
    ratios.append(drig/dheur); dh_all.append(dheur); dr_all.append(drig)
    if drig>LIM and dheur<=LIM: flips+=1
f.close()
ratios=np.array(ratios)
print("=== AISLADO: misma alpha canonica en ambos lados (solo difiere la torsion) ===")
print("Casos analizados: %d"%len(cand))
print("cociente RIGUROSO/SIMPLIFICADO  mediana=%.3f  P10=%.3f  P90=%.3f  max=%.3f"
      %(np.median(ratios),np.percentile(ratios,10),np.percentile(ratios,90),ratios.max()))
print("error |1-cociente|              mediana=%.1f%%  P90=%.1f%%  max=%.1f%%"
      %(100*np.median(np.abs(1-ratios)),100*np.percentile(np.abs(1-ratios),90),100*np.abs(1-ratios).max()))
print("flips (heur<=lim<rig) con alpha canonica consistente: %d"%flips)
