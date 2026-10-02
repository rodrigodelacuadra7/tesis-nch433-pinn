# -*- coding: utf-8 -*-
"""Sobre los casos al borde del limite de deriva, calcula la torsion accidental
RIGUROSA (CM +-5%, re-eig acoplado) y verifica si algun veredicto cumple->no cumple
cambiaria respecto a la heuristica e_acc/r_giro almacenada."""
import h5py, numpy as np
G=9.80665; I=1.0; Ro=11.0; ZETA=0.05; LIM=0.002
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}

def gen_eigh(Ks,Ms):
    L=np.linalg.cholesky(Ms); Li=np.linalg.inv(L)
    A=Li@Ks@Li.T; A=0.5*(A+A.T)
    w,Psi=np.linalg.eigh(A); return np.maximum(w,1e-9),Li.T@Psi

def cqc_rho(om):
    b=om[None,:]/np.maximum(om[:,None],1e-12)
    num=8*ZETA**2*(1+b)*b**1.5; den=(1-b**2)**2+4*ZETA**2*b*(1+b)**2
    R=np.where(den>0,num/den,0.0); R[om<=0,:]=0; R[:,om<=0]=0; return R

def sa_spectrum(T,Ts,s,z):
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
    return np.clip(S*A0*a*I/R,S*A0*I/6,0.35*S*A0)*G

def corner_drift(K,M,N,recel,s,z,arm,gdir,h):
    Mc=M.copy()
    for j in range(N):
        m=M[3*j,3*j]
        if gdir==1:
            Mc[3*j+1,3*j+2]+=m*recel; Mc[3*j+2,3*j+1]+=m*recel; Mc[3*j+2,3*j+2]+=m*recel**2
        else:
            Mc[3*j,3*j+2]+=-m*recel; Mc[3*j+2,3*j]+=-m*recel; Mc[3*j+2,3*j+2]+=m*recel**2
    Ks=K[:3*N,:3*N]; Ms=Mc[:3*N,:3*N]
    w2,Phi=gen_eigh(Ks,Ms); om=np.sqrt(w2)
    iota=np.zeros(3*N); iota[gdir::3]=1.0
    Mphi=Ms@Phi; Mn=np.einsum('ir,ir->r',Phi,Mphi)
    Gam=(Phi.T@(Ms@iota))/np.maximum(Mn,1e-30)
    part=Gam**2*Mn/M[gdir::3,gdir::3].diagonal().sum()
    Ts=2*np.pi/om[np.argmax(part)]
    Sd=sa_spectrum(2*np.pi/om,Ts,s,z)/np.maximum(om**2,1e-9)
    rho=cqc_rho(om); C=Phi*(Gam*Sd)[None,:]
    edge=C[gdir::3,:]+C[2::3,:]*arm
    u=np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',edge,rho,edge),0))
    dd=np.empty(N); dd[0]=u[0]/h; dd[1:]=np.abs(np.diff(u))/h
    return dd.max()

def rig_drift_case(f,feat,i):
    X=f['inputs/X'][i]; d=dict(zip(feat,X)); N=int(f['scalars/N_pisos'][i])
    s=smap[int(d['suelo'])]; z=int(d['zona']); h=float(d['h_story_m'])
    nu=int(d['n_unid_lado']); Lx=(4*nu+4)*d['L_mod_m']+d['L_nucleo_m']; Ly=2*d['prof_depto_m']+d['ancho_corredor_m']
    K=f['stiffness/K_global'][i].astype(float); M=f['stiffness/M_global'][i].astype(float)
    best=0.0
    for gdir,Lperp,arm in [(0,Ly,Ly/2),(1,Lx,Lx/2)]:
        e=0.05*Lperp
        for sgn in (+1,-1):
            best=max(best,corner_drift(K,M,N,sgn*e,s,z,arm,gdir,h))
    return best

f=h5py.File('dataset_familyB_OPENSEES_v6_20260523_1226.h5','r')
feat=[t.decode() for t in f['inputs/input_features'][:]]
N=f['scalars/N_pisos'][:]; cumple=f['scalars/cumple'][:]
dx=f['response/drift_x'][:]; dy=f['response/drift_y'][:]
dmx=np.array([dx[i,:N[i]].max() for i in range(len(N))])
dmy=np.array([dy[i,:N[i]].max() for i in range(len(N))])
dmax_heur=np.maximum(dmx,dmy)

# verificar equivalencia cumple <-> deriva
viol=dmax_heur>LIM
print('cumple==0 coincide con deriva>limite en %d/10000 casos (%.1f%%)'%((viol==(cumple==0)).sum(),100*(viol==(cumple==0)).mean()))

# candidatos a flip: hoy cumplen y deriva entre 80% y 100% del limite
cand=np.nonzero((cumple==1)&(dmax_heur>0.80*LIM))[0]
print('candidatos a flip (cumplen, deriva>80%% lim): %d'%len(cand))
flips=[]; ratios=[]
for k,i in enumerate(cand):
    drig=rig_drift_case(f,feat,i)
    ratios.append(drig/dmax_heur[i])
    if drig>LIM: flips.append((int(i),dmax_heur[i],drig))
    if k%50==0: print('  procesados %d/%d'%(k,len(cand)),flush=True)
ratios=np.array(ratios)
print('\nRESULTADO sobre %d candidatos:'%len(cand))
print('  deriva RIG/HEUR: mediana=%.2f  P90=%.2f  max=%.2f'%(np.median(ratios),np.percentile(ratios,90),ratios.max()))
print('  casos que se DARIAN VUELTA (cumple->no cumple con torsion rigurosa): %d'%len(flips))
for i,dh,dr in flips[:30]:
    print('    caso %5d: deriva heur=%.5f -> rig=%.5f'%(i,dh,dr))
f.close()
