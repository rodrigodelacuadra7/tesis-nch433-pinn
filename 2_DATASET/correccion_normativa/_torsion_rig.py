# -*- coding: utf-8 -*-
"""Compara torsion accidental HEURISTICA (e_acc/r_giro) vs RIGUROSA (CM desplazado
+-5%, re-eig acoplado, espectral+CQC, desplazamiento en la ESQUINA real).
Usa K_global, M_global (54x54) ya guardadas. No re-corre OpenSees."""
import h5py, numpy as np
G=9.80665; I=1.0; Ro=11.0; ZETA=0.05

def gen_eigh(Ks,Ms):
    """Eig generalizado K psi = w^2 M psi via Cholesky (M SPD). numpy.eigh robusto."""
    L=np.linalg.cholesky(Ms); Li=np.linalg.inv(L)
    A=Li@Ks@Li.T; A=0.5*(A+A.T)
    w,Psi=np.linalg.eigh(A)
    Phi=Li.T@Psi
    return np.maximum(w,1e-9),Phi
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}

def cqc_rho(om):
    b=om[None,:]/np.maximum(om[:,None],1e-12)
    num=8*ZETA**2*(1+b)*b**1.5; den=(1-b**2)**2+4*ZETA**2*b*(1+b)**2
    R=np.where(den>0,num/den,0.0); R[om<=0,:]=0; R[:,om<=0]=0; return R

def sa_spectrum(T,Ts,s,z):
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
    smin=S*A0*I/6; smax=0.35*S*A0
    return np.clip(S*A0*a*I/R,smin,smax)*G

def solve_dir(K,M,N, recel, s,z, arm_edge, gdir):
    Mc=M.copy()
    for j in range(N):
        m=M[3*j,3*j]
        if gdir==1:   # Y load -> offset ex
            ex=recel
            Mc[3*j+1,3*j+2]+=m*ex; Mc[3*j+2,3*j+1]+=m*ex; Mc[3*j+2,3*j+2]+=m*ex**2
        else:         # X load -> offset ey
            ey=recel
            Mc[3*j,3*j+2]+=-m*ey; Mc[3*j+2,3*j]+=-m*ey; Mc[3*j+2,3*j+2]+=m*ey**2
    Ks=K[:3*N,:3*N]; Ms=Mc[:3*N,:3*N]
    w2,Phi=gen_eigh(Ks,Ms)
    om=np.sqrt(w2)
    iota=np.zeros(3*N); iota[gdir::3]=1.0
    Mphi=Ms@Phi
    Mn=np.einsum('ir,ir->r',Phi,Mphi)
    Gam=(Phi.T@(Ms@iota))/np.maximum(Mn,1e-30)
    Mdir_tot=M[gdir::3,gdir::3].diagonal().sum()
    part=Gam**2*Mn/Mdir_tot
    Ts=2*np.pi/om[np.argmax(part)]
    Sa=sa_spectrum(2*np.pi/om,Ts,s,z); Sd=Sa/np.maximum(om**2,1e-9)
    rho=cqc_rho(om)
    C=Phi*(Gam*Sd)[None,:]
    trans=C[gdir::3,:]
    theta=C[2::3,:]
    edge=trans+theta*arm_edge
    def cqc(modal):
        return np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',modal,rho,modal),0))
    return cqc(trans),cqc(edge),om,Ts

f=h5py.File('dataset_familyB_OPENSEES_v6_20260523_1226.h5','r')
feat=[t.decode() for t in f['inputs/input_features'][:]]
for i in [2100,3034]:
    X=f['inputs/X'][i]; d=dict(zip(feat,X)); N=int(f['scalars/N_pisos'][i])
    s=smap[int(d['suelo'])]; z=int(d['zona']); h=d['h_story_m']
    nu=int(d['n_unid_lado']); Lx=(4*nu+4)*d['L_mod_m']+d['L_nucleo_m']; Ly=2*d['prof_depto_m']+d['ancho_corredor_m']
    rg=np.sqrt((Lx**2+Ly**2)/12)
    K=f['stiffness/K_global'][i].astype(float); M=f['stiffness/M_global'][i].astype(float)
    print('='*70); print('CASO %d: N=%d suelo=%s zona=%d Lx=%.1f Ly=%.1f h=%.2f r_giro=%.2f'%(i,N,s,z,Lx,Ly,h,rg))
    for gdir,lbl,Lperp,arm in [(0,'X (lado largo)',Ly,Ly/2),(1,'Y (lado corto)',Lx,Lx/2)]:
        e_acc=0.05*Lperp
        u_cm0,_,om0,Ts=solve_dir(K,M,N,0.0,s,z,arm,gdir)
        frac=e_acc/rg
        u_edge_heur=u_cm0*(1+frac)
        _,u_edgeA,_,_=solve_dir(K,M,N,+e_acc,s,z,arm,gdir)
        _,u_edgeB,_,_=solve_dir(K,M,N,-e_acc,s,z,arm,gdir)
        u_edge_rig=np.maximum(u_edgeA,u_edgeB)
        def drift(u):
            dd=np.empty(N); dd[0]=u[0]/h; dd[1:]=np.abs(np.diff(u))/h; return dd
        dr_cm=drift(u_cm0); dr_heur=drift(u_edge_heur); dr_rig=drift(u_edge_rig)
        add_heur=u_edge_heur[-1]-u_cm0[-1]; add_rig=u_edge_rig[-1]-u_cm0[-1]
        print('  --- Sismo %s: e_acc=%.2fm brazo_esquina=%.1fm frac_heur=%.3f T*=%.3fs'%(lbl,e_acc,arm,frac,Ts))
        print('      U_techo CM(trasl)   = %7.3f mm'%(u_cm0[-1]*1000))
        print('      U_techo borde HEUR  = %7.3f mm   (add torsion %6.3f mm)'%(u_edge_heur[-1]*1000,add_heur*1000))
        print('      U_techo esquina RIG = %7.3f mm   (add torsion %6.3f mm)'%(u_edge_rig[-1]*1000,add_rig*1000))
        print('      deriva_max CM=%.6f  HEUR=%.6f  RIG=%.6f'%(dr_cm.max(),dr_heur.max(),dr_rig.max()))
        if add_rig>1e-12:
            rel='HEUR sobreestima' if add_heur>=add_rig else 'HEUR SUBESTIMA'
            print('      -> add torsion HEUR/RIG = %.2fx  (%s)'%(add_heur/add_rig,rel))
f.close()
