# -*- coding: utf-8 -*-
"""Recalcula la respuesta espectral (Vb, U, deriva) con la alpha CANONICA NCh433
(T0, p, Ro) sobre la data modal ya guardada en el HDF5 v6. No re-corre OpenSees.
Replica compute_spectral_response (incl. torsion accidental e_acc/r_giro)."""
import h5py, numpy as np
G=9.80665; I=1.0; Ro=11.0; ZETA=0.05; DRIFT_LIMIT=0.002
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}

def cqc_rho(wi,wj,z=ZETA):
    if wi<=0 or wj<=0: return 0.0
    b=wj/wi
    return 8*z**2*(1+b)*b**1.5/((1-b**2)**2+4*z**2*b*(1+b)**2)
def cqc(vals,oms):
    vals=np.asarray(vals); s=0.0; K=len(vals)
    for i in range(K):
        for j in range(K): s+=cqc_rho(oms[i],oms[j])*vals[i]*vals[j]
    return np.sqrt(max(s,0.0))

def respuesta(N,h,Lx,Ly,s,z,fm,T,om,mpx,mpy,Phix,Phiy,corregir):
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    Tsx=T[np.argmax(mpx)]; Tsy=T[np.argmax(mpy)]
    def sa(Ts):
        if corregir: x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
        else:        x=T/Tp; a=(1+4.5*x**n)/(1+x**3); R=1+Ts/(0.10*T0+Ts/(Ro-1))
        smin=S*A0*I/6; smax=0.35*S*A0
        return np.clip(S*A0*a*I/R,smin,smax)*G
    Sa_x=sa(Tsx); Sa_y=sa(Tsy)
    Sd_x=Sa_x/np.maximum(om**2,1e-6); Sd_y=Sa_y/np.maximum(om**2,1e-6)
    K=len(T); Ux=np.zeros(N); Uy=np.zeros(N)
    for pp in range(1,N+1):
        Mx=[];My=[]
        for r in range(K):
            Mn=sum(fm[q]*(Phix[q,r]**2+Phiy[q,r]**2) for q in range(N)); Mn=max(Mn,1e-30)
            Gx=sum(fm[q]*Phix[q,r] for q in range(N))/Mn
            Gy=sum(fm[q]*Phiy[q,r] for q in range(N))/Mn
            Mx.append(Gx*Phix[pp-1,r]*Sd_x[r]); My.append(Gy*Phiy[pp-1,r]*Sd_y[r])
        Ux[pp-1]=cqc(Mx,om); Uy[pp-1]=cqc(My,om)
    Mtot=fm.sum()
    Vbx=cqc(mpx*Mtot*Sa_x,om)/1000.0; Vby=cqc(mpy*Mtot*Sa_y,om)/1000.0
    r_g=np.sqrt((Lx**2+Ly**2)/12.0); fx=0.05*Ly/r_g; fy=0.05*Lx/r_g
    Uxt=np.abs(Ux)*(1+fx); Uyt=np.abs(Uy)*(1+fy)
    dx=np.zeros(N); dy=np.zeros(N); dx[0]=Uxt[0]/h; dy[0]=Uyt[0]/h
    for i in range(1,N):
        dx[i]=abs(Uxt[i]-Uxt[i-1])/h; dy[i]=abs(Uyt[i]-Uyt[i-1])/h
    return dict(Vbx=Vbx,Vby=Vby,Ux=Uxt,Uy=Uyt,dx=dx,dy=dy)

def geom(i,feat,X):
    nu=int(X[i,feat.index('n_unid_lado')])
    Lx=(4*nu+4)*X[i,feat.index('L_mod_m')]+X[i,feat.index('L_nucleo_m')]
    Ly=2*X[i,feat.index('prof_depto_m')]+X[i,feat.index('ancho_corredor_m')]
    return Lx,Ly

if __name__=='__main__':
    f=h5py.File('dataset_familyB_OPENSEES_v6_20260523_1226.h5','r')
    feat=[t.decode() for t in f['inputs/input_features'][:]]; X=f['inputs/X'][:]
    N=f['scalars/N_pisos'][:]; fmA=f['inputs/floor_masses'][:]
    T=f['modal/T_r'][:]; om=f['modal/omega_r'][:]
    mpx=f['modal/mass_part_x'][:]; mpy=f['modal/mass_part_y'][:]
    Phix=f['modal/Phi_x'][:]; Phiy=f['modal/Phi_y'][:]
    Vbx0=f['response/Vb_x_kN'][:]; Ux0=f['response/Ux_m'][:]; dx0=f['response/drift_x'][:]
    print(f"{'caso':>5} | {'Vbx_rep':>8} {'Vbx_h5':>8} | {'Uroof_rp':>8} {'Uroof_h5':>8} | {'dmax_rp':>8} {'dmax_h5':>8}")
    for i in [2100,3034,5821,6478]:
        nn=int(N[i]); s=smap[int(X[i,feat.index('suelo')])]; z=int(X[i,feat.index('zona')])
        Lx,Ly=geom(i,feat,X); fm=fmA[i][:nn]
        r=respuesta(nn,X[i,feat.index('h_story_m')],Lx,Ly,s,z,fm,T[i],om[i],mpx[i],mpy[i],
                    Phix[i][:nn,:],Phiy[i][:nn,:],corregir=False)
        print(f"{i:>5} | {r['Vbx']:>8.1f} {Vbx0[i]:>8.1f} | {r['Ux'][-1]:>8.4f} {Ux0[i][nn-1]:>8.4f} | {r['dx'].max():>8.5f} {dx0[i][:nn].max():>8.5f}")
    f.close()
