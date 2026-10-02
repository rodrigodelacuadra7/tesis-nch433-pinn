# -*- coding: utf-8 -*-
"""VERIFICACION EXHAUSTIVA C4:
(1) que mi reconstruccion (cap_smax=True) reproduce EXACTO la deriva del HDF5
    corregido de produccion. Si calza -> el contrafactual (cap_smax=False) es valido.
(2) cuantifica de nuevo el impacto con la version validada."""
import h5py, numpy as np
G=9.80665; I=1.0; Ro=11.0; ZETA=0.05; LIM=0.002
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}

def cqc_combine(vals,oms):
    vals=np.asarray(vals,float); s=0.0; K=len(vals)
    for i in range(K):
        wi=oms[i]
        for j in range(K):
            wj=oms[j]
            if wi<=0 or wj<=0: continue
            b=wj/wi; r=8*ZETA**2*(1+b)*b**1.5/((1-b**2)**2+4*ZETA**2*b*(1+b)**2)
            s+=r*vals[i]*vals[j]
    return np.sqrt(max(s,0.0))

def respuesta(N,h,Lx,Ly,s,z,fm,T,om,mpx,mpy,Phix,Phiy,Phith,cap_smax):
    # replica EXACTA de compute_spectral_response con alpha canonica (corregida)
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    Tsx=T[np.argmax(mpx)]; Tsy=T[np.argmax(mpy)]
    smin=S*A0*I/6; smax=0.35*S*A0
    def sa(Ts):
        x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
        val=S*A0*a*I/R
        if cap_smax: val=np.maximum(smin,np.minimum(smax,val))
        else:        val=np.maximum(smin,val)
        return val*G
    Sa_x=sa(Tsx); Sa_y=sa(Tsy)
    Sd_x=Sa_x/np.maximum(om**2,1e-6); Sd_y=Sa_y/np.maximum(om**2,1e-6)
    K=len(T); Ux=np.zeros(N); Uy=np.zeros(N)
    for piso in range(1,N+1):
        Mx=[];My=[]
        for r in range(K):
            Mn=sum(fm[q]*(Phix[q,r]**2+Phiy[q,r]**2) for q in range(N)); Mn=max(Mn,1e-30)
            Gx=sum(fm[q]*Phix[q,r] for q in range(N))/Mn
            Gy=sum(fm[q]*Phiy[q,r] for q in range(N))/Mn
            Mx.append(Gx*Phix[piso-1,r]*Sd_x[r]); My.append(Gy*Phiy[piso-1,r]*Sd_y[r])
        Ux[piso-1]=cqc_combine(Mx,om); Uy[piso-1]=cqc_combine(My,om)
    r_g=np.sqrt((Lx**2+Ly**2)/12.0)
    Uxt=np.abs(Ux)*(1+0.05*Ly/r_g); Uyt=np.abs(Uy)*(1+0.05*Lx/r_g)
    dx=np.zeros(N); dy=np.zeros(N); dx[0]=Uxt[0]/h; dy[0]=Uyt[0]/h
    for i in range(1,N):
        dx[i]=abs(Uxt[i]-Uxt[i-1])/h; dy[i]=abs(Uyt[i]-Uyt[i-1])/h
    return max(dx.max(),dy.max())

f=h5py.File('dataset_familyB_OPENSEES_v6_CORREGIDO.h5','r')
feat=[t.decode() for t in f['inputs/input_features'][:]]; X=f['inputs/X'][:]
N=f['scalars/N_pisos'][:]; fmA=f['inputs/floor_masses'][:]
T=f['modal/T_r'][:]; om=f['modal/omega_r'][:]
mpx=f['modal/mass_part_x'][:]; mpy=f['modal/mass_part_y'][:]
Phix=f['modal/Phi_x'][:]; Phiy=f['modal/Phi_y'][:]; Phith=f['modal/Phi_theta'][:]
dxs=f['response/drift_x'][:]; dys=f['response/drift_y'][:]
dstored=np.array([max(dxs[i,:N[i]].max(),dys[i,:N[i]].max()) for i in range(len(N))])
ih=feat.index('h_story_m')
def geom(i):
    nu=int(X[i,feat.index('n_unid_lado')])
    return ((4*nu+4)*X[i,feat.index('L_mod_m')]+X[i,feat.index('L_nucleo_m')],
            2*X[i,feat.index('prof_depto_m')]+X[i,feat.index('ancho_corredor_m')])

# VERIFICACION (1): reproduccion sobre muestra
np.random.seed(0); muestra=np.random.choice(len(N),60,replace=False)
err=[]
for i in muestra:
    nn=int(N[i]); s=smap[int(X[i,feat.index('suelo')])]; z=int(X[i,feat.index('zona')])
    Lx,Ly=geom(i); fm=fmA[i][:nn]
    d=respuesta(nn,X[i,ih],Lx,Ly,s,z,fm,T[i],om[i],mpx[i],mpy[i],
                Phix[i][:nn,:],Phiy[i][:nn,:],Phith[i][:nn,:],cap_smax=True)
    err.append(abs(d-dstored[i])/max(dstored[i],1e-9))
err=np.array(err)
print("=== VERIF (1): mi reconstruccion (cap=True) vs HDF5 corregido ===")
print(f"  error relativo: mediana={100*np.median(err):.3f}%  max={100*err.max():.3f}%  (60 casos)")
print(f"  -> {'CALZA: reconstruccion valida' if err.max()<0.01 else 'NO CALZA - revisar'}")
f.close()
