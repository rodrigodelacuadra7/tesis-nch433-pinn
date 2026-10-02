# -*- coding: utf-8 -*-
"""C4: el codigo capa Sa por ordenada en [smin,smax] y usa ese Sa para Vb Y
desplazamientos. NCh433:2026 6.3.6.2 dice que Cmax NO rige para desplazamientos.
Mide cuanto cambia la deriva (y el cumplimiento) si NO se capa el espectro de
desplazamientos en smax (lo correcto segun norma). Usa alpha canonica (corregida)."""
import h5py, numpy as np
G=9.80665; I=1.0; Ro=11.0; ZETA=0.05; LIM=0.002
SP={'A':(0.90,0.15,0.20,1.00,2.0),'B':(1.00,0.30,0.35,1.33,1.5),
    'C':(1.05,0.40,0.45,1.40,1.6),'D':(1.20,0.75,0.85,1.80,1.0)}
smap={0:'A',1:'B',2:'C',3:'D'}

def rho_mat(om):
    b=om[None,:]/np.maximum(om[:,None],1e-12)
    num=8*ZETA**2*(1+b)*b**1.5; den=(1-b**2)**2+4*ZETA**2*b*(1+b)**2
    R=np.where(den>0,num/den,0.0); R[om<=0,:]=0; R[:,om<=0]=0; return R

def drift_case(N,h,Lx,Ly,s,z,fm,T,om,mpx,mpy,Phix,Phiy,cap_smax):
    S,T0,Tp,n,p=SP[s]; A0={1:0.20,2:0.30,3:0.40}[z]
    Tsx=T[np.argmax(mpx)]; Tsy=T[np.argmax(mpy)]
    smin=S*A0*I/6; smax=0.35*S*A0
    def sa(Ts):
        x=T/T0; a=(1+4.5*x**p)/(1+x**3); R=1+Ts/(0.10*T0+Ts/Ro)
        val=S*A0*a*I/R
        if cap_smax: val=np.clip(val,smin,smax)
        else:        val=np.maximum(val,smin)   # NCh433: piso si, techo no, para desplaz.
        return val*G
    Sa_x=sa(Tsx); Sa_y=sa(Tsy)
    Sd_x=Sa_x/np.maximum(om**2,1e-6); Sd_y=Sa_y/np.maximum(om**2,1e-6)
    rho=rho_mat(om)
    Mn=(fm[:,None]*(Phix**2+Phiy**2)).sum(0); Mn=np.maximum(Mn,1e-30)
    Gx=(fm[:,None]*Phix).sum(0)/Mn; Gy=(fm[:,None]*Phiy).sum(0)/Mn
    Mx=Phix*(Gx*Sd_x)[None,:]; My=Phiy*(Gy*Sd_y)[None,:]
    Ux=np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',Mx,rho,Mx),0))
    Uy=np.sqrt(np.maximum(np.einsum('pi,ij,pj->p',My,rho,My),0))
    r_g=np.sqrt((Lx**2+Ly**2)/12.0)
    Uxt=np.abs(Ux)*(1+0.05*Ly/r_g); Uyt=np.abs(Uy)*(1+0.05*Lx/r_g)
    dx=np.empty(N); dy=np.empty(N); dx[0]=Uxt[0]/h; dy[0]=Uyt[0]/h
    dx[1:]=np.abs(np.diff(Uxt))/h; dy[1:]=np.abs(np.diff(Uyt))/h
    return max(dx.max(),dy.max())

f=h5py.File('dataset_familyB_OPENSEES_v6_CORREGIDO.h5','r')
feat=[t.decode() for t in f['inputs/input_features'][:]]; X=f['inputs/X'][:]
N=f['scalars/N_pisos'][:]; fmA=f['inputs/floor_masses'][:]
T=f['modal/T_r'][:]; om=f['modal/omega_r'][:]
mpx=f['modal/mass_part_x'][:]; mpy=f['modal/mass_part_y'][:]
Phix=f['modal/Phi_x'][:]; Phiy=f['modal/Phi_y'][:]
ih=feat.index('h_story_m')
def geom(i):
    nu=int(X[i,feat.index('n_unid_lado')])
    return ((4*nu+4)*X[i,feat.index('L_mod_m')]+X[i,feat.index('L_nucleo_m')],
            2*X[i,feat.index('prof_depto_m')]+X[i,feat.index('ancho_corredor_m')])

dxs=f['response/drift_x'][:]; dys=f['response/drift_y'][:]
dstored=np.array([max(dxs[i,:N[i]].max(),dys[i,:N[i]].max()) for i in range(len(N))])
d_cap=np.zeros(len(N)); d_nocap=np.zeros(len(N))
for i in range(len(N)):
    nn=int(N[i]); s=smap[int(X[i,feat.index('suelo')])]; z=int(X[i,feat.index('zona')])
    Lx,Ly=geom(i); fm=fmA[i][:nn]
    a=(nn,X[i,ih],Lx,Ly,s,z,fm,T[i],om[i],mpx[i],mpy[i],Phix[i][:nn,:],Phiy[i][:nn,:])
    d_cap[i]=drift_case(*a,cap_smax=True)
    d_nocap[i]=drift_case(*a,cap_smax=False)
f.close()
# VERIF: la version vectorizada (cap=True) reproduce el HDF5?
ev=np.abs(d_cap-dstored)/np.maximum(dstored,1e-9)
print(f"VERIF vectorizada vs HDF5: error mediana={100*np.median(ev):.3f}%  max={100*ev.max():.3f}%  -> {'OK' if ev.max()<0.01 else 'REVISAR'}\n")

dd=100*(d_nocap-d_cap)/np.maximum(d_cap,1e-9)
binds=dd>0.5   # casos donde smax topaba (deriva sube al quitar el tope)
print("=== C4: impacto de quitar el tope smax en el espectro de DESPLAZAMIENTOS ===")
print(f"Casos donde smax afecta la deriva (>0.5%): {binds.sum()} ({100*binds.mean():.1f}%)")
print(f"Aumento de deriva al corregir: mediana={np.median(dd[binds]):.1f}%  P90={np.percentile(dd[binds],90):.1f}%  max={dd[binds].max():.1f}%" if binds.sum() else "  (ninguno)")
# cumple: pasa de cumple a no-cumple?
cumple_cap=d_cap<=LIM; cumple_nocap=d_nocap<=LIM
flips=int((cumple_cap&~cumple_nocap).sum())
print(f"Veredictos que cambiarian cumple->NO cumple al corregir C4: {flips} de {len(N)} ({100*flips/len(N):.2f}%)")
ids=np.nonzero(cumple_cap&~cumple_nocap)[0]
for i in ids[:15]: print(f"   caso {i}: deriva {d_cap[i]:.5f} -> {d_nocap[i]:.5f}")
