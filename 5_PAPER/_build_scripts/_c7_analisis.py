# -*- coding: utf-8 -*-
"""C7: el 'T1' guardado (slot 1) vs el periodo fundamental REAL (omega mas bajo).
En edificios torsionalmente acoplados el modo de periodo mas largo se clasifica
como torsional, dejando en slot1 un modo superior. Cuantifica el problema sobre
el dataset corregido (modal identico al original)."""
import h5py, numpy as np
H5="dataset_familyB_OPENSEES_v6_CORREGIDO.h5"
f=h5py.File(H5,'r')
feat=[t.decode() for t in f['inputs/input_features'][:]]
X=f['inputs/X'][:]; N=f['scalars/N_pisos'][:].astype(int)
Tr=f['modal/T_r'][:]                      # (10000,18) periodos en slots clasificados
opad=f['stiffness/omega_pad'][:]          # (10000,54) frecuencias crudas ascendentes
mtypes=f['modal/mode_types'][:]           # (10000,18) tipo por slot
b2=X[:,feat.index('activar_B2')]
# T1 guardado = slot 0
T1_stored=Tr[:,0]
# fundamental real = periodo mas largo = 2pi/omega_min (primer omega valido >0)
omega_min=np.where(opad>1e-6,opad,np.inf).min(1)
T1_true=2*np.pi/omega_min
ratio=T1_stored/T1_true
# tipo del slot 0
slot0=np.array([mtypes[i,0].decode() for i in range(len(N))])

print("=== C7 sobre dataset CORREGIDO ===")
print("slot0 tipos:",{t:int((slot0==t).sum()) for t in set(slot0)})
print()
for thr,lbl in [(0.99,'T1_stored < 99% del real'),(0.90,'<90%'),(0.70,'<70%'),(0.50,'<50% (SEVERO)')]:
    m=ratio<thr
    print(f"{lbl:28s}: {m.sum():5d} casos ({100*m.mean():.1f}%)   B2=0 entre ellos: {int((m&(b2==0)).sum())}/{int(m.sum())}")
print()
sev=ratio<0.50
print(f"Casos SEVEROS (ratio<0.5): {sev.sum()}")
ids=np.nonzero(sev)[0]
print("ids severos:",ids[:40].tolist())
print()
print("Ejemplos severos (caso: T1_guardado -> T1_real, slot0_tipo, N, B2):")
for i in ids[:12]:
    print(f"  {i:5d}: {T1_stored[i]:.4f} -> {T1_true[i]:.4f}  ({slot0[i]}, N={N[i]}, B2={int(b2[i])})")
print()
# impacto: en estos casos, cuanto vale el T1_MAPE si lo midieras contra el real?
err_vs_stored=np.abs(T1_stored-T1_stored)  # 0 por def
err_real=100*np.abs(T1_stored-T1_true)/T1_true
print(f"Error del 'T1' guardado vs fundamental real (solo casos C7 ratio<0.9): mediana={np.median(err_real[ratio<0.9]):.1f}%  max={err_real[ratio<0.9].max():.1f}%")
print(f"Sobre TODO el dataset, el T1 guardado coincide con el fundamental real en {100*(ratio>=0.99).mean():.1f}% de casos")
f.close()
