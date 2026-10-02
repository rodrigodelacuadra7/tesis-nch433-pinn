# ============================================================================
#  MODELO OPENSEES - MUROS SHELL - CASO 3570 (Familia B, 10 pisos)
#  Geometria de muros EXACTA (wall_df del PKL de origen, misma fuente que el GT)
#  Metodologia identica al caso 2100: diafragma rigido + masa concentrada +
#  E reducido (eta=0.35). Validacion externa shell (equivalente a Robot).
# ============================================================================
import numpy as np
import openseespy.opensees as ops
import time

# ----------------------------------------------------------------------------
#  1. PARAMETROS DEL CASO 3570
# ----------------------------------------------------------------------------
N_PISOS   = 10
H_STORY   = 2.90                  # m
H_TOTAL   = N_PISOS * H_STORY     # 29.00 m

# Hormigon fc=40 MPa, E=4700*sqrt(fc), agrietamiento DS61 (eta=0.35)
E_BRUTO   = 2.9725e+10              # Pa  (= 4700*sqrt(40) MPa)
ETA       = 0.35
E_FIS     = ETA * E_BRUTO
NU        = 0.20
RHO_MAT   = 0.0

# Espesores por grupo (m)
T_NUCLEO  = 0.300   # B1
T_BORDE   = 0.250   # B2
T_MID     = 0.150   # B3, B4

# Masas e inercias rotacionales por piso (HDF5, exactas del GT OpenSees)
MASS = [990678.125]*9 + [919915.375]          # kg
IROT = [5.572460e+08]*9 + [5.174427e+08]           # kg*m2

# Centro de masa (centroide de la planta = Lx/2, Ly/2)
CX, CY = 40.159, 8.643            # m

MESH = 1.4
TOL  = 1e-3

# Periodos ground truth (OpenSees columna ancha, para comparar)
T_GT = {'T1_Y': 0.5118, 'T2_Tors': 0.3455, 'T3_X': 0.1863}

# ----------------------------------------------------------------------------
#  2. GEOMETRIA DE MUROS  (label, grupo, x_orig, y_orig, largo, espesor, dir)
#     dir='Y' -> corre en Y (espesor en X) ; dir='X' -> corre en X (espesor en Y)
# ----------------------------------------------------------------------------
def get_walls():
    w = []
    # B1 nucleo
    w.append(("B1-1",'B1',37.000,9.574,4.900,T_NUCLEO,'Y'))
    w.append(("B1-2",'B1',38.667,9.574,4.900,T_NUCLEO,'Y'))
    w.append(("B1-3",'B1',40.633,9.574,4.900,T_NUCLEO,'Y'))
    w.append(("B1-4",'B1',42.600,9.574,4.900,T_NUCLEO,'Y'))
    w.append(("B1-5",'B1',37.000,14.174,1.967,T_NUCLEO,'X'))
    w.append(("B1-6",'B1',38.967,14.174,1.967,T_NUCLEO,'X'))
    w.append(("B1-7",'B1',40.933,14.174,1.967,T_NUCLEO,'X'))
    # B2 fachadas extremas (Y)
    w.append(("B2-L",'B2',0.000,0.000,17.286,T_BORDE,'Y'))
    w.append(("B2-R",'B2',80.069,0.000,17.286,T_BORDE,'Y'))
    # B3 medianeros superiores (Y)
    B3T = [7.325, 14.725, 22.125, 29.525, 50.225, 57.625, 65.025, 72.425]
    for k,xo in enumerate(B3T):
        w.append((f"B3-T{k+1}",'B3',xo,9.480,7.805,T_MID,'Y'))
    # B3 medianeros inferiores (Y)
    B3B = [7.325, 14.725, 22.125, 29.525, 36.925, 42.825, 50.225, 57.625, 65.025, 72.425]
    for k,xo in enumerate(B3B):
        w.append((f"B3-B{k+1}",'B3',xo,0.000,7.830,T_MID,'Y'))
    # B4 pasillo (X)
    w.append(("B4-SUP-L",'B4',0.250,9.480,36.750,T_MID,'X'))
    w.append(("B4-SUP-R",'B4',42.900,9.480,37.169,T_MID,'X'))
    w.append(("B4-INF-L",'B4',0.250,7.680,36.750,T_MID,'X'))
    w.append(("B4-INF-R",'B4',42.900,7.680,37.169,T_MID,'X'))
    return w

# ----------------------------------------------------------------------------
#  Estaciones de malla: divisiones regulares (~mesh) MAS puntos de cruce
# ----------------------------------------------------------------------------
def line_stations(s0, s1, mesh, crossings):
    L = s1 - s0
    n = max(1, int(round(L/mesh)))
    pts = set(round(s0 + L*i/n, 4) for i in range(n+1))
    for c in crossings:
        if s0 - TOL < c < s1 + TOL:
            pts.add(round(c, 4))
    return sorted(pts)

# ----------------------------------------------------------------------------
#  3. CONSTRUCCION DEL MODELO
# ----------------------------------------------------------------------------
def build_model(mesh=MESH):
    ops.wipe()
    ops.model('basic','-ndm',3,'-ndf',6)
    ops.nDMaterial('ElasticIsotropic', 1, E_FIS, NU, RHO_MAT)

    walls = get_walls()
    sec_for_t = {}; st = 1
    for t in sorted(set(round(w[5],3) for w in walls)):
        ops.section('PlateFiber', st, 1, t)
        sec_for_t[t] = st; st += 1

    z_levels = [round(ip*H_STORY, 6) for ip in range(N_PISOS+1)]

    node_map = {}; nid = [0]
    def gn(x,y,z):
        k = (round(x,3), round(y,3), round(z,6))
        if k not in node_map:
            nid[0]+=1; ops.node(nid[0], x, y, z); node_map[k]=nid[0]
        return node_map[k]

    et = [0]
    def shell(a,b,c,d,t):
        et[0]+=1; ops.element('ShellMITC4', et[0], a,b,c,d, sec_for_t[t])

    walls_Y = [(xo, yo, yo+L) for lab,g,xo,yo,L,t,d in walls if d=='Y']
    walls_X = [(yo, xo, xo+L) for lab,g,xo,yo,L,t,d in walls if d=='X']

    for lab,g,xo,yo,L,t,d in walls:
        nZ = max(1, int(round(H_STORY/mesh)))
        if d=='Y':
            x = xo
            crossings = [wy for (wy,wx0,wx1) in walls_X if wx0-TOL < x < wx1+TOL]
            stations = line_stations(yo, yo+L, mesh, crossings)
            for ip in range(N_PISOS):
                zbot=z_levels[ip]; ztop=z_levels[ip+1]
                for iz in range(nZ):
                    za=round(zbot+(ztop-zbot)*iz/nZ,6)
                    zb=round(zbot+(ztop-zbot)*(iz+1)/nZ,6)
                    for il in range(len(stations)-1):
                        ya=stations[il]; yb=stations[il+1]
                        shell(gn(x,ya,za),gn(x,yb,za),gn(x,yb,zb),gn(x,ya,zb), round(t,3))
        else:
            y = yo
            crossings = [wx for (wx,wy0,wy1) in walls_Y if wy0-TOL < y < wy1+TOL]
            stations = line_stations(xo, xo+L, mesh, crossings)
            for ip in range(N_PISOS):
                zbot=z_levels[ip]; ztop=z_levels[ip+1]
                for iz in range(nZ):
                    za=round(zbot+(ztop-zbot)*iz/nZ,6)
                    zb=round(zbot+(ztop-zbot)*(iz+1)/nZ,6)
                    for il in range(len(stations)-1):
                        xa=stations[il]; xb=stations[il+1]
                        shell(gn(xa,y,za),gn(xb,y,za),gn(xb,y,zb),gn(xa,y,zb), round(t,3))

    nfix=0
    for k,n in node_map.items():
        if abs(k[2]-z_levels[0])<1e-9:
            ops.fix(n, 1,1,1,1,1,1); nfix+=1

    masters = []
    for ip in range(1, N_PISOS+1):
        z = z_levels[ip]; mt = 900000+ip
        ops.node(mt, CX, CY, z)
        ops.mass(mt, MASS[ip-1], MASS[ip-1], 0.0, 0.0, 0.0, IROT[ip-1])
        cn = [n for k,n in node_map.items() if abs(k[2]-z)<1e-6]
        if cn:
            ops.rigidDiaphragm(3, mt, *cn)
        ops.fix(mt, 0,0,1,1,1,0)
        if ip in (1, N_PISOS):
            print(f"  piso {ip}: {len(cn)} nodos esclavos")
        masters.append(mt)
    print(f"  base empotrada: {nfix} nodos")
    return nid[0], et[0], masters

# ----------------------------------------------------------------------------
#  4. ANALISIS MODAL
# ----------------------------------------------------------------------------
def run_modal(nmodes=12):
    # En edificios bajos los GDL con masa son pocos (3 x N_PISOS) y ARPACK no
    # converge si se piden demasiados modos. Se acota y se prueba con fallback.
    nmax = min(nmodes, max(3, 3*N_PISOS - 12))
    for nm in range(nmax, 2, -1):
        try:
            ops.system('UmfPack')
            ops.numberer('RCM')
            ops.constraints('Transformation')
            eigs = ops.eigen('-genBandArpack', nm)
            Ts = [2*np.pi/np.sqrt(e) for e in eigs]
            return Ts, eigs
        except Exception:
            ops.wipeAnalysis()
            continue
    raise RuntimeError("eigen no convergio con ningun nmodes")

# ----------------------------------------------------------------------------
#  5. CLASIFICACION DE MODOS por masa participante
# ----------------------------------------------------------------------------
def classify_modes(masters, nmodes):
    Mtot = sum(MASS); Itot = sum(IROT)
    info = []
    for m in range(1, nmodes+1):
        Lx = Ly = Lrz = 0.0
        for ip, mt in enumerate(masters):
            ev = ops.nodeEigenvector(mt, m)
            Lx  += MASS[ip] * ev[0]
            Ly  += MASS[ip] * ev[1]
            Lrz += IROT[ip] * ev[5]
        Mx, My, Mrz = Lx**2, Ly**2, Lrz**2
        frac_x  = Mx/Mtot if Mtot else 0
        frac_y  = My/Mtot if Mtot else 0
        frac_rz = Mrz/Itot if Itot else 0
        dom = max([('trans_X',frac_x),('trans_Y',frac_y),('torsional',frac_rz)],
                  key=lambda p:p[1])
        info.append((m, dom[0], frac_x, frac_y, frac_rz))
    return info

# ----------------------------------------------------------------------------
#  6. EJECUCION PRINCIPAL
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print(f"MODELO SHELL OPENSEES")
    print("="*60)

    t0 = time.time()
    nn, ne, masters = build_model(mesh=MESH)
    print(f"Nodos: {nn} | Shells: {ne} | construccion: {time.time()-t0:.1f}s")

    t0 = time.time()
    Ts, eigs = run_modal(nmodes=12)
    nmodes = len(Ts)
    print(f"Analisis modal: {time.time()-t0:.1f}s  ({nmodes} modos)\n")

    modes = classify_modes(masters, nmodes=nmodes)

    print(f"{'Modo':>4} {'T [s]':>9} {'tipo':>11} {'%X':>7} {'%Y':>7} {'%Rz':>7}")
    for (m,tipo,fx,fy,frz),T in zip(modes,Ts):
        print(f"{m:>4} {T:>9.4f} {tipo:>11} {100*fx:>6.1f} {100*fy:>6.1f} {100*frz:>6.1f}")

    def first(tipo):
        for (m,t,fx,fy,frz),T in zip(modes,Ts):
            if t==tipo: return T
        return float('nan')
    T1y, T2t, T3x = first('trans_Y'), first('torsional'), first('trans_X')

    print("\n--- Comparacion con ground truth OpenSees (columna ancha) ---")
    print(f"  T1 (Y)    GT={T_GT['T1_Y']:.4f}  shell={T1y:.4f}  "
          f"d={100*(T1y-T_GT['T1_Y'])/T_GT['T1_Y']:+.1f}%")
    print(f"  T2 (Tors) GT={T_GT['T2_Tors']:.4f}  shell={T2t:.4f}  "
          f"d={100*(T2t-T_GT['T2_Tors'])/T_GT['T2_Tors']:+.1f}%")
    print(f"  T3 (X)    GT={T_GT['T3_X']:.4f}  shell={T3x:.4f}  "
          f"d={100*(T3x-T_GT['T3_X'])/T_GT['T3_X']:+.1f}%")
