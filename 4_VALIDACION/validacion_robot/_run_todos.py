import glob, numpy as np, io, contextlib, re
casos = [3034,3078,3357,3362,3570,4051,4827,5362,5671,5821,
         6182,6478,6624,6679,6804,7139,7362,7801,8358,9177]
print(f"{'caso':>5} {'N':>3} | {'T1y_sh':>7} {'T1y_GT':>7} {'d%':>6} | {'T2t_sh':>7} {'T2t_GT':>7} | {'T3x_sh':>7} {'T3x_GT':>7} {'d%':>6}")
print("-"*78)
for cid in casos:
    src = open(f'_celda_shell_{cid}.py',encoding='utf-8').read().split('if __name__')[0]
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            exec(src, ns)
            nn,ne,masters = ns['build_model'](mesh=1.4)
            Ts,eigs = ns['run_modal'](nmodes=12)
            modes = ns['classify_modes'](masters, nmodes=len(Ts))
        except Exception as e:
            print(f"{cid:>5}  ERROR: {e}"); continue
    def first(t):
        for (m,ty,fx,fy,frz),T in zip(modes,Ts):
            if ty==t: return T
        return float('nan')
    y,tt,x = first('trans_Y'),first('torsional'),first('trans_X')
    G = ns['T_GT']; N = ns['N_PISOS']
    dy = 100*(y-G['T1_Y'])/G['T1_Y']; dx = 100*(x-G['T3_X'])/G['T3_X']
    print(f"{cid:>5} {N:>3} | {y:>7.4f} {G['T1_Y']:>7.4f} {dy:>+5.1f} | {tt:>7.4f} {G['T2_Tors']:>7.4f} | {x:>7.4f} {G['T3_X']:>7.4f} {dx:>+5.1f}")
