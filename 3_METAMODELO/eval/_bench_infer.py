# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F, time
N_INPUTS, N_MODOS, N_MAX = 18, 18, 18

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.net(x))

class PINNModal_v4(nn.Module):
    def __init__(self, n_inputs, n_modos=18, n_max_pisos=18, hidden=256, hidden_T1=128):
        super().__init__()
        self.n_modos=n_modos; self.n_max_pisos=n_max_pisos
        self.encoder_T1 = nn.Sequential(nn.Linear(n_inputs,hidden_T1), nn.LayerNorm(hidden_T1),
            nn.SiLU(), ResBlock(hidden_T1), ResBlock(hidden_T1))
        self.head_T1 = nn.Sequential(nn.Linear(hidden_T1,32), nn.SiLU(), nn.Linear(32,1))
        self.encoder = nn.Sequential(nn.Linear(n_inputs,hidden), nn.LayerNorm(hidden), nn.SiLU(),
            ResBlock(hidden), ResBlock(hidden), ResBlock(hidden), ResBlock(hidden))
        self.head_T_rest = nn.Sequential(nn.Linear(hidden,128), nn.SiLU(), nn.Linear(128,n_modos-1))
        self.head_Phi = nn.Sequential(nn.Linear(hidden,512), nn.SiLU(), nn.Linear(512,n_modos*n_max_pisos*3))
        self.head_resp = nn.Sequential(nn.Linear(hidden,256), nn.SiLU(), nn.Linear(256,n_max_pisos*4))
        self.head_Vb = nn.Sequential(nn.Linear(hidden,64), nn.SiLU(), nn.Linear(64,2))
    def forward(self, X, mask):
        B=X.shape[0]
        h_T1=self.encoder_T1(X); T1=self.head_T1(h_T1)
        h=self.encoder(X); Td=F.softplus(self.head_T_rest(h))
        parts=[T1]
        for r in range(self.n_modos-1): parts.append(parts[-1]-Td[:,r:r+1])
        logT=torch.cat(parts,dim=1)
        Phi=self.head_Phi(h).view(B,self.n_modos,self.n_max_pisos,3)
        Phi=Phi*mask.unsqueeze(1).unsqueeze(-1)
        Phi=Phi/Phi.norm(dim=2,keepdim=True).clamp(min=1e-8)
        Phi=Phi.permute(0,2,1,3)
        resp=self.head_resp(h).view(B,self.n_max_pisos,4)
        Ux=resp[:,:,0]*mask; Uy=resp[:,:,1]*mask
        ratio=F.softplus(resp[:,:,2])*mask; dy=resp[:,:,3]*mask
        Vb=self.head_Vb(h)
        return logT,Phi[:,:,:,0],Phi[:,:,:,1],Phi[:,:,:,2],Ux,Uy,ratio,dy,Vb,ratio

@torch.no_grad()
def bench(dev, B, reps=50):
    m=PINNModal_v4(N_INPUTS).to(dev).eval()
    X=torch.randn(B,N_INPUTS,device=dev); mk=torch.ones(B,N_MAX,device=dev)
    for _ in range(10): m(X,mk)
    if dev=='cuda': torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(reps): m(X,mk)
    if dev=='cuda': torch.cuda.synchronize()
    dt=(time.perf_counter()-t0)/reps
    return dt, dt/B

print("GPU disponible:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("\n=== INFERENCIA METAMODELO ===")
for dev in (['cuda','cpu'] if torch.cuda.is_available() else ['cpu']):
    dt1,_=bench(dev,1)
    dtB,per=bench(dev,10000,reps=20)
    print(f"[{dev.upper()}] 1 edificio (latencia): {dt1*1e3:.3f} ms")
    print(f"[{dev.upper()}] lote 10.000: {dtB*1e3:.1f} ms total  ->  {per*1e6:.2f} us/edificio ({per*1e3:.4f} ms)")
