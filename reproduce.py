
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

STUDENT_ID = "2023270177"
SEED = 270177
DESIGN_SEED = 270178
TARGET_WAVELENGTH = 600
WAVELENGTHS = np.arange(400., 801., 10.)
N_H, N_L, N_S, N_0 = 2.30, 1.45, 1.52, 1.00

def tmm_reflectance(thicknesses):
    d = np.asarray(thicknesses, dtype=float)
    one = d.ndim == 1
    if one: d = d[None, :]
    wl = WAVELENGTHS
    ns = [N_H, N_L, N_H, N_L]
    M11 = np.ones((len(d), len(wl)), complex)
    M12 = np.zeros_like(M11); M21 = np.zeros_like(M11); M22 = np.ones_like(M11)
    for k,n in enumerate(ns):
        delta = 2*np.pi*n*d[:,k,None]/wl[None,:]
        c, s = np.cos(delta), np.sin(delta)
        A,B,C,D = c,1j*s/n,1j*n*s,c
        M11,M12,M21,M22 = (M11*A+M12*C, M11*B+M12*D,
                           M21*A+M22*C, M21*B+M22*D)
    r = (N_0*M11+N_0*N_S*M12-M21-N_S*M22) / \
        (N_0*M11+N_0*N_S*M12+M21+N_S*M22)
    R = np.abs(r)**2
    return R[0] if one else R

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,41))
    def forward(self,x): return self.net(x)

def train(X,Y,n_train):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model=MLP()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=nn.MSELoss()
    tr=TensorDataset(torch.tensor((X[:n_train]-40)/140,dtype=torch.float32),
                     torch.tensor(Y[:n_train],dtype=torch.float32))
    vaX=torch.tensor((X[4000:4500]-40)/140,dtype=torch.float32)
    vaY=torch.tensor(Y[4000:4500],dtype=torch.float32)
    loader=DataLoader(tr,batch_size=128,shuffle=True,
                      generator=torch.Generator().manual_seed(SEED))
    best=1e9; best_state=None; wait=0
    for epoch in range(140):
        model.train()
        for xb,yb in loader:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): v=loss_fn(model(vaX),vaY).item()
        if v<best:
            best=v; wait=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=18: break
    model.load_state_dict(best_state)
    return model

def main():
    rng=np.random.default_rng(SEED)
    X=rng.uniform(40,180,(5000,4)).astype(np.float32)
    Y=tmm_reflectance(X).astype(np.float32)
    p=rng.permutation(5000)
    X,Y=X[p],Y[p]
    for n in [500,1000,2000,4000]:
        m=train(X,Y,n)
        with torch.no_grad():
            pred=m(torch.tensor((X[4500:]-40)/140,dtype=torch.float32)).numpy()
        mse=np.mean((pred-Y[4500:])**2)
        mae=np.mean(np.abs(pred-Y[4500:]))
        print(n,"MSE",mse,"RMSE",np.sqrt(mse),"MAE",mae)
    m=train(X,Y,4000)
    drng=np.random.default_rng(DESIGN_SEED)
    C=drng.uniform(40,180,(10000,4)).astype(np.float32)
    with torch.no_grad():
        P=m(torch.tensor((C-40)/140,dtype=torch.float32)).numpy()
    idx=np.argmin(abs(WAVELENGTHS-TARGET_WAVELENGTH))
    top=np.argsort(P[:,idx])[-10:][::-1]
    R=tmm_reflectance(C[top])[:,idx]
    order=np.argsort(R)[-5:][::-1]
    print("Top-5 verified designs:")
    for rank,j in enumerate(order,1):
        print(rank,C[top[j]],"MLP",P[top[j],idx],"TMM",R[j])

if __name__=="__main__":
    main()
