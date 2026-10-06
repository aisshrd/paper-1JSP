"""
Первопорядковая поправка к диссипации: D_2(a) = gamma_4 a^2 (1 - c a + O(a^2)),
f = M(1 + a h), h = He_4/sqrt(24).
   c = -A/gamma_4,   A = (lam/2m) ∫ M^{⊗m} dR [ (1/2) S Δ[h]^2 - Δ[h] Δ[h^2] ],  S = Σ_i (h(v_i)+h(v_i')).
m=2,3: точные квадратуры; m=4: Монте-Карло по Хаару (QR гауссовых матриц).
"""
import numpy as np, json
from numpy.polynomial.hermite_e import hermeval, hermegauss
from numpy.polynomial.legendre import leggauss
from fractions import Fraction
LAM = 1.0
def He(n, x):
    c = np.zeros(n+1); c[n] = 1; return hermeval(x, c)
h = lambda x: He(4, x)/np.sqrt(24.0)
h2 = lambda x: h(x)**2

def gh_grid(m, n=16):
    x, w = hermegauss(n); w = w/np.sqrt(2*np.pi)
    grids = np.meshgrid(*([x]*m), indexing='ij'); V = np.stack([g.ravel() for g in grids])   # (m, NV)
    W = np.ones(V.shape[1])
    for g in np.meshgrid(*([w]*m), indexing='ij'): W = W*g.ravel()
    return V, W

def rotations(m, n_mc=200000, seed=0):
    if m == 2:
        th = 2*np.pi*(np.arange(16)+0.5)/16
        R = np.stack([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).transpose(2, 0, 1)
        return R, np.full(len(th), 1/len(th))
    if m == 3:
        na, nb = 14, 8
        al = 2*np.pi*(np.arange(na)+0.5)/na; xb, wb = leggauss(nb); be = np.arccos(xb)
        A, B, G = np.meshgrid(al, be, al, indexing='ij')
        Wt = (np.ones(na)[:, None, None]*wb[None, :, None]*np.ones(na)[None, None, :])/(na*na*2)
        ca, sa, cb, sb, cg, sg = np.cos(A), np.sin(A), np.cos(B), np.sin(B), np.cos(G), np.sin(G)
        R = np.stack([ca*cb*cg-sa*sg, -ca*cb*sg-sa*cg, ca*sb, sa*cb*cg+ca*sg, -sa*cb*sg+ca*cg, sa*sb, -sb*cg, sb*sg, cb], -1)
        return R.reshape(-1, 3, 3), Wt.ravel()
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_mc, m, m)); Q, Rr = np.linalg.qr(Z)
    d = np.sign(np.einsum('nii->ni', Rr)); Q = Q*d[:, None, :]
    return Q, np.full(n_mc, 1/n_mc)

def compute(m, chunk=64, n_mc=200000):
    V, W = gh_grid(m); R, WR = rotations(m, n_mc)
    hv = h(V); h2v = h2(V)                      # (m, NV)
    A_S = 0.0; A_X = 0.0; G4 = 0.0
    for s in range(0, len(WR), chunk):
        Rc, wr = R[s:s+chunk], WR[s:s+chunk]
        Vp = np.einsum('cij,jv->civ', Rc, V)   # (chunk, m, NV)
        hp, h2p = h(Vp), h2(Vp)
        Dh = hp.sum(1) - hv.sum(0)[None, :]; Dh2 = h2p.sum(1) - h2v.sum(0)[None, :]
        S = hp.sum(1) + hv.sum(0)[None, :]
        wt = wr[:, None]*W[None, :]
        G4 += np.sum(wt*Dh**2); A_S += np.sum(wt*S*Dh**2); A_X += np.sum(wt*Dh*Dh2)
    pref = LAM/(2*m)
    gamma4 = pref*G4; A = pref*(0.5*A_S - A_X); c = -A/gamma4
    return dict(m=m, gamma4=float(gamma4), A=float(A), c=float(c), A_S=float(pref*A_S), A_X=float(pref*A_X))

res = {}
for m in (2, 3):
    r = compute(m); res[m] = r
    print(m, r, 'gamma4 theory', (m-1)/(m+2), 'c rational?', Fraction(r['c']).limit_denominator(2000), flush=True)
# m=4 Монте-Карло с оценкой разброса
vals = [compute(4, n_mc=60000)['c'] for _ in range(1)]
r4 = compute(4, n_mc=60000); res[4] = r4
cs = []
for seed in range(3):
    V, W = gh_grid(4); R, WR = rotations(4, 40000, seed)
    # быстрый повтор
    hv = h(V); h2v = h2(V); G4 = A_S = A_X = 0.0
    for s in range(0, len(WR), 64):
        Rc, wr = R[s:s+64], WR[s:s+64]; Vp = np.einsum('cij,jv->civ', Rc, V); hp, h2p = h(Vp), h2(Vp)
        Dh = hp.sum(1)-hv.sum(0)[None]; Dh2 = h2p.sum(1)-h2v.sum(0)[None]; S = hp.sum(1)+hv.sum(0)[None]
        wt = wr[:, None]*W[None]; G4 += np.sum(wt*Dh**2); A_S += np.sum(wt*S*Dh**2); A_X += np.sum(wt*Dh*Dh2)
    cs.append(-(0.5*A_S-A_X)/G4); print('m=4 seed', seed, 'gamma4', G4/8, 'c', cs[-1], flush=True)
res[4]['c_samples'] = cs; res[4]['c_mean'] = float(np.mean(cs)); res[4]['c_std'] = float(np.std(cs))
print('m=4: c =', np.mean(cs), '+-', np.std(cs), ' gamma4 theory 3/6=0.5')

# ---- проверка конечной амплитудой (точная D_2 при малых a) для m=2 и m=3 ----
def D2_finite(m, a, n=16):
    V, W = gh_grid(m, n); R, WR = rotations(m)
    hv = h(V); fr = 1 + a*hv; lf = np.log(fr).sum(0)
    D = 0.0
    for s in range(0, len(WR), 64):
        Rc, wr = R[s:s+64], WR[s:s+64]; Vp = np.einsum('cij,jv->civ', Rc, V); frp = 1 + a*h(Vp)
        Wm = np.sqrt(np.prod(fr, 0)[None]*np.prod(frp, 1)); r = 0.5*(np.log(frp).sum(1) - lf[None])
        D += np.sum(wr[:, None]*W[None]*Wm*(2*r)**2)
    return LAM/(2*m)*D
fit = {}
for m in (2, 3):
    g4 = (m-1)/(m+2); rows = []
    for a in (0.005, 0.01, 0.02, 0.04):
        D = D2_finite(m, a); rows.append((a, D, (1-D/(g4*a*a))/a))
        print(f'm={m} a={a}: D2/(g4 a^2)={D/(g4*a*a):.5f}  c_est={(1-D/(g4*a*a))/a:.4f}', flush=True)
    fit[m] = rows
json.dump(dict(res={str(k): v for k, v in res.items()}, fit={str(k): v for k, v in fit.items()}), open('../data/c_coeff.json', 'w'), indent=1)
