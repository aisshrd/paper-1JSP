"""
Трёхчастичная модель Каца: численная часть.
Квадратуры: Гаусс--Эрмит по (v,v*,v**), углы Эйлера (равномерно по alpha,gamma; Гаусс--Лежандр по cos beta).
lambda = 1.
"""
import numpy as np, json, time, sympy as sp
from numpy.polynomial.hermite_e import hermeval, hermegauss
from numpy.polynomial.legendre import leggauss
from math import factorial

LAM = 1.0
# ---------- квадратура по SO(3) (мера Хаара) ----------
def haar_nodes(na=12, nb=8):
    al = 2*np.pi*(np.arange(na)+0.5)/na; ga = al.copy()
    xb, wb = leggauss(nb); be = np.arccos(xb)
    A, B, G = np.meshgrid(al, be, ga, indexing='ij')
    Wt = (np.ones(na)[:, None, None] * wb[None, :, None] * np.ones(na)[None, None, :]) / (na*na*2)
    ca, sa, cb, sb, cg, sg = np.cos(A), np.sin(A), np.cos(B), np.sin(B), np.cos(G), np.sin(G)
    # R = Rz(alpha) Ry(beta) Rz(gamma); первая строка:
    R11 = ca*cb*cg - sa*sg; R12 = -ca*cb*sg - sa*cg; R13 = ca*sb
    R21 = sa*cb*cg + ca*sg; R22 = -sa*cb*sg + ca*cg; R23 = sa*sb
    R31 = -sb*cg;           R32 = sb*sg;             R33 = cb
    R = np.stack([R11, R12, R13, R21, R22, R23, R31, R32, R33], -1).reshape(-1, 3, 3)
    return R, Wt.ravel()

R, WR = haar_nodes()
print('Haar check <x^2>,<x^4>,<x^2y^2>,<x^6>,<x^4y^2>,<x^2y^2z^2>:',
      [round(float(np.sum(WR*g)), 6) for g in (R[:,0,0]**2, R[:,0,0]**4, R[:,0,0]**2*R[:,0,1]**2,
                                            R[:,0,0]**6, R[:,0,0]**4*R[:,0,1]**2, R[:,0,0]**2*R[:,0,1]**2*R[:,0,2]**2)],
      'theory 1/3,1/5,1/15,1/7,1/35,1/105')

# ---------- квадратура Гаусса--Эрмита ----------
def gh(n):
    x, w = hermegauss(n); return x, w/np.sqrt(2*np.pi)   # вес M(v)
xg, wg = gh(16)
V1, V2, V3 = np.meshgrid(xg, xg, xg, indexing='ij'); V1, V2, V3 = V1.ravel(), V2.ravel(), V3.ravel()
WV = (wg[:, None, None]*wg[None, :, None]*wg[None, None, :]).ravel()

def He(n, x):
    c = np.zeros(n+1); c[n] = 1; return hermeval(x, c)

# ---------- 1. Спектр: gamma_n = 1 - 3<x^n> ----------
spec = {}
for n in range(1, 9):
    # <He_n(R11 v + R12 v* + R13 v**)> по v*,v** и Хаару, делённое на He_n(v): берём v = 1.3
    v = 1.3
    val = 0.0
    for i in range(3):
        arg = R[:, None, i, 0]*v + R[:, None, i, 1]*xg[None, :]  # ещё v** ниже
        # усредняем по v* и v** отдельно (сепарабельно) — полная сумма
        s = 0.0
        for j, vs in enumerate(xg):
            for k, vss in enumerate(xg):
                pass
        # быстрее: используем лемму напрямую численно только по Хаару и v*,v** сеткой
    # прямой расчёт на сетке (v* , v**) и Хааре:
    acc = 0.0
    for i in range(3):
        arg = (R[:, i, 0][:, None]*v + R[:, i, 1][:, None]*V2[None, :] + R[:, i, 2][:, None]*V3[None, :])
        acc += np.sum(WR[:, None]*WV[None, :]*He(n, arg))
    gam = 1 - acc/He(n, v)
    spec[n] = (float(gam), float(1-3/(n+1)) if n % 2 == 0 else 1.0)
print('spectrum', spec)

# ---------- 2. Моментная иерархия (символьно) ----------
# d m_n/dt = lam [ 3 < (x v + y v* + z v**)^n >_{S^2, f} - m_n ]  для симметричного f
x, y, z = sp.symbols('x y z')
def haar_mono(a, b, c):
    if a % 2 or b % 2 or c % 2: return sp.Integer(0)
    n = (a+b+c)//2
    return sp.Integer(sp.factorial2(a-1)*sp.factorial2(b-1)*sp.factorial2(c-1))/sp.factorial2(2*n+1)
m = {k: sp.Symbol(f'm{k}') for k in range(0, 13)}; m[0] = 1; m[2] = 1
def moment_rhs(n):
    poly = sp.Poly(sp.expand((x*sp.Symbol('v') + y*sp.Symbol('w') + z*sp.Symbol('u'))**n), x, y, z)
    tot = 0
    for (a, b, c), coef in poly.terms():
        hm = haar_mono(a, b, c)
        if hm == 0: continue
        # coef содержит v^a w^b u^c -> m_a m_b m_c (нечётные моменты = 0)
        if a % 2 or b % 2 or c % 2: continue
        cc = sp.Poly(coef, sp.Symbol('v'), sp.Symbol('w'), sp.Symbol('u')).terms()[0][1]
        tot += cc*hm*(m[a] if a else 1)*(m[b] if b else 1)*(m[c] if c else 1)
    return sp.simplify(tot - m[n])
mom_eqs = {n: moment_rhs(n) for n in (4, 6, 8)}
print('moment eqs:', mom_eqs)

# точное решение иерархии для m4, m6 (линейные с источником)
t = sp.Symbol('t'); m40, m60 = sp.symbols('m40 m60')
m4t = 3 + (m40-3)*sp.exp(-sp.Rational(2,5)*t)
rhs6 = mom_eqs[6].subs(m[4], m4t)
M6 = sp.Function('M6')
sol6 = sp.dsolve(sp.Eq(M6(t).diff(t), rhs6.subs(m[6], M6(t))), M6(t), ics={M6(0): m60})
print('m6(t) =', sp.simplify(sol6.rhs))
m6t = sp.simplify(sol6.rhs)

# ---------- 3. Метрика на подмногообразии (a4,a6) ----------
DEGS = [1, 3, 4, 5, 6, 7, 8, 9, 10]
def basis_vals(vv):
    return np.stack([He(k, vv)/np.sqrt(factorial(k)) for k in DEGS], -1)

def f_ratio(a, vv):   # f/M = 1 + a4 He4/sqrt(24) + a6 He6/sqrt(720)
    return 1 + a[0]*He(4, vv)/np.sqrt(24) + a[1]*He(6, vv)/np.sqrt(720)

def metric_and_diss(a, chunk=200):
    """g_ij (двойственная форма, xi в полиномах) и точная/квадратичная диссипация"""
    Ksub = np.zeros((len(DEGS), len(DEGS)))
    Dex = 0.0; Dq = 0.0
    B0 = basis_vals(V1) + basis_vals(V2) + basis_vals(V3)         # xi + xi* + xi**
    h1, h2, h3 = f_ratio(a, V1), f_ratio(a, V2), f_ratio(a, V3)
    pre = np.maximum(h1*h2*h3, 0.0)
    lf = np.log(np.maximum(h1, 1e-300)) + np.log(np.maximum(h2, 1e-300)) + np.log(np.maximum(h3, 1e-300))
    for s in range(0, len(WR), chunk):
        Rc = R[s:s+chunk]; wr = WR[s:s+chunk]
        Vp = Rc[:, 0, 0][:, None]*V1 + Rc[:, 0, 1][:, None]*V2 + Rc[:, 0, 2][:, None]*V3
        Vsp = Rc[:, 1, 0][:, None]*V1 + Rc[:, 1, 1][:, None]*V2 + Rc[:, 1, 2][:, None]*V3
        Vssp = Rc[:, 2, 0][:, None]*V1 + Rc[:, 2, 1][:, None]*V2 + Rc[:, 2, 2][:, None]*V3
        hp = np.maximum(f_ratio(a, Vp)*f_ratio(a, Vsp)*f_ratio(a, Vssp), 0.0)
        W3 = np.sqrt(pre[None, :]*hp)                                  # без гауссова веса (он в WV, инвариантен)
        wtot = (wr[:, None]*WV[None, :])*W3
        D3 = basis_vals(Vp) + basis_vals(Vsp) + basis_vals(Vssp) - B0[None, :, :]   # (chunk, NV, nb)
        Ksub += np.einsum('cv,cvi,cvj->ij', wtot, D3, D3)
        ok = (hp > 0) & (pre[None, :] > 0)
        r = np.where(ok, 0.5*(np.log(np.maximum(hp, 1e-300)) - lf[None, :]), 0.0)
        Dex += np.sum(np.where(ok, wtot*r*np.sinh(r), 0.0)); Dq += np.sum(wtot*r*r)
    # <xi,K xi> = (lam/6) ∫ W3 Δ^2   (из Psi* = (2lam/3)∫W3(cosh(Δ/2)-1) => квадр. часть (lam/12)∫W3Δ^2 = 1/2<xi,Kxi>)
    Ksub *= LAM/6
    B = np.array([[np.sum(WV*f_ratio(a, V1)*0 + WV*He(4, V1)/np.sqrt(24)*basis_vals(V1)[:, i]) for i in range(len(DEGS))],
                  [np.sum(WV*He(6, V1)/np.sqrt(720)*basis_vals(V1)[:, i]) for i in range(len(DEGS))]])
    C = np.linalg.solve(0.5*(Ksub+Ksub.T), B.T)
    g = B @ C; g = 0.5*(g+g.T)
    return g, (2*LAM/3)*Dex, (2*LAM/3)*Dq

t0 = time.time()
g0, _, _ = metric_and_diss((0.0, 0.0))
print('g at origin', g0, ' theory diag(5/2, 7/4)', time.time()-t0)

da = 0.05
grid4 = np.arange(0, 0.30+1e-9, da); grid6 = np.arange(0, 0.20+1e-9, da)
G = np.zeros((len(grid4), len(grid6), 2, 2)); DEX = np.zeros((len(grid4), len(grid6))); DQ = DEX.copy()
for i, a4 in enumerate(grid4):
    for j, a6 in enumerate(grid6):
        G[i, j], DEX[i, j], DQ[i, j] = metric_and_diss((a4, a6))
    print('row', i, time.time()-t0, flush=True)
np.save('../data/G3.npy', G); np.save('../data/D3.npy', np.stack([DEX, DQ]))

# кривизна (Бриоски)
Eg, Fg, Gg = G[:, :, 0, 0], G[:, :, 0, 1], G[:, :, 1, 1]
d = lambda A, ax: np.gradient(A, da, axis=ax)
Eu, Ev = d(Eg, 0), d(Eg, 1); Fu, Fv = d(Fg, 0), d(Fg, 1); Gu, Gv = d(Gg, 0), d(Gg, 1)
Evv, Fuv, Guu = d(Ev, 1), d(Fu, 1), d(Gu, 0)
det = Eg*Gg - Fg**2
def det3(A):
    return (A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1]) - A[0,1]*(A[1,0]*A[2,2]-A[1,2]*A[2,0]) + A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
M1 = np.array([[-0.5*Evv+Fuv-0.5*Guu, 0.5*Eu, Fu-0.5*Ev],[Fv-0.5*Gu, Eg, Fg],[0.5*Gv, Fg, Gg]])
M2 = np.array([[np.zeros_like(Eg), 0.5*Ev, 0.5*Gu],[0.5*Ev, Eg, Fg],[0.5*Gu, Fg, Gg]])
Kg = (det3(M1)-det3(M2))/det**2
np.save('../data/K3.npy', Kg)
print('curvature interior', Kg[1:-1, 1:-1])

# ---------- 4. Точная моментная траектория и kappa_eff ----------
m4f = sp.lambdify(t, m4t.subs(m40, 3+0.30*np.sqrt(24)))      # a4(0)=0.30
m6f = sp.lambdify(t, m6t.subs({m40: 3+0.30*np.sqrt(24), m60: 15*1 - 0 + 0}))  # a6(0)=0 -> m6 = 15 m4 - 30 + a6*sqrt(720)
# a6 = (m6 - 15 m4 + 30)/sqrt(720): при a6(0)=0, m60 = 15*m40 - 30
m60v = 15*(3+0.30*np.sqrt(24)) - 30
m6f = sp.lambdify(t, m6t.subs({m40: 3+0.30*np.sqrt(24), m60: m60v}))
ts = np.linspace(0, 6, 13)
traj = []
for tt in ts:
    a4 = (m4f(tt)-3)/np.sqrt(24); a6 = (m6f(tt) - 15*m4f(tt) + 30)/np.sqrt(720)
    _, Dex, Dq = metric_and_diss((a4, a6))
    traj.append((float(tt), float(a4), float(a6), float(Dex), float(Dq)))
    print('traj', traj[-1], flush=True)
traj = np.array(traj)
kap = -0.5*np.gradient(np.log(traj[:, 3]), traj[:, 0])
json.dump(dict(spec=spec, g0=g0.tolist(), moment_eqs={k: str(v) for k, v in mom_eqs.items()}, m6t=str(m6t),
               traj=traj.tolist(), kappa=kap.tolist(), grid4=grid4.tolist(), grid6=grid6.tolist(),
               K_interior=Kg[1:-1, 1:-1].tolist()), open('../data/three_body.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
im = ax[0].pcolormesh(grid4[1:-1], grid6[1:-1], Kg[1:-1, 1:-1].T, shading='nearest', cmap='RdBu_r')
plt.colorbar(im, ax=ax[0], label='гауссова кривизна'); ax[0].set_xlabel('$a_4$'); ax[0].set_ylabel('$a_6$')
ax[0].plot(traj[:, 1], traj[:, 2], 'k.-', ms=3, label='точная моментная траектория'); ax[0].legend(fontsize=7)
ax[1].semilogy(traj[:, 0], traj[:, 3], 'o-', label=r'$D_{exact}$'); ax[1].semilogy(traj[:, 0], traj[:, 4], 's--', label=r'$D_2$')
ax[1].set_xlabel('$t$'); ax[1].legend()
ax[2].plot(traj[:, 0], kap, 'o-'); ax[2].axhline(0.4, color='k', ls=':', label=r'$\gamma_4=2/5$'); ax[2].set_xlabel('$t$'); ax[2].set_ylabel(r'$\kappa_{eff}$'); ax[2].legend()
plt.tight_layout(); plt.savefig('../figures/fig_3body.png'); print('done')
