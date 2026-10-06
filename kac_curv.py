"""
Индуцированная метрика на двумерном моментном подмногообразии
  f_a = M (1 + a4 He4/sqrt(4!) + a6 He6/sqrt(6!)),
гауссова кривизна (формула Бриоски), геодезические и сравнение с
линеаризованным потоком  a_n' = -gamma_n a_n.
"""
import numpy as np, json, time
from numpy.polynomial.hermite_e import hermeval
from math import factorial
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
exec(open('kac_geom.py').read().split('results = {}')[0])

M = maxw(V)
def He(n):
    c = np.zeros(n + 1); c[n] = 1
    return hermeval(V, c) / np.sqrt(factorial(n))
E4, E6 = M * He(4), M * He(6)          # касательные векторы d f/d a4, d f/d a6
def f_of(a):
    return M + a[0] * E4 + a[1] * E6

DEGS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]   # без ядра {1, v^2}
P = np.stack([He(k) for k in DEGS], 1)
def metric(a):
    """g_ij = sup_xi [2<xi,df_i> - <xi,K xi>] в подпространстве полиномов (нижняя оценка, гладкая)"""
    f = np.maximum(f_of(a), 0.0)
    K = Konsager(f)
    Ks = P.T @ K @ P * H; B = P.T @ np.stack([E4, E6], 1) * H
    C = np.linalg.solve(0.5 * (Ks + Ks.T), B)
    g = B.T @ C
    return 0.5 * (g + g.T)

da = 0.03
grid4 = np.arange(0.0, 0.36 + 1e-9, da); grid6 = np.arange(0.0, 0.24 + 1e-9, da)
n4, n6 = len(grid4), len(grid6)
G = np.zeros((n4, n6, 2, 2))
t0 = time.time()
for i, a4 in enumerate(grid4):
    for j, a6 in enumerate(grid6):
        G[i, j] = metric((a4, a6))
print('metric grid done', time.time() - t0, flush=True)
np.save('../data/Ggrid.npy', G); np.save('../data/agrid4.npy', grid4); np.save('../data/agrid6.npy', grid6)

Eg, Fg, Gg = G[:, :, 0, 0], G[:, :, 0, 1], G[:, :, 1, 1]
print('g at origin', G[0, 0], ' theory diag(1/gamma4, 1/gamma6) =', 4.0, 8 / 3)

# ---- гауссова кривизна (формула Бриоски) ----
def d(A, ax): return np.gradient(A, da, axis=ax)
Eu, Ev = d(Eg, 0), d(Eg, 1); Fu, Fv = d(Fg, 0), d(Fg, 1); Gu, Gv = d(Gg, 0), d(Gg, 1)
Evv, Fuv, Guu = d(Ev, 1), d(Fu, 1), d(Gu, 0)
det = Eg * Gg - Fg**2
M1 = np.array([[-0.5 * Evv + Fuv - 0.5 * Guu, 0.5 * Eu, Fu - 0.5 * Ev],
               [Fv - 0.5 * Gu, Eg, Fg],
               [0.5 * Gv, Fg, Gg]])
M2 = np.array([[np.zeros_like(Eg), 0.5 * Ev, 0.5 * Gu],
               [0.5 * Ev, Eg, Fg],
               [0.5 * Gu, Fg, Gg]])
def det3(A):
    return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))
Kgauss = (det3(M1) - det3(M2)) / det**2
np.save('../data/Kgauss.npy', Kgauss)
print('Gaussian curvature near origin', Kgauss[1, 1], ' range (inner)', Kgauss[1:-1, 1:-1].min(), Kgauss[1:-1, 1:-1].max())

# ---- геодезические ----
gi = {k: RegularGridInterpolator((grid4, grid6), A, bounds_error=False, fill_value=None)
      for k, A in dict(E=Eg, F=Fg, G=Gg, Eu=Eu, Ev=Ev, Fu=Fu, Fv=Fv, Gu=Gu, Gv=Gv).items()}
def christoffel(a):
    p = np.array(a)[None, :]
    E_, F_, G_ = gi['E'](p)[0], gi['F'](p)[0], gi['G'](p)[0]
    gm = np.array([[E_, F_], [F_, G_]]); ginv = np.linalg.inv(gm)
    dg = np.zeros((2, 2, 2))   # dg[k, i, j] = d g_ij / d a_k
    dg[0] = [[gi['Eu'](p)[0], gi['Fu'](p)[0]], [gi['Fu'](p)[0], gi['Gu'](p)[0]]]
    dg[1] = [[gi['Ev'](p)[0], gi['Fv'](p)[0]], [gi['Fv'](p)[0], gi['Gv'](p)[0]]]
    Gam = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                Gam[k, i, j] = 0.5 * sum(ginv[k, l] * (dg[i, l, j] + dg[j, l, i] - dg[l, i, j]) for l in range(2))
    return Gam, gm
def geo_rhs(t, y):
    a, v = y[:2], y[2:]
    Gam, _ = christoffel(a)
    acc = -np.einsum('kij,i,j->k', Gam, v, v)
    return np.concatenate([v, acc])

def length(curve):
    Lsum = 0.0
    for p0, p1 in zip(curve[:-1], curve[1:]):
        mid = 0.5 * (p0 + p1); _, gm = christoffel(mid); dv = p1 - p0
        Lsum += np.sqrt(dv @ gm @ dv)
    return Lsum

def shoot(a_start, a_target, iters=40):
    """пристрелка: геодезическая из a_start в a_target"""
    v = np.array(a_target) - np.array(a_start)
    for _ in range(iters):
        sol = solve_ivp(geo_rhs, (0, 1), np.concatenate([a_start, v]), rtol=1e-8, atol=1e-10, dense_output=True)
        end = sol.y[:2, -1]
        err = np.array(a_target) - end
        if np.linalg.norm(err) < 1e-7: break
        # якобиан по конечным разностям
        J = np.zeros((2, 2))
        for k in range(2):
            vv = v.copy(); vv[k] += 1e-5
            s2 = solve_ivp(geo_rhs, (0, 1), np.concatenate([a_start, vv]), rtol=1e-8, atol=1e-10)
            J[:, k] = (s2.y[:2, -1] - end) / 1e-5
        v = v + np.linalg.solve(J, err)
    ts = np.linspace(0, 1, 201)
    return sol.sol(ts)[:2].T

gam4, gam6 = 0.25, 0.375
out = []
for a_start in [(0.30, 0.03), (0.15, 0.21), (0.33, 0.21), (0.06, 0.21)]:
    geo = shoot(np.array(a_start), np.array([0.0, 0.0]))
    ts = np.linspace(0, 40, 4001)
    flow = np.stack([a_start[0] * np.exp(-gam4 * ts), a_start[1] * np.exp(-gam6 * ts)], 1)
    line = np.stack([np.linspace(a_start[0], 0, 201), np.linspace(a_start[1], 0, 201)], 1)
    Lg, Lf, Ll = length(geo), length(flow), length(line)
    out.append(dict(start=list(a_start), L_geo=float(Lg), L_flow=float(Lf), L_line=float(Ll)))
    np.save(f'../data/geo_{a_start[0]:+.2f}_{a_start[1]:+.2f}.npy', geo)
    print(a_start, 'geodesic', Lg, 'flow', Lf, 'coord-line', Ll, flush=True)
json.dump(dict(g0=G[0, 0].tolist(), K0=float(Kgauss[1, 1]),
               Kmin=float(Kgauss[1:-1, 1:-1].min()), Kmax=float(Kgauss[1:-1, 1:-1].max()),
               geodesics=out), open('../data/curv.json', 'w'), indent=1)
