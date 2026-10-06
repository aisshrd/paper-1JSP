"""
Геометрия больших уклонений для модели Каца.
Всё в единицах lambda = 1.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermeval
import json, sys

LAM = 1.0
L, H = 7.0, 0.07          # сетка по скорости
V = np.arange(-L, L + 1e-9, H); N = len(V)
M_TH = 48                  # число углов
TH = 2 * np.pi * (np.arange(M_TH) + 0.5) / M_TH
WTH = 1.0 / M_TH

def maxw(v, K=1.0):
    return np.exp(-v**2 / (2 * K)) / np.sqrt(2 * np.pi * K)

def bkw(v, a):
    """Точное решение Каца--Больцмана (тип Крука--Ву): a in (0,1/3]."""
    K = 1 - 2 * a
    return maxw(v, K) * ((K - a) / K + a * v**2 / K**2)

def dbkw_da(v, a, eps=1e-6):
    return (bkw(v, a + eps) - bkw(v, a - eps)) / (2 * eps)

# ---------- интерполяция на сетке ----------
def interp_idx(x):
    """линейная интерполяция: индексы и веса, вне сетки -> вес 0"""
    t = (x + L) / H
    i0 = np.floor(t).astype(int)
    w1 = t - i0
    ok = (i0 >= 0) & (i0 < N - 1)
    i0c = np.clip(i0, 0, N - 2)
    w0 = np.where(ok, 1 - w1, 0.0); w1 = np.where(ok, w1, 0.0)
    return i0c, w0, w1

def interp(g, x):
    i0, w0, w1 = interp_idx(x)
    return w0 * g[i0] + w1 * g[i0 + 1]

# предвычисленные образы столкновений: массивы (M_TH, N, N): [theta, u, v*]
U = V[:, None]; VS = V[None, :]
VP = np.stack([U * np.cos(t) - VS * np.sin(t) for t in TH])      # v'
VSP = np.stack([U * np.sin(t) + VS * np.cos(t) for t in TH])     # v*'
I0P, W0P, W1P = interp_idx(VP)
I0S, W0S, W1S = interp_idx(VSP)

def coll_vals(g):
    """значения g в v' и v*' на всей коллизионной сетке"""
    gp = W0P * g[I0P] + W1P * g[I0P + 1]
    gsp = W0S * g[I0S] + W1S * g[I0S + 1]
    return gp, gsp

def Delta(xi):
    xp, xsp = coll_vals(xi)
    return xp + xsp - xi[None, :, None] - xi[None, None, :]

def Wgeo(f):
    fp, fsp = coll_vals(f)
    return np.sqrt(np.maximum(f[None, :, None] * f[None, None, :] * fp * fsp, 0.0))

def Q_coll(f):
    """интеграл столкновений Каца"""
    fp, fsp = coll_vals(f)
    integrand = fp * fsp - f[None, :, None] * f[None, None, :]
    return LAM * WTH * H * integrand.sum(axis=(0, 2))

def entropy(f):
    m = f > 0
    return H * np.sum(f[m] * np.log(f[m]))

def logf(f):
    return np.log(np.maximum(f, 1e-300))

def rfield(f):
    """r = (1/2) log(f'f*'/(f f*)) на коллизионной сетке (там, где W>0)"""
    lf = logf(f)
    return 0.5 * Delta(lf)

def Psi_star(f, xi):
    return LAM * WTH * H * H * np.sum(Wgeo(f) * (np.cosh(0.5 * Delta(xi)) - 1))

def DPsi_star(f, xi):
    """D_xi Psi*(f,xi)(u) = -2 lam ∫ W sinh(Delta/2)"""
    W = Wgeo(f); D = Delta(xi)
    return -2 * LAM * WTH * H * np.sum(W * np.sinh(0.5 * D), axis=(0, 2))

def Konsager(f, weight=None):
    """матрица оператора Онзагера K(f): (K xi)(u) = -lam ∫∫ w(u,v*,th) Delta[xi]"""
    w = Wgeo(f) if weight is None else weight
    Kmat = np.zeros((N, N))
    diag = -LAM * WTH * H * w.sum(axis=(0, 2))          # член -xi(u)
    Kmat[np.arange(N), np.arange(N)] -= diag              # -(-xi(u)) => +
    # член -xi(v*): столбец j получает -(-lam) sum_th w[.,i,j]
    Kmat += LAM * WTH * H * w.sum(axis=0)
    # члены +xi(v') и +xi(v*'): K[i, idx] += -lam * w * weight
    for (I0, W0, W1) in ((I0P, W0P, W1P), (I0S, W0S, W1S)):
        for t in range(M_TH):
            c = -LAM * WTH * H * w[t]                    # (N,N) по (i,j)
            rows = np.repeat(np.arange(N), N)
            np.add.at(Kmat, (rows, I0[t].ravel()), (c * W0[t]).ravel())
            np.add.at(Kmat, (rows, I0[t].ravel() + 1), (c * W1[t]).ravel())
    return Kmat

def Lambda_logmean(f):
    """логарифмическое среднее Λ(f f*, f'f*') = W sinh r / r (Эрбар)"""
    W = Wgeo(f); r = rfield(f)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(np.abs(r) < 1e-8, 1.0, np.sinh(r) / r)
    return np.where(W > 0, W * ratio, 0.0)

def solve_K(Kmat, rhs):
    """решение K xi = rhs с учётом ядра {1, v^2}: наименьшие квадраты"""
    xi, *_ = np.linalg.lstsq(Kmat, rhs, rcond=1e-10)
    return xi

results = {}

# ======== 1. Проверка точного решения и интеграла столкновений ========
a0 = 0.30
f0 = bkw(V, a0)
Qn = Q_coll(f0)
dfdt_exact = (-a0 / 8) * dbkw_da(V, a0)
err_Q = np.max(np.abs(Qn - dfdt_exact)) / np.max(np.abs(dfdt_exact))
results['check_Q_vs_exact'] = float(err_Q)
results['mass'] = float(H * f0.sum()); results['energy'] = float(H * (V**2 * f0).sum())

# ======== 2. Тождество D_xi Psi*(f,-DE) = Q(f) ========
Qc = DPsi_star(f0, -logf(f0))
results['check_cosh_gradflow'] = float(np.max(np.abs(Qc - Qn)) / np.max(np.abs(Qn)))
# квадратичное (геом. среднее) приближение: K(-log f)
Kg = Konsager(f0)
Qq = Kg @ (-logf(f0))
results['err_quadratic_geomean_a0.30'] = float(np.linalg.norm(Qq - Qn) / np.linalg.norm(Qn))
# лог-среднее (Эрбар): точно
Kl = Konsager(f0, Lambda_logmean(f0))
Ql = Kl @ (-logf(f0))
results['err_quadratic_logmean_a0.30'] = float(np.linalg.norm(Ql - Qn) / np.linalg.norm(Qn))
# симметричность и положительность K
results['K_asym'] = float(np.max(np.abs(Kg - Kg.T)) / np.max(np.abs(Kg)))
ev = np.linalg.eigvalsh(0.5 * (Kg + Kg.T))
results['K_min_eig'] = float(ev.min()); results['K_max_eig'] = float(ev.max())

# ======== 3. Спектр K_M на полиномах Эрмита ========
M = maxw(V)
KM = Konsager(M)
spec = {}
for n in range(1, 9):
    c = np.zeros(n + 1); c[n] = 1
    Hn = hermeval(V, c)
    lhs = KM @ Hn
    # gamma_n = <M Hn, K Hn>/<M Hn, M Hn> (проекция)
    gam = H * np.sum(lhs * Hn) / (H * np.sum(M * Hn * Hn))
    from math import factorial
    def dfact(k):
        return 1 if k <= 0 else k * dfact(k - 2)
    theo = 1 - 2 * (dfact(n - 1) / dfact(n) if n % 2 == 0 else 0.0)
    spec[n] = (float(gam), float(theo))
results['hermite_spectrum'] = spec

# ======== 4-5. Вдоль решения BKW: диссипации, отношения, kappa_eff, длина ========
alphas = np.concatenate([np.linspace(0.02, 0.30, 15), [0.32, 0.33]])
rows = []
for a in alphas:
    f = bkw(V, a)
    W = Wgeo(f); r = rfield(f)
    mask = W > 0
    r = np.where(mask, r, 0.0)
    D_exact = LAM * WTH * H * H * np.sum(W * r * np.sinh(r))
    D_quad = LAM * WTH * H * H * np.sum(W * r * r)
    Psi_st = LAM * WTH * H * H * np.sum(W * (np.cosh(r) - 1))
    Psi_pr = D_exact - Psi_st
    # прямая проверка: -dE/dt = -<log f, Q>
    D_direct = -H * np.sum(logf(f) * Q_coll(f))
    # квадратичная норма скорости потока в геометрии геом. среднего
    K = Konsager(f)
    Qf = Q_coll(f)
    xi = solve_K(K, Qf)
    speed2_geo = H * np.sum(xi * Qf)
    # индуцированная метрика на семействе: g(a) = <xi_a, d_a f>
    dfa = dbkw_da(V, a)
    xia = solve_K(K, dfa)
    g_a = H * np.sum(xia * dfa)
    # лог-среднее
    KL = Konsager(f, Lambda_logmean(f))
    xiL = solve_K(KL, Qf)
    speed2_log = H * np.sum(xiL * Qf)
    # r-статистика (весовая)
    wsum = np.sum(W); rmax = np.max(np.abs(r)); rrms = np.sqrt(np.sum(W * r * r) / wsum)
    Ent = entropy(f)
    rows.append(dict(a=float(a), D_exact=float(D_exact), D_direct=float(D_direct),
                     D_quad=float(D_quad), Psi_star=float(Psi_st), Psi=float(Psi_pr),
                     speed2_geo=float(speed2_geo), speed2_log=float(speed2_log),
                     g_a=float(g_a), rmax=float(rmax), rrms=float(rrms), E=float(Ent)))
    print(f"a={a:.3f} Dex={D_exact:.5f} Ddir={D_direct:.5f} Dq={D_quad:.5f} "
          f"|Q|2geo={speed2_geo:.5f} |Q|2log={speed2_log:.5f} g={g_a:.4f} rrms={rrms:.3f}", flush=True)
results['bkw_rows'] = rows

# kappa_eff = -(1/2) d/dt log D,  da/dt = -a/8  =>  kappa = (a/16) d log D / da
A = np.array([r_['a'] for r_ in rows]); Dx = np.array([r_['D_exact'] for r_ in rows])
Dq = np.array([r_['D_quad'] for r_ in rows]); G = np.array([r_['g_a'] for r_ in rows])
kap_ex = (A / 16) * np.gradient(np.log(Dx), A)
kap_q = (A / 16) * np.gradient(np.log(Dq), A)
# длина траектории ℓ(a) = ∫_0^a sqrt(g) da'  (от равновесия)
from scipy.integrate import cumulative_trapezoid
Aext = np.concatenate([[0.0], A]); Gext = np.concatenate([[0.0], G])
ell = cumulative_trapezoid(np.sqrt(np.maximum(Gext, 0)), Aext, initial=0.0)[1:]
results['kappa_exact'] = kap_ex.tolist(); results['kappa_quad'] = kap_q.tolist()
results['ell'] = ell.tolist(); results['ell_lin'] = (2 * np.sqrt(6) * A**2).tolist()
# энтропия относительно равновесия
E_eq = entropy(M)
results['E_rel'] = [float(r_['E'] - E_eq) for r_ in rows]

# ======== 6. Геодезическая vs поток (демонстрация) ========
def geodesic_rhs(f, xi):
    K = Konsager(f)
    W = Wgeo(f); D = Delta(xi)
    fdot = K @ xi
    xidot = -(LAM / 4) * WTH * H * np.sum(W * D * D, axis=(0, 2)) / np.maximum(f, 1e-12)
    xidot = np.where(f > 1e-8, xidot, 0.0)
    return fdot, xidot

def run_geodesic(f_init, xi_init, T=1.5, dt=0.05):
    f, xi = f_init.copy(), xi_init.copy()
    traj = [(0.0, f.copy())]
    t = 0.0
    while t < T - 1e-12:
        k1 = geodesic_rhs(f, xi)
        k2 = geodesic_rhs(f + dt/2*k1[0], xi + dt/2*k1[1])
        k3 = geodesic_rhs(f + dt/2*k2[0], xi + dt/2*k2[1])
        k4 = geodesic_rhs(f + dt*k3[0], xi + dt*k3[1])
        f = f + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        xi = xi + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        t += dt
        traj.append((t, f.copy()))
    return traj

ag = 0.30
fg = bkw(V, ag); Kg = Konsager(fg); Qg = Q_coll(fg); xig = solve_K(Kg, Qg)
speed0 = H * np.sum(xig * Qg)
geo = run_geodesic(fg, xig, T=1.5, dt=0.05)
geo_cmp = []
for (t, fgeo) in geo:
    a_t = ag * np.exp(-t / 8)
    fflow = bkw(V, a_t)
    # скорость вдоль геодезической (должна сохраняться)
    geo_cmp.append(dict(t=float(t), l2_dev=float(np.sqrt(H * np.sum((fgeo - fflow)**2))),
                        l2_flow_move=float(np.sqrt(H * np.sum((fflow - fg)**2))),
                        mass=float(H * fgeo.sum()), energy=float(H * np.sum(V**2 * fgeo)),
                        minf=float(fgeo.min())))
results['geodesic_vs_flow'] = geo_cmp
results['geo_speed0'] = float(speed0)
np.save('../data/geo_traj.npy', np.array([g[1] for g in geo]))
np.save('../data/V.npy', V)

with open('../data/results.json', 'w') as fh:
    json.dump(results, fh, indent=1)
print(json.dumps({k: v for k, v in results.items() if k not in ('bkw_rows', 'geodesic_vs_flow')}, indent=1))
