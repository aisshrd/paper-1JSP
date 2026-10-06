"""
Закрытие уровня A. lambda = 1.
 (1) Расходимость kappa_eff у границы alpha -> 1/3: константы D0, D1 и сравнение с численным kappa.
 (2) Самоподобное решение 3-частичной модели: рекурсия для Psi_3(x), радиус сходимости, положительность f.
 (3) Верхняя граница глобальной константы сжатия: inf D/(2(E-E_eq)) по семействам.
 (4) Термостат: спектр численно; предел скользящих столкновений: R_D -> 1.
 (5) Немаксвелловское ядро |v-v*|^beta: коэффициент c_beta.
 (6) Геодезические в полном многообразии: схема без 1/f (умножение на f), Галёркин по xi.
"""
import numpy as np, json, time
from numpy.polynomial.hermite_e import hermeval, hermegauss
from math import factorial
from fractions import Fraction
exec(open('kac_geom.py').read().split('results = {}')[0])
out = {}
def hn(n, x):
    c = np.zeros(n+1); c[n] = 1; return hermeval(x, c)/np.sqrt(factorial(n))

def rfield2(f):
    """r из логарифмов интерполированных значений (согласовано с W; важно у нуля f)"""
    fp, fsp = coll_vals(f)
    return 0.5*(logf(fp) + logf(fsp) - logf(f)[None, :, None] - logf(f)[None, None, :])
def D_exact(f):
    W = Wgeo(f); r = np.where(W > 0, rfield2(f), 0.0)
    return LAM*WTH*H*H*np.sum(W*r*np.sinh(r))
def E_rel(f):
    M = maxw(V); m = f > 0
    return H*np.sum(f[m]*np.log(f[m]/M[m]))

# ===================== (1) расходимость kappa_eff =====================
# D(eps) = D0 - D1 sqrt(eps) + ..., eps = 1/3 - alpha ; D1 = 2 pi Q0(0) sqrt(c1/c2), c1/c2 = 3
f0 = bkw(V, 1/3 - 1e-12)
D0 = D_exact(f0)
Q0 = Q_coll(f0)                       # интеграл столкновений при alpha=1/3
i0 = np.argmin(np.abs(V)); Qgain0 = Q0[i0]   # при v=0: f(0)=0, потерь нет, Q(0)=gain
D1 = 2*np.pi*Qgain0*np.sqrt(3.0)
print(f'D0={D0:.5f}  Q(0)|a=1/3 = {Qgain0:.5f}  D1={D1:.5f}  D1/D0={D1/D0:.4f}', flush=True)
rows = []
for eps in (0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125):
    a = 1/3 - eps; de = eps*0.2
    Dp, Dm = D_exact(bkw(V, a + de)), D_exact(bkw(V, a - de))
    D = D_exact(bkw(V, a))
    kappa = (LAM*a/16)*(np.log(Dp) - np.log(Dm))/(2*de)
    kappa_pred = (LAM*a/16)*D1/(2*np.sqrt(eps)*D)
    rows.append((eps, float(D), float(kappa), float(kappa_pred), float((D0 - D)/np.sqrt(eps))))
    print(f'eps={eps:.5f} D={D:.5f} kappa_num={kappa:.4f} kappa_asymp={kappa_pred:.4f} (D0-D)/sqrt(eps)={rows[-1][4]:.4f} (D1 theory {D1:.4f})', flush=True)
out['divergence'] = dict(D0=float(D0), Q0=float(Qgain0), D1=float(D1), rows=rows)

# ===================== (2) самоподобное решение, m тел =====================
def sphere_moment(beta, m):
    """<prod n_i^{2 beta_i}> on S^{m-1} = prod (2b-1)!! / prod_{i<|beta|}(m+2i)"""
    from math import prod
    def dfact(k): return 1 if k <= 0 else k*dfact(k-2)
    s = sum(beta); num = prod(dfact(2*b-1) for b in beta); den = prod(m+2*i for i in range(s))
    return Fraction(num, den)
def psi_series(m, K, gamma4, c2=Fraction(1), c3=Fraction(0)):
    """Psi(x)=1+sum c_k x^k, c_1=0, c_2=1 (нормировка); уравнение: -(gamma4/2) x Psi' = <prod Psi(x n_i^2)> - Psi"""
    from itertools import product
    c = [Fraction(0)]*(K+1); c[0] = Fraction(1); c[2] = c2; c[3] = c3
    for k in range(4, K+1):
        # collect coefficient of x^k in <prod_i Psi(x n_i^2)> excluding terms with c_k itself
        tot = Fraction(0)
        # partitions of k into m parts (each part 0 or >=2), using c's
        def rec(i, rem, parts):
            nonlocal tot
            if i == m:
                if rem == 0:
                    if k in parts and parts.count(k) == 1 and sum(1 for p in parts if p) == 1: return  # the c_k term
                    coef = Fraction(1)
                    for p in parts: coef *= c[p]
                    if coef == 0: return
                    tot += coef*sphere_moment(parts, m)
                return
            for p in [0] + list(range(2, rem+1)):
                rec(i+1, rem-p, parts+[p])
        rec(0, k, [])
        # equation: -(g4/2) k c_k = m <n1^{2k}> c_k + tot - c_k
        lin = m*sphere_moment([k]+[0]*(m-1), m) - 1 + Fraction(gamma4)*k/2
        c[k] = -tot/lin if lin != 0 else Fraction(0)
    return c
g4 = {2: Fraction(1, 4), 3: Fraction(2, 5), 4: Fraction(1, 2)}
series = {}
for m in (2, 3, 4):
    c = psi_series(m, 24, g4[m]); series[m] = [float(x) for x in c]
    print(f'm={m}: c =', [str(x) for x in c[:9]], flush=True)
# m=2 должно дать (1-x)e^x = 1 - x^2/2 - x^3/3 - x^4/8 ... в нормировке c2=1: Psi=(1+ (-2x)... проверка: (1-y)e^y с y=-2x? нет: (1-y)e^y = 1 - y^2/2 - y^3/3 - ..., значит c2=-1/2 при y=x; нормируем c2=1 => y = i*sqrt2 x? Просто сравним отношение c_k/c_2^{k/2}
import math
ref = [1, 0, -0.5, -1/3, -1/8, -1/30, -1/144, -1/840]   # (1-y)e^y coefficients
ck = psi_series(2, 10, g4[2], c2=Fraction(-1, 2), c3=Fraction(-1, 3))
print('Kac with c2=-1/2,c3=-1/3 (resonant free parameter):', [str(x) for x in ck[:8]], ' (1-x)e^x:', ref)
# при масштабировании y = s x: c_k = ref_k s^k; s^2 = -2 => c3 = -1/3 s^3 = -1/3 * (-2)^{3/2}: комплексно -> знак c2 отрицателен физически

# радиус сходимости (оценка по отношениям)
for m in (2, 3, 4):
    cs = np.array(series[m]); nz = [k for k in range(4, 25) if cs[k] != 0]
    rat = [abs(cs[k]/cs[k-2])**0.5 if cs[k-2] != 0 else np.nan for k in nz]
    print(f'm={m} |c_k/c_(k-2)|^(1/2) tail:', np.round(rat[-4:], 4))
out['series'] = {str(m): series[m] for m in series}
# положительность f для m=3: phi = e^{-k^2/2} Psi(alpha k^2) с c2 -> -1/2 (физический знак, как у Каца), обратное Фурье
def f_from_psi(cs, alpha, kmax=12, nk=4001):
    k = np.linspace(-kmax, kmax, nk); x = alpha*k**2
    Psi = sum(cs[j]*(-0.5)**(j//2)*x**j if j % 2 == 0 else 0 for j in range(len(cs)))  # c2 = -1/2 нормировка: c_k * (c2)^{k/2}
    phi = np.exp(-k**2/2)*Psi
    fv = np.array([np.trapezoid(phi*np.cos(k*v), k)/(2*np.pi) for v in V])
    return fv, phi
pos = {}
for m in (2, 3):
    for alpha in (0.05, 0.1, 0.15, 0.2, 0.3):
        fv, phi = f_from_psi(series[m], alpha)
        m4 = H*np.sum(V**4*fv); mass = H*np.sum(fv)
        pos[f'{m}_{alpha}'] = dict(minf=float(fv.min()), mass=float(mass), m4=float(m4))
        print(f'm={m} alpha={alpha}: min f={fv.min():.2e} mass={mass:.4f} m4={m4:.4f}', flush=True)
out['positivity'] = pos
# проверка a8/a4^2 для m=3 (предсказание -(15/2)sqrt(2/35))
fv, _ = f_from_psi(series[3], 0.1)
a4 = (H*np.sum(V**4*fv) - 3)/np.sqrt(24); m6 = H*np.sum(V**6*fv); m8 = H*np.sum(V**8*fv); m4 = H*np.sum(V**4*fv)
a8 = (m8 - 28*m6 + 210*m4 - 315)/np.sqrt(40320)
print('m=3 self-similar: a8/a4^2 =', a8/a4**2, ' predicted', -(15/2)*np.sqrt(2/35))
out['a8_over_a4sq'] = dict(num=float(a8/a4**2), pred=float(-(15/2)*np.sqrt(2/35)))

# ===================== (3) граница глобальной константы =====================
M = maxw(V)
ratios = []
for a in np.linspace(0.02, 0.8, 14):
    f = np.maximum(M*(1 + a*hn(4, V)), 0); ratios.append(('a4', float(a), float(D_exact(f)/(2*E_rel(f)))))
for a in np.linspace(0.02, 0.5, 8):
    f = np.maximum(M*(1 + a*hn(6, V)), 0); ratios.append(('a6', float(a), float(D_exact(f)/(2*E_rel(f)))))
for al in np.linspace(0.02, 0.333, 12):
    f = bkw(V, al); ratios.append(('bkw', float(al), float(D_exact(f)/(2*E_rel(f)))))
# смеси двух гауссиан с энергией 1 (широкое семейство)
for w in (0.1, 0.3, 0.5):
    for s1 in (0.3, 0.5, 0.7):
        s2 = np.sqrt((1 - (1-w)*s1**2)/w)
        f = (1-w)*maxw(V, s1**2) + w*maxw(V, s2**2); ratios.append((f'mix w={w}', float(s1), float(D_exact(f)/(2*E_rel(f)))))
for r_ in ratios: print(r_, flush=True)
out['ratios'] = ratios
print('inf ratio =', min(r_[2] for r_ in ratios))

# ===================== (4) термостат и скользящие столкновения =====================
mu = 0.7
# термостат: <h_n, T h_n> где (T phi)(u) = ∫dw dθ [phi(u')M(w') - phi(u)M(w)] линеаризовано около M: даёт mu(<cos^n>-1)
xg, wg = hermegauss(30); wg = wg/np.sqrt(2*np.pi)
spec_th = {}
for n in (1, 2, 3, 4, 6):
    # eigenvalue of thermostat part on He_n: ∫ M(w) He_n(u c - w s) dw dθ - He_n(u)
    u = 1.1; acc = 0.0
    for t in TH:
        acc += np.sum(wg*hn(n, u*np.cos(t) - xg*np.sin(t)))/M_TH
    lam_th = mu*(acc/hn(n, u) - 1)
    spec_th[n] = (float(-lam_th), float(mu*(1 - np.mean(np.cos(TH)**n))))
print('thermostat gamma_n^(th) numeric/theory:', spec_th)
out['thermo'] = spec_th
# скользящие: b(θ) = гауссиан ширины sig в θ (и -θ), R_D для f_alpha, alpha=0.3
def RD_grazing(f, sig, nth=64):
    th = np.linspace(-4*sig, 4*sig, nth); b = np.exp(-th**2/(2*sig**2)); b /= b.sum()
    lf = logf(f); tot_e = 0.0; tot_q = 0.0
    for t, bw in zip(th, b):
        vp = U*np.cos(t) - VS*np.sin(t); vsp = U*np.sin(t) + VS*np.cos(t)
        fp, fsp = interp(f, vp), interp(f, vsp)
        W = np.sqrt(np.maximum(f[:, None]*f[None, :]*fp*fsp, 0)); ok = W > 0
        r = np.where(ok, 0.5*(logf(fp) + logf(fsp) - lf[:, None] - lf[None, :]), 0)
        tot_e += bw*np.sum(W*r*np.sinh(r)); tot_q += bw*np.sum(W*r*r)
    return tot_e/tot_q
graz = {}
f = bkw(V, 0.3)
for sig in (1.0, 0.5, 0.25, 0.125, 0.0625):
    graz[sig] = float(RD_grazing(f, sig)); print(f'grazing sig={sig}: R_D-1 = {graz[sig]-1:.5f}', flush=True)
out['grazing'] = {str(k): v for k, v in graz.items()}

# ===================== (5) немаксвелловское ядро =====================
def D2_kernel(f, beta):
    """(lam/4) ∫ sqrt(B B') W Δ[log f]^2,  B=|v-v*|^beta"""
    lf = logf(f); tot = 0.0
    for it, t in enumerate(TH):
        vp, vsp = VP[it], VSP[it]
        fp, fsp = interp(f, vp), interp(f, vsp)
        W = np.sqrt(np.maximum(f[:, None]*f[None, :]*fp*fsp, 0)); ok = W > 0
        B = np.abs(U - VS)**beta; Bp = np.abs(vp - vsp)**beta
        D = np.where(ok, logf(fp) + logf(fsp) - lf[:, None] - lf[None, :], 0)
        tot += np.sum(np.sqrt(B*Bp)*W*D*D)
    return 0.25*LAM*WTH*H*H*tot
cbeta = {}
for beta in (0.0, 0.5, 1.0):
    vals = {}
    for a in (0.01, 0.02, 0.04):
        f = M*(1 + a*hn(4, V)); vals[a] = D2_kernel(f, beta)
    g4b = None
    # gamma_4^(beta) как предел D/a^2 (экстраполяция по двум точкам)
    q = {a: vals[a]/a**2 for a in vals}
    g4b = (4*q[0.01] - q[0.02])/3   # линейная экстраполяция в a->0 по (0.01,0.02)
    c_est = {a: float((1 - q[a]/g4b)/a) for a in vals}
    cbeta[beta] = dict(gamma4=float(g4b), c_est=c_est)
    print(f'beta={beta}: gamma4^(beta)={g4b:.5f}  c_est={c_est}  (Maxwell: 3sqrt6={3*np.sqrt(6):.4f})', flush=True)
out['cbeta'] = {str(k): v for k, v in cbeta.items()}

json.dump(out, open('../data/closeA.json', 'w'), indent=1)
import sys; sys.exit(0)
# ===================== (6) геодезические: схема f*xi_dot, Галёркин по xi =====================
DEG = [1, 3, 4, 5, 6, 7, 8, 9, 10]
P = np.stack([hn(k, V) for k in DEG], 1)            # базис для xi
def geo_rhs(f, cvec):
    xi = P @ cvec
    K = Konsager(f); fdot = K @ xi
    W = Wgeo(f); D = Delta(xi)
    rhs_fxidot = -(LAM/4)*WTH*H*np.sum(W*D*D, axis=(0, 2))     # = f * xi_dot (гладко)
    # Галёркин: sum_k cdot_k <p_j, f p_k> = <p_j, f xi_dot>
    G = (P.T*(f*H)) @ P; b = P.T @ (rhs_fxidot*H)
    cdot = np.linalg.solve(G + 1e-12*np.eye(len(DEG)), b)
    return fdot, cdot, 0.5*H*np.sum(xi*fdot)
ag = 0.30; fg = bkw(V, ag); Kg = Konsager(fg); Qg = Q_coll(fg)
# начальная скорость = Q (как у потока): xi in span(P): min || K P c - Q ||
c0, *_ = np.linalg.lstsq(Kg @ P, Qg, rcond=None)
f, cvec, t, dt = fg.copy(), c0.copy(), 0.0, 0.01
geo_log = []
E0 = geo_rhs(f, cvec)[2]
while t < 3.0 - 1e-9:
    k1 = geo_rhs(f, cvec); k2 = geo_rhs(f + dt/2*k1[0], cvec + dt/2*k1[1])
    k3 = geo_rhs(f + dt/2*k2[0], cvec + dt/2*k2[1]); k4 = geo_rhs(f + dt*k3[0], cvec + dt*k3[1])
    f = f + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); cvec = cvec + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]); t += dt
    if abs(t/0.25 - round(t/0.25)) < 1e-9:
        Eg = geo_rhs(f, cvec)[2]; ff = bkw(V, ag*np.exp(-t/8))
        geo_log.append(dict(t=float(t), speed2=float(Eg), speed2_rel=float(Eg/E0), minf=float(f.min()), mass=float(H*f.sum()),
                            energy=float(H*np.sum(V**2*f)), l2_geo_flow=float(np.sqrt(H*np.sum((f-ff)**2))), l2_flow_move=float(np.sqrt(H*np.sum((ff-fg)**2)))))
        print(geo_log[-1], flush=True)
out['geodesic'] = geo_log
np.save('../data/geo_full_f.npy', f)
json.dump(out, open('../data/closeA.json', 'w'), indent=1)
print('done')
