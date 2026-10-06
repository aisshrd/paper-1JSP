"""
Модели за пределами Каца: численная часть.
 (1) неупругая модель Каца (Пульвиренти--Тосcани), p=1: энергия, нарушение симметрии H,
     разложение дрейфа на "градиентную" часть -1/2 C DE и остаток;
 (2) модель Карлемана: точные формулы (для графиков).
"""
import numpy as np, json
from numpy.polynomial.hermite_e import hermeval
from math import factorial
exec(open('kac_geom.py').read().split('results = {}')[0])

# ---------- общая машинерия для произвольного отображения столкновений ----------
def make_maps(pexp):
    """v' = v c|c|^p - v* s|s|^p ; v*' = v s|s|^p + v* c|c|^p"""
    C = np.cos(TH); S = np.sin(TH)
    cp = C * np.abs(C)**pexp; sp = S * np.abs(S)**pexp
    VPp = np.stack([U * cp[t] - VS * sp[t] for t in range(M_TH)])
    VSPp = np.stack([U * sp[t] + VS * cp[t] for t in range(M_TH)])
    return VPp, VSPp

class Model:
    def __init__(self, pexp):
        self.pexp = pexp
        self.VP, self.VSP = make_maps(pexp)
        self.I0P, self.W0P, self.W1P = interp_idx(self.VP)
        self.I0S, self.W0S, self.W1S = interp_idx(self.VSP)
        self.beta = np.mean(np.abs(np.cos(TH))**(2*pexp+2) + np.abs(np.sin(TH))**(2*pexp+2))
    def cv(self, g):
        gp = self.W0P * g[self.I0P] + self.W1P * g[self.I0P + 1]
        gsp = self.W0S * g[self.I0S] + self.W1S * g[self.I0S + 1]
        return gp, gsp
    def Delta(self, xi):
        xp, xsp = self.cv(xi)
        return xp + xsp - xi[None, :, None] - xi[None, None, :]
    def deposit(self, A):
        """Q-приход: (lam/2) ∫∫ A(v,v*,θ) [δ(v'-u)+δ(v*'-u)] -> вектор по u (транспонированная интерполяция)"""
        out = np.zeros(N)
        for (I0, W0, W1) in ((self.I0P, self.W0P, self.W1P), (self.I0S, self.W0S, self.W1S)):
            np.add.at(out, I0.ravel(), (A * W0).ravel())
            np.add.at(out, I0.ravel() + 1, (A * W1).ravel())
        return out
    def Q(self, f):
        A = LAM * WTH * H * f[None, :, None] * f[None, None, :]
        gain = 0.5 * self.deposit(A)
        loss = LAM * f * (H * f.sum())
        return gain - loss
    def H_LD(self, f, p):
        """H на множестве столкновений, где все четыре f > 1e-10 (иначе log f не определён)"""
        D = self.Delta(p)
        fp, fsp = self.cv(f)
        mask = (f[None, :, None] > 1e-10) & (f[None, None, :] > 1e-10) & (fp > 1e-10) & (fsp > 1e-10)
        A = f[None, :, None] * f[None, None, :] * (np.exp(np.where(mask, D, 0.0)) - 1)
        return 0.5 * LAM * WTH * H * H * np.sum(np.where(mask, A, 0.0))
    def DPsi_sym(self, f):
        """D_xi Psi*_sym(f,-DE) с W=sqrt(f f* f' f*'), без использования симметрии обращения:
           (lam/2)[ deposit(W sinh(Δ/2)) - 2∫ W sinh(Δ/2) dv* ],  Δ/2 = -r"""
        fp, fsp = self.cv(f)
        W = np.sqrt(np.maximum(f[None, :, None] * f[None, None, :] * fp * fsp, 0.0))
        ok = W > 0
        r = np.where(ok, 0.5 * (logf(fp) + logf(fsp) - logf(f)[None, :, None] - logf(f)[None, None, :]), 0.0)
        A = np.where(ok, W * np.sinh(-r), 0.0) * LAM * WTH * H
        return 0.5 * self.deposit(A) - A.sum(axis=(0, 2))
    def C_matrix(self, f):
        """ковариация шума: <p, C p> = lam ∫∫ f f* Δ[p]^2 ;  C = Hess_p H(f,0)"""
        w = f[None, :, None] * f[None, None, :] * np.ones((M_TH, 1, 1))
        # <p,Cp> = lam ∫ ff* Δ^2  => C = 2 * K-конструкция с весом ff* и множителем lam/... :
        # K_w из kac_geom даёт <xi,K xi> = (lam/4) ∫ w Δ^2, поэтому C = 4 K_w
        return 2 * self._Kw(w)
    def _Kw(self, w):
        Kmat = np.zeros((N, N))
        diag = -LAM * WTH * H * w.sum(axis=(0, 2))
        Kmat[np.arange(N), np.arange(N)] -= diag
        Kmat += LAM * WTH * H * w.sum(axis=0)
        for (I0, W0, W1) in ((self.I0P, self.W0P, self.W1P), (self.I0S, self.W0S, self.W1S)):
            for t in range(M_TH):
                c = -LAM * WTH * H * w[t]
                rows = np.repeat(np.arange(N), N)
                np.add.at(Kmat, (rows, I0[t].ravel()), (c * W0[t]).ravel())
                np.add.at(Kmat, (rows, I0[t].ravel() + 1), (c * W1[t]).ravel())
        return Kmat

def rk4(model, f, dt):
    k1 = model.Q(f); k2 = model.Q(f + dt/2*k1); k3 = model.Q(f + dt/2*k2); k4 = model.Q(f + dt*k3)
    return f + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def He(n):
    c = np.zeros(n + 1); c[n] = 1
    return hermeval(V, c) / np.sqrt(factorial(n))

out = {}
# ---- проверка: упругая модель (p=0) воспроизводит Q из kac_geom ----
el = Model(0.0)
f0 = bkw(V, 0.25)
out['check_Q_elastic'] = float(np.max(np.abs(el.Q(f0) - Q_coll(f0))) / np.max(np.abs(Q_coll(f0))))

# ---- неупругая, p=1 ----
inel = Model(1.0)
out['beta_p1'] = float(inel.beta); out['energy_rate_theory'] = float(LAM * (1 - inel.beta))
f = bkw(V, 0.0 + 1e-9); f = maxw(V)          # старт из максвеллиана
dt, T = 0.02, 6.0
ts, en, kurt, symdef, nongrad, nongrad_el = [], [], [], [], [], []
p_test = 0.3 * He(4) * np.exp(-V**2/8)                         # пробный импульс для проверки симметрии
t = 0.0
while t <= T + 1e-9:
    E2 = H * np.sum(V**2 * f); E4 = H * np.sum(V**4 * f)
    ts.append(t); en.append(E2); kurt.append(E4 / E2**2)
    if abs(t / 0.5 - round(t / 0.5)) < 1e-9:
        # нарушение симметрии обращения времени
        Hp = inel.H_LD(f, p_test); Hs = inel.H_LD(f, logf(f) - p_test)
        symdef.append((t, float(Hp), float(Hs)))
        # разложение дрейфа: Q = -1/2 C DE + Nrem
        G = inel.DPsi_sym(f); Qf = inel.Q(f)
        nongrad.append((t, float(np.linalg.norm(Qf - G) / np.linalg.norm(Qf)), float(H*np.sum(V**2*Qf)), float(H*np.sum(V**2*G))))
        print(f"t={t:.2f} E2={E2:.4f} kurt={E4/E2**2:.3f} H(p)={Hp:.4e} H(logf-p)={Hs:.4e} |Q-G|/|Q|={nongrad[-1][1]:.3f}", flush=True)
    f = rk4(inel, f, dt); t += dt
out['inelastic'] = dict(t=ts, energy=en, kurtosis=kurt, symdef=symdef, nongrad=nongrad)
np.save('../data/f_inel_final.npy', f)

# та же диагностика для упругой модели вдоль BKW (для сравнения)
for a in [0.1, 0.2, 0.3]:
    fa = bkw(V, a); G = el.DPsi_sym(fa); Qf = el.Q(fa)
    GW = Konsager(fa) @ (-(logf(fa) + 1))
    Hp = el.H_LD(fa, p_test); Hs = el.H_LD(fa, logf(fa) - p_test)
    nongrad_el.append((a, float(np.linalg.norm(Qf - G) / np.linalg.norm(Qf)), float(np.linalg.norm(Qf - GW) / np.linalg.norm(Qf)), float(Hp), float(Hs)))
    print('elastic', a, nongrad_el[-1], flush=True)
out['elastic_ref'] = nongrad_el
json.dump(out, open('../data/models_results.json', 'w'), indent=1)

# ---- графики ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
ts = np.array(ts); en = np.array(en)
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].semilogy(ts, en, label='численно'); ax[0].semilogy(ts, np.exp(-LAM*(1-inel.beta)*ts), 'k--', label=r'$e^{-\lambda(1-\langle\beta\rangle)t}$')
ax[0].set_xlabel('$t$'); ax[0].set_ylabel(r'$\langle v^2\rangle$'); ax[0].legend()
ax[1].plot(ts, kurt); ax[1].axhline(3, color='k', ls=':'); ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'$\langle v^4\rangle/\langle v^2\rangle^2$')
ff = np.load('../data/f_inel_final.npy'); s = np.sqrt(H*np.sum(V**2*ff))
ax[2].semilogy(V/s, ff*s, label=f'профиль при t={T}'); ax[2].semilogy(V, maxw(V), 'k--', label='гауссиан')
ax[2].set_xlim(-6, 6); ax[2].set_ylim(1e-6, 1); ax[2].set_xlabel(r'$v/\sqrt{\langle v^2\rangle}$'); ax[2].legend()
plt.tight_layout(); plt.savefig('../figures/fig_inel.png')

# Карлеман: точные формулы
d = np.linspace(1e-3, 0.999, 400); sig = 1.0
r = 2*np.arctanh(d)
D_ex = 2*sig*d*np.arctanh(d); D_q = 2*sig*(1-d**2)/4*r**2  # 2σ uv r^2, uv=(1-d^2)/4
kap = sig*(1 + d/((1-d**2)*np.arctanh(d)))
dist = np.arcsin(d)/np.sqrt(2*sig)
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].semilogy(d, D_ex, label=r'$D_{exact}=2\sigma\delta\,\mathrm{artanh}\,\delta$'); ax[0].semilogy(d, D_q, '--', label=r'$D_2=2\sigma uv\,r^2$')
ax[0].set_xlabel(r'$\delta=v-u$'); ax[0].legend(fontsize=7)
ax[1].plot(d, kap/sig); ax[1].axhline(2, color='k', ls=':'); ax[1].set_ylim(1.5, 8); ax[1].set_xlabel(r'$\delta$'); ax[1].set_ylabel(r'$\kappa_{eff}/\sigma$')
ax[2].plot(d, dist*np.sqrt(2*sig), label=r'$\sqrt{2\sigma}\,d(u,1/2)=\arcsin\delta$'); ax[2].plot(d, D_ex/D_q, '--', label=r'$R_D=\sinh r/r$')
ax[2].set_xlabel(r'$\delta$'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('../figures/fig_carleman.png')
print('done')
