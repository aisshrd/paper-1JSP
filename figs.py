import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
exec(open('kac_geom.py').read().split('results = {}')[0])
R = json.load(open('../data/results.json'))
rows = R['bkw_rows']; A = np.array([r['a'] for r in rows])

# 1. профили BKW
plt.figure(figsize=(5, 3.2))
for a in [0.1, 0.2, 0.3, 1/3]:
    plt.plot(V, bkw(V, a), label=fr'$\alpha={a:.3g}$')
plt.plot(V, maxw(V), 'k--', label='Максвеллиан')
plt.xlim(-4, 4); plt.xlabel('$v$'); plt.ylabel('$f_\\alpha(v)$'); plt.legend(); plt.tight_layout()
plt.savefig('../figures/fig_bkw.png')

# 2. Q vs точное
a0 = 0.30; f0 = bkw(V, a0)
plt.figure(figsize=(5, 3.2))
plt.plot(V, Q_coll(f0), label='$Q(f_\\alpha)$, численно')
plt.plot(V, (-a0/8)*dbkw_da(V, a0), '--', label=r'$\partial_t f_\alpha$, точное решение')
plt.plot(V, DPsi_star(f0, -logf(f0)), ':', label=r'$D_\xi\Psi^*(f,-DE)$')
plt.xlim(-4, 4); plt.xlabel('$v$'); plt.legend(); plt.tight_layout()
plt.savefig('../figures/fig_Q.png')

# 3. спектр Эрмита
sp = R['hermite_spectrum']
ns = sorted(int(k) for k in sp)
plt.figure(figsize=(5, 3.0))
plt.bar(np.array(ns)-0.18, [sp[str(n)][1] for n in ns], width=0.36, label=r'$\gamma_n=1-2\langle\cos^n\theta\rangle$')
plt.bar(np.array(ns)+0.18, [sp[str(n)][0] for n in ns], width=0.36, label='численно')
plt.xlabel('$n$'); plt.ylabel(r'$\gamma_n$'); plt.legend(); plt.tight_layout()
plt.savefig('../figures/fig_spec.png')

# 4. диссипации и отношения
Dx = np.array([r['D_exact'] for r in rows]); Dq = np.array([r['D_quad'] for r in rows])
Ps = np.array([r['Psi_star'] for r in rows]); Pp = np.array([r['Psi'] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
ax[0].semilogy(A, Dx, 'o-', label=r'$-dE/dt=\lambda\int W r\sinh r$')
ax[0].semilogy(A, Dq, 's--', label=r'квадратичная $\lambda\int W r^2$')
ax[0].semilogy(A, 1.5*A**4, 'k:', label=r'линейная теория $1.5\,\alpha^4$')
ax[0].semilogy(A, Ps, '^-', ms=3, label=r'$\Psi^*(f,-DE)$'); ax[0].semilogy(A, Pp, 'v-', ms=3, label=r'$\Psi(f,\dot f)$')
ax[0].set_xlabel(r'$\alpha$'); ax[0].legend(fontsize=7)
ax[1].plot(A, Dx/Dq, 'o-', label=r'$R_D=D_{exact}/D_{quad}$')
ax[1].plot(A, [r['rrms'] for r in rows], 's--', label=r'$r_{rms}$ (вес $W$)')
ax[1].axhline(1.05, color='gray', lw=0.8, ls=':')
ax[1].set_xlabel(r'$\alpha$'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('../figures/fig_diss.png')

# 5. kappa_eff
kx = np.array(R['kappa_exact']); kq = np.array(R['kappa_quad'])
plt.figure(figsize=(5, 3.0))
plt.plot(A[1:], kx[1:], 'o-', label=r'$\kappa_{eff}$ (точная диссипация)')
plt.plot(A[1:], kq[1:], 's--', label=r'$\kappa_{eff}$ (квадратичная)')
plt.axhline(0.25, color='k', ls=':', label=r'линейная щель $\gamma_4=1/4$')
plt.xlabel(r'$\alpha$'); plt.ylabel(r'$\kappa_{eff}$'); plt.legend(); plt.tight_layout()
plt.savefig('../figures/fig_kappa.png')

# 6. длина и нормы скорости (rcond=1e-8)
sp_geo, sp_log = [], []
for a in A:
    f = bkw(V, a); Qf = Q_coll(f)
    K = Konsager(f); xi, *_ = np.linalg.lstsq(K, Qf, rcond=1e-8); sp_geo.append(H*np.sum(xi*Qf))
    KL = Konsager(f, Lambda_logmean(f)); xi, *_ = np.linalg.lstsq(KL, Qf, rcond=1e-8); sp_log.append(H*np.sum(xi*Qf))
sp_geo, sp_log = np.array(sp_geo), np.array(sp_log)
json.dump(dict(a=A.tolist(), speed2_geo=sp_geo.tolist(), speed2_log=sp_log.tolist(), D_exact=Dx.tolist(), D_quad=Dq.tolist()),
          open('../data/speeds.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
ax[0].plot(A, R['ell'], 'o-', label=r'$\ell(\alpha)=\int_0^\alpha\sqrt{g}\,d\alpha$')
ax[0].plot(A, R['ell_lin'], 'k:', label=r'$2\sqrt{6}\,\alpha^2$ (линейная)')
ax[0].set_xlabel(r'$\alpha$'); ax[0].set_ylabel('длина'); ax[0].legend()
m = A >= 0.14
ax[1].plot(A[m], sp_geo[m]/Dx[m], 'o-', label=r'$\|Q\|^2_{geo}/D_{exact}$')
ax[1].plot(A[m], sp_log[m]/Dx[m], 's-', label=r'$\|Q\|^2_{log}/D_{exact}$ (Эрбар)')
ax[1].axhline(1, color='k', ls=':'); ax[1].set_xlabel(r'$\alpha$'); ax[1].legend()
plt.tight_layout(); plt.savefig('../figures/fig_len.png')

# 7. кривизна и геодезические
G = np.load('../data/Ggrid.npy'); g4 = np.load('../data/agrid4.npy'); g6 = np.load('../data/agrid6.npy')
Kg = np.load('../data/Kgauss.npy')
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
im = ax[0].pcolormesh(g4[1:-1], g6[1:-1], Kg[1:-1, 1:-1].T, shading='nearest', cmap='RdBu_r', vmin=-8, vmax=8)
plt.colorbar(im, ax=ax[0], label='гауссова кривизна')
ax[0].set_xlabel('$a_4$'); ax[0].set_ylabel('$a_6$')
import glob
for fn in sorted(glob.glob('../data/geo_+*.npy')):
    geo = np.load(fn); ax[1].plot(geo[:, 0], geo[:, 1], 'b-', lw=1.2)
    a0_ = geo[0]; ts = np.linspace(0, 40, 400)
    ax[1].plot(a0_[0]*np.exp(-0.25*ts), a0_[1]*np.exp(-0.375*ts), 'r--', lw=1)
ax[1].plot([], [], 'b-', label='геодезическая'); ax[1].plot([], [], 'r--', label='линеаризованный поток')
ax[1].set_xlabel('$a_4$'); ax[1].set_ylabel('$a_6$'); ax[1].legend(); ax[1].set_xlim(-0.01, 0.36); ax[1].set_ylim(-0.01, 0.25)
plt.tight_layout(); plt.savefig('../figures/fig_curv.png')
print('figs done')
