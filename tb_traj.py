import numpy as np, json
src = open('three_body.py').read().split('t0 = time.time()')[0]
exec(src)
grid4 = np.arange(0, 0.30+1e-9, 0.05); grid6 = np.arange(0, 0.20+1e-9, 0.05); da=0.05
Kg = np.load('../data/K3.npy'); G = np.load('../data/G3.npy')
a40, a60 = 0.30, 0.15
ts = np.linspace(0, 6, 13); traj = []
for tt in ts[::1]:
    a4 = a40*np.exp(-2*tt/5); a6 = a60*np.exp(-4*tt/7)
    _, Dex, Dq = metric_and_diss((a4, a6)); traj.append((float(tt), a4, a6, float(Dex), float(Dq))); print(traj[-1], flush=True)
traj = np.array(traj); kap = -0.5*np.gradient(np.log(traj[:,3]), traj[:,0]); kapq = -0.5*np.gradient(np.log(traj[:,4]), traj[:,0])
json.dump(dict(spec=spec, moment_eqs={k: str(v) for k,v in mom_eqs.items()}, m6t=str(m6t),
               traj=traj.tolist(), kappa=kap.tolist(), kappa_q=kapq.tolist(), K_interior=Kg[1:-1,1:-1].tolist(),
               G=G.tolist()), open('../data/three_body.json','w'), indent=1)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
im = ax[0].pcolormesh(grid4[1:-1], grid6[1:-1], Kg[1:-1, 1:-1].T, shading='nearest', cmap='RdBu_r', vmin=-8, vmax=8)
plt.colorbar(im, ax=ax[0], label='гауссова кривизна'); ax[0].set_xlabel('$a_4$'); ax[0].set_ylabel('$a_6$')
ax[0].plot(traj[:,1], traj[:,2], 'k.-', ms=3, label='точная траектория'); ax[0].legend(fontsize=7)
ax[1].semilogy(traj[:,0], traj[:,3], 'o-', label=r'$D_{exact}$'); ax[1].semilogy(traj[:,0], traj[:,4], 's--', label=r'$D_2$')
ax[1].semilogy(traj[:,0], traj[0,3]*np.exp(-0.8*traj[:,0]), 'k:', label=r'$e^{-2\gamma_4 t}$'); ax[1].set_xlabel('$t$'); ax[1].legend()
ax[2].plot(traj[:,0], kap, 'o-', label='точная'); ax[2].plot(traj[:,0], kapq, 's--', label='квадратичная'); ax[2].axhline(0.4, color='k', ls=':', label=r'$\gamma_4=2/5$')
ax[2].set_xlabel('$t$'); ax[2].set_ylabel(r'$\kappa_{eff}$'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('../figures/fig_3body.png'); print('done')
