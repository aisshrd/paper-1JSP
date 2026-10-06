import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
A=json.load(open('../data/closeA2.json')); B=json.load(open('../data/closeA.json')); C=json.load(open('../data/cbeta.json'))
fig,ax=plt.subplots(1,4,figsize=(13,3.1))
r=np.array(A['div']['rows']); eps=r[:,0]
ax[0].loglog(eps,r[:,2],'o-',label=r'$\kappa_{eff}$ (числ.)'); ax[0].loglog(eps,(1/96)*A['div']['D1']/A['div']['D0']/np.sqrt(eps),'k--',label=r'$\lambda D_1/(96 D_0)\,\varepsilon^{-1/2}$')
ax[0].set_xlabel(r'$\varepsilon=1/3-\alpha$'); ax[0].legend(fontsize=8)
x=np.linspace(0,6,300)
for m,ls in ((2,'--'),(3,'-'),(4,':')):
    cs=B['series'][str(m)]; Psi=sum(cs[j]*(-0.5)**(j//2)*x**j for j in range(0,len(cs),2)); ax[1].plot(x,Psi,ls,label=f'$m={m}$')
ax[1].plot(x,(1-x/2)*np.exp(x/2)*0+ (1-x)*np.exp(x)*0,alpha=0)  # placeholder
ax[1].set_xlabel('$x=\\alpha k^2$'); ax[1].set_ylabel(r'$\Psi_m(x)$, $c_2=-1/2$'); ax[1].set_ylim(-3,1.2); ax[1].legend()
sig=np.array([float(k) for k in B['grazing']]); rd=np.array([B['grazing'][k]-1 for k in B['grazing']])
ax[2].loglog(sig,rd,'o-',label='$R_D-1$'); ax[2].loglog(sig,rd[-1]*(sig/sig[-1])**2,'k:',label=r'$\propto\sigma_\theta^2$'); ax[2].set_xlabel(r'ширина ядра $\sigma_\theta$'); ax[2].legend()
bs=sorted(float(k) for k in C); ax[3].plot(bs,[C[str(b) if str(b) in C else k]['c'] for b,k in zip(bs,C)],'o-'); ax[3].axhline(3*np.sqrt(6),color='k',ls=':',label=r'$3\sqrt{6}$ (Максвелл)')
ax[3].set_xlabel(r'$\beta$ в $|v-v_*|^\beta$'); ax[3].set_ylabel(r'$c_\beta$'); ax[3].legend()
plt.tight_layout(); plt.savefig('../figures/fig_A5.png'); print('ok')
