import numpy as np, json
from numpy.polynomial.hermite_e import hermeval, hermegauss
xg,wg=hermegauss(28); wg=wg/np.sqrt(2*np.pi)
X,Y=np.meshgrid(xg,xg,indexing='ij'); X,Y=X.ravel(),Y.ravel(); WW=np.outer(wg,wg).ravel()
th=2*np.pi*(np.arange(64)+0.5)/64
h=lambda x: hermeval(x,[0,0,0,0,1])/np.sqrt(24)
res={}
for beta in (0.0,0.25,0.5,1.0,1.5,2.0):
    G4=AS=AX=0.0
    for t in th:
        xp=X*np.cos(t)-Y*np.sin(t); yp=X*np.sin(t)+Y*np.cos(t)
        w=WW*np.sqrt(np.abs(X-Y)**beta*np.abs(xp-yp)**beta)
        Dh=h(xp)+h(yp)-h(X)-h(Y); Dh2=h(xp)**2+h(yp)**2-h(X)**2-h(Y)**2; S=h(xp)+h(yp)+h(X)+h(Y)
        G4+=np.sum(w*Dh**2); AS+=np.sum(w*S*Dh**2); AX+=np.sum(w*Dh*Dh2)
    g4=0.25*G4/len(th); c=-(0.5*AS-AX)/G4
    res[beta]=dict(gamma4=float(g4),c=float(c),AS_over_AX=float(AS/AX))
    print(f'beta={beta}: gamma4^(beta)={g4:.5f}  c_beta={c:.5f}  A_S/A_X={AS/AX:.5f}',flush=True)
json.dump(res,open('../data/cbeta.json','w'),indent=1)
