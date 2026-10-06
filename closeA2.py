import numpy as np, json
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
exec(open('kac_geom.py').read().split('results = {}')[0])
out={}
# ---- (1) точная D(alpha) = -∫ log f_alpha(v) Q(v) dv с адаптивной квадратурой ----
def f_an(v,a):
    K=1-2*a; return np.exp(-v**2/(2*K))/np.sqrt(2*np.pi*K)*((K-a)/K+a*v**2/K**2)
def D_quad(a):
    Q=Q_coll(bkw(V,a)); Qs=CubicSpline(V,Q)
    g=lambda v: -np.log(max(f_an(v,a),1e-300))*Qs(v)
    val=0.0
    for (lo,hi) in [(-7,-1),(-1,-0.05),(-0.05,0.05),(0.05,1),(1,7)]:
        val+=quad(g,lo,hi,limit=200)[0]
    return val
D0=D_quad(1/3-1e-14)
Q0=Q_coll(bkw(V,1/3-1e-12)); Qs0=CubicSpline(V,Q0); Qg0=float(Qs0(0.0))
D1=2*np.pi*Qg0*np.sqrt(3.0)
print(f'D0={D0:.6f} Q(0)={Qg0:.6f} D1={D1:.6f}')
rows=[]
for eps in (0.04,0.02,0.01,0.005,0.0025,0.00125,0.000625):
    a=1/3-eps; de=eps*0.1
    D=D_quad(a); Dp=D_quad(a+de); Dm=D_quad(a-de)
    kap=(a/16)*(np.log(Dp)-np.log(Dm))/(2*de); kap_as=(a/16)*D1/(2*np.sqrt(eps)*D)
    rows.append((eps,D,kap,kap_as,(D0-D)/np.sqrt(eps)))
    print(f'eps={eps:.6f} D={D:.6f} kappa={kap:.4f} kappa_asymp={kap_as:.4f} (D0-D)/sqrt(eps)={(D0-D)/np.sqrt(eps):.4f}',flush=True)
out['div']=dict(D0=D0,Q0=Qg0,D1=D1,rows=rows)
# ---- (5) c_beta через квадратуру Гаусса--Эрмита ----
from numpy.polynomial.hermite_e import hermeval, hermegauss
from math import factorial
xg,wg=hermegauss(24); wg=wg/np.sqrt(2*np.pi)
X,Y=np.meshgrid(xg,xg,indexing='ij'); X,Y=X.ravel(),Y.ravel(); WW=np.outer(wg,wg).ravel()
th=2*np.pi*(np.arange(48)+0.5)/48
h4=lambda x: hermeval(x,[0,0,0,0,1])/np.sqrt(24)
def D2b(a,beta):
    fr=1+a*h4(X); frY=1+a*h4(Y); tot=0.0
    for t in th:
        xp=X*np.cos(t)-Y*np.sin(t); yp=X*np.sin(t)+Y*np.cos(t)
        frp=1+a*h4(xp); fryp=1+a*h4(yp)
        W=np.sqrt(np.maximum(fr*frY*frp*fryp,1e-300)); r=0.5*(np.log(frp)+np.log(fryp)-np.log(fr)-np.log(frY))
        B=np.abs(X-Y)**beta; Bp=np.abs(xp-yp)**beta
        tot+=np.sum(WW*np.sqrt(B*Bp)*W*(2*r)**2)
    return 0.25*tot/len(th)
cb={}
for beta in (0.0,0.5,1.0,2.0):
    q={a:D2b(a,beta)/a**2 for a in (0.0025,0.005,0.01,0.02)}
    # gamma4^(beta) = lim q; Richardson: q(a)=g - g c a + O(a^2)
    g=2*q[0.0025]-q[0.005]; g2=2*q[0.005]-q[0.01]
    c=(1-q[0.01]/g)/0.01; c2=(1-q[0.005]/g)/0.005
    cb[beta]=dict(gamma4=float(g),gamma4_alt=float(g2),c_from_0p01=float(c),c_from_0p005=float(c2),q={str(k):float(v) for k,v in q.items()})
    print(f'beta={beta}: gamma4^(beta)={g:.5f} ({g2:.5f})  c={c2:.4f} (a=0.005), {c:.4f} (a=0.01)   Maxwell 3sqrt6={3*np.sqrt(6):.4f}',flush=True)
out['cbeta']=cb
json.dump(out,open('../data/closeA2.json','w'),indent=1)
