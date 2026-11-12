import os
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
os.makedirs(FIG, exist_ok=True)
import numpy as np
from style import *
from coeffs import *
from scipy.optimize import brentq
rhoH=s*(s+b+3)/(s-b-1)
# critical rho values
disc=lambda r: (lambda L: np.max(np.abs(L.imag)))(lams(r))
rho1=brentq(lambda r: np.max(np.abs(lams(r).imag))-1e-9 if True else 0, 1.2,1.5) if False else None
from numpy.polynomial import polynomial as P
def discr(r):
    a,bb,c,d=1,s+b+1,b*(s+r),2*s*b*(r-1)
    return 18*a*bb*c*d-4*bb**3*d+bb**2*c**2-4*a*c**3-27*a**2*d**2
rho1=brentq(discr,1.01,2.0)
rsw=brentq(lambda r: hF(r)-hNS(r),2.5,5)
rG=brentq(lambda r: dNS(r)[0],7,10)
print('rho1',rho1,'rsw',rsw,'hsw',hF(rsw),'rG',rG,'hG',hNS(rG),'rhoH',rhoH)
# curves
r0=np.geomspace(0.1,0.999,300); h0=[min(4/(s+1+np.sqrt((s-1)**2+4*s*r)),2/b) for r in r0]
rA=np.linspace(1.001,rhoH-1e-4,1500); HF=np.array([hF(r) for r in rA]); HN=np.array([hNS(r) if r>rho1 else np.nan for r in rA])
HE=np.fmin(HF,np.nan_to_num(HN,nan=np.inf))
fig,ax=plt.subplots(2,1,figsize=(6.2,5.6),sharex=True,gridspec_kw={'height_ratios':[2.1,1],'hspace':0.1})
a=ax[0]
a.fill_between(r0,1e-4,h0,color=REG['stable'],lw=0); a.fill_between(rA,1e-4,HE,color=REG['stable'],lw=0)
a.axvspan(rhoH,45,color=REG['escape'],lw=0)
a.text(39,0.0011,'$E_\\pm$ unstable for every $h$',ha='center',va='bottom',rotation=90,fontsize=7.5,color='0.3')
a.text(0.3,0.03,'$E_0$ stable',ha='center',fontsize=8,color=BLUE); a.text(5,0.003,'$E_\\pm$ stable',ha='center',fontsize=8,color=BLUE)
a.plot(r0,h0,c=VERM,lw=1.6,ls=(0,(4,2)),label='flip at $E_0$ (subcritical)')
m1=rA<rsw; a.plot(rA[m1],HF[m1],c=GREEN,lw=1.8,label='flip at $E_\\pm$ (supercritical)')
a.plot(rA[~m1],HF[~m1],c=GREEN,lw=0.8,alpha=0.45)
m2=(rA>rsw)&(rA<rG); m3=rA>=rG; m4=(rA>rho1)&(rA<rsw)
a.plot(rA[m2],HN[m2],c=BLUE,lw=1.8,label='Neimark--Sacker (supercritical)')
a.plot(rA[m3],HN[m3],c=PURPLE,lw=1.8,ls=(0,(4,2)),label='Neimark--Sacker (subcritical)')
a.plot(rA[m4],HN[m4],c=BLUE,lw=0.8,alpha=0.45)
a.plot(rsw,hF(rsw),'D',c='k',ms=5,zorder=6); a.annotate('$\\rho_{sw}$',(rsw,hF(rsw)),xytext=(rsw*1.25,0.33),fontsize=8.5,arrowprops=dict(arrowstyle='-',lw=0.5,color='0.3'))
a.plot(rG,hNS(rG),'*',c='k',ms=9,zorder=6); a.annotate('$\\rho_G$ (GNS)',(rG,hNS(rG)),xytext=(rG*0.45,0.017),fontsize=8.5,arrowprops=dict(arrowstyle='-',lw=0.5,color='0.3'))
for xv,l in [(1,'$1$'),(rhoH,'$\\rho_H$')]:
    a.axvline(xv,c='0.4',lw=0.6,ls=':')
a.text(rhoH,1.02,'$\\rho_H$',transform=a.get_xaxis_transform(),ha='center',va='bottom',fontsize=8.5)
# example markers
ex=[(0.5,0.19002,'Ex 1'),(2,0.17790,'Ex 2'),(5,0.10282,'Ex 3'),(10,0.030955,'Ex 4')]
for (xr,hh,l) in ex:
    a.plot(xr,hh,'o',mfc='white',mec='k',ms=5,zorder=7); a.text(xr*(1.08 if l=='Ex 4' else 0.93),hh*(1.0 if l=='Ex 4' else 0.62),l,fontsize=7.5,ha=('left' if l=='Ex 4' else 'right'),va=('center' if l=='Ex 4' else 'baseline'))
for (xr,hh) in [(10,0.07),(28,0.01)]:
    a.plot(xr,hh,'s',mfc=ORANGE,mec='k',ms=5,zorder=7)
a.text(10.9,0.07,'Ex 5',fontsize=7.5,va='center'); a.text(28,0.0072,'Ex 5',fontsize=7.5,ha='center',va='top')
a.set_yscale('log'); a.set_ylim(3e-4,1.0); a.set_xscale('log'); a.set_xlim(0.1,45); a.set_ylabel('$h$')
a.legend(loc='lower left',fontsize=7.2)
panel(a,'(a)')
# panel b: coefficients
a=ax[1]
l0=[]
for r in r0:
    lm=(-(s+1)-np.sqrt((s-1)**2+4*s*r))/2; h1=2/abs(lm); l0.append(-(h1/b)*(s+lm)**2/(s*r+(s+lm)**2))
a.plot(r0,l0,c=VERM,lw=1.4,ls=(0,(4,2)),label='$\\ell_0$')
rF=np.linspace(1.01,rsw,200); a.plot(rF,[lF(r) for r in rF],c=GREEN,lw=1.6,label='$\\ell_F$')
rN=np.linspace(rsw,rhoH-0.02,600); dn=np.array([dNS(r)[0] for r in rN])
a.plot(rN[rN<rG],dn[rN<rG],c=BLUE,lw=1.6,label='$d$'); a.plot(rN[rN>=rG],dn[rN>=rG],c=PURPLE,lw=1.6,ls=(0,(4,2)))
a.axhline(0,c='0.3',lw=0.6); a.plot(rG,0,'*',c='k',ms=9)
a.set_yscale('symlog',linthresh=1e-5); a.set_ylabel('coefficient'); a.set_xlabel('$\\rho$')
a.axvspan(rhoH,45,color=REG['escape'],lw=0); a.axvline(1,c='0.4',lw=0.6,ls=':'); a.axvline(rhoH,c='0.4',lw=0.6,ls=':')
a.legend(loc='upper left',ncol=3,fontsize=7.5); panel(a,'(b)')
fig.savefig(os.path.join(FIG,'overview.pdf'),**SAVE)
# resonance check
v=[hNS(r)**2*(b*(s+r)-2*s*b*(r-1)*hNS(r)) for r in rN]; print('max h^2 m on NS-first branch',max(v))
print('lF range',min(lF(r) for r in rF),max(lF(r) for r in rF))
