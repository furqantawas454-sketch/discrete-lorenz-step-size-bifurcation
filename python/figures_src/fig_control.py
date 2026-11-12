import os
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
os.makedirs(FIG, exist_ok=True)
import numpy as np
from style import *
from vec import s,b
def coeffs(k1,k2,h,r):
    a2=-3+k1+h*(1+b+s)
    a1=3-2*k1+h*(-2+b*(-2+k1)+k1+k2-2*s)+b*h**2*(r+s)
    a0=-1+k1+2*b*h**3*(r-1)*s+h*(1+b-k1-b*k1-k2+s)-b*h**2*(k2*(r-2)+r-k1*r+s)
    return a2,a1,a0
def jury(k1,k2,h,r):
    a2,a1,a0=coeffs(k1,k2,h,r)
    return (1+a2+a1+a0>0)&(-(-1+a2-a1+a0)>0)&(np.abs(a0)<1)&(np.abs(a0**2-1)>np.abs(a0*a2-a1))
def G(X,h,r): x,y,z=X; return np.array([x+h*s*(y-x), y+h*(x*(r-z)-y), z+h*(x*y-b*z)])
def Ep(r): q=np.sqrt(b*(r-1)); return np.array([q,q,r-1])
def run(r,h,k1,k2,N=6000,warm=3000,radius=3.0):
    E=Ep(r); X=np.array([1.,1.,1.]); xs=[]; on=None
    for n in range(N):
        u=0.0
        if on is None and n>=warm and np.linalg.norm(X-E)<radius: on=n
        if on is not None: u=-k1*(X[0]-E[0])-k2*(X[1]-E[1])
        X=G(X,h,r); X[0]+=u; xs.append(X[0])
    return np.array(xs),on,np.linalg.norm(X-E)
cases=[(10,0.07,0.5,0.5,6000),(28,0.01,0.5,0.0,12000)]
fig,ax=plt.subplots(2,2,figsize=(6.6,5.1),gridspec_kw={'width_ratios':[1,1.35]})
for i,(r,h,k1,k2,N) in enumerate(cases):
    K1,K2=np.meshgrid(np.linspace(-1.2,2.2,800),np.linspace(-2.5,2.5,800))
    M=jury(K1,K2,h,r).astype(float)
    a=ax[i,0]; a.contourf(K1,K2,M,levels=[0.5,1.5],colors=[SKY],alpha=0.55); a.contour(K1,K2,M,levels=[0.5],colors=BLUE,linewidths=1.0)
    a.plot(0,0,'s',c='k',ms=4.5,mfc='white',mew=1); a.text(-0.1,-0.38,'no control',fontsize=7.5,ha='right')
    a.plot(k1,k2,'o',c=VERM,ms=6,mec='white',mew=0.8); a.text(k1+0.08,k2+0.15,'$(%.1f,%.1f)$'%(k1,k2),fontsize=7.5,color=VERM)
    cx,cy=[(1.15,1.45),(1.15,-0.25)][i]; a.text(cx,cy,'stable',ha='center',fontsize=8.5,color=BLUE,style='italic')
    a.set_xlabel('$k_1$'); a.set_ylabel('$k_2$'); a.set_title('$\\rho=%g$, $h=%g$'%(r,h),fontsize=9.5); panel(a,'(%s)'%'ac'[i])
    xs,on,dist=run(r,h,k1,k2,N=N)
    a=ax[i,1]; n=np.arange(len(xs))
    a.axvspan(on,len(xs),color=REG['stable'],lw=0,zorder=0)
    a.plot(n[:on+1],xs[:on+1],c=ORANGE,lw=0.4,label='uncontrolled'); a.plot(n[on:],xs[on:],c=BLUE,lw=1.0,label='controlled')
    a.axhline(Ep(r)[0],c='0.4',lw=0.6,ls=':'); a.axvline(on,c=VERM,lw=0.9,ls=(0,(4,2)))
    a.text(on,1.01,'control on, $n=%d$'%on,transform=a.get_xaxis_transform(),ha='center',va='bottom',fontsize=8,color=VERM)
    a.text(len(xs)*0.99,Ep(r)[0]+0.6,'$x^*$',ha='right',fontsize=8.5,color='0.3')
    a.set_xlim(0,len(xs)); a.set_xlabel('$n$'); a.set_ylabel('$x_n$'); panel(a,'(%s)'%'bd'[i])
    if i==0: a.legend(loc='upper right',ncol=1,fontsize=7.5,bbox_to_anchor=(1.0,0.97))
    print(r,h,'switch-on',on,'final dist',dist,'chosen gains inside', bool(jury(k1,k2,h,r)))
fig.tight_layout(); fig.savefig(os.path.join(FIG,'ex5_control.pdf'),**SAVE)
