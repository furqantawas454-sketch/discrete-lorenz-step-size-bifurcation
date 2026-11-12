"""vec.py -- vectorized (over h) bifurcation and Lyapunov scan used by the figure scripts."""
import numpy as np
s,b=10.0,8/3
def vstep(x,y,z,h,r):
    return x+h*s*(y-x), y+h*(x*(r-z)-y), z+h*(x*y-b*z)
def bif(r,hs,X0,tr=20000,keep=300,bound=1e4,mle_n=20000):
    """vectorised over h: returns list of x samples and MLE per h"""
    n=len(hs); h=np.array(hs,float)
    x=np.full(n,X0[0],float); y=np.full(n,X0[1],float); z=np.full(n,X0[2],float)
    alive=np.ones(n,bool)
    def chk():
        bad=~(np.isfinite(x)&np.isfinite(y)&np.isfinite(z)) | (np.abs(x)>bound)|(np.abs(y)>bound)|(np.abs(z)>bound)
        alive[bad]=False; x[~alive]=0; y[~alive]=0; z[~alive]=0
    for k in range(tr):
        x,y,z=vstep(x,y,z,h,r)
        if k%50==0: chk()
    chk()
    # MLE
    vx=np.ones(n); vy=0.3*np.ones(n); vz=-0.2*np.ones(n); S=np.zeros(n)
    xs=np.empty((keep,n)); zs=np.empty((keep,n))
    for k in range(mle_n):
        jx=(1-h*s)*vx+h*s*vy; jy=h*(r-z)*vx+(1-h)*vy-h*x*vz; jz=h*y*vx+h*x*vy+(1-h*b)*vz
        x,y,z=vstep(x,y,z,h,r)
        nv=np.sqrt(jx**2+jy**2+jz**2); nv[nv==0]=1; S+=np.log(nv); vx,vy,vz=jx/nv,jy/nv,jz/nv
        if k%50==0: chk()
        if k>=mle_n-keep: xs[k-(mle_n-keep)]=x; zs[k-(mle_n-keep)]=z
    chk()
    lam=S/mle_n; lam[~alive]=np.nan; xs[:,~alive]=np.nan; zs[:,~alive]=np.nan
    return xs,zs,lam,alive
