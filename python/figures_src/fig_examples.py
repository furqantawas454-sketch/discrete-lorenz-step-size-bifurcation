import os
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
os.makedirs(FIG, exist_ok=True)
import numpy as np
from scipy.optimize import fsolve
from vec import bif, s, b
from style import *
from matplotlib import cm
from matplotlib.colors import Normalize
np.seterr(all='ignore')
def G(X,h,r): x,y,z=X; return np.array([x+h*s*(y-x), y+h*(x*(r-z)-y), z+h*(x*y-b*z)])
def Ep(r): q=np.sqrt(b*(r-1)); return np.array([q,q,r-1])
Sm=np.diag([-1,-1,1.])
def traj(r,h,X0,tr,n,ctrl=None):
    X=np.array(X0,float)
    for k in range(tr): X=G(X,h,r)
    out=np.empty((n,3))
    for k in range(n): X=G(X,h,r); out[k]=X
    return out
def twopanel(): 
    fig,ax=plt.subplots(2,1,figsize=(5.6,4.4),sharex=True,gridspec_kw={'height_ratios':[2.2,1],'hspace':0.08}); return fig,ax
# ---------- Example 1 ----------
r=0.5; lm=(-(s+1)-np.sqrt((s-1)**2+4*s*r))/2; h1=2/abs(lm)
hs=np.linspace(0.17,0.20,1200); xs,zs,lam,al=bif(r,hs,[0.01,0,0],tr=20000,keep=200)
cyc=[]; X=np.array([0.7265,-0.0389,-0.0106])
for h in np.linspace(h1*(1-1e-3),0.17,300): X=fsolve(lambda U: G(U,h,r)-Sm@U,X,xtol=1e-13); cyc.append((h,X[0]))
cyc=np.array(cyc)
fig,ax=twopanel()
regions(ax[0],[(0.17,h1,'stable','stable origin'),(h1,0.20,'escape','escape')])
regions(ax[1],[(0.17,h1,'stable',None),(h1,0.20,'escape',None)],labels=False)
ax[0].plot(hs[hs<h1],0*hs[hs<h1],c=BLUE,lw=1.6,label='stable fixed point $E_0$')
ax[0].plot(cyc[:,0],cyc[:,1],c=VERM,lw=1.2,ls=(0,(4,2)),label='unstable period-2 orbit')
ax[0].plot(cyc[:,0],-cyc[:,1],c=VERM,lw=1.2,ls=(0,(4,2)))
ax[0].plot([h1],[0],'o',mfc='white',mec='k',ms=5,zorder=5)
crit(ax[0],h1,r'$h_1$'); crit(ax[1],h1,None)
ax[0].set_ylim(-5.2,5.2); ax[0].set_ylabel('$x$'); ax[0].legend(loc='upper right',bbox_to_anchor=(1.0,0.86),fontsize=7.5)
mlepanel(ax[1],hs,lam); ax[1].set_xlabel('$h$'); ax[1].set_xlim(0.17,0.20)
panel(ax[0],'(a)'); panel(ax[1],'(b)')
fig.savefig(os.path.join(FIG,'ex1_bif.pdf'),**SAVE); plt.close(fig)
# ---------- Example 2 ----------
r=2; hF=0.1779004904305973; hs=np.linspace(0.170,0.1855,1400)
xs,zs,lam,al=bif(r,hs,Ep(r)+[0.01,0,0],tr=40000,keep=300,mle_n=30000)
hesc=hs[np.where(~al)[0][0]]; hsec=0.18413
fig,ax=twopanel()
sp=[(0.170,hF,'stable','$E_\\pm$ stable'),(hF,hsec,'p2','stable period-2 orbit'),(hsec,hesc,'qp',None),(hesc,0.1855,'escape',None)]
regions(ax[0],sp); regions(ax[1],sp,labels=False)
bifdots(ax[0],hs,xs,BLUE,s=1.6)
crit(ax[0],hF,r'$h_F$'); crit(ax[1],hF,None)
ax[0].annotate('invariant curves',xy=(0.18425,5.9),xytext=(0.1722,5.9),fontsize=7.5,color='0.25',arrowprops=dict(arrowstyle='->',lw=0.6,color='0.3'))
ax[0].set_ylabel('$x$'); mlepanel(ax[1],hs,lam); ax[1].set_xlabel('$h$'); ax[1].set_xlim(0.170,0.1855)
panel(ax[0],'(a)'); panel(ax[1],'(b)')
fig.savefig(os.path.join(FIG,'ex2_bif.pdf'),**SAVE); plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(6.4,2.9))
for a,h,t in [(ax[0],0.1830,'$h=0.1830$'),(ax[1],0.1843,'$h=0.1843$')]:
    o=traj(r,h,Ep(r)+[0.01,0,0],60000,6000)
    if len(np.unique(np.round(o[:,0],5)))<50:
        a.plot(o[:2,0],o[:2,1],'o',c=BLUE,ms=6,zorder=3,label='period-2 orbit')
        a.annotate('',xy=o[1,:2],xytext=o[0,:2],arrowprops=dict(arrowstyle='->',color='0.5',lw=0.7,connectionstyle='arc3,rad=0.25'))
        a.annotate('',xy=o[0,:2],xytext=o[1,:2],arrowprops=dict(arrowstyle='->',color='0.5',lw=0.7,connectionstyle='arc3,rad=0.25'))
    else:
        a.scatter(o[:,0],o[:,1],s=0.8,c=np.arange(len(o))%2,cmap=matplotlib.colors.ListedColormap([BLUE,GREEN]),marker='.',lw=0,rasterized=True)
    a.plot(*Ep(r)[:2],'X',c=VERM,ms=7,mec='white',mew=0.6,zorder=4); a.text(Ep(r)[0]+0.2,Ep(r)[1]-0.12,'$E_+$',color=VERM,fontsize=9)
    a.set_xlim(-1,7); a.set_ylim(1.2,3.4); a.set_xlabel('$x$'); a.set_ylabel('$y$'); a.set_title(t,fontsize=9.5)
panel(ax[0],'(a)'); panel(ax[1],'(b)'); ax[0].legend(loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIG,'ex2_phase.pdf'),**SAVE); plt.close(fig)
# ---------- Example 3 ----------
r=5; hNS=0.1028203879486139; hs=np.linspace(0.095,0.1575,1400)
xs,zs,lam,al=bif(r,hs,Ep(r)+[0.01,0,0],tr=30000,keep=300,mle_n=30000)
hesc=hs[np.where(~al)[0][0]]; hmerge=0.12377
fig,ax=twopanel()
sp=[(0.095,hNS,'stable','$E_\\pm$\nstable'),(hNS,hmerge,'qp','two invariant curves'),(hmerge,hesc,'qp','one symmetric curve, windows'),(hesc,0.1575,'escape','')]
regions(ax[0],sp); regions(ax[1],sp,labels=False)
ax[0].axvline(hmerge,c='0.55',lw=0.6,ls=':')
bifdots(ax[0],hs,xs,BLUE,s=0.35)
crit(ax[0],hNS,r'$h_{NS}$'); crit(ax[1],hNS,None)
ax[0].set_ylabel('$x$'); ax[0].set_ylim(-7,8.2); mlepanel(ax[1],hs,lam); ax[1].set_xlabel('$h$'); ax[1].set_xlim(0.095,0.1575)
panel(ax[0],'(a)'); panel(ax[1],'(b)')
fig.savefig(os.path.join(FIG,'ex3_bif.pdf'),**SAVE); plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(6.4,5.2))
for a,h,t,lab in [(ax[0,0],0.110,'$h=0.110$','(a)'),(ax[0,1],0.118,'$h=0.118$','(b)'),(ax[1,0],0.124,'$h=0.124$','(c)'),(ax[1,1],0.154,'$h=0.154$','(d)')]:
    for X0,c in [(Ep(r)+[0.01,0,0],BLUE),(Sm@Ep(r)+[-0.01,0,0],ORANGE)]:
        o=traj(r,h,X0,60000,5000)
        per=len(np.unique(np.round(o[:,0],5)))<50
        a.scatter(o[:,0],o[:,2],s=(16 if per else 0.9),c=c,marker='.' if not per else 'o',lw=0,rasterized=True)
    for E,n in [(Ep(r),'$E_+$'),(Sm@Ep(r),'$E_-$')]:
        a.plot(E[0],E[2],'X',c=VERM,ms=7,mec='white',mew=0.6,zorder=4); a.text(E[0]+0.3,E[2]+0.3,n,color=VERM,fontsize=8.5)
    a.set_xlabel('$x$'); a.set_ylabel('$z$'); a.set_title(t,fontsize=9.5); panel(a,lab)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'ex3_phase.pdf'),**SAVE); plt.close(fig)
# ---------- Example 4 ----------
r=10; hNS=0.03095511134839678; hs=np.linspace(0.020,0.0875,1400)
xsA,zsA,lamA,alA=bif(r,hs,Ep(r)+[0.01,0,0],tr=40000,keep=250,mle_n=30000)
xsB,zsB,lamB,alB=bif(r,hs,[1.0,1.0,1.0],tr=40000,keep=250,mle_n=30000)
hesc=hs[np.where(~alB)[0][0]]; hon=0.02791
fig,ax=twopanel()
sp=[(0.020,hon,'stable','$E_\\pm$\nstable'),(hon,hNS,'p2',None),(hNS,hesc,'chaos','chaos (with periodic windows)'),(hesc,0.0875,'escape',None)]
REG['p2']='#f1ecf7'
regions(ax[0],sp); regions(ax[1],sp,labels=False)
bifdots(ax[0],hs,xsB,ORANGE,s=0.3,alpha=0.8); bifdots(ax[0],hs,xsA,BLUE,s=0.9)
ax[0].plot([],[],'o',c=ORANGE,ms=4,label='start at $(1,1,1)$'); ax[0].plot([],[],'o',c=BLUE,ms=4,label='start near $E_+$')
ax[0].legend(loc='lower right',bbox_to_anchor=(1.0,1.07),ncol=2,handletextpad=0.2,columnspacing=0.8)
ax[0].annotate('bistable',xy=(0.0294,-11.8),xytext=(0.037,-12.4),fontsize=7.5,color='0.25',va='center',arrowprops=dict(arrowstyle='->',lw=0.6,color='0.3'))
crit(ax[0],hNS,r'$h_{NS}$'); crit(ax[1],hNS,None)
ax[0].set_ylabel('$x$'); ax[0].set_ylim(-13.5,15)
l=np.array(lamB); ax[1].fill_between(hs,0,np.where(l>0,l,0),color=VERM,alpha=0.5,lw=0); ax[1].fill_between(hs,0,np.where(l<0,l,0),color=BLUE,alpha=0.3,lw=0)
ax[1].plot(hs,lamB,c=ORANGE,lw=0.8); ax[1].plot(hs,lamA,c=BLUE,lw=0.8); ax[1].axhline(0,c='0.3',lw=0.6); ax[1].set_ylabel(r'$\lambda_{\max}$')
ax[1].set_xlabel('$h$'); ax[1].set_xlim(0.020,0.0875); panel(ax[0],'(a)'); panel(ax[1],'(b)')
fig.savefig(os.path.join(FIG,'ex4_bif.pdf'),**SAVE); plt.close(fig)
fig=plt.figure(figsize=(7.2,3.0))
gs=fig.add_gridspec(1,3,width_ratios=[1.35,1,1],wspace=0.38)
o=traj(r,0.07,[1,1,1],40000,40000)
ax3=fig.add_subplot(gs[0],projection='3d')
ax3.scatter(o[:,0],o[:,1],o[:,2],s=0.12,c=o[:,2],cmap='viridis',lw=0,rasterized=True,alpha=0.9)
for E in [Ep(r),Sm@Ep(r)]: ax3.scatter([E[0]],[E[1]],[E[2]],marker='X',s=40,c=VERM,edgecolors='white',linewidths=0.5,depthshade=False)
ax3.view_init(elev=18,azim=-60); ax3.grid(False)
for axis in (ax3.xaxis,ax3.yaxis,ax3.zaxis): axis.pane.fill=False; axis.pane.set_edgecolor('0.75')
ax3.set_xlabel('$x$',labelpad=-7); ax3.set_ylabel('$y$',labelpad=-7); ax3.set_zlabel('$z$',labelpad=-9)
ax3.tick_params(labelsize=6.5,pad=-3); ax3.set_box_aspect((1,1,0.8))
ax3.text2D(0.02,0.98,'(a)',transform=ax3.transAxes,fontsize=10,fontweight='bold'); ax3.set_title('$h=0.070$',fontsize=9.5,y=0.98)
a=fig.add_subplot(gs[1])
a.scatter(o[:,0],o[:,2],s=0.15,c=o[:,2],cmap='viridis',lw=0,rasterized=True)
for E,n in [(Ep(r),'$E_+$'),(Sm@Ep(r),'$E_-$')]: a.plot(E[0],E[2],'X',c=VERM,ms=7,mec='white',mew=0.6,zorder=4)
a.set_xlabel('$x$'); a.set_ylabel('$z$'); a.set_title('$h=0.070$',fontsize=9.5); panel(a,'(b)')
a=fig.add_subplot(gs[2])
o2=traj(r,0.03,[1,1,1],40000,30000); a.scatter(o2[:,0],o2[:,2],s=0.15,c=ORANGE,lw=0,rasterized=True,alpha=0.7)
for E in [Ep(r),Sm@Ep(r)]:
    p=traj(r,0.03,E+[0.01,0,0],40000,1); a.plot(p[0,0],p[0,2],'o',c=BLUE,ms=6,mec='white',mew=0.7,zorder=5)
a.plot([],[],'o',c=BLUE,ms=5,label='stable $E_\\pm$'); a.plot([],[],'s',c=ORANGE,ms=4,label='chaotic attractor'); a.legend(loc='upper right',fontsize=7,ncol=1,handletextpad=0.3)
a.set_xlabel('$x$'); a.set_ylabel('$z$'); a.set_title('$h=0.030$',fontsize=9.5); a.set_ylim(-0.5,26); panel(a,'(c)')
fig.savefig(os.path.join(FIG,'ex4_phase.pdf'),**SAVE); plt.close(fig)
print('done',hesc)
