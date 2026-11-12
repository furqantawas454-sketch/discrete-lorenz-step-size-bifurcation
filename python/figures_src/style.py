"""style.py -- common plotting style (Okabe-Ito colour-blind safe palette)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Okabe-Ito colour-blind safe palette
BLUE='#0072B2'; ORANGE='#E69F00'; VERM='#D55E00'; GREEN='#009E73'; SKY='#56B4E9'; PURPLE='#CC79A7'; GREY='#6b6b6b'
REG={'stable':'#e8f1fa','p2':'#eaf6ef','qp':'#fdf3e1','chaos':'#fbe6e1','escape':'#ececec'}
plt.rcParams.update({'font.size':9.5,'font.family':'serif','mathtext.fontset':'cm','axes.linewidth':0.7,
    'axes.labelsize':10.5,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
    'xtick.major.size':3.5,'ytick.major.size':3.5,'xtick.minor.visible':True,'ytick.minor.visible':True,
    'xtick.minor.size':1.8,'ytick.minor.size':1.8,'legend.fontsize':8,'legend.frameon':False,
    'axes.titlesize':10,'axes.titleweight':'normal','savefig.pad_inches':0.03})
SAVE=dict(dpi=500,bbox_inches='tight',metadata={'CreationDate':None})
def panel(ax,lab): ax.text(0.01,1.02,lab,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='left')
def regions(ax,spans,ymax_text=None,labels=True):
    for (a,b_,key,txt) in spans:
        ax.axvspan(a,b_,color=REG[key],zorder=0,lw=0)
        if labels and txt: ax.text((a+b_)/2,0.97,txt,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=7.5,color='0.25')
def crit(ax,h,lab,top=True,color='0.25'):
    ax.axvline(h,c=color,lw=0.8,ls=(0,(4,2)),zorder=1)
    if lab: ax.text(h,1.01,lab,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8.5,color=color)
def mlepanel(ax,hs,lam,color_line='k'):
    import numpy as np
    l=np.array(lam,float)
    ax.fill_between(hs,0,np.where(l>0,l,0),color=VERM,alpha=0.55,lw=0)
    ax.fill_between(hs,0,np.where(l<0,l,0),color=BLUE,alpha=0.30,lw=0)
    ax.plot(hs,l,c=color_line,lw=0.6); ax.axhline(0,c='0.3',lw=0.6)
    ax.set_ylabel(r'$\lambda_{\max}$')
def bifdots(ax,hs,xs,color,s=0.35,alpha=1):
    import numpy as np
    H=np.broadcast_to(hs,xs.shape); m=np.isfinite(xs)
    ax.scatter(H[m],xs[m],s=s,c=color,marker='.',lw=0,alpha=alpha,rasterized=True)
