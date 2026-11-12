"""coeffs.py -- thresholds and normal form coefficients used by the overview figure."""
import numpy as np
s,b=10.0,8/3
def Bq(u,v,h): return h*np.array([0,-(u[0]*v[2]+u[2]*v[0]), u[0]*v[1]+u[1]*v[0]])
def J(X,h,r): x,y,z=X; return np.array([[1-h*s,h*s,0],[h*(r-z),1-h,-h*x],[h*y,h*x,1-h*b]])
def Ep(r): q=np.sqrt(b*(r-1)); return np.array([q,q,r-1])
def lams(r): return np.roots([1,s+b+1,b*(s+r),2*s*b*(r-1)])
def hF(r):
    L=lams(r); re=L[np.abs(L.imag)<1e-12].real; return min(2/np.abs(re))
def hNS(r):
    L=lams(r); c=L[np.abs(L.imag)>1e-12]
    return np.inf if len(c)==0 else min(-2*c.real/np.abs(c)**2)
def lF(r):
    h=hF(r); A=J(Ep(r),h,r); w,V=np.linalg.eig(A); i=np.argmin(abs(w+1)); q=np.real(V[:,i]); q/=np.linalg.norm(q)
    w2,W=np.linalg.eig(A.T); j=np.argmin(abs(w2+1)); p=np.real(W[:,j]); p/=p@q
    return -0.5*p@Bq(q,np.linalg.solve(A-np.eye(3),Bq(q,q,h)),h)
def dNS(r):
    h=hNS(r); A=J(Ep(r),h,r); w,V=np.linalg.eig(A); i=np.argmax(w.imag); mu=w[i]; q=V[:,i]/np.linalg.norm(V[:,i])
    w2,W=np.linalg.eig(A.T); j=np.argmin(abs(w2-np.conj(mu))); p=W[:,j]; p=p/np.conj(np.conj(p)@q)
    th=np.angle(mu); I=np.eye(3)
    t=2*np.conj(p)@Bq(q,np.linalg.solve(I-A,Bq(q,np.conj(q),h)),h)+np.conj(p)@Bq(np.conj(q),np.linalg.solve(np.exp(2j*th)*I-A,Bq(q,q,h)),h)
    return 0.5*np.real(np.exp(-1j*th)*t), h, th
if __name__=='__main__':
    print('lF(2)',lF(2.0),'expect 0.0086757'); print('d(5)',dNS(5.0)[0],'expect -0.0016278'); print('d(10)',dNS(10.0)[0],'expect 3.79e-5')
    print('hF(2)',hF(2.0),'hNS(5)',hNS(5.0))
