# Extracted from common19.py for portable use. See manifests/code_origins.csv.
import numpy as np
from scipy.stats import norm

def cm(y,a,w=None):return np.bincount(4*y+a,weights=w,minlength=16).reshape(4,4)

def macro(c):
 c=np.asarray(c);den=c.sum(-1)+c.sum(-2);tp=c.diagonal(axis1=-2,axis2=-1)
 return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)

def influence(y,a):
 n=len(y);c=cm(y,a);p=c.sum(1)/n;q=c.sum(0)/n;tp=c.diagonal()/n;d=p+q
 if np.any(d<=0):raise ValueError('zero true+predicted support: Macro-F1 convention 0, IF nonregular')
 f=2*tp/d
 # Sparse evaluation of the sum of four class influence functions.
 v=-(f[y]/d[y]+f[a]/d[a]);same=y==a;v[same]+=2/d[y[same]]
 return macro(c),v/4

def groupsums(x,g):return np.bincount(g,weights=x)

def variances(psi,z,correct=False):
 n=len(psi);v={'iid':float(psi@psi/n**2)}
 if correct:v['iid']*=n/(n-1)
 for k in ['fact','source','target','pair']:
  s=groupsums(psi,z[k]);v[k]=float(s@s/n**2)
  if correct:v[k]=v[k]*len(s)/(len(s)-1) if len(s)>1 else None
 v['two_way']=v['source']+v['target']-v['pair'] if all(v[k] is not None for k in ['source','target','pair']) else None
 return v

def interval(d,v,alpha=.05):
 if v is None:return {'status':'INSUFFICIENT_CLUSTERS','variance':None,'se':None,'lo':None,'hi':None,'halfwidth':None,'cross_zero':None}
 if v<=0:return {'status':'NEGATIVE_VARIANCE' if v<0 else 'DEGENERATE_ZERO','variance':v,'se':None,'lo':None,'hi':None,'halfwidth':None,'cross_zero':None}
 se=float(np.sqrt(v));h=float(norm.ppf(1-alpha/2)*se)
 return {'status':'ESTIMATED_ASSUMPTION_CONDITIONAL','variance':v,'se':se,'lo':d-h,'hi':d+h,'halfwidth':h,'cross_zero':bool(d-h<=0<=d+h)}
