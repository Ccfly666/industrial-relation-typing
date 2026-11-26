# Extracted from prepare_context.py for portable use. See manifests/code_origins.csv.
import hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler
INV=np.array([0,2,1,3])

def relation_features(pool,fit,ev,y):
 u=pool['u'];v=pool['v'];groups=pool['groups'];nS=len(pool['source_keys']);nT=len(pool['target_keys']);units=sorted(set(groups[fit]),key=lambda g:hashlib.sha256(f'MS15|16|{g}'.encode()).hexdigest());mp={g:i//8 for i,g in enumerate(units)};blocks=np.array([mp[g]for g in groups[fit]]);cs=np.zeros((nS,4),np.int64);ct=np.zeros((nT,4),np.int64);np.add.at(cs,(u[fit],y),1);np.add.at(ct,(v[fit],y),1);sk=blocks*nS+u[fit];tk=blocks*nT+v[fit];su,si=np.unique(sk,return_inverse=True);tu,ti=np.unique(tk,return_inverse=True);ds=np.zeros((len(su),4),int);dt=np.zeros((len(tu),4),int);np.add.at(ds,(si,y),1);np.add.at(dt,(ti,y),1);pos=np.r_[fit,ev];s=cs[u[pos]].copy();t=ct[v[pos]].copy();s[:len(fit)]-=ds[si];t[:len(fit)]-=dt[ti];assert(s>=0).all()and(t>=0).all();degree=np.column_stack([s.sum(1),t.sum(1)]);raw=np.column_stack([np.log1p(degree),degree>0,s/degree[:,0,None].clip(1),t[:,INV]/degree[:,1,None].clip(1)]);sc=StandardScaler().fit(raw[:len(fit)]);r=sc.transform(raw).astype(np.float32)
 assert not set(groups[fit])&set(groups[ev]);assert not set(zip(u[fit],v[fit]))&set(zip(u[ev],v[ev]));return raw,r,degree,sc,blocks
