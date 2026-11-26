# Extracted from build_evidence.py for portable use. See manifests/code_origins.csv.
import numpy as np
cfg={"temperatures":[1,5,10]}

def summarize(sim,allow,labels):
 one=np.eye(4)[labels];counts=allow@one;parts=[];qs=[];strength=[];ess=[];ent=[]
 for beta in cfg['temperatures']:
  w=np.exp(beta*(sim-1))*allow;total=w.sum(1);mass=w@one;q=mass/total.clip(1e-30)[:,None];qs.append(q);strength.append(mass/counts.clip(1));ess.append(np.log1p(total**2/(w*w).sum(1).clip(1e-30)));ent.append(-(q*np.log(q.clip(1e-30))).sum(1))
 return counts,np.stack(qs,1),np.stack(strength,1),np.stack(ess,1),np.stack(ent,1)
