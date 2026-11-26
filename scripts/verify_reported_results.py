"""Public main-result replay. Restricted inputs are supplied explicitly by the user."""
import argparse
from pathlib import Path
from method import check_scores
def check_intervals(directory,out):
 import json,hashlib,csv
 import numpy as np
 from paired_stats import influence,variances,interval
 root=Path(__file__).resolve().parents[1];manifest=json.loads((root/'manifests/original54_replay.json').read_text(encoding='utf-8'));cache={};records=[]
 def load(h):
  if h not in cache:
   if len(cache)>16:cache.clear()
   p=directory/(h+'.npz')
   with p.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!=h:raise ValueError('Asset SHA mismatch')
   with np.load(p,allow_pickle=False) as z:cache[h]=dict(z)
  return cache[h]
 for r in manifest:
  dep=load(r['dependency_sha']);psis=[];deltas=[]
  for seed in r['seeds']:
   psi=np.zeros(len(dep['y']));delta=0.
   for t in seed['terms']:
    q=load(t['sha256']);lut={int(row):i for i,row in enumerate(q['rows'])}
    if len(lut)!=len(dep['rows']) or set(lut)!=set(map(int,dep['rows'])):raise ValueError('Query pairing mismatch')
    ix=np.array([lut[int(row)] for row in dep['rows']]);m,v=influence(dep['y'],q[t['probability_key']][ix].argmax(1));delta+=t['sign']*m;psi+=t['sign']*v
   psis.append(psi);deltas.append(delta)
  estimate=float(np.mean(deltas));var=variances(np.mean(psis,0),dep)['two_way'];ci=interval(estimate,var,.05/r['family_size']);err=max(abs(estimate-r['estimate']),*(abs(ci[k]-v) for k,v in r['expected'].items()))
  if err>5e-13:raise ValueError('Original interval mismatch: '+str((r['task'],r['comparison'],err)))
  records.append(dict(task=r['task'],family=r['family'],comparison=r['comparison'],max_abs_error=err,estimate=estimate,**ci))
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 print('Verified original conditional intervals:',len(records))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path);p.add_argument('--stat-assets',type=Path);p.add_argument('--out',type=Path,default=Path('interval_verification.csv'));a=p.parse_args()
 if a.predictions:check_scores(a.predictions)
 if a.stat_assets:check_intervals(a.stat_assets,a.out)
 if not a.predictions and not a.stat_assets:p.error('Supply --predictions and/or --stat-assets')
