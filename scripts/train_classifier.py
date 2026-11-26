"""Explicit single-scope MLP path or complete nested suite over prepared scope files."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from net import Net
from paired_stats import cm,macro
def load(p):
 with np.load(p,allow_pickle=False) as z:return dict(z)
def trajectory(scope,arm,lr,epochs,seed,device='cpu',validation=False):
 if not 1<=epochs<=160:raise ValueError('This frozen reproduction supports budgets through160 only')
 if len(np.unique(scope['fit_rows']))!=len(scope['fit_rows']) or len(set(scope['y_fit']))!=4:raise ValueError('Invalid fit identity/class support')
 if scope['x_fit'].shape[1]!=146 or scope['x_eval'].shape[1]!=146:raise ValueError('Expected146 coordinates')
 if not np.isfinite(scope['x_fit']).all() or not np.isfinite(scope['x_eval']).all():raise ValueError('Nonfinite input')
 torch.set_num_threads(4);torch.manual_seed(seed)
 if device=='cuda':torch.cuda.manual_seed_all(seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 n=len(scope['y_fit']);x=np.r_[scope['x_fit'],scope['x_eval']].astype(np.float32);cut={'AB':71,'ABR':83,'Full':146}[arm];x[:,cut:]=0;xx=torch.from_numpy(x).to(device);yy=torch.as_tensor(scope['y_fit'],device=device);net=Net('R1' if arm=='ABR' else arm).to(device);opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.001);curves=[];started=time.perf_counter();prob=None
 for ep in range(1,epochs+1):
  net.train();order=np.random.default_rng(np.random.SeedSequence([seed,ep])).permutation(n)
  for j in range(0,n,4096):
   ix=torch.as_tensor(order[j:j+4096],device=device);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(xx[ix],None,None,None),yy[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  if ep%5==0 or ep==epochs:
   net.eval()
   with torch.inference_mode():prob=torch.softmax(torch.cat([net(xx[j:j+4096],None,None,None) for j in range(n,len(x),4096)]),1).cpu().numpy()
   curves.append(dict(epoch=ep,macro=float(macro(cm(scope['y_eval'],prob.argmax(1)))) if validation else None,updates=ep*math.ceil(n/4096)))
 op=prob.astype(np.float64);zero=scope['degree_eval'].sum(1)==0
 if 'fallback_probability' not in scope:raise ValueError('An explicitly bound common fallback probability array is required')
 op[zero]=scope['fallback_probability'][zero]
 return dict(rows=scope['eval_rows'],probabilities=op,raw_probabilities=prob),curves,{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()},time.perf_counter()-started
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--suite',type=Path);p.add_argument('--task',default='E');p.add_argument('--out',required=True,type=Path);p.add_argument('--arm',choices=['AB','ABR','Full'],default='Full');p.add_argument('--lr',type=float,default=.001);p.add_argument('--epochs',type=int,default=80);p.add_argument('--budget',type=int,choices=[80,160],default=160);p.add_argument('--seed',type=int,default=2026111601);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 def run(inp,out,arm,lr,ep,seed,val=False):
  out.mkdir(parents=True,exist_ok=True);pred,curve,weights,seconds=trajectory(load(inp),arm,lr,ep,seed,a.device,val);np.savez_compressed(out/'prediction.npz',**pred);np.savez_compressed(out/'weights.npz',**weights);(out/'run.json').write_text(json.dumps(dict(curves=curve,seconds=seconds,arm=arm,lr=lr,epochs=ep,seed=seed)),encoding='utf-8');return curve
 if a.suite:
  for o in range(5):
   for arm in ['AB','ABR','Full']:
    trials=[]
    for lr in [.0003,.001]:
     ll=[run(a.suite/f'{a.task}_o{o}i{i}.npz',a.out/f'{a.task}_o{o}_{arm}'/f'inner_{lr}_{i}',arm,lr,a.budget,2026111601,True) for i in range(3)]
     for j,ep in enumerate(range(5,a.budget+1,5)):trials.append(dict(lr=lr,epochs=ep,mean=float(np.mean([r[j]['macro'] for r in ll]))))
    best=max(trials,key=lambda r:(r['mean'],-r['lr'],-r['epochs']));(a.out/f'{a.task}_o{o}_{arm}'/'selection.json').write_text(json.dumps(best),encoding='utf-8')
    for seed in [2026111601,2026111602,2026111603]:run(a.suite/f'{a.task}_o{o}.npz',a.out/f'{a.task}_o{o}_{arm}'/f'seed{seed}',arm,best['lr'],best['epochs'],seed)
 elif a.input:run(a.input,a.out,a.arm,a.lr,a.epochs,a.seed)
 else:p.error('--input or --suite is required')
if __name__=='__main__':main()
