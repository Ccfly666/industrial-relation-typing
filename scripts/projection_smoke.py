"""Synthetic A64 fit test, not an industrial representation experiment."""
from pathlib import Path
import json,time
import numpy as np
from method import fit_a64
if __name__=='__main__':
 rng=np.random.default_rng(2026112605);alphabet=np.array(list('abcdefghijklmnopqrstuvwxyz'));fields=np.array([[''.join(rng.choice(alphabet,18))+' item '+str(i)+' field '+str(j) for j in range(4)] for i in range(96)]);fit=np.arange(80);y=np.arange(80)%4;start=time.perf_counter();a=fit_a64(fields,fit,y,'cpu');assert a.shape==(96,64) and np.isfinite(a).all();assert abs(a[fit].mean(0)).max()<1e-5
 p=Path(__file__).resolve().parents[1]/'verification/projection_smoke.json';p.write_text(json.dumps(dict(status='PASS',synthetic=True,N=96,fit=80,dimensions=64,seconds=time.perf_counter()-start,formal_projection_fits=0,centered_norm_type='Python float64 scalar as original solver contract')),encoding='utf-8');print('SYNTHETIC_A64_FIT_PASS',a.shape)
