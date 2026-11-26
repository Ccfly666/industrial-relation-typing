"""License-independent runnable component acceptance in a clean environment."""
from pathlib import Path
import sys,json,unittest
import numpy as np
import torch
from method import smoke
from train_classifier import trajectory
from projection import retract
def main():
 root=Path(__file__).resolve().parents[1];smoke();suite=unittest.defaultTestLoader.discover(str(root/'tests'));result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise SystemExit(1)
 generator=np.random.default_rng(2026112602);xx=generator.normal(size=(80,146)).astype(np.float32);y=np.arange(64)%4;degree=np.ones((16,2),int);degree[:3]=0;base=np.tile([.1,.2,.3,.4],(16,1));scope=dict(x_fit=xx[:64],x_eval=xx[64:],y_fit=y,y_eval=np.arange(16)%4,fit_rows=np.arange(64),eval_rows=np.arange(64,80),degree_eval=degree,fallback_probability=base);checks=[]
 for arm in ['AB','ABR','Full']:
  pred,curve,weights,seconds=trajectory(scope,arm,.001,5,2026111601,'cpu');np.testing.assert_array_equal(pred['probabilities'][:3],base[:3]);np.testing.assert_allclose(pred['probabilities'].sum(1),1,atol=1e-6);assert np.isfinite(pred['probabilities']).all();checks.append(dict(arm=arm,epochs=5,seconds=seconds,weights=len(weights),fallback_exact=True))
 q=retract(torch.randn(80,64,dtype=torch.float64));assert float(abs(q.T@q-torch.eye(64)).max())<1e-12
 out=root/'verification';out.mkdir(exist_ok=True);(out/'synthetic_acceptance.json').write_text(json.dumps(dict(status='PASS',source_tests=result.testsRun,head_checks=checks,projection_retraction=True,python=sys.version,torch=torch.__version__,numpy=np.__version__),indent=2),encoding='utf-8');print('CLEAN_SYNTHETIC_ACCEPTANCE_PASS',result.testsRun)
if __name__=='__main__':main()
