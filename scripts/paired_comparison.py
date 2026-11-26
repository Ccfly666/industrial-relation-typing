import numpy as np
def combine(y,rows,models,terms,influence):
 if len(np.unique(rows))!=len(rows):raise ValueError('Duplicate query identity')
 total=0.;psi=np.zeros(len(y))
 for key,sign in terms:
  if key not in models:raise ValueError('Missing prediction term '+str(key))
  m=models[key];idx={int(r):i for i,r in enumerate(m['rows'])}
  if len(idx)!=len(rows) or set(idx)!=set(map(int,rows)):raise ValueError('Query identity mismatch')
  ix=np.array([idx[int(r)] for r in rows]);p=m['probabilities'][ix]
  if not np.array_equal(m['y'][ix],y):raise ValueError('Reference label mismatch')
  score,part=influence(y,p.argmax(1));total+=sign*score;psi+=sign*part
 return total,psi
