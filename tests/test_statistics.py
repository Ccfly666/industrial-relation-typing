import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from paired_comparison import combine
from paired_stats import influence
class Paired(unittest.TestCase):
 def setUp(self):
  self.y=np.array([0,1,2,3,0,1,2,3]);self.rows=np.arange(8);a=np.eye(4)[self.y];b=np.roll(a,1,axis=1);self.models={'a':dict(rows=self.rows,y=self.y,probabilities=a),'b':dict(rows=self.rows,y=self.y,probabilities=b)}
 def test_signs_and_same_predictions(self):
  d,p=combine(self.y,self.rows,self.models,[('a',1),('b',-1),('b',-1),('a',1)],influence);self.assertAlmostEqual(d,2.)
  d,p=combine(self.y,self.rows,self.models,[('a',1),('b',-1),('a',-1),('b',1)],influence);self.assertEqual(d,0);np.testing.assert_array_equal(p,0)
 def test_permutation_and_missing(self):
  before=combine(self.y,self.rows,self.models,[('a',1),('b',-1)],influence);m=self.models['b'];ix=np.arange(8)[::-1];self.models['b']={k:v[ix] for k,v in m.items()};after=combine(self.y,self.rows,self.models,[('a',1),('b',-1)],influence);self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
  with self.assertRaises(ValueError):combine(self.y,self.rows,self.models,[('missing',1)],influence)
 def test_nonregular(self):
  with self.assertRaises(ValueError):influence(np.array([0,0]),np.array([0,0]))
if __name__=='__main__':unittest.main()
