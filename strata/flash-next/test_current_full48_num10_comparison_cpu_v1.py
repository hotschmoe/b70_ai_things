import unittest,numpy as np
import compare_current_full48_num10_v1 as a
class Controls(unittest.TestCase):
 def test_exact_and_changed(self):
  x=np.array([1,2],dtype='<f4');self.assertTrue(a.metrics(x,x)['bitwise_equal']);v=a.metrics(x,x+1);self.assertFalse(v['bitwise_equal']);self.assertEqual(v['max_absolute'],1);self.assertFalse(v['numeric_pass_claim'])
 def test_bad_shape_nonfinite(self):
  for x,y in((np.zeros(2),np.zeros(3)),(np.array([float('nan')]),np.ones(1))):
   with self.assertRaises(ValueError):a.metrics(x,y)
if __name__=='__main__':unittest.main()
