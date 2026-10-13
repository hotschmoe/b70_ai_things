import unittest
import numpy as np
from analyze_native62_QSA35_first_jump_v1 import residual_update
class Controls(unittest.TestCase):
 def test_saved_residual_update_is_explicit_fp64(self):
  before=np.array([1.,2.],dtype='<f4');after=np.nextafter(before,np.float32(np.inf));delta=residual_update(before,after);self.assertEqual(delta.dtype,np.dtype('float64'));np.testing.assert_array_equal(delta,after.astype(np.float64)-before.astype(np.float64))
 def test_source_one_cell_attention_independent_of_query(self):
  from qsa_owned_state_storage_v1 import QsaOwnedState
  s=QsaOwnedState();s.keys=[np.ones((2,256),dtype='<f4')];s.values=[np.full((2,256),.125,dtype='<f4')];gate=np.zeros((24,256),dtype='<f4');first=s.attention(np.zeros((24,256)),gate,np.array([0]));second=s.attention(np.ones((24,256)),gate,np.array([0]));np.testing.assert_array_equal(first,second);np.testing.assert_array_equal(first,np.full((24,256),.0625,dtype='<f4'))
if __name__=='__main__':unittest.main()
