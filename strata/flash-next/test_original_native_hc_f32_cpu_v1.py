import unittest
from types import SimpleNamespace
import numpy as np
import original_native_hc_f32_v1 as a
class Controls(unittest.TestCase):
 def test_lane_rounding_differs_from_double_sum(self):
  w=np.zeros(96,dtype='<f4');x=np.ones(96,dtype='<f4');w[0]=2**24;w[32]=1;w[64]=-2**24;w[1]=1
  self.assertEqual(a.dot32(w,x),1.0);self.assertEqual(float(w.astype('float64')@x.astype('float64')),2.0)
 def test_fused_product_survives_cancellation(self):
  w=np.zeros(64,dtype='<f4');x=np.zeros(64,dtype='<f4');w[0]=-1;x[0]=1;w[32]=1+2**-23;x[32]=1-2**-23
  self.assertEqual(a.dot32(w,x),-2**-46);self.assertEqual(float(np.float32(w[32]*x[32])+np.float32(-1)),0.0)
 def test_original_row_projection_never_calls_packet_projection(self):
  matrix=np.vstack([np.ones(32),np.arange(32)]).astype('<f4')
  provider=SimpleNamespace(shape=lambda name:(32,2),rows=lambda name,indices:matrix[list(indices)])
  projector=SimpleNamespace(p=provider,tile=4096,project=lambda *args:(_ for _ in()).throw(AssertionError('Q81 path forbidden for HC')))
  np.testing.assert_array_equal(a.project_original_rows(projector,'original-HC',np.ones(32)),np.array([32,496],dtype='<f4'))
 def test_norm_and_invalid_inputs(self):
  r=np.ones((4,64),dtype='<f4');value=a.normalize_streams(r,r,1e-6);np.testing.assert_array_equal(value['square_sum'],np.full(4,64,dtype='<f4'));self.assertEqual(value['normalized'].dtype,np.dtype('<f4'))
  with self.assertRaises(ValueError):a.normalize_streams(r[:3],r[:3],1e-6)
  r[0,0]=np.nan
  with self.assertRaises(ValueError):a.normalize_streams(r,np.ones_like(r),1e-6)
 def test_HC_write_has_original_FMA_shape(self):
  result=a.hc_write_original(np.ones((4,2560),dtype='<f4'),np.full(2560,2,dtype='<f4'),np.zeros(4,dtype='<f4'));np.testing.assert_array_equal(result,np.full((4,2560),3,dtype='<f4'))
if __name__=='__main__':unittest.main()
