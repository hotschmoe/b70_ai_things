import unittest
import numpy as np
import analyze_current59_first_hc_packet_boundary_v1 as a
class Controls(unittest.TestCase):
 def test_exact_two_ulp_half_boundary(self):
  x=np.zeros(32,dtype='<f4');y=x.copy();x[0]=7.206923961639404;y[0]=7.206924915313721
  own=bytearray(36);native=bytearray(36);own[:2]=np.float16(.056732177734375).tobytes();native[:2]=np.float16(.0567626953125).tobytes();own[4]=127;native[4]=127
  got=a.packet_metrics(bytes(own),bytes(native),x,y);self.assertEqual(got['codes_different'],0);self.assertEqual(got['scale_blocks_different'],1);row=got['scale_boundaries'][0];self.assertEqual(row['absmax_ULP_distance'],2);self.assertLess(row['own_absmax_div127_F64_minus_midpoint'],0);self.assertGreater(row['native_absmax_div127_F64_minus_midpoint'],0);self.assertEqual(got['algebraic_code_delta_squared_norm'],0)
 def test_signed_ulp(self):
  x=np.array([-1.,0.,1.],dtype='<f4');y=np.nextafter(x,np.float32(np.inf));np.testing.assert_array_equal(a.ulp_distance(x,y),[1,1,1])
 def test_packet_shape_refusal(self):
  with self.assertRaises(ValueError):a.packet_metrics(bytes(36),bytes(72),np.zeros(32),np.zeros(64))
if __name__=='__main__':unittest.main()
