"""CPU exact arithmetic/source/record controls; no compiler/helper/GPU/model read."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np
import native_rms_rsqrt37_hypotheses_v1 as h
import native_rms_rsqrt37_proposal_v1 as p
class Tests(unittest.TestCase):
 def test_exact_rsqrt(self):
  for x,y in [(1.,1.),(4.,.5),(.25,2.),(16.,.25)]:self.assertEqual(h.rounded_rsqrt(x),y)
 def test_argument_rejects_invalid_or_notF32(self):
  for x in (True,0.,-1.,float('nan'),float('inf'),1.000000000001):
   with self.assertRaises(ValueError):h.rounded_rsqrt(x)
 def test_two_arguments_threeRS_preregistered(self):
  row=h.variants(np.zeros((4,2560),dtype='<f4'));self.assertEqual(set(row),{'source_FMA_XOR','legacy_square_FP64_mean'});self.assertTrue(all(set(v)=={'argument','host_sqrtf_reciprocal','mathematically_rounded_rsqrt','legacy_FP64_sqrt_reciprocal'} for v in row.values()))
 def test_invalid_owned_geometry(self):
  with self.assertRaises(ValueError):h.variants(np.zeros((1,2560)))
 def test_actual_builder_source_not_compile_commands(self):
  f=p.builder_flags();self.assertIn('-fp-model=precise',f['FLAGS']);self.assertEqual(p.compile_argv()[0],'/opt/intel/oneapi/compiler/2026.1/bin/icpx');self.assertNotIn('-ffast-math',p.compile_argv())
 def test_saved_independent_artifact_only(self):
  r,b=p.saved_original_input();self.assertEqual(r.shape,(4,2560));self.assertFalse(b['captured_native_operand_used']);self.assertFalse(b['fresh_model_payload_read']);self.assertFalse(b['full_model_math_qualified'])
 def make(self,root):
  for route in range(3):
   out=root/('r'+str(route));out.mkdir()
   for name,n in [('actual_hc_rs',4),('actual_hc_xn_normones',10240),('separate_square_sum_shadow',4),('separate_argument_shadow',4)]: (out/(name+'.f32')).write_bytes(np.ones(n,dtype='<f4').tobytes())
 def test_direct_graph_known_negative(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);self.make(root);self.assertTrue(p.compare_routes(root)['bitwise_direct_graph_repeat_equal']);path=root/'r2/actual_hc_rs.f32';raw=bytearray(path.read_bytes());raw[0]^=1;path.write_bytes(raw)
   with self.assertRaises(ValueError):p.compare_routes(root)
 def test_incomplete_nonfinite_output(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);self.make(root);(root/'r0/actual_hc_rs.f32').write_bytes(np.full(4,np.nan,dtype='<f4').tobytes())
   with self.assertRaises(ValueError):p.compare_routes(root)
 def test_source(self):p.source_binding()
if __name__=='__main__':unittest.main()
