"""Source preregistration and synthetic complete device-field controls."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import native_rms_device_ops_proposal_v1 as p
class Tests(unittest.TestCase):
 def fixture(self,root):
  out=root/'candidate';old=root/'original'
  for route in range(3):
   (out/f'r{route}').mkdir(parents=True);(old/f'r{route}').mkdir(parents=True)
   for name,n in p.FIELDS.items():
    raw=np.full(n//4,1.,dtype='<f4').tobytes();(out/f'r{route}'/(name+'.f32')).write_bytes(raw)
    if name not in p.OPERATIONS:(old/f'r{route}'/(name+'.f32')).write_bytes(raw)
  return out,old
 def test_exact_source_additions_only(self):self.assertTrue(p.derivation()['actual_HC_call_and_original_shadow_unchanged'])
 def test_exact_preregistered_three_operations(self):self.assertEqual(list(p.OPERATIONS.values()),['sycl::rsqrt(x)','sycl::native::rsqrt(x)','1.0f/sycl::sqrt(x)'])
 def test_complete_direct_repeat_all_candidates(self):
  with tempfile.TemporaryDirectory() as t:
   out,old=self.fixture(Path(t));r=p.compare_outputs(out,old);self.assertTrue(r['all_routes_bitwise']);self.assertIsNone(r['tolerance_gate']);self.assertFalse(r['internal_HC_argument_observed'])
 def test_candidate_difference_reported_without_pass_threshold(self):
  with tempfile.TemporaryDirectory() as t:
   out,old=self.fixture(Path(t))
   for route in range(3):(out/f'r{route}'/'device_native_rsqrt_shadow.f32').write_bytes(np.full(4,2.,dtype='<f4').tobytes())
   r=p.compare_outputs(out,old);self.assertFalse(r['comparisons']['device_native_rsqrt_shadow']['bytes_equal_actual_HC_RS'])
 def test_missing_changed_HC_argument_nonfinite_or_repeat_rejected(self):
  for kind in ('missing','HC','argument','nonfinite','repeat'):
   with tempfile.TemporaryDirectory() as t:
    out,old=self.fixture(Path(t));name='device_rsqrt_shadow' if kind in ('missing','nonfinite','repeat') else 'actual_hc_rs' if kind=='HC' else 'separate_argument_shadow';path=out/'r1'/(name+'.f32')
    if kind=='missing':path.unlink()
    else:path.write_bytes(np.full(4,float('nan') if kind=='nonfinite' else 2.,dtype='<f4').tobytes())
    with self.assertRaises((ValueError,FileNotFoundError)):p.compare_outputs(out,old)
 def test_failed_original_run_not_accepted(self):
  with patch.object(p.prior,'finalized_binding',return_value={'actual_RMS_observed':False,'full_model_math_qualified':False}):
   with self.assertRaises(ValueError):p.prior_binding('/CPU_SYNTHETIC')
 def test_flags_unchanged_except_source(self):
  original=['FLAG','/leaf/native_rms_rsqrt37_gpu_v1.cpp','LIB']
  with patch.object(p.original,'compile_argv',return_value=list(original)):self.assertEqual(p.compile_argv(),['FLAG','/leaf/native_rms_rsqrt37_device_ops_entry_v1.cpp','LIB'])
if __name__=='__main__':unittest.main()
