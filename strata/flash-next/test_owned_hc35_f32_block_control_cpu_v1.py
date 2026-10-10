"""Own arithmetic/source controls only; no model, compiled helper or device."""
import copy,struct,unittest
from unittest.mock import patch
import numpy as np
import owned_hc35_f32_block_control_v1 as own

class Controls(unittest.TestCase):
 def test_source_rms_square_FMA_schedule_differs_from_FP64_store(self):
  values=np.ones(2560,dtype='<f4');values[0]=10000
  row=own.rms_argument(values,1e-6)
  self.assertEqual(row['square_sum_f32'],100002480.)
  legacy=own.host.f32(np.sum(values.astype(np.float64)**2))
  self.assertEqual(legacy,100002560.);self.assertNotEqual(row['square_sum_f32'],legacy)
 def test_rms_exact_uniform_expected_argument(self):
  row=own.rms_argument([1.]*2560,1e-6)
  self.assertEqual(row,{'square_sum_f32':2560.,'mean_f32':1.,'rsqrt_argument_f32':own.host.f32(1+own.host.f32(1e-6))})
 def test_bad_shape_nonfinite_and_zero_argument_failclosed(self):
  for values,eps in [([1.]*31,1e-6),([float('nan')]*32,1e-6),([0.]*32,0.)]:
   with self.assertRaises(ValueError):own.rms_argument(values,eps)
 def test_mix_stream_layout_and_packet_extent(self):
  x=np.array([[1.]*32,[2.]*32,[3.]*32,[4.]*32]);gate=np.zeros((4,32));r=own.mixed_candidates(x,gate)
  self.assertEqual(set(r),{'host_expf_separate_mix','host_expf_fused_mix'})
  for value in r.values():
   self.assertEqual(value.tobytes(),np.full(32,1.25,dtype='<f4').tobytes());self.assertEqual(len(own.encode(value.tobytes(),1,32)),36)
  with self.assertRaises(ValueError):own.mixed_candidates(x[:3],gate[:3])
 def test_public_tokens_has_no_native_input_or_state_argument(self):
  with self.assertRaises(TypeError):own.OwnedHcBlockControl.tokens(object(),[1],captured_input=np.ones(32))
  with self.assertRaises(TypeError):own.OwnedHcBlockControl.tokens(object(),[1],native_state={})
 def test_actual_provider_identity_host_admission_required_before_rows(self):
  class Provider:
   actual_source=True;source_identity_sha256='a'*64
   def rows(self,*args):raise AssertionError('No source payload read on bad admission')
  with patch.object(own,'derive_schedule',return_value={}):
   with self.assertRaises(ValueError):own.OwnedHcBlockControl(Provider(),'b'*64,[],{},'/CPU_ONLY')
   with self.assertRaises(ValueError):own.OwnedHcBlockControl(Provider(),'a'*64,[],{},'/CPU_ONLY')
 def test_token_schedule_failure_before_host_or_original_rows(self):
  value=object.__new__(own.OwnedHcBlockControl);value.args=();value.env={}
  with patch.object(own,'derive_schedule',side_effect=ValueError('Unsupported token/config')):
   with self.assertRaises(ValueError):value.tokens([False])
 def test_same_input_original_projection_F32_not_FP64_product_store(self):
  # Cancellation from fixed source lane0 FMA accumulation and XOR16-first tree.
  w=np.ones(32,dtype='<f4');x=np.zeros(32,dtype='<f4');x[0]=2**24;x[16]=-2**24;x[8]=1
  self.assertEqual(own.dot(w,x),1.)
  self.assertEqual(own.COLUMNS,tuple(range(1120,1152)))
 def test_intrinsic_candidates_explicit_limiting_sigmoid(self):
  self.assertEqual(own.sigmoid_candidate(0.),.5);self.assertEqual(own.sigmoid_candidate(-1000.),0.);self.assertEqual(own.sigmoid_candidate(1000.),1.)
  self.assertEqual(own.rsqrt_candidate(4.),.5)
  with self.assertRaises(ValueError):own.sigmoid_candidate(float('nan'))

if __name__=='__main__':unittest.main()
