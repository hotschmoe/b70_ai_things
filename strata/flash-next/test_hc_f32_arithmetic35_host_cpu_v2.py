"""Analytic host-fmaf/source-schedule controls; C++ remains uncompiled/unrun."""
import ast,json,math,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import hc_f32_arithmetic35_host_v2 as q
pack=lambda values:struct.pack('<%df'%len(values),*values)
scalar=lambda raw:struct.unpack('<f',raw)[0]

class AnalyticTests(unittest.TestCase):
 def test_host_fmaf_fused_known_exact_negative(self):
  a=1+2**-23;b=1-2**-23;c=-1.
  self.assertEqual(q.host_fma(a,b,c),-2**-46)
  self.assertEqual(q.f32(q.f32(a*b)+c),0.)
  self.assertNotEqual(q.host_fma(a,b,c),q.f32(q.f32(a*b)+c))
 def test_host_fmaf_RNE_ties_even(self):
  self.assertEqual(q.host_fma(1.,2**-24,1.),1.)
  self.assertEqual(q.host_fma(1.,2**-24,1+2**-23),1+2**-22)
 def test_host_gradual_underflow_known_subnormal(self):
  tiny=2**-149
  self.assertEqual(q.host_fma(tiny,1.,0.),tiny)
  self.assertEqual(q.host_fma(tiny,.5,tiny),2*tiny)
  self.assertEqual(q.f32(q.f32(tiny*.5)+tiny),tiny)
 def test_single_column_down_and_up_row_analytic(self):
  for k in (320,10240):
   w=[0.]*k;x=[0.]*k;w[k-1]=.125;x[k-1]=-8.
   self.assertEqual(scalar(q.dot32_f32(pack(w),pack(x))),-1.)
 def test_XOR_tree_exact_dyadic_alllanes(self):
  w=[1.]*32;x=[float(i-16)/8 for i in range(32)]
  self.assertEqual(scalar(q.dot32_f32(pack(w),pack(x))),-2.)
 def test_lane_strided_cancellation_and_wrong_order_negative(self):
  w=[1.]*96;x=[0.]*96;x[0]=2**24;x[32]=1.;x[64]=-2**24
  self.assertEqual(math.fsum(x),1.)
  self.assertEqual(scalar(q.dot32_f32(pack(w),pack(x))),0.)
  x[32],x[64]=x[64],x[32]
  self.assertEqual(scalar(q.dot32_f32(pack(w),pack(x))),1.)
 def test_XOR_mask_order_cancellation_negative(self):
  x=[0.]*32;x[0]=2**24;x[1]=1.;x[16]=-2**24
  self.assertEqual(math.fsum(x),1.)
  self.assertEqual(scalar(q.dot32_f32(pack([1.]*32),pack(x))),1.)
  # Wrong pairing loses1 before cancellation: first XOR1 pairs A+1, then
  # final XOR16 cancels A with -A. Exact source tree pairs A and -A first.
  wrong=x
  for mask in (1,2,4,8,16):wrong=[q.f32(wrong[lane]+wrong[lane^mask]) for lane in range(32)]
  self.assertEqual(wrong[0],0.)
 def test_FTZ_DAZ_behavior_fails_before_actual_operation(self):
  original=q._fma
  for mode in ('FTZ','DAZ'):
   calls=[]
   def fake(a,b,c):
    calls.append((a,b,c))
    if mode=='FTZ' and (a,b,c)==(2**-126,.5,0.):return 0.
    if mode=='DAZ' and (a,b,c)==(2**-149,2**24,0.):return 0.
    return original(a,b,c)
   with patch.object(q,'_fma',side_effect=fake):
    with self.assertRaisesRegex(ValueError,mode):q.host_fma(3.,4.,5.)
   self.assertNotIn((3.,4.,5.),calls)
 def test_gradual_probes_rechecked_after_earlier_positive(self):
  self.assertTrue(q.require_gradual_f32())
  with patch.object(q,'_fma',return_value=0.):
   with self.assertRaises(ValueError):q.dot32_f32(pack([1.]*32),pack([1.]*32))
 def test_Q8halfscale_signed_codes_decode_analytic(self):
  for code in (-128,-127,-1,0,1,127):
   codes=[0]*32;codes[17]=code;raw=struct.pack('<e',.125)+struct.pack('32b',*codes);x=[0.]*32;x[17]=8.
   self.assertEqual(scalar(q.q8dot32_f32(raw,pack(x))),float(code))
 def test_Q8halfsubnormal_scale_no_hidden_BF16_conversion(self):
  codes=[0]*32;codes[0]=1;raw=struct.pack('<e',2**-24)+struct.pack('32b',*codes);x=[0.]*32;x[0]=2**24
  self.assertEqual(scalar(q.q8dot32_f32(raw,pack(x))),1.)
 def test_elementwise_FMA_reports_separate_multiply_negative(self):
  result=q.fma_f32(pack([1+2**-23]),pack([1-2**-23]),pack([-1.]));self.assertEqual(scalar(result),-2**-46)
 def test_badextent_nonfinite_and_unimplemented_ops_failclosed(self):
  for w,x in [(pack([1.]),pack([2.])),(bytes(128),pack([float('nan')]*32)),(bytes(128),bytes(256))]:
   with self.assertRaises(ValueError):q.dot32_f32(w,x)
  with self.assertRaises(ValueError):q.q8dot32_f32(struct.pack('<e',float('nan'))+bytes(32),bytes(128))
  with self.assertRaises(ValueError):q.fma_f32(pack([1.]),bytes(8),pack([0.]))
 def test_rounding_mode_failclosed(self):
  with patch.object(q,'_getround',return_value=1):
   with self.assertRaises(ValueError):q.host_fma(1.,2.,3.)
 def test_frozen_original_model_not_imported_or_called(self):
  tree=ast.parse(Path(q.__file__).read_text());calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)];self.assertFalse(any(name.endswith('.tokens') for name in calls));self.assertNotIn('Full48OwnedComposition',Path(q.__file__).read_text());self.assertNotIn('exp(',Path(q.__file__).read_text());self.assertNotIn('rsqrt(',Path(q.__file__).read_text())
 def test_source_plan_binding(self):self.assertEqual(q.source_binding()['native_graph_or_intrinsic_qualification'],False)

class AdapterTests(unittest.TestCase):
 def test_explicit_mock_helper_rounding_and_identity_only(self):
  with tempfile.TemporaryDirectory() as name:
   helper=Path(name)/'helper';helper.write_bytes(b'\x7fELFCPU_MOCK_NOT_EXECUTABLE');digest=q.sha(helper)
   def run(cmd,**kwargs):
    Path(cmd[6]).write_bytes(pack([-2**-46]));return type('CPU',(),{'stdout':json.dumps({'op':'fma','elements':1,'rounding':'FE_TONEAREST','flush_to_zero':False,'denormals_are_zero':False,'device_intrinsics_qualified':False,'model_math_qualified':False})})()
   with patch.object(q.subprocess,'run',side_effect=run) as mock:
    raw,result=q.run_helper(helper,digest,'fma',pack([1+2**-23]),pack([1-2**-23]),pack([-1.]));self.assertEqual(scalar(raw),-2**-46);self.assertFalse(result['implementation_arithmetic_qualified']);self.assertFalse(result['model_math_qualified']);self.assertIsNone(result['tolerance_gate']);self.assertEqual(mock.call_count,1);self.assertTrue(result['host_gradual_input_output_probes_passed']);self.assertFalse(result['host_direct_MXCSR_flags_observed'])
 def test_unknown_intrinsic_not_executed(self):
  with tempfile.TemporaryDirectory() as name:
   helper=Path(name)/'helper';helper.write_bytes(b'\x7fELFCPU_MOCK')
   with patch.object(q.subprocess,'run',side_effect=AssertionError('No unqualified intrinsic execution')):
    with self.assertRaises(ValueError):q.run_helper(helper,q.sha(helper),'exp',pack([1.]),pack([1.]))

if __name__=='__main__':unittest.main()
