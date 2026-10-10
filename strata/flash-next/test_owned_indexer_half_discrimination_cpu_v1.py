"""Synthetic arithmetic/protocol tests, never a device measurement."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import owned_indexer_half_discrimination_v1 as d
class Controls(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name);self.raw=np.linspace(-2,2,512,dtype='<f4').tobytes();self.expected=d.preregistered(self.raw)
  (self.root/'consumed-raw.f32').write_bytes(self.raw)
  for r in range(3):
   p=self.root/('route'+str(r));p.mkdir()
   for name,value in [('expression.f32',self.expected['identity_F32']),('materialized.f32',self.expected['FP16_RNE_widened_F32']),('materialized.f16',self.expected['FP16_RNE']),('input.f32',self.raw)]: (p/name).write_bytes(value)
  self.log='HALF37_DEVICE backend=level_zero affinity=0 fp16=1 vendor=CPU driver=synthetic name=fixture\n'+''.join('HALF37_FRAME route=%d graph_replay=%d fields=4 own_input_restored=1 values=512\n'%(r,r) for r in range(3))+'HALF37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 compiler_lowering_qualified=0 full_model_math_qualified=0\n'
 def test_descriptive_independent_hypotheses_without_threshold(self):
  result=d.recollect(self.root,self.raw,self.log);self.assertTrue(result['fields']['expression.f32']['equals_identity_F32']);self.assertTrue(result['fields']['materialized.f32']['equals_FP16_RNE_widened_F32']);self.assertFalse(result['candidate_reference_modified']);self.assertIsNone(result['tolerance_gate'])
 def test_input_echo_mutation_rejected(self):
  (self.root/'consumed-raw.f32').write_bytes(bytes(2048))
  with self.assertRaises(ValueError):d.recollect(self.root,self.raw,self.log)
 def test_replay_field_mutation_rejected(self):
  (self.root/'route2/expression.f32').write_bytes(bytes(2048))
  with self.assertRaises(ValueError):d.recollect(self.root,self.raw,self.log)
 def test_error_missing_free_or_duplicate_frame_refused(self):
  for log in (self.log+'HALF37_ERROR bad\n',self.log.replace('owned_allocations_freed=1','owned_allocations_freed=0'),self.log+self.log.splitlines()[1]+'\n'):
   with self.assertRaises(ValueError):d.recollect(self.root,self.raw,log)
 def test_wrong_extent_nonfinite_overflow_refused(self):
  for raw in (bytes(4),np.full(512,np.nan,dtype='<f4').tobytes(),np.full(512,1e6,dtype='<f4').tobytes()):
   with self.assertRaises(ValueError):d.original_input(raw)
 def test_ties_even_and_sign_preregistered_without_measurement(self):
  x=np.zeros(512,dtype='<f4');x[:4]=[1+2**-11,1+3*2**-11,-1-2**-11,-0.0];h=np.frombuffer(d.preregistered(x.tobytes())['FP16_RNE'],dtype='<u2');self.assertEqual(h[:4].tolist(),[0x3c00,0x3c02,0xbc00,0x8000])
 def test_actual_source_flags_and_no_new_math_switches(self):
  binding=d.source_binding();argv=binding['leaf_argv'];self.assertIn('-fp-model=precise',argv);self.assertFalse(binding['runtime_ready']);self.assertEqual(set(binding['production_object_flags']['objects']),{'native_qsa.dp.cpp','native_rope.dp.cpp','native_qsa_indexer.dp.cpp','qsa_decode_attn.dp.cpp','qsa.dp.cpp'});self.assertIn('/leaf/owned_indexer_half_discrimination_gpu_v1.cpp',argv)
if __name__=='__main__':unittest.main()
