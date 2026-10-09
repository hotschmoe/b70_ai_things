#!/usr/bin/env python3
"""Real role geometry synthetic own-state CPU controls; no weights/GPU/native pass."""
import gc,unittest
import numpy as np
from full48_owned_composition_storage_v1 import Full48OwnedComposition,OriginalStorageProjector
from full48_synthetic_original_roles_v1 import SyntheticOriginalRoles
from prefill_first_gdn_owned_storage_v1 import f16_rne
from native_storage_first_gdn_estimate_v1 import bf16_rne

class CompositionTests(unittest.TestCase):
 def setUp(self):self.p=SyntheticOriginalRoles();self.model=Full48OwnedComposition(self.p,'a'*64)
 def tearDown(self):gc.collect()
 def test_synthetic_fastpath_equal_dense_original_row_algebra(self):
  x=np.linspace(-.17,.23,2560);name='blk.2.ffn_gate_exps.weight'
  for seam in ('original','f16','bf16'):
   dense=self.p.rows(name,range(7));dense=f16_rne(dense) if seam=='f16' else bf16_rne(dense) if seam=='bf16' else dense
   self.assertTrue(np.array_equal(dense@x,self.p.synthetic_matmul(name,range(7),x,seam)))
 def test_prefix1_real_role_geometry_all48_own_state_head(self):
  result=self.model.tokens([19]);self.assertEqual(len(result['trace']),1);self.assertEqual(len(result['trace'][0]['layers']),48);self.assertEqual(result['first_generated_logits'].shape,(248320,));self.assertEqual(len(result['gdn_states_owned']),36);self.assertEqual(len(result['qsa_states_owned']),12);self.assertTrue(np.isfinite(result['first_generated_logits']).all());self.assertFalse(result['captured_inputs_used']);self.assertFalse(result['actual_original_payload_used']);self.assertFalse(result['full_model_math_qualified']);self.assertIsNone(result['tolerance_gate']);self.assertEqual(result['last_two_owned'],[-1,19]);self.assertLessEqual(self.p.max_rows_bytes,64<<20)
 def test_phase_and_history_owned_prefix2(self):
  a=self.model.tokens([19,22]);self.assertEqual([r['route'] for r in a['trace']],['prefill','verifier']);state=a['gdn_states_owned'][0]['recurrent'].copy();head=a['first_generated_logits'].copy();del a;gc.collect();b=self.model.tokens([22,19]);self.assertFalse(np.array_equal(state,b['gdn_states_owned'][0]['recurrent']));self.assertFalse(np.array_equal(head,b['first_generated_logits']));self.assertEqual(b['last_two_owned'],[22,19])
 def test_repeated_prefix_zero_reset_and_no_input_state_api(self):
  a=self.model.tokens([19]);head=a['first_generated_logits'].copy();del a;gc.collect();b=self.model.tokens([19]);self.assertTrue(np.array_equal(head,b['first_generated_logits']))
  with self.assertRaises(TypeError):self.model.tokens([19],state=b['gdn_states_owned'])
 def test_prefix4_and8_actual_phase_roster(self):
  for ids in ([19,22,23,25],[19,22,23,25,27,29,31,33]):
   result=self.model.tokens(ids);self.assertEqual([row['route'] for row in result['trace']],['prefill']*(len(ids)-1)+['verifier']);self.assertEqual(result['last_two_owned'],ids[-2:]);self.assertEqual(len(result['qsa_states_owned'][3].keys),len(ids));self.assertEqual(len(result['trace'][-1]['layers']),48);self.assertFalse(result['nativebitwise_qualified']);del result;gc.collect()
 def test_true_prefill_not_decode_substitute(self):
  embedding=self.p.rows('token_embd.weight',[19])[0];residual=np.broadcast_to(embedding,(4,2560));mixed=self.model.hc_read(residual,'blk.0.hc_attn_')['mixed'];gdn=self.model.gdn[0];a,_=gdn.mixer(mixed,gdn.initial_state(),'prefill');b,_=gdn.mixer(mixed,gdn.initial_state(),'verifier');self.assertFalse(np.array_equal(a['conv'],b['conv']));self.assertFalse(np.array_equal(a['recurrent'],b['recurrent']))
 def test_role_route_source_identity_negative(self):
  for options in ({'STRATA_PREFILL_MMQ':'1'},{'STRATA_PREFILL_BF16X2':'1'},{'STRATA_SYCL_NATIVE_HC':'0'},{'STRATA_GDN_REC_HEADS':'0'},{'STRATA_WMMA_GEMM':'1'}):
   with self.assertRaises(NotImplementedError):Full48OwnedComposition(self.p,'a'*64,runtime_options=options)
  with self.assertRaises(ValueError):Full48OwnedComposition(self.p,'wrong')
  self.p.reader.tensors['blk.47.ffn_down_exps.weight'][1]['type']='Q5_1'
  with self.assertRaises(ValueError):Full48OwnedComposition(self.p,'a'*64)
 def test_nonfinite_original_projection_refused(self):
  with self.assertRaises(ValueError):self.model.projector.project('output.weight',np.full(2560,np.nan),activation='q8_1')
 def test_prefix_roster_and_noninteger_negative(self):
  for ids in ([],[1,2,3],[1]*9,[-1],[248320],[True]):
   with self.assertRaises(ValueError):self.model.tokens(ids)

if __name__=='__main__':unittest.main()
