"""Tiny known-answer math and source/provenance negatives, no model/device."""
import copy,unittest
import numpy as np
import owned_layer3_qsa_contract_v1 as q
class Tests(unittest.TestCase):
 def test_known_norm256_two_stage(self):
  for n in (128,256):
   row=np.ones(n,dtype='<f4');r=q.norm256_argument(row,np.float32(1e-6));self.assertEqual(r['square_sum_F32'],float(n));self.assertEqual(r['mean_F32'],1.);self.assertEqual(r['argument_F32'],float(np.float32(1+np.float32(1e-6))));self.assertFalse(r['device_reduction_observed'])
 def test_distinct_warp_sums_known_answer(self):
  row=np.repeat(np.arange(1,9,dtype='<f4'),32);r=q.norm256_argument(row,np.float32(1e-6));self.assertEqual(r['square_sum_F32'],6528.);self.assertEqual(r['mean_F32'],25.5)
 def test_norm_column_epsilon_nonfinite_refused(self):
  for row,eps in [(np.ones(64),1e-6),(np.full(256,np.nan),1e-6),(np.ones(256),0.)]:
   with self.assertRaises(ValueError):q.norm256_argument(row,eps)
 def test_source_multiply_order_explicit(self):
  x=np.linspace(.125,2.,256,dtype='<f4');g=np.linspace(.25,3.,256,dtype='<f4');rs=np.float32(1.234567);self.assertEqual(q.weighted_norm_from_own_rs(x,g,rs).tobytes(),np.asarray(np.asarray(rs*x,dtype='<f4')*g,dtype='<f4').tobytes())
 def test_FP16_storage_and_all_owned_cells(self):
  x=np.full((4,2,256),1.0001,dtype='<f4');k,v=q.fp16_pool_owned(x,x);self.assertEqual(k.tobytes(),x.astype('<f2').astype('<f4').tobytes());self.assertEqual(q.all_cell_ids(4),[0,1,2,3]);self.assertEqual(q.all_cell_ids(1),[0])
 def manifest(self):return {'geometry':q.geometry_contract(),'original_weights_and_embedding_only':True,'own_zero_KV_history':True,'captured_inputs_used':False,'captured_states_used':False,'captured_selected_IDs_used':False,'ULP_fit_or_tolerance_used':False,'own_projection_roles':['attn_q.weight','attn_k.weight','attn_v.weight','indexer.k_proj.weight','indexer.q_proj.weight'],'primitive_labels':['projected_own','norm_argument_own','normalized_own','RoPE_own','FP16_KV_own','indexer_own','scores_own','softmax_own','attention_FMA_own','native_gate_expression_own','Q81_packet_own']}
 def test_actual_intrinsics_not_claimed_by_contract(self):self.assertFalse(q.admission(self.manifest())['actual_runtime_or_math_qualified'])
 def test_captured_state_route_fit_and_false_geometry_refused(self):
  for field in ('captured_inputs_used','captured_states_used','captured_selected_IDs_used','ULP_fit_or_tolerance_used','geometry','own_projection_roles','primitive_labels'):
   r=self.manifest()
   if field=='geometry':r[field]['norm_threads']=32
   elif field in ('own_projection_roles','primitive_labels'):r[field].reverse()
   else:r[field]=True
   with self.assertRaises(ValueError):q.admission(r)
if __name__=='__main__':unittest.main()
