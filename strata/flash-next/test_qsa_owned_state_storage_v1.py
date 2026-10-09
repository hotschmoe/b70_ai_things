#!/usr/bin/env python3
"""Bounded synthetic QSA history/selection/checkpoint tests; no model reads."""
import copy,unittest
import numpy as np
from qsa_owned_state_storage_v1 import QsaOwnedState,QsaGeometry,rope,QsaConditionalProjection
from types import SimpleNamespace

class QsaTests(unittest.TestCase):
 def setUp(self):
  self.g=QsaGeometry(heads=24,kv_heads=2,dim=4,rot=4,idx_heads=4,idx_dim=4,topk=8);self.s=QsaOwnedState(self.g,max_cells=128)
 def inputs(self,index):
  g=self.g;rng=np.random.default_rng(index+61)
  return [rng.normal(size=(24,4)),rng.normal(size=(2,4)),rng.normal(size=(2,4)),np.zeros((24,4)),np.full(4,index+.124),np.zeros((4,4))]
 def step(self,index,evaluate=True):return self.s.advance(index,*self.inputs(index),evaluate_attention=evaluate)
 def test_pool4_first_spare_tail_dead(self):
  self.step(0);dead=self.s.dead.copy();self.assertTrue(np.array_equal(self.s.pooled[0],dead));self.assertEqual(self.s.block_pos,0)
  for i in (1,2):self.step(i)
  self.assertTrue(np.array_equal(self.s.pooled[0],dead));self.step(3);self.assertEqual(len(self.s.pooled),2);self.assertTrue(np.array_equal(self.s.pooled[1],dead))
  self.step(4);self.assertTrue(np.array_equal(self.s.dead,dead));self.assertEqual(self.s.tail[0,0],float(np.float16(4.124)))
 def test_64_history_boundary_is_16_fourcell_pages(self):
  for i in range(65):result=self.step(i,evaluate=False)
  self.assertEqual(result['completed_indexer_blocks'],16);self.assertEqual(self.s.resident_pool()['k'].shape,(17,2,4,4));self.assertEqual(self.s.block_pos,60)
 def test_causal_tail_bias_lowIDties_ascending(self):
  for i in range(14):result=self.step(i,evaluate=False)
  self.assertEqual(result['selected_ids_owned'].tolist(),list(range(9))+[12,13]);self.assertTrue(np.min(result['scores'][12:])>np.max(result['scores'][:12]));self.assertTrue(np.all(result['selected_ids_owned']<14))
 def test_exact_boundary_empty_spare_never_selected(self):
  for i in range(16):result=self.step(i,evaluate=False)
  self.assertEqual(result['selected_ids_owned'].tolist(),list(range(11)));self.assertTrue(np.all(result['scores']==0))
 def test_perhead_relu_not_relu_ofsum(self):
  self.step(0,evaluate=False);self.s.dead=np.array([1.,0,0,0]);self.s.pooled[0]=self.s.dead.copy();q=np.zeros((4,4));q[0,0]=2;q[1,0]=-3
  scores,_=self.s.scores_and_selection(q);self.assertEqual(scores[0],1e9) # F32 1e9 loses +2; test complete block below
  for i in (1,2,3):self.step(i,evaluate=False)
  self.s.pooled[0]=np.array([1.,0,0,0]);scores,_=self.s.scores_and_selection(q);self.assertEqual(scores[0],2.)
 def test_24_query_heads_divide_to_twoKV(self):
  q=np.zeros((24,4));k=np.zeros((2,4));v=np.array([[2.]*4,[-3.]*4]);result=self.s.advance(1,q,k,v,q,np.zeros(4),np.zeros((4,4)))
  out=result['attention_gated_F32'];self.assertTrue(np.all(out[:12]==1));self.assertTrue(np.all(out[12:]==-1.5));self.assertFalse(np.array_equal(out,np.tile([[1.]*4,[-1.5]*4],(12,1))))
 def test_checkpoint_main_slot_resume_equals_uninterrupted(self):
  for i in range(7):self.step(i,evaluate=False)
  snap=self.s.checkpoint();slot=QsaOwnedState(self.g,max_cells=128);slot.restore(snap,list(range(7)))
  expected=self.step(7);actual=slot.advance(7,*self.inputs(7));self.assertTrue(np.array_equal(expected['attention_gated_F32'],actual['attention_gated_F32']));self.assertEqual(slot.history_digest(),self.s.history_digest())
 def test_wrong_checkpoint_tokens_tail_dead_blockpos_rejected(self):
  for i in range(7):self.step(i,evaluate=False)
  snap=self.s.checkpoint();other=QsaOwnedState(self.g,max_cells=128)
  with self.assertRaises(ValueError):other.restore(snap,list(reversed(range(7))))
  for field in ('tail','dead','block_pos'):
   bad=copy.deepcopy(snap)
   if field=='block_pos':bad[field]+=4
   else:bad[field].flat[0]+=.1
   with self.assertRaises(ValueError):other.restore(bad,list(range(7)))
 def test_reordered_records_even_when_labels_same_rejected(self):
  for i in range(7):self.step(i,evaluate=False)
  snap=self.s.checkpoint();snap['records'][0]['inputs'],snap['records'][1]['inputs']=snap['records'][1]['inputs'],snap['records'][0]['inputs']
  with self.assertRaises(ValueError):QsaOwnedState(self.g,max_cells=128).restore(snap,list(range(7)))
 def test_reset_independent_session_history(self):
  a=self.step(0);self.step(1);self.s.reset();b=self.step(0);self.assertTrue(np.array_equal(a['attention_gated_F32'],b['attention_gated_F32']));self.assertEqual(len(self.s.keys),1)
 def test_fp16_storage_and_partialneox(self):
  self.step(0);self.assertTrue(np.array_equal(self.s.keys[0],self.s.keys[0].astype('<f2').astype(np.float64)))
  g=QsaGeometry(dim=8,rot=4,idx_dim=8);x=np.arange(8,dtype=float);y=rope(x,3,g);self.assertTrue(np.array_equal(x[4:],y[4:]));self.assertFalse(np.array_equal(x[:4],y[:4]))
 def test_supplied_selectedIDs_invalid_refused(self):
  self.step(0)
  with self.assertRaises(ValueError):self.s.attention(np.zeros((24,4)),np.zeros((24,4)),np.array([1]))
 def test_actual_2048plus3_selection_width_boundary(self):
  g=QsaGeometry(heads=24,kv_heads=2,dim=4,rot=4,idx_heads=4,idx_dim=4,topk=2048);state=QsaOwnedState(g,max_cells=2112)
  zeros=[np.zeros((24,4)),np.zeros((2,4)),np.zeros((2,4)),np.zeros((24,4)),np.zeros(4),np.zeros((4,4))]
  for index in range(2052):result=state.advance(index,*zeros,evaluate_attention=False)
  self.assertEqual(len(result['selected_ids_owned']),2051);self.assertEqual(result['selected_ids_owned'][-1],2050);self.assertEqual(len(state.keys),2052)
 def test_header_role_validation_reads_no_weights(self):
  tensors={'blk.3.'+role:(None,{'type':kind,'shape_ggml_order':list(shape)},None) for role,(kind,shape) in QsaConditionalProjection.ROLES.items()}
  def forbid(*args):raise AssertionError('No model weights read')
  provider=SimpleNamespace(reader=SimpleNamespace(tensors=tensors),rows=forbid);QsaConditionalProjection(provider)
  tensors['blk.3.indexer.q_proj.weight'][1]['type']='F32'
  with self.assertRaises(ValueError):QsaConditionalProjection(provider)
 def test_scope_fullmodel_native_gates_false(self):
  q=self.s.qualification();self.assertFalse(q['captured_selected_ids_used']);self.assertTrue(q['conditional_on_supplied_layer_inputs']);self.assertFalse(q['full_model_math_qualified']);self.assertIsNone(q['tolerance_gate'])

if __name__=='__main__':unittest.main()
