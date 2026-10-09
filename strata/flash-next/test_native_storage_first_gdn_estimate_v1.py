#!/usr/bin/env python3
"""CPU source controls; fixture weights, not actual model qualification."""
import unittest
import numpy as np
from original_first_gdn_layer_v1 import Geometry
from original_gguf_vector_decoder_v2 import decode as weights_decode
from independent_q8_1_activation_v1 import encode,decode
from native_storage_first_gdn_estimate_v1 import OwnPrefixStorageEstimate,PACKET_ROLES,BF16_ROLES,bf16_rne

class ToyRows:
 actual_source=False
 def __init__(self,g):
  n=g.embd;d=n*g.streams;c=(2*g.key_heads+g.value_heads)*g.state;z=g.value_heads*g.state;p='blk.0.'
  shapes={'token_embd.weight':(n,7),'attn_qkv.weight':(n,c),'attn_gate.weight':(n,z),'ssm_out.weight':(z,n),'ssm_alpha.weight':(n,g.value_heads),'ssm_beta.weight':(n,g.value_heads),'ssm_a':(g.value_heads,),'ssm_dt.bias':(g.value_heads,),'ssm_norm.weight':(g.state,),'ssm_conv1d.weight':(g.conv_kernel,c),'ffn_gate_inp.weight':(n,g.experts),'ffn_gate_inp_shexp.weight':(n,)}
  for role in ('gate','up','down'):
   shape=(g.ffn,n) if role=='down' else (n,g.ffn)
   shapes['ffn_'+role+'_shexp.weight']=shape;shapes['ffn_'+role+'_exps.weight']=shape+(g.experts,)
  for half in ('attn','ffn'):
   for role,shape in [('norm',(d,)),('down',(d,g.low_rank)),('up',(g.low_rank,d)),('inject',(d,g.streams))]:shapes['hc_'+half+'_'+role+'.weight']=shape
  self.shapes={name if name.startswith('token_') else p+name:shape for name,shape in shapes.items()};rng=np.random.default_rng(13);self.values={}
  for name,shape in self.shapes.items():
   self.values[name]=rng.normal(0,.15,size=(int(np.prod(shape[1:])) if len(shape)>1 else 1,shape[0]))
  self.values[p+'ssm_a'][:]=-.2;self.values[p+'ssm_norm.weight'][:]=1
 def shape(self,name):return self.shapes[name]
 def kind(self,name):return PACKET_ROLES.get(name.removeprefix('blk.0.'),'F32')
 def rows(self,name,indices):return self.values[name][list(indices)].copy()

class StorageTests(unittest.TestCase):
 def setUp(self):
  self.g=Geometry(embd=32,streams=2,low_rank=32,state=8,key_heads=4,value_heads=4,experts=4,topk=2,ffn=32)
  self.provider=ToyRows(self.g)
 def test_own_prefix_state_from_zero_repeated(self):
  layer=OwnPrefixStorageEstimate(self.provider,self.g);a=layer.tokens([1,2],allow_decode_surrogate=True);b=layer.tokens([1,2],allow_decode_surrogate=True)
  self.assertTrue(np.array_equal(a['rows'][-1]['state']['recurrent'],b['rows'][-1]['state']['recurrent']))
  self.assertEqual(np.count_nonzero(a['rows'][0]['details']['mixer']['incoming_recurrent']),0)
  self.assertGreater(np.count_nonzero(a['rows'][1]['details']['mixer']['incoming_recurrent']),0)
  self.assertFalse(a['full_model_math_qualified']);self.assertFalse(a['native_exp_emulated']);self.assertIsNone(a['tolerance_gate'])
 def test_divergent_own_suffix_changes_state(self):
  layer=OwnPrefixStorageEstimate(self.provider,self.g);a=layer.tokens([1,2],allow_decode_surrogate=True);b=layer.tokens([1,3],allow_decode_surrogate=True)
  self.assertFalse(np.array_equal(a['rows'][-1]['state']['recurrent'],b['rows'][-1]['state']['recurrent']))
 def test_reject_actual_prefill_history_substitution(self):
  layer=OwnPrefixStorageEstimate(self.provider,self.g)
  with self.assertRaises(NotImplementedError):layer.tokens([1,2])
  surrogate=layer.tokens([1,2],allow_decode_surrogate=True)
  self.assertFalse(surrogate['captured_prefix_history_comparison_admissible'])
  self.assertFalse(surrogate['actual_prefill_history_supported'])
 def test_dense_shared_expert_packet_boundaries(self):
  layer=OwnPrefixStorageEstimate(self.provider,self.g);result=layer.tokens([1]);roles={e['role'] for e in result['packet_events']}
  self.assertEqual(roles,set(PACKET_ROLES));self.assertFalse(any('hc_' in e['role'] for e in result['packet_events']))
  self.assertTrue(all(not e['stored_sum_used'] for e in result['packet_events']))
 def test_bf16_ties_and_weight_seam(self):
  bits=np.array([0x3f808000,0x3f818000],dtype='<u4');got=bf16_rne(bits.view('<f4')).astype('<f4').view('<u4')
  self.assertEqual(got.tolist(),[0x3f800000,0x3f820000])
  layer=OwnPrefixStorageEstimate(self.provider,self.g);name='blk.0.ssm_alpha.weight'
  self.assertTrue(np.array_equal(layer.p.rows(name,[0]),bf16_rne(self.provider.rows(name,[0]))))
 def test_reject_nonselected_affine_consumer(self):
  self.provider.kind=lambda name:'Q5_0'
  with self.assertRaises(ValueError):OwnPrefixStorageEstimate(self.provider,self.g)
 def test_nonfinite_seam_rejected(self):
  with self.assertRaises(ValueError):bf16_rne([np.nan])
 def test_Q51_quantized_min_algebra_not_stored_sum(self):
  weight=np.zeros(24,dtype=np.uint8);weight[:4]=np.array([.25,.5],dtype='<f2').view(np.uint8);weight[8:]=0x12
  x=np.linspace(-1,2,32,dtype='<f4');packet=encode(x.tobytes(),1,32);image=decode(packet,1,32)
  raw_codes=np.r_[np.full(16,2),np.full(16,1)];codes=image['codes'].reshape(-1).astype(np.int64);scale=float(image['scale'][0]);w=weights_decode(weight.tobytes(),'Q5_1')
  expect=scale*(.25*int(raw_codes@codes)+.5*int(codes.sum()))
  self.assertAlmostEqual(float(w@image['reconstructed_activation'].reshape(-1)),expect,places=12)
  altered=bytearray(packet);altered[2:4]=np.array([700.],dtype='<f2').tobytes()
  self.assertTrue(np.array_equal(decode(bytes(altered),1,32)['reconstructed_activation'],image['reconstructed_activation']))
  wrong=.25*scale*int(raw_codes@codes)+.5*float(image['stored_sum'][0]);self.assertGreater(abs(wrong-expect),1e-5)

 def test_Q4K_quantized_min_algebra(self):
  weight=np.zeros(144,dtype=np.uint8);weight[:4]=np.array([.25,.125],dtype='<f2').view(np.uint8)
  weight[4:8]=1;weight[8:12]=2;weight[12:16]=0x21;weight[16:]=0x43
  x=np.linspace(-.9,1.7,256,dtype='<f4');packet=encode(x.tobytes(),1,256);image=decode(packet,1,256)
  codes=image['codes'].astype(np.int64);dw=.25;minimum=.25
  wcode=np.tile(np.r_[np.full(32,3),np.full(32,4)],4).reshape(8,32)
  expected=np.sum(image['scale']*(dw*np.sum(wcode*codes,axis=1)-minimum*np.sum(codes,axis=1)))
  actual=weights_decode(weight.tobytes(),'Q4_K')@image['reconstructed_activation'].reshape(-1)
  self.assertAlmostEqual(float(actual),float(expected),places=12)

if __name__=='__main__':unittest.main()
