#!/usr/bin/env python3
"""Synthetic source/storage controls; no actual GGUF payload or GPU work."""
import unittest
import numpy as np
from original_first_gdn_layer_v1 import Geometry
from test_native_storage_first_gdn_estimate_v1 import ToyRows
from native_storage_first_gdn_estimate_v1 import OwnPrefixStorageEstimate,bf16_rne
from prefill_first_gdn_owned_storage_v1 import OwnedPrefillGdnStorage,f16_rne

class StorageRows(ToyRows):
 def kind(self,name):
  role=name.removeprefix('blk.0.')
  return 'Q8_0' if role in ('token_embd.weight','hc_attn_down.weight','hc_attn_up.weight','attn_qkv.weight','attn_gate.weight','ssm_out.weight') else super().kind(name)
 def __init__(self,g):
  super().__init__(g)
  # Synthetic Q8_0 encoded-real half-scale times signed codes, not random F64
  # falsely labelled original weights. Other F32 source roles explicitly stored.
  for name,values in self.values.items():
   if self.kind(name)=='Q8_0':
    blocks=values.reshape(-1,32);d=(np.max(np.abs(blocks),axis=1)/127).astype('<f2').astype(np.float64);codes=np.clip(np.rint(blocks/d[:,None]),-127,127);self.values[name]=(codes*d[:,None]).reshape(values.shape)
   else:self.values[name]=values.astype('<f4').astype(np.float64)

class PrefillTests(unittest.TestCase):
 def setUp(self):
  self.g=Geometry(embd=32,streams=2,low_rank=32,state=8,key_heads=4,value_heads=4,experts=4,topk=2,ffn=32);self.p=StorageRows(self.g);self.layer=OwnedPrefillGdnStorage(self.p,self.g)
 def test_true_prefill_then_last_verifier_routes(self):
  result=self.layer.tokens([1,2,3,4]);self.assertEqual([row['route'] for row in result['rows']],['prefill']*3+['verifier']);self.assertEqual({event['position'] for event in result['packet_events']},{3});self.assertFalse(result['captured_inputs_used']);self.assertIsNone(result['tolerance_gate']);self.assertFalse(result['full_model_math_qualified'])
 def test_own_repeated_prefix_resets_from_zero(self):
  a=self.layer.tokens([1,2]);b=self.layer.tokens([1,2]);self.assertTrue(np.array_equal(a['rows'][-1]['incoming_state_owned']['recurrent'],b['rows'][-1]['incoming_state_owned']['recurrent']));self.assertEqual(np.count_nonzero(a['rows'][0]['incoming_state_owned']['recurrent']),0);self.assertGreater(np.count_nonzero(a['rows'][1]['incoming_state_owned']['recurrent']),0)
 def test_divergent_history_and_order_changes_state(self):
  a=self.layer.tokens([1,2,3]);b=self.layer.tokens([2,1,3]);c=self.layer.tokens([1,4,3]);self.assertFalse(np.array_equal(a['rows'][-1]['incoming_state_owned']['recurrent'],b['rows'][-1]['incoming_state_owned']['recurrent']));self.assertFalse(np.array_equal(a['rows'][-1]['incoming_state_owned']['conv'],c['rows'][-1]['incoming_state_owned']['conv']))
 def test_decoder_substitution_changes_earlier_state(self):
  result=self.layer.tokens([1,2]);mixed=result['rows'][0]['hc_attention']['mixed'];wrong,_=self.layer.mixer(mixed,self.layer.initial_state(),'verifier');actual=result['rows'][0]['outgoing_state_owned'];self.assertFalse(np.array_equal(wrong['recurrent'],actual['recurrent']));self.assertFalse(np.array_equal(wrong['conv'],actual['conv']))
 def test_F16_input_and_weight_rounding_both_required(self):
  x=np.linspace(-.777,1.233,32);name='attn_qkv.weight';actual=self.layer.project(name,x,'f16','f16');no_input=self.layer.project(name,x,'f16','f32');no_weight=self.layer.project(name,x,'original','f16');self.assertFalse(np.array_equal(actual,no_input));self.assertFalse(np.array_equal(actual,no_weight))
 def test_BF16_alpha_input_differs_from_verifier_F32(self):
  x=np.linspace(-.771,1.239,32);prefill=self.layer.project('ssm_alpha.weight',x,'bf16','bf16');wrong=self.layer.project('ssm_alpha.weight',x,'bf16','f32');self.assertFalse(np.array_equal(prefill,wrong))
 def test_prefill_output_operand_F16_and_core_y_semantics(self):
  row=self.layer.tokens([1,2])['rows'][0];details=row['mixer'];self.assertEqual(details['prefill_y_buffer_semantics'],'scaled_core_F32');self.assertTrue(np.array_equal(details['output_operand_storage'],f16_rne(details['normalized_gated_mathematical_F32'])));self.assertFalse(np.array_equal(details['core_scaled'],details['normalized_gated_mathematical_F32']))
 def test_half_F32_before_F16_and_overflow(self):
  x=1.0004882812501;self.assertEqual(float(f16_rne([x])[0]),1.);self.assertNotEqual(float(np.float16(x)),1.)
  with self.assertRaises(ValueError):f16_rne([70000.])
 def test_alternative_routes_failclosed_even_serial_flag_zero(self):
  for options in ({'STRATA_PREFILL_BF16X2':'1'},{'STRATA_BF16_TC':'2'},{'STRATA_GDN_REC_HEADS':'0'},{'STRATA_GDN_REC_HEADS':''}):
   with self.assertRaises(NotImplementedError):OwnedPrefillGdnStorage(self.p,self.g,runtime_options=options)
 def test_no_supplied_state_interface(self):
  with self.assertRaises(TypeError):self.layer.tokens([1],state=self.layer.initial_state())
 def test_own_state_is_F32_storage(self):
  row=self.layer.tokens([1,2])['rows'][0]
  for value in row['outgoing_state_owned'].values():self.assertTrue(np.array_equal(value,value.astype('<f4').astype(np.float64)))
 def test_lastincoming_equals_prioroutgoing(self):
  rows=self.layer.tokens([1,2,3,4])['rows']
  for index in range(1,len(rows)):
   for name in ('recurrent','conv'):self.assertTrue(np.array_equal(rows[index]['incoming_state_owned'][name],rows[index-1]['outgoing_state_owned'][name]))

if __name__=='__main__':unittest.main()
