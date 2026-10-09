#!/usr/bin/env python3
"""CPU synthetic source/storage adapter; no original model payload or GPU."""
import unittest
import numpy as np
from original_first_gdn_layer_v1 import Geometry
from ffn_full48_owned_storage_v1 import OwnedFfnStorage

class ToyOriginal:
 actual_source=False
 def __init__(self,g,layer):self.g=g;self.layer=layer
 def role(self,name):return name.split('.ffn_')[1].removesuffix('.weight')
 def shape(self,name):
  g=self.g;role=self.role(name)
  if role=='gate_inp':return (g.embd,g.experts)
  if role=='gate_inp_shexp':return (g.embd,)
  base=(g.ffn,g.embd) if role.startswith('down_') else (g.embd,g.ffn)
  return base+(g.experts,) if role.endswith('_exps') else base
 def kind(self,name):
  role=self.role(name)
  if role.startswith('gate_inp'):return 'F32'
  if role.endswith('_shexp'):return 'Q8_0'
  if role.startswith('down_'):return 'Q8_0' if self.layer in (2,4,30,46,47) else 'Q5_1'
  return 'Q5_K' if self.layer==2 else 'Q4_K'
 def rows(self,name,indices):
  size=self.shape(name)[0];result=[]
  for index in indices:
   # Synthetic exact dyadic encoded-real weight values; no claimdecoder oracle.
   values=(np.arange(size)%7-3)*(index%11+1)/128
   if self.kind(name)=='F32':values+=.0000123*(index+1)
   result.append(values)
  return np.asarray(result)

class FfnTests(unittest.TestCase):
 def setUp(self):self.g=Geometry(embd=32,streams=4,low_rank=32,state=8,key_heads=4,value_heads=4,experts=4,topk=2,ffn=32);self.x=np.linspace(-.233,.367,32)
 def model(self,layer):return OwnedFfnStorage(ToyOriginal(self.g,layer),layer,self.g)
 def test_all48_original_format_exceptions_and_owned_routing(self):
  for layer in range(48):
   model=self.model(layer);result=model.run(self.x,'verifier');self.assertEqual(len(result['router_ids_owned']),2);self.assertFalse(result['supplied_intermediates_used']);self.assertFalse(result['full_model_math_qualified']);self.assertIsNone(result['tolerance_gate'])
   key='blk.%d.ffn_gate_exps.weight'%layer;self.assertEqual(result['original_format_contract'][key],'Q5_K' if layer==2 else 'Q4_K')
 def test_prefill_F16_path_differs_from_verifier_Q8_1(self):
  model=self.model(2);a=model.run(self.x,'prefill');b=model.run(self.x,'verifier');self.assertFalse(np.array_equal(a['block_output'],b['block_output']))
 def test_prefill_hidden_saturation_and_nan_notmasked(self):
  model=self.model(0);value=model.hidden(np.full(32,10000.),np.full(32,10000.),'prefill');self.assertTrue(np.all(value==65504))
  with self.assertRaises(ValueError):model.hidden(np.full(32,np.nan),np.ones(32),'prefill')
 def test_input_changes_owned_selection_or_output(self):
  model=self.model(4);a=model.run(self.x,'verifier');b=model.run(-self.x,'verifier');self.assertFalse(np.array_equal(a['block_output'],b['block_output']))
 def test_source_type_and_prefill_route_failclosed(self):
  provider=ToyOriginal(self.g,2);provider.kind=lambda name:'Q4_K'
  with self.assertRaises(ValueError):OwnedFfnStorage(provider,2,self.g)
  with self.assertRaises(NotImplementedError):OwnedFfnStorage(ToyOriginal(self.g,2),2,self.g,prefill_profile='guess')
 def test_runtime_alternative_route_rejected(self):
  for flag in ('STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_PF_FUSED','STRATA_PREFILL_MMQ'):
   with self.assertRaises(NotImplementedError):OwnedFfnStorage(ToyOriginal(self.g,0),0,self.g,runtime_options={flag:'1'})

if __name__=='__main__':unittest.main()
