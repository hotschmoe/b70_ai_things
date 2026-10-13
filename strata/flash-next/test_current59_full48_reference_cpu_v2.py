import unittest
from unittest.mock import patch
import numpy as np
import current59_full48_mode0_reference_v2 as v
import test_original_grouped_quantized_reference_cpu_v1 as fixture
from independent_q8_1_activation_v1 import encode
class Provider:
 actual_source=True
 @property
 def reader(self):
  from types import SimpleNamespace
  return SimpleNamespace(tensors={name:(None,{"type":self.kind(name)},None)for name in ("blk.0.ffn_gate_exps.weight","blk.0.ffn_down_shexp.weight")})
 def shape(self,name):return (256,2)if name.endswith('shexp.weight')else(256,2,512)
 def kind(self,name):return 'Q8_0'if name.endswith('shexp.weight')else'Q4_K'
class Routes(unittest.TestCase):
 def test_real_grouped_dispatch_own_packet(self):
  p=v.ConsumerProjector(Provider(),None);raw=fixture.Controls().packets('Q4_K',2,256);x=np.arange(256,dtype='<f4')/100
  with patch.object(v,'original_packed_rows',return_value=raw)as read:
   y=p.project('blk.0.ffn_gate_exps.weight',x,activation='q8_1',expert=7)
  self.assertEqual(read.call_args.args[2],[14,15]);np.testing.assert_array_equal(y,v.project_packets(raw,encode(x.tobytes(),1,256),'Q4_K',2,256));self.assertEqual(p.events[0]['expert'],7);self.assertFalse(p.events[0]['stored_s_used'])
 def test_shared_actual_wide32_route(self):
  p=v.ConsumerProjector(Provider(),None);raw=fixture.Controls().packets('Q8_0',2,256);x=np.arange(256,dtype='<f4')/100
  with patch.object(v,'original_rows',return_value=raw):y=p.project('blk.0.ffn_down_shexp.weight',x,activation='q8_1')
  np.testing.assert_array_equal(y,v.wide32(raw,encode(x.tobytes(),1,256),2,256));self.assertEqual(p.contract_events[0]['arithmetic_lane'],'shared_Q80_Q81_half16_wide32')
 def test_explicit_current_only(self):
  p=v.ConsumerProjector(Provider(),None)
  with self.assertRaises(ValueError):p.project('blk.0.ffn_gate_exps.weight',np.ones(256),activation='f32',expert=7)
  with self.assertRaises(ValueError):p.project('blk.0.ffn_gate_exps.weight',np.ones(256),activation='q8_1',expert=True)
if __name__=='__main__':unittest.main()
