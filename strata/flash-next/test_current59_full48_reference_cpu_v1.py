import unittest
import numpy as np
import hc_f32_arithmetic35_host_v2 as arithmetic
from original_q80_q81_wide32_reference_v1 import bf16_gdn_ab_linear_owned
from current59_full48_mode0_reference_v1 import bf16_source_rows,ConsumerProjector,InlineHalfMode0State
class CpuRows:
 def rows(self,w,x):
  out=[]
  for row in w:
   lanes=np.zeros(32,np.float32)
   for lane in range(32):
    value=0.
    for col in range(lane,len(x),32):value=float(arithmetic._fma(float(row[col]),float(x[col]),value))
    lanes[lane]=value
   for mask in(16,8,4,2,1):lanes=np.asarray([np.float32(lanes[i]+lanes[i^mask])for i in range(32)],np.float32)
   out.append(lanes[0])
  return np.asarray(out,np.float32)
class ConsumerTests(unittest.TestCase):
 def test_group8_transport_preserves_executed_first_GDN_arithmetic(self):
  rng=np.random.default_rng(3);w=rng.normal(size=(2,2560)).astype(np.float32);x=rng.normal(size=2560).astype(np.float32)
  a=bf16_source_rows(w,x,CpuRows(),'gdn_AB_group8');b=bf16_gdn_ab_linear_owned(w,x)['output'];self.assertEqual(a.tobytes(),b.tobytes())
 def test_MMVF_block_tree_constant_and_unknown_topology(self):
  a=bf16_source_rows(np.ones((2,2560),np.float32),np.ones(2560,np.float32),CpuRows(),'BF16_MMVF_pair2_block256');self.assertEqual(a.tobytes(),np.asarray([2560,2560],dtype='<f4').tobytes())
  with self.assertRaises(ValueError):bf16_source_rows(np.ones((1,2560)),np.ones(2560),CpuRows(),'unknown')
 def test_declared_unknown_consumer_fails_and_no_mode1_storage_object(self):
  class Provider:
   def shape(self,name):return (2560,1)
   def kind(self,name):return 'F32'
  p=ConsumerProjector(Provider(),CpuRows())
  with self.assertRaisesRegex(ValueError,'Unclassified'):p.project('blk.0.unknown.weight',np.ones(2560))
  self.assertNotIn('materialize',InlineHalfMode0State._pool.__code__.co_names)
