#!/usr/bin/env python3
"""Only synthetic packed blocks/source indexing; no actual model/GPU."""
import unittest
import numpy as np
from q5k_q8_8lane_storage_estimate_v1 import row_dot
from independent_q8_1_activation_v1 import encode,decode
from original_gguf_vector_decoder_v2 import decode as weights

class LaneTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(239);self.raw=bytearray(176);self.raw[:4]=np.asarray([.125,.0625],dtype='<f2').tobytes();self.raw[4:]=rng.integers(0,256,172,dtype=np.uint8).tobytes();self.x=np.linspace(-.937,1.873,256,dtype='<f4');self.packet=encode(self.x.tobytes(),1,256)
 def test_q5k_integer_layout_and_native_partial_mathematics(self):
  row=row_dot(bytes(self.raw),'Q5_K',self.packet);wanted=weights(bytes(self.raw),'Q5_K')@decode(self.packet,1,256)['reconstructed_activation'].reshape(-1);self.assertTrue(row['all_columns_once']);self.assertEqual(row['piece_count'],16);self.assertEqual(row['unrounded_FP64_piece_algebra'],float(wanted));self.assertFalse(row['native_compiler_FMA_or_reductions_qualified']);self.assertIsNone(row['tolerance_gate'])
 def test_q8_symmetric_onehot_exact_dyadic(self):
  b=np.asarray([.125],dtype='<f2').tobytes()+np.arange(-16,16,dtype=np.int8).tobytes();x=np.zeros(32,dtype='<f4');x[5]=1.;packet=encode(x.tobytes(),1,32);row=row_dot(b,'Q8_0',packet);expected=float(weights(b,'Q8_0')@decode(packet,1,32)['reconstructed_activation'].reshape(-1));self.assertEqual(row['output_estimate'],expected);self.assertEqual(row['piece_count'],4)
 def test_row_2560_and_640_source_geometry(self):
  packet=encode(np.tile(self.x,10).astype('<f4').tobytes(),1,2560);row=row_dot(bytes(self.raw)*10,'Q5_K',packet);self.assertEqual(row['piece_count'],160)
  raw=np.asarray([.00390625],dtype='<f2').tobytes()+bytes(range(32));packet=encode(np.linspace(-.27,.37,640,dtype='<f4').tobytes(),1,640);row=row_dot(raw*20,'Q8_0',packet);self.assertEqual(row['piece_count'],80)
 def test_stored_raw_s_has_no_effect(self):
  before=row_dot(bytes(self.raw),'Q5_K',self.packet);changed=bytearray(self.packet)
  for block in range(8):changed[block*36+2:block*36+4]=np.asarray([11.5+block],dtype='<f2').tobytes()
  self.assertEqual(before,row_dot(bytes(self.raw),'Q5_K',bytes(changed)))
 def test_fifth_bit_min_and_badextent_fail_controls(self):
  before=row_dot(bytes(self.raw),'Q5_K',self.packet)['output_estimate'];raw=self.raw[:];raw[16:48]=bytes(32);self.assertNotEqual(before,row_dot(bytes(raw),'Q5_K',self.packet)['output_estimate']);raw=self.raw[:];raw[2:4]=bytes(2);self.assertNotEqual(before,row_dot(bytes(raw),'Q5_K',self.packet)['output_estimate'])
  for raw,kind,lane in [(bytes(self.raw[:-1]),'Q5_K',8),(bytes(self.raw),'Q4_K',8),(bytes(self.raw),'Q5_K',32)]:
   with self.assertRaises(ValueError):row_dot(raw,kind,self.packet,lane)

if __name__=='__main__':unittest.main()
