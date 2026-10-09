#!/usr/bin/env python3
"""Source-bound Q5_K mathematical seam controls; no weights/GPU/runtime writes."""
import hashlib,subprocess,unittest
from pathlib import Path
import numpy as np
from original_gguf_vector_decoder_v2 import decode as weight_decode
from independent_q8_1_activation_v1 import encode,decode
BASE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T185206Z-d5q32zc7')
SOURCE=BASE/'source/sycl/src/kernels/cuda/iq_kernels.dp.cpp'

def affine(raw,packet,stored_sum=False):
 d,dm=np.frombuffer(raw[:4],dtype='<f2').astype(np.float64);sc=raw[4:16];image=decode(packet,1,256);codes=image['codes'].reshape(8,32).astype(np.int64);scale=image['scale'].reshape(8).astype(np.float64);s=image['stored_sum'].reshape(8).astype(np.float64);result=0.
 for g in range(8):
  sg=sc[g]&63 if g<4 else (sc[g+4]&15)|((sc[g-4]>>6)<<4)
  mg=sc[g+4]&63 if g<4 else (sc[g+4]>>4)|((sc[g]>>6)<<4)
  qw=np.array([((raw[48+(g//2)*32+j]>>(4*(g%2)))&15)|(((raw[16+j]>>g)&1)<<4) for j in range(32)],dtype=np.int64)
  result+=d*int(sg)*scale[g]*int(qw@codes[g])-dm*int(mg)*(s[g] if stored_sum else scale[g]*int(codes[g].sum()))
 return result

class AffineTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(532);self.raw=bytearray(176);self.raw[:4]=np.array([.125,.0625],dtype='<f2').tobytes();self.raw[4:]=rng.integers(0,256,172,dtype=np.uint8).tobytes();x=np.linspace(-.937,1.873,256,dtype='<f4');self.packet=encode(x.tobytes(),1,256)
 def test_code_sum_equals_original_real_weights_times_reconstructed_x(self):
  wanted=float(weight_decode(bytes(self.raw),'Q5_K')@decode(self.packet,1,256)['reconstructed_activation'].reshape(-1));self.assertAlmostEqual(affine(self.raw,self.packet),wanted,places=10)
 def test_stored_sum_changes_do_not_change_code_sum_consumer(self):
  before=affine(self.raw,self.packet);changed=bytearray(self.packet)
  for block in range(8):changed[block*36+2:block*36+4]=np.array([11.5+block],dtype='<f2').tobytes()
  self.assertEqual(before,affine(self.raw,bytes(changed)));self.assertNotEqual(before,affine(self.raw,bytes(changed),True))
 def test_raw_sum_replacement_is_a_different_math_contract(self):self.assertGreater(abs(affine(self.raw,self.packet)-affine(self.raw,self.packet,True)),1e-5)
 def test_high_bits_and_min_correction_are_not_optional(self):
  raw=self.raw[:];raw[16:48]=bytes(32);self.assertNotEqual(affine(raw,self.packet),affine(self.raw,self.packet));raw=self.raw[:];raw[2:4]=bytes(2);self.assertNotEqual(affine(raw,self.packet),affine(self.raw,self.packet))
 def test_actual_compiled_source_uses_ds0_integer_sums(self):
  text=SOURCE.read_text();impl=text[text.index('__dpct_inline__ float vec_dot_q5_K_q8_1_impl_vmmq('):text.index('// the 6-bit scales')];call=text[text.index('__dpct_inline__ float vec_dot_q5_K_q8_1('):text.index('// Q5_1:')]
  self.assertIn('ggml_cuda_dp4a(0x01010101',impl);self.assertIn('sumf_m += d8[i] * (dot2 * m[i])',impl);self.assertIn('d8[i] = bq8i->ds[0]',call);self.assertNotIn('ds[1]',impl+call)
 def test_actual_SYCL_Q5K_default_lanes_path(self):
  text=SOURCE.read_text();self.assertNotIn('kSplit<13> = true',text);self.assertIn('#define STRATA_EXPERT_LANES 8',text);self.assertIn('launch_gu_lanes<TG>',text);self.assertIn('return vec_dot_q5_K_q8_1(v, y, kbx, iqs)',text)
 def test_pinned_primary_ggml_MMVQ_same_correction(self):
  text=subprocess.check_output(['git','-C',str(BASE/'ggml-source'),'show','3cf03257f219afbe7334045ff7c6a06ac68c627d:ggml/src/ggml-cuda/vecdotq.cuh'],text=True);start=text.index('float vec_dot_q5_K_q8_1_impl_vmmq');end=text.index('// contiguous v/x',start);body=text[start:end];self.assertIn('ggml_cuda_dp4a(0x01010101',body);self.assertIn('sumf_m += d8[i] * (dot2 * m[i])',body);self.assertNotIn('ds8',body)
if __name__=='__main__':unittest.main()
