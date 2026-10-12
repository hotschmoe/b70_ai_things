import unittest
import numpy as np
import original_q80_q81_wide32_reference_v1 as r
class Arithmetic(unittest.TestCase):
 def packets(self,rows,cols):
  blocks=cols//32;w=np.zeros((rows,blocks,34),np.uint8);x=np.zeros((blocks,36),np.uint8)
  w[:,:,:2]=np.full((rows,blocks),.5,dtype='<f2').view(np.uint8).reshape(rows,blocks,2)
  x[:,:2]=np.full(blocks,.25,dtype='<f2').view(np.uint8).reshape(blocks,2)
  w[:,:,2:]=np.int8(2).view(np.uint8);x[:,4:]=np.int8(3).view(np.uint8)
  return w,x
 def test_integer_dot_scale_and_ignored_sum(self):
  w,x=self.packets(2,640);out=r.project_packets(w.tobytes(),x.tobytes(),2,640)
  self.assertEqual(out.tobytes(),np.full(2,640*.5*.25*6,dtype='<f4').tobytes())
  x[:,2:4]=np.arange(40,dtype=np.uint8).reshape(20,2)
  self.assertEqual(out.tobytes(),r.project_packets(w.tobytes(),x.tobytes(),2,640).tobytes())
 def test_independent_scalar_source_order(self):
  rng=np.random.default_rng(4);rows=3;cols=1024;w,x=self.packets(rows,cols)
  w[:,:,2:]=rng.integers(-127,128,(rows,32,32),dtype=np.int16).astype(np.int8).view(np.uint8)
  x[:,4:]=rng.integers(-127,128,(32,32),dtype=np.int16).astype(np.int8).view(np.uint8)
  scales=np.asarray([2**-10,2**-3,1,16]*8,dtype='<f2');x[:,:2]=scales.view(np.uint8).reshape(32,2)
  result=r.project_packets(w.tobytes(),x.tobytes(),rows,cols);expected=[]
  for row in range(rows):
   lanes=np.zeros(32,np.float32)
   for lane in range(32):
    for kb in range(lane//2,32,16):
     at=16*(lane%2);dot=sum(int(a)*int(b)for a,b in zip(w[row,kb,2:].view(np.int8)[at:at+16],x[kb,4:].view(np.int8)[at:at+16]))
     term=np.float32(np.float32(np.float32(.5)*np.float32(scales[kb]))*np.float32(dot));lanes[lane]=np.float32(lanes[lane]+term)
   for mask in (16,8,4,2,1):lanes=np.asarray([np.float32(lanes[i]+lanes[i^mask])for i in range(32)],np.float32)
   expected.append(lanes[0])
  self.assertEqual(result.tobytes(),np.asarray(expected,dtype='<f4').tobytes())
 def test_bounds_and_owned_operand(self):
  w,x=self.packets(1,32)
  with self.assertRaises(ValueError):r.project_packets(w.tobytes()[:-1],x.tobytes(),1,32)
  with self.assertRaises(ValueError):r.project_packets(w.tobytes(),x.tobytes(),True,32)
  with self.assertRaises(ValueError):r.project_owned(w.tobytes(),np.full(32,np.nan),1)
  result=r.project_owned(w.tobytes(),np.linspace(-1,1,32,dtype=np.float32),1)
  self.assertFalse(result['full_model_math_qualified']);self.assertFalse(result['device_compiler_contraction_qualified'])
 def test_bf16_group8_owned_linear(self):
  w=np.ones((2,256),np.float32);x=np.ones(256,np.float32)
  result=r.bf16_gdn_ab_linear_owned(w,x)
  self.assertEqual(result['output'].tobytes(),np.asarray([256,256],dtype='<f4').tobytes())
  self.assertFalse(result['native_exp_log_softplus_modeled']);self.assertFalse(result['full_model_math_qualified'])
  with self.assertRaises(ValueError):r.bf16_gdn_ab_linear_owned(w,x[:-1])
