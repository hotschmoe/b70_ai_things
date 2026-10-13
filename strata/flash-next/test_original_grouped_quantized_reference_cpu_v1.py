import unittest
import numpy as np
import original_grouped_quantized_reference_v1 as g
from independent_q8_1_activation_v1 import encode
from original_gguf_vector_decoder_v2 import decode

class Controls(unittest.TestCase):
 def packets(self,kind,rows,cols):
  qk,width,_=g.GEOMETRY[kind];rng=np.random.default_rng(35);b=rng.integers(0,256,(rows,cols//qk,width),dtype=np.uint8)
  for i in range(rows):
   for j in range(cols//qk):
    b[i,j,:2]=np.frombuffer(np.float16(.03125*(j+1)).tobytes(),np.uint8)
    if kind!='Q8_0':b[i,j,2:4]=np.frombuffer(np.float16(.015625).tobytes(),np.uint8)
  return b.tobytes()
 def test_original_code_unpack_algebra(self):
  for kind in g.GEOMETRY:
   cols=256;raw=self.packets(kind,2,cols);d,m,sc,mn,c=g.unpack(raw,kind,2,cols)
   if kind in('Q4_K','Q5_K'):logical=d[:,:,None,None].astype(np.float64)*sc[:,:,:,None]*c-m[:,:,None,None].astype(np.float64)*mn[:,:,:,None]
   else:logical=d[:,:,None].astype(np.float64)*c+(0 if m is None else m[:,:,None].astype(np.float64))
   np.testing.assert_array_equal(logical.reshape(-1),decode(raw,kind))
 def scalar(self,raw,pkt,kind,rows,cols):
  d,m,sc,mn,c=g.unpack(raw,kind,rows,cols);x=np.frombuffer(pkt,np.uint8).reshape(-1,36);xd=x[:,:2].copy().view('<f2').reshape(-1).astype('<f4');q=x[:,4:].view(np.int8).astype(int);qk,_,ipb=g.GEOMETRY[kind];f=np.float32;out=[]
  for r in range(rows):
   a=np.zeros(8,np.float32)
   for lane in range(8):
    for k in range(lane,cols//qk*ipb,8):
     kb,p=divmod(k,ipb)
     if kind in('Q4_K','Q5_K'):
      sd=f(0);sm=f(0)
      for group in(2*(p//4),2*(p//4)+1):
       indices=list(range(4*(p%4),4*(p%4)+4))+list(range(16+4*(p%4),20+4*(p%4)));dot=sum(int(c[r,kb,group,j])*int(q[kb*8+group,j])for j in indices);s=sum(int(q[kb*8+group,j])for j in indices)
       sd=f(sd+f(xd[kb*8+group]*f(dot*int(sc[r,kb,group]))));sm=f(sm+f(xd[kb*8+group]*f(s*int(mn[r,kb,group]))))
      term=f(f(d[r,kb]*sd)-f(m[r,kb]*sm))
     else:
      indices=list(range(8*p,8*p+8))
      if kind=='Q5_1':indices+=list(range(16+8*p,24+8*p))
      dot=sum(int(c[r,kb,j])*int(q[kb,j])for j in indices);term=f(f(d[r,kb]*xd[kb])*f(dot))
      if kind=='Q5_1':term=f(term+f(f(m[r,kb]*xd[kb])*f(sum(int(q[kb,j])for j in indices))))
     a[lane]=f(a[lane]+term)
   for mask in(4,2,1):a=np.asarray([f(a[j]+a[j^mask])for j in range(8)],np.float32)
   out.append(a[0])
  return np.asarray(out,np.float32)
 def test_separate_scalar_lane_chain(self):
  for kind in g.GEOMETRY:
   raw=self.packets(kind,3,512);x=np.random.default_rng(9).normal(size=512).astype('<f4');pkt=encode(x.tobytes(),1,512)
   np.testing.assert_array_equal(g.project_packets(raw,pkt,kind,3,512),self.scalar(raw,pkt,kind,3,512))
 def test_quantized_sum_not_original_sum(self):
  raw=self.packets('Q5_1',1,256);pkt=bytearray(encode(np.linspace(-.6,.7,256,dtype='<f4').tobytes(),1,256));prior=g.project_packets(raw,bytes(pkt),'Q5_1',1,256)
  for i in range(8):pkt[i*36+2:i*36+4]=np.float16(100).tobytes()
  np.testing.assert_array_equal(prior,g.project_packets(raw,bytes(pkt),'Q5_1',1,256))
 def test_bounds(self):
  raw=self.packets('Q4_K',1,256);pkt=encode(np.ones(256,dtype='<f4').tobytes(),1,256)
  for args in((raw[:-1],pkt,'Q4_K',1,256,8),(raw,pkt[:-1],'Q4_K',1,256,8),(raw,pkt,'Q4_K',1,256,32)):
   with self.assertRaises(ValueError):g.project_packets(*args)
if __name__=='__main__':unittest.main()
