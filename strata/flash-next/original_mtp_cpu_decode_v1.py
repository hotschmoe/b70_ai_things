"""Independent original MTP Q5_0/Q6_K extension; existing decoder unchanged."""
import numpy as np
import original_gguf_vector_decoder_v2 as original
GEOMETRY={**original.GEOMETRY,'Q5_0':(32,22),'Q6_K':(256,210)}
def decode(raw,kind,lane='ggml_f32'):
 if kind not in ('Q5_0','Q6_K'):return original.decode(raw,kind,lane)
 require_lane=lane in('ggml_f32','original_fp64')
 if not require_lane or len(raw)%GEOMETRY[kind][1]:raise ValueError('Whole original blocks/known lane required')
 p=np.frombuffer(raw,dtype=np.uint8).reshape(-1,GEOMETRY[kind][1]);n=len(p)
 rnd=lambda x:x.astype(np.float32).astype(np.float64) if lane=='ggml_f32' else x
 if kind=='Q5_0':
  d=p[:,:2].copy().view('<f2').reshape(n,1).astype(np.float64)
  lo=np.concatenate((p[:,6:]&15,p[:,6:]>>4),axis=1)
  hi=np.unpackbits(p[:,2:6],axis=1,bitorder='little')*16
  out=rnd(d*((lo|hi).astype(np.int16)-16))
 else:
  low=p[:,:128].reshape(n,2,64);high=p[:,128:192].reshape(n,2,32)
  codes=np.stack(((low[:,:,:32]&15)|((high&3)<<4),
                  (low[:,:,32:]&15)|(((high>>2)&3)<<4),
                  (low[:,:,:32]>>4)|(((high>>4)&3)<<4),
                  (low[:,:,32:]>>4)|(((high>>6)&3)<<4)),axis=2).astype(np.int16)-32
  scales=p[:,192:208].copy().view(np.int8).reshape(n,2,4,2).astype(np.float64)
  scales=np.repeat(scales,16,axis=3)
  d=p[:,208:].copy().view('<f2').reshape(n,1,1,1).astype(np.float64)
  out=rnd(rnd(d*scales)*codes)
 out=out.reshape(-1)
 if not np.isfinite(out).all():raise ValueError('Nonfinite original block')
 return out
