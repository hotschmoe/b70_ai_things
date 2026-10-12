"""Independent source-expression Q8_0 x Q8_1 wide32 reference candidate.

Original packed weights + caller-owned operands only. Not a device intrinsic or
compiler-contraction oracle. No captured operand primary reference API.
"""
import os
import numpy as np
from independent_q8_1_activation_v1 import encode

def require(ok,message):
 if not ok:raise ValueError(message)
def project_packets(original_q80_rows,own_q81,rows,cols):
 require(type(rows)is int and type(cols)is int and 0<rows<=248320 and 0<cols<=12288 and cols%32==0,'Bounded original wide32 geometry required')
 blocks=cols//32
 require(type(original_q80_rows)is bytes and len(original_q80_rows)==rows*blocks*34 and len(original_q80_rows)<=64<<20,'Complete bounded ORIGINAL Q8_0 rows required')
 require(type(own_q81)is bytes and len(own_q81)==blocks*36,'Complete own Q8_1 packet required')
 w=np.frombuffer(original_q80_rows,dtype=np.uint8).reshape(rows,blocks,34)
 x=np.frombuffer(own_q81,dtype=np.uint8).reshape(blocks,36)
 wd=w[:,:,:2].copy().view('<f2').reshape(rows,blocks).astype('<f4')
 xd=x[:,:2].copy().view('<f2').reshape(blocks).astype('<f4')
 require(np.isfinite(wd).all()and np.isfinite(xd).all(),'Nonfinite original scale refused')
 wq=w[:,:,2:].view(np.int8).astype(np.int32)
 xq=x[:,4:].view(np.int8).astype(np.int32)
 lanes=np.zeros((rows,32),dtype='<f4')
 # Wide32Q8 LPB2: lane2*sub+half, kb=sub +16*k; each half16
 # is an exact signed integer dot, followed by two rounded F32 products.
 for sub in range(16):
  for half in range(2):
   lane=2*sub+half;start=16*half
   for kb in range(sub,blocks,16):
    dot=np.sum(wq[:,kb,start:start+16]*xq[kb,start:start+16],axis=1,dtype=np.int32)
    scale=np.asarray(wd[:,kb]*xd[kb],dtype='<f4')
    term=np.asarray(scale*dot.astype('<f4'),dtype='<f4')
    lanes[:,lane]=np.asarray(lanes[:,lane]+term,dtype='<f4')
 positions=np.arange(32)
 for mask in (16,8,4,2,1):lanes=np.asarray(lanes+lanes[:,positions^mask],dtype='<f4')
 require(np.isfinite(lanes[:,0]).all(),'Nonfinite source-expression reduction refused')
 return lanes[:,0].copy()
def project_owned(original_q80_rows,own_operand,rows):
 operand=np.asarray(own_operand,dtype='<f4')
 require(operand.ndim==1 and np.isfinite(operand).all(),'Finite caller-owned original operand required')
 packet=encode(operand.tobytes(),1,len(operand))
 return {'output':project_packets(original_q80_rows,packet,rows,len(operand)),'own_packet':packet,'reference_lane':'independent_Q80_Q81_wide32_separate_F32_products_adds_v1','integer_half16_dot':True,'lane_block_stride':16,'XOR_masks':[16,8,4,2,1],'stored_Q81_sum_used':False,'device_compiler_contraction_qualified':False,'native_bitwise_qualified':False,'full_model_math_qualified':False}
def original_rows(reader,name,indices):
 """Root-only original bytes; all reads remain inside caller fresh full4 bracket."""
 shard,tensor,sig=reader.tensors[name];indices=list(indices);shape=tensor['shape_ggml_order']
 require(tensor['type']=='Q8_0'and shape[0]%32==0,'Original Q8_0 consumer required')
 count=int(np.prod(shape[1:],dtype=np.int64));width=shape[0]//32*34
 require(indices and len(indices)*width<=min(reader.max_read_bytes,64<<20) and all(type(i)is int and 0<=i<count for i in indices),'Bounded original row indices required')
 fd=os.open(shard['path'],os.O_RDONLY|os.O_NOFOLLOW)
 try:
  def stable():
   s=os.fstat(fd);require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==sig,'Original source row identity changed')
  stable();packets=[]
  for i in indices:
   packet=os.pread(fd,width,tensor['absolute_offset']+i*width);require(len(packet)==width,'Short original Q8_0 row');packets.append(packet)
  stable();return b''.join(packets)
 finally:os.close(fd)

def bf16_gdn_ab_linear_owned(original_weights,own_operand):
 """Independent pre-nonlinearity GDN alpha/beta linear values, not native exp."""
 import hc_f32_arithmetic35_host_v2 as scalar
 from native_storage_first_gdn_estimate_v1 import bf16_rne
 scalar.require_gradual_f32()
 w=np.asarray(bf16_rne(original_weights),dtype='<f4');x=np.asarray(own_operand,dtype='<f4')
 require(w.ndim==2 and x.ndim==1 and w.shape[1]==len(x) and 0<len(x)<=10240 and len(x)%8==0 and 0<w.shape[0]<=1024 and np.isfinite(w).all()and np.isfinite(x).all(),'Bounded own BF16 GDN linear dimensions/finite required')
 output=[]
 # verify_kernels.dp.cpp:gdn_ab_multi_kernel: lane owns groups of8,
 # j=lane+32*k; each group has eight explicit FMAs, then XOR16..1.
 for row in w:
  lanes=np.zeros(32,dtype='<f4')
  for lane in range(32):
   acc=0.0
   for j in range(lane,len(x)//8,32):
    for k in range(8):acc=float(scalar._fma(float(row[j*8+k]),float(x[j*8+k]),acc))
   lanes[lane]=acc
  positions=np.arange(32)
  for mask in (16,8,4,2,1):lanes=np.asarray(lanes+lanes[positions^mask],dtype='<f4')
  output.append(lanes[0])
 scalar.require_gradual_f32()
 return {'output':np.asarray(output,dtype='<f4'),'reference_lane':'independent_original_BF16_GDN_AB_group8_FMA_XOR32_v1','native_exp_log_softplus_modeled':False,'device_FMA_intrinsics_qualified':False,'captured_inputs_used':False,'full_model_math_qualified':False}

class OwnFirstGdnArithmeticProjector:
 """Explicit opt-in first-GDN projector, preserving every other old role.

Construct a NEW OwnedGdn(provider, this_projector, 0, geometry, options).
Pass only its own HC mixed operand and fresh own states. This does not modify
any executed full48 class/function/global or broaden the consumer dispatch.
"""
 def __init__(self,original_projector):
  self.base=original_projector;self.p=original_projector.p;self.tile=original_projector.tile;self.events=[]
 def require(self,*args):return self.base.require(*args)
 def vector(self,*args):return self.base.vector(*args)
 def project(self,name,x,weight='original',activation='f32',expert=None):
  q80={'blk.0.attn_qkv.weight','blk.0.attn_gate.weight','blk.0.ssm_out.weight'}
  bf16={'blk.0.ssm_alpha.weight','blk.0.ssm_beta.weight'}
  if name not in q80|bf16:return self.base.project(name,x,weight,activation,expert)
  require(expert is None,'First GDN projection cannot consume expert selection')
  shape=self.p.shape(name);require(len(shape)==2,'Original first GDN matrix required')
  x=np.asarray(x,dtype='<f4');require(x.shape==(shape[0],)and np.isfinite(x).all(),'Own original first-GDN operand differs')
  output=np.empty(shape[1],dtype='<f4')
  if name in q80:
   require(weight=='original'and activation=='q8_1','Only source verifier Q80/Q81 route supported')
   require(self.base.kind(name)=='Q8_0','Original Q8_0 role changed')
   packet=encode(x.tobytes(),1,len(x));step=max(1,min(self.tile,64<<20)//(shape[0]//32*34+shape[0]*4*4))
   for at in range(0,shape[1],step):
    n=min(step,shape[1]-at);raw=original_rows(self.p.reader,name,list(range(at,at+n)));output[at:at+n]=project_packets(raw,packet,n,len(x))
   self.events.append({'role':name,'own_packet_bytes':len(packet),'reference_lane':'Q80_Q81_wide32_source_expression','stored_sum_used':False,'device_compiler_contraction_qualified':False})
  else:
   require(weight=='bf16'and activation=='f32','Only native BF16-weight/F32-input GDN AB route supported')
   require(self.base.kind(name)=='F32','Original GDN alpha/beta F32 role changed')
   step=max(1,self.tile//(shape[0]*8))
   for at in range(0,shape[1],step):
    n=min(step,shape[1]-at);raw=self.p.rows(name,range(at,at+n));output[at:at+n]=bf16_gdn_ab_linear_owned(raw,x)['output']
   self.events.append({'role':name,'reference_lane':'BF16_group8_FMA_XOR32','device_intrinsic_rounding_qualified':False})
  return output.astype(np.float64)
