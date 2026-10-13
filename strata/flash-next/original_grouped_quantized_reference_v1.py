"""Independent original packed Q4K/Q5K/Q51/Q80 grouped lane8 expressions.
Own operands only. Source-order F32 candidate; compiler contraction unqualified.
"""
import os
import numpy as np
from independent_q8_1_activation_v1 import encode

GEOMETRY={'Q4_K':(256,144,16),'Q5_K':(256,176,16),'Q5_1':(32,24,2),'Q8_0':(32,34,4)}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def f32(x):return np.asarray(x,dtype='<f4')
def original_packed_rows(reader,name,indices):
 shard,t,sig=reader.tensors[name];kind=t['type'];require(kind in GEOMETRY,'Unsupported grouped original format');qk,width,_=GEOMETRY[kind];shape=t['shape_ggml_order'];require(shape[0]%qk==0,'Incomplete original grouped blocks');size=shape[0]//qk*width;indices=list(indices);count=int(np.prod(shape[1:],dtype=np.int64))
 require(indices and len(indices)*size<=min(reader.max_read_bytes,64<<20)and all(type(i)is int and 0<=i<count for i in indices),'Bounded original packed rows required')
 fd=os.open(shard['path'],os.O_RDONLY|os.O_NOFOLLOW)
 try:
  def stable():
   s=os.fstat(fd);require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==sig,'Original packed-row source changed')
  stable();out=[]
  for i in indices:
   b=os.pread(fd,size,t['absolute_offset']+i*size);require(len(b)==size,'Short original packed row');out.append(b)
  stable();return b''.join(out)
 finally:os.close(fd)
def unpack(raw,kind,rows,cols):
 qk,width,_=GEOMETRY[kind];require(type(raw)is bytes and len(raw)==rows*(cols//qk)*width and len(raw)<=64<<20,'Complete bounded original grouped packets required');b=np.frombuffer(raw,np.uint8).reshape(rows,cols//qk,width)
 half=lambda at:b[:,:,at:at+2].copy().view('<f2').reshape(rows,cols//qk).astype('<f4')
 d=half(0);m=None;sc=mn=None
 if kind=='Q8_0':codes=b[:,:,2:].view(np.int8).astype(np.int32)
 elif kind=='Q5_1':
  m=half(2);low=np.concatenate((b[:,:,8:]&15,b[:,:,8:]>>4),axis=2);high=np.unpackbits(b[:,:,4:8],axis=2,bitorder='little')*16;codes=(low|high).astype(np.int32)
 else:
  m=half(2);p=b[:,:,4:16];sc=np.concatenate((p[:,:,:4]&63,(p[:,:,8:]&15)|((p[:,:,:4]>>6)<<4)),axis=2).astype(np.int32);mn=np.concatenate((p[:,:,4:8]&63,(p[:,:,8:]>>4)|((p[:,:,4:8]>>6)<<4)),axis=2).astype(np.int32)
  start=16 if kind=='Q4_K'else 48;q=b[:,:,start:].reshape(rows,cols//qk,4,32);codes=np.stack((q&15,q>>4),axis=3).reshape(rows,cols//qk,8,32)
  if kind=='Q5_K':codes=codes|(((b[:,:,16:48,None].transpose(0,1,3,2)>>np.arange(8,dtype=np.uint8)[None,None,:,None])&1)*16)
  codes=codes.astype(np.int32)
 require(np.isfinite(d).all()and(m is None or np.isfinite(m).all()),'Nonfinite packed original scales');return d,m,sc,mn,codes

def project_packets(raw,packet,kind,rows,cols,lanes=8):
 require(kind in GEOMETRY and type(rows)is int and 0<rows<=10240 and type(cols)is int and 0<cols<=2560 and cols%GEOMETRY[kind][0]==0 and type(lanes)is int and lanes==8,'Exact current grouped lane8 geometry required')
 require(type(packet)is bytes and len(packet)==cols//32*36,'Complete own activation packet required')
 d,m,sc,mn,codes=unpack(raw,kind,rows,cols);x=np.frombuffer(packet,np.uint8).reshape(cols//32,36);xd=x[:,:2].copy().view('<f2').reshape(-1).astype('<f4');xq=x[:,4:].view(np.int8).astype(np.int32);require(np.isfinite(xd).all(),'Nonfinite own activation scale')
 qk,_,ipb=GEOMETRY[kind];nb=cols//qk;acc=np.zeros((rows,lanes),dtype='<f4')
 for lane in range(lanes):
  for k in range(lane,nb*ipb,lanes):
   kb,p=divmod(k,ipb)
   if kind in('Q4_K','Q5_K'):
    group=2*(p//4);idx=np.r_[np.arange(4*(p%4),4*(p%4)+4),np.arange(16+4*(p%4),20+4*(p%4))];sd=np.zeros(rows,dtype='<f4');sm=sd.copy()
    for i in range(2):
     g=group+i;u=xq[kb*8+g,idx];dot=np.sum(codes[:,kb,g,idx]*u,axis=1,dtype=np.int32);sumu=int(u.sum(dtype=np.int32));sd=f32(sd+f32(xd[kb*8+g]*(dot*sc[:,kb,g]).astype('<f4')));sm=f32(sm+f32(xd[kb*8+g]*(sumu*mn[:,kb,g]).astype('<f4')))
    term=f32(f32(d[:,kb]*sd)-f32(m[:,kb]*sm))
   elif kind=='Q5_1':
    idx=np.r_[np.arange(8*p,8*p+8),np.arange(16+8*p,24+8*p)];u=xq[kb,idx];dot=np.sum(codes[:,kb,idx]*u,axis=1,dtype=np.int32);sumu=int(u.sum(dtype=np.int32));term=f32(f32(dot.astype('<f4')*f32(d[:,kb]*xd[kb]))+f32(np.float32(sumu)*f32(m[:,kb]*xd[kb])))
   else:
    idx=slice(8*p,8*p+8);dot=np.sum(codes[:,kb,idx]*xq[kb,idx],axis=1,dtype=np.int32);term=f32(f32(d[:,kb]*xd[kb])*dot.astype('<f4'))
   acc[:,lane]=f32(acc[:,lane]+term)
 positions=np.arange(lanes)
 for mask in(4,2,1):acc=f32(acc+acc[:,positions^mask])
 require(np.isfinite(acc[:,0]).all(),'Nonfinite grouped source expression');return acc[:,0].copy()

def project_owned(raw,operand,kind,rows):
 x=f32(operand);require(x.ndim==1 and np.isfinite(x).all(),'Own finite activation required');packet=encode(x.tobytes(),1,len(x));return {'output':project_packets(raw,packet,kind,rows,len(x)),'own_packet':packet,'stored_Q81_sum_used':False,'source_lane_count':8,'device_compiler_contraction_qualified':False,'captured_inputs_used':False}
