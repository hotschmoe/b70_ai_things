"""Independent own-prefix storage estimate, explicitly NOT native arithmetic oracle.

Original packed weights retained; native Q8_1 storage and BF16-RNE seams explicit.
FP64 dots/recurrence plus F32 stores and CPU exp are estimates of device arithmetic.
No captured activations/states enter tokens(). No full-layer/model pass API exists.
"""
import hashlib
import numpy as np
from original_first_gdn_layer_v1 import FirstGdnLayer,Geometry
from original_math_scalar import gdn_step
from independent_q8_1_activation_v1 import encode,decode

BF16_ROLES=('ssm_alpha.weight','ssm_beta.weight','ffn_gate_inp.weight','ffn_gate_inp_shexp.weight')
PACKET_ROLES={'attn_qkv.weight':'Q8_0','attn_gate.weight':'Q8_0','ssm_out.weight':'Q8_0','ffn_gate_shexp.weight':'Q8_0','ffn_up_shexp.weight':'Q8_0','ffn_down_shexp.weight':'Q8_0','ffn_gate_exps.weight':'Q4_K','ffn_up_exps.weight':'Q4_K','ffn_down_exps.weight':'Q5_1'}
UNSUPPORTED=('device native::exp rounding','native FP32 FMA and reduction grouping','HC projection/reduction FP32 order','router topk/ties/native exp reduction','MoE rank-ordered FMA combine','actual earlier-row prefill F16/BF16 GEMM histories (no decode-Q8_1 substitution)')


def f32(x):
 value=np.asarray(x,dtype=np.float32)
 if not np.isfinite(value).all():raise ValueError('Nonfinite F32 storage estimate')
 return value.astype(np.float64)


def bf16_rne(x):
 value=np.asarray(x,dtype='<f4')
 if not np.isfinite(value).all():raise ValueError('Nonfinite BF16 seam')
 bits=value.view('<u4');rounded=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)
 result=rounded.view('<f4')
 if not np.isfinite(result).all():raise ValueError('BF16 overflow seam')
 return result.astype(np.float64)


class WeightSeams:
 def __init__(self,original):self.original=original;self.actual_source=original.actual_source
 def shape(self,name):return self.original.shape(name)
 def rows(self,name,indices):
  values=self.original.rows(name,indices)
  return bf16_rne(values) if name.removeprefix('blk.0.') in BF16_ROLES else values
 def kind(self,name):
  if hasattr(self.original,'reader'):return self.original.reader.tensors[name][1]['type']
  return self.original.kind(name)


class OwnPrefixStorageEstimate(FirstGdnLayer):
 def __init__(self,provider,geometry=Geometry(),tile_bytes=64<<20):
  super().__init__(WeightSeams(provider),geometry=geometry,tile_bytes=tile_bytes)
  self.lane='native_storage_estimate_unqualified';self.packet_events=[]
  for role,want in PACKET_ROLES.items():
   if self.p.kind(self.prefix+role)!=want:raise ValueError('Selected native consumer format differs: '+role)
  for role in BF16_ROLES:
   if self.p.kind(self.prefix+role)!='F32':raise ValueError('Selected original F32-to-BF16 seam differs: '+role)
 def project(self,name,x,expert=None):
  role=name.removeprefix(self.prefix);x=f32(x)
  if role in PACKET_ROLES:
   raw=np.asarray(x,dtype='<f4').tobytes();packet=encode(raw,1,len(x));image=decode(packet,1,len(x))
   x=image['reconstructed_activation'].reshape(-1)
   self.packet_events.append({'role':role,'expert':expert,'input_sha256':hashlib.sha256(raw).hexdigest(),'packet_sha256':hashlib.sha256(packet).hexdigest(),'cols':len(x),'weight_type':PACKET_ROLES[role],'stored_sum_used':False,'consumer_algebra':'original_real_weights dot Q8_1 d*codes; NOT native F32 reducer'})
  return f32(super().project(name,x,expert))
 @staticmethod
 def sigmoid(x):
  # CPU exp is deliberately labelled unsupported native-exp emulation.
  with np.errstate(over='ignore'):
   out=np.float32(1)/(np.float32(1)+np.exp(-np.asarray(x,dtype=np.float32)))
  return f32(out)
 @classmethod
 def silu(cls,x):
  with np.errstate(over='ignore'):
   value=np.asarray(x,dtype=np.float32);out=value/(np.float32(1)+np.exp(-value))
  return f32(out)
 def hc_read(self,residual,half):
  details=super().hc_read(f32(residual),half)
  return {key:f32(value) for key,value in details.items()}
 def hc_write(self,residual,block,inject):return f32(super().hc_write(f32(residual),f32(block),f32(inject)))
 def mixer(self,x,state):
  g=self.g;s=g.state;hk=g.key_heads;hv=g.value_heads;c=2*hk*s+hv*s
  history=f32(state['conv']);incoming=f32(state['recurrent'])
  if history.shape!=(g.conv_kernel-1,c) or incoming.shape!=(hv,s,s):raise ValueError('Own state geometry differs')
  qkv=self.project(self.prefix+'attn_qkv.weight',x);z=self.project(self.prefix+'attn_gate.weight',x).reshape(hv,s)
  alpha=self.project(self.prefix+'ssm_alpha.weight',x);beta=self.sigmoid(self.project(self.prefix+'ssm_beta.weight',x))
  # Estimated CPU softplus. Device native exp/log/FMA differences remain unqualified.
  log_decay=f32(f32(np.logaddexp(np.float32(0),np.asarray(alpha+self.vector(self.prefix+'ssm_dt.bias'),dtype=np.float32)))*self.vector(self.prefix+'ssm_a'))
  full=np.concatenate([history,qkv[None,:]]);conv=np.empty(c)
  tile=max(1,self.tile_bytes//(g.conv_kernel*8))
  for start in range(0,c,tile):
   count=min(tile,c-start);weights=self.p.rows(self.prefix+'ssm_conv1d.weight',range(start,start+count))
   conv[start:start+count]=f32(np.sum(full[:,start:start+count]*weights.T,axis=0))
  h=self.silu(conv);q=h[:hk*s].reshape(hk,s);k=h[hk*s:2*hk*s].reshape(hk,s);v=h[2*hk*s:].reshape(hv,s)
  q=f32(q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+g.eps));k=f32(k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+g.eps))
  next_state,core=gdn_step(incoming,q,k,v,log_decay,beta,lane='original_fp64')
  next_state=f32(next_state);core=f32(core/np.sqrt(s));norm=f32(core*self.vector(self.prefix+'ssm_norm.weight')/np.sqrt(np.mean(core*core,axis=1,keepdims=True)+g.eps));gated=f32(norm*self.sigmoid(z))
  output=self.project(self.prefix+'ssm_out.weight',gated.reshape(-1))
  return {'recurrent':next_state,'conv':f32(full[1:])},output,{'qkv':qkv,'conv_silu':h,'q':q,'k':k,'v':v,'log_decay':log_decay,'beta':beta,'core':core,'normalized':norm,'gated':gated,'output':output,'incoming_recurrent':incoming.copy(),'incoming_conv':history.copy()}
 def expert(self,x,expert,shared=False):
  output,details=super().expert(x,expert,shared);return f32(output),{key:f32(value) for key,value in details.items()}
 def ffn(self,x):
  output,details=super().ffn(f32(x));return f32(output),details
 def tokens(self,token_ids,*,allow_decode_surrogate=False):
  # Each invocation reconstructs its own zero state and all preceding embeddings.
  ids=list(token_ids)
  if not 1<=len(ids)<=8 or any(type(token) is not int or not 0<=token<self.p.shape('token_embd.weight')[1] for token in ids):raise ValueError('Bounded independent prefix differs')
  if len(ids)>1 and not allow_decode_surrogate:raise NotImplementedError('Actual prefix earlier rows use prefill F16/BF16 GEMMs; repeated verifier estimator is not their own-state oracle')
  self.packet_events=[];rows=[];state=self.initial_state()
  for ordinal,token in enumerate(ids):
   begin=len(self.packet_events);embedding=f32(self.p.rows('token_embd.weight',[token])[0]);residual=np.broadcast_to(embedding,(self.g.streams,self.g.embd)).copy()
   result,state,details=self.step(residual,state)
   for event in self.packet_events[begin:]:event.update(token=token,position=ordinal)
   rows.append({'token':token,'position':ordinal,'details':details,'state':{key:value.copy() for key,value in state.items()}})
  return {'lane':self.lane,'rows':rows,'packet_events':list(self.packet_events),'incoming_states_supplied':False,'captured_inputs_used':False,'native_exp_emulated':False,'native_reduction_emulated':False,'own_prefix_estimate_executed':True,'producer_route':'decode_only_surrogate' if len(ids)>1 else 'first_zero_state_verifier_estimate','actual_prefill_history_supported':False,'captured_prefix_history_comparison_admissible':len(ids)==1,'original_own_state_reference_qualified':False,'complete_layer_operators_qualified':False,'full_model_math_qualified':False,'unsupported':list(UNSUPPORTED),'tolerance_gate':None}
