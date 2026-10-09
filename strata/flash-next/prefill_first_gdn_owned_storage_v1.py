"""Owned embedding -> earlier SYCL prefill rows -> final verifier storage model.
Mathematical estimates with explicit F16/BF16/F32 boundaries, NOT device oracle.
No captured input/state is accepted. No pass threshold/full-model qualification.
"""
import hashlib
import numpy as np
from original_first_gdn_layer_v1 import Geometry
from native_storage_first_gdn_estimate_v1 import f32,bf16_rne
from independent_q8_1_activation_v1 import encode,decode

UNSUPPORTED=('oneMKL GEMM F32 accumulation/tiling','native exp/log1p/rsqrt device rounding','HC subgroup FMA/XOR reduction','GDN FMA/subgroup and partial-group reduction','actual BF16 GEMM device branch selection','other layers/MoE/prefix-cache/session state')


def f16_rne(x):
 """SYCL hf: F32 -> F16 RNE -> exact widening; never saturate overflow."""
 source=np.asarray(f32(x),dtype='<f4')
 with np.errstate(over='ignore'):value=source.astype('<f2')
 if not np.isfinite(value).all():raise ValueError('F16 storage overflow/nonfinite; no saturation substitution')
 return value.astype(np.float64)


class OwnedPrefillGdnStorage:
 def __init__(self,provider,geometry=Geometry(),tile_bytes=64<<20,runtime_options=None):
  self.p=provider;self.g=geometry;self.prefix='blk.0.';self.tile_bytes=tile_bytes;self.events=[]
  if type(tile_bytes) is not int or not 0<tile_bytes<=64<<20:raise ValueError('Projection tile cap differs')
  if provider.actual_source and geometry!=Geometry():raise ValueError('Actual model geometry differs')
  options={} if runtime_options is None else dict(runtime_options)
  forbidden=('STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_REC_HEADS','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD')
  if ('STRATA_GDN_REC_HEADS' in options and options['STRATA_GDN_REC_HEADS'] is not None) or any(options.get(name) not in (None,'','0') for name in forbidden):raise NotImplementedError('Alternative BF16/recurrence profile unmodeled')
  self.validate_roles()
 def kind(self,name):
  return self.p.reader.tensors[name][1]['type'] if hasattr(self.p,'reader') else self.p.kind(name)
 def validate_roles(self):
  g=self.g;n=g.embd;d=n*g.streams;c=(2*g.key_heads+g.value_heads)*g.state;z=g.value_heads*g.state
  roles={'token_embd.weight':('Q8_0',(n,None)),'hc_attn_norm.weight':('F32',(d,)),'hc_attn_down.weight':('Q8_0',(d,g.low_rank)),'hc_attn_up.weight':('Q8_0',(g.low_rank,d)),'hc_attn_inject.weight':('F32',(d,g.streams)),'attn_qkv.weight':('Q8_0',(n,c)),'attn_gate.weight':('Q8_0',(n,z)),'ssm_out.weight':('Q8_0',(z,n)),'ssm_alpha.weight':('F32',(n,g.value_heads)),'ssm_beta.weight':('F32',(n,g.value_heads)),'ssm_dt.bias':('F32',(g.value_heads,)),'ssm_a':('F32',(g.value_heads,)),'ssm_norm.weight':('F32',(g.state,)),'ssm_conv1d.weight':('F32',(g.conv_kernel,c))}
  for role,(kind,want) in roles.items():
   name=role if role.startswith('token_') else self.prefix+role;shape=self.p.shape(name)
   if self.kind(name)!=kind or len(shape)!=len(want) or any(a!=b for a,b in zip(shape,want) if b is not None):raise ValueError('Selected original role/shape/storage differs '+name)
 def vector(self,role):return f32(self.p.rows(self.prefix+role,[0])[0])
 @staticmethod
 def sigmoid(x):
  # CPU mathematical exp evaluated in F64, then F32 stage stores. Not native::exp.
  with np.errstate(over='ignore'):return f32(1/(1+np.exp(-np.asarray(x,dtype=np.float64))))
 def project(self,role,x,weight_storage='original',input_storage='f32'):
  name=self.prefix+role;shape=self.p.shape(name);x=f32(x)
  if input_storage=='f16':x=f16_rne(x)
  elif input_storage=='bf16':x=bf16_rne(x)
  elif input_storage=='q8_1':
   raw=np.asarray(x,dtype='<f4').tobytes();packet=encode(raw,1,len(x));x=decode(packet,1,len(x))['reconstructed_activation'].reshape(-1)
   self.events.append({'role':role,'route':'verifier_Q8_1','input_sha256':hashlib.sha256(raw).hexdigest(),'packet_sha256':hashlib.sha256(packet).hexdigest()})
  elif input_storage!='f32':raise ValueError('Unknown activation storage')
  if len(shape)!=2 or len(x)!=shape[0]:raise ValueError('Projection dimensions differ')
  rows=shape[1];count=max(1,self.tile_bytes//(shape[0]*8));out=np.empty(rows)
  for start in range(0,rows,count):
   amount=min(count,rows-start);weights=self.p.rows(name,range(start,start+amount))
   if weight_storage=='f16':weights=f16_rne(weights)
   elif weight_storage=='bf16':weights=bf16_rne(weights)
   elif weight_storage!='original':raise ValueError('Unknown weight storage')
   out[start:start+amount]=weights@x
  return f32(out)
 def hc_attention(self,embedding):
  g=self.g;r=np.broadcast_to(f32(embedding),(g.streams,g.embd)).copy();norm=self.vector('hc_attn_norm.weight').reshape(r.shape)
  rs=f32(1/np.sqrt(f32(np.mean(r*r,axis=1,keepdims=True))+g.eps));xn=f32(f32(r*norm)*rs)
  down=self.project('hc_attn_down.weight',xn.reshape(-1));low=f32(down/g.streams);low=f32(low*self.sigmoid(low));gate=self.project('hc_attn_up.weight',low).reshape(r.shape);inject=self.project('hc_attn_inject.weight',xn.reshape(-1))
  total=np.zeros(g.embd)
  for stream in range(g.streams):total=f32(total+f32(xn[stream]*self.sigmoid(gate[stream])))
  mixed=f32(total/g.streams)
  return {'residual':r,'normalized':xn,'low':low,'gate':gate,'inject':inject,'mixed':mixed}
 def initial_state(self):
  g=self.g;return {'recurrent':np.zeros((g.value_heads,g.state,g.state)),'conv':np.zeros((g.conv_kernel-1,(2*g.key_heads+g.value_heads)*g.state))}
 def recurrent_estimate(self,state,q,k,v,log_decay,beta):
  """Source stage order, F32 state/update/core storage, F64 dot estimates."""
  g=self.g;qh=np.arange(g.value_heads)%g.key_heads;keys=k[qh];queries=q[qh]
  decay=f32(np.exp(log_decay));kv=f32(np.einsum('hi,hij->hj',keys,state));delta=f32(f32(v-f32(decay[:,None]*kv))*beta[:,None])
  # Source FMA(g,old,k*delta): separately rounded k*delta, then update store.
  update=f32(keys[:,:,None]*delta[:,None,:]);next_state=f32(decay[:,None,None]*state+update)
  core=f32(np.einsum('hi,hij->hj',queries,next_state)/np.sqrt(g.state))
  return next_state,core
 def mixer(self,mixed,state,route):
  if route not in ('prefill','verifier'):raise ValueError('Explicit producer route required')
  g=self.g;hk=g.key_heads;hv=g.value_heads;s=g.state;c=(2*hk+hv)*s
  recurrent=f32(state['recurrent']);history=f32(state['conv'])
  if recurrent.shape!=(hv,s,s) or history.shape!=(g.conv_kernel-1,c):raise ValueError('Own state layout differs')
  quant='f16' if route=='prefill' else 'q8_1';weights='f16' if route=='prefill' else 'original';small_input='bf16' if route=='prefill' else 'f32'
  qkv=self.project('attn_qkv.weight',mixed,weights,quant);z=self.project('attn_gate.weight',mixed,weights,quant).reshape(hv,s)
  alpha=self.project('ssm_alpha.weight',mixed,'bf16',small_input);beta_pre=self.project('ssm_beta.weight',mixed,'bf16',small_input);beta=self.sigmoid(beta_pre)
  soft_arg=f32(alpha+self.vector('ssm_dt.bias'));softplus=np.empty_like(soft_arg);high=soft_arg>20;softplus[high]=soft_arg[high];softplus[~high]=np.log1p(np.exp(soft_arg[~high]));log_decay=f32(f32(softplus)*self.vector('ssm_a'))
  full=np.concatenate([history,qkv[None,:]]);conv=np.empty(c);chunk=max(1,self.tile_bytes//(g.conv_kernel*8))
  for start in range(0,c,chunk):
   amount=min(chunk,c-start);w=self.p.rows(self.prefix+'ssm_conv1d.weight',range(start,start+amount));conv[start:start+amount]=f32(np.sum(full[:,start:start+amount]*w.T,axis=0))
  # The source prefill and verifier conv both use s/(1+native_exp(-s)).
  with np.errstate(over='ignore'):h=f32(conv/(1+np.exp(-conv)))
  q=h[:hk*s].reshape(hk,s);k=h[hk*s:2*hk*s].reshape(hk,s);v=h[2*hk*s:].reshape(hv,s)
  q=f32(q/np.sqrt(f32(np.sum(f32(q*q),axis=1,keepdims=True))+g.eps));k=f32(k/np.sqrt(f32(np.sum(f32(k*k),axis=1,keepdims=True))+g.eps))
  next_state,core=self.recurrent_estimate(recurrent,q,k,v,log_decay,beta)
  rs=f32(1/np.sqrt(f32(np.mean(f32(core*core),axis=1,keepdims=True))+g.eps));normalized=f32(f32(core*rs)*self.vector('ssm_norm.weight'));gated=f32(normalized*self.sigmoid(z))
  output_storage=f16_rne(gated) if route=='prefill' else gated
  output=self.project('ssm_out.weight',output_storage.reshape(-1),weights,quant)
  # Default SYCL pipelined prefill y retains scaled CORE, not normalized gated.
  details={'route':route,'mixed_f32':f32(mixed),'mixed_f16':f16_rne(mixed) if route=='prefill' else None,'mixed_bf16':bf16_rne(mixed) if route=='prefill' else None,'qkv':qkv,'z':z,'alpha':alpha,'beta_pre':beta_pre,'log_decay':log_decay,'beta':beta,'conv_silu':h,'q':q,'k':k,'v':v,'core_scaled':core,'normalized_gated_mathematical_F32':gated,'prefill_y_buffer_semantics':'scaled_core_F32' if route=='prefill' else 'verifier_gated_F32','output_operand_storage':output_storage,'output':output}
  return {'recurrent':next_state,'conv':f32(full[1:])},details
 def tokens(self,token_ids):
  ids=list(token_ids)
  if not 1<=len(ids)<=8 or any(type(token) is not int or not 0<=token<self.p.shape('token_embd.weight')[1] for token in ids):raise ValueError('Bounded own prefix differs')
  self.events=[];state=self.initial_state();rows=[]
  for position,token in enumerate(ids):
   embedding=f32(self.p.rows('token_embd.weight',[token])[0]);hc=self.hc_attention(embedding);before={name:value.copy() for name,value in state.items()};route='verifier' if position==len(ids)-1 else 'prefill';start=len(self.events)
   state,details=self.mixer(hc['mixed'],state,route)
   for event in self.events[start:]:event.update(position=position,token=token)
   rows.append({'token':token,'position':position,'route':route,'embedding':embedding,'hc_attention':hc,'incoming_state_owned':before,'outgoing_state_owned':{name:value.copy() for name,value in state.items()},'mixer':details})
  return {'lane':'SYCL_prefill_storage_math_then_verifier_estimate_unqualified','rows':rows,'packet_events':list(self.events),'earlier_row_routes':['prefill']*(len(ids)-1),'last_row_route':'verifier','captured_inputs_used':False,'incoming_states_supplied':False,'storage_lane_implemented':True,'actual_runtime_BF16_GEMM_branch_observed':False,'native_exp_or_reductions_or_GEMM_emulated':False,'original_own_state_reference_qualified':False,'complete_layer_math_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None,'unsupported':list(UNSUPPORTED),'state_layout':'logical[head,i,j]; source physical[i,head,j]=transpose(1,0,2)'}
