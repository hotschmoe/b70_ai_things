"""Owned original-role full48 causal storage mathematics, NOT native GPU oracle.
Public tokens accepts IDs only. No captured inputs/state/routing/head accepted.
"""
import hashlib,re
from pathlib import Path
import numpy as np
from original_first_gdn_layer_v1 import Geometry,OriginalTensorRows
from original_gguf_vector_decoder_v2 import rows as original_decoded_rows
from native_storage_first_gdn_estimate_v1 import f32,bf16_rne
from prefill_first_gdn_owned_storage_v1 import OwnedPrefillGdnStorage,f16_rne
from qsa_owned_state_storage_v1 import QsaConditionalProjection,QsaOwnedState,QsaGeometry
from ple_owned_history_storage_v1 import OwnedPleHistory
from ffn_full48_owned_storage_v1 import OwnedFfnStorage
from independent_q8_1_activation_v1 import encode,decode

UNSUPPORTED=('device native exp/log/rsqrt/RoPE rounding','native FMA/subgroup and dot/reduction grouping',
 'oneMKL GEMM accumulation/tiling and batch shape dependent rounding','native router/selection ties',
 'actual BF16/noMMQ/noFUSED dispatch not witnessed','actual GPU checkpoint/session reconstruction',
 'native bitwise equivalence or a newly inferred tolerance gate')

class BoundOriginalFull48Rows(OriginalTensorRows):
 """Lazy original provider bound to the identity receipt supplied to composition.
 Callers must separately authorize payload reads and ensure current source guards.
 """
 def __init__(self,manifest,identity):
  self.source_identity_sha256=hashlib.sha256(Path(identity).read_bytes()).hexdigest()
  super().__init__(manifest,identity)

class OriginalStorageProjector:
 def __init__(self,provider,tile_bytes=64<<20):
  if not 0<tile_bytes<=64<<20:raise ValueError('Decoded row tile bound differs')
  self.p=provider;self.tile=tile_bytes;self.events=[]
 def kind(self,name):return self.p.reader.tensors[name][1]['type'] if hasattr(self.p,'reader') else self.p.kind(name)
 def require(self,name,kind,shape):
  got=self.p.shape(name)
  if self.kind(name)!=kind or len(got)!=len(shape) or any(a!=b for a,b in zip(got,shape) if b is not None):raise ValueError('Original role/type/shape differs '+name)
 def vector(self,name):
  shape=self.p.shape(name)
  if len(shape)!=1 or shape[0]*8>self.tile:raise ValueError('Vector outside bounded source role')
  return f32(self.p.rows(name,[0])[0])
 def project(self,name,x,weight='original',activation='f32',expert=None):
  shape=self.p.shape(name);x=f32(x)
  if x.ndim!=1 or len(x)!=shape[0] or len(shape) not in (1,2,3):raise ValueError('Original projection input differs '+name)
  if activation=='q8_1':
   raw=np.asarray(x,dtype='<f4').tobytes();packet=encode(raw,1,len(x));x=decode(packet,1,len(x))['reconstructed_activation'].reshape(-1)
   self.events.append({'role':name,'expert':expert,'packet_sha256':hashlib.sha256(packet).hexdigest(),'input_sha256':hashlib.sha256(raw).hexdigest(),'stored_s_used':False})
  elif activation=='f16':x=f16_rne(x)
  elif activation=='bf16':x=bf16_rne(x)
  elif activation!='f32':raise ValueError('Unknown original activation storage')
  if weight not in ('original','f16','bf16'):raise ValueError('Unknown original weight storage')
  if (len(shape)==3)!=(expert is not None):raise ValueError('Expert role/index differs')
  count=shape[1] if len(shape)>1 else 1;offset=0
  if expert is not None:
   if type(expert) is not int or not 0<=expert<shape[2]:raise ValueError('Expert bound differs')
   offset=expert*count
  # Explicit synthetic-only analytical original-row provider. Actual storage
  # providers always use the bounded independent original decoder row path.
  if not self.p.actual_source and hasattr(self.p,'synthetic_matmul'):
   return f32(self.p.synthetic_matmul(name,range(offset,offset+count),x,weight))
  step=self.tile//(shape[0]*8)
  if step<1:raise ValueError('Original row exceeds tile bound')
  out=np.empty(count)
  for start in range(0,count,step):
   n=min(step,count-start);indices=range(offset+start,offset+start+n)
   w=np.asarray(original_decoded_rows(self.p.reader,name,list(indices),'ggml_f32') if self.p.actual_source and weight=='f16' else self.p.rows(name,indices))
   if w.shape!=(n,shape[0]) or not np.isfinite(w).all():raise ValueError('Original decoded tile differs')
   if weight=='f16':w=f16_rne(w)
   elif weight=='bf16':w=bf16_rne(w)
   out[start:start+n]=w@x
  return f32(out)

class OwnedGdn(OwnedPrefillGdnStorage):
 def __init__(self,provider,projector,layer,geometry,options):
  # All36 roles share the frozen layer0 equations; only namespace is generalized.
  self.p=provider;self.g=geometry;self.prefix='blk.%d.'%layer;self.tile_bytes=projector.tile;self.events=[];self.projector=projector
  if any(options.get(key) not in (None,'','0') for key in ('STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD')) or options.get('STRATA_GDN_REC_HEADS') is not None:raise NotImplementedError('Alternative GDN branch unmodeled')
  self.validate_roles()
 def project(self,role,x,weight_storage='original',input_storage='f32'):
  return self.projector.project(self.prefix+role,x,weight_storage,input_storage)

class OwnedQsa(QsaConditionalProjection):
 def __init__(self,provider,projector,layer):
  super().__init__(provider,projector.tile,layer);self.projector=projector
 def project(self,role,x,route):
  if route not in ('prefill','verifier'):raise ValueError('Explicit QSA producer route required')
  kind,_=self.ROLES[role]
  return self.projector.project(self.prefix+role,x,'f16' if kind=='Q8_0' and route=='prefill' else 'original','f16' if kind=='Q8_0' and route=='prefill' else 'q8_1' if kind=='Q8_0' else 'bf16' if route=='prefill' else 'f32')

class OwnedPle(OwnedPleHistory):
 def __init__(self,provider,projector,identity):super().__init__(provider,identity,max_tokens=8,tile_bytes=projector.tile);self.projector=projector
 def project(self,role,embedding):return self.projector.project('blk.1.'+role,embedding)

class OwnedFfn(OwnedFfnStorage):
 def __init__(self,provider,projector,layer,geometry,options):super().__init__(provider,layer,geometry,projector.tile,runtime_options=options);self.projector=projector
 def project(self,role,x,route,expert=None):
  if route not in ('prefill','verifier'):raise ValueError('Explicit FFN producer route required')
  name=self.prefix+'ffn_'+role+'.weight';kind=self.contract[name]
  weight='bf16' if kind=='F32' else 'f16' if route=='prefill' else 'original'
  activation=('bf16' if route=='prefill' else 'f32') if kind=='F32' else 'f16' if route=='prefill' else 'q8_1'
  return self.projector.project(name,x,weight,activation,expert)

class Full48OwnedComposition:
 def __init__(self,provider,source_identity_sha256,geometry=Geometry(),tile_bytes=64<<20,runtime_options=None):
  if not re.fullmatch('[0-9a-f]{64}',source_identity_sha256) or geometry!=Geometry():raise ValueError('Actual role geometry/source binding required')
  self.p=provider;self.identity=source_identity_sha256;self.g=geometry;self.options={} if runtime_options is None else dict(runtime_options);self.projector=OriginalStorageProjector(provider,tile_bytes)
  allowed={'STRATA_SYCL_NATIVE_HC':'1'}
  supported_options=set(allowed)|{'STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD','STRATA_GDN_REC_HEADS'}
  if set(self.options)-supported_options:raise NotImplementedError('Unreviewed runtime math/storage option')
  if provider.actual_source and (getattr(provider,'source_identity_sha256',None)!=source_identity_sha256 or provider.source_contract['effective_expert_scale']!=1.0 or provider.source_contract['original_rms_epsilon']!=geometry.eps):raise ValueError('Original provider identity/metadata differs')
  if any(self.options.get(key,'1')!='1' for key in allowed):raise NotImplementedError('Strict original HC/PLE branch required')
  for key in ('STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD'):
   if self.options.get(key) not in (None,'','0'):raise NotImplementedError('Alternative source dispatch '+key)
  if self.options.get('STRATA_GDN_REC_HEADS') is not None:raise NotImplementedError('Alternative recurrence profile')
  g=self.g;d=g.embd*g.streams;self.projector.require('token_embd.weight','Q8_0',(g.embd,248320));self.projector.require('output.weight','Q8_0',(g.embd,248320))
  for layer in range(48):
   for half in ('attn','ffn'):
    for role,kind,shape in [('norm','F32',(d,)),('down','Q8_0',(d,g.low_rank)),('up','Q8_0',(g.low_rank,d)),('inject','F32',(d,g.streams))]:self.projector.require('blk.%d.hc_%s_%s.weight'%(layer,half,role),kind,shape)
  for role,kind,shape in [('norm','F32',(d,)),('down','Q8_0',(d,g.low_rank)),('up','Q8_0',(g.low_rank,d))]:self.projector.require('output_hc_'+role+'.weight',kind,shape)
  self.gdn={layer:OwnedGdn(provider,self.projector,layer,g,self.options) for layer in range(48) if layer%4!=3};self.qsa={layer:OwnedQsa(provider,self.projector,layer) for layer in range(3,48,4)};self.ffn={layer:OwnedFfn(provider,self.projector,layer,g,self.options) for layer in range(48)};self.ple=OwnedPle(provider,self.projector,self.identity)
 @staticmethod
 def sigmoid(x):
  with np.errstate(over='ignore'):return f32(1/(1+np.exp(-np.asarray(x,dtype=np.float64))))
 def hc_read(self,residual,stem,inject=True):
  g=self.g;r=f32(residual)
  if r.shape!=(4,2560):raise ValueError('Owned residual geometry differs')
  norm=self.projector.vector(stem+'norm.weight').reshape(r.shape);rs=f32(1/np.sqrt(f32(np.mean(f32(r*r),axis=1,keepdims=True))+g.eps));xn=f32(f32(r*norm)*rs);down=self.projector.project(stem+'down.weight',xn.reshape(-1));low=f32(down/4);low=f32(low*self.sigmoid(low));gate=self.projector.project(stem+'up.weight',low).reshape(r.shape);total=np.zeros(g.embd)
  for stream in range(4):total=f32(total+f32(xn[stream]*self.sigmoid(gate[stream])))
  result={'mixed':f32(total/4),'normalized':xn,'down':down,'gate':gate}
  if inject:result['inject']=self.projector.project(stem+'inject.weight',xn.reshape(-1))
  return result
 def hc_write(self,residual,block,inject):return f32(f32(residual)+f32(f32(block)[None,:]*f32(2*self.sigmoid(f32(inject)/4))[:,None]))
 def tokens(self,token_ids):
  ids=list(token_ids)
  if len(ids) not in (1,2,4,8) or any(type(token) is not int or not 0<=token<248320 for token in ids):raise ValueError('Own prefix1/2/4/8 token roster required')
  self.projector.events=[];self.ple.reset();states={layer:model.initial_state() for layer,model in self.gdn.items()};qstates={layer:QsaOwnedState(QsaGeometry(),model.vector('indexer.k_norm.weight'),max_cells=8) for layer,model in self.qsa.items()};trace=[]
  for position,token in enumerate(ids):
   route='verifier' if position==len(ids)-1 else 'prefill';embedding=f32(self.p.rows('token_embd.weight',[token])[0]);residual=np.broadcast_to(embedding,(4,2560)).copy();layers=[]
   for layer in range(48):
    before=residual.copy();ple_info=None
    if layer==1:ple_info=self.ple.advance(token,residual);residual=ple_info['postprojection']['result']
    attention=self.hc_read(residual,'blk.%d.hc_attn_'%layer)
    if layer in self.gdn:states[layer],detail=self.gdn[layer].mixer(attention['mixed'],states[layer],route);block=detail['output']
    else:detail=self.qsa[layer].step(qstates[layer],token,attention['mixed'],route);block=detail['block_output']
    after_attn=self.hc_write(residual,block,attention['inject']);ffn_read=self.hc_read(after_attn,'blk.%d.hc_ffn_'%layer);ffn=self.ffn[layer].run(ffn_read['mixed'],route);residual=self.hc_write(after_attn,ffn['block_output'],ffn_read['inject'])
    layers.append({'layer':layer,'family':'QSA' if layer%4==3 else 'GDN','route':route,'input':before,'attention':after_attn,'ffn':residual.copy(),'selected_ids_owned':detail.get('selected_ids_owned'),'router_ids_owned':ffn['router_ids_owned'].copy(),'PLE_rows_owned':None if ple_info is None else ple_info['row_ids_owned']})
   trace.append({'position':position,'token':token,'route':route,'layers':layers})
  final=self.hc_read(residual,'output_hc_',inject=False);logits=self.projector.project('output.weight',final['mixed'],activation='q8_1')
  return {'lane':'owned_original48_storage_mathematics_unqualified','ids':ids,'trace':trace,'first_generated_logits':logits,'gdn_states_owned':states,'qsa_states_owned':qstates,'ple_history_owned':self.ple.history.copy(),'last_two_owned':list(self.ple.last_two),'packet_events':list(self.projector.events),'captured_inputs_used':False,'captured_states_or_selected_ids_used':False,'state_initialized_from_zero':True,'actual_original_payload_used':self.p.actual_source,'actual_runtime_dispatch_witnessed':False,'nativebitwise_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None,'unsupported':list(UNSUPPORTED),'source_identity_sha256':self.identity}
