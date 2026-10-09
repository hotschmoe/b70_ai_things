"""Stateless owned router/expert FFN storage adapter for48 original role namespaces.
Explicit SYCL noMMQ prefill and verifier images; mathematical estimates only.
No supplied logits/IDs/weights/GU/HQ/down intermediates, no nativebitwise oracle.
"""
import numpy as np
from original_first_gdn_layer_v1 import Geometry
from original_gguf_vector_decoder_v2 import rows as decoded_rows
from native_storage_first_gdn_estimate_v1 import f32,bf16_rne
from prefill_first_gdn_owned_storage_v1 import f16_rne
from independent_q8_1_activation_v1 import encode,decode

class OwnedFfnStorage:
 def __init__(self,provider,layer,geometry=Geometry(),tile_bytes=64<<20,prefill_profile='SYCL_f16_noMMQ_noFUSED',runtime_options=None):
  if type(layer) is not int or not 0<=layer<48 or not 0<tile_bytes<=64<<20:raise ValueError('Layer/tilebound differs')
  if prefill_profile!='SYCL_f16_noMMQ_noFUSED':raise NotImplementedError('Prefill route not sourceaudited')
  options={} if runtime_options is None else runtime_options
  if any(options.get(name) not in (None,'','0') for name in ('STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_PF_FUSED','STRATA_PREFILL_MMQ')):raise NotImplementedError('Alternative FFN operand/producer route')
  if provider.actual_source and geometry!=Geometry():raise ValueError('Actual FFN geometry differs')
  self.p=provider;self.layer=layer;self.g=geometry;self.prefix='blk.'+str(layer)+'.';self.tile=tile_bytes
  g=geometry;self.contract={}
  gu='Q5_K' if layer==2 else 'Q4_K';down='Q8_0' if layer in (2,4,30,46,47) else 'Q5_1'
  for role,kind,shape in [('gate_exps',gu,(g.embd,g.ffn,g.experts)),('up_exps',gu,(g.embd,g.ffn,g.experts)),('down_exps',down,(g.ffn,g.embd,g.experts)),('gate_shexp','Q8_0',(g.embd,g.ffn)),('up_shexp','Q8_0',(g.embd,g.ffn)),('down_shexp','Q8_0',(g.ffn,g.embd)),('gate_inp','F32',(g.embd,g.experts)),('gate_inp_shexp','F32',(g.embd,))]:
   name=self.prefix+'ffn_'+role+'.weight';actual_kind=self.kind(name);actual_shape=self.p.shape(name)
   if actual_kind!=kind or tuple(actual_shape)!=shape:raise ValueError('Original mixed-format FFN role differs '+name)
   self.contract[name]=kind
 def kind(self,name):return self.p.reader.tensors[name][1]['type'] if hasattr(self.p,'reader') else self.p.kind(name)
 def source_rows(self,name,indices,route):
  if route=='prefill' and hasattr(self.p,'reader'):return decoded_rows(self.p.reader,name,list(indices),'ggml_f32')
  return self.p.rows(name,indices)
 def project(self,role,x,route,expert=None):
  if route not in ('prefill','verifier'):raise ValueError('Explicit FFN actual route required')
  name=self.prefix+'ffn_'+role+'.weight';shape=self.p.shape(name);kind=self.contract[name];x=f32(x)
  if len(x)!=shape[0] or (len(shape)==3)!=(expert is not None):raise ValueError('Projection input/expert shape differs')
  if kind=='F32':x=bf16_rne(x) if route=='prefill' else x
  elif route=='prefill':x=f16_rne(x)
  else:x=decode(encode(np.asarray(x,dtype='<f4').tobytes(),1,len(x)),1,len(x))['reconstructed_activation'].reshape(-1)
  count=shape[1] if len(shape)>1 else 1;out=np.empty(count);tile=max(1,self.tile//(shape[0]*8));base=0 if expert is None else int(expert)*count
  if expert is not None and (not 0<=expert<shape[2]):raise ValueError('Expert index outside original source')
  for start in range(0,count,tile):
   amount=min(tile,count-start);w=self.source_rows(name,range(base+start,base+start+amount),route)
   w=bf16_rne(w) if kind=='F32' else f16_rne(w) if route=='prefill' else w
   out[start:start+amount]=w@x
  return f32(out)
 @staticmethod
 def sigmoid(value):
  with np.errstate(over='ignore'):return f32(1/(1+np.exp(-np.asarray(value,dtype=np.float64))))
 def hidden(self,gate,up,route):
  with np.errstate(over='ignore',invalid='ignore'):
   a=np.asarray(gate,dtype='<f4');u=np.asarray(up,dtype='<f4');activation=(a/(np.float32(1)+np.exp(-a)));hidden=activation*u
  if np.isnan(hidden).any():raise ValueError('NaN hidden preserved/rejected, not saturated')
  if route=='prefill':return f16_rne(np.clip(hidden,-65504,65504)) # Actual hf_sat also clamps infinity.
  return f32(hidden)
 def expert(self,mixed,route,expert=None):
  shared=expert is None;suffix='shexp' if shared else 'exps';kw={} if shared else {'expert':int(expert)}
  gate=self.project('gate_'+suffix,mixed,route,**kw);up=self.project('up_'+suffix,mixed,route,**kw);hidden=self.hidden(gate,up,route);down=self.project('down_'+suffix,hidden,route,**kw)
  return down,{'gate':gate,'up':up,'hidden_estimate':hidden,'down_estimate':down,'raw_fused_hidden_observed':False}
 def run(self,mixed,route):
  mixed=f32(mixed)
  if mixed.shape!=(self.g.embd,):raise ValueError('Owned HC mixed input shape differs')
  logits=self.project('gate_inp',mixed,route);probabilities=f32(np.exp(f32(logits-logits.max())));probabilities=f32(probabilities/f32(probabilities.sum()));ids=np.lexsort((np.arange(self.g.experts),-probabilities))[:self.g.topk];weights=f32(probabilities[ids]/max(float(f32(probabilities[ids].sum())),6.103515625e-5))
  outputs=[];details=[]
  for expert in ids:out,detail=self.expert(mixed,route,int(expert));outputs.append(out);details.append(detail)
  shared,shared_detail=self.expert(mixed,route);shared_logit=self.project('gate_inp_shexp',mixed,route);shared_gated=f32(shared*self.sigmoid(shared_logit))
  combined=f32(np.sum(np.stack(outputs)*weights[:,None],axis=0)+shared_gated)
  return {'block_output':combined,'router_logits':logits,'router_ids_owned':ids,'router_weights_owned':weights,'experts':details,'shared':shared_detail,'shared_gate_logit_estimate':shared_logit,'route':route,'original_format_contract':dict(self.contract),'actual_runtime_route_observed':False,'supplied_intermediates_used':False,'native_exp_FMA_GEMM_reductions_emulated':False,'source_format_nativebitwise_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None,'scope':'Owned original-role FFN mathematical/storage estimate from its HC mixed input; standalone supplied mixed is conditional, composition must prove upstream ownership'}
