"""Bounded conditional QSA history/selection model; no full-model/native oracle.
Inputs are explicitly per-layer projected values or supplied attention inputs.
Own FP16 KV, indexer tail/dead/pooled/positions, causal selection and checkpoints.
"""
import copy,hashlib,json
from dataclasses import dataclass,asdict
import numpy as np
from native_storage_first_gdn_estimate_v1 import f32,bf16_rne
from prefill_first_gdn_owned_storage_v1 import f16_rne
from independent_q8_1_activation_v1 import encode,decode

@dataclass(frozen=True)
class QsaGeometry:
 heads:int=24
 kv_heads:int=2
 dim:int=256
 rot:int=64
 idx_heads:int=4
 idx_dim:int=128
 block:int=4
 topk:int=2048
 page:int=4
 theta:float=1e7
 eps:float=float(np.float32(1e-6))

UNSUPPORTED=('device native exp/rsqrt/sin/cos and RoPE table rounding','native score/attention/GEMM/FMA reduction','actual score tie changes from device rounding','multimodal axis positions/YaRN/alternative KV formats','whole-model residual generation and all48-layer composition','actual main/slot GPU checkpoint byte reconstruction')


def rms(x,gamma,epsilon):
 x=f32(x);return f32(x*np.asarray(gamma)/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+epsilon))


def rope(x,pos,g):
 out=f32(x).copy();half=g.rot//2;angle=pos*g.theta**(-2*np.arange(half)/g.rot);a=out[...,:half].copy();b=out[...,half:g.rot].copy()
 out[...,:half]=a*np.cos(angle)-b*np.sin(angle);out[...,half:g.rot]=a*np.sin(angle)+b*np.cos(angle)
 return f32(out)


class QsaOwnedState:
 def __init__(self,geometry=QsaGeometry(),indexer_key_gamma=None,max_cells=2112,pos_base=0):
  g=geometry
  if min(g.heads,g.kv_heads,g.dim,g.idx_heads,g.idx_dim,g.rot,g.topk)<=0 or g.dim%4 or g.eps<=0 or g.theta<=0 or g.heads%g.kv_heads or g.rot%2 or g.rot>min(g.dim,g.idx_dim) or g.block!=4 or g.page!=4 or not 1<=max_cells<=4096 or not 0<=pos_base<max_cells:raise ValueError('Bounded resident/text QSA geometry differs')
  self.g=g;self.max_cells=max_cells;self.pos_base=pos_base;self.gamma=np.ones(g.idx_dim) if indexer_key_gamma is None else f32(indexer_key_gamma)
  if self.gamma.shape!=(g.idx_dim,):raise ValueError('Indexer key norm shape differs')
  self.reset()
 def reset(self):
  self.keys=[];self.values=[];self.records=[];self.tail=np.zeros((3,self.g.idx_dim));self.dead=np.zeros(self.g.idx_dim);self.pooled=[];self.block_pos=0
 def _pool(self,raw,pos):
  g=self.g;incoming=f16_rne(raw);slot=pos%4
  if slot<3:self.tail[slot]=incoming
  if pos==0 or slot==3:
   total=incoming.copy() if pos==0 else self.tail[0].copy()
   for index in range(1,4):total=f32(total+(incoming if pos==0 or index==3 else self.tail[index]))
   mean=f32(total*.25);anchor=0 if pos==0 else self.pos_base+4*(pos//4);normalized=rope(rms(mean,self.gamma,g.eps),anchor,g);block=pos//4
   while len(self.pooled)<=block:self.pooled.append(np.zeros(g.idx_dim))
   self.pooled[block]=normalized
   if pos==0:self.dead=normalized.copy()
   else:
    while len(self.pooled)<=block+1:self.pooled.append(np.zeros(g.idx_dim))
    self.pooled[block+1]=self.dead.copy();self.block_pos=anchor
 def scores_and_selection(self,indexer_query):
  g=self.g;n=len(self.keys);nb=n//4;query=f32(indexer_query)
  if query.shape!=(g.idx_heads,g.idx_dim):raise ValueError('Indexer query shape differs')
  scores=np.empty(n)
  for block in range(nb+1):
   lo=4*block;hi=min(n,lo+4)
   if lo==hi:continue
   key=self.dead if block==nb else self.pooled[block];score=f32(np.maximum(query@key,0).sum()).item()
   if block==nb and n%4:score=f32(score+1e9).item()
   scores[lo:hi]=score
  width=min(n,g.topk+3);selected=np.sort(np.lexsort((np.arange(n),-scores))[:width])
  return scores,selected
 def attention(self,query,gate,selected):
  g=self.g;n=len(self.keys)
  if len(set(map(int,selected)))!=len(selected) or np.any(selected<0) or np.any(selected>=n) or not np.array_equal(selected,np.sort(selected)):raise ValueError('Causal selected IDs invalid')
  q=f32(query);gate=f32(gate)
  if q.shape!=(g.heads,g.dim) or gate.shape!=(g.heads,g.dim):raise ValueError('Query/gate shape differs')
  keys=np.stack(self.keys);values=np.stack(self.values);result=np.empty_like(q)
  for head in range(g.heads):
   kv=head//(g.heads//g.kv_heads);logits=keys[selected,kv]@q[head]/np.sqrt(g.dim);weights=np.exp(logits-logits.max());weights/=weights.sum();result[head]=weights@values[selected,kv]
  with np.errstate(over='ignore'):gated=f32(f32(result)/(1+np.exp(-gate)))
  # Native resident default keeps gated attention F32 for output Q8_1 projection.
  return gated
 def advance(self,token,query,key,value,gate,indexer_raw,indexer_query,evaluate_attention=True):
  pos=len(self.keys);g=self.g
  if type(token) is not int or pos+self.pos_base>=self.max_cells:raise ValueError('Token history/capacity differs')
  shapes=[(g.heads,g.dim),(g.kv_heads,g.dim),(g.kv_heads,g.dim),(g.heads,g.dim),(g.idx_dim,),(g.idx_heads,g.idx_dim)]
  inputs=[f32(x) for x in (query,key,value,gate,indexer_raw,indexer_query)]
  if any(value.shape!=want for value,want in zip(inputs,shapes)):raise ValueError('Projected conditional input shape differs')
  query,key,value,gate,indexer_raw,indexer_query=inputs
  self.keys.append(f16_rne(key));self.values.append(f16_rne(value));self._pool(indexer_raw,pos)
  self.records.append({'token':token,'inputs':[value.astype('<f4') for value in inputs]});scores,selected=self.scores_and_selection(indexer_query)
  output=self.attention(query,gate,selected) if evaluate_attention else None
  return {'position':pos,'token':token,'scores':scores,'selected_ids_owned':selected,'attention_gated_F32':output,'completed_indexer_blocks':(pos+1)//4,'indexer_spare_row':(pos+1)//4,'block_pos':self.block_pos,'scope':'conditional projected per-layer inputs; selection/state owned, not whole-model','selection_device_qualified':False,'full_model_math_qualified':False}
 def resident_pool(self):
  g=self.g;n=len(self.keys);pages=(n+3)//4;k=np.zeros((pages*4,g.kv_heads,g.dim),dtype='<f2');v=k.copy()
  if n:k[:n]=np.stack(self.keys);v[:n]=np.stack(self.values)
  return {'k':k.reshape(pages,4,g.kv_heads,g.dim).transpose(0,2,1,3).copy(),'v':v.reshape(pages,4,g.kv_heads,g.dim).transpose(0,2,1,3).copy(),'page_table':np.arange(pages,dtype=np.int32),'layout':'[page,kv_head,cell_in_page,dim] FP16'}
 def history_digest(self):
  h=hashlib.sha256(json.dumps({'geometry':asdict(self.g),'pos_base':self.pos_base},sort_keys=True).encode('ascii'));h.update(np.asarray(self.gamma,dtype='<f4').tobytes())
  for record in self.records:
   h.update(np.asarray([record['token']],dtype='<i8').tobytes())
   for value in record['inputs']:h.update(np.asarray(value,dtype='<f4').tobytes())
  return h.hexdigest()
 def checkpoint(self):
  return {'geometry':asdict(self.g),'pos_base':self.pos_base,'gamma':self.gamma.copy(),'history_sha256':self.history_digest(),'records':copy.deepcopy(self.records),'keys':copy.deepcopy(self.keys),'values':copy.deepcopy(self.values),'tail':self.tail.copy(),'dead':self.dead.copy(),'pooled':copy.deepcopy(self.pooled),'block_pos':self.block_pos,'scope':'owned mathematical history snapshot; actualGPU checkpoint unqualified'}
 def restore(self,snapshot,expected_tokens):
  if snapshot['geometry']!=asdict(self.g) or snapshot['pos_base']!=self.pos_base:raise ValueError('Checkpoint identity/geometry differs')
  if not np.array_equal(snapshot['gamma'],self.gamma) or [record['token'] for record in snapshot['records']]!=list(expected_tokens):raise ValueError('Checkpoint norm or expected committed token prefix differs')
  replay=QsaOwnedState(self.g,self.gamma,self.max_cells,self.pos_base)
  for record in snapshot['records']:replay.advance(record['token'],*record['inputs'],evaluate_attention=False)
  if replay.history_digest()!=snapshot['history_sha256']:raise ValueError('Checkpoint input history digest differs')
  for name in ('keys','values','tail','dead','pooled'):
   if not np.array_equal(np.asarray(snapshot[name]),np.asarray(getattr(replay,name))):raise ValueError('Checkpoint history reconstruction differs '+name)
  if snapshot['block_pos']!=replay.block_pos:raise ValueError('Checkpoint block position differs')
  self.keys=replay.keys;self.values=replay.values;self.records=replay.records;self.tail=replay.tail;self.dead=replay.dead;self.pooled=replay.pooled;self.block_pos=replay.block_pos
 def qualification(self):return {'owned_history_selection_model':True,'conditional_on_supplied_layer_inputs':True,'captured_selected_ids_used':False,'native_intrinsics_or_reductions_emulated':False,'actual_main_slot_checkpoint_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None,'unsupported':list(UNSUPPORTED)}


class QsaConditionalProjection:
 """Original QSA roles with explicit storage, accepting supplied HC mixed input.
 No actual source reads until rows/project are called; no HC/wholemodel claim.
 """
 ROLES={'attn_q.weight':('Q8_0',(2560,12288)),'attn_k.weight':('Q8_0',(2560,512)),'attn_v.weight':('Q8_0',(2560,512)),'attn_output.weight':('Q8_0',(6144,2560)),'indexer.k_proj.weight':('BF16',(2560,128)),'indexer.q_proj.weight':('BF16',(2560,512)),'attn_q_norm.weight':('F32',(256,)),'attn_k_norm.weight':('F32',(256,)),'indexer.q_norm.weight':('F32',(128,)),'indexer.k_norm.weight':('F32',(128,))}
 def __init__(self,provider,tile_bytes=64<<20,layer=3):
  if layer not in range(3,48,4):raise ValueError('Owned original QSA layer index differs')
  self.p=provider;self.tile_bytes=tile_bytes;self.g=QsaGeometry();self.prefix='blk.'+str(layer)+'.'
  if not 0<tile_bytes<=64<<20:raise ValueError('Projection tile cap differs')
  for role,(kind,shape) in self.ROLES.items():
   name=self.prefix+role;actual=provider.reader.tensors[name][1]
   if actual['type']!=kind or tuple(actual['shape_ggml_order'])!=shape:raise ValueError('Exact original layer3 role differs '+role)
 def vector(self,role):return f32(self.p.rows(self.prefix+role,[0])[0])
 def project(self,role,x,route):
  if route not in ('prefill','verifier'):raise ValueError('Projection route differs')
  kind,shape=self.ROLES[role];x=f32(x)
  if kind=='Q8_0':
   if route=='prefill':x=f16_rne(x)
   elif route=='verifier':x=decode(encode(np.asarray(x,dtype='<f4').tobytes(),1,len(x)),1,len(x))['reconstructed_activation'].reshape(-1)
   else:raise ValueError('Projection route differs')
  elif kind=='BF16':x=bf16_rne(x) if route=='prefill' else x
  else:raise ValueError('Projection role type differs')
  out=np.empty(shape[1]);chunk=max(1,self.tile_bytes//(shape[0]*8))
  for start in range(0,shape[1],chunk):
   count=min(chunk,shape[1]-start);w=self.p.rows(self.prefix+role,range(start,start+count))
   if kind=='Q8_0' and route=='prefill':w=f16_rne(w)
   out[start:start+count]=w@x
  return f32(out)
 def step(self,state,token,mixed,route='verifier',evaluate_attention=True):
  if state.g!=self.g:raise ValueError('Actual layer3 geometry differs')
  pos=state.pos_base+len(state.keys);qg=self.project('attn_q.weight',mixed,route).reshape(24,2,256);q=rope(rms(qg[:,0],self.vector('attn_q_norm.weight'),self.g.eps),pos,self.g);k=rope(rms(self.project('attn_k.weight',mixed,route).reshape(2,256),self.vector('attn_k_norm.weight'),self.g.eps),pos,self.g);v=self.project('attn_v.weight',mixed,route).reshape(2,256)
  raw=self.project('indexer.k_proj.weight',mixed,route);iq=rope(rms(self.project('indexer.q_proj.weight',mixed,route).reshape(4,128),self.vector('indexer.q_norm.weight'),self.g.eps),pos,self.g)
  if not np.array_equal(state.gamma,self.vector('indexer.k_norm.weight')):raise ValueError('Owned indexer key gamma differs')
  result=state.advance(token,q,k,v,qg[:,1],raw,iq,evaluate_attention=evaluate_attention)
  if evaluate_attention:result['block_output']=self.project('attn_output.weight',result['attention_gated_F32'].reshape(-1),route)
  result['conditional_on_supplied_HC_attention_input']=True;result['projection_route']=route;return result
