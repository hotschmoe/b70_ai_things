"""Conditional layer1 PLE from owned committed tokens/history and original rows.
Hash is uint64 exact; math/storage estimates reuse frozen ple_postprojection.
No captured states/rowIDs accepted; lazy payloads, no native/fullmodel oracle.
"""
import copy,hashlib,json,re
from dataclasses import dataclass,asdict
import numpy as np
from original_math_scalar import ple_postprojection
from native_storage_first_gdn_estimate_v1 import f32

@dataclass(frozen=True)
class PleHashConsts:
 multipliers:tuple=(23703573157769,20109073645365,8052911324071)
 vocab:tuple=(20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171)
 offsets:tuple=(0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275)
 eos:int=248044


def ngram_rows(token,prev_oldest_first,consts=PleHashConsts()):
 if type(token) is not int or not 0<=token<=0x7fffffff or len(prev_oldest_first)!=2 or any(type(t) is not int or not -(1<<31)<=t<1<<31 for t in prev_oldest_first):raise ValueError('Int32 token/oldest-first predecessor scope differs')
 context=[token];cut=False
 for prior in reversed(prev_oldest_first):
  cut=cut or prior<0 or prior==consts.eos;context.append(consts.eos if cut else prior)
 result=[];mask=(1<<64)-1
 for length in (2,3):
  mixed=0
  for index in range(length):mixed^=(context[index]*consts.multipliers[index])&mask
  for head in range((length-2)*8,(length-1)*8):result.append(mixed%consts.vocab[head]+consts.offsets[head])
 return result


class OwnedPleHistory:
 ROLES={'per_layer_token_embd.weight':('IQ4_NL',(160,320001536)),'blk.1.ple_key.weight':('Q8_0',(2560,10240)),'blk.1.ple_value.weight':('Q8_0',(2560,2560)),'blk.1.ple_conv1d.weight':('F32',(4,10240)),'blk.1.ple_norm_key.weight':('F32',(10240,)),'blk.1.ple_norm_query.weight':('F32',(10240,)),'blk.1.ple_norm_conv.weight':('F32',(10240,))}
 def __init__(self,provider,source_identity_sha256,consts=PleHashConsts(),max_tokens=64,tile_bytes=64<<20):
  if not re.fullmatch('[0-9a-f]{64}',source_identity_sha256) or not 1<=max_tokens<=256 or not 0<tile_bytes<=64<<20:raise ValueError('Bounded identity/size differs')
  self.p=provider;self.source_identity=source_identity_sha256;self.consts=consts;self.max_tokens=max_tokens;self.tile=tile_bytes
  if consts!=PleHashConsts():raise ValueError('Selected original ngram constants differ')
  for name,(kind,shape) in self.ROLES.items():
   actual=provider.reader.tensors[name][1]
   if actual['type']!=kind or tuple(actual['shape_ggml_order'])!=shape:raise ValueError('Original PLE role/type/shape differs '+name)
  self.reset()
 def reset(self):self.tokens=[];self.records=[];self.last_two=[-1,-1];self.history=np.zeros((9,10240))
 def vector(self,role):return f32(self.p.rows('blk.1.'+role,[0])[0]).reshape(4,2560)
 def project(self,role,embedding):
  name='blk.1.'+role;count=self.ROLES[name][1][1];chunk=max(1,self.tile//(2560*8));output=np.empty(count)
  # Strict source_exact PLE uses original Q8_0 against F32 input, WITHOUT
  # Q8_1 activation quantization or BF16/F16 projection storage.
  for start in range(0,count,chunk):
   amount=min(chunk,count-start);w=self.p.rows(name,range(start,start+amount));output[start:start+amount]=w@embedding
  return f32(output)
 def advance(self,token,layer_residual):
  if type(token) is not int or not 0<=token<248320 or len(self.tokens)>=self.max_tokens:raise ValueError('Committed token bound differs')
  residual=f32(layer_residual)
  if residual.shape!=(4,2560):raise ValueError('Supplied layer1 residual shape differs')
  indices=ngram_rows(token,self.last_two,self.consts);rows=self.p.rows('per_layer_token_embd.weight',indices)
  if rows.shape!=(16,160):raise ValueError('Original table head-row shape differs')
  embedding=f32(rows).reshape(2560);key=self.project('ple_key.weight',embedding).reshape(4,2560);value=self.project('ple_value.weight',embedding)
  weights=f32(self.p.rows('blk.1.ple_conv1d.weight',range(10240))).T
  if weights.shape!=(4,10240):raise ValueError('Original F32 channel/tap storage differs')
  before=self.history.copy();details=ple_postprojection(key,value,residual,self.vector('ple_norm_key.weight'),self.vector('ple_norm_query.weight'),self.vector('ple_norm_conv.weight'),before,weights,lane='declared_storage')
  if not all(np.isfinite(value).all() for value in details.values()):raise ValueError('Nonfinite PLE math/storage stage')
  self.history=f32(details['next_history']);previous=list(self.last_two);self.last_two=[self.last_two[1],token];self.tokens.append(token);self.records.append({'token':token,'residual':residual.astype('<f4')})
  return {'position':len(self.tokens)-1,'token':token,'predecessors_oldest_first':previous,'row_ids_owned':indices,'embedding_F32':embedding,'key_projection_F32':key,'value_projection_F32':value,'incoming_history_owned':before,'postprojection':details,'last_two_owned':list(self.last_two),'source_identity_sha256':self.source_identity,'conditional_on_supplied_layer1_residual':True,'raw_native_conv_buffer_observed':False,'full_model_math_qualified':False}
 def digest(self):
  h=hashlib.sha256(self.source_identity.encode('ascii')+json.dumps(asdict(self.consts),sort_keys=True).encode('ascii'))
  for record in self.records:h.update(np.asarray([record['token']],dtype='<i8').tobytes());h.update(record['residual'].tobytes())
  return h.hexdigest()
 def checkpoint(self):return {'source_identity_sha256':self.source_identity,'consts':asdict(self.consts),'tokens':list(self.tokens),'records':copy.deepcopy(self.records),'last_two':list(self.last_two),'history':self.history.copy(),'input_digest':self.digest()}
 def restore(self,snapshot,expected_committed_tokens):
  if snapshot['source_identity_sha256']!=self.source_identity or snapshot['consts']!=asdict(self.consts) or snapshot['tokens']!=list(expected_committed_tokens):raise ValueError('PLE checkpoint source/token/constant identity differs')
  replay=type(self)(self.p,self.source_identity,self.consts,self.max_tokens,self.tile)
  for record in snapshot['records']:replay.advance(record['token'],record['residual'])
  if replay.tokens!=snapshot['tokens'] or replay.last_two!=snapshot['last_two'] or replay.digest()!=snapshot['input_digest'] or not np.array_equal(replay.history,snapshot['history']):raise ValueError('PLE checkpoint owned history reconstruction differs')
  self.tokens=replay.tokens;self.records=replay.records;self.last_two=replay.last_two;self.history=replay.history
 def qualification(self):return {'owned_token_hash_history':True,'captured_state_or_row_ids_used':False,'conditional_on_supplied_layer1_residual':True,'hash_uint64_wrap_XOR_modulo_contract':True,'actual_original_payload_reads_qualified':False,'native_intrinsic_quant_or_reductions_emulated':False,'actual_main_slot_checkpoint_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None,'unsupported':['native projection/FMA/RMS/dot reduction and exp rounding','alternative table formats/nonexact PLE projection routes','upstream residual/GDN/QSA/FFN/all48 composition','actual GPU checkpoint/lifecycle/cache continuity']}
