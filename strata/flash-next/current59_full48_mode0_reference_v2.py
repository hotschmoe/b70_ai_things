"""Current59 independent full48 own-state with explicit grouped integer products.
Preserves frozen V1; no native operand feeds or decoded BLAS projection fallback.
"""
import re,hashlib
import numpy as np
from current59_full48_mode0_reference_v1 import ConsumerProjector as PriorProjector,Current59Full48OwnState as PriorState,OwnedPle,require
from original_grouped_quantized_reference_v1 import GEOMETRY,original_packed_rows,project_packets
from original_q80_q81_wide32_reference_v1 import original_rows,project_packets as wide32
from independent_q8_1_activation_v1 import encode

class ConsumerProjector(PriorProjector):
 def project(self,name,x,weight='original',activation='f32',expert=None):
  match=re.fullmatch(r'blk\.\d+\.ffn_(gate|up|down)_(exps|shexp)\.weight',name)
  if match is None:return super().project(name,x,weight,activation,expert)
  kind=self.kind(name);shape=self.p.shape(name);x=np.asarray(x,dtype='<f4');shared=match[2]=='shexp'
  require(weight=='original'and activation=='q8_1'and x.shape==(shape[0],)and np.isfinite(x).all(),'Independent FFN verifier ownQ81/original weights required')
  require((len(shape)==3)==(expert is not None)and len(shape)==(2 if shared else 3),'Independent shared/grouped original role shape differs')
  require(kind=='Q8_0'if shared else kind in GEOMETRY,'Current shared/grouped format unsupported')
  count=shape[1];offset=0
  if not shared:require(type(expert)is int and 0<=expert<shape[2],'Own router expert index required');offset=expert*count
  packet=encode(x.tobytes(),1,len(x));out=np.empty(count,dtype='<f4');step=max(1,self.tile//(shape[0]*24))
  for at in range(0,count,step):
   n=min(step,count-at);indices=list(range(offset+at,offset+at+n))
   if shared:out[at:at+n]=wide32(original_rows(self.p.reader,name,indices),packet,n,len(x))
   else:out[at:at+n]=project_packets(original_packed_rows(self.p.reader,name,indices),packet,kind,n,len(x))
  self.events.append({'role':name,'expert':expert,'packet_sha256':hashlib.sha256(packet).hexdigest(),'stored_s_used':False})
  self.record(name,'shared_Q80_Q81_half16_wide32'if shared else'grouped_'+kind+'_integer_parts_lane8_XOR4_2_1_source_expression',kind,'Q81_own_packet','original_packed',['device_compiler_FMA_contraction_unqualified','device_F32_intrinsics_unqualified'])
  return out.astype(np.float64)

class Current59Full48OwnState(PriorState):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.projector=ConsumerProjector(self.p,self.rows_backend)
  for collection in(self.gdn,self.qsa,self.ffn):
   for model in collection.values():model.projector=self.projector
  self.ple=OwnedPle(self.p,self.projector,self.identity)
 def tokens(self,ids):
  result=super().tokens(ids);result['unsupported']=[x for x in result['unsupported']if x!='grouped_Q4K_Q5K_Q51_Q80_integer_lane_schedule'];result['unsupported'].append('grouped_integer_source_expression_compiler_contraction_and_device_F32_intrinsics');result['decoded_FP64_projection_fallback_used']=False;return result
