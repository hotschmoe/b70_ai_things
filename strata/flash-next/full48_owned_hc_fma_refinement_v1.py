"""NEW independent original48 HC-only arithmetic refinement, no native inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
import hc35_host_bulk_fma_v1 as bulk
import qualify_hc35_host_bulk_runtime_v1 as qualification
import owned_hc35_f32_block_control_v2 as hc
from full48_owned_composition_storage_v2 import RouteAwareFull48OwnedComposition
from native_storage_first_gdn_estimate_v1 import f32
HERE=Path(__file__).resolve().parent
PLAN=HERE/'full48-owned-hc-fma-refinement-source-plan-v1.json'
def require(ok,message):
 if not ok:raise ValueError(message)
def source_binding(source):
 p=json.loads(PLAN.read_text())
 for n,want in p['files'].items():require(bulk.sha(HERE.parents[1]/n)==want,'HC-only refinement source changed '+n)
 return {'files':p['files'],'HC_source':hc.source_binding(source)}
def finite(value,shape):
 a=np.asarray(value,dtype='<f4');require(a.shape==shape and np.isfinite(a).all(),'Owned HC operand shape/finite differs');return a
class OwnedHcFmaRefinement(RouteAwareFull48OwnedComposition):
 def __init__(self,provider,identity,args,env,source_root,bulk_build_root,tile_bytes=64<<20):
  self.bulk_root=Path(bulk_build_root).resolve();self.bulk_binding=qualification.finalized_binding(self.bulk_root);self.helper=self.bulk_root/'hc35-host-bulk';self.helper_sha=bulk.sha(self.helper);self.refinement_source=source_binding(source_root);self.hc_calls=0
  super().__init__(provider,identity,args,env,source_root,tile_bytes)
  self.epsilon_contract=hc.epsilon_contract(provider) if provider.actual_source else {'epsilon':hc.EPSILON,'synthetic_only':True}
 def invoke(self,op,k,m,mode,a,b,c=None):
  value,receipt=bulk.run_bulk(self.helper,self.helper_sha,op,k,m,mode,hc.raw(a),hc.raw(b),hc.raw(c) if c is not None else None)
  return np.frombuffer(value,dtype='<f4').copy()
 def projection(self,name,operand):
  x=finite(operand,(self.p.shape(name)[0],));shape=self.p.shape(name);require(len(shape)==2 and shape[0]%32==0,'Original HC matrix role required')
  count=shape[1];out=np.empty(count,dtype='<f4');step=min(10240,self.projector.tile//(shape[0]*8));require(step>0,'Original HC decoded row tile exceeds bound')
  for start in range(0,count,step):
   n=min(step,count-start);weights=finite(self.p.rows(name,range(start,start+n)),(n,shape[0]));out[start:start+n]=self.invoke('matrixdot',shape[0],n,'shared',weights,x)
  return out
 def hc_read(self,residual,stem,inject=True):
  require(type(inject) is bool,'Explicit HC inject branch required');r=finite(residual,(4,2560));norm=finite(self.projector.vector(stem+'norm.weight').reshape(4,2560),(4,2560))
  rs=np.asarray([hc.rsqrt_candidate(hc.rms_argument(row,self.epsilon_contract['epsilon'])['rsqrt_argument_f32']) for row in r],dtype='<f4');xn=f32(f32(r*norm)*rs[:,None]);down=self.projection(stem+'down.weight',xn.reshape(-1));scaled=f32(down/4);low=np.asarray([hc.host.f32(float(x)*hc.sigmoid_candidate(float(x))) for x in scaled],dtype='<f4');gate=self.projection(stem+'up.weight',low).reshape(4,2560)
  total=np.zeros(2560,dtype='<f4')
  for stream in range(4):
   sig=np.asarray([hc.sigmoid_candidate(float(x)) for x in gate[stream]],dtype='<f4');total=self.invoke('fma',2560,1,'paired',xn[stream],sig,total)
  row={'mixed':f32(total/4),'normalized':xn,'down':down,'gate':gate}
  if inject:row['inject']=self.projection(stem+'inject.weight',xn.reshape(-1))
  self.hc_calls+=1;return row
 def hc_write(self,residual,block,inject):
  r=finite(residual,(4,2560));b=finite(block,(2560,));i=finite(inject,(4,));gain=np.asarray([hc.host.f32(2*hc.sigmoid_candidate(hc.host.f32(float(x)/4))) for x in i],dtype='<f4');return self.invoke('fma',2560,4,'paired',np.broadcast_to(b,r.shape),np.broadcast_to(gain[:,None],r.shape),r).reshape(r.shape)
 def tokens(self,token_ids):
  ids=list(token_ids);require(len(ids) in (1,2,4,8) and all(type(x) is int and 0<=x<248320 for x in ids),'Accepted original token IDs only')
  before=qualification.finalized_binding(self.bulk_root);require(before==self.bulk_binding and source_binding(self.source_root)==self.refinement_source,'Current qualified bulk/source prerequisite changed')
  self.hc_calls=0;result=super().tokens(ids)
  require(qualification.finalized_binding(self.bulk_root)==before and source_binding(self.source_root)==self.refinement_source,'Bulk/source runtime changed during original48 computation')
  result.update(lane='owned_original48_hc_fma_refinement_candidate_v1',HC_bulk_binding=before,HC_refinement_source=self.refinement_source,HC_epsilon=self.epsilon_contract,HC_calls=self.hc_calls,HC_FMA_XOR_schedule_used=True,HC_write_fused=True,HC_mix_fused=True,HC_host_rsqrt_expf_device_qualified=False,non_HC_primitives_changed=False,captured_inputs_used=False,captured_states_or_selected_ids_used=False,nativebitwise_qualified=False,full_model_math_qualified=False,tolerance_gate=None)
  return result
