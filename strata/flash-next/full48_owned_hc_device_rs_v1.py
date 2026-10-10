"""Independent original HC device-arithmetic reference, all non-HC inherited."""
import hashlib
import numpy as np
import full48_owned_hc_fma_refinement_v1 as prior
from full48_owned_hc_fma_refinement_v1 import require,finite,f32,hc
from owned_hc_device_rs_protocol_v1 import OwnDeviceRsClient
class OwnedHcDeviceRsReference(prior.OwnedHcFmaRefinement):
 def __init__(self,provider,identity,args,env,source_root,bulk_build_root,device_rs,tile_bytes=64<<20):
  require(isinstance(device_rs,OwnDeviceRsClient),'Declared own RS protocol client required')
  if provider.actual_source:device_rs.require_actual()
  self.device_rs=device_rs;self.owned_rms_records=[]
  super().__init__(provider,identity,args,env,source_root,bulk_build_root,tile_bytes)
 def resolve_owned_rs(self,residual,stem):
  values=[hc.rms_argument(row,self.epsilon_contract['epsilon']) for row in residual];arguments=np.asarray([v['rsqrt_argument_f32'] for v in values],dtype='<f4');rs=self.device_rs.rsqrt_owned(arguments,stem);self.owned_rms_records.append({'role':stem,'source_FMA_XOR_arguments':values,'residual_sha256':hashlib.sha256(hc.raw(residual)).hexdigest(),'device_sequence':len(self.device_rs.records),'argument_LE_F32_hex':arguments.tobytes().hex(),'captured_operand_used':False});return rs
 def hc_read(self,residual,stem,inject=True):
  require(type(inject) is bool,'Explicit HC inject branch required');r=finite(residual,(4,2560));norm=finite(self.projector.vector(stem+'norm.weight').reshape(4,2560),(4,2560))
  rs=self.resolve_owned_rs(r,stem);xn=f32(f32(r*norm)*rs[:,None]);down=self.projection(stem+'down.weight',xn.reshape(-1));scaled=f32(down/4);low=np.asarray([hc.host.f32(float(x)*hc.sigmoid_candidate(float(x))) for x in scaled],dtype='<f4');gate=self.projection(stem+'up.weight',low).reshape(4,2560)
  total=np.zeros(2560,dtype='<f4')
  for stream in range(4):
   sig=np.asarray([hc.sigmoid_candidate(float(x)) for x in gate[stream]],dtype='<f4');total=self.invoke('fma',2560,1,'paired',xn[stream],sig,total)
  row={'mixed':f32(total/4),'normalized':xn,'down':down,'gate':gate}
  if inject:row['inject']=self.projection(stem+'inject.weight',xn.reshape(-1))
  self.hc_calls+=1;return row
 def tokens(self,token_ids):
  ids=list(token_ids);require(len(ids)==4,'This exploratory fullmodel arm admits exact prefix4 only');begin=len(self.device_rs.records);result=super().tokens(ids);result.update(lane='independent_original48_HC_device_rsqrt_reference_v1',HC_device_RS_records=self.device_rs.records[begin:],HC_owned_argument_records=self.owned_rms_records,HC_device_arithmetic_reference=True,HC_device_operation='sycl::rsqrt',HC_internal_argument_observed=False,universal_correct_rounding_claim=False,ULP_adjustment_or_lookup_used=False,non_HC_primitives_changed=False,captured_inputs_used=False,captured_states_or_selected_ids_used=False,full_model_math_qualified=False,tolerance_gate=None);return result
