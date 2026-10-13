"""Readonly saved-own/native HC/packet localization; no model/provider execution."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import run_current59_first_gdn_reference_v1 as first
from source43_num10_original_reference_v3 import SHAPES,PACKETS
OWN_SHA='620a4ec5f3c8a1bf053afe8529658eb6e02a01a792d1ea403b0013cc6c32a504'
HERE=Path(__file__).resolve().parent
PLAN=HERE/'current59-first-hc-packet-analysis-source-plan-v1.json'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source():
 before=sha(PLAN);v=json.loads(PLAN.read_text())
 for n,w in v['files'].items():require(sha(first.ROOT/n)==w,'Analysis source changed '+n)
 require(sha(PLAN)==before,'Analysis plan changed');return before

def ulp_distance(a,b):
 def order(x):
  bits=np.asarray(x,dtype='<f4').view('<u4').astype(np.int64);return np.where(bits>=0x80000000,0x80000000-(bits&0x7fffffff),0x80000000+bits)
 return np.abs(order(a)-order(b))
def field_metrics(a,b):
 x=np.asarray(a,dtype='<f4');y=np.asarray(b,dtype='<f4');u=ulp_distance(x,y);result=first.metrics(x,y);result.update(differing_F32_elements=int(np.count_nonzero(x.view('<u4')!=y.view('<u4'))),max_ULP_distance=int(u.max()),ULP_distance_p50=float(np.quantile(u,.5)),ULP_distance_p95=float(np.quantile(u,.95)))
 return result

def packet_parts(data):
 b=np.frombuffer(data,np.uint8).reshape(-1,36);return b[:,:2].copy().view('<f2').reshape(-1).astype(np.float64),b[:,2:4].copy().view('<f2').reshape(-1).astype(np.float64),b[:,4:].view(np.int8).astype(np.int32)
def packet_metrics(own,native,own_input,native_input):
 od,os,oq=packet_parts(own);nd,ns,nq=packet_parts(native);require(oq.shape==nq.shape and own_input.size==oq.size and native_input.size==oq.size,'Saved packet/own operand dimensions differ');x=np.asarray(own_input,dtype='<f4').reshape(-1,32);y=np.asarray(native_input,dtype='<f4').reshape(-1,32);code=nq-oq;scale=nd-od
 orecon=od[:,None]*oq;nrecon=nd[:,None]*nq;sd=scale[:,None]*oq;qd=nd[:,None]*code
 require(np.array_equal(nrecon-orecon,sd+qd),'Diagnostic algebra decomposition failed')
 rows=[]
 for i in np.flatnonzero(scale!=0):
  lo,hi=sorted((float(od[i]),float(nd[i])));mid=(lo+hi)/2;om=float(np.abs(x[i]).max());nm=float(np.abs(y[i]).max())
  rows.append({'block':int(i),'own_scale':float(od[i]),'native_scale':float(nd[i]),'relative_scale_change':float(scale[i]/od[i])if od[i]!=0 else None,'scale_midpoint':mid,'own_absmax':om,'native_absmax':nm,'absmax_difference':nm-om,'absmax_ULP_distance':int(ulp_distance(np.float32(om),np.float32(nm))),'own_absmax_div127_F32':float(np.float32(np.float32(om)/np.float32(127))),'native_absmax_div127_F32':float(np.float32(np.float32(nm)/np.float32(127))),'own_absmax_div127_F64_minus_midpoint':om/127-mid,'native_absmax_div127_F64_minus_midpoint':nm/127-mid,'own_argmax_in_block':int(np.argmax(np.abs(x[i]))),'native_argmax_in_block':int(np.argmax(np.abs(y[i]))),'codes_differ_in_block':int(np.count_nonzero(code[i]))})
 return {'blocks':len(od),'scale_blocks_different':int(np.count_nonzero(scale)),'stored_sum_blocks_different':int(np.count_nonzero(os!=ns)),'code_blocks_different':int(np.count_nonzero(np.any(code!=0,axis=1))),'codes_different':int(np.count_nonzero(code)),'max_code_distance':int(np.abs(code).max()),'scale_boundaries':rows,'own_vs_native_F32_input':field_metrics(x,y),'reconstructed_activation':first.metrics(orecon,nrecon),'algebraic_scale_delta_squared_norm':float(np.square(sd).sum()),'algebraic_code_delta_squared_norm':float(np.square(qd).sum()),'algebraic_scale_code_cross_term':float(2*np.sum(sd*qd)),'stored_sum_used_in_Q80_projection':False,'conditional_diagnostic_only':True,'native_operand_used_as_primary_reference':False,'FP64_decomposition_is_not_native_arithmetic':True}

def analyze(a):
 source_sha=source();root=a.own_root.resolve();require(sha(root/'report.json')==OWN_SHA,'Exact executed own firstGDN report required');report=json.loads((root/'report.json').read_text());require(report['passed']is True and report['errors']==[]and report['captured_inputs_used']is False and report['full_model_math_qualified']is False,'Original own computation report scope differs');native,pins=first.targets(a.native_root,a.producer_report_sha256,a.public_binding,a.public_binding_sha256);pins[root/'report.json']=OWN_SHA;own={};ownpack={}
 try:
  for n in SHAPES:
   p=root/(n+'.f32');desc=report['arrays'][n];require(desc['shape']==list(SHAPES[n])and report['artifacts'][p.name]==desc['sha256'],'Original own field/report join differs');data=first.consume(p,int(np.prod(SHAPES[n]))*4,desc['sha256']);pins[p]=desc['sha256'];own[n]=np.frombuffer(data,dtype='<f4').reshape(SHAPES[n]).copy()
  for n,size in PACKETS.items():
   p=root/(n+'.q81');want=report['artifacts'][p.name];ownpack[n]=first.consume(p,size,want);pins[p]=want
  rows=[]
  for target in native:
   fs={n:field_metrics(x,target['fields'][n])for n,x in own.items()};pm={n:packet_metrics(ownpack[n],target['packets'][n],own['attn_mixed'if n=='attn_input_q81'else'gdn_output_gated'],target['fields']['attn_mixed'if n=='attn_input_q81'else'gdn_output_gated'])for n in PACKETS};rows.append({'request':target['ordinal'],'fields':fs,'packets':pm})
  return {'schema':1,'passed':True,'CONFIG':{'source_generation':59,'materialized_mode':0,'literal_prefix':[248045],'fresh_requests':4,'mode':'readonly saved independent own-state versus admitted native comparison targets'},'COMMAND':{'argv_scope':'saved arrays/packets/source only','provider_or_model_reads':False},'RESULT':{'comparisons':rows,'own_report_sha256':OWN_SHA,'source_plan_sha256':source_sha,'consumed_artifacts':{str(p):v for p,v in pins.items()}},'VERDICT':{'numeric_pass':False,'full_math_qualified':False,'rounding_cause_uniquely_observed':False,'native_RMS_sum_argument_scale_observed':False,'first_normalization_contract':'same lane32 FMA/XOR source schedule; hostF32(1/sqrtf) versus SYCL rsqrt and compiler divisions remain unqualified','HC_sigmoid_contract':'host expf versus SYCL exp and division lowering remain unqualified','packet_discontinuity_is_observed_not_a_quantization_fix':True,'native_values_are_diagnostic_targets_only':True,'current_model_or_health_authority':False}}
 finally:
  require(source()==source_sha,'Closing analysis source changed')
  for p,w in pins.items():require(sha(p)==w,'Consumed saved analysis target changed '+str(p))
def main():
 p=argparse.ArgumentParser()
 for n in('own-root','native-root','public-binding','output'):p.add_argument('--'+n,type=Path,required=True)
 for n in('producer-report-sha256','public-binding-sha256'):p.add_argument('--'+n,required=True)
 a=p.parse_args();require(not a.output.exists(),'New readonly diagnostic output required');result=analyze(a);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='ascii')
if __name__=='__main__':main()
