"""Saved admitted62/independent own-state first-jump analysis; no provider/GPU."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from native62_hc_rms_comparison_v2 import consume,metrics,require,sha,OWN,OWN_SHA
from serial37_canonical_json_v3 import read_unique
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'native62-QSA35-first-jump-analysis-source-plan-v1.json'
PRODUCER_SHA='18f8aec9cd3a23c0d47c67ef84255f90c9b1d61b3eb88c24e18e44e15c392646'
def source_binding():
 before=sha(PLAN)
 for n,w in read_unique(PLAN)['files'].items():require(sha(ROOT/n)==w,'Firstjump source changed '+n)
 require(sha(PLAN)==before,'Firstjump plan changed');return before

def residual_update(old,new):return np.asarray(new,dtype=np.float64)-np.asarray(old,dtype=np.float64)
def analyze(a):
 source=source_binding();root=a.root.resolve();require(sha(root/'report.json')==PRODUCER_SHA and sha(a.public_binding)==a.public_sha256,'Exact completed producer/public report SHA required');report=read_unique(root/'report.json');pub=read_unique(a.public_binding);require(report['passed']is True and report['errors']==[]and pub['passed']is True and pub['producer_report_sha256']==PRODUCER_SHA and pub['source_generation']==62 and pub['materialized_mode']==0 and pub['numeric_pass_assigned']is False,'Actual current62 diagnostic admission required');require(sha(OWN/'report.json')==OWN_SHA,'Exact independent own report required');own=read_unique(OWN/'report.json');pins={root/'report.json':PRODUCER_SHA,Path(a.public_binding):a.public_sha256,OWN/'report.json':OWN_SHA};raw=pub['current_mode0_targets']['native62-hc-rms-ON'];comparison=pub['ON_independent_full48_comparison'];require(comparison['independent_own_report_sha256']==OWN_SHA,'Public independent-reference association differs')
 def load_own(name,shape):
  d=own['arrays'][name];p=OWN/(name+'.f32');require(d['shape']==list(shape)and own['artifacts'][p.name]==d['sha256'],'Independent owned array shape/hash differs');data=consume(p,int(np.prod(shape))*4,d['sha256']);pins[p]=d['sha256'];return np.frombuffer(data,dtype='<f4').reshape(shape).copy()
 def load_native(f):
  p=Path(f['path']);require(p.resolve()==p and p.is_relative_to(root)and report['artifacts'][str(p.relative_to(root))]==f['sha256']and f['bytes']==40960,'Actual observed phase producer/shape join differs');data=consume(p,40960,f['sha256']);pins[p]=f['sha256'];return np.frombuffer(data,dtype='<f4').reshape(4,2560).copy()
 try:
  owned={(l,k):load_own('own-l%d-%s'%(l,k),(4,2560))for l in range(32,37)for k in('input','attention','ffn')};rows=[]
  for r in raw['requests']:
   native={}
   for frame in raw['P30']['frames']:
    if frame['binding']['request']==r['ordinal']:
     for f in frame['fields']:
      if 32<=f['layer']<=36:native[(f['layer'],f['phase'])]=load_native(f)
   require(set(native)==set(owned),'Exact five-layer/threephase native observations required');layers=[]
   for l in range(32,37):
    saved=next(x for x in comparison['comparisons'][r['ordinal']-1]['layers']if x['layer']==l);phase={k:metrics(owned[(l,k)],native[(l,k)])for k in('input','attention','ffn')};require(all(phase[k]==saved[k]for k in phase),'Actual recomputed phase metrics differ from publicadmission')
    layers.append({'layer':l,'kind':'QSA'if l%4==3 else'GDN','phases':phase,'attention_saved_residual_update':metrics(residual_update(owned[(l,'input')],owned[(l,'attention')]),residual_update(native[(l,'input')],native[(l,'attention')])),'FFN_saved_residual_update':metrics(residual_update(owned[(l,'attention')],owned[(l,'ffn')]),residual_update(native[(l,'attention')],native[(l,'ffn')])),'updates_are_FP64_difference_of_saved_F32_residuals_not_captured_block_outputs':True})
   continuity=[{'previous_layer':l-1,'layer':l,'own_prevFFN_nextInput_bitwise':owned[(l-1,'ffn')].tobytes()==owned[(l,'input')].tobytes(),'native_prevFFN_nextInput_bitwise':native[(l-1,'ffn')].tobytes()==native[(l,'input')].tobytes()}for l in range(33,37)];rows.append({'request':r['ordinal'],'layers':layers,'continuity':continuity})
  values=load_own('own-qsa35-values',(1,2,256));keys=load_own('own-qsa35-keys',(1,2,256));require(np.array_equal(values.astype('<f2').astype('<f4'),values),'Own KV values must retain actualdeclared F16storage image')
  return {'schema':1,'CONFIG':{'source_generation':62,'materialized_mode':0,'explicit_sqrt_divide':1,'literal_prefix':[248045],'actual_requests':4,'kind':'saved admitted arrays; no new original weight/provider or GPU'},'COMMAND':{'producer_report_sha256':PRODUCER_SHA,'public_report_sha256':a.public_sha256,'own_full48_report_sha256':OWN_SHA,'source_plan_sha256':source},'RESULT':{'requests':rows,'own_QSA35_KV_shape':list(values.shape),'own_QSA35_values_are_F16_representable':True,'head_metrics_from_original_public':comparison['comparisons'][0]['head'],'OFF_bitwise_actual59_retained':pub['baseline_original59_tensor_bitwise'],'consumed_sha256':{str(p):w for p,w in pins.items()}},'VERDICT':{'first_requested_material_jump':'QSA35 input~1e-12 toattention~4e-9 thenFFN~2.65e-8','numeric_pass_assigned':False,'tolerance_fitted':False,'actual_QSA35_intermediates_observed':False,'single_causal_cell_is_source_and_ownstate_condition_not_actual_selection_observation':True,'candidate_seams':['HCmixed->Q81 block scale/code threshold','originalQ80 projected V and q_full gate arithmetic','Vprojection->residentFP16 conversion','nativeexp sigmoid gatedattention','gatedF32->outputQ81 scale/code threshold','originalQ80 output projection andHCwrite'], 'score_Q_K_RoPE_effect':'One valid causal cell gives weight1 mathematically; requires actualstep/selectedIDs to admit condition, not a GPUresultclaim','minimal_next_own_CPU_probe':'Observe independentown QSA35 HCmixed/packets/Vprojected/gate/gated/output withunchangedown weights/state; no primarynativefeeds','minimal_missing_native_if_needed':['hc_mixed','mixed_q81','v_projected','q_full_projected','Vphysicalcell0','gated','gated_q81','output_projected','step','selected_ids'],'existing_hook':'QSA3 atverify908 l==3, notlayer35; cannotreuseold3captureas35','no_new_GPU_capture_requested':True,'native_operands_as_primary':False}}
 finally:
  require(source_binding()==source,'Closing firstjump source changed')
  for p,w in pins.items():require(sha(p)==w,'Consumed saved firstjump array changed '+str(p))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in('root','public-binding','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--public-sha256',required=True);a=p.parse_args();require(not a.output.exists(),'New readonly analysis output required');a.output.write_text(json.dumps(analyze(a),indent=2,allow_nan=False)+'\n',encoding='ascii')
