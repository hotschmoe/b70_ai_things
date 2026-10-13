"""Root-only independent current59 source-expression full48 prefix1 candidate."""
import argparse,signal,time,traceback
from pathlib import Path
import numpy as np
import run_current59_first_gdn_reference_v1 as first
from current59_full48_mode0_reference_v3 import Current59Full48OwnState
from serial37_canonical_json_v3 import read_unique
require=first.require;sha=first.sha;write=first.write;ROOT=first.ROOT;HERE=first.HERE
PLAN=HERE/'current59-full48-mode0-reference-source-plan-v3.json'
def source_binding():
 source=sha(PLAN)
 for name,want in read_unique(PLAN)['files'].items():require(sha(ROOT/name)==want,'Current59 full48 reference source changed '+name)
 require(sha(PLAN)==source,'Current59 full48 plan changed');return source

def full_targets(a):
 _,pins=first.targets(a.native_root,a.producer_report_sha256,a.public_binding,a.public_binding_sha256);public=read_unique(a.public_binding);report=read_unique(a.native_root/'report.json');raw=public['current_mode0_targets'];rows=[]
 for own in raw['requests']:
  fields={};head=None
  for frame in raw['P30']['frames']:
   if frame['binding']['request']!=own['ordinal']:continue
   require(frame['binding']['pid']==1 and frame['binding']['first_position']==0 and frame['binding']['rows']==1,'Actual current first-row P30 owner/position differs')
   for f in frame['fields']:
    p=Path(f['path']);require(p.resolve()==p and p.parent==a.native_root/'source59-mode0-fidelity/p30','Current P30 path escaped');data=first.consume(p,40960,f['sha256']);require(report['artifacts'][str(p.relative_to(a.native_root))]==f['sha256'],'Original P30 artifact mismatch');pins[p]=f['sha256'];key=(f['layer'],f['phase']);require(key not in fields,'Duplicate current P30 phase/layer');fields[key]=np.frombuffer(data,dtype='<f4').reshape(4,2560).copy()
  require(set(fields)=={(l,k)for l in range(48)for k in('input','attention','ffn')},'Current complete all48/3phase targets required')
  target=own['meta']['logits'][0];p=Path(target['path']);data=first.consume(p,248320*4,target['sha256']);require(report['artifacts'][str(p.relative_to(a.native_root))]==target['sha256'],'Actual complete current head artifact mismatch');pins[p]=target['sha256'];head=np.frombuffer(data,dtype='<f4').copy()
  rows.append({'ordinal':own['ordinal'],'fields':fields,'head':head})
 return rows,pins

def execute(a):
 require(__debug__ and a.allow_unqualified_estimates is True,'Explicit unoptimized estimate-contract optin required');a.native_root=a.native_root.resolve();out=a.output.resolve();require(not out.exists(),'New full48 own-state output required');out.mkdir();report={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'numeric_pass_assigned':False,'full_model_math_qualified':False,'captured_inputs_used':False,'materialized_mode':0,'mode1_reference_relabelled':False};pins={};source=None
 try:
  source=source_binding();require(sha(a.model_identity)==a.expected_identity_sha256,'Exact fresh original model identity required');targets,pins=full_targets(a);pub=read_unique(a.public_binding);recipe=read_unique(a.native_root/'source59-mode0-fidelity/recipe.json')
  require(recipe['materialized_mode']==0 and recipe['native_env']['STRATA_SYCL_NATIVE_HC']=='1'and all(recipe['native_env'].get(k,'0')=='0'for k in('STRATA_QFUSE','STRATA_HC_Q8','STRATA_HC_Q8_INJECT','STRATA_GR_V3','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD','STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_SYCL_NATIVE_GROUP_F16','STRATA_EXPERT_SPLIT','STRATA_EXPERT_V2','STRATA_EXPERT_STREAM_B','STRATA_EXPERT_PAIR_REUSE','STRATA_LFUSE','STRATA_SHARED_BF16')),'Current supported ordinary consumer math flags required')
  require(all(recipe['native_env'].get(k,'8') in('','8')for k in('STRATA_GU_LANES','STRATA_DOWN_LANES')),'Actual grouped lane8 required')
  from current_full48_own_state_bulk_v1 import first_HC_gate
  first_HC_gate(a.first_HC_report,first.FIRST_HC_SHA,first.ELF_SHA)
  provider=first.BoundOriginalFull48Rows(HERE/'original-gguf-reference-foundation-plan-v1.json',a.model_identity);model=Current59Full48OwnState(provider,a.expected_identity_sha256,a.bulk_elf,first.ELF_SHA,allow_unqualified_estimates=True);owned=model.tokens([248045]);report['computation_terminal_epoch']=time.time();arrays={}
  def save(name,value):
   x=np.asarray(value,dtype='<f4');require(np.isfinite(x).all(),'Independent full48 nonfinite '+name);p=out/(name+'.f32');p.write_bytes(x.tobytes());arrays[name]={'shape':list(x.shape),'bytes':x.nbytes,'sha256':sha(p)}
  for l in owned['layers']:
   for key in('input','attention','ffn'):save('own-l%d-%s'%(l['layer'],key),l[key])
  save('own-first-head',owned['head'])
  for layer,state in owned['gdn_states_owned'].items():
   for key,value in state.items():save('own-gdn%d-%s'%(layer,key),value)
  for layer,state in owned['qsa_states_owned'].items():
   for key in('keys','values','tail','dead','pooled'):save('own-qsa%d-%s'%(layer,key),getattr(state,key))
  save('own-PLE-history',owned['ple_history_owned'])
  comparisons=[]
  for target in targets:
   layers=[{'layer':l['layer'],**{key:first.metrics(l[key],target['fields'][(l['layer'],key)])for key in('input','attention','ffn')}}for l in owned['layers']];comparisons.append({'request':target['ordinal'],'layers':layers,'head':first.metrics(owned['head'],target['head'])})
  report.update(arrays=arrays,comparisons=comparisons,consumer_contracts=owned['consumer_contracts'],unsupported=owned['unsupported'],packet_events=owned['packet_events'],decoded_FP64_projection_fallback_used=owned['decoded_FP64_projection_fallback_used'],bulk_calls=owned['bulk_calls'],source_plan_sha256=source,native_producer_report_sha256=a.producer_report_sha256,native_recovery_sha256=a.public_binding_sha256,model_identity_sha256=a.expected_identity_sha256,accepted_ids=[248045],source_generation=59,own_state_initialized_from_zero=True,mode0_inline_half_actual_compiler_qualified=False,first_prefix_only=True,actual_original_payload_used=True)
 except BaseException as exc:
  report['errors'].append(type(exc).__name__+': '+str(exc));report['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'projection_role':getattr(exc,'reference_role_context',None)}
 finally:
  try:
   require(source is None or source_binding()==source,'Closing full48 source changed')
   for p,want in pins.items():require(sha(p)==want,'Consumed original target changed '+str(p))
   require(sha(a.model_identity)==a.expected_identity_sha256,'Caller identity changed')
  except BaseException as exc:report['errors'].append('read-close: '+str(exc))
  try:
   from run_source44_num38_original_reference_v2 import closing_identity
   report['postCPU_full4']=closing_identity(out,time.time())
  except BaseException as exc:report['errors'].append('fresh-closing: '+str(exc))
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifacts']={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='report.json'};write(out/'report.json',report)
 require(report['passed'],'Current59 full48 computation/identity/source failed');return report

def main():
 p=argparse.ArgumentParser()
 for name in('native-root','public-binding','model-identity','bulk-elf','first-HC-report','output'):p.add_argument('--'+name,type=Path,required=True)
 for name in('producer-report-sha256','public-binding-sha256','expected-identity-sha256'):p.add_argument('--'+name,required=True)
 p.add_argument('--allow-unqualified-estimates',action='store_true');a=p.parse_args()
 for sig in(signal.SIGHUP,signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(_ for _ in()).throw(InterruptedError('Controlled full48 original reference interruption')))
 execute(a)
if __name__=='__main__':main()
