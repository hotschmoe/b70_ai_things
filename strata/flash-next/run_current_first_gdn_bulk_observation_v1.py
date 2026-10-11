#!/usr/bin/env python3
"""Root-only own first HC/GDN locals vs saved43 targets, one final full4 close."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
import run_current_full48_own_state_bulk_v1 as full
import current_first_gdn_bulk_observation_v1 as observe
import compare_current_full48_num10_v1 as comparison
from source43_num10_original_reference_v3 import SHAPES,PACKETS
from serial37_canonical_json_v3 import read_unique
require=full.require;sha=full.sha;ROOT=full.ROOT;HERE=full.HERE;PLAN=HERE/'current-first-gdn-bulk-observation-source-plan-v1.json'
def source_binding():
 p=read_unique(PLAN)
 for name,want in p['files'].items():require(sha(ROOT/name)==want,'First GDN observation source changed '+name)
 full.source_binding()
 return sha(PLAN)
def targets(root):
 root=Path(root).resolve();report=read_unique(root/'report.json');require(sha(root/'report.json')==comparison.NATIVE_SHA and report['passed']is True and report['errors']==[],'Exact historical43 targets required');artifact=lambda p:comparison.artifact(root,report,p)
 corpus=read_unique(artifact(root/'plan.snapshot.json'));require(sha(root/'plan.snapshot.json')==report['plan_sha256'],'Original target corpus changed');source=read_unique(artifact(root/'source-plan.snapshot.json'));require(sha(root/'source-plan.snapshot.json')==report['source_plan_sha256']and sha(artifact(root/'producer.snapshot.py'))==source['files']['strata/flash-next/run_source43_num10_zero_v3.py'],'Actual historical43 source/producer changed');directory=root/'source43-num10-ON';recipe=read_unique(artifact(directory/'recipe.json'));result=read_unique(artifact(directory/'result.json'));phase=read_unique(artifact(directory/'phase_receipt.json'));artifact(directory/'command.json');require(recipe['engine_generation']==43 and type(recipe['engine_generation'])is int and recipe['materialized_mode']==1 and type(recipe['materialized_mode'])is int and result['passed']is True and result['engine_rc']==0 and type(result['engine_rc'])is int and result['errors']==[]and result['forced_cleanup']is False and result['removal']['removed']is True and phase['passed']is True and phase['result_sha256']==sha(directory/'result.json')and phase['command_sha256']==sha(directory/'command.json'),'Original target mode1/terminal/phase differs');requests=read_unique(artifact(directory/'requests.json'));require([r['ordinal']for r in requests]==[1,2,3,4],'Four original requests required');out=[]
 for row in requests:
  i=row['ordinal'];raw=row['raw'];require(raw['ids']==[248045]and type(raw['fresh'])is int and raw['fresh']==1 and raw['pin']is None and raw['stop_sent']is False and read_unique(artifact(directory/('request-%d.json'%i)))==raw,'Original fresh request changed');frame=read_unique(artifact(directory/'layer0'/('l0q8-1-r%d-s0.json'%i)));require(frame['pid']==1 and frame['request']==i and frame['rows']==1 and frame['position']==0 and frame['token']==248045 and frame['gen_ids']==[248045]and frame['binding_sha256']==report['source_plan_sha256']and frame['completed_nonce']==(1<<32)|i,'Original target owner/frame differs');require(len(frame['fields'])==33 and len({f['name']for f in frame['fields']})==33,'Complete original33 roster required');values={};packets={}
  for f in frame['fields']:
   if f['name']not in SHAPES and f['name']not in PACKETS:continue
   name=f['name'];p=Path(f['file']);require(p.parent==Path('/results/layer0'),'Original raw target escaped');path=artifact(directory/'layer0'/p.name);data=path.read_bytes();require(sha(path)==report['artifacts'][str(path.relative_to(root))]and len(data)==f['bytes'],'Original target changed during read')
   if name in SHAPES:require(len(data)==int(np.prod(SHAPES[name]))*4,'Exact original target shape required');values[name]=np.frombuffer(data,dtype='<f4').reshape(SHAPES[name]).copy();require(np.isfinite(values[name]).all(),'Nonfinite native target')
   else:require(len(data)==PACKETS[name],'Exact original packet extent required');packets[name]=data
  require(set(values)==set(SHAPES)and set(packets)==set(PACKETS)and not np.any(values['gdn_state_before'])and not np.any(values['gdn_conv_before']),'Own-zero fresh target roster required');out.append({'request':i,'fields':values,'packets':packets})
 return out,recipe,corpus,report

def execute(a):
 out=a.output.resolve();require(not out.exists(),'New first GDN output required');out.mkdir();report={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'captured_inputs_used':False,'full_model_math_qualified':False,'original_arithmetic_changed':False,'GPU_execution':False}
 try:
  source=source_binding();require(sha(a.model_identity)==a.expected_identity_sha256 and sha(a.config)==a.expected_config_sha256,'Exact freshly supplied identity/config required');cfg=read_unique(a.config);native,recipe,corpus,nr=targets(a.native_root);require(recipe['native_argv'][2:]==cfg['args']and all(recipe['native_env'].get(k)==v for k,v in cfg['env'].items()),'Actual same43 math args/env required');contract=full.candidate.source_contract(43)
  for name,want in contract['consumed_source_sha256'].items():require(corpus['source43']['current_all70_source_sha256'][name]==want,'Recorded native43 math source differs '+name)
  full.candidate.first_HC_gate(a.first_HC_report,full.FIRST_SHA,full.ELF_SHA);full.candidate.schedule([248045],cfg['args'],cfg['env']);provider=full.BoundOriginalFull48Rows(HERE/'original-gguf-reference-foundation-plan-v1.json',a.model_identity);model=full.candidate.CurrentFull48OwnStateBulk(provider,a.expected_identity_sha256,cfg['args'],cfg['env'],generation=43,rows_dot_elf=a.bulk_elf,rows_dot_sha256=full.ELF_SHA,first_HC_report=a.first_HC_report,first_HC_report_sha256=full.FIRST_SHA);owned=observe.independent_first(model);report['computation_terminal_epoch']=time.time();arrays={}
  for group in('fields','own_intermediates'):
   for name,value in owned[group].items():
    x=np.asarray(value,dtype='<f4');require(np.isfinite(x).all(),'Nonfinite own observation');p=out/(name+'.f32');p.write_bytes(x.tobytes());arrays[name]={'shape':list(x.shape),'sha256':sha(p),'native_target_observed':name in SHAPES}
  for name,data in owned['packets'].items():(out/(name+'.q81')).write_bytes(data)
  report.update(arrays=arrays,comparisons=[{'request':row['request'],'fields':{name:comparison.metrics(value,row['fields'][name])for name,value in owned['fields'].items()},'packet_bitwise':{name:data==row['packets'][name]for name,data in owned['packets'].items()}}for row in native],native_target_report_sha256=comparison.NATIVE_SHA,model_identity_sha256=a.expected_identity_sha256,config_sha256=a.expected_config_sha256,source_plan_sha256=source,own_state_zero=True,source_generation=43,materialized_mode=1,independent_computations=1,native_comparisons=4,bulk_calls=model.rows_backend.calls,native_intrinsics_qualified=False,extra_HC_GDN_native_targets_observed=False)
  model.rows_backend.recheck();require(full.candidate.source_contract(43)==contract and source_binding()==source and sha(a.model_identity)==a.expected_identity_sha256 and sha(a.config)==a.expected_config_sha256,'Closing own input/source changed');targets(a.native_root)
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  try:
   from run_source44_num38_original_reference_v2 import closing_identity
   report['postCPU_full4']=closing_identity(out,time.time())
  except BaseException as exc:report['errors'].append('fresh-closing: '+type(exc).__name__+': '+str(exc))
  try:report['closing_source_plan_sha256']=source_binding()
  except BaseException as exc:report['errors'].append('closing-source: '+type(exc).__name__+': '+str(exc))
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifacts']={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='report.json'};(out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='ascii')
 require(report['passed'],'Own first GDN observation/closing failed');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--native-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--expected-config-sha256',required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--expected-identity-sha256',required=True);p.add_argument('--bulk-elf',type=Path,required=True);p.add_argument('--first-HC-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(__debug__,'Unoptimized Python required')
 for sig in(signal.SIGHUP,signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(_ for _ in()).throw(InterruptedError('Controlled original first GDN CPU interruption')))
 execute(a)
if __name__=='__main__':main()
