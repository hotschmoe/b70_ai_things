#!/usr/bin/env python3
"""Closed harness40 paired2 OFF/ON and complete actual serial49 readonly gate."""
import copy,json,os,sys
from pathlib import Path
import audit_batch_numerical_suite_v40 as audit
import batch_numerical_execution_v40 as ctrl
import private_native_protocol_recollection_v2 as protocol
from batch_numerical_prefixes_v2 import compare_all
from batch_numerical_proofs_v40 import read,write,sha,require,validate_artifacts,native_row_event,exact_producer_counters,owner_proofs
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE_PLAN=HERE/'private-native-offon-source37-source-plan-v1.json'
H40_SHA='914c2f3c745e93a84f606b4155e1636e2b447971245011a3ea8297bc2a52c62a'
def source_binding():
 plan=read(SOURCE_PLAN);require(sha(HERE/'batch-current-source37-v4-source-plan.json')==H40_SHA and plan['harness_source_plan_sha256']==H40_SHA,'Exact frozen harness40/source37 reader required')
 for name,digest in plan['files'].items():require(sha(ROOT/name)==digest,'Private reader frozen source changed '+name)
 return {'source_plan_sha256':sha(SOURCE_PLAN),'files':plan['files']}
def matched_plans(off,on):
 require(off['harness_generation']==on['harness_generation']==40 and off['slots']==on['slots']==2 and off['kind']==on['kind']=='native' and off['diagnostic']==0 and on['diagnostic']==1 and off['cards']==on['cards']==[0,1],'Exact actual source37harness40 paired2 native OFF/ON required')
 ignored={'diagnostic','research_alias','registry_binding'};require(set(off)==set(on) and {k:v for k,v in off.items() if k not in ignored}=={k:v for k,v in on.items() if k not in ignored},'Matched OFF/ON complete source/args/env/corpus/baseline/math/settings differ')
 for p in (off,on):
  alias=ctrl.api.experimental_alias({'args':p['args'],'env':p['env']},2,bool(p['diagnostic']),p['lane']);require(p['research_alias']==alias and p['registry_binding']==ctrl.api.registry_gate(alias),'Exact accepted OFF/ON source alias association differs')
 require(off['registry_binding']['sha256']==on['registry_binding']['sha256'],'Actual global registry changed betweenarms')
 return True
def native_command_binding(root,parent,plan):
 import shlex
 root=Path(root).resolve();child=root/'child';binding=sha(child/'plan.snapshot.json');name='b70-prefix-'+str(parent['child_pid'])+'-native2';env=dict(plan['env']);env.update(STRATA_ARTIFACT_IDENTITY_SHA256=binding,STRATA_BATCH_FIDELITY_DIAG=str(plan['diagnostic']));engine=Path(plan['engine_root']);model=ROOT/read(HERE/'model-lock.json')['destination']
 command=['docker','run','-i','--name',name,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(model)+':/model:ro','-v',str(child.resolve())+':/results']
 for key,value in sorted(env.items()):command+=['-e',key+'='+str(value)]
 command += [plan['image'],'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+plan['args'])]
 require(read(child/'command.json')==command,'Actual complete native launch/PID/image/env/mount/source command differs')
 return {'container':name,'command_sha256':sha(child/'command.json'),'effective_environment':env,'actual_physical_cards':plan['cards']}
def serial_roster_path(on_root):
 on_root=Path(on_root).resolve();return on_root.parent/(on_root.name+'.serial49-roster-v1.json')
def serial_binding(off_root,on_root,plan,batch,roster_path):
 off_root=Path(off_root).resolve();on_root=Path(on_root).resolve();roster_path=Path(roster_path).resolve();require(roster_path==serial_roster_path(on_root) and not roster_path.is_symlink(),'Exact NEW external serial49 roster path required')
 r=read(roster_path);require(r['schema']==1 and r['off_root']==str(off_root) and r['on_root']==str(on_root) and r['on_parent_sha256']==sha(on_root/'parent-qualification.json') and r['off_parent_sha256']==sha(off_root/'parent-qualification.json') and r['jobs_sha256']==sha(on_root/'child/serial-jobs.json'),'Actual source/ON/OFF selected jobs roster association differs')
 jobs=read(on_root/'child/serial-jobs.json')['jobs'];required=list(range((len(jobs)+5)//6));require(jobs and type(r['groups']) is list and len(r['groups'])==len(required),'Every actual selected serialgroup required')
 serial={};seen=[];bindings=[]
 for row in r['groups']:
  root=Path(row['root']).resolve();require(root not in (on_root,off_root) and row['group_index'] in required and row['group_index'] not in seen and row['parent_sha256']==sha(root/'parent-qualification.json') and row['plan_sha256']==sha(root/'input-plan.snapshot.json'),'Actual unique source-bound serial group/parent differs')
  parent,source,child=audit.parent_arm(root);require(source['kind']=='serial' and source['slots']==2 and source['group_index']==row['group_index'] and Path(source['batch_parent']).resolve()==on_root and source['engine_receipt_sha256']==plan['engine_receipt_sha256'] and source['prepared_sha256']==plan['prepared_sha256'] and source['cards']==plan['cards'] and source['args']==plan['args'] and source['env']==plan['env'],'Actual serial control topology/math/source/context differs')
  data=audit.serial_vectors(root);require(not(set(serial)&set(data)),'Duplicate actual serial49 vectors');serial.update(data);seen.append(row['group_index']);bindings.append({'root':str(root),'parent_sha256':row['parent_sha256'],'group_index':row['group_index'],'vector_count':len(data)})
 require(sorted(seen)==required and len(batch)==len(serial)==len(jobs)*49,'Complete selected jobs times full48+head required')
 compared=compare_all(batch,serial);require(compared['passed'],'Actual allselected full49 serial byte equality failed; no tolerance waiver')
 return {'roster_path':str(roster_path),'roster_sha256':sha(roster_path),'groups':bindings,'jobs':jobs,'comparisons':compared,'actual_complete_serial49_qualified':True,'original_independent_model_math_qualified':False}
def finalized_binding(off_root,on_root):
 require(__debug__ and sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE','0') in ('','0'),'Strict readonly source/parser assertions required');source=source_binding();off_root=Path(off_root).resolve();on_root=Path(on_root).resolve();require(off_root!=on_root,'Distinct immutable OFF/ON roots required')
 off_parent,off,off_child=audit.parent_arm(off_root);on_parent,on,on_child=audit.parent_arm(on_root);matched_plans(off,on)
 outputs={};commands={}
 for label,root,parent,plan,child in [('OFF',off_root,off_parent,off,off_child),('ON',on_root,on_parent,on,on_child)]:
  require(child['initial_native_collection_and_teardown_passed'] is True and child['engine_rc']==0 and child['state']['ExitCode']==0 and child['state']['Running'] is False and not child['state'].get('OOMKilled') and not child['state'].get('Error') and child['removed'] is True and child['error'] is None and child.get('failure_diagnostic') is None,'Actual native collector/EOF/normal ownedterminal failure')
  commands[label]=native_command_binding(root,parent,plan);validate_artifacts(root/'child',child['artifact_bindings']);trace=(root/'child/engine.combined.log').read_text();outputs[label]=protocol.recollect(plan,trace,read(root/'child/requests.json'))
 histories=protocol.compare_histories(outputs['OFF'],outputs['ON'],on['cancel_index']);trace=(on_root/'child/engine.combined.log').read_text();stages=[(i,a,b) for i,(a,b) in enumerate(on['stage_ranges'])];real_n=native_row_event(trace,2);policy={(2001+i,'admission'):{'reused':0,'read_from':0,'reread_to':-1} for i in range(2)};counters=exact_producer_counters(trace,{2001,2002},policy)
 require(outputs['ON']['actual_serial_jobs']==read(on_root/'child/serial-jobs.json'),'Actual complete selected consumed-prefix serial jobs changed')
 batch=audit.vectors(on_root);serial=serial_binding(off_root,on_root,on,batch,serial_roster_path(on_root));owners=owner_proofs(trace,stages,2,'source37',True)
 # Recheck original artifact/current source/identity after all raw comparisons.
 for root in (off_root,on_root):audit.parent_arm(root)
 require(source_binding()==source,'Private current source changed during readonlywork')
 return {'schema':1,'passed':True,'source_lane':'source37','harness_generation':40,'engine_receipt_sha256':on['engine_receipt_sha256'],'cards':[0,1],'off_parent_sha256':sha(off_root/'parent-qualification.json'),'on_parent_sha256':sha(on_root/'parent-qualification.json'),'source_binding':source,'actual_launch_bindings':commands,'actual_complete_histories':outputs,'matched_native_histories':histories,'actual_ON_requested_Nrow':real_n,'actual_ON_counters':counters,'actual_ON_complete_raw49':{'vectors':len(batch),'selected_jobs':len(serial['jobs'])},'actual_complete_serial49':serial,'actual_ON_owner_logical_frees':owners,'actual_OFF_raw49_observed':False,'actual_OFF_Nrow_event_observed':False,'actual_stdin_cancellation_bytes_logged':False,'cancellation_proof':'Pinned original send predicate reconstructed plus actual matching cancelterminal acknowledgement; no claimed independentlylogged sendtime','full_model_math_qualified':False,'broad_quality_qualified':False,'concurrent_cache_qualified':False,'latency_qualified':False}
def prepare_serial_roster(off_root,on_root,groups,output):
 off_root=Path(off_root).resolve();on_root=Path(on_root).resolve();output=Path(output).resolve();require(output==serial_roster_path(on_root) and not output.exists(),'Exact NEW external roster output required');source_binding();a,off,_=audit.parent_arm(off_root);b,on,_=audit.parent_arm(on_root);matched_plans(off,on);rows=[]
 for root in groups:
  root=Path(root).resolve();parent,plan,_=audit.parent_arm(root);require(plan['kind']=='serial' and Path(plan['batch_parent']).resolve()==on_root,'Actual serialgroup source differs');rows.append({'root':str(root),'group_index':plan['group_index'],'parent_sha256':sha(root/'parent-qualification.json'),'plan_sha256':sha(root/'input-plan.snapshot.json')})
 value={'schema':1,'off_root':str(off_root),'on_root':str(on_root),'off_parent_sha256':sha(off_root/'parent-qualification.json'),'on_parent_sha256':sha(on_root/'parent-qualification.json'),'jobs_sha256':sha(on_root/'child/serial-jobs.json'),'groups':rows};write(output,value);return value
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--off-root',type=Path,required=True);p.add_argument('--on-root',type=Path,required=True);p.add_argument('--serial-root',type=Path,action='append');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.serial_root:result=prepare_serial_roster(a.off_root,a.on_root,a.serial_root,a.output)
 else:require(not a.output.exists() and not a.output.resolve().is_relative_to(a.off_root.resolve()) and not a.output.resolve().is_relative_to(a.on_root.resolve()),'NEW readonly report outside original arms required');result=finalized_binding(a.off_root,a.on_root);write(a.output,result)
 print(json.dumps({'passed':result.get('passed',False),'full_model_math_qualified':False,'output':str(a.output)}))
