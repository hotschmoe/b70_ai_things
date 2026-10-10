"""NEW cacheOFF fresh numerical counterpart to genuine positive API schema6/9."""
import argparse,copy,os,sys,time
from pathlib import Path
import batch_numerical_execution_v7 as original
import batch_numerical_proofs_v7 as proof
import batch_api_cache_positive_v2 as positive
import validate_batch_api_cache_positive_v2 as positive_reader
import run_batch_serial_cacheoff_v8 as serial
from audit_batch_numerical_suite_v7 import snapshot_plan_join,vectors
ROOT=original.ROOT;HERE=original.HERE;c1=original.c1;sha=proof.sha;read=proof.read;write=proof.write;require=proof.require
lane_contract=original.lane_contract;expected_stage_ranges=original.expected_stage_ranges;verify_model_identity=original.verify_model_identity
SOURCE_PLAN=HERE/'serial-positive-cacheoff-source-plan-v10.json'
CONTROL_SCOPE='fresh cacheOFF GEN1 full49 numerical control; actual positive collector prefix source only'
REGISTRY_SCOPE='actual positive collector accepted entry/current global digest; control native has no API served alias'

def collector_header(directory):
 """Reject impossible/old/missing producer metadata before fullscan or GPU."""
 root=Path(directory).resolve();parent=read(root/'parent-qualification.json');plan=read(root/'input-plan.snapshot.json');child=read(root/'child/report.json')
 require(plan.get('schema')==6 and plan.get('harness_generation')==9 and plan.get('api_cache_positive_generation')==2 and plan['kind']=='api' and plan['diagnostic']==1 and plan['slots']==2,'Only genuine NEW positiveAPI2 ON collector metadata accepted')
 require(child.get('schema')==6 and child.get('harness_generation')==9 and child.get('api_cache_positive_generation')==2 and 'plan_sha256' in child,'Actual positive producer declared snapshotSHA required before modelhash/GPU')
 snapshot_plan_join(root,parent,plan,child)
 require(parent['passed'] is True and parent['scoped_collection_or_serial_arm_qualified'] is True and parent['child_return_code']==0 and parent['owned_containers_terminal'] is True and parent['forced_cleanup'] is False and parent['interrupted'] is False and not parent['errors'] and all(parent[k] is True for k in ('pre_health_passed','post_health_passed','kernel_fault_gate_passed')) and parent['source_guard_generation']==3,'Actual positive collector source/health/owned metadata failed')
 require(parent['wrapper_sha256']==sha(HERE/'qualify_batch_api_cache_positive_v2.py') and parent['controller_sha256']==sha(Path(positive.__file__))==plan['driver_sha256'] and parent['child_report_sha256']==sha(root/'child/report.json') and plan['API_source_plan_sha256']==sha(positive.SOURCE_PLAN),'Actual positive collector source/report metadata association differs')
 require(child['collection_and_teardown_passed'] is True and child['error'] is None and child['actual_cached_state_handoff_qualified'] is False and child['full_model_math_qualified'] is False and child['actual_last_live_handoffs'],'Actual positive collection missing or premature arithmetic qualification claimed')
 require(positive.contract.registry_gate(plan['research_alias'])==plan['registry_binding'],'Actual positive collector current registry entry/append association differs')
 return parent,plan,child

def collector_binding(directory):
 parent,plan,child=collector_header(directory);p,c,q,binding=positive_reader.finalized_binding(directory);require(p==parent and c==child and q==plan,'Actual positive collector changed across raw/current proof admission')
 require(binding['actual_positive_transfer_records_observed'] is True and binding['actual_cached_state_handoff_qualified'] is False and binding['full_model_math_qualified'] is False,'Actual positive collection scope differs');raw=vectors(Path(directory));require(raw,'Actual positive collector full49 source corpus required')
 return parent,plan,child,binding,raw

def prepared_config(collector_plan):
 # Only cache/observer geometry changes; core arithmetic/source remain identical.
 args=list(collector_plan['args']);env=dict(collector_plan['env'])
 positive.api.set_arg(args,'--prompt-cache',0);env.update(STRATA_BATCH_FULL_STATE_CHAIN='0',STRATA_BATCH_PUBLIC_PREFIX='0')
 return args,env

def cheap_plan(plan):
 require(__debug__ and sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE','0') in ('','0'),'Frozen assert gates require unoptimized Python')
 require(plan.get('schema')==7 and plan.get('harness_generation')==10 and plan.get('serial_positive_generation')==10 and plan['kind']=='serial' and plan['driver_sha256']==sha(Path(__file__)),'Explicit NEW positive cacheOFF serial producer required')
 require(plan['serial_source_plan_sha256']==sha(SOURCE_PLAN),'Exact positive serial sourceplan identity differs')
 for name,want in read(SOURCE_PLAN)['files'].items():require(sha(ROOT/name)==want,'Frozen positive serial dependency changed '+name)
 parent,source,child=collector_header(plan['batch_parent']);require(plan['collector_header_sha256']==sha(Path(plan['batch_parent'])/'parent-qualification.json') and plan['collector_report_sha256']==sha(Path(plan['batch_parent'])/'child/report.json') and plan['collector_plan_sha256']==parent['plan_sha256'],'Actual positive collector complete snapshot/report/header identity differs')
 require(plan['slots']==source['slots']==2 and plan['diagnostic']==1,'Actual positive2 serial scope differs')
 require(plan['prepared']==source['prepared'] and plan['prepared_sha256']==source['prepared_sha256'] and plan['engine_receipt_sha256']==source['engine_receipt_sha256'] and plan['engine_root']==source['engine_root'] and plan['cards']==source['cards'] and plan['stage_ranges']==source['stage_ranges'] and plan['image']==source['image'] and plan['pack']==source['pack'] and plan['lane']==source['lane']=='source35','Exact matching positive collector source/topology differs')
 require(plan['registry_binding']==source['registry_binding'] and plan['research_alias']==source['research_alias'],'Exact positive collector current registry association differs')
 require(plan['control_recipe_scope']==CONTROL_SCOPE and plan['registry_binding_scope']==REGISTRY_SCOPE,'Numerical control/source registry scope differs')
 args,env=prepared_config(source);require(plan['args']==args and plan['env']==env,'Only declared cacheOFF/observer profile adaptation accepted');serial.config(plan)
 jobs=read(Path(plan['batch_parent'])/'child/serial-jobs.json')['jobs'];selected=jobs[plan['group_index']*6:plan['group_index']*6+6];require(type(plan['group_index']) is int and 0<=plan['group_index']<(len(jobs)+5)//6 and 1<=len(selected)<=6,'Actual selected positive source group absent')
 require(plan['serial_jobs_sha256']==sha(Path(plan['batch_parent'])/'child/serial-jobs.json') and plan['selected_actual_jobs']==selected and plan['serial_group_job_count']==len(selected) and plan['max_new_by_request']==[1]*len(selected),'Complete actual source jobs/selected GEN1 budget differs')
 return parent,source,child

def manifest_binding(plan):
 parent,source,child=cheap_plan(plan);p,q,c,binding,raw=collector_binding(plan['batch_parent']);require(p==parent and q==source and c==child and binding==plan['collector_finalized_binding'],'Actual current source/raw positive collector join changed')
 prepared,chain=proof.genuine_baseline(Path(plan['prepared']),plan['lane']);require(chain==plan['baseline_source_proof']==source['baseline_source_proof'] and prepared['engine_receipt_sha256']==plan['engine_receipt_sha256'],'Exact genuine source35/C113 baseline changed');return chain

def prepare(a):
 parent,source,child=collector_header(a.batch_parent);require(a.prepared.resolve()==Path(source['prepared']).resolve(),'Serial mustuse matching actual positive collector preparation')
 p,q,c,binding,raw=collector_binding(a.batch_parent);require(p==parent and q==source and c==child,'Actual positive collector changed duringprepare');jobs=read(a.batch_parent/'child/serial-jobs.json')['jobs'];require(type(a.group_index) is int and 0<=a.group_index<(len(jobs)+5)//6,'Actual bounded source group required');selected=jobs[a.group_index*6:a.group_index*6+6];prepared,chain=proof.genuine_baseline(a.prepared,source['lane']);require(chain==source['baseline_source_proof'],'Matching currentC113 baseline differs');identity=verify_model_identity(a.model_identity,read(HERE/'model-lock.json'),[Path(r['path']) for r in prepared['model_shards']]);args,env=prepared_config(source)
 plan={k:copy.deepcopy(source[k]) for k in ('prepared','prepared_sha256','engine_receipt_sha256','engine_root','cards','stage_ranges','image','pack','lane','source_plan_sha256','registry_binding','research_alias')};plan.update(schema=7,harness_generation=10,serial_positive_generation=10,kind='serial',diagnostic=1,slots=2,driver_sha256=sha(Path(__file__)),serial_source_plan_sha256=sha(SOURCE_PLAN),baseline_source_proof=chain,model_identity=identity,args=args,env=env,batch_parent=str(a.batch_parent.resolve()),collector_header_sha256=sha(a.batch_parent/'parent-qualification.json'),collector_report_sha256=sha(a.batch_parent/'child/report.json'),collector_plan_sha256=parent['plan_sha256'],collector_finalized_binding=binding,serial_jobs_sha256=sha(a.batch_parent/'child/serial-jobs.json'),group_index=a.group_index,selected_actual_jobs=selected,serial_group_job_count=len(selected),max_new_by_request=[1]*len(selected),actual_cached_state_handoff_qualified=False,full_model_math_qualified=False)
 plan.update(control_recipe_scope=CONTROL_SCOPE,registry_binding_scope=REGISTRY_SCOPE);manifest_binding(plan);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',plan)

def completed_report(plan,output,result):
 require(read(output/'plan.snapshot.json')==plan,'Actual serial snapshot differs before seal');require(len(result['comparisons'])==49*plan['serial_group_job_count'],'Actual full49 serial group count differs')
 result.update(producer_pid=os.getpid(),producer_interpreter=str(Path(sys.executable).resolve()),producer_interpreter_sha256=sha(Path(sys.executable).resolve()))
 result.update(schema=7,harness_generation=10,serial_positive_generation=10,plan_sha256=sha(output/'plan.snapshot.json'),collection_and_teardown_passed=result['passed'],finished_epoch=time.time(),actual_serial_job_count=plan['serial_group_job_count'],actual_matched_full49_vector_pairs=len(result['comparisons']),actual_cached_state_handoff_qualified=False,full_model_math_qualified=False,cache_qualification_granted=False);result['artifact_bindings']=proof.artifact_bindings(output);write(output/'report.json',result);return result

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);result=serial.run(plan,Path(plan['batch_parent'])/'child',a.output,a.pre_health,plan['group_index']);report=completed_report(plan,a.output,result);return 0 if report['passed'] else 1

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--batch-parent',type=Path,required=True);a.add_argument('--group-index',type=int,required=True);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
