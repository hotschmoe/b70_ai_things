#!/usr/bin/env python3
"""One bounded actual native/API/serial arm; each requires its own parent gates."""
import argparse,json,os,sys,time,signal
from pathlib import Path
from types import SimpleNamespace
import c1_serve_controller_combined_v137 as c1
import serial_prefix_qualification_v6 as base
from source_page_watchdog_v3 import guard as page_guard
from batch_numerical_proofs_v37 import genuine_baseline,engine_binding,read,write,sha,require,PLAN_SHA,artifact_bindings,lane_contract,providers,source_observers_off,topology_baselines,case_source_gate
import run_batch_numerical_pilot_v37 as native
import run_batch_api_controls_v37 as api
import run_batch_serial_controls_v37 as serial
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
expected_stage_ranges=base.expected_stage_ranges
DEPENDENCIES=['batch37_runtime_evidence_v1.py','audit_batch_numerical_suite_v37.py','parse_usm_logical_free_trace.py','audit_fidelity_observer_coverage.py','collect_fidelity_observer.py','batch_numerical_execution_v37.py','qualify_batch_numerical_v37.py','batch_numerical_proofs_v37.py','run_batch_numerical_pilot_v37.py','run_batch_api_controls_v37.py','run_batch_serial_controls_v37.py','batch_numerical_protocol_v2.py','batch_numerical_prefixes_v2.py','batch_api_trace_v2.py','batch_api_client_v2.py','batch_api_prefixes_v2.py','audit_batch_capture_lifecycle_v2.py','audit_slot_owner_trace_v1.py','audit_batch_fidelity_coverage_v2.py','audit_batch_fidelity_coverage_v4.py','audit_batch_fidelity_coverage_v5.py','audit_batch_fidelity_coverage_v6.py','audit_stage_mirror_owner_trace_v1.py','merged_numerical_protocol_v2.py','serial_prefix_qualification_v6.py','layer0_numerical_qualification_v9.py','source_page_watchdog_v3.py','c1_trace_contract.py','c1_serve_controller_combined_v137.py','qualify_c1_serving_combined_v137_v2.py','c137_journal_binding_v2.py','c1-combined-v137-parent-source-plan-v2.json','c1-combined-v137-source-plan.json','c1-combined-v137-parent-source-plan.json','hosttrace36-batchstamp37-engine-build-plan-v1.json']

def verify_model_identity(path,lock,shards):
 identity=base.verify_model_identity(path,lock,shards);identity['current_known_pages']=page_guard(shards[2]);return identity

def output_budget_contract(kind,slots,native_budgets,api_budgets,serial_job_count=None):
 require(kind in ('native','api','serial') and type(slots) is int and slots in (2,4,6),'V37 actual native/API/serial2/4/6 kind required')
 require(native_budgets==[32]*slots and all(type(v) is int for v in native_budgets),'Native bounded diagnostic mustdeclare actual32 perrequest')
 require(len(api_budgets)==slots and all(type(v) is int and 2<=v<=64 for v in api_budgets) and 64 in api_budgets,'Separate API longsurvivor64 declaration required')
 require(serial_job_count is None or kind=='serial' and type(serial_job_count) is int and 1<=serial_job_count<=6,'Serial actualselected group jobcount outside1..6')
 return {'harness_generation':37,'native_diagnostic_max_new_by_request':list(native_budgets),'api_max_new_by_request':list(api_budgets),'actual_submission_budget':list(api_budgets if kind=='api' else [1]*serial_job_count if kind=='serial' and serial_job_count is not None else [] if kind=='serial' else native_budgets),'natural_completion_qualified':False,'native_64_claim':False,'serial_first_head_max_new':1,'actual_serial_group_job_count':serial_job_count}

def manifest_binding(plan):
 require(__debug__ and sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE','0') in ('','0'),'Strict frozen parser requires assertions enabled; optimized Python refused')
 source_observers_off(plan['env'])
 require(plan.get('schema')==4 and plan.get('harness_generation')==37,'FrozenV6/older harness plan refused')
 budget=output_budget_contract(plan['kind'],plan['slots'],plan['native_diagnostic_max_new_by_request'],plan['api_max_new_by_request'],plan.get('serial_group_job_count'));require(plan['output_budget_contract']==budget and plan['max_new_by_request']==budget['actual_submission_budget'] and plan['max_new']==32,'Explicit V37 native32/API64 plan budget differs')
 require(set(plan['dependency_sha256'])==set(DEPENDENCIES),'Complete immutable V37 dependency closure required')
 require(sha(plan['registry_binding']['path'])==plan['registry_binding']['sha256'],'Derived eval registry changed')
 require(plan['driver_sha256']==sha(Path(__file__)) and plan['source_plan_sha256']==lane_contract(plan['lane'])['sha256'],'Immutable arm source changed')
 for name,digest in plan['dependency_sha256'].items():require(sha(HERE/name)==digest,'Consumed harness dependency changed '+name)
 topology=topology_baselines(plan['topology_baselines']['onecard_root'],plan['topology_baselines']['pair_root'],plan['lane']);require(topology==plan['topology_baselines'],'Current genuineC137 one/pair sameSDK prerequisite changed')
 require(plan['case_source_binding']=={'source_plan_sha256':PLAN_SHA,'source_count':64,'header_count':28,'patch_count':37,'fresh_abi_count':8,'runtime_python_count':6},'Actual newcase source generation metadata differs')
 if plan['slots']>2:
  from audit_batch_numerical_suite_v37 import paired_two_control
  require(paired_two_control(plan['paired_two_control'],plan['engine_receipt_sha256'])==plan['paired_two_control_binding'],'Actual completed paired2 OFF/ON/full49 prerequisite changed')
 prepared,binding=genuine_baseline(Path(plan['prepared']),plan['lane']);require(binding['prepared_sha256']==plan['prepared_sha256'] and prepared['engine_receipt_sha256']==plan['engine_receipt_sha256'],'Actual matching corrected current-PLE32/C137 baseline changed')
 if plan['kind']=='serial':
  from audit_batch_numerical_suite_v37 import parent_arm,vectors
  root=Path(plan['batch_parent']);parent,collector_plan,_=parent_arm(root);require(collector_plan['diagnostic']==1 and collector_plan['kind'] in ('native','api') and collector_plan['slots']==plan['slots'] and collector_plan['engine_receipt_sha256']==plan['engine_receipt_sha256'],'Serial actual V37 collector/topology/source differs');require(vectors(root),'Serial requires recollected full raw native48/head source corpus');require(parent['passed'] and parent['plan_sha256']==plan['batch_parent_plan_sha256'] and sha(root/'parent-qualification.json')==plan['batch_parent_sha256'],'Actual source-qualified collector parent changed');require(sha(root/'child/serial-jobs.json')==plan['serial_jobs_sha256'],'Actual collected consumed-prefix jobs changed');jobs=read(root/'child/serial-jobs.json')['jobs'];require(plan['serial_group_job_count']==len(jobs[plan['group_index']*6:plan['group_index']*6+6]),'Serial actualsubmitted group jobcount changed')
 return binding

def prepare(a):
 topology=topology_baselines(a.one_card_baseline,a.pair_baseline,a.lane)
 prepared,binding=genuine_baseline(a.prepared,a.lane);require(prepared['engine_receipt_sha256']==topology['engine_receipt_sha256'],'Chosen preparation not sameSDK as finalizedone/pair');spec=read(a.spec);case_binding=case_source_gate(spec,Path(prepared['engine_receipt']).parent);require(spec['source_lane']==a.lane,'Case source lane differs');n=spec['slots'];require(type(n) is int and n in (2,4,6),'Requested2/4/6 scope required');cfg=read(a.prepared/'server-config.json');args=list(cfg['args'])
 for key,value in [('--max-context',2048),('--prefill',64),('--batch',n),('--batch-groups',1),('--prompt-cache',0),('--conversation-cache-mib',0),('--adapt-every',0)]:api.set_arg(args,key,value)
 env=dict(cfg['env']);env.update(STRATA_BATCH_FIDELITY_DIAG='1',STRATA_BATCH_FIDELITY_ARM='/results/ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',STRATA_SLOT_OWNER_TRACE='1',STRATA_BATCH_FULL_STATE_CHAIN='0',STRATA_BATCH_PUBLIC_PREFIX='0',SYCL_UR_TRACE='2')
 env.update(STRATA_PLE_INPUT33='0',STRATA_PREFIX30='0',STRATA_CRITICAL_PATH_TRACE='0');source_observers_off(env)
 require('STRATA_CKPT_REREAD' not in env,'Checkpoint reread presence invalidates exact -1 counter policy');env['STRATA_MIRROR_OWNER_TRACE']='1' if a.lane=='source37' else '0'
 for key in ['STRATA_FIDELITY_DIAG','STRATA_LAYER0_Q8_DIAG']:env[key]='0'
 tokens=spec['tokens'];messages=spec['messages'];require(len(tokens['warm'])==len(tokens['target'])==len(messages['warm'])==len(messages['target'])==n,'Exact logical native/API cohorts required')
 for ids in tokens['warm']+tokens['target']:require(1<=len(ids)<=2048 and all(type(t) is int and 0<=t<248320 for t in ids),'Bounded actual submitted token vectors required')
 require(all(x[0]!=y[0] for x in tokens['warm'] for y in tokens['target']),'Warm native complete prefixes must be unrelated to target')
 require(len({json.dumps(m,sort_keys=True) for m in messages['target']})==n,'Unique API target messages needed for actual call joins')
 require(spec.get('schema')==2 and spec.get('harness_generation')==37,'New explicit V37 transport-budget case required; frozenV6 refused')
 budget=output_budget_contract(a.kind,n,spec['native_diagnostic_max_new_by_request'],spec['api_max_new_by_request']);require(spec['max_new_by_request']==spec['api_max_new_by_request'],'Case generic max_new is explicitly the API budget only')
 require(0<=spec['cancel_index']<n,'Bounded real cancellation logicalindex required')
 for row in spec['actual_counter_policy']:
  require(0<=row['logical_index']<n and row['role'] in ('admission','solo_migration') and row['values']=={'reused':0,'read_from':0,'reread_to':-1},'Pilot exact source no-cache/no-checkpoint counters must0/0/-1; reuse/read_from guards unchanged')
 require({(row['logical_index'],row['role']) for row in spec['actual_counter_policy']}=={(i,'admission') for i in range(n)}|{(1,'solo_migration')} and len(spec['actual_counter_policy'])==n+1,'Exact complete admission/survivor counter roster required')
 require(len(spec['api_token_ids']['warm'])==len(spec['api_token_ids']['target'])==n and spec['api_token_ids']['target']==tokens['target'],'Actual API/native target input corpus identity differs')
 for name,digest in spec['source_sha256'].items():require(sha(Path(prepared['engine_receipt']).parent/'source'/name)==digest,'Case consumed source identity changed '+name)
 require(spec['tokenizer_sha256']==read(a.prepared/'artifact-identity.json')['tokenizer_files'],'Case tokenizer/template identity changed')
 pair2_binding=None
 if n>2:
  from audit_batch_numerical_suite_v37 import paired_two_control
  require('paired_two_control' in spec,'Actual completed paired2 OFF/ON/full49 control required before4/6')
  pair2_binding=paired_two_control(spec['paired_two_control'],prepared['engine_receipt_sha256'])
 identity=verify_model_identity(a.model_identity,read(HERE/'model-lock.json'),[Path(row['path']) for row in prepared['model_shards']])
 plan={'schema':4,'harness_generation':37,'topology_baselines':topology,'case_source_binding':case_binding,'native_diagnostic_max_new_by_request':spec['native_diagnostic_max_new_by_request'],'api_max_new_by_request':spec['api_max_new_by_request'],'output_budget_contract':budget,'lane':a.lane,'kind':a.kind,'diagnostic':a.diagnostic,'prepared':str(a.prepared.resolve()),'prepared_sha256':binding['prepared_sha256'],'baseline_source_proof':binding,'engine_root':str(Path(prepared['engine_receipt']).parent),'engine_receipt_sha256':prepared['engine_receipt_sha256'],'source_plan_sha256':lane_contract(a.lane)['sha256'],'driver_sha256':sha(Path(__file__)),'native_driver_sha256':sha(Path(native.__file__)),'dependency_sha256':{name:sha(HERE/name) for name in DEPENDENCIES},'cards':prepared['cards'],'args':args,'env':env,'image':prepared['runtime']['image'],'pack':prepared['pack'],'model_identity':identity,'slots':n,'stage_ranges':expected_stage_ranges(args),'api_token_ids':spec['api_token_ids'],'tokens':tokens,'messages':messages,'cancel_index':spec['cancel_index'],'max_new_by_request':budget['actual_submission_budget'],'actual_counter_policy':spec['actual_counter_policy'],'port':spec['port'],'spec_sha256':sha(a.spec),'max_new':32,'full_model_math_qualified':False,'public_cache_lane_required_separately':True,'serving_mirror_owner_proof':'Mandatory actual same-process mirror31 ownership under corrected current-PLE32'}
 if n>2:plan['paired_two_control']=spec['paired_two_control'];plan['paired_two_control_binding']=pair2_binding
 alias=api.experimental_alias({'args':args,'env':env},n,bool(a.diagnostic),a.lane);plan['registry_binding']=api.registry_gate(alias);plan['research_alias']=alias
 if a.kind=='serial':
  require(a.batch_parent and a.group_index is not None,'Actual finalized collector parent/group required');
  from audit_batch_numerical_suite_v37 import parent_arm,vectors
  parent,source,_=parent_arm(a.batch_parent);require(source['kind'] in ('native','api') and vectors(a.batch_parent),'Actual finalized V37 native/API collector raw corpus required');require(parent['passed'],'Collector source/lifecycle parent failed');source=read(a.batch_parent/'input-plan.snapshot.json');require(source.get('harness_generation')==37 and source['diagnostic']==1 and source['slots']==n and source['engine_receipt_sha256']==plan['engine_receipt_sha256'],'Actual diagnostic collector/source configuration differs')
  jobs=read(a.batch_parent/'child/serial-jobs.json')['jobs'];require(0<=a.group_index<(len(jobs)+5)//6,'Bounded actual serial group not present');plan.update(batch_parent=str(a.batch_parent.resolve()),batch_parent_sha256=sha(a.batch_parent/'parent-qualification.json'),batch_parent_plan_sha256=parent['plan_sha256'],serial_jobs_sha256=sha(a.batch_parent/'child/serial-jobs.json'),group_index=a.group_index,serial_group_job_count=len(jobs[a.group_index*6:a.group_index*6+6]));plan['output_budget_contract']=output_budget_contract('serial',n,spec['native_diagnostic_max_new_by_request'],spec['api_max_new_by_request'],plan['serial_group_job_count']);plan['max_new_by_request']=plan['output_budget_contract']['actual_submission_budget']
 manifest_binding(plan);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',plan)
 print('PREPARED single actual-process arm; no execution or mathematical qualification')

def run(a):
 c1.leased([0,1]);plan=read(a.plan);manifest_binding(plan)
 def stop(signum,frame):api.ABORT.set();raise KeyboardInterrupt('Parent requested owned arm stop')
 for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,stop)
 if plan['kind']=='native':return native.run(SimpleNamespace(plan=a.plan,output=a.output,pre_health=a.pre_health,diagnostic=plan['diagnostic']))
 if plan['kind']=='api':return 0 if api.run(plan,a.output,a.pre_health,bool(plan['diagnostic']))['collection_and_teardown_passed'] else 1
 result=serial.run(plan,Path(plan['batch_parent'])/'child',a.output,a.pre_health,plan['group_index']);write(a.output/'plan.snapshot.json',plan);result.update(collection_and_teardown_passed=result['passed'],finished_epoch=time.time(),full_model_math_qualified=False);result['artifact_bindings']=artifact_bindings(a.output);write(a.output/'report.json',result);return 0 if result['passed'] else 1

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True);a=sub.add_parser('prepare');a.add_argument('--lane',choices=['source37'],required=True);a.add_argument('--kind',choices=['native','api','serial'],required=True);a.add_argument('--diagnostic',type=int,choices=[0,1],required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--one-card-baseline',type=Path,required=True);a.add_argument('--pair-baseline',type=Path,required=True);a.add_argument('--spec',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--batch-parent',type=Path);a.add_argument('--group-index',type=int);a.add_argument('--output',type=Path,required=True);a=sub.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
