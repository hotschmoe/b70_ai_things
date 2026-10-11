"""NEW source40 full-cache phase controller; all missing gates remain explicit."""
import argparse,copy,importlib,json,os,sys
from pathlib import Path
import c1_serve_controller_combined_v140_v3 as c1
import full_cache_shared_baseline_admission_v8 as baseline
from serial37_canonical_json_v3 import canonical
import full_cache_shared_admission_v2 as fixture
from full_cache_shared_phase_contract_v2 import schedules,scenario_schedule,MANDATORY
import run_full_cache_shared_runtime_v8 as api
ROOT=fixture.ROOT;HERE=fixture.HERE;sha=fixture.sha;read=fixture.read;require=fixture.require
ADMITTED_PLAN_BYTES=True
SOURCE_PLAN=HERE/'full-cache-shared-runtime-source-plan-v8.json'
ENGINE_PLAN=HERE/'native-layer3-qsa-observer-engine-build-plan-v2.json'
ENGINE_SHA='d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6'
CASE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/full-cache-shared-authentic-case-v2.json')
FIXTURE=CASE.parent/'full-cache-shared-tokenizer-fixture-v2'
CASE_SHA='02c1a0299f4a2eaca10b8f017f154dec2df77bdde802ec93e2ec8d6857e93eaf'
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
def source_binding():
 plan=read(SOURCE_PLAN)
 require('evals/configs/models.yaml' not in plan['files'],'New source ledger cannot treat mutable canonical registry as fixed source bytes')
 for n,w in plan['files'].items():require(sha(ROOT/n)==w,'Full shared runtime source changed '+n)
 from full_cache_shared_registry_source_v4 import inherited_source
 inherited=inherited_source();require(sha(ENGINE_PLAN)==ENGINE_SHA,'Actual source40 engine plan changed');return {'source_plan_sha256':sha(SOURCE_PLAN),'files':plan['files'],'explicit_registry_source_admission':inherited}
def authentic_case():
 data,binding=fixture.fixture_admission(FIXTURE);case=read(CASE);require(sha(CASE)==CASE_SHA and case['authentic_fixture_binding']==binding and case['shared_boundary']==fixture.shared_boundary(data['fixtures']),'Exact genuine shared fixture/case association differs')
 for phase,rows in data['fixtures'].items():require(case['phases'][phase]['rows']==rows,'Actual phase fixture bytes differ')
 return case
def option(args,k):require(args.count(k)==1,'Exactly one actual option required '+k);return args[args.index(k)+1]
def set_arg(args,k,v):
 if k in args:require(args.count(k)==1,'Duplicate inherited option '+k);args[args.index(k)+1]=str(v)
 else:args.extend([k,str(v)])
def recipe(prepared,selected_rids):
 cfg=read(Path(prepared)/'server-config.json');args=list(cfg['args']);env=dict(cfg['env'])
 require('STRATA_QSA3_TARGET_DIR' not in env and 'STRATA_QSA3_TARGET_BINDING_SHA256' not in env,'No inherited QSA40 target observation controls')
 require('STRATA_VERIFY_EAGER' not in env and 'STRATA_CKPT_REREAD' not in env and '--pipeline-windows' not in args and '--mtp' not in args,'Unqualified full-cache math/graph route presence')
 for key in ('STRATA_KV_GROW','STRATA_PREFILL_EQUAL','STRATA_SPLIT_SMALL_OWN','STRATA_SPLIT_SMALL_MAX','STRATA_CACHE_MESSAGE_BOUNDARY'):require(key not in env,'Undeclared cache/shape override presence '+key)
 for k,v in [('--max-context',2048),('--prefill',64),('--batch',2),('--batch-groups',1),('--prompt-cache',3),('--conversation-cache-mib',512),('--conversation-cache-slots',2),('--adapt-every',0),('--lookup-chain',0),('--suffix-draft',0),('--prompt-cache-root',2048),('--prompt-cache-every',16384),('--turn-token',248045),('--tail-role-token',-1)]:set_arg(args,k,v)
 env.update(STRATA_QSA3_TARGET='0',STRATA_CRITICAL_PATH_TRACE='0',STRATA_FULL_CACHE_OBSERVER38='1',STRATA_FULL_CACHE_CAPTURE_RIDS38=','.join(map(str,selected_rids)),STRATA_PLE_INPUT33='0',STRATA_PREFIX30='0',STRATA_LAYER0_Q8_DIAG='0',STRATA_FIDELITY_DIAG='0',STRATA_PREFIX_DIAG='0',STRATA_PREFIX_LIFECYCLE_DIAG='0',STRATA_BATCH_FULL_STATE_CHAIN='1',STRATA_BATCH_CHAIN_MIB='8192',STRATA_BATCH_TRANSFER_MIB='512',STRATA_BATCH_PUBLIC_PREFIX='1',STRATA_BATCH_FIDELITY_DIAG='1',STRATA_BATCH_FIDELITY_ARM='/results/ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',STRATA_SLOT_OWNER_TRACE='1',STRATA_MIRROR_OWNER_TRACE='1',SYCL_UR_TRACE='2')
 env.pop('STRATA_PREFIX_DIAG_ARM',None);env.pop('STRATA_PREFIX_LIFECYCLE_DIAG_ARM',None)
 return args,env
def expected_stage_ranges(args):
 from batch_numerical_execution_v40 import expected_stage_ranges as fn
 return fn(args)
def lane_contract(lane):require(lane=='source40','Current source40 only');return {'sha256':ENGINE_SHA,'plan':ENGINE_PLAN}
def verify_model_identity(*a,**k):
 from batch_numerical_execution_v40 import verify_model_identity as fn
 return fn(*a,**k)
def manifest_binding(plan):
 source=source_binding();require(__debug__ and not sys.flags.optimize and os.environ.get('PYTHONOPTIMIZE','0') in ('','0'),'Unoptimized strict assertions required')
 require(plan['schema']=='full-cache-shared-runtime-v3' and plan['driver_sha256']==sha(Path(__file__)) and plan['source_binding']==source and plan['kind']=='api' and plan['lane']=='source40' and type(plan['diagnostic'])is int and plan['diagnostic']==1 and plan['slots']==2,'Explicit full shared source40 producer required')
 prepared,proof=baseline.finalized_binding(Path(plan['prepared']));require(canonical(proof)==canonical(plan['baseline_binding']) and sha(Path(plan['prepared'])/'prepared.json')==plan['prepared_sha256'] and prepared['engine_receipt_sha256']==plan['engine_receipt_sha256'],'Actual independently verified current1403 baseline association changed')
 require(prepared['combined_generation']['consumer_generation']==1403 and prepared['combined_generation']['native_QSA_target_predicate_namespace_fix40_v2'] is True and prepared['combined_generation']['source_count']==67 and len(prepared['combined_generation']['header_payloads'])==31 and prepared['combined_generation']['default_off_native_layer3_QSA_targets40'] is True and prepared['combined_generation']['actual_native_layer3_QSA_targets_qualified'] is False and prepared['combined_generation']['plan_sha256']==ENGINE_SHA and prepared['combined_generation']['earlier_stage_batch_observer_stamp37'] is True and prepared['combined_generation']['default_off_full_cache_observer38'] is True and prepared['combined_generation']['default_off_logical_cache_memory39'] is True,'Actual source40 stamp binding absent')
 case=authentic_case();require(plan['authentic_case_sha256']==CASE_SHA and plan['scenario_binding']==scenario_schedule(case,plan['scenario'],plan['capture_pass']) and plan['schedule']==plan['scenario_binding']['phases'] and plan['mandatory_requirements']==list(MANDATORY),'Whole explicit shared phase/requirement contract changed')
 args,env=recipe(plan['prepared'],plan['scenario_binding']['selected_capture_pass']['selected_actual_RIDs']);require(plan['args']==args and plan['env']==env and canonical(plan['stage_ranges'])==canonical(expected_stage_ranges(args)) and plan['cards']==prepared['cards'] and plan['image']==prepared['runtime']['image'] and plan['pack']==prepared['pack'],'Exact actual full shared derived recipe/SDK image/placement differs')
 require(type(plan['total_text_budget_bytes'])is int and plan['total_text_budget_bytes']==32<<30 and plan['capture_disk_quota_scope']=='captures_array_files_only','Exact declared text/capture budgets required')
 require(plan['research_alias']=='qwen38-flash-next-unsloth-udq4-k-xl-strata-source40-full-cache-shared272-v3','Detailed primary/research identity required');registry=ROOT/'evals/configs/models.yaml';require(plan['registry_binding']=={'path':str(registry),'sha256':sha(registry),'served_model_id':plan['research_alias']},'Actual current registry association changed')
 from registry_c140_shared_association_v3 import association
 require(plan['registry_association']==association(registry,True),'Exact deliberate source40 registry association changed; no original global gate override')
 return {'prepared_sha256':plan['prepared_sha256'],'engine_receipt_sha256':plan['engine_receipt_sha256'],'current_baseline':proof,'source':source,'genuine_current_source40':True,'full_cache_runtime_qualified':False}
def prepare(a):
 require(type(a.port)is int and 1024<=a.port<=65535,'Bounded exact integer TCP port required');require(not a.output.exists(),'Fresh plan output required');source=source_binding();case=authentic_case();scenario=scenario_schedule(case,a.scenario,a.capture_pass);schedule=scenario['phases'];prepared,proof=baseline.finalized_binding(a.prepared);args,env=recipe(a.prepared,scenario['selected_capture_pass']['selected_actual_RIDs']);identity=read(a.model_identity);require(identity['passed'] is True and len(identity['rows'])==4,'Current all4 source identity required');alias='qwen38-flash-next-unsloth-udq4-k-xl-strata-source40-full-cache-shared272-v3';registry=ROOT/'evals/configs/models.yaml'
 from registry_c140_shared_association_v3 import association
 registry_association=association(registry,True)
 plan={'registry_association':registry_association,'schema':'full-cache-shared-runtime-v3','driver_sha256':sha(Path(__file__)),'kind':'api','lane':'source40','diagnostic':1,'slots':2,'prepared':str(a.prepared.resolve()),'prepared_sha256':sha(a.prepared/'prepared.json'),'engine_receipt_sha256':prepared['engine_receipt_sha256'],'actual_baseline_parent_generation_required':1403,'baseline_binding':proof,'source_binding':source,'authentic_case_sha256':CASE_SHA,'scenario':a.scenario,'capture_pass':a.capture_pass,'scenario_binding':scenario,'schedule':schedule,'mandatory_requirements':list(MANDATORY),'args':args,'env':env,'stage_ranges':expected_stage_ranges(args),'cards':prepared['cards'],'image':prepared['runtime']['image'],'pack':prepared['pack'],'port':a.port,'research_alias':alias,'registry_binding':{'path':str(registry),'sha256':sha(registry),'served_model_id':alias},'model_identity':{'path':str(a.model_identity.resolve()),'sha256':sha(a.model_identity)},'full_cache_runtime_qualified':False,'checkpoint_victim_observer_source_available':True,'actual_checkpoint_victim_observed':False,'new_runtime_prime_policy':'prime0 authentic sole row cold0; prime1 repeat same authenticated input and require shared272 readback; declared before execution','capture_disk_quota_bytes':512<<20,'capture_disk_quota_scope':'captures_array_files_only','total_text_budget_bytes':32<<30,'total_text_budget_scope':'trace_and_combined_plus_bounded_overflow_failure_record','observer_USM_budget_increased':False}
 manifest_binding(plan);a.output.mkdir(parents=True);write(a.output/'plan.json',plan)
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--scenario',choices=['shared','independent','cancellation','eviction','history','stale'],required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--capture-pass',type=int,default=0);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--port',type=int,default=28740);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--expected-plan-sha256',required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.mode=='prepare':prepare(a);return 0
 from full_cache_shared_plan_snapshot_v8 import Snapshot
 snapshot=Snapshot(a.plan,a.expected_plan_sha256);plan=snapshot.plan;manifest=manifest_binding(plan);snapshot.verify();c1.leased([0,1]);from full_cache_shared_health_handoff_v8 import child_wait,post_ack_seal
 handoff=child_wait(plan,manifest,a.output);seal=post_ack_seal(plan,sys.modules[__name__],handoff);snapshot.verify();return int(not api.run(plan,a.output,a.output.parent/'leaf-health.json',snapshot.raw,handoff,seal)['collection_and_teardown_passed'])
if __name__=='__main__':raise SystemExit(main())
