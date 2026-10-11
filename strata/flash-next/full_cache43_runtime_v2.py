#!/usr/bin/env python3
"""Genuine fresh43 mode1 two-slot shared-cache actor. No historical GPU transfer."""
import argparse,json,os,sys
from pathlib import Path
import c1_serve_controller_combined_v140_v3 as c1
import source43_sdk_binding_v1 as sdk
import full_cache43_baseline_v1 as baseline
import registry_source43_cache175_association_v1 as registry
from serial37_canonical_json_v3 import canonical
import full_cache_shared_runtime_v12 as software
from source43_cache_port_recipe_v1 import recipe as current_recipe
import source43_cache_watchdog_v1 as watchdog
ROOT=software.ROOT;HERE=software.HERE;sha=software.sha;read=software.read;write=software.write;require=software.require
SOURCE_PLAN=HERE/'full-cache43-two-slot-source-plan-v2.json';ENGINE_PLAN=sdk.PLAN;ENGINE_SHA=sdk.PLAN_SHA;ADMITTED_PLAN_BYTES=True
set_arg=software.set_arg
CASE=software.CASE;CASE_SHA=software.CASE_SHA;MANDATORY=software.MANDATORY

def source_binding():
 plan=read(SOURCE_PLAN);require('evals/configs/models.yaml'not in plan['files'],'Canonical registry is explicit current association')
 for n,w in plan['files'].items():require(sha(ROOT/n)==w,'Current43 cache source changed '+n)
 return {'source_plan_sha256':sha(SOURCE_PLAN),'files':plan['files'],'source_generation':43,'source40_GPU_proof_transferred':False}

def authentic_case():return software.authentic_case()
def expected_stage_ranges(args):return software.expected_stage_ranges(args)
def lane_contract(lane):require(lane=='source43','Actual source43 lane only');return {'plan':ENGINE_PLAN,'sha256':ENGINE_SHA}
def verify_model_identity(path,lock,shards):
 value=software.verify_model_identity(path,lock,shards);watchdog.guard(shards[2]);return value

def recipe(prepared,selected_rids):return current_recipe(read(Path(prepared)/'server-config.json'),2,selected_rids)
def baseline_binding(prepared):return baseline.finalized_binding(prepared)
def prepare(a):
 unoptimized();source=source_binding();require(type(a.port)is int and 1024<=a.port<=65535 and not a.output.exists(),'Fresh bounded cache43 actor plan required');m,proof=baseline_binding(a.prepared);actual_sdk=sdk.binding();require(canonical(m['source43_sdk'])==canonical(actual_sdk),'Actual70/32/43/eight SDK required');case=authentic_case();scenario=software.scenario_schedule(case,a.scenario,a.capture_pass);args,env=recipe(a.prepared,scenario['selected_capture_pass']['selected_actual_RIDs']);identity=read(a.model_identity);require(identity['passed']is True and len(identity['rows'])==4,'Actual complete source identity required');reg=registry.association(ROOT/'evals/configs/models.yaml');plan={'schema':'full-cache-shared-runtime-v3','driver_sha256':sha(__file__),'kind':'api','lane':'source43','diagnostic':1,'slots':2,'source_generation':43,'materialized_mode':1,'actual_source43_sdk':actual_sdk,'prepared':str(a.prepared.resolve()),'prepared_sha256':sha(a.prepared/'prepared.json'),'engine_receipt_sha256':m['engine_receipt_sha256'],'baseline_binding':proof,'source_binding':source,'authentic_case_sha256':CASE_SHA,'scenario':a.scenario,'capture_pass':a.capture_pass,'scenario_binding':scenario,'schedule':scenario['phases'],'mandatory_requirements':list(MANDATORY),'args':args,'env':env,'stage_ranges':expected_stage_ranges(args),'cards':m['cards'],'image':m['runtime']['image'],'pack':m['pack'],'port':a.port,'research_alias':registry.ALIAS,'registry_binding':{'path':str(ROOT/'evals/configs/models.yaml'),'sha256':reg['current_registry_sha256'],'served_model_id':registry.ALIAS},'registry_association':reg,'model_identity':{'path':str(a.model_identity.resolve()),'sha256':sha(a.model_identity)},'capture_disk_quota_bytes':512<<20,'capture_disk_quota_scope':'captures_array_files_only','total_text_budget_bytes':32<<30,'total_text_budget_scope':'trace_and_combined_plus_bounded_overflow_failure_record','observer_USM_budget_increased':False,'full_cache_runtime_qualified':False,'physical_residency_qualified':False};manifest_binding(plan);a.output.mkdir(parents=True);write(a.output/'plan.json',plan);return plan

def unoptimized():
 require(__debug__ and not sys.flags.optimize and os.environ.get('PYTHONOPTIMIZE','0')in('','0'),'Unoptimized strict assertions required')

def budget_gate(plan):
 require(type(plan['total_text_budget_bytes'])is int and plan['total_text_budget_bytes']==32<<30 and plan['total_text_budget_scope']=='trace_and_combined_plus_bounded_overflow_failure_record','Exact declared text budget/scope required');require(type(plan['capture_disk_quota_bytes'])is int and plan['capture_disk_quota_bytes']==512<<20 and plan['capture_disk_quota_scope']=='captures_array_files_only','Exact declared capture array quota/scope required')

def manifest_binding(plan):
 unoptimized();budget_gate(plan);source=source_binding();require(plan['schema']=='full-cache-shared-runtime-v3'and plan['driver_sha256']==sha(__file__)and canonical(plan['source_binding'])==canonical(source)and plan['lane']=='source43'and plan['kind']=='api'and type(plan['slots'])is int and plan['slots']==2 and type(plan['diagnostic'])is int and plan['diagnostic']==1 and type(plan['source_generation'])is int and plan['source_generation']==43 and type(plan['materialized_mode'])is int and plan['materialized_mode']==1,'Exact actual43 two-slot mode1 cache actor required');m,proof=baseline_binding(Path(plan['prepared']));current=sdk.binding();require(canonical(current)==canonical(plan['actual_source43_sdk'])==canonical(m['source43_sdk'])and plan['prepared_sha256']==sha(Path(plan['prepared'])/'prepared.json')and plan['engine_receipt_sha256']==m['engine_receipt_sha256']and canonical(proof)==canonical(plan['baseline_binding']),'Actual43 baseline/source/SDK association changed');case=authentic_case();scenario=software.scenario_schedule(case,plan['scenario'],plan['capture_pass']);args,env=recipe(plan['prepared'],scenario['selected_capture_pass']['selected_actual_RIDs']);require(plan['authentic_case_sha256']==CASE_SHA and canonical(plan['scenario_binding'])==canonical(scenario)and canonical(plan['schedule'])==canonical(scenario['phases'])and plan['mandatory_requirements']==list(MANDATORY)and plan['args']==args and plan['env']==env and plan['stage_ranges']==expected_stage_ranges(args),'Whole original phase/input/current recipe changed');reg=registry.association(ROOT/'evals/configs/models.yaml');require(plan['registry_association']==reg and plan['research_alias']==registry.ALIAS and plan['registry_binding']=={'path':str(ROOT/'evals/configs/models.yaml'),'sha256':reg['current_registry_sha256'],'served_model_id':registry.ALIAS}and plan['cards']==m['cards']and plan['image']==m['runtime']['image']and plan['pack']==m['pack'],'Current actual175 alias/topology/image/pack changed');lock=read(HERE/'model-lock.json');shards=[ROOT/lock['destination']/r['path']for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];require(sha(plan['model_identity']['path'])==plan['model_identity']['sha256'],'Current identity intake receipt changed');verify_model_identity(Path(plan['model_identity']['path']),lock,shards);return {'prepared_sha256':plan['prepared_sha256'],'engine_receipt_sha256':plan['engine_receipt_sha256'],'current_baseline':proof,'source':source,'genuine_current_source43':True,'full_cache_runtime_qualified':False}

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True);a=sub.add_parser('prepare');a.add_argument('--scenario',choices=['shared','independent','cancellation','eviction','history','stale'],required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--capture-pass',type=int,default=0);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--port',type=int,default=28743);a.add_argument('--output',type=Path,required=True);a=sub.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--expected-plan-sha256',required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.mode=='prepare':prepare(a);return 0
 from full_cache43_plan_snapshot_v2 import Snapshot
 from full_cache43_health_handoff_v2 import child_wait,post_ack_seal
 import run_full_cache43_runtime_v2 as api
 snapshot=Snapshot(a.plan,a.expected_plan_sha256);plan=snapshot.plan;manifest=manifest_binding(plan);snapshot.verify();c1.leased([0,1]);handoff=child_wait(plan,manifest,a.output);seal=post_ack_seal(plan,sys.modules[__name__],handoff);snapshot.verify();return int(not api.run(plan,a.output,a.output.parent/'leaf-health.json',snapshot.raw,handoff,seal)['collection_and_teardown_passed'])
if __name__=='__main__':raise SystemExit(main())
