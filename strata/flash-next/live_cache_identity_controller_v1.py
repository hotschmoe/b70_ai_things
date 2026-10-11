"""New root-only live identity lane on immutable full shared source40 data."""
import argparse,copy,json,sys
from pathlib import Path
import full_cache_shared_runtime_v8 as base
from live_cache_identity_namespace_v1 import digest,namespace
from live_cache_identity_experiment_v1 import schedule
ROOT,HERE=base.ROOT,base.HERE
read,write,sha,require,c1=base.read,base.write,base.sha,base.require,base.c1
ENGINE_PLAN,ENGINE_SHA=base.ENGINE_PLAN,base.ENGINE_SHA
ADMITTED_PLAN_BYTES=True
SOURCE_PLAN=HERE/'live-cache-identity-refusal-source-plan-v1.json'
expected_stage_ranges=base.expected_stage_ranges
verify_model_identity=base.verify_model_identity

def source_binding():
 plan=read(SOURCE_PLAN)
 for name,value in plan['files'].items():require(sha(ROOT/name)==value,'New live identity source changed '+name)
 return {'source_plan_sha256':sha(SOURCE_PLAN),'files':plan['files']}
def owner_namespace(plan):
 prepared=read(Path(plan['prepared'])/'prepared.json');artifact=read(Path(plan['prepared'])/'artifact-identity.json');lock=read(HERE/'model-lock.json');tokenizer=artifact['tokenizer_files']
 return namespace({'model_sha256':digest({'revision':lock['revision'],'publisher4':[(r['path'],r['sha256'],r['size'])for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')],'native_pack_receipt_sha256':prepared['pack_receipt_sha256']}),'tokenizer_sha256':digest({k:v for k,v in tokenizer.items()if k!='chat_template.jinja'}),'template_sha256':tokenizer['chat_template.jinja'],'source_sha256':digest({'engine_receipt_sha256':plan['engine_receipt_sha256'],'base_source':plan['source_binding'],'new_guard_files':{name:sha(HERE/name)for name in ('live_cache_identity_namespace_v1.py','live_cache_loaded_owner_v1.py','full_cache_live_identity_api_v1.py')}})})
def original_plan(plan):
 return copy.deepcopy(plan['immutable_original_source_plan'])
def family_schedule(original):
 phases=copy.deepcopy(original['schedule'][:3]);repeat=copy.deepcopy(phases[2]);repeat['name']='repeat_original';phases.append(repeat)
 from full_cache_shared_capture_roster_v2 import actor_passes
 roster=actor_passes(phases);require(len(roster['passes'])==1,'Bounded whole live identity actor capture required');selected=roster['passes'][0];cursor=0
 for row in phases:
  row['expected_actual_RID_set']=list(range(cursor+1,cursor+1+len(row['rows'])));cursor+=len(row['rows']);row['raw_capture_requested']=row['name']in selected['selected_phases']
 require(cursor==5 and selected['selected_actual_RIDs']==[3,4,5],'Actual warm2/prime0/prime1/repeat3 full49 owner roster required')
 return phases,roster,selected
def manifest_binding(plan):
 require(plan['driver_sha256']==sha(Path(__file__)) and plan['live_identity_source']==source_binding(),'Exact new namespace controller/source required');original=original_plan(plan);proof=base.manifest_binding(original);require(plan['live_identity_namespace']==owner_namespace(original) and plan['live_identity_experiment']==schedule(plan['live_identity_namespace']),'Original loaded identity namespace/preregistered refusal stages changed');phases,roster,selected=family_schedule(original);require(plan['schedule']==phases and plan['scenario_binding']['capture_roster']==roster and plan['scenario_binding']['selected_capture_pass']==selected,'Exact whole dedicated identity-refusal family required');env=dict(original['env']);env['STRATA_FULL_CACHE_CAPTURE_RIDS38']='3,4,5';require(plan['env']==env and plan['args']==original['args'],'No model/math route change in identity test')
 return {'current_base_source40':proof,'new_live_identity_source':source_binding(),'explicit_namespace_and_loaded_owner_guard':True,'native_model_fingerprint_cache_key_qualified':False,'actual_different_weights_loaded':False,'full_cache_runtime_qualified':False}
def prepare(args):
 source_binding();require(not args.output.exists(),'Fresh live identity preparation required');args.output.mkdir();base.prepare(argparse.Namespace(scenario='shared',capture_pass=0,prepared=args.prepared,model_identity=args.model_identity,port=args.port,output=args.output/'original-input-plan'));plan=read(args.output/'original-input-plan/plan.json');owner=owner_namespace(plan);original=copy.deepcopy(plan);phases,roster,selected=family_schedule(original);plan.update(driver_sha256=sha(Path(__file__)),live_identity_source=source_binding(),live_identity_namespace=owner,live_identity_experiment=schedule(owner),immutable_original_source_plan=original,schedule=phases);plan['scenario_binding'].update(phases=phases,capture_roster=roster,selected_capture_pass=selected);plan['env']['STRATA_FULL_CACHE_CAPTURE_RIDS38']='3,4,5';manifest_binding(plan);write(args.output/'plan.json',plan)
def main():
 parser=argparse.ArgumentParser();subs=parser.add_subparsers(dest='mode',required=True);p=subs.add_parser('prepare');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=28900);p=subs.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pre-health',type=Path,required=True);p.add_argument('--expected-plan-sha256',required=True);args=parser.parse_args()
 if args.mode=='prepare':prepare(args);return 0
 from full_cache_shared_plan_snapshot_v8 import Snapshot
 snapshot=Snapshot(args.plan,args.expected_plan_sha256);plan=snapshot.plan;manifest=manifest_binding(plan);snapshot.verify();c1.leased([0,1])
 from full_cache_shared_health_handoff_v8 import child_wait,post_ack_seal
 handoff=child_wait(plan,manifest,args.output);seal=post_ack_seal(plan,sys.modules[__name__],handoff);snapshot.verify()
 import run_live_cache_identity_actor_v1 as actor
 return int(not actor.run(plan,args.output,args.output.parent/'leaf-health.json',snapshot.raw,handoff,seal)['collection_and_teardown_passed'])
def __getattr__(name):
 return getattr(base,name)
if __name__=='__main__':raise SystemExit(main())
