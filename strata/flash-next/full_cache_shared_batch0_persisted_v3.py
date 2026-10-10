"""NEW actual source40V2 batch0 persisted purpose; independent proofs mandatory."""
import argparse,copy,os,sys
from pathlib import Path
import full_cache_shared_runtime_v3 as shared
from serial37_canonical_json_v3 import canonical
ROOT=shared.ROOT;HERE=shared.HERE;c1=shared.c1;read=shared.read;write=shared.write;sha=shared.sha;require=shared.require
ADMITTED_PLAN_BYTES=True;ENGINE_PLAN=shared.ENGINE_PLAN;ENGINE_SHA=shared.ENGINE_SHA
ALIAS='qwen38-flash-next-unsloth-udq4-k-xl-strata-source40-batch0-persisted-v3'
source_binding=shared.source_binding;lane_contract=shared.lane_contract;expected_stage_ranges=shared.expected_stage_ranges;verify_model_identity=shared.verify_model_identity

def recipe(prepared):
 args,env=shared.recipe(prepared,[1]);shared.set_arg(args,'--batch',0)
 env.update(STRATA_FULL_CACHE_OBSERVER38='0',STRATA_BATCH_FIDELITY_DIAG='0',STRATA_BATCH_FULL_STATE_CHAIN='0',STRATA_BATCH_PUBLIC_PREFIX='0',STRATA_FIDELITY_DIAG='1',STRATA_FIDELITY_DIAG_ACTIVATIONS='1',STRATA_PREFIX_DIAG='1',STRATA_PREFIX_LIFECYCLE_DIAG='1',STRATA_FIDELITY_DIAG_ARM='/results/ARM',STRATA_FIDELITY_DIAG_DIR='/results/captures',STRATA_PREFIX_DIAG_ARM='/results/ARM')
 for key in ('STRATA_FULL_CACHE_CAPTURE_RIDS38','STRATA_BATCH_FIDELITY_ARM','STRATA_BATCH_FIDELITY_DIR'):env.pop(key,None)
 from full_cache_shared_batch0_api_trace_v3 import cfg_gate
 cfg_gate({'args':args,'env':env,'parallel':1,'slot_save_path':'/results/sessions'});return args,env

def schedule(case):
 row=copy.deepcopy(case['phases']['prime_shared']['rows'][0]);n=len(row['ids']);require(272<n<=2048,'Authentic independent saved-prefix input required')
 return {'prime':{'row':row,'fresh':1,'pin':None,'expected_reused':0,'max_new':1},'save':{'action':'save','filename':'valid.bin','expected_saved_tokens':n},'wrong_restore':{'action':'restore_wrong_model','filename':'wrong.bin','expected_saved_tokens':n},'after_wrong':{'row':row,'fresh':0,'pin':None,'expected_reused':n-1,'max_new':1},'valid_restore':{'action':'restore_valid','filename':'valid.bin','expected_saved_tokens':n},'after_valid':{'row':row,'fresh':0,'pin':None,'expected_reused':n-1,'max_new':1},'preregistered_phase_order':['prime','save','wrong_restore','after_wrong','valid_restore','after_valid'],'restored_or_other_actor_state_used_by_fresh_control':False,'all3_full49_and_independent_fresh49_required':True,'full_cache_runtime_qualified':False}

def manifest_binding(plan):
 source=source_binding();require(__debug__ and not sys.flags.optimize and os.environ.get('PYTHONOPTIMIZE','0') in ('','0'),'Unoptimized public gates required')
 require(plan['schema']=='source40-batch0-persisted-v3' and plan['kind']=='api' and plan['lane']=='source40' and plan['slots']==0 and plan['diagnostic']==1 and plan['driver_sha256']==sha(__file__) and plan['source_binding']==source,'Explicit new serial persisted controller required')
 prepared,proof=shared.baseline.finalized_binding(Path(plan['prepared']));require(prepared['combined_generation']['plan_sha256']==ENGINE_SHA and prepared['combined_generation']['consumer_generation']==1403 and prepared['combined_generation']['native_QSA_target_predicate_namespace_fix40_v2'] is True and prepared['engine_receipt_sha256']==plan['engine_receipt_sha256'] and sha(Path(plan['prepared'])/'prepared.json')==plan['prepared_sha256'] and canonical(proof)==canonical(plan['baseline_binding']),'Genuine corrected source40/current1403 baseline required')
 args,env=recipe(plan['prepared']);require(plan['args']==args and plan['env']==env and plan['cards']==prepared['cards'] and plan['image']==prepared['runtime']['image'] and plan['pack']==prepared['pack'] and canonical(plan['stage_ranges'])==canonical(expected_stage_ranges(args)),'Actual original source40 serial recipe/topology differs')
 require(plan['authentic_case_sha256']==shared.CASE_SHA and plan['schedule']==schedule(shared.authentic_case()) and plan['research_alias']==ALIAS,'Actual input-only authentic fixture and preregistered entire persisted schedule differ')
 from registry_c140_shared_association_v3 import association
 require(plan['registry_association']==association(ROOT/'evals/configs/models.yaml',True),'Actual exact source40 registry association differs')
 return {'source':source,'current_baseline':proof,'prepared_sha256':plan['prepared_sha256'],'engine_receipt_sha256':plan['engine_receipt_sha256'],'source40_serial_PCL_scope':True,'full_cache_runtime_qualified':False}

def prepare(args):
 require(not args.output.exists() and type(args.port)is int and 1024<=args.port<=65535,'New bounded original purpose plan required');source=source_binding();case=shared.authentic_case();prepared,proof=shared.baseline.finalized_binding(args.prepared);actual_args,env=recipe(args.prepared)
 from registry_c140_shared_association_v3 import association
 plan={'schema':'source40-batch0-persisted-v3','kind':'api','lane':'source40','slots':0,'diagnostic':1,'driver_sha256':sha(__file__),'source_binding':source,'prepared':str(args.prepared.resolve()),'prepared_sha256':sha(args.prepared/'prepared.json'),'engine_receipt_sha256':prepared['engine_receipt_sha256'],'baseline_binding':proof,'args':actual_args,'env':env,'cards':prepared['cards'],'image':prepared['runtime']['image'],'pack':prepared['pack'],'stage_ranges':expected_stage_ranges(actual_args),'model_identity':{'path':str(args.model_identity.resolve()),'sha256':sha(args.model_identity)},'port':args.port,'research_alias':ALIAS,'authentic_case_sha256':shared.CASE_SHA,'schedule':schedule(case),'registry_association':association(ROOT/'evals/configs/models.yaml',True),'actual_current1403_baseline_required':True,'full_cache_runtime_qualified':False}
 manifest_binding(plan);write(args.output,plan)

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--port',type=int,default=28770);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--expected-plan-sha256',required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.mode=='prepare':prepare(a);return 0
 from full_cache_shared_plan_snapshot_v3 import Snapshot
 from full_cache_shared_health_handoff_v3 import child_wait,post_ack_seal
 from run_full_cache_shared_batch0_persisted_v3 import run
 snapshot=Snapshot(a.plan,a.expected_plan_sha256);plan=snapshot.plan;manifest=manifest_binding(plan);snapshot.verify();c1.leased([0,1]);handoff=child_wait(plan,manifest,a.output);seal=post_ack_seal(plan,sys.modules[__name__],handoff);snapshot.verify();return int(not run(plan,a.output,a.output.parent/'leaf-health.json',snapshot.raw,handoff,seal)['collection_and_teardown_passed'])
if __name__=='__main__':raise SystemExit(main())
