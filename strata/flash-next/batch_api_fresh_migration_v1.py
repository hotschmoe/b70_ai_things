#!/usr/bin/env python3
"""NEW API capability-on fresh recompute adapter, frozen V7 lifecycle unchanged."""
import argparse,copy,signal,sys,os
from pathlib import Path
from types import SimpleNamespace
import batch_numerical_execution_v7 as base
import run_batch_api_fresh_migration_v1 as api
import api_fresh_migration_contract_v1 as contract
ROOT=base.ROOT;HERE=base.HERE;c1=base.c1;sha=base.sha;read=base.read;write=base.write;require=base.require
lane_contract=base.lane_contract;expected_stage_ranges=base.expected_stage_ranges;verify_model_identity=base.verify_model_identity
SOURCE_PLAN=contract.SOURCE_PLAN
FROZEN_MANIFEST=base.manifest_binding

def recipe(prepared,slots,diagnostic):
 cfg=read(Path(prepared)/'server-config.json');args=list(cfg['args']);env=dict(cfg['env'])
 for key,val in [('--max-context',2048),('--prefill',64),('--batch',slots),('--batch-groups',1),('--prompt-cache',3),('--conversation-cache-mib',0),('--adapt-every',0)]:api.set_arg(args,key,val)
 env.update(contract.PROFILE_ENV);env.update(STRATA_PLE_INPUT33='0',STRATA_PREFIX30='0',STRATA_MIRROR_OWNER_TRACE='1',STRATA_FIDELITY_DIAG='0',STRATA_LAYER0_Q8_DIAG='0',STRATA_SLOT_OWNER_TRACE='1',STRATA_BATCH_FIDELITY_DIAG=str(int(diagnostic)),STRATA_BATCH_FIDELITY_ARM='/results/ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',SYCL_UR_TRACE='2')
 return args,env

def frozen_admission_view(plan):
 view=copy.deepcopy(plan);view.update(schema=4,harness_generation=7,driver_sha256=sha(Path(base.__file__)))
 return view

def manifest_binding(plan):
 require(__debug__ and sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE','0') in ('','0'),'Frozen parser assertion gates required')
 require(plan.get('schema')==5 and plan.get('harness_generation')==8 and plan.get('api_fresh_recompute_generation')==1 and plan['kind']=='api' and plan['driver_sha256']==sha(Path(__file__)),'Explicit NEW API fresh recompute producer required')
 require(plan['API_source_plan_sha256']==sha(SOURCE_PLAN) and plan['API_driver_sha256']==sha(Path(api.__file__)),'Exact new API source closure/runner identity differs')
 for name,want in read(SOURCE_PLAN)['files'].items():require(sha(ROOT/name)==want,'Immutable API recompute source changed '+name)
 args,env=recipe(plan['prepared'],plan['slots'],bool(plan['diagnostic']));require(plan['args']==args and plan['env']==env,'Exact current prepared API recipe differs')
 prepared=read(Path(plan['prepared'])/'prepared.json');require(plan['image']==prepared['runtime']['image'] and plan['pack']==prepared['pack'],'Exact API image/pack differs')
 scope=contract.preflight(args,env,plan['API_request_policy'],plan['actual_counter_policy'],plan['slots']);require(plan['API_migration_scope']==scope and plan['research_alias']==api.experimental_alias({'args':args,'env':env},plan['slots'],bool(plan['diagnostic'])),'Declared API capability/recompute scope differs')
 require(contract.registry_gate(plan['research_alias'])==plan['registry_binding'],'Actual NEW alias registry association differs')
 # The original strict source, budget, model and dependency admission is retained.
 # This view only adapts producer schema/identity for that frozen admission call.
 return FROZEN_MANIFEST(frozen_admission_view(plan))

def prepare(a):
 require(a.kind=='api' and a.lane=='source35','Only explicit API source35 lane supported')
 spec=read(a.spec);args,env=recipe(a.prepared,spec['slots'],bool(a.diagnostic));scope=contract.preflight(args,env,contract.REQUEST_POLICY,spec['actual_counter_policy'],spec['slots']);contract.registry_gate(api.experimental_alias({'args':args,'env':env},spec['slots'],bool(a.diagnostic)))
 # CPU-only delegation uses unchanged V7 prepare for all original input/model gates.
 # Transient module hooks are restored; no tracked or runtime source is rewritten.
 old_api,old_manifest=base.api,base.manifest_binding
 def chosen_alias(cfg,slots,diagnostic,lane):
  chosen_args,chosen_env=recipe(a.prepared,slots,diagnostic);cfg['args'][:]=chosen_args;cfg['env'].clear();cfg['env'].update(chosen_env);return api.experimental_alias(cfg,slots,diagnostic,lane)
 def admission(plan):
  plan.update(schema=5,harness_generation=8,api_fresh_recompute_generation=1,driver_sha256=sha(Path(__file__)),API_source_plan_sha256=sha(SOURCE_PLAN),API_driver_sha256=sha(Path(api.__file__)),API_request_policy=copy.deepcopy(contract.REQUEST_POLICY),API_migration_scope=scope,actual_cached_state_handoff_qualified=False)
  return manifest_binding(plan)
 base.api=SimpleNamespace(set_arg=api.set_arg,experimental_alias=chosen_alias,registry_gate=contract.registry_gate);base.manifest_binding=admission
 try:base.prepare(a)
 finally:base.api,base.manifest_binding=old_api,old_manifest

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1])
 def stop(signum,frame):api.ABORT.set();raise KeyboardInterrupt('Owned API arm stop')
 for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(signum,stop)
 result=api.run(plan,a.output,a.pre_health,bool(plan['diagnostic']));require(result['plan_sha256']==sha(a.output/'plan.snapshot.json'),'Producer did not declare actual snapshotSHA')
 return 0 if result['collection_and_teardown_passed'] else 1

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--lane',choices=['source35'],required=True);a.add_argument('--kind',choices=['api'],required=True);a.add_argument('--diagnostic',type=int,choices=[0,1],required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--spec',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
