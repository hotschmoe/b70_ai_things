#!/usr/bin/env python3
"""CacheOFF numerical serial supplement; visible original V7 source preparation.
No old failedserial cache PASS transfer or falsified raw lifecycle fields.
"""
import argparse,copy,json,signal,sys,time
from pathlib import Path
from types import SimpleNamespace
import batch_numerical_execution_v7 as preparation
import batch_numerical_proofs_v7 as proof
import run_batch_serial_cacheoff_v8 as serial
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
c1=preparation.c1;sha=proof.sha;read=proof.read;write=proof.write;require=proof.require
lane_contract=proof.lane_contract;expected_stage_ranges=preparation.expected_stage_ranges;verify_model_identity=preparation.verify_model_identity
SOURCE_PLAN=HERE/'batch-serial-cacheoff-v8-source-plan.json'

def prepared_config(plan):
 cfg=read(Path(plan['prepared'])/'server-config.json');args=list(cfg['args']);env=dict(cfg['env'])
 for key,value in [('--max-context',2048),('--prefill',64),('--batch',plan['slots']),('--batch-groups',1),('--prompt-cache',0),('--conversation-cache-mib',0),('--adapt-every',0)]:preparation.api.set_arg(args,key,value)
 env.update(STRATA_BATCH_FIDELITY_DIAG='1',STRATA_BATCH_FIDELITY_ARM='/results/ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',STRATA_SLOT_OWNER_TRACE='1',STRATA_BATCH_FULL_STATE_CHAIN='0',STRATA_BATCH_PUBLIC_PREFIX='0',SYCL_UR_TRACE='2',STRATA_PLE_INPUT33='0',STRATA_PREFIX30='0',STRATA_MIRROR_OWNER_TRACE='1',STRATA_FIDELITY_DIAG='0',STRATA_LAYER0_Q8_DIAG='0')
 return args,env

def manifest_binding(plan):
 require(__debug__ and sys.flags.optimize==0,'Strict numerical frozen assert gates required');require(plan.get('serial_supplement_generation')==8 and plan['kind']=='serial' and plan['driver_sha256']==sha(Path(__file__)),'New cacheOFF serial V8 supplement required')
 for name,want in read(SOURCE_PLAN)['files'].items():require(sha(ROOT/name)==want,'Immutable serial supplement changed '+name)
 base=plan['V7_source_preparation'];require(base['kind']=='serial','V7 source preparation mustbe serial');binding=preparation.manifest_binding(base)
 for key,value in base.items():
  if key!='driver_sha256':require(plan[key]==value,'Serial supplemental plan changed actual V7 preparation field '+key)
 require(plan['driver_sha256']!=base['driver_sha256'],'New controller identity cannot borrow V7 runtime label');args,env=prepared_config(plan);require(plan['args']==args and plan['env']==env,'Exact numerical baseprepared args/env differs');serial.config(plan);return binding

def prepare(a):
 # Frozen V7 CPU metadata preparation/admission generates a NEW temporary recipe.
 preparation.prepare(SimpleNamespace(prepared=a.prepared,lane='source35',kind='serial',diagnostic=1,spec=a.spec,model_identity=a.model_identity,batch_parent=a.batch_parent,group_index=a.group_index,output=a.output))
 path=a.output/'plan.json';base=read(path);plan=copy.deepcopy(base);plan.update(serial_supplement_generation=8,driver_sha256=sha(Path(__file__)),V7_source_preparation=base,cache_qualification_granted=False,legacy_pin_zero_replay_only=False);manifest_binding(plan);write(path,plan);print(json.dumps({'newserialsupplement':str(path),'GPU_executed':False}))

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);result=serial.run(plan,Path(plan['batch_parent'])/'child',a.output,a.pre_health,plan['group_index']);write(a.output/'plan.snapshot.json',plan);result.update(collection_and_teardown_passed=result['passed'],finished_epoch=time.time(),full_model_math_qualified=False,serial_supplement_generation=8,cache_qualification_granted=False,source_lifecycle_scope='fresh cacheOFF GEN1 actual flags/counters/full49 only');result['artifact_bindings']=proof.artifact_bindings(a.output);write(a.output/'report.json',result);return 0 if result['passed'] else 1

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--spec',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--batch-parent',type=Path,required=True);a.add_argument('--group-index',type=int,required=True);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
