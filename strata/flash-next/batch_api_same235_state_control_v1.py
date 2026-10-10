"""NEW source preparation for exact serial fresh0/fresh1 state controls."""
import argparse,copy
from pathlib import Path
import batch_api_cache_positive_buffered_v6 as origin
import same235_state_control_v1 as controls
import run_batch_api_same235_state_control_v1 as api
ROOT=origin.ROOT;HERE=origin.HERE;c1=origin.c1;sha=origin.sha;read=origin.read;write=origin.write;require=origin.require
lane_contract=origin.lane_contract;expected_stage_ranges=origin.expected_stage_ranges;verify_model_identity=origin.verify_model_identity
SOURCE_PLAN=controls.PLAN

def manifest_binding(plan):
 controls.source_binding();require(plan.get('same235_state_control_generation')==1 and plan['driver_sha256']==sha(__file__) and plan['state_control_source_plan_sha256']==sha(SOURCE_PLAN) and plan['state_control_runner_sha256']==sha(api.__file__),'Explicit NEW same235 state control producer required');original=plan['positive_V6_source_preparation'];binding=origin.manifest_binding(original)
 for k,v in original.items():
  if k!='driver_sha256':require(plan[k]==v,'Control changed matched original V6 configuration '+k)
 require(plan['serial_state_control_jobs']==controls.jobs(),'Exact same235 serial jobs/source inputs differ');return binding

def prepare(a):
 original=read(a.origin_plan);origin.manifest_binding(original);plan=copy.deepcopy(original);plan.update(driver_sha256=sha(__file__),same235_state_control_generation=1,state_control_source_plan_sha256=sha(SOURCE_PLAN),state_control_runner_sha256=sha(api.__file__),positive_V6_source_preparation=original,serial_state_control_jobs=controls.jobs(),actual_cached_state_handoff_qualified=False,underlying_EOS_cause_established=False);manifest_binding(plan);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',plan)

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);result=api.run(plan,a.output,a.pre_health,bool(plan['diagnostic']));require(result['plan_sha256']==sha(a.output/'plan.snapshot.json'),'Actual control producer snapshotSHA differs');return int(not result['collection_and_teardown_passed'])
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--origin-plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
