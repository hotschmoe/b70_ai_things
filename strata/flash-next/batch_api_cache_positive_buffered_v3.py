"""Explicit NEW buffered observer/terminal gate, unchanged positive model recipe."""
import argparse,copy
from pathlib import Path
import batch_api_cache_positive_v2 as base
import run_batch_api_cache_positive_buffered_v3 as api
ROOT=base.ROOT;HERE=base.HERE;c1=base.c1;sha=base.sha;read=base.read;write=base.write;require=base.require
lane_contract=base.lane_contract;expected_stage_ranges=base.expected_stage_ranges;verify_model_identity=base.verify_model_identity
SOURCE_PLAN=HERE/'api-buffered-positive-source-plan-v3.json'

def manifest_binding(plan):
 require(plan.get('buffered_trace_generation')==3 and plan.get('terminal_association_generation')==1 and plan['driver_sha256']==sha(Path(__file__)) and plan['buffered_source_plan_sha256']==sha(SOURCE_PLAN),'Explicit NEW buffered positive source producer required')
 for name,want in read(SOURCE_PLAN)['files'].items():require(sha(ROOT/name)==want,'Frozen buffered source changed '+name)
 original=plan['positive_V2_source_preparation'];binding=base.manifest_binding(original)
 for name,value in original.items():
  if name!='driver_sha256':require(plan[name]==value,'Buffered plan changed matched V2 recipe/metadata '+name)
 require(plan['buffered_runner_sha256']==sha(Path(api.__file__)) and plan['buffered_tracer_sha256']==sha(HERE/'batch_api_trace_v3.py'),'Actual NEW buffered runner/tracer source identity differs')
 return binding

def prepare(a):
 base.prepare(a);path=a.output/'plan.json';original=read(path);plan=copy.deepcopy(original);plan.update(driver_sha256=sha(Path(__file__)),positive_V2_source_preparation=original,buffered_trace_generation=3,terminal_association_generation=1,buffered_source_plan_sha256=sha(SOURCE_PLAN),buffered_runner_sha256=sha(Path(api.__file__)),buffered_tracer_sha256=sha(HERE/'batch_api_trace_v3.py'),matched_buffer_IO_savings_qualified=False);manifest_binding(plan);write(path,plan)

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);result=api.run(plan,a.output,a.pre_health,bool(plan['diagnostic']));require(result['plan_sha256']==sha(a.output/'plan.snapshot.json'),'Actual buffered producer planSHA missing/wrong');return 0 if result['collection_and_teardown_passed'] else 1

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--lane',choices=['source35'],required=True);a.add_argument('--kind',choices=['api'],required=True);a.add_argument('--diagnostic',type=int,choices=[0,1],required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--spec',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
