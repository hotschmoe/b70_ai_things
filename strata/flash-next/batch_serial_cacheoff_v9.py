#!/usr/bin/env python3
"""Future controller declares actual child snapshotSHA and copycounts explicitly.
No executed V8 source/evidence changes or retroactive producer field claims.
"""
import argparse,copy,time
from pathlib import Path
import batch_serial_cacheoff_v8 as base
ROOT=base.ROOT;HERE=base.HERE;c1=base.c1;sha=base.sha;read=base.read;write=base.write;require=base.require
lane_contract=base.lane_contract;expected_stage_ranges=base.expected_stage_ranges;verify_model_identity=base.verify_model_identity
SOURCE_PLAN=HERE/'cacheoff-v8-adjudication-and-v9-source-plan.json'

def manifest_binding(plan):
 require(plan.get('future_controller_generation')==9 and plan['driver_sha256']==sha(Path(__file__)),'Future declared snapshot producer V9 required')
 for name,want in read(SOURCE_PLAN)['files'].items():require(sha(ROOT/name)==want,'Frozen future controller dependency changed '+name)
 original=plan['V8_source_preparation'];binding=base.manifest_binding(original)
 for key,value in original.items():
  if key!='driver_sha256':require(plan[key]==value,'Future V9 changed actual V8 preparation field '+key)
 return binding

def prepare(a):
 base.prepare(a);path=a.output/'plan.json';original=read(path);plan=copy.deepcopy(original);plan.update(future_controller_generation=9,driver_sha256=sha(Path(__file__)),V8_source_preparation=original);manifest_binding(plan);write(path,plan)

def completed_report(plan,output,result):
 path=Path(output)/'plan.snapshot.json';write(path,plan);actual_jobs=read(Path(plan['batch_parent'])/'child/serial-jobs.json')['jobs'][plan['group_index']*6:plan['group_index']*6+6];pairs=len(result['comparisons']);require(pairs==49*len(actual_jobs),'Future actual full49 copy/comparison count differs')
 result.update(collection_and_teardown_passed=result['passed'],finished_epoch=time.time(),full_model_math_qualified=False,serial_supplement_generation=8,future_controller_generation=9,cache_qualification_granted=False,plan_sha256=sha(path),actual_serial_job_count=len(actual_jobs),actual_matched_full49_vector_pairs=pairs,source_lifecycle_scope='fresh cacheOFF GEN1 actual flags/counters/full49 only; declared real snapshotSHA');result['artifact_bindings']=base.proof.artifact_bindings(Path(output));write(Path(output)/'report.json',result);return result

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);result=base.serial.run(plan,Path(plan['batch_parent'])/'child',a.output,a.pre_health,plan['group_index']);report=completed_report(plan,a.output,result);return 0 if report['passed'] else 1

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--spec',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--batch-parent',type=Path,required=True);a.add_argument('--group-index',type=int,required=True);a.add_argument('--output',type=Path,required=True);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
