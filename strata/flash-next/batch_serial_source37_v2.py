#!/usr/bin/env python3
"""Explicit source37 fresh cacheOFF serial successor to frozen harness40.
The original preparation and its source/SDK/baseline gates remain nested evidence.
"""
import argparse,copy,time
from pathlib import Path
import batch_numerical_execution_v40 as origin
import run_batch_serial_source37_v1 as serial
from serial37_selection_v1 import selected_jobs
ROOT=origin.ROOT;HERE=origin.HERE;c1=origin.c1
sha=origin.sha;read=origin.read;write=origin.write;require=origin.require
lane_contract=origin.lane_contract;expected_stage_ranges=origin.expected_stage_ranges
verify_model_identity=origin.verify_model_identity;output_budget_contract=origin.output_budget_contract
SOURCE_PLAN=HERE/'batch-serial-source37-source-plan-v2.json'

def manifest_binding(plan):
 require(plan.get('serial_source37_generation')==2 and plan['kind']=='serial' and plan['driver_sha256']==sha(Path(__file__)),'Explicit new source37 serial controller required')
 for name,want in read(SOURCE_PLAN)['files'].items():require(sha(ROOT/name)==want,'Frozen serial successor dependency changed '+name)
 old=plan['harness40_source_preparation'];binding=origin.manifest_binding(old)
 require(old['kind']=='serial' and old['lane']=='source37','Genuine harness40 source37 serial preparation required')
 for key,value in old.items():
  if key!='driver_sha256':require(plan[key]==value,'Serial successor changed original preparation '+key)
 serial.config(plan);selected_jobs(plan,read(Path(plan['batch_parent'])/'child/serial-jobs.json')['jobs']);return binding

def prepare(a):
 origin.prepare(a);path=a.output/'plan.json';old=read(path);plan=copy.deepcopy(old)
 plan.update(driver_sha256=sha(Path(__file__)),serial_source37_generation=2,harness40_source_preparation=old,cache_qualification_granted=False,legacy_pin_zero_replay_only=False)
 jobs=read(Path(plan['batch_parent'])/'child/serial-jobs.json')['jobs'][plan['group_index']*6:plan['group_index']*6+6];receipt=getattr(a,'first_job_adjudication',None)
 indices=list(range(1,len(jobs))) if receipt is not None else list(range(len(jobs)))
 plan.update(actual_serial_selected_indices=indices,actual_serial_job_count=len(indices),actual_serial_submission_budgets=[1]*len(indices),first_job_adjudication={'path':str(receipt.resolve()),'sha256':sha(receipt)} if receipt is not None else None)
 manifest_binding(plan);write(path,plan)

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);result=serial.run(plan,Path(plan['batch_parent'])/'child',a.output,a.pre_health,plan['group_index']);write(a.output/'plan.snapshot.json',plan)
 count=plan['actual_serial_job_count'];require(len(result['comparisons'])==49*count,'Actual full49 serial comparison count differs')
 result.update(collection_and_teardown_passed=result['passed'],finished_epoch=time.time(),plan_sha256=sha(a.output/'plan.snapshot.json'),actual_serial_job_count=count,actual_matched_full49_vector_pairs=49*count,full_model_math_qualified=False,serial_source37_generation=2,cache_qualification_granted=False)
 result['artifact_bindings']=origin.artifact_bindings(a.output);write(a.output/'report.json',result);return 0 if result['passed'] else 1

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare')
 a.add_argument('--lane',choices=['source37'],required=True);a.add_argument('--kind',choices=['serial'],default='serial');a.add_argument('--diagnostic',type=int,choices=[1],default=1)
 for name in ('prepared','one-card-baseline','pair-baseline','spec','model-identity','batch-parent','output'):a.add_argument('--'+name,type=Path,required=True)
 a.add_argument('--one-card-adjudication',type=Path);a.add_argument('--group-index',type=int,required=True)
 a.add_argument('--first-job-adjudication',type=Path)
 a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
 a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
