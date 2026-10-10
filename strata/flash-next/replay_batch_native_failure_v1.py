#!/usr/bin/env python3
"""Readonly preserved-log native roster/raw coverage replay; no model/device access.
Writes optional NEW result outside evidence; never rewrites native receipts.
"""
import argparse,hashlib,json,traceback
from pathlib import Path
from batch_numerical_protocol_v2 import Roster
from batch_numerical_prefixes_v2 import serial_jobs
from batch_numerical_proofs_v6 import native_row_event,exact_producer_counters
from audit_batch_fidelity_coverage_v2 import coverage as frozen_coverage
from audit_batch_fidelity_coverage_v5 import coverage as scoped_coverage

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def replay(root):
 root=Path(root).resolve();plan=json.loads((root/'plan.snapshot.json').read_text());trace=(root/'engine.combined.log').read_text();failed=json.loads((root/'report.json').read_text());
 if failed['plan_sha256']!=sha(root/'plan.snapshot.json'):raise ValueError('Preserved failed report/plan changed')
 for name,binding in failed['artifact_bindings'].items():
  path=root/name
  if not path.is_file() or path.is_symlink() or path.stat().st_size!=binding['bytes'] or sha(path)!=binding['sha256']:raise ValueError('Preserved failure artifact changed: '+name)
 roster=Roster(plan['slots']);warm=[1001+i for i in range(plan['slots'])];targets=[2001+i for i in range(plan['slots'])]
 for i,rid in enumerate(warm):roster.submit(rid,i,plan['tokens']['warm'][i],32)
 armed=False;armidx=0;stopped=False;commands=[];last=None;results={}
 try:
  for number,line in enumerate(trace.splitlines(),1):
   last=(number,line)
   if line.startswith('HARNESS ARM'):
    if roster.owners:raise ValueError('Preserved warm submissions were not terminal before ARM')
    armed=True;armidx=len(roster.events)
    for i,rid in enumerate(targets):roster.submit(rid,i,plan['tokens']['target'][i],32)
   roster.consume(line)
   if armed and plan['cancel_index'] is not None:
    cancel=targets[plan['cancel_index']]
    if not stopped and roster.requests[cancel]['admitted'] and any(int(e['rows'])>=2 and int(e['active_mask'])&(1<<roster.requests[cancel]['slot']) for e in roster.events[armidx:]):commands.append({'log_line':number,'command':roster.cancel(cancel),'scope':'readonly simulation of source consumer condition; actual input write not independently recorded'});stopped=True
  results['roster']={'status':'complete_readonly_replay','warm_generated':[len(roster.requests[r]['generated']) for r in warm],'target_generated':[len(roster.requests[r]['generated']) for r in targets],'target_done':[roster.requests[r]['done'] for r in targets],'armed_completed_events':len(roster.events)-armidx}
 except BaseException as exc:results['roster']={'status':'FAILED','exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'last_log_line':last}
 stages=[(0,0,48)] if '--layer-split' not in plan['args'] else [(0,0,32),(1,32,48)]
 steps=[('frozen_whole_trace_coverage',lambda:frozen_coverage(trace,targets,stages,root/'captures')),('new_ARM_scoped_coverage',lambda:scoped_coverage(trace,targets,stages,root/'captures')),('native_row_event',lambda:native_row_event(trace,plan['slots'])),('exact_counters',lambda:exact_producer_counters(trace,set(targets),{(rid,'admission'):{'reused':0,'read_from':0,'reread_to':-1} for rid in targets})),('serial_jobs',lambda:serial_jobs(trace,roster))]
 for label,call in steps:
  try:results[label]={'status':'complete_readonly_replay','result':call()}
  except BaseException as exc:results[label]={'status':'FAILED','exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
 return {'schema':1,'source_child_root':str(root),'source_plan_sha256':sha(root/'plan.snapshot.json'),'source_trace_sha256':sha(root/'engine.combined.log'),'preserved_failed_report_sha256':sha(root/'report.json'),'results':results,'simulated_cancel_commands':commands,'actual_driver_target_max_new':32,'planned_max_new_by_request':plan.get('max_new_by_request'),'planned_maxnew_mismatch_cause_proven':False,'scope':'readonly parser/coverage diagnostics only; no rewrite of failed parent/child proof, no serial numerical equivalence/math/concurrency/speed qualification','model_or_GPU_execution':False,'preserved_artifact_bindings_unchanged':True}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child-root',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();report=replay(a.child_root)
 if a.output:
  if a.output.exists():raise ValueError('New diagnostic output required')
  a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='ascii')
 print(json.dumps({key:{k:v for k,v in value.items() if k not in ('result','traceback')} for key,value in report['results'].items()},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
