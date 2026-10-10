#!/usr/bin/env python3
"""ARM-scoped initial BGEN observer audit. Frozen V2 numerical/raw gates unchanged.
Source35 batch event ordinal resets when all private slots become idle; its
identity is batch-local, not globally unique across warm-to-ARM sessions.
"""
import hashlib
from pathlib import Path
from audit_batch_fidelity_coverage_v2 import coverage as strict_initial
HERE=Path(__file__).resolve().parent
V2_SHA='2d3719ad4f020811462e5a5e8f2d93da950658d9a901e9aaa9777d2e4439e24a'

def require(ok,msg):
 if not ok:raise ValueError(msg)

def scoped_trace(trace,requested_rows,marker='HARNESS ARM'):
 require(type(requested_rows) is int and requested_rows in (2,4,6),'Bounded actualbatch2/4/6 required');lines=trace.splitlines();marks=[i for i,line in enumerate(lines) if line.startswith(marker)];require(len(marks)==1,'Exactly one same-process warm-to-ARM boundary required');at=marks[0];warm={};target={}
 require(not any(line.startswith(('SBF request ','SBF resume ','SBF replay ','SBF row ','SBF vector ','SBF allocation ','SBF migration_begin ','SBF solo_event ')) for line in lines[:at]),'Unarmed warmup published target observer data')
 for index,line in enumerate(lines):
  if not line.startswith('SBF batch_event '):continue
  f=dict(word.split('=',1) for word in line.split()[2:] if '=' in word)
  key=(int(f['pid']),int(f['enginegen']),int(f['event']));rows=int(f['rows']);mask=int(f['active_mask']);require(all(v>0 for v in key),'Actual batch event identity absent')
  require(f['completed']=='1' and 1<=rows<=requested_rows and 0<mask<(1<<requested_rows) and mask.bit_count()==rows,'Actual completed private-slot batch geometry invalid')
  scope=warm if index<at else target;require(key not in scope,'Duplicate completed batch event within '+('warm' if index<at else 'armed')+' session atline'+str(index+1));scope[key]=f
 require(warm and target,'Actual warm and armed body events required');require(any(int(f['rows'])>=2 for f in warm.values()),'No actual multi-row warm body');require(any(int(f['rows'])==requested_rows for f in target.values()),'No actual requested-N armed body')
 warm_engines={(k[0],k[1]) for k in warm};target_engines={(k[0],k[1]) for k in target};require(len(warm_engines)==1 and target_engines==warm_engines,'Actual sameprocess native engine changed across ARM')
 return '\n'.join(lines[at+1:]),{'ARM_line':at+1,'warm_completed_events':len(warm),'armed_completed_events':len(target),'same_engine_PID_generation':list(next(iter(warm_engines))),'warm_to_armed_ordinal_overlap_count':len(set(warm)&set(target)),'batch_ordinal_scope':'separate warm and armed sessions; no within-session duplicates allowed','frozen_strict_target_coverage_unchanged':True,'full_model_math_qualified':False}

def coverage(trace,expected_rids,stages,capture_root,requested_rows=None):
 require(hashlib.sha256((HERE/'audit_batch_fidelity_coverage_v2.py').read_bytes()).hexdigest()==V2_SHA,'Frozen original V2 coverage source changed')
 n=len(set(expected_rids)) if requested_rows is None else requested_rows;require(n==len(set(expected_rids)),'Requested initialprivate slots/RID roster differs')
 target,boundary=scoped_trace(trace,n)
 try:result=strict_initial(target,expected_rids,stages,capture_root)
 except BaseException as exc:
  # Preserve exception cause, including formerly empty AssertionError details.
  raise ValueError('Strict armed initial observer coverage failed: '+type(exc).__name__+': '+str(exc)) from exc
 return {**result,'warm_to_ARM_event_scope':boundary,'whole_trace_warm_and_target_not_one_event_namespace':True}
