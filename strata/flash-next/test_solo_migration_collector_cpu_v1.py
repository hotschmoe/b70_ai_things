#!/usr/bin/env python3
"""Rehashed synthetic raw producer output, strict negative collector controls."""
import hashlib,json
from pathlib import Path
from audit_batch_fidelity_coverage_v3 import coverage
HERE=Path(__file__).resolve().parent

def main():
 latest=json.loads(Path('/mnt/vm_8tb/b70/build/strata-observer26-cpu-latest-v1.json').read_bytes());plan=json.loads((HERE/'solo-migration-observer-source-draft-v1.json').read_bytes());assert latest['patch_sha256']==plan['patch_sha256'];out=Path(latest['output']);trace=(out/'trace.log').read_text();assert hashlib.sha256(trace.encode()).hexdigest()==latest['trace_sha256'];stages=[(0,0,32),(1,32,48)]
 report=coverage(trace,[11,12],[11,12],stages,out/'capture');assert report['migration_vectors']==98 and report['initial']['vectors']==196 and report['total_raw_bytes']==17756160
 negatives=0
 def bad(text):
  nonlocal negatives
  try:coverage(text,[11,12],[11,12],stages,out/'capture')
  except (AssertionError,KeyError,ValueError):negatives+=1;return
  raise AssertionError('Invalid source/body/slot/phase/coverage accepted')
 lines=trace.splitlines();pick=lambda prefix:next(l for l in lines if l.startswith(prefix));begin=pick('SBF migration_begin ');event=pick('SBF solo_event ');closed=pick('SBF slot_closed ');vec=next(l for l in lines if l.startswith('SBF vector ') and 'solo_migration_' in l);replay=next(l for l in lines if l.startswith('SBF replay ') and 'admission=2' in l);row=next(l for l in lines if l.startswith('SBF row ') and 'admission=2' in l);span=next(l for l in lines if l.startswith('SBF span ') and 'solo_migration_target_verify' in l);resume=next(l for l in lines if l.startswith('SBF resume ') and 'slot=6' in l)
 for removed in [begin,event,closed,vec,replay,row,span,resume]:bad(trace.replace(removed,'',1))
 for duplicate in [begin,event,closed,vec,replay,row]:bad(trace+'\n'+duplicate)
 bad(trace.replace(closed,closed.replace('cancelled=1','cancelled=0'),1));bad(trace.replace(begin,begin.replace('source_slot=0','source_slot=1'),1));bad(trace.replace(begin,begin.replace('source_slotgen=1','source_slotgen=9'),1));bad(trace.replace(replay,replay.replace('active_mask=64','active_mask=1'),1));bad(trace.replace(replay,replay.replace('stage_context_verified=1','stage_context_verified=0'),1));bad(trace.replace(row,row.replace('slot=6','slot=0'),1));bad(trace.replace(event,event.replace('main_session=1','main_session=0'),1));bad(trace.replace(resume,resume.replace('read_from=101','read_from=0'),1))
 result={'CONFIG':'synthetic actual-header raw callbacks298?294 vectors/2RID/2stage/3samples, no model/GPU','COMMAND':'python3 strata/flash-next/test_solo_migration_collector_cpu_v1.py (after header mock test)','RESULT':{'initial_vectors':196,'solo_migration_vectors':98,'total_bytes':17756160,'negative_controls':negatives,'raw_full48_head_after_completed_solo_body':True},'VERDICT':'PASS CPU supplied synthetic collection only; native GPU/serial/original math, actual slot transfer/source/lifecycle/cache unqualified','full_model_math_qualified':False}
 result['CONFIG']=result['CONFIG'].replace('298?294','294');(HERE/'solo-migration-collector-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS synthetic full raw solo collector and controls; no actual migration qualification')
if __name__=='__main__':main()
