#!/usr/bin/env python3
"""Synthetic real-protocol grammar for solo->slot->solo consumed prefixes."""
import copy,json
from pathlib import Path
from batch_api_prefixes_v1 import prefix_jobs
HERE=Path(__file__).resolve().parent

def main():
 events=[]
 def emit(kind,**kw):events.append({'kind':kind,'sequence':len(events)+1,'engine_pid':7,**kw})
 def send(line):emit('native_send',line=line,call=1)
 def recv(line):emit('native_receive',line=line)
 def vectors(phase,pos,token):
  for l in range(-1,48):recv(f'SBF vector rid=10 slotgen=1 phase={phase}_{"logits_before_sampler" if l==-1 else "residual"} layer={l} pos={pos} token={token}')
 emit('engine_begin',call=1,rendered_matches_submitted=True,submitted_ids=[101,102]);send('GEN 64 temperature=0 rid=10 101,102');recv('T 17 rid=10 slotgen=0');recv('DONE 1 2 1 0 cancel rid=10 slotgen=0');send('BGEN 0 63 temperature=0 rid=10 101,102,17');vectors('admission',2,17);recv('T 18 rid=10 slotgen=1');recv('DONE 1 3 1 0 length rid=10 slotgen=1');recv('BADM 0 1 rid=10 slotgen=1');vectors('batch_step',3,18);recv('SBF batch_event rows=2 completed=1');recv('BT 0 19 rid=10 slotgen=1');send('BSTOP 0 rid=10 slotgen=1');recv('BT 0 20 rid=10 slotgen=1');recv('BDONE 0 3 cancel 1 rid=10 slotgen=1');send('GEN 60 temperature=0 rid=10 101,102,17,18,19,20');vectors('solo_migration',5,20);recv('T 21 rid=10 slotgen=0');recv('DONE 1 6 1 0 length rid=10 slotgen=0')
 result=prefix_jobs(events);assert result['all_segments_terminal'] and result['actual_completed_multirow_events']==1 and [j['ids'] for j in result['jobs']]==[[101,102,17],[101,102,17,18],[101,102,17,18,19,20]]
 negatives=0
 def bad(rows):
  nonlocal negatives
  try:prefix_jobs(rows)
  except (ValueError,KeyError,AssertionError):negatives+=1;return
  raise AssertionError('Invalid continuation/input/history accepted')
 for text,replacement in [('101,102,17,18,19,20','101,102,99,18,19,20'),('T 17 rid=10 slotgen=0','T 17 rid=99 slotgen=0'),('BT 0 19 rid=10 slotgen=1','BT 1 19 rid=10 slotgen=1'),('BDONE 0 3 cancel 1','BDONE 0 9 cancel 1'),('layer=0 pos=5 token=20','layer=0 pos=5 token=99')]:
  rows=copy.deepcopy(events)
  for row in rows:
   if 'line' in row:row['line']=row['line'].replace(text,replacement)
  bad(rows)
 bad([r for r in events if r.get('line')!='DONE 1 2 1 0 cancel rid=10 slotgen=0']);bad([r for r in events if 'layer=0 pos=5 ' not in r.get('line','')])
 receipt={'CONFIG':'synthetic native solo/slot/solo tags and all49 vectors perphase; no API/socket/model/GPU','COMMAND':'python3 strata/flash-next/test_batch_api_prefixes_cpu_v1.py','RESULT':{'exact_actual_segment_consumed_prefix_jobs':3,'negative_controls':negatives,'migration_and_cancel_raw_collector_still_required':True},'VERDICT':'PASS CPU history grammar only; actual concurrent/migration/math/source/owner gates absent','full_model_math_qualified':False}
 (HERE/'batch-api-prefix-cpu-receipt-v1.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS synthetic solo/slot/solo input history controls; no actual migration')
if __name__=='__main__':main()
