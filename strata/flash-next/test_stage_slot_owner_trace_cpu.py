#!/usr/bin/env python3
"""Synthetic chronological slot owner/host trace controls; no device execution."""
import copy,json,tempfile
from pathlib import Path
from collect_stage_slot_owner_oracle import collect
ROOT=Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='slot-trace-cpu-') as name:
 d=Path(name);source=ROOT/'strata/flash-next/stage-slot-owner-oracle-plan.json';plan=json.loads(source.read_text());p=d/'plan.json';p.write_text(json.dumps(plan))
 rows=['<--- urContextGetNativeHandle(.hContext = 0xa, .phNativeContext = 0xf (11)) -> UR_RESULT_SUCCESS;'];number=1
 for label,qn in [('normal-0-0',12),('normal-0-1',12),('normal-0-2',12),('partial-zero-0',1),('partial-throw-0',1),('probe-0',0)]:
  addresses=[]
  roles=['probe'] if label.startswith('probe') else ['slot_arena']+['host_step','host_pos']*qn
  for role in roles:
   ptr='0x%x'%(0x1000+number*0x100);number+=1;size=128 if role in ['probe','slot_arena'] else 32;alloc='urUSMHostAlloc' if role.startswith('host') else 'urUSMDeviceAlloc'
   rows.append('<--- '+alloc+'(.hContext = 0xa, .size = '+str(size)+', .ppMem = 0xf ('+ptr+')) -> UR_RESULT_SUCCESS;')
   rows.append('UPLOAD_USM '+json.dumps({'event':'owner_register','stage':label,'ze_context':'0xb','pointer':ptr,'bytes':size,'role':role,'name':role}));addresses.append(ptr)
  rows.append('UPLOAD_USM '+json.dumps({'event':'destroy_begin','stage':label}))
  for ptr in addresses:rows.append('<--- urUSMFree(.hContext = 0xa, .pMem = '+ptr+') -> UR_RESULT_SUCCESS;')
  rows.append('UPLOAD_USM '+json.dumps({'event':'destroy_end','stage':label,'owning_destructor_returned':True}))
 raw=d/'raw.json';raw.write_text(json.dumps({'owner_and_probe_passed':True,'model_weights_loaded':False,'inference_or_graph_retirement_qualified':False,'owner_registrations':82,'stages':1,'cells':64}));log=d/'trace.log';log.write_text('\n'.join(rows)+'\n');r=collect(raw,log,p,'owner_card0');assert r['passed'] and all(r['negative_controls'].values())
 bad=rows[:];bad.pop(next(i for i,x in enumerate(bad) if '<--- urUSMFree' in x));log.write_text('\n'.join(bad));assert not collect(raw,log,p,'owner_card0')['passed']
 log.write_text('');assert not collect(raw,log,p,'owner_card0')['passed']
 log.write_text('\n'.join(rows).replace('partial-throw-0','unplanned-partial'));assert not collect(raw,log,p,'owner_card0')['passed']
 print('PASS CPU synthetic82 device/host owners, chronological context frees; missing/double/failedfree negatives, empty trace and wrong role-stage roster rejected. No GPU/allocator result.')
