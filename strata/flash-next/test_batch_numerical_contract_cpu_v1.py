#!/usr/bin/env python3
"""Synthetic strict native parser/prefix/pipe controls; no native/API/GPU calls."""
import copy,io,json,tempfile
from pathlib import Path
from batch_numerical_protocol_v1 import Roster
from batch_numerical_prefixes_v1 import serial_jobs,compare_raw
from batch_api_trace_v1 import PipeObserver
HERE=Path(__file__).resolve().parent

def main():
 negatives=0
 def bad(fn):
  nonlocal negatives
  try:fn()
  except (ValueError,KeyError,AssertionError):negatives+=1;return
  raise AssertionError('Invalid batch control accepted')
 for slots in [2,4,6]:
  r=Roster(slots);r.consume(f'INFO batch_slots={slots} batch_requested={slots} batch_protocol=2');r.capacity();bad(lambda:r.submit(1,slots,[10],3))
  r.submit(10,0,[101,102],3);r.submit(11,1,[201,202],3)
  for rid,slot in [(10,0),(11,1)]:
   r.consume(f'T 17 rid={rid} slotgen=1');r.consume(f'DONE 1 2 1 0 length 0 0 0 0 0 0 0 0 2 0 rid={rid} slotgen=1');r.consume(f'BADM {slot} 1 rid={rid} slotgen=1')
  r.consume('SBF batch_event pid=1 enginegen=99 event=1 rows=2 active_mask=3 completed=1');r.consume('BT 0 18 rid=10 slotgen=1');assert r.prefix(10,2,17)==[101,102,17]
  bad(lambda:r.prefix(10,2,18));bad(lambda:r.consume('BT 1 99 rid=10 slotgen=1'));bad(lambda:r.consume('BT 0 99 rid=10 slotgen=2'));bad(lambda:r.consume('BDONE 0 9 length 1 rid=10 slotgen=1'))
  stop=r.cancel(11);assert stop=='BSTOP 1 rid=11 slotgen=1';r.consume('BT 1 18 rid=11 slotgen=1');r.consume('BDONE 1 2 cancel 1 rid=11 slotgen=1');r.consume('BT 0 19 rid=10 slotgen=1');r.consume('BDONE 0 3 length 1 rid=10 slotgen=1');bad(lambda:r.consume('BT 0 20 rid=10 slotgen=1'))
  trace='\n'.join(f'SBF vector rid=10 phase=batch_step_{"logits_before_sampler" if l==-1 else "residual"} layer={l} pos=2 token=17' for l in range(-1,48));assert serial_jobs(trace,r)['jobs'][0]['ids']==[101,102,17];bad(lambda:serial_jobs(trace.replace('layer=0 ','layer=1 ',1),r))
  badcap=Roster(slots);badcap.consume('INFO batch_slots=0 batch_requested=0 batch_protocol=2');bad(badcap.capacity)
 calls=[];raw=io.StringIO('INFO batch_slots=2\nBT 0 17 rid=10 slotgen=1\n');pipe=PipeObserver(raw,lambda kind,line:calls.append((kind,line)),'receive');assert list(pipe)==['INFO batch_slots=2\n','BT 0 17 rid=10 slotgen=1\n'] and len(calls)==2
 sink=io.StringIO();writer=PipeObserver(sink,lambda k,l:calls.append((k,l)),'send');writer.write('BSTOP 0 rid=10 slotgen=1\n');assert sink.getvalue()=='BSTOP 0 rid=10 slotgen=1\n'
 with tempfile.TemporaryDirectory(prefix='batch-native-contract-cpu-') as name:
  import struct;root=Path(name);a=root/'a';b=root/'b';a.write_bytes(struct.pack('<2f',1,2));b.write_bytes(a.read_bytes());assert compare_raw(a,b)['bitwise_equal'];b.write_bytes(struct.pack('<2f',1,2.0000002));different=compare_raw(a,b);assert not different['bitwise_equal'] and different['first_different_float']==1 and different['nmse']<1e-6
 result={'CONFIG':'synthetic2/4/6 native protocol/consumed prefixes/unchanged IO proxy; no GPU/API/weights','COMMAND':'python3 strata/flash-next/test_batch_numerical_contract_cpu_v1.py','RESULT':{'negative_controls':negatives,'native_capacity_rid_slotgen_cancel_history':True,'per_actual_input_prefix_full48_head_jobs':True,'tiny_float_difference_rejects_byte_gate':True,'pipe_bytes_unchanged':True},'VERDICT':'PASS CPU contracts only; actual serving/warm->ARM/native cancel/migration/source/allocator/math unqualified','full_model_math_qualified':False}
 (HERE/'batch-numerical-contract-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS synthetic batch control/prefix gates; no actual concurrency qualification')
if __name__=='__main__':main()
