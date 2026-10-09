#!/usr/bin/env python3
"""Bounded CPU source-proof controls; no model payload, GPU, SDK or actual serve."""
import copy,json,tempfile
from pathlib import Path
import batch_numerical_proofs_v2 as p
from audit_batch_capture_lifecycle_v2 import audit

def rejected(fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):return
 raise AssertionError('Deliberate control was accepted')

def main():
 controls=0
 trace='SBF batch_event rows=2 active_mask=3 completed=1\nHARNESS ARM warm terminal\nSBF batch_event rows=4 active_mask=15 completed=1'
 assert p.native_row_event(trace,4)['warm_multirow_completed']
 for bad in [trace.replace('HARNESS ARM','MISSING'),trace.replace('rows=2','rows=1'),trace.replace('rows=4','rows=2'),trace.replace('completed=1','completed=0'), 'SBF vector rid=1\n'+trace]:rejected(lambda:p.native_row_event(bad,4));controls+=1
 policy={(7,'admission'):{'reused':0,'read_from':0,'reread_to':0},(7,'solo_migration'):{'reused':2,'read_from':2,'reread_to':0}}
 counter='SBF resume rid=7 slot=0 reused=0 read_from=0 reread_to=0\nSBF resume rid=7 slot=6 reused=2 read_from=2 reread_to=0'
 assert len(p.exact_producer_counters(counter,{7},policy))==2
 for bad in [counter.replace('reused=2','reused=1'),counter.splitlines()[0],counter+'\n'+counter.splitlines()[0]]:rejected(lambda:p.exact_producer_counters(bad,{7},policy));controls+=1
 hist={'a':{'submitted_ids':[1,2],'sampling':{'temperature':0},'max_new':4,'native_generated_ids':[3,4],'native_finish':'length','native_cancel_completed':False}}
 assert p.off_on_histories(hist,copy.deepcopy(hist),set())['passed']
 for field,value in [('submitted_ids',[1,9]),('native_generated_ids',[3,5]),('max_new',5),('native_finish','stop')]:
  bad=copy.deepcopy(hist);bad['a'][field]=value;rejected(lambda:p.off_on_histories(hist,bad,set()));controls+=1
 short=copy.deepcopy(hist);short['a']['native_generated_ids']=[3];short['a']['native_cancel_completed']=True;long=copy.deepcopy(hist);long['a']['native_cancel_completed']=True
 assert p.off_on_histories(short,long,{'a'})['passed']
 rejected(lambda:p.off_on_histories(hist,long,{'a'}));controls+=1
 size=6*(48*10240+248320)*4+4736
 log=f'<--- urUSMDeviceAlloc(.hContext = 0x1, .size = {size}, .ppMem = 0x9 (0x100)) -> UR_RESULT_SUCCESS;\nSBF allocation stage=0 lb=0 le=48 pointer=0x100 bytes={size} rows=6 head=1 owner_queue=verifier_cs\nSBF release_begin stage=0 pointer=0x100 bytes={size}\n<--- urUSMFree(.hContext = 0x1, .pMem = 0x100) -> UR_RESULT_SUCCESS;\nSBF release_returned stage=0'
 result=audit(log,[(0,0,48)])
 assert result['passed'],result
 for bad in [log.replace('.pMem = 0x100','.pMem = 0x101'),log.replace('bytes='+str(size),'bytes='+str(size-1)),log.replace('SBF release_returned stage=0',''),log+'\nSBF release_returned stage=0',log.replace('.hContext = 0x1, .pMem','.hContext = 0x2, .pMem')]:assert not audit(bad,[(0,0,48)])['passed'];controls+=1
 assert not audit(log,[(0,0,32),(1,32,48)])['passed'];controls+=1
 parent={'child_return_code':0,'owned_containers_terminal':True,'forced_cleanup':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'errors':[],'post_health_finished_epoch':11}
 gates=['all_requested_2_4_6_cases','actual_Nrow_events','all_private_RID_raw_head48','exact_consumed_prefixes_and_counters','actual_serial_raw_byte_equal','diagnostic_off_on_output_equal','actual_native_cancel_and_solo_migration','all_observer_slot_mirror_logical_frees']
 child={k:True for k in gates};child['finished_epoch']=10;post={'passed':True,'started':12};assert p.passing_suite(parent,child,post)
 for key in gates:
  bad=dict(child);bad[key]=False;assert not p.passing_suite(parent,bad,post);controls+=1
 assert not p.passing_suite(parent,child,{'passed':True,'started':10.5});controls+=1
 with tempfile.TemporaryDirectory() as d:
  path=Path(d);(path/'receipt.json').write_text(json.dumps({'build_rc':0,'external_source_unchanged':True,'plan_snapshot_unchanged':True,'plan_sha256':'old28'}));rejected(lambda:p.engine_binding(path));controls+=1
 print(json.dumps({'passed':True,'negative_controls':controls,'scope':'CPU source/conditional contracts only; no genuine prepare or actual serving qualification'},ensure_ascii=True))
if __name__=='__main__':main()
