#!/usr/bin/env python3
"""Negative controls for independent V6 external record binding; no GPU/model."""
import copy,json
from pathlib import Path
from audit_v6_actual_request_records import strengthen,digest,fnv

def rejected(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid record accepted')

def fixture():
 ids=[11,12,13];prefix=[dict(event='selection',pid=9,request=1,prompt_tokens=3,actual_reused=0,token_fnv_le32=fnv(ids)),dict(event='finish',pid=9,request=1,prompt_tokens=3,actual_reused=0,cancelled=False)]
 stage=dict(event='stage_span',pid=9,request=1,phase='verify_body',device=0,lb=0,le=48,lo=0,hi=3,complete=False,commit_proven=False)
 pcl=[dict(event='begin',pid=9,request=1,tokens=3,input_sha256_le32=digest(ids)),stage,dict(stage,complete=True),dict(event='committed_live',pid=9,request=1,tokens=4,ids=[11,12,13,17],sha256_le32=digest([11,12,13,17]),ids_truncated=False,phase='complete')]
 return dict(ids=ids,output_ids=[17,248046],done='DONE 2 3 0 0 stop',cancel_requested=None,stderr=['PREFIX_DIAG '+json.dumps(e) for e in prefix]+['PCL '+json.dumps(e) for e in pcl]+['SFD request pid=9 request=1 tokens=3 ids=11,12,13 shape=4x2560','SFD vector pid=9 request=1 stage=0 lb=0 le=48 phase=first_logits_before_sampler layer=-1 pos=2 token=13 bytes=993280 file=CPU_UNUSED'])

def main():
 good=fixture();result=strengthen(good,[(0,48)],True,True);assert result['exact_enter_return_equal']
 checks=[]
 def mutate(name,modify):
  bad=copy.deepcopy(good);modify(bad);rejected(lambda:strengthen(bad,[(0,48)],True,True));checks.append(name)
 def event_edit(record,prefix,event,edit):
  for index,line in enumerate(record['stderr']):
   if line.startswith(prefix):
    value=json.loads(line[len(prefix):])
    if value.get('event')==event:edit(value);record['stderr'][index]=prefix+json.dumps(value)
 mutate('extra_unmatched_enter',lambda r:r['stderr'].append(next(line for line in r['stderr'] if line.startswith('PCL ') and 'stage_span' in line and '"complete": false' in line)))
 mutate('missing_enter',lambda r:r['stderr'].__setitem__(slice(None),[line for line in r['stderr'] if not(line.startswith('PCL ') and 'stage_span' in line and '"complete": false' in line)]))
 mutate('PCL_selfcoherent_wrong_pid',lambda r:r['stderr'].__setitem__(slice(None),[line.replace('"pid": 9','"pid": 10') if line.startswith('PCL ') else line for line in r['stderr']]))
 mutate('SFD_wrong_pid',lambda r:r['stderr'].__setitem__(slice(None),[line.replace('pid=9','pid=10') if line.startswith('SFD ') else line for line in r['stderr']]))
 mutate('PREFIX_finish_wrong_request',lambda r:event_edit(r,'PREFIX_DIAG ','finish',lambda e:e.__setitem__('request',2)))
 mutate('PREFIX_wrong_source_fnv',lambda r:event_edit(r,'PREFIX_DIAG ','selection',lambda e:e.__setitem__('token_fnv_le32','0'*16)))
 mutate('PCL_wrong_source_sha',lambda r:event_edit(r,'PCL ','begin',lambda e:e.__setitem__('input_sha256_le32','0'*64)))
 mutate('wrong_span_device',lambda r:event_edit(r,'PCL ','stage_span',lambda e:e.__setitem__('device',1)))
 mutate('committed_source_suffix_wrong',lambda r:event_edit(r,'PCL ','committed_live',lambda e:e.update(ids=[11,12,13,18],sha256_le32=digest([11,12,13,18]))))
 cancelled=copy.deepcopy(good);cancelled['cancel_requested']='prefill';event_edit(cancelled,'PREFIX_DIAG ','finish',lambda e:e.__setitem__('cancelled',True));event_edit(cancelled,'PCL ','committed_live',lambda e:e.__setitem__('phase','prefill'));cancelled['stderr']=[line for line in cancelled['stderr'] if not(line.startswith('PCL ') and 'stage_span' in line and '"complete": true' in line)]
 partial=strengthen(cancelled,[(0,48)],True,True);assert partial['cancelled'] and not partial['exact_enter_return_equal'] and partial['partial_entries_explicitly_unobserved'] and partial['cancelled_partial_work_is_zero'] is False
 off=copy.deepcopy(good);off['stderr']=[line for line in off['stderr'] if line.startswith('PREFIX_DIAG ')];assert strengthen(off,[(0,48)],False,False)['passed']
 receipt={'CONFIG':'CPU synthetic metadata controls only; not model records','COMMAND':'python3 strata/flash-next/test_audit_v6_actual_request_records_cpu.py','RESULT':{'positive_exact_binding':True,'rejected':checks,'cancel_partial_explicit_unobserved':True,'flag0_scope_explicit_unobserved':True},'VERDICT':'PASS stricter acceptance negative controls; no GPU/model/source edits'}
 Path(__file__).with_name('v6-actual-record-auditor-cpu-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS exact enter/return and cross PREFIX/PCL/SFD source binding; partial cancellation unobserved')
if __name__=='__main__':main()
