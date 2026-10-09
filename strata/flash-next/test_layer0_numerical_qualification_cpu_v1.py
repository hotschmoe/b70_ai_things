#!/usr/bin/env python3
"""Partial frame/source/nonce/lifecycle numerical contracts; CPU only, no model."""
import copy,hashlib,json,struct,tempfile
from pathlib import Path
import layer0_numerical_qualification_v1 as q
from audit_layer0_capture_lifecycle_v1 import audit_text


def reject(fn):
 try:fn()
 except (ValueError,KeyError):return
 raise AssertionError('Invalid numerical evidence accepted')


def main():
 from unittest.mock import patch
 from audit_v6_actual_request_records import strengthen,digest,fnv
 engine=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T130403Z-8b4zuyvp');built=q.engine_binding(engine,q.PLAN_SOURCE_SHA);assert built['build_rc']==0
 oldprep=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f07-20261009/c1-onecard-combined20-prepared-v1');reject(lambda:q.candidate_binding({'engine_root':str(engine),'prepared':str(oldprep),'engine_receipt_sha256':q.sha(engine/'receipt.json'),'prepared_sha256':q.sha(oldprep/'prepared.json')}))
 with tempfile.TemporaryDirectory(prefix='layer0-numerical-cpu-') as name:
  directory=Path(name);(directory/'layer0').mkdir();layout=directory/'strata/flash-next/layer0_numerical_contract_v1.hpp';layout.parent.mkdir(parents=True);layout.write_text((q.ROOT/'strata/flash-next/layer0_numerical_contract_v1.hpp').read_text().replace('40960','40964',1))
  with patch.object(q,'ROOT',directory):reject(q.field_contract)
  binding='a'*64;frame={'schema':1,'pid':9,'request':1,'stage':0,'layer':0,'rows':1,'position':0,'token':248045,'reused':0,'binding_sha256':binding,'gen_ids':[248045],'graph_key':2,'completed_nonce':(9<<32)|1,'request_replay_marker_verified':True,'raw_fused_hidden_observed':False,'complete_preregistered_layout':False,'full_model_math_qualified':False,'fields':[]}
  for contract in q.field_contract():
   item=dict(contract,provenance='actual_buffer' if contract['observed'] else 'UNOBSERVED_no_producer_hook',file='/results/layer0/'+contract['name']+'.bin' if contract['observed'] else '')
   if contract['observed']:
    raw=b'\0'*contract['bytes']
    if contract['name']=='router_ids':raw=struct.pack('<10i',*range(10))
    (directory/'layer0'/Path(item['file']).name).write_bytes(raw)
   frame['fields'].append(item)
  metadata=directory/'layer0/frame.json';q.write(metadata,frame);raw={'ids':[248045],'stderr':['SFD request pid=9 request=1 tokens=1 ids=248045','L0Q8 frame pid=9 request=1 stage=0 layer=0 position=0 token=248045 metadata=/results/layer0/frame.json']};checked=q.collect_frame(raw,directory,binding);assert len(checked['observed'])==26 and len(checked['unobserved'])==7
  for key,value in [('completed_nonce',0),('graph_key',0),('full_model_math_qualified',True),('raw_fused_hidden_observed',True),('position',1),('reused',1),('stage',1),('pid',10)]:
   wrong=copy.deepcopy(frame);wrong[key]=value;q.write(metadata,wrong);reject(lambda:q.collect_frame(raw,directory,binding))
  wrong=copy.deepcopy(frame);wrong['fields'][24].update(observed=True,file='/results/layer0/unobserved.bin',provenance='actual_buffer');q.write(metadata,wrong);reject(lambda:q.collect_frame(raw,directory,binding));q.write(metadata,frame)
  file=directory/'layer0/residual_input.bin';old=file.read_bytes();file.write_bytes(old[:-4]);reject(lambda:q.collect_frame(raw,directory,binding));file.write_bytes(old)
  ids=directory/'layer0/router_ids.bin';old_ids=ids.read_bytes();ids.write_bytes(struct.pack('<10i',*([0]*10)));reject(lambda:q.collect_frame(raw,directory,binding));ids.write_bytes(old_ids)
 for terminal,outtoken in [('length',17),('stop',248046)]:
  ids0=[248045];prefix=[dict(event='selection',pid=9,request=1,prompt_tokens=1,actual_reused=0,token_fnv_le32=fnv(ids0)),dict(event='finish',pid=9,request=1,prompt_tokens=1,actual_reused=0,cancelled=False)]
  span=dict(event='stage_span',pid=9,request=1,phase='verify_body',device=0,lb=0,le=48,lo=0,hi=1,complete=False,commit_proven=False)
  events=[dict(event='begin',pid=9,request=1,tokens=1,input_sha256_le32=digest(ids0)),span,dict(span,complete=True),dict(event='committed_live',pid=9,request=1,tokens=1,ids=ids0,sha256_le32=digest(ids0),ids_truncated=False,phase='complete')]
  raw0={'ids':ids0,'output_ids':[outtoken],'done':'DONE 1 1 0 0 '+terminal,'cancel_requested':None,'stderr':['PREFIX_DIAG '+json.dumps(e) for e in prefix]+['PCL '+json.dumps(e) for e in events]+['SFD request pid=9 request=1 tokens=1 ids=248045']}
  assert strengthen(raw0,[(0,48)],True,True)['passed'] # Last predicted output is unconsumed in both source paths.
 trace='''<--- urUSMDeviceAlloc(.hContext = 0x111, .size = 7023104, .ppMem = 0x222 (0x1000)) -> UR_RESULT_SUCCESS;
L0Q8 allocation stage=0 pointer=0x1000 bytes=7023104 owner_queue=verifier_cs
L0Q8 release_begin stage=0 pointer=0x1000 bytes=7023104
<--- urUSMFree(.hContext = 0x111, .pMem = 0x1000) -> UR_RESULT_SUCCESS;
L0Q8 release_returned stage=0
''';assert audit_text(trace)['passed']
 for label,wrong in [('missing_free','\n'.join(line for line in trace.splitlines() if 'urUSMFree' not in line)),('wrong_context',trace.replace('urUSMFree(.hContext = 0x111','urUSMFree(.hContext = 0x112')),('failed_free',trace.replace('urUSMFree(.hContext = 0x111, .pMem = 0x1000) -> UR_RESULT_SUCCESS','urUSMFree(.hContext = 0x111, .pMem = 0x1000) -> UR_RESULT_ERROR_UNKNOWN')),('double_free',trace.replace('L0Q8 release_returned','<--- urUSMFree(.hContext = 0x111, .pMem = 0x1000) -> UR_RESULT_SUCCESS;\nL0Q8 release_returned')),('omitted56controlkeybytes',trace.replace('7023104','7023048'))]:assert not audit_text(wrong)['passed'],label
 q.write(Path(__file__).with_name('layer0-numerical-orchestration-cpu-receipt-v1.json'),{'CONFIG':'actual21 linked source hashes readonly, old20 C1 rejection, synthetic partialframes/UR chronology','COMMAND':'python3 strata/flash-next/test_layer0_numerical_qualification_cpu_v1.py','RESULT':{'actual21_engine_source_identity':True,'old20_upload_C1_rejected':True,'exact26of33_layout_and_nonce_control_negatives':True,'all7023104B_including56_control_keys_owner_context_frees':True,'logical_negative_controls':True},'VERDICT':'PASS CPU/source contracts; synthetic frames not model outputs; no GPU/full-math/actual21 numerical qualification','full_model_math_qualified':False})
 print('PASS actual21 source/old20 rejection,26of33 nonce/provenance andwholebuffer owner/free negatives; CPU only')
if __name__=='__main__':main()
