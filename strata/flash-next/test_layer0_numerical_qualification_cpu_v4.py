#!/usr/bin/env python3
"""Synthetic V4 field, pointer, packet, page and owner controls. No model reads."""
import copy,hashlib,json,struct,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import layer0_numerical_qualification_v4 as q
import qualify_layer0_numerical_v4 as parent
from audit_layer0_capture_lifecycle_v2 import audit_text,live_blob_bindings
from independent_q8_1_activation_v1 import encode,require_packet_exact
from verify_layer0_packets_v3 import verify,PAIRS,hidden_metrics
from source_page_watchdog_v3 import inspect_pages,KNOWN_PAGES


def reject(fn):
 try:fn()
 except (ValueError,KeyError,AssertionError,FileNotFoundError):return
 raise AssertionError('Bad evidence accepted')


def alloc(kind,context,address,size):
 return f'<--- {kind}(.hContext = {context}, .size = {size}, .ppMem = 0x222 ({hex(address)})) -> UR_RESULT_SUCCESS;\n'


def main():
 negatives=0
 assert q.post_hash_boundary({'started':3.051},{'finished_epoch':2.95},{'finished_epoch':3.05})==3.05
 for start in [2.9,3.04,float('nan')]:
  reject(lambda:q.post_hash_boundary({'started':start},{'finished_epoch':2.95},{'finished_epoch':3.05}));negatives+=1
 assert len(parent.source_plan_binding()['expected_patched_source_sha256'])==51
 assert parent.sha(Path(q.__file__))==parent.L0_NUMERICAL_SHA
 with tempfile.TemporaryDirectory(prefix='layer0-v4-cpu-') as name:
  root=Path(name);(root/'layer0').mkdir();binding='a'*64
  pointers=[0x10000000+i*0x400000 for i in range(10)];tiers=[1]*5+[2]*5
  trace=alloc('urUSMDeviceAlloc','0x111',0x1000,7023304)+'L0Q8 allocation stage=0 pointer=0x1000 bytes=7023304 owner_queue=verifier_cs\n'
  for address,tier in zip(pointers,tiers):trace+=alloc('urUSMDeviceAlloc' if tier==1 else 'urUSMHostAlloc','0x111',address,3072000)
  trace+='L0Q8 frame pid=9 request=1 stage=0 metadata=frame.json\n'
  end='L0Q8 release_begin stage=0 pointer=0x1000 bytes=7023304\n<--- urUSMFree(.hContext = 0x111, .pMem = 0x1000) -> UR_RESULT_SUCCESS;\nL0Q8 release_returned stage=0\n'
  assert audit_text(trace+end)['passed'];assert len(live_blob_bindings(trace,9,1,pointers,tiers)['source_live_ranges'])==10
  for wrong in [(trace+end).replace('urUSMDeviceAlloc(.hContext = 0x111, .size = 7023304','urUSMHostAlloc(.hContext = 0x111, .size = 7023304'),(trace+end).replace('7023304','7023104'),(trace+end).replace('7023304','7023048'),(trace+end).replace('urUSMFree(.hContext = 0x111','urUSMFree(.hContext = 0x112'),trace+'L0Q8 release_returned stage=0\n']:
   assert not audit_text(wrong)['passed'];negatives+=1
  for wrong_addresses,wrong_tiers,wrong_trace in [(pointers,tiers,trace+'L0Q8 frame pid=9 request=1 metadata=frame.json\n'),(pointers,[2]*10,trace),([pointers[0]+4]+pointers[1:],tiers,trace),(pointers,tiers,trace.replace('.hContext = 0x111, .size = 3072000','.hContext = 0x112, .size = 3072000')),(pointers,tiers,trace.replace('L0Q8 frame','<--- urUSMFree(.hContext = 0x111, .pMem = '+hex(pointers[0])+') -> UR_RESULT_SUCCESS;\nL0Q8 frame'))]:
   reject(lambda:live_blob_bindings(wrong_trace,9,1,wrong_addresses,wrong_tiers));negatives+=1
  frame={'schema':1,'pid':9,'request':1,'stage':0,'layer':0,'rows':1,'position':0,'token':7,'reused':0,'binding_sha256':binding,'gen_ids':[7],'graph_key':2,'completed_nonce':(9<<32)|1,'request_replay_marker_verified':True,'raw_fused_hidden_observed':False,'complete_preregistered_layout':True,'producer_mapping_verified':True,'full_model_math_qualified':False,'expert_blob_addresses':pointers,'expert_tiers':tiers,'fields':[]}
  for contract in q.field_contract():
   item=dict(contract,file='/results/layer0/'+contract['name']+'.bin');data=b'\0'*contract['bytes']
   if contract['name']=='router_ids':data=struct.pack('<10i',*range(10))
   if contract['name']=='expert_entry_map':data=struct.pack('<30i',*[v for i in range(10) for v in (i,0,i)])
   (root/'layer0'/Path(item['file']).name).write_bytes(data);frame['fields'].append(item)
  metadata=root/'layer0/frame.json';q.write(metadata,frame);(root/'engine.combined.log').write_text(trace+end)
  raw={'ids':[7],'stderr':['SFD request pid=9 request=1 tokens=1 ids=7','L0Q8 frame pid=9 request=1 metadata=/results/layer0/frame.json']}
  checked=q.collect_frame(raw,root,binding);assert checked['source_value_fields']==31 and checked['derived_value_fields']==2 and len(checked['observed'])==33
  for key,value in [('completed_nonce',0),('graph_key',1),('producer_mapping_verified',False),('raw_fused_hidden_observed',True),('full_model_math_qualified',True),('request',2),('position',1),('reused',1),('expert_tiers',[0]*10),('expert_blob_addresses',[pointers[0]]*10)]:
   bad=copy.deepcopy(frame);bad[key]=value;q.write(metadata,bad);reject(lambda:q.collect_frame(raw,root,binding));negatives+=1
  bad=copy.deepcopy(frame);bad['fields'][25]['provenance']='actual_buffer';q.write(metadata,bad);reject(lambda:q.collect_frame(raw,root,binding));negatives+=1
  q.write(metadata,frame);mapping=root/'layer0/expert_entry_map.bin';mapping.write_bytes(struct.pack('<30i',*([0]*30)));reject(lambda:q.collect_frame(raw,root,binding));negatives+=1
  # Both pages tested with temporary synthetic source only, never real shards.
  pagefile=root/'pages';a=b'A'*4096;b=b'B'*4096;pagefile.write_bytes(a+b);pages=((0,hashlib.sha256(a).hexdigest()),(4096,hashlib.sha256(b).hexdigest()));assert inspect_pages(pagefile,pages)['passed']
  for offset in (0,4096):
   data=bytearray(a+b);data[offset+2796]^=0x20;pagefile.write_bytes(data);assert not inspect_pages(pagefile,pages)['passed'];negatives+=1
  assert [x[0] for x in KNOWN_PAGES]==[3857879040,39437303808]
  # Five packet seams across four synthetic prefixes, source file SHA bound.
  rows=[]
  for prefix in [1,2,4,8]:
   fields={c['name']:{'name':c['name'],'path':str(root/f'{prefix}-{c["name"]}'),'sha256':''} for c in q.field_contract()}
   payloads={c['name']:b'\0'*c['bytes'] for c in q.field_contract()}
   for source,packet,count,cols in PAIRS:
    x=np.linspace(-1,1,count*cols,dtype=np.float32).reshape(count,cols)
    if source.endswith('_DERIVED'):x[:]=0
    payloads[source]=x.astype('<f4').tobytes();payloads[packet]=encode(payloads[source],count,cols)
   for key,data in payloads.items():Path(fields[key]['path']).write_bytes(data);fields[key]['sha256']=hashlib.sha256(data).hexdigest()
   rows.append({'prefix':prefix,'layer0':{'observed':list(fields.values())}})
  gu=np.linspace(-3,3,1280,dtype=np.float32).reshape(1,2,640);g=gu[:,0];u=gu[:,1];hidden=((g/(1+np.exp(-g)))*u).astype('<f4');assert hidden_metrics(gu.astype('<f4').tobytes(),hidden.tobytes(),1)['raw_hidden_proven'] is False
  reject(lambda:hidden_metrics(gu.astype('<f4').tobytes(),(hidden+1).astype('<f4').tobytes(),1));negatives+=1
  packets=verify(rows);assert packets['passed'] and len(packets['results'])==4 and packets['raw_fused_hidden_proven'] is False
  # An intrinsic-near-half-step hidden change can alter exact Q8 codes even
  # while passing loose float error gates. Packet equality remains mandatory.
  h=np.zeros(32,dtype='<f4');h[0]=127;h[1]=np.nextafter(np.float32(.5),np.float32(0));other=h.copy();other[1]=np.float32(.5)
  assert np.max(np.abs(h-other))<1e-6 and encode(h.tobytes(),1,32)!=encode(other.tobytes(),1,32)
  reject(lambda:require_packet_exact(h.tobytes(),encode(other.tobytes(),1,32),1,32));negatives+=1
  manifest={'runtime':{'python_sources':{}}}
  for name in q.API_PYTHON_SOURCES:
   f=root/'source'/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(name);manifest['runtime']['python_sources'][name]=q.sha(f)
  assert len(q.api_source_binding(root,manifest))==6
  bad=copy.deepcopy(manifest);del bad['runtime']['python_sources']['serve/batch_request_identity.py'];reject(lambda:q.api_source_binding(root,bad));negatives+=1
  bad=copy.deepcopy(manifest);bad['runtime']['python_sources']['serve/batch_request_identity.py']='0'*64;reject(lambda:q.api_source_binding(root,bad));negatives+=1
  legacy=root/'legacy-upload.json';q.write(legacy,{'passed':True});reject(lambda:q.c1.strict_v2_upload_provenance(legacy,root/'missing-oracle',root/'missing-engine'));negatives+=1
  # Missing SDK receipts fail before C1 validation or any source payload read.
  with patch.object(q.c1,'validate_prepared',side_effect=AssertionError('must not read model')):
   reject(lambda:q.engine_binding(root/'absent-sdk',q.PLAN_SOURCE_SHA));negatives+=1
 result={'CONFIG':'synthetic33fields/31source2DERIVED,10tier pointers,20packets,two temporarypages; no model payload/GPU/SDK','COMMAND':'python3 strata/flash-next/test_layer0_numerical_qualification_cpu_v4.py','RESULT':{'negative_controls':negatives,'five_packet_seams_four_prefixes':True,'intrinsic_rounding_exact_packet_failure':True,'complete7023304_owner_extent':True,'both_known_page_controls':True,'numpy':np.__version__},'VERDICT':'PASS CPU contracts only; actual combined SDK/C1/preparation/packet/operator/source/current page qualification absent','full_model_math_qualified':False}
 q.write(Path(__file__).with_name('layer0-numerical-orchestration-cpu-receipt-v4.json'),result);print('PASS synthetic V4 controls; no model payload/GPU/math qualification')
if __name__=='__main__':main()
