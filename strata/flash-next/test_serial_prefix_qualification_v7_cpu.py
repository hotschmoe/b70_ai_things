#!/usr/bin/env python3
"""CPU metadata negative controls, true patched-source binding, tokenizer-only fixture."""
import array,copy,hashlib,json,subprocess,tempfile
from pathlib import Path
import serial_prefix_qualification_v7 as q
ROOT=q.ROOT

def rejected(fn):
 try:fn()
 except (ValueError,IndexError,KeyError,AssertionError):return
 raise AssertionError('Negative control accepted')

def main():
 with tempfile.TemporaryDirectory(prefix='prefix-v6-cpu-') as name:
  directory=Path(name);cap=directory/'captures';cap.mkdir();ranges=[(0,24),(24,48)]
  (cap/'logits.f32').write_bytes(array.array('f',[1.0]*248320).tobytes())
  for layer in range(48):(cap/f'layer{layer}.f32').write_bytes(array.array('f',[2.0]*10240).tobytes())
  request={'ids':[11,12,13],'fresh':1,'stderr':[],'output_ids':[17,18],'LP':['LP -0.1 17:-0.1'],'done':'DONE 2 3 0 0 stop','cancel_requested':None}
  ledger=[dict(event='selection',stages=2),dict(event='evaluated_span',lo=0,hi=3,prompt_rows=3),dict(event='finish',prompt_tokens=3,actual_reused=0,candidate_resume=0,evaluated_prompt_rows=3,stage_prompt_rows=6,cancelled=False)]
  request['stderr']=['PREFIX_DIAG '+json.dumps(e) for e in ledger]+['SFD request pid=9 request=1 tokens=3 ids=11,12,13 shape=4x2560','SFD vector pid=9 request=1 stage=1 lb=24 le=48 phase=first_logits_before_sampler layer=-1 pos=2 token=13 bytes=993280 file=/results/captures/logits.f32']
  for layer in range(48):request['stderr'].append('SFD vector pid=9 request=1 stage=%d lb=%d le=%d phase=first_window_residual layer=%d pos=2 token=13 bytes=40960 file=/results/captures/layer%d.f32'%(layer//24,layer//24*24,(layer//24+1)*24,layer,layer))
  begin=dict(event='begin',pid=9,request=1,tokens=3,input_sha256_le32=q.le32_digest(request['ids']))
  commit=dict(event='committed_live',pid=9,request=1,tokens=4,ids=[11,12,13,17],sha256_le32=q.le32_digest([11,12,13,17]),ids_truncated=False,phase='complete',finish='stop',published=True,chain_updated=True,live_reusable=True)
  spans=[dict(event='stage_span',pid=9,request=1,device=stage,lb=bounds[0],le=bounds[1],lo=0,hi=3,complete=True,phase='verify_body',commit_proven=False) for stage,bounds in enumerate(ranges)]
  request['stderr']+=['PCL '+json.dumps(e) for e in [begin,*[dict(e,complete=False) for e in spans],*spans,commit]]
  assert q.ARMED_REQUESTS['real_live']==4 and q.ARMED_REQUESTS['cancel_decode']==6 and q.ARMED_REQUESTS['cancel_prefill_isolation']==6 and all(0<n<=6 for n in q.ARMED_REQUESTS.values())
  meta=q.extract(request,cap,True,True,ranges);assert meta['lifecycle']['body_completion_only']
  bad=copy.deepcopy(request);bad['stderr']=[line for line in bad['stderr'] if not line.startswith('PCL ')];rejected(lambda:q.extract(bad,cap,True,True,ranges))
  bad=copy.deepcopy(request);bad['stderr']=[line for line in bad['stderr'] if '"device": 1' not in line];rejected(lambda:q.extract(bad,cap,True,True,ranges))
  bad=copy.deepcopy(request);wrong=copy.deepcopy(commit);wrong['ids'][-1]=19;bad['stderr'][-1]='PCL '+json.dumps(wrong);rejected(lambda:q.extract(bad,cap,True,True,ranges))
  bad=copy.deepcopy(request);wrong=copy.deepcopy(commit);wrong['live_reusable']=False;bad['stderr'][-1]='PCL '+json.dumps(wrong);rejected(lambda:q.extract(bad,cap,True,True,ranges))
  bad=copy.deepcopy(request);bad['cancel_requested']='prefill';wrong=copy.deepcopy(commit);wrong['phase']='decode';wrong['finish']='cancel';bad['stderr'][2]='PREFIX_DIAG '+json.dumps(dict(ledger[-1],cancelled=True));bad['stderr'][-1]='PCL '+json.dumps(wrong);rejected(lambda:q.extract(bad,cap,True,True,ranges))
  bad=copy.deepcopy(request);bad['stderr']=[line.replace('\"device\": 1','\"device\": 0') if line.startswith('PCL ') else line for line in bad['stderr']];rejected(lambda:q.extract(bad,cap,True,True,ranges))
  target=meta['lifecycle']['commit']['sha256_le32'];cache=dict(event='cache',pid=9,request=1,action='admit',source_tokens_sha256_le32=target,snapshot_instance=7,snapshot_metadata_sha256='a'*64,identity_observed=True,budget=512<<20,slots=1,retained_bytes=1024,entries=1,held=0,stage_ranges=[[0,24],[24,48]],stage_descriptor_truncated=False)
  trace=[dict(meta={'lifecycle':{'cache_events':[cache,dict(cache,action='evict',entries=0,retained_bytes=0)]}})]
  q.keyed_eviction(trace,meta,dict(ledger={'actual_reused':0}))
  rejected(lambda:q.keyed_eviction([],meta,dict(ledger={'actual_reused':0})))
  wrong=copy.deepcopy(trace);wrong[0]['meta']['lifecycle']['cache_events'][-1]['snapshot_instance']=8;rejected(lambda:q.keyed_eviction(wrong,meta,dict(ledger={'actual_reused':0})))
  bad=copy.deepcopy(request);bad['stderr'].insert(-1,'PCL '+json.dumps(dict(cache,retained_bytes=(512<<20)+1)));rejected(lambda:q.extract(bad,cap,True,True,ranges))
  # Frozen math/lifecycle parsing and serial groups are kept deliberately intact.
  import ast
  old=ast.parse((ROOT/'strata/flash-next/serial_prefix_qualification_v6.py').read_text());new=ast.parse(Path(q.__file__).read_text())
  for name in ['extract_numeric','extract','le32_digest','keyed_eviction','continue_fixture','compare_vectors','compare_requests','decode_output','topology_args']:
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
  env={'STRATA_STAGE_MIRRORS':'1','STRATA_STAGE_MIRROR_SEGMENT_MIB':'1024'};assert q.observer_env_contract(env)
  for key,value in [('STRATA_VERIFY_EAGER','0'),('STRATA_VERIFY_EAGER','1'),('STRATA_PREFIX30','1'),('STRATA_PLE_INPUT33','1'),('STRATA_BATCH_PUBLIC_PREFIX','1'),('STRATA_CKPT_REREAD','0'),('STRATA_STATE_HASH','0'),('STRATA_STAGE_MIRROR_SEGMENT_MIB','512')]:
   rejected(lambda:q.observer_env_contract(dict(env,**{key:value})))
  from unittest.mock import patch
  generation={'mock':'SOURCE35_ONLY'};baseline={'combined_generation':generation,'engine_receipt_sha256':'a'*64}
  with patch.object(q.c1,'combined_generation_gate',lambda p:generation),patch.object(q,'sha',lambda p:'a'*64),patch.object(q,'read',lambda p:{'mock':'compiled receipt'}):
   assert q.engine_identity(directory,baseline)=={'mock':'compiled receipt'}
   rejected(lambda:q.engine_identity(directory,dict(baseline,combined_generation={'mock':'OLD_SOURCE33'})))
   rejected(lambda:q.engine_identity(directory,dict(baseline,engine_receipt_sha256='b'*64)))
  with patch.object(q,'read',lambda p:{'schema':6,'passed':True}):
   rejected(lambda:q.basic_receipt_binding(directory/'legacy.json',{}))
  # Exercise the real prerequisite reader with tiny synthetic lifecycle artifacts.
  import shutil,time
  parentdir=directory/'basic-parent';child=parentdir/'child';child.mkdir(parents=True)
  sourceplan=q.c1.COMBINED_PLAN_SHA
  basicplan={'engine_receipt_sha256':'a'*64,'combined_plan_sha256':sourceplan,'controller_sha256':q.sha(Path(q.__file__)),'matched_geometry':{'CPU_SYNTHETIC':True},'cards':[0,1],'args':['--layer-split','24','--split-device','1'],'tokens':{'cases':{k:request['ids'] for k in ['A','B','C']}},'image':'CPU_SYNTHETIC','prepared':str(directory/'prepared')}
  q.write(child/'plan.snapshot.json',basicplan);q.write(parentdir/'input-plan.snapshot.json',basicplan)
  groups=[];collected=[]
  for group,diag,acts in [('basic_off',False,False),('basic_logits',True,False),('basic_layers',True,True)]:
   out=child/group;out.mkdir();shutil.copytree(cap,out/'captures');rows=[]
   for key in ['A','B','C']:
    raw=copy.deepcopy(request);raw['label']=key
    if not diag:raw['stderr']=[v for v in raw['stderr'] if not v.startswith(('SFD ','PCL '))]
    elif not acts:raw['stderr']=[v for v in raw['stderr'] if 'phase=first_window_residual' not in v]
    rows.append({'raw':raw,'meta':q.extract(raw,out/'captures',diag,acts,ranges)})
   q.write(out/'requests.json',rows);collected.append(rows);groups.append({'name':group,'diagnostic':int(diag),'activations':int(acts),'passed':True})
  eq=[]
  for off,logits,layers in zip(*collected):
   eq.append({'label':off['raw']['label'],'flag0_on':q.compare_requests(off['raw'],logits['raw'],off['meta'],logits['meta'],False),'activations0_1':q.compare_requests(logits['raw'],layers['raw'],logits['meta'],layers['meta'])})
  q.write(child/'basic-equivalence.json',eq)
  health={'passed':True,'finished_epoch':2,'files':[]}
  for name in ['strict','pair']:
   path=parentdir/(name+'.log');path.write_text('CPU_SYNTHETIC');health['files'].append({'path':str(path),'sha256':q.sha(path)})
  q.write(parentdir/'post-health.json',health)
  parent={'passed':True,'child_return_code':0,'owned_containers_terminal':True,'forced_cleanup':False,'post_health_passed':True,'kernel_fault_gate_passed':True,'errors':[],'controller_sha256':basicplan['controller_sha256'],'plan_sha256':q.sha(parentdir/'input-plan.snapshot.json'),'child_terminal_epoch':1,'known_pages_before_hash':{'passed':True,'rows':[{},{}],'epoch':3},'known_pages_after_hash':{'passed':True,'rows':[{},{}],'epoch':6},'wrapper_sha256':q.sha(ROOT/'strata/flash-next/qualify_serial_prefix_v7.py'),'pre_health_passed':True,'interrupted':False}
  q.write(parentdir/'parent-qualification.json',parent);q.write(parentdir/'post-model-identity.json',{'passed':True,'rows':[{}]*4,'started':4,'finished':5})
  basic={'schema':7,'group':'basic','passed':True,'post_health_passed':True,'numerical_and_teardown_passed':True,'plan_sha256':q.sha(child/'plan.snapshot.json'),'results':groups,'post_health_sha256':q.sha(parentdir/'post-health.json')};q.write(child/'report.json',basic)
  with patch.object(q,'source35_admission',lambda path:({'model_shards':[]},{})),patch.object(q,'verify_model_identity',lambda *args:{}):
   assert q.basic_receipt_binding(child/'report.json',basicplan)==basic
   wrong=copy.deepcopy(eq);wrong[0]['flag0_on']['passed']=False;q.write(child/'basic-equivalence.json',wrong);rejected(lambda:q.basic_receipt_binding(child/'report.json',basicplan));q.write(child/'basic-equivalence.json',eq)
   wrong=copy.deepcopy(parent);wrong['owned_containers_terminal']=False;q.write(parentdir/'parent-qualification.json',wrong);rejected(lambda:q.basic_receipt_binding(child/'report.json',basicplan));q.write(parentdir/'parent-qualification.json',parent)
   path=child/'basic_layers'/'requests.json';q.write(path,collected[-1][:-1]);rejected(lambda:q.basic_receipt_binding(child/'report.json',basicplan));q.write(path,collected[-1])
   (parentdir/'pair.log').write_text('TAMPERED');rejected(lambda:q.basic_receipt_binding(child/'report.json',basicplan))
  assert q.ARMED_REQUESTS=={'basic':3,'root':6,'pin':6,'turn':6,'parked':6,'eviction':6,'cancel':6,'cancel_decode':6,'cancel_prefill_isolation':6,'real_live':4}
  text=Path(q.__file__).read_text();assert "if a.group!='basic':" in text and 'basic_receipt_binding(a.one_card_receipt,plan,False)' in text
  print('PASS source35 V7 CPU controls: actual-work spans, all48/fullhead coverage, lifecycle identity, cancellation phase, exact eviction victim, source35 engine binding, eager0/observer rejection, legacy basic rejection, frozen parser/math helpers unchanged. No runtime/model/GPU/Docker touched.')
if __name__=='__main__':main()
