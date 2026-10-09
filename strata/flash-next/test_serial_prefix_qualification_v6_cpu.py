#!/usr/bin/env python3
"""CPU metadata negative controls, true patched-source binding, tokenizer-only fixture."""
import array,copy,hashlib,json,subprocess,tempfile
from pathlib import Path
import serial_prefix_qualification_v6 as q
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
  # Positive engine_identity path uses real staged source bytes and explicitly
  # synthetic build receipt/executable. This is not a compiled0019 engine gate.
  plan=q.read(ROOT/'strata/flash-next/prefix-lifecycle-source-plan.json');base=Path(plan['base_source']).parent;receipt=q.read(base/'receipt.json');fake=directory/'mock-engine';(fake/'build').mkdir(parents=True);(fake/'build/strata').write_text('CPU protocol-only identity stub\n')
  import shutil
  shutil.copytree(Path(plan['overlay']),fake/'source');receipt['patched_source_sha256']=plan['patched_files'];receipt['binary_sha256']={str(fake/'build/strata'):q.sha(fake/'build/strata')};receipt['patches'].append(dict(path=str(ROOT/plan['patch']),sha256=plan['patch_sha256']));q.write(fake/'receipt.json',receipt)
  q.engine_identity(fake,{'engine_receipt':str(base/'receipt.json')})
  rejected(lambda:q.engine_identity(base,{'engine_receipt':str(base/'receipt.json')}))
  # Actual tokenizer/template, CPU-only pinned runtime, synthetic natural seed.
  prepared=q.read(Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/c1-onecard-corrected-streams-prepared-v2/prepared.json'))
  oldplan=q.read(Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/serial-prefix-onecard-v5-prepared-v1/plan.json'))
  # The old exported seed prompt is unchanged by V6. Derive the canonical live
  # token prefix from exact template plus a synthetic reply; no model execution.
  fixture=ROOT/'strata/flash-next/serial_prefix_prompt_fixture_v6.py'
  code='import sys,json;sys.path[:0]=["/src/tools","/src/serve"];from strata_tokenizer import Tokenizer;from frontend import ChatTemplate;from pathlib import Path;p=Path("/pack/tokenizer");v=json.loads((p/"vocab.json").read_text());t=[None]*len(v);[(t.__setitem__(i,k)) for k,i in v.items()];cfg=json.loads((p/"tokenizer.json").read_text());tok=Tokenizer(t,(p/"merges.txt").read_text().split("\\n"),json.loads((p/"token_type.json").read_text()),cfg["pre"],cfg["special_ids"]);print(json.dumps(tok.encode("42")))'
  command=['docker','run','--rm','--network','none','--user','1000:1000','-v',str(base/'source')+':/src:ro','-v',prepared['pack']+':/pack:ro',prepared['runtime']['image'],'exec /opt/b70-c1-python/bin/python -c '+__import__('shlex').quote(code)]
  answer=json.loads(subprocess.check_output(command,text=True));seed_ids=oldplan['tokens']['cases']['A'];seed=dict(ids=seed_ids,output_ids=answer+[q.read(Path(prepared['pack'])/'tokenizer/tokenizer.json')['special_ids']['tokenizer.ggml.eos_token_id']],done='DONE 2 3 0 0 stop',decoded_output_without_special_tokens='42',lifecycle_committed_ids=seed_ids+answer)
  continuation=q.continue_fixture({'engine_root':str(base),'pack':prepared['pack'],'image':prepared['runtime']['image'],'tokens':oldplan['tokens']},seed,directory);assert continuation['expected_live_reuse']==len(seed['lifecycle_committed_ids'])
  wrong=copy.deepcopy(seed);wrong['lifecycle_committed_ids'][0]+=1
  payload=directory/'bad-seed.json';q.write(payload,wrong)
  badcommand=['docker','run','--rm','--network','none','--user','1000:1000','-v',str(base/'source')+':/src:ro','-v',prepared['pack']+':/pack:ro','-v',str(fixture)+':/fixture.py:ro','-v',str(payload)+':/seed.json:ro',prepared['runtime']['image'],'exec /opt/b70-c1-python/bin/python /fixture.py --continue-json /seed.json']
  result=subprocess.run(badcommand,capture_output=True,text=True);assert result.returncode!=0
  result={'CONFIG':'CPU synthetic raw/lifecycle/receipt controls and actual tokenizer/template runtime, no devices/model','COMMAND':'python3 strata/flash-next/test_serial_prefix_qualification_v6_cpu.py','RESULT':{'strict_raw_and_lifecycle_positive':True,'missing_wrong_phase_stage_digest_budget_victim_rejected':True,'new_source_positive_mock_engine_identity':True,'old0018_generation_rejected':True,'actual_tokenizer_template_synthetic_continuation':True,'wrong_committed_prefix_rejected':True},'VERDICT':'PASS CPU contracts only;0019 compilation/full390/GPU/real generated continuation remain unqualified','synthetic_engine_receipt_explicit':True,'frozen_v5_sha256':q.sha(ROOT/'strata/flash-next/serial_prefix_qualification_v5.py')}
  q.write(ROOT/'strata/flash-next/serial-prefix-v6-cpu-receipt.json',result)
  print('PASS V6 CPU lifecycle/stage/phase/victim negative controls and actual tokenizer synthetic continuation; no GPU/model')
if __name__=='__main__':main()
