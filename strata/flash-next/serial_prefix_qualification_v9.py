#!/usr/bin/env python3
"""V9 remaining serial cache suite; corrected cold-establish parking protocol. Parent owns GPU health.
prepare is CPU-only. run is an explicit GPU action and checks inherited leases.
"""
import argparse,array,hashlib,json,math,os,queue,re,shlex,signal,subprocess,sys,threading,time
from pathlib import Path
import serial_prefix_qualification_v8 as v8
import c1_serve_controller_combined_v13 as c1
import qualify_c1_serving_combined_v13 as c113
from run_source_upload_oracle_full import verify_model_identity
from audit_fidelity_observer_coverage import coverage as strict_observer_coverage
ROOT=Path(__file__).resolve().parents[2]
COVERAGE_SHA='c23f73131b4d525c7e322f913ae6f0b600b07073c0125b9d1d66824725dbec47'
def sha(p):return c1.sha(p)
def read(p):return c1.read(p)
def require(ok,msg):c1.require(ok,msg)
def write(p,j):c1.write(p,j)

ARMED_REQUESTS={'basic':3,'root':6,'pin':6,'turn':6,'parked':6,'eviction':6,'cancel':6,'cancel_decode':6,'cancel_prefill_isolation':6,'real_live':4}

GEOMETRY={'--max-context':'2048','--prefill':'64','--ple-row-cache':'65536','--kv':'fp16','--prompt-cache':'3','--prompt-cache-root':'1','--prompt-cache-every':'1000000000','--batch':'0','--suffix-draft':'0','--lookup-chain':'0','--adapt-swaps':'0','--adapt-every':'0'}
def geometry_contract(args):
 values={flag:args[args.index(flag)+1] for flag in GEOMETRY if flag in args}
 require(values==GEOMETRY,'Expected shared2048/64 bounded numerical geometry differs')
 require('--no-prefill-borrow' in args,'Matched geometry requires no prefill borrowing')
 return {'id':'serial-pilot-ctx2048-prefill64-ple65536-v1','values':values,'no_prefill_borrow':True,'production8192_qualified':False}

def expected_stage_ranges(args):
 if '--layer-split' not in args:return [(0,48)]
 cut=int(args[args.index('--layer-split')+1]);require(0<cut<48,'Expected layer split differs')
 require('--split-device' in args and args[args.index('--split-device')+1]=='1','Expected two-device split differs')
 return [(0,cut),(cut,48)]

def pilot_alias(cards,args):
 geometry_contract(args)
 bounds=expected_stage_ranges(args)
 require(len(cards)==len(bounds),'Pilot physical-card and stage roster differ')
 card_label='_'.join(map(str,cards));layer_label='_'.join(map(str,[bounds[0][0]]+[hi for lo,hi in bounds]))
 return ('qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-source-hc-ple-mtp0-fp16kv'
         '-pilotctx2048-pf64-ple65536-seg1024-adapt0-borrow0-prefix-diag0013-35-nativeGEN-v8-source35-pinAbsent'
         '-cards'+card_label+'-layers'+layer_label+'-lifecycle0019-v8-source35-v9-remaining-coldPark')

def observer_env_contract(env):
 require('STRATA_VERIFY_EAGER' not in env,'Normal graph requires STRATA_VERIFY_EAGER absent, including value0')
 require(not any(k in env for k in ('STRATA_CKPT_REREAD','STRATA_STATE_HASH')),'Diagnostic state reset/hash forbidden')
 for key in ('STRATA_BATCH_FULL_STATE_CHAIN','STRATA_BATCH_PUBLIC_PREFIX','STRATA_BATCH_FIDELITY_DIAG','STRATA_SLOT_OWNER_TRACE','STRATA_MIRROR_OWNER_TRACE','STRATA_LAYER0_Q8_DIAG','STRATA_PREFIX30','STRATA_PLE_INPUT33'):
  require(env.get(key,'0')=='0','Unmatched source35 observer/math flag: '+key)
 require(env.get('STRATA_STAGE_MIRRORS')=='1' and env.get('STRATA_STAGE_MIRROR_SEGMENT_MIB')=='1024','Matched segmented mirror route required')
 return True

def source35_admission(prepared):
 m=c1.validate_prepared(Path(prepared));c113.source_pins()
 proof=c113.validate_final_source_proof(Path(prepared),m)
 observer_env_contract(read(Path(prepared)/'server-config.json')['env'])
 return m,proof

def engine_identity(root,baseline):
 generation=c1.combined_generation_gate(root)
 require(generation==baseline['combined_generation'],'Prepared/actual source35 integrated generation differs')
 require(sha(root/'receipt.json')==baseline['engine_receipt_sha256'],'Prepared/actual source35 receipt differs')
 return read(root/'receipt.json')

def matched_upload_engine_gate(prepared,root):
 require(prepared['engine_receipt_sha256']==sha(root/'receipt.json') and prepared['executable_sha256']==sha(root/'build/strata'),'Source35 requires completed full390 source-upload gate for this exact SDK generation')

REMAINING_GROUPS=('turn','parked','eviction','cancel','cancel_decode','cancel_prefill_isolation','real_live')
V8_CONTROLLER_SHA='c0bac8dbfca4f84e9883db87855e3eea832c0d1422cd9e0d52c4ee103933a9bc'

def source_plan_binding():
 path=ROOT/'strata/flash-next/serial-prefix-source35-v9-source-plan.json';manifest=read(path)
 require(manifest['groups']==list(REMAINING_GROUPS),'V9 frozen source roster differs')
 for name,digest in manifest['files'].items():require(sha(ROOT/name)==digest,'V9 frozen source dependency changed '+name)
 return sha(path)

def prepare(a):
 """CPU-only clone of a frozen genuine V8 prepared plan; no new tokenizer run."""
 suite_source=source_plan_binding();base=read(a.base_plan)
 require(base['schema']==8 and base['controller_sha256']==V8_CONTROLLER_SHA==sha(Path(v8.__file__)),'Frozen genuine V8 base controller differs')
 require(base['token_fixture_sha256']==sha(ROOT/'strata/flash-next/serial_prefix_prompt_fixture_v6.py'),'Exact token fixture changed')
 require(base['coverage_parser_sha256']==COVERAGE_SHA==sha(ROOT/'strata/flash-next/audit_fidelity_observer_coverage.py'),'Coverage source changed')
 require(base['matched_geometry']==geometry_contract(base['args']),'Frozen base geometry differs')
 cases=base['tokens']['cases'];vocab=base['tokens']['vocab']
 require(vocab==248320 and all(k in cases for k in ['A','B','C','pin_A','pin_B','turn_A','turn_B','evict_A','evict_B','evict_C','evict_D']),'Complete static token corpus required')
 require(all(ids and len(ids)+base['max_new']<=2048 and all(type(i)is int and 0<=i<vocab for i in ids) for ids in cases.values()),'Token corpus extent/vocabulary differs')
 for key,left,right in [('root','A','B'),('pin','pin_A','pin_B'),('turn_shared','turn_A','turn_B')]:
  n=base['tokens'][key];require(0<n<min(len(cases[left]),len(cases[right])) and cases[left][:n]==cases[right][:n],'Declared shared prefix differs '+key)
 j=dict(base);j.update(suite_source_plan_sha256=suite_source,research_alias=pilot_alias(base['cards'],base['args']),schema=9,status='CPU prepared from genuine frozen V8; no GPU launch',controller_sha256=sha(Path(__file__)),base_v8_plan_path=str(a.base_plan.resolve()),base_v8_plan_sha256=sha(a.base_plan),base_v8_controller_sha256=V8_CONTROLLER_SHA,groups=list(REMAINING_GROUPS),client_protocol='V9 cold-establish fresh=0 pin=0 preserves outgoing parking; defaults absent pin')
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',j);print(json.dumps({'plan':str(a.output/'plan.json'),'GPU_executed':False}))


def preflight(plan,basic_receipt,one_card_receipt=None):
 require(plan['schema']==9 and plan['groups']==list(REMAINING_GROUPS) and plan['base_v8_controller_sha256']==V8_CONTROLLER_SHA,'Exact V9 remaining plan roster/source differs')
 require(plan['suite_source_plan_sha256']==source_plan_binding(),'V9 preregistered source plan changed')
 base_path=Path(plan['base_v8_plan_path']);require(sha(base_path)==plan['base_v8_plan_sha256'],'Genuine frozen V8 base plan changed')
 base=read(base_path);require(base['schema']==8 and base['controller_sha256']==V8_CONTROLLER_SHA==sha(Path(v8.__file__)),'V8 base controller/source differs')
 changed={'schema','status','controller_sha256','groups','client_protocol','research_alias'}
 require(all(plan[k]==value for k,value in base.items() if k not in changed),'V9 changed an undeclared V8 math/runtime/token field')
 require(plan['controller_sha256']==sha(Path(__file__)) and plan['research_alias']==pilot_alias(plan['cards'],plan['args']),'V9 controller/research alias differs')
 require(basic_receipt,'Finalized same-topology V8 basic required before any health')
 basic_receipt_binding(basic_receipt,plan,True)
 if len(plan['cards'])>1:
  require(one_card_receipt,'Finalized one-card V8 basic required before paired health')
  basic_receipt_binding(one_card_receipt,plan,False)
 return {'passed':True,'scope':'CPU V9/V8 plan and raw prerequisite admission; no GPU work','groups':list(REMAINING_GROUPS)}

def cold_recollected_binding(saved,fresh):
 require(json.dumps(saved,sort_keys=True)==json.dumps(fresh,sort_keys=True),'Postterminal cold raw/vector metadata changed')
 return True

def cold_establish_binding(raw,meta):
 require(raw['fresh']==0 and raw['pin']==0 and raw['command']==request_command(raw['ids'],64,0,0),'Cold-establish exact nonfresh pin-zero protocol differs')
 require(meta['ledger']['actual_reused']==0 and meta['ledger']['evaluated_prompt_rows']==len(raw['ids']) and meta['selection']['candidate_resume']==0 and meta['selection']['actual_reused']==0,'Cold-establish reused/read-from-zero coverage differs')
 return True

def request_command(ids,max_new=64,fresh=0,pin=None):
 require(type(max_new) is int and max_new>0 and type(fresh) is int and fresh in (0,1),'GEN requires positive max_new and fresh0|1')
 require(ids and all(type(token) is int and 0<=token<248320 for token in ids),'GEN requires exact model token IDs')
 require(pin is None or (type(pin) is int and 0<=pin<len(ids)),'Optional pin must be absent or explicit decimal0..prompt-1')
 keys='GEN %d fresh=%d '%(max_new,fresh)
 if pin is not None:keys+='pin=%d '%pin
 return keys+'ckpt=1 temperature=0 logprobs=20 '+' '.join(map(str,ids))

class Protocol:
 def __init__(self,command,directory,ready_timeout=900):
  self.directory=directory;self.stdout=[];self.stderr=[];self.q=queue.Queue();self.log=directory/'engine.stderr.log';self.err=self.log.open('w')
  self.p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,start_new_session=True)
  self.threads=[]
  def pump(stream,rows,events):
   for line in stream:
    line=line.rstrip('\n');rows.append(line)
    if events:self.q.put((time.monotonic(),line))
    else:self.err.write(line+'\n');self.err.flush()
   if events:self.q.put((time.monotonic(),None))
  for stream,rows,events in [(self.p.stdout,self.stdout,True),(self.p.stderr,self.stderr,False)]:
   t=threading.Thread(target=pump,args=(stream,rows,events),daemon=True);t.start();self.threads.append(t)
  deadline=time.monotonic()+ready_timeout
  while True:
   _,line=self.q.get(timeout=max(.001,deadline-time.monotonic()))
   require(line is not None,'Engine ended before readiness')
   if line.startswith('ERR'):raise ValueError(line)
   if line.startswith('READY'):self.ready=line;break
 def send(self,line):self.p.stdin.write(line+'\n');self.p.stdin.flush()
 def request(self,label,ids,fresh=0,pin=None,cancel=None,max_new=64):
  start=time.monotonic();err_start=len(self.stderr);rows=[];times=[];stopped=False
  command=request_command(ids,max_new,fresh,pin);self.send(command)
  deadline=start+900
  while True:
   stamp,line=self.q.get(timeout=max(.001,deadline-time.monotonic()));require(line is not None,'Unexpected engine EOF')
   rows.append(line);times.append(stamp-start)
   if line.startswith('ERR'):raise ValueError(line)
   if cancel and not stopped and ((cancel=='prefill' and line.startswith('PP ')) or (cancel=='decode' and line.startswith('T '))):self.send('STOP');stopped=True
   if line.startswith('DONE '):break
  # DONE follows all diagnostic writes; reader thread may lag without GPU effect.
  for _ in range(200 if (self.directory/'ARM').exists() else 0):
   if any('"event":"finish"' in x for x in self.stderr[err_start:]):break
   time.sleep(.005)
  raw={'label':label,'ids':ids,'fresh':fresh,'pin':pin,'cancel_requested':cancel,'stop_sent':stopped,'command':command,'stdout':rows,'stdout_seconds':times,'stderr':self.stderr[err_start:],'output_ids':[int(x.split()[1]) for x in rows if x.startswith('T ')],'LP':[x for x in rows if x.startswith('LP ')],'done':rows[-1],'ready':self.ready}
  write(self.directory/(label+'.json'),raw);return raw
 def close(self):
  if self.p.poll() is None:self.send('QUIT')
  try:rc=self.p.wait(timeout=90)
  except subprocess.TimeoutExpired:self.p.terminate();self.p.wait(timeout=20);rc=self.p.returncode
  for t in self.threads:t.join(timeout=5)
  self.err.close();(self.directory/'engine.stdout.log').write_text('\n'.join(self.stdout)+'\n',encoding='ascii')
  return rc

def extract_numeric(request,capture_dir,require_vectors=True,require_activations=False,stage_ranges=None):
 led=[json.loads(x[len('PREFIX_DIAG '):]) for x in request['stderr'] if x.startswith('PREFIX_DIAG ')]
 finish=[x for x in led if x.get('event')=='finish'];require(len(finish)==1,'Missing/ambiguous completed work ledger')
 f=finish[0];require(f['prompt_tokens']==len(request['ids']),'Prompt length differs')
 selection=[x for x in led if x.get('event')=='selection'];require(len(selection)==1,'Missing/ambiguous selection')
 if stage_ranges is not None:require(selection[0]['stages']==len(stage_ranges),'Selected engine stage count differs from configured topology')
 spans=[x for x in led if x.get('event')=='evaluated_span'];require(sum(x['prompt_rows'] for x in spans)==f['evaluated_prompt_rows'],'Ledger total differs from spans')
 if not f['cancelled']:
  require(f['actual_reused']+f['evaluated_prompt_rows']==len(request['ids']),'Actual evaluated/reused prompt coverage incomplete')
  covered=[i for x in spans for i in range(max(0,x['lo']),min(len(request['ids']),x['hi']))]
  require(covered==list(range(f['actual_reused'],len(request['ids']))),'Prompt spans overlap or leave gaps')
  require(f['stage_prompt_rows']==f['evaluated_prompt_rows']*selection[0]['stages'],'Stage work multiplier differs')
 if request['fresh']:require(f['actual_reused']==0,'fresh did not reset work')
 vectors=[]
 begins=[x for x in request['stderr'] if x.startswith('SFD request ')]
 if require_vectors:
  require(len(begins)==1,'Missing/ambiguous fidelity request identity')
  fields=dict(re.findall(r'(\w+)=([^ ]+)',begins[0]));require([int(x) for x in fields['ids'].split(',')]==request['ids'],'Captured source input IDs differ')
 for line in request['stderr']:
  if not line.startswith('SFD vector '):continue
  row=dict(re.findall(r'(\w+)=([^ ]+)',line));rel=Path(row['file']).name;p=capture_dir/rel
  require(p.is_file() and not p.is_symlink() and p.stat().st_size==int(row['bytes']),'Raw vector missing/truncated')
  require(not require_vectors or row['request']==fields['request'] and row['pid']==fields['pid'],'Vector request ownership differs')
  if stage_ranges is not None:
   stage=int(row['stage']);require(0<=stage<len(stage_ranges) and (int(row['lb']),int(row['le']))==stage_ranges[stage],'Raw vector actual configured owner range differs')
  require(0<=int(row['pos'])<len(request['ids']) and int(row['token'])==request['ids'][int(row['pos'])],'Raw vector source token/position differs')
  floats=array.array('f');floats.frombytes(p.read_bytes())
  if sys.byteorder!='little':floats.byteswap()
  require(all(math.isfinite(v) for v in floats),'Captured nonfinite vector')
  row['path']=str(p);row['sha256']=sha(p);vectors.append(row)
 require(len({v['path'] for v in vectors})==len(vectors),'Raw capture file reused under another descriptor')
 logits=[x for x in vectors if x['phase']=='first_logits_before_sampler'];
 if require_vectors and not f['cancelled']:require(len(logits)==1 and int(logits[0]['bytes'])==248320*4 and int(logits[0]['pos'])==len(request['ids'])-1 and int(logits[0]['token'])==request['ids'][-1],'Full first-token logit coverage/position differs')
 if logits and not f['cancelled']:
  values=array.array('f');values.frombytes(Path(logits[0]['path']).read_bytes())
  if sys.byteorder!='little':values.byteswap()
  require(request['output_ids'] and 0<=request['output_ids'][0]<248320 and values[request['output_ids'][0]]==max(values),'Captured first logits do not predict emitted greedy token')
 if logits:
  require(int(logits[0]['stage'])==selection[0]['stages']-1 and int(logits[0]['le'])==48,'Full vocabulary head stage quota differs')
  if stage_ranges is not None:require((int(logits[0]['lb']),int(logits[0]['le']))==stage_ranges[-1],'Full vocabulary head stage bounds differ')
 residuals=[x for x in vectors if x['phase']=='first_window_residual'];
 if require_activations and not f['cancelled']:require(len(residuals)==48,'Activations1 requires all48 first-window layers; empty/partial coverage is unobserved, not equal')
 if residuals:
  require(sorted(int(x['layer']) for x in residuals)==list(range(48)) and all(int(x['bytes'])==10240*4 for x in residuals),'Complete first-window layer residual coverage differs')
  descriptors={int(x['stage']):(int(x['lb']),int(x['le'])) for x in residuals}
  require(sorted(descriptors)==list(range(selection[0]['stages'])),'Captured stage count differs')
  if stage_ranges is not None:require([descriptors[i] for i in sorted(descriptors)]==stage_ranges,'Captured stage owner bounds differ from actual engine arguments')
  cursor=0
  for stage,(lo,hi) in sorted(descriptors.items()):require(lo==cursor and lo<hi<=48,'Stage layer bounds overlap or leave gap');cursor=hi
  require(cursor==48 and all(descriptors[int(x['stage'])]==(int(x['lb']),int(x['le'])) and int(x['lb'])<=int(x['layer'])<int(x['le']) and int(x['pos'])==len(request['ids'])-1 and int(x['token'])==request['ids'][-1] for x in residuals),'Layer residual identity/ownership differs')
  if logits:require(int(logits[0]['stage'])==len(descriptors)-1 and int(logits[0]['le'])==48,'Logit head stage differs')
 coverage=None
 if require_vectors:
  ranges=stage_ranges if stage_ranges is not None else [(0,48)] if selection[0]['stages']==1 else None
  require(ranges is not None,'Explicit split-stage ranges required for armed coverage')
  coverage=strict_observer_coverage({'requests':[fields],'vectors':vectors},1,require_activations,dict(enumerate(ranges)),[int(fields['request'])] if f['cancelled'] else [])
  require(coverage['passed'],'Strict observer coverage failed: '+str(coverage['errors']))
 return {'coverage':coverage,'ledger':f,'selection':selection[0],'spans':spans,'vectors':vectors,'logits':logits,'residuals':residuals}

def le32_digest(ids):
 import struct
 return hashlib.sha256(b''.join(struct.pack('<I',int(t)) for t in ids)).hexdigest()

def extract(request,capture_dir,require_vectors=True,require_activations=False,stage_ranges=None):
 result=extract_numeric(request,capture_dir,require_vectors,require_activations,stage_ranges)
 if not require_vectors:
  require(not any(line.startswith('PCL ') for line in request['stderr']),'Lifecycle flag0 unexpectedly produced captures')
  result['lifecycle']={'observed':False};return result
 events=[json.loads(line[4:]) for line in request['stderr'] if line.startswith('PCL ')]
 begins=[e for e in events if e['event']=='begin'];commits=[e for e in events if e['event']=='committed_live']
 require(len(begins)==1 and len(commits)==1,'Missing/ambiguous lifecycle begin/commit provenance')
 begin=begins[0];commit=commits[0]
 require(begin['input_sha256_le32']==le32_digest(request['ids']) and begin['tokens']==len(request['ids']),'Lifecycle actual input differs')
 for event in events:
  require(event['pid']==begin['pid'] and event['request']==begin['request'],'Lifecycle event request/PID differs')
  require(event['event']!='skip','Lifecycle observer quota exceeded; coverage unobserved')
 require(not commit['ids_truncated'] and commit['tokens']==len(commit['ids']) and commit['sha256_le32']==le32_digest(commit['ids']),'Committed prefix token provenance differs')
 require(commit['published']==(commit['chain_updated'] and commit['live_reusable']),'Published/updated/reusable semantics differ')
 if result['ledger']['cancelled']:
  require(commit['phase']==request['cancel_requested'],'STOP was consumed in another phase; cancellation target unobserved')
 else:
  require(commit['phase']=='complete' and commit['chain_updated'] and commit['live_reusable'],'Noncancelled live chain was not reusable')
 caches=[e for e in events if e['event']=='cache']
 for event in caches:
  require(0<=event['retained_bytes']<=event['budget'] and 0<=event['entries']<=event['slots'] and 0<=event['held']<=event['budget']-event['retained_bytes'],'Observed cache memory/entry budget exceeded')
  if event['action'] in {'admit','take','evict','superseded'} and event['identity_observed']:
   require(event['snapshot_instance']>0 and not event['stage_descriptor_truncated'] and event['stage_ranges']==[list(x) for x in stage_ranges],'Snapshot observed ownership/stage ranges differ')
 spans=[e for e in events if e['event']=='stage_span'];completed=[e for e in spans if e['complete']]
 for event in spans:
  bounds=(event['lb'],event['le']);require(bounds in stage_ranges and event['device']==stage_ranges.index(bounds),'Actual runtime stage device/range ownership differs')
 from collections import Counter
 span_key=lambda e:(e['phase'],e['device'],e['lb'],e['le'],e['lo'],e['hi'])
 entered=Counter(span_key(e) for e in spans if not e['complete']);returned=Counter(span_key(e) for e in completed)
 require(all(entered[key]>=count for key,count in returned.items()),'Completed stage body lacks matching actual entry')
 if not result['ledger']['cancelled']:
  for bounds in stage_ranges:
   selected=[e for e in completed if (e['lb'],e['le'])==tuple(bounds)]
   covered=sorted(i for event in selected for i in range(max(result['ledger']['actual_reused'],event['lo']),min(len(request['ids']),event['hi'])))
   require(covered==list(range(result['ledger']['actual_reused'],len(request['ids']))),'Actual per-stage completed body prompt spans missing/duplicated')
 result['lifecycle']={'observed':True,'begin':begin,'commit':commit,'cache_events':caches,'stage_spans':spans,'body_completion_only':True,'complete_state_proof':False}
 request['lifecycle_committed_ids']=commit['ids']
 if request['label'].startswith('independent-context') or request['label'] in {'other-evict_A','other-evict_C','other-evict_D'}:cold_establish_binding(request,result)
 return result

def keyed_eviction(requests,target_meta,after_meta,budget=512<<20,slots=1):
 target=target_meta['lifecycle']['commit']['sha256_le32'];admitted=[];evicted=[]
 for request in requests:
  for event in request['meta']['lifecycle']['cache_events']:
   require(event['budget']==budget and event['slots']==slots,'Declared eviction group memory configuration differs')
   if event['source_tokens_sha256_le32']!=target:continue
   require(event['identity_observed'],'Target snapshot identity never observed')
   if event['action']=='admit':admitted.append(event)
   if event['action']=='evict':evicted.append(event)
 require(admitted,'Target was never successfully admitted; cold miss is not proof of eviction')
 require(any(v['snapshot_instance']==a['snapshot_instance'] and v['snapshot_metadata_sha256']==a['snapshot_metadata_sha256'] for a in admitted for v in evicted),'No exact admitted target snapshot victim evidence')
 require(after_meta['ledger']['actual_reused']==0,'Evicted target did not reconstruct cold')
 return {'target_source_sha256':target,'admissions':admitted,'victims':evicted,'bounded_memory':True}

def continue_fixture(plan,raw,directory):
 source=Path(plan['engine_root'])/'source';payload=directory/'continuation-seed.json';write(payload,raw)
 fixture=ROOT/'strata/flash-next/serial_prefix_prompt_fixture_v6.py'
 command=['docker','run','--rm','--network','none','--user','1000:1000','-v',str(source)+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(fixture)+':/fixture.py:ro','-v',str(payload)+':/seed.json:ro',plan['image'],'exec /opt/b70-c1-python/bin/python /fixture.py --continue-json /seed.json']
 response=json.loads(subprocess.check_output(command,text=True));require(response['tokenizer_sha256']==plan['tokens']['tokenizer_sha256'],'Continuation tokenizer/template source differs')
 write(directory/'continuation-token-receipt.json',response);return response

def compare_vectors(a,b):
 ra=Path(a['path']).read_bytes();rb=Path(b['path']).read_bytes();require(len(ra)==len(rb),'Vector shape differs')
 x=array.array('f');x.frombytes(ra);y=array.array('f');y.frombytes(rb)
 if sys.byteorder!='little':x.byteswap();y.byteswap()
 require(all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y),'Nonfinite logits/residual')
 d=[abs(float(u)-float(v)) for u,v in zip(x,y)]
 return {'bitwise_equal':ra==rb,'floats':len(x),'max_abs':max(d,default=0),'rmse':math.sqrt(sum(t*t for t in d)/len(d)) if d else 0,'left_sha256':a['sha256'],'right_sha256':b['sha256']}

class ComparisonError(ValueError):
 def __init__(self,message,evidence):super().__init__(message);self.evidence=evidence

def compare_requests(left,right,lmeta,rmeta,full=True):
 result={'output_ids_equal':left['output_ids']==right['output_ids'],'LP20_equal':left['LP']==right['LP'],'finish_equal':left['done'].split()[5]==right['done'].split()[5],'natural_completion':left['done'].split()[5]=='stop' and right['done'].split()[5]=='stop'}
 flags=list(result.values())
 if full:
  result['first_logits']=compare_vectors(lmeta['logits'][0],rmeta['logits'][0]);flags.append(result['first_logits']['bitwise_equal'])
  if lmeta['residuals'] and rmeta['residuals']:
   x={int(v['layer']):v for v in lmeta['residuals']};y={int(v['layer']):v for v in rmeta['residuals']};result['first_window_residuals']={str(i):compare_vectors(x[i],y[i]) for i in range(48)}
   differing=[i for i in range(48) if not result['first_window_residuals'][str(i)]['bitwise_equal']]
   result['first_divergent_residual_layer']=differing[0] if differing else None;flags.append(not differing)
 result['passed']=all(flags)
 if not result['passed']:raise ComparisonError('Strict output/logit/residual/natural-completion comparison failed',result)
 return result

def option(args,flag,value):
 if flag in args:args[args.index(flag)+1]=str(value)
 else:args += [flag,str(value)]

def decode_output(pack,ids):
 # Exact exported token strings; GPT-2 byte inverse used by the source tokenizer.
 p=Path(pack)/'tokenizer';vocab=read(p/'vocab.json');reverse={i:t for t,i in vocab.items()};types=read(p/'token_type.json')
 bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];extra=0
 for b in range(256):
  if b not in bs:bs.append(b);cs.append(256+extra);extra+=1
 inverse={chr(c):b for b,c in zip(bs,cs)}
 selected=[reverse[i] for i in ids if types[i] not in (3,4)]
 return bytes(inverse[c] for token in selected for c in token).decode('utf-8',errors='strict')

def run_process(plan,m,output,name,diag,activations,group):
 directory=output/name;directory.mkdir();capture=directory/'captures';capture.mkdir();arm=directory/'ARM'
 args=list(plan['args']);parking=group in {'eviction','parked'}
 option(args,'--conversation-cache-mib',512 if parking else 0);option(args,'--conversation-cache-slots',1 if parking else 0)
 env=dict(read(Path(plan['prepared'])/'server-config.json')['env']);observer_env_contract(env)
 env.update({'STRATA_FIDELITY_DIAG':str(diag),'STRATA_FIDELITY_DIAG_ACTIVATIONS':str(activations),'STRATA_FIDELITY_DIAG_ARM':'/results/ARM','STRATA_FIDELITY_DIAG_DIR':'/results/captures','STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_LIFECYCLE_DIAG':str(diag),'STRATA_PREFIX_DIAG_ARM':'/results/ARM','STRATA_ARTIFACT_IDENTITY_SHA256':sha(output/'plan.snapshot.json')})
 engine=Path(plan['engine_root']);container='b70-prefix-'+str(os.getpid())+'-'+name
 command=['docker','run','-i','--name',container,'--label','b70.prefix.plan='+sha(output/'plan.snapshot.json'),'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',m['pack']+':/pack:ro','-v',str(ROOT/read(ROOT/'strata/flash-next/model-lock.json')['destination'])+':/model:ro','-v',str(directory)+':/results']
 for k,v in sorted(env.items()):command += ['-e',k+'='+v]
 command += [plan['image'],'cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+args)]
 write(directory/'command.json',command);p=None;rows=[];error=None
 try:
  p=Protocol(command,directory)
  # Identical useful warmups execute before ARM. They do not consume six-request capture quota.
  for key in ['A','B','C']:p.request('warm-'+key,plan['tokens']['cases'][key],fresh=1,max_new=1)
  arm.write_text('ARM after READY and fixed warmups\n',encoding='ascii')
  cases=plan['tokens']['cases'];requests=[]
  def req(label,key,fresh=0,pin=None,cancel=None):
   require(len(requests)<ARMED_REQUESTS[group]<=6,'Armed request roster/capture cap exceeded before submission')
   r=p.request(label,cases[key],fresh,pin,cancel);r['decoded_output_without_special_tokens']=decode_output(m['pack'],r['output_ids']);write(directory/(label+'.json'),r);meta=extract(r,capture,diag==1,activations==1,expected_stage_ranges(args));requests.append({'raw':r,'meta':meta});write(directory/'requests.json',requests);return r,meta
  if group=='basic':
   for key in ['A','B','C']:req(key,key,fresh=1)
  elif group in {'root','pin','turn'}:
   ka,kb=('pin_A','pin_B') if group=='pin' else ('turn_A','turn_B') if group=='turn' else ('A','B')
   pin=plan['tokens']['pin'] if group=='pin' else None
   ref,rm=req('reference',kb,1,pin)
   req('establish',ka,1,pin)
   hit,hm=req('hit',kb,0,pin)
   expected=plan['tokens']['pin' if group=='pin' else 'turn_shared' if group=='turn' else 'root']
   require(hm['ledger']['actual_reused']==expected,'Actual selected prefix differs from targeted '+group+' boundary')
   rows.append(compare_requests(ref,hit,rm,hm))
   # Repeat the matched triple in the same warmed process; no latency comparison.
   ref2,rm2=req('reference2',kb,1,pin);req('establish2',ka,1,pin);hit2,hm2=req('hit2',kb,0,pin)
   require(hm2['ledger']['actual_reused']==expected,'Second matched hit boundary differs')
   rows.append(compare_requests(ref2,hit2,rm2,hm2));rows.append(compare_requests(ref,ref2,rm,rm2))
  elif group=='parked':
   expected=plan['tokens']['turn_boundaries']['B']
   for suffix in ['', '2']:
    ref,rm=req('reference'+suffix,'B',1)
    req('independent-context'+suffix,'A',0,0)
    hit,hm=req('restored'+suffix,'B')
    require(hm['selection']['route']=='ram_snapshot' and hm['ledger']['actual_reused']==expected,'Complete parked snapshot restore route/boundary was not exercised')
    rows.append(compare_requests(ref,hit,rm,hm))
  elif group=='eviction':
   ref,rm=req('reference','evict_B',1)
   for key in ['evict_A','evict_C','evict_D']:req('other-'+key,key,0,0)
   after,am=req('after-eviction','evict_B')
   write(directory/'keyed-eviction.json',keyed_eviction(requests,rm,am))
   require(am['ledger']['evictions']>rm['ledger']['evictions'] and am['ledger']['actual_reused']==0,'No actual bounded eviction/cold reconstruction evidence')
   final,fm=req('fresh-final','evict_B',1);rows += [compare_requests(ref,after,rm,am),compare_requests(after,final,am,fm)]
  elif group=='cancel':
   ref,rm=req('reference','pin_B',1)
   cancelled,cm=req('cancel-prefill','pin_B',1,cancel='prefill')
   require(cancelled['stop_sent'] and cm['ledger']['cancelled'],'Prefill STOP did not cancel')
   after,am=req('after-prefill','pin_B');fresh,fm=req('fresh-after-prefill','pin_B',1)
   rows += [compare_requests(ref,after,rm,am),compare_requests(after,fresh,am,fm)]
   cancelled,cm=req('cancel-decode','A',1,cancel='decode')
   require(cancelled['stop_sent'] and cm['ledger']['cancelled'],'Decode STOP did not cancel')
   final,fm=req('after-decode-fresh','pin_B',1);rows.append(compare_requests(ref,final,rm,fm))
  elif group in {'cancel_decode','cancel_prefill_isolation'}:
   phase='decode' if group=='cancel_decode' else 'prefill'
   key='A' if phase=='decode' else 'pin_B'
   refa,rma=req('reference-same',key,1);refb,rmb=req('reference-unrelated','C',1)
   stopped,sm=req('cancelled',key,1,cancel=phase)
   require(stopped['stop_sent'] and sm['ledger']['cancelled'],'STOP did not cause targeted cancellation')
   replay,pm=req('nonfresh-same',key);other,om=req('nonfresh-unrelated','C');fresh,fm=req('fresh-confirm-unrelated','C',1)
   rows += [compare_requests(refa,replay,rma,pm),compare_requests(refb,other,rmb,om),compare_requests(other,fresh,om,fm)]
  elif group=='real_live':
   seed,seedmeta=req('seed','A',1);continuation=continue_fixture(plan,seed,directory);cases['actual_continuation']=continuation['continuation_ids']
   ref,rm=req('fresh-continuation','actual_continuation',1)
   seed2,seedmeta2=req('seed-again','A',1);rows.append(compare_requests(seed,seed2,seedmeta,seedmeta2))
   require(seed2['lifecycle_committed_ids']==continuation['seed_committed_ids'],'Actual seed committed chain changed')
   hit,hm=req('nonfresh-live-continuation','actual_continuation')
   require(hm['selection']['route']=='live' and hm['ledger']['actual_reused']==continuation['expected_live_reuse'],'Real continuation did not reuse exact committed live prefix')
   rows.append(compare_requests(ref,hit,rm,hm))
  require(len(requests)==ARMED_REQUESTS[group]<=6,'Declared exact armed request roster incomplete or exceeded')
  write(directory/'requests.json',requests)
 except Exception as e:
  error=str(e)
  if isinstance(e,ComparisonError):rows.append(e.evidence)
 finally:
  rc=None
  try:rc=p.close() if p else None
  except Exception as e:error=(error+'; ' if error else '')+'protocol close: '+str(e)
  try:
   state=c1.inspected(container)['State'];require(not state['Running'],'Engine remains live')
   require(state['ExitCode']==0 and not state.get('OOMKilled'),'Engine did not exit normally')
   subprocess.run(['docker','rm',container],check=True,capture_output=True,text=True)
   removed=not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+container+'$'],text=True).strip()
   require(removed,'Container remains registered')
  except Exception as e:
   # Owned cleanup only; retain failure. No broad process kill or recovery from this controller.
   subprocess.run(['docker','stop','--time','30',container],capture_output=True,text=True,timeout=50)
   subprocess.run(['docker','rm',container],capture_output=True,text=True,timeout=30)
   error=(error+'; ' if error else '')+'teardown: '+str(e);state=None;removed=False
  result={'group':group,'name':name,'diagnostic':diag,'activations':activations,'passed':error is None and rc==0,'error':error,'terminal':state,'removed':removed,'engine_rc':rc,'comparisons':rows,'transport':'native GEN','capture_bound':6,'cancelled_work_counter_scope':'completed spans only'}
  write(directory/'result.json',result)
 return result

def basic_receipt_binding(path,plan,same_topology=True):
 # Preserve the qualified V8 raw/source/lifecycle prerequisite verifier intact.
 require(sha(Path(v8.__file__))==V8_CONTROLLER_SHA,'Frozen V8 prerequisite verifier changed')
 return v8.basic_receipt_binding(path,plan,same_topology)


def run(a):
 plan=read(a.plan);require(plan['controller_sha256']==sha(Path(__file__)),'Controller changed after preparation')
 require(plan['token_fixture_sha256']==sha(ROOT/'strata/flash-next/serial_prefix_prompt_fixture_v6.py'),'Token fixture code changed')
 require(plan['coverage_parser_sha256']==sha(ROOT/'strata/flash-next/audit_fidelity_observer_coverage.py')==COVERAGE_SHA,'Strict coverage parser changed')
 m,_=source35_admission(Path(plan['prepared']));require(sha(Path(plan['prepared'])/'c1-source-identity-proof-v13.json')==plan['c113_source_proof_sha256'] and plan['combined_plan_sha256']==c1.COMBINED_PLAN_SHA,'C113 source35 final admission changed');require(sha(Path(plan['prepared'])/'prepared.json')==plan['prepared_sha256'],'Original prepared source changed')
 engine_identity(Path(plan['engine_root']),m);require(sha(Path(plan['engine_root'])/'receipt.json')==plan['engine_receipt_sha256']==m['engine_receipt_sha256'],'Corrected source-upload/build receipt changed')
 require(plan['matched_geometry']==geometry_contract(plan['args']),'Matched geometry changed')
 require(plan['cards']==m['cards'] and expected_stage_ranges(plan['args'])==expected_stage_ranges(read(Path(plan['prepared'])/'server-config.json')['args']),'Actual prepared topology/device split changed')
 require(plan['base_prepared_alias']==m['alias'] and plan['research_alias']==pilot_alias(plan['cards'],plan['args']),'Pilot/base identity metadata changed')
 require(sha(ROOT/'strata/flash-next/run_source_upload_oracle_full.py')==plan['identity_parser_sha256'],'Full identity verifier changed')
 verify_model_identity(Path(plan['model_identity']['path']),read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']])
 c1.leased(plan['cards']);h=read(a.pre_health);require(h.get('passed') and set(plan['cards'])<=set(h['cards']) and 0<=time.time()-h['finished_epoch']<=300,'Fresh parent-owned pre-health missing')
 require(h.get('files'),'Parent pre-health has no source command/log evidence')
 for f in h['files']:require(sha(f['path'])==f['sha256'],'Parent health evidence changed')
 if len(plan['cards'])>1:
  require(a.one_card_receipt,'Matched combined one-card diagnostic required before pair')
  r=read(a.one_card_receipt);require(r.get('passed') and r.get('numerical_and_teardown_passed') and r.get('post_health_passed') and r.get('group') in {'basic','all'} and len(r['cards'])==1 and r['engine_receipt_sha256']==plan['engine_receipt_sha256'],'Matched one-card evidence absent')
  basic_receipt_binding(a.one_card_receipt,plan,False)
  one_plan=read(a.one_card_receipt.parent/'plan.snapshot.json');require(one_plan.get('matched_geometry')==plan['matched_geometry'],'One-card reference numerical geometry differs')
 preflight(plan,a.basic_receipt,a.one_card_receipt)
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.snapshot.json',plan)
 stopping=[False]
 def stop(sig,frame):
  if not stopping[0]:stopping[0]=True;raise InterruptedError('Parent stop signal '+str(sig))
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
 require(plan['schema']==9 and plan['groups']==list(REMAINING_GROUPS) and plan['base_v8_controller_sha256']==V8_CONTROLLER_SHA,'Exact remaining-suite plan required')
 groups=list(REMAINING_GROUPS) if a.group=='remaining' else [a.group]
 results=[];error=None
 if a.group!='basic':
  require(a.basic_receipt,'Basic flag-off/on equivalence must pass before prefix fixture')
  basic_receipt_binding(a.basic_receipt,plan,True)
  b=read(a.basic_receipt);require(b.get('passed') and b.get('numerical_and_teardown_passed') and b.get('post_health_passed') and b.get('group') in {'basic','all'} and b['engine_receipt_sha256']==plan['engine_receipt_sha256'] and b['cards']==plan['cards'],'Matched basic receipt missing')
 try:
  for group in groups:
   verify_model_identity(Path(plan['model_identity']['path']),read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']])
   if group=='basic':
    local=[]
    for name,diag,acts in [('basic_off',0,0),('basic_logits',1,0),('basic_layers',1,1)]:
     r=run_process(plan,m,a.output,name,diag,acts,'basic');results.append(r);require(r['passed'],'Basic process failed: '+str(r['error']));local.append(read(a.output/name/'requests.json'))
    pairs=[]
    for off,logits,layers in zip(*local):
     pairs.append({'label':off['raw']['label'],'flag0_on':compare_requests(off['raw'],logits['raw'],off['meta'],logits['meta'],False),'activations0_1':compare_requests(logits['raw'],layers['raw'],logits['meta'],layers['meta'])})
    write(a.output/'basic-equivalence.json',pairs)
   else:
    r=run_process(plan,m,a.output,group,1,1,group);results.append(r);require(r['passed'],group+' failed: '+str(r['error']))
   verify_model_identity(Path(plan['model_identity']['path']),read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']])
 except Exception as e:
  error=str(e)
  if isinstance(e,ComparisonError):write(a.output/'failed-comparison.json',e.evidence)
 report={'schema':9,'status':'awaiting parent post-health' if error is None else 'FAILED strict serial gate','passed':False,'numerical_and_teardown_passed':error is None and all(r['passed'] for r in results),'post_health_passed':False,'error':error,'cards':plan['cards'],'group':a.group,'engine_receipt_sha256':plan['engine_receipt_sha256'],'matched_geometry':plan['matched_geometry'],'primary_model_name':plan['primary_model_name'],'research_alias':plan['research_alias'],'base_prepared_alias':plan['base_prepared_alias'],'plan_sha256':sha(a.output/'plan.snapshot.json'),'pre_health_sha256':sha(a.pre_health),'finished_epoch':time.time(),'results':results,'transport':'native GEN; no API/concurrency/shelf/clean latency claim','flag0_full_logits_observed':False,'complete_prefix_qualified':False,'new_engine390_source_upload_qualified':True}
 write(a.output/'report.json',report);print(json.dumps({'report':str(a.output/'report.json'),'numerical_and_teardown_passed':report['numerical_and_teardown_passed']}))
 if error:raise SystemExit(1)

def finalize(a):
 p=a.output/'report.json';r=read(p);h=read(a.post_health);require(h.get('passed') and set(r['cards'])<=set(h['cards']) and h.get('finished_epoch',0)>=r['finished_epoch'],'Parent post-health does not cover completed fixture/devices')
 require(h.get('files'),'Parent post-health has no source command/log evidence')
 for row in h['files']:require(sha(row['path'])==row['sha256'],'Post-health artifact changed')
 require(r['numerical_and_teardown_passed'],'Strict numerical/teardown gate failed')
 plan=read(a.output/'plan.snapshot.json');m=read(Path(plan['prepared'])/'prepared.json')
 for group in ('parked','eviction'):
  if not any(x['name']==group for x in r['results']):continue
  directory=a.output/group;rows=read(directory/'requests.json')
  wanted=['independent-context','independent-context2'] if group=='parked' else ['other-evict_A','other-evict_C','other-evict_D']
  require(len(rows)==ARMED_REQUESTS[group] and [row['raw']['label'] for row in rows if row['raw']['label'] in wanted]==wanted,'Complete postterminal cold request roster differs')
  for row in rows:
   raw=row['raw']
   if raw['label'].startswith('independent-context') or raw['label'] in {'other-evict_A','other-evict_C','other-evict_D'}:
    require(read(directory/(raw['label']+'.json'))==raw,'Cold raw request copy changed')
    fresh=extract(raw,directory/'captures',True,True,expected_stage_ranges(plan['args']));cold_recollected_binding(row['meta'],fresh)
 identity=verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']])
 require(read(a.model_identity)['started']>=max(r['finished_epoch'],h['finished_epoch']),'Full post-run model hash scan must start after fixture and post-health')
 r['post_model_identity']=identity
 r.update(passed=True,post_health_passed=True,post_health_sha256=sha(a.post_health),status='PASSED bounded serial native fixture; no concurrency/API qualification');r['bounded_serial_remaining_passed']=r['group']=='remaining';r['complete_prefix_qualified']=False;write(p,r)

def topology_args(args):
 out=[];i=0
 while i<len(args):
  if args[i] in {'--layer-split','--split-device'}:i+=2;continue
  if args[i]=='--trim-stage-weights':i+=1;continue
  out.append(args[i]);i+=1
 return out

def compare_topologies(a):
 report={'schema':9,'passed':False,'scope':'CPU comparison of completed matched-geometry native reports, not an independent model reference','pairs':[],'error':None}
 try:
  left,right=read(a.one_card),read(a.two_card);lp=read(a.one_card.parent/'plan.snapshot.json');rp=read(a.two_card.parent/'plan.snapshot.json')
  for receipt,path,plan in [(left,a.one_card,lp),(right,a.two_card,rp)]:
   require(receipt.get('passed') and receipt.get('post_health_passed') and receipt.get('numerical_and_teardown_passed'),'Input native fixture did not qualify with health/teardown')
   require(receipt['plan_sha256']==sha(path.parent/'plan.snapshot.json'),'Input plan changed')
   require(plan['controller_sha256']==sha(Path(__file__)),'Input controller generation differs')
   require(plan['coverage_parser_sha256']==sha(ROOT/'strata/flash-next/audit_fidelity_observer_coverage.py')==COVERAGE_SHA,'Input strict coverage generation differs')
   require(plan['matched_geometry']==geometry_contract(plan['args']),'Input geometry incomplete')
   require(plan['research_alias']==pilot_alias(plan['cards'],plan['args']),'Input pilot identity metadata differs')
  require(len(left['cards'])==1 and len(right['cards'])==2,'One/two-card ordering differs')
  require(left['engine_receipt_sha256']==right['engine_receipt_sha256'],'Compared source/binary generations differ')
  require(lp['matched_geometry']==rp['matched_geometry'] and topology_args(lp['args'])==topology_args(rp['args']),'Numerical-control arguments differ beyond layer split/device/trim')
  require(lp['tokens']==rp['tokens'] and lp['token_fixture_sha256']==rp['token_fixture_sha256'],'Source token/template fixture differs')
  le=read(Path(lp['prepared'])/'server-config.json')['env'];re_=read(Path(rp['prepared'])/'server-config.json')['env']
  math_env=lambda env:{k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','ONEAPI_')) and k!='STRATA_ARTIFACT_IDENTITY_SHA256'}
  require(math_env(le)==math_env(re_),'Effective native/SYCL math environments differ')
  require(left['group']==right['group'],'Compared fixture groups differ')
  names=['basic_off','basic_logits','basic_layers'] if left['group']=='basic' else [r['name'] for r in left['results']]
  require(names==(['basic_off','basic_logits','basic_layers'] if right['group']=='basic' else [r['name'] for r in right['results']]),'Compared process roster differs')
  for name in names:
   lroot=a.one_card.parent/name;rroot=a.two_card.parent/name;lrows=read(lroot/'requests.json');rrows=read(rroot/'requests.json');require(len(lrows)==len(rrows),'Request coverage differs')
   for lrow,rrow in zip(lrows,rrows):
    x,y=lrow['raw'],rrow['raw'];require(x['label']==y['label'] and x['ids']==y['ids'] and x['fresh']==y['fresh'] and x['pin']==y['pin'],'Compared actual source requests differ')
    if x['cancel_requested'] or y['cancel_requested']:continue # cancellation work/point is asynchronous, not matched numeric parity.
    full=name!='basic_off';lm=extract(x,lroot/'captures',full,full and name!='basic_logits',expected_stage_ranges(lp['args']));rm=extract(y,rroot/'captures',full,full and name!='basic_logits',expected_stage_ranges(rp['args']))
    report['pairs'].append({'process':name,'label':x['label'],'comparison':compare_requests(x,y,lm,rm,full)})
  require(report['pairs'],'No numerical pairs compared');report['passed']=True
 except Exception as e:
  report['error']=str(e)
  if isinstance(e,ComparisonError):report['failed_comparison']=e.evidence
 write(a.output,report)
 if not report['passed']:raise SystemExit(1)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 q=sub.add_parser('prepare');q.add_argument('--base-plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('run');q.add_argument('--plan',type=Path,required=True);q.add_argument('--pre-health',type=Path,required=True);q.add_argument('--one-card-receipt',type=Path);q.add_argument('--basic-receipt',type=Path);q.add_argument('--group',choices=list(REMAINING_GROUPS)+['remaining'],default='remaining');q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('finalize');q.add_argument('--output',type=Path,required=True);q.add_argument('--post-health',type=Path,required=True);q.add_argument('--model-identity',type=Path,required=True)
 q=sub.add_parser('compare-topologies');q.add_argument('--one-card',type=Path,required=True);q.add_argument('--two-card',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args();{'prepare':prepare,'run':run,'finalize':finalize,'compare-topologies':compare_topologies}[a.mode](a)
if __name__=='__main__':main()
