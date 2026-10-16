#!/usr/bin/env python3
"""S113 ordered native-prompt-start pair + existing functionals; no native-token-ID claim."""
import argparse,concurrent.futures,hashlib,json,math,re,sys,threading,time,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parents[1];R=D.parent;REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
sys.path.insert(0,str(REPO/'strata/flash-next'));from bench_serving import get,send
BASE='http://127.0.0.1:18085';ALIAS='qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata'
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def identity():
 x=get(BASE,'/v1/models');ids=[i['id'] for i in x['data']]
 assert ids[0]=='hotschmoe-dd' and ALIAS in ids and ALIAS in (REPO/'evals/configs/models.yaml').read_text();return x
def post(body):
 body.setdefault('strata_tune',{})['spec_min_p']=0
 q=urllib.request.Request(BASE+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=1800) as f:return json.load(f)
def functional(out):
 assert (out/'prefill/complete.json').exists();dst=out/'functional';dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  rows=list(pool.map(lambda ab:send(BASE,'hotschmoe-dd',[{'role':'user','content':f'Return only the integer: {ab[0]} * {ab[1]}.'}],32,spec_min_p=0),[(17,19),(23,29)]))
 save(dst/'concurrent-math.json',rows);assert [r['text'].strip() for r in rows]==['323','667']
 source=R/'prefill-quantum-s33/q0/functional'
 thought=json.loads((source/'thinking.json').read_text())['request'];tr=post(thought);save(dst/'thinking.json',{'request':thought,'response':tr})
 msg=tr['choices'][0]['message'];assert msg['content'].strip()=='391' and msg.get('reasoning_content')
 first=json.loads((source/'tool-first.json').read_text())['request'];fr=post(first);save(dst/'tool-first.json',{'request':first,'response':fr})
 msg=fr['choices'][0]['message'];calls=msg.get('tool_calls');assert len(calls or [])==1 and calls[0]['function']['name']=='lookup_build_status' and json.loads(calls[0]['function']['arguments'])=={'project':'b70-demo'}
 final=json.loads((source/'tool-final.json').read_text())['request'];final['messages']=first['messages']+[msg,{'role':'tool','tool_call_id':calls[0]['id'],'content':json.dumps({'project':'b70-demo','status':'passed','tests_passed':17,'tests_failed':0})}]
 tr=post(final);save(dst/'tool-final.json',{'request':final,'response':tr});txt=tr['choices'][0]['message']['content'].lower();assert all(x in txt for x in ['17','0','passed','failed'])
 save(dst/'health.json',get(BASE,'/health'));save(dst/'verdict.json',{'status':'PASS','arithmetic':True,'thinking':True,'tool_roundtrip':True})

VALUE_MARKER='strata: prefill attention value pipeline selected (nq32 cap2051 INT8 page4)'
def value_marker(out,expected):
 found=[line for line in (out/'engine.log').read_text().splitlines() if 'prefill attention value pipeline selected' in line]
 assert all(line==VALUE_MARKER for line in found) and len(found)<=(1 if expected else 0),'Duplicate/wrong/OFF value HOST marker'
 return {'selected':bool(found),'count':len(found),'raw':found,'scope':'HOST selection only, not independent GPU dispatch/call count'}

WORK=['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']
QSA_RE=re.compile(r'^strata QSA prefill tiled: enabled=(0|1) nq=256 idx=4x128 R4 stride=50002 active=4097\.\.50001 first_active=(\d+) tile=8 key_wg=256 SG32$',re.M)
SERVE_RE=re.compile(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms .*?, (\d+) generated in ([\d.]+) ms .*?, drafts accepted (\d+) of (\d+), (\d+) checkpoints',re.M)
def work(row):
 assert all(k in row['timings'] and isinstance(row['timings'][k],(int,float)) and not isinstance(row['timings'][k],bool) and math.isfinite(row['timings'][k]) and row['timings'][k]>=0 for k in WORK)
 return {k:row['timings'][k] for k in WORK}
def emitted(row):
 return ''.join(c.get('choices',[{}])[0].get('delta',{}).get('content') or '' for c in row['chunks'] if c.get('choices'))
def row_coherence(x,y):
 assert x['request']==y['request'] and x['text']==y['text']==emitted(x)==emitted(y)
 assert x['usage']==y['usage'] and work(x)==work(y),'Control/candidate native work mismatch; no retiming/retry'
def qsa_marker(out,expected):
 found=QSA_RE.findall((out/'engine.log').read_text())
 assert len(found)==1 and found[0][0]==str(expected) and int(found[0][1])==4097,'Missing/duplicate/wrong QSA first-eligible host marker'
 return {'enabled':int(found[0][0]),'first_active':int(found[0][1]),'domain':'nq256/stride50002/active4097..50001','scope':'host selection only, not independent GPU dispatch'}
def capture_stats(log):
 return {'logged_capture_layouts':re.findall(r'^strata verify: captured (?:the \d+-token window \(upload no error, sync no error\)|the batch window over slots [0-9,]+)$',log,re.M),
 'native_window_rows':re.findall(r'^strata batch: (\d+) windows, avg ([0-9.]+) rows,',log,re.M),
 'paired_solo':re.findall(r'^strata batch pair head: (\d+) windows drafted as a pair, (\d+) deferred but solo$',log,re.M)}
def off_candidates(out):
 full=(out/'engine.log').read_text()
 assert not re.search(r'^strata (?:Q6 A4:|IQ2_XS multi:|MTP pair graph:)',full,re.M),'Unrelated candidate HOST marker'
 center=re.findall(r'^strata Q6 center: .*$',full,re.M)
 assert center in [[],['strata Q6 center: enabled=0 C6 A2 R1 SG32 PF0 shapes=6144x2560,2560x10240']]
def prefill(out,warm=False):
 if not warm:assert json.loads((out/'warm4k/complete.json').read_text())['status']=='PASS_LOCAL_AND_AVAILABLE_CROSS_ARM_GATES'
 if warm:
  assert value_marker(out,int(out.name=='mode1'))['count']==0,'Value marker already present before first warm request'
  assert not QSA_RE.findall((out/'engine.log').read_text())
  off_candidates(out)
 dst=out/('warm4k'if warm else'prefill');dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
 reference=R/'peer-gather-serving-s38/mode0/screen';entries=[]
 schedule=[('p4096','warm')]if warm else[(case,repeat)for case in ['p4096','p32768']for repeat in range(3)]
 for case,repeat in schedule:
  name='warm4k'if warm else f'{case}-{repeat}'
  body=json.loads((D/'warm-request.json').read_text())if warm else json.loads((reference/(name+'.json')).read_text())['request']
  body.setdefault('strata_tune',{})['spec_min_p']=0
  assert body['max_tokens']==256 and len(body['messages'])==1
  before=(out/'engine.log').stat().st_size;save(dst/(name+'-start.json'),{'byte_offset':before,'epoch_ns':time.time_ns(),'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),'unmeasured_warm':warm})
  row=send(BASE,body['model'],body['messages'],256,spec_min_p=0);save(dst/(name+'.json'),row)
  assert row['request']==body,'Reconstructed common fixed-policy request differs'
  ww=work(row);expected_prompt=row['usage']['prompt_tokens']if warm else(4059 if case=='p4096'else 32728)
  assert row['text']==emitted(row)and row['text']and row['usage']['prompt_tokens']==expected_prompt and row['usage']['completion_tokens']==256
  assert ww['cache_n']<=64 and row['usage']['prompt_tokens_details']['cached_tokens']==ww['cache_n']
  assert ww['prompt_n']+ww['cache_n']==expected_prompt and ww['predicted_n']==256
  assert all(math.isfinite(row[k])for k in ['ttft_s','wall_s','client_decode_tps'])
  if out.name=='mode1':row_coherence(json.loads((D/'mode0'/dst.name/(name+'.json')).read_text()),row)
  deadline=time.monotonic()+30
  while True:
   raw=(out/'engine.log').read_bytes()[before:];log=raw.decode('ascii');native=SERVE_RE.findall(log)
   if len(native)==1 and int(native[0][0])==expected_prompt and int(native[0][4])==256:break
   if time.monotonic()>=deadline:
    (dst/(name+'-partial-engine.log')).write_bytes(raw);raise RuntimeError('Native summary incomplete; preserve server, no retry/generation')
   time.sleep(.1)
  (dst/(name+'-engine.log')).write_bytes(raw)
  n=native[0];assert tuple(map(int,[n[1],n[2],n[4],n[6],n[7]]))==(ww['cache_n'],ww['prompt_n'],ww['predicted_n'],ww['draft_n_accepted'],ww['draft_n'])
  parts=[]
  for m in re.finditer(r'^strata batch prefill quantum: allocated (\d+) requested (\d+) effective (\d+) eligible (\d+) from (\d+) to (\d+)$',log,re.M):
   C,requested,Q,eligible,start,end=map(int,m.groups());assert C==Q==12288 and requested==0 and eligible==0 and 0<=start<end<=expected_prompt
   parts.append({'raw':m.group(0),'allocated_C':C,'effective_Q':Q,'from':start,'to':end,'source_derived_chunks':[{'from':x,'to':min(x+Q,end)}for x in range(start,end,Q)],'interval_scope':'Observed native read_part bounds; subdivisions source-derived, not callback timestamps.'})
  assert parts and not re.search(r'ERR batch MTP|DEVICE_LOST|capture.*(?:failed|produced no graph)',log,re.I)
  marker=None
  if case=='p4096':assert not QSA_RE.findall((out/'engine.log').read_text()),'4K must preserve old scorer path'
  else:
   marker=qsa_marker(out,0)
   assert any(part['to']>=16384 and any(c['to']-c['from']>=256 for c in part['source_derived_chunks'])for part in parts)
  value=value_marker(out,int(out.name=='mode1'))
  if out.name=='mode1':assert value['count']==1,'First4K warm did not activate value HOST marker; stop remaining requests'
  value['first_emission_request']='warm4k'if value['selected']else None
  off_candidates(out)
  entry={'name':name,'status':'PASS_LOCAL_GATES','unmeasured_warm':warm,'VALUE_marker':value,'usage':row['usage'],'work':ww,'capture_stats':capture_stats(log),'native_summary_raw':SERVE_RE.search(log).group(0),'native_part_intervals':parts,'QSA_marker':marker,'native_prompt_tps':row['timings']['prompt_per_second'],'native_prompt_ms':row['timings']['prompt_ms'],'ttft_s':row['ttft_s'],'wall_s':row['wall_s'],'content_event_gap_ms':row['content_event_gap_ms'],'output_sha256':row['output_sha256']}
  if out.name=='mode1':
   control=json.loads((D/'mode0'/dst.name/(name+'-verdict.json')).read_text())
   checks={k:control[k]==entry[k]for k in ['work','usage','native_part_intervals','capture_stats']}
   save(dst/(name+'-control-parity.json'),{'status':'PASS'if all(checks.values())else'FAIL_STOP_REMAINING_REQUESTS','checks':checks,'request_text_SSE_work_checked':True,'normalization':None})
   assert all(checks.values()),'Native/capture mismatch; stop remaining requests and normal-stop/health'
  entries.append(entry);save(dst/(name+'-verdict.json'),entry)
 save(dst/'complete.json',{'status':'PASS_LOCAL_AND_AVAILABLE_CROSS_ARM_GATES','entries':entries,'cap':256,'warm_tokenization':'observed/matched, no old4059 assumption'if warm else None,'VALUE_marker':value_marker(out,int(out.name=='mode1')),'first_value_emission_request':'warm4k'if out.name=='mode1'else None,'not_independent_GPU_dispatch':True})
 print('S113 phase complete; no extra warmups, requests or retries.',flush=True)
p=argparse.ArgumentParser();p.add_argument('arm',choices=['0','1']);p.add_argument('phase',choices=['warm','prefill','functional']);a=p.parse_args();out=D/f'mode{a.arm}'
if a.phase=='warm':prefill(out,True)
elif a.phase=='prefill':prefill(out,False)
else:functional(out)
