#!/usr/bin/env python3
"""S93 ordered native-prompt-start pair + existing functionals; no native-token-ID claim."""
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
 q=urllib.request.Request(BASE+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=1800) as f:return json.load(f)
def pair(out,repeat):
 parent=out/'coding';parent.mkdir(exist_ok=True);dst=parent/('warm' if repeat==0 else f'repeat{repeat}');dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
 initial=(out/'engine.log').read_bytes();save(dst/'window-start.json',{'byte_offset':len(initial),'epoch_ns':time.time_ns()})
 def request(i):
  ref=json.loads((R.parent/f'flashnext_ista_gsq_rco_iq3xxs_strata_accounting_20261014/concurrency/coding-0-{i}.json').read_text())
  messages=ref['request']['messages']
  call_start=time.monotonic_ns();epoch_start=time.time_ns()
  life={'client_call_start_monotonic_ns':call_start,'client_call_start_epoch_ns':epoch_start,'model':'hotschmoe-dd','messages':messages,'max_new_tokens':256}
  save(dst/f'coding-{i}-lifetime.json',life)
  try:row=send(BASE,'hotschmoe-dd',messages,256)
  except Exception as e:
   life.update(client_call_end_monotonic_ns=time.monotonic_ns(),client_call_end_epoch_ns=time.time_ns(),error_type=type(e).__name__);save(dst/f'coding-{i}-lifetime.json',life);raise
  life.update(client_call_end_monotonic_ns=time.monotonic_ns(),client_call_end_epoch_ns=time.time_ns());save(dst/f'coding-{i}-lifetime.json',life)
  row['client_lifetime']=life
  required=['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']
  assert all(k in row['timings'] and isinstance(row['timings'][k],(int,float)) and not isinstance(row['timings'][k],bool) and math.isfinite(row['timings'][k]) and row['timings'][k]>=0 for k in required),'Missing/nonfinite/negative native work field'
  row['prefix_matches_s16']=ref['text'].startswith(row['text']) if repeat==0 else None;save(dst/f'coding-{i}.json',row)
  assert row['usage']['completion_tokens']==256 and row['usage']['prompt_tokens']==107
  if repeat==0:assert row['prefix_matches_s16']
  assert all(math.isfinite(row[k]) for k in ['ttft_s','wall_s','client_decode_tps'])
  assert all(math.isfinite(v) for v in row['timings'].values() if isinstance(v,(int,float)))
  return row
 start=time.monotonic()
 gate={'status':'WAITING_NATIVE_PROMPT_START','fresh_engine_byte_offset':len(initial),'only_request0_outstanding':True}
 save(dst/'admission-gate.json',gate)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  first=pool.submit(request,0);deadline=time.monotonic()+60
  while True:
   with (out/'engine.log').open('rb') as engine:
    engine.seek(len(initial));suffix=engine.read()
   m=re.search(rb'^strata trace: prompt start (\d+) -1\n',suffix,re.M)
   if m:
    gate.update(native_prompt_start_line=m.group(0).decode('ascii').rstrip(),native_prompt_count_minus1=int(m[1]),marker_byte_offset=len(initial)+m.start(),marker_observed_monotonic_ns=time.monotonic_ns(),marker_observed_epoch_ns=time.time_ns())
    if first.done():
     gate['status']='FAIL_FIRST_COMPLETED_BEFORE_RELEASE';save(dst/'admission-gate.json',gate);first.result();raise RuntimeError(gate['status'])
    gate.update(status='NATIVE_PROMPT_START_OBSERVED',request1_submit_monotonic_ns=time.monotonic_ns());save(dst/'admission-gate.json',gate)
    second=pool.submit(request,1);break
   if first.done():
    gate['status']='FAIL_NO_NATIVE_PROMPT_START_BEFORE_FIRST_COMPLETION';save(dst/'admission-gate.json',gate);first.result();raise RuntimeError(gate['status'])
   if time.monotonic()>=deadline:
    gate['status']='OBSERVATION_DEADLINE_PRESERVE_SERVER_NO_RETRY';save(dst/'admission-gate.json',gate);raise RuntimeError(gate['status'])
   time.sleep(.005) # Poll actual flushed marker; elapsed delay is NOT the admission proof.
  rows=[first.result(),second.result()]
 life=[r['client_lifetime'] for r in rows]
 overlap=min(x['client_call_end_monotonic_ns'] for x in life)-max(x['client_call_start_monotonic_ns'] for x in life)
 assert life[0]['client_call_start_monotonic_ns']<=gate['marker_observed_monotonic_ns']<=life[1]['client_call_start_monotonic_ns']<life[0]['client_call_end_monotonic_ns']
 assert overlap>0,'Client lifetimes did not overlap; do not reinterpret as paired workload'
 gate.update(status='PASS_NATIVE_PROMPT_START_ORDER_AND_CLIENT_OVERLAP',client_overlap_ns=overlap,request0_lifetime=life[0],request1_lifetime=life[1]);save(dst/'admission-gate.json',gate)
 elapsed=time.monotonic()-start
 for _ in range(100):
  log=(out/'engine.log').read_bytes()[len(initial):].decode('utf8');pairs=re.findall(r'strata batch pair head: (\d+) windows drafted as a pair, (\d+) deferred but solo',log)
  native_windows=re.findall(r'^strata batch: (\d+) windows, avg ',log,re.M)
  if pairs and native_windows:break
  time.sleep(.1)
 else:raise RuntimeError('Native summary incomplete: preserve server; do not regenerate requests')
 (dst/'engine-segment.log').write_text(log,encoding='utf8')
 assert sum(int(x) for x,y in pairs)>0 and 'ERR batch MTP' not in log
 marker=f'strata Q6 center: enabled=0 C6 A2 R1 SG32 PF0 shapes=6144x2560,2560x10240'
 full=(out/'engine.log').read_text()
 markers=re.findall(r'^strata Q6 center: .*$',full,re.M)
 assert markers==[marker],'First-eligible host selection marker missing/duplicated/wrong mode'
 assert not re.search(r'ERR batch MTP|DEVICE_LOST|capture.*(?:failed|produced no graph)',full,re.I)
 captures=re.findall(r'^strata verify: captured (?:the \d+-token window \(upload no error, sync no error\)|the batch window over slots [0-9,]+)$',log,re.M)
 native_rows=re.findall(r'^strata batch: (\d+) windows, avg ([0-9.]+) rows,',log,re.M)
 assert len(native_rows)==len(native_windows)
 save(dst/'pair.json',{'unmeasured_warm_coherence_only':True,'q6_host_selection_marker':marker,'logged_capture_layouts':captures,'native_window_rows':native_rows,'status':'PASS_HTTP_TEXT_COUNTS_FINITE_AND_PAIRED','wall_s':elapsed,'output_sha256':[r['output_sha256'] for r in rows],
  'usage':[r['usage'] for r in rows],'work_timings':[{k:r['timings'][k] for k in ['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']} for r in rows],
  'native_windows':sum(map(int,native_windows)),'paired':sum(int(x) for x,y in pairs),'solo':sum(int(y) for x,y in pairs),'raw_native_token_ids_observed':False,'aggregate_output_tps':sum(r['usage']['completion_tokens'] for r in rows)/elapsed,'stream_metrics':[{'ttft_s':r['ttft_s'],'completion_wall_s':r['wall_s'],'decode_tps':r['client_decode_tps'],'whole_request_output_tps':r['usage']['completion_tokens']/r['wall_s'],'content_event_gap_ms':r['content_event_gap_ms']} for r in rows],
  'admission_gate':gate,'client_overlap_ns':overlap,'repeat':repeat,'exact_gate':'full UTF8 text/SSE content plus usage/native work; not raw token IDs'})
 print('S93 pair complete; release quiet CPU lease before functional requests.',flush=True)
def functional(out):
 assert (out/'prefill/complete.json').exists();dst=out/'functional';dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  rows=list(pool.map(lambda ab:send(BASE,'hotschmoe-dd',[{'role':'user','content':f'Return only the integer: {ab[0]} * {ab[1]}.'}],32),[(17,19),(23,29)]))
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
def prefill(out):
 assert (out/'coding/warm/pair.json').exists()
 dst=out/'prefill';dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
 reference=R/'peer-gather-serving-s38/mode0/screen';entries=[];first_value_emission_request=None
 assert not QSA_RE.findall((out/'engine.log').read_text()),'Short warm must not consume eligible QSA marker'
 for case in ['p4096','p32768']:
  for repeat in range(3):
   name=f'{case}-{repeat}';ref=json.loads((reference/(name+'.json')).read_text());body=ref['request']
   assert body['max_tokens']==256 and len(body['messages'])==1
   before=(out/'engine.log').stat().st_size;save(dst/(name+'-start.json'),{'byte_offset':before,'epoch_ns':time.time_ns(),'retained_S38_request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()})
   row=send(BASE,body['model'],body['messages'],body['max_tokens']);save(dst/(name+'.json'),row)
   assert row['request']==body,'Reconstructed request differs from exact retained S38 body'
   ww=work(row);expected_prompt=4059 if case=='p4096' else 32728
   assert row['text']==emitted(row) and row['text'] and row['usage']['prompt_tokens']==expected_prompt and row['usage']['completion_tokens']==256
   assert ww['cache_n']<=64 and row['usage']['prompt_tokens_details']['cached_tokens']==ww['cache_n']
   assert ww['prompt_n']+ww['cache_n']==expected_prompt and ww['predicted_n']==256
   assert all(math.isfinite(row[k]) for k in ['ttft_s','wall_s','client_decode_tps'])
   if out.name=='mode1':row_coherence(json.loads((D/'mode0/prefill'/(name+'.json')).read_text()),row)
   deadline=time.monotonic()+30
   while True:
    raw=(out/'engine.log').read_bytes()[before:];log=raw.decode('ascii');native=SERVE_RE.findall(log)
    if len(native)==1 and int(native[0][0])==expected_prompt and int(native[0][4])==256:break
    if time.monotonic()>=deadline:
     (dst/(name+'-partial-engine.log')).write_bytes(raw)
     raise RuntimeError('Native completion observation incomplete; preserve same server, no retry/generation')
    time.sleep(.1)
   (dst/(name+'-engine.log')).write_bytes(raw)
   n=native[0];assert tuple(map(int,[n[1],n[2],n[4],n[6],n[7]]))==(ww['cache_n'],ww['prompt_n'],ww['predicted_n'],ww['draft_n_accepted'],ww['draft_n'])
   parts=[]
   for m in re.finditer(r'^strata batch prefill quantum: allocated (\d+) requested (\d+) effective (\d+) eligible (\d+) from (\d+) to (\d+)$',log,re.M):
    C,requested,Q,eligible,start,end=map(int,m.groups());assert C==Q==12288 and requested==0 and eligible==0 and 0<=start<end<=expected_prompt
    chunks=[{'from':x,'to':min(x+Q,end)} for x in range(start,end,Q)]
    parts.append({'raw':m.group(0),'allocated_C':C,'effective_Q':Q,'from':start,'to':end,'source_derived_chunks':chunks,'interval_scope':'observed native read_part bounds; subdivisions derived from current Q0/C loop, not measured callback timestamps'})
   assert parts,'Native part bounds missing'
   assert not re.search(r'ERR batch MTP|DEVICE_LOST|capture.*(?:failed|produced no graph)',log,re.I)
   marker=None
   if case=='p4096':assert not QSA_RE.findall((out/'engine.log').read_text()),'4K must preserve old scorer path'
   else:
    marker=qsa_marker(out,0)
    assert any(part['to']>=16384 and any(c['to']-c['from']>=256 for c in part['source_derived_chunks']) for part in parts)
   value=value_marker(out,int(out.name=='mode1'))
   if value['count']==1 and first_value_emission_request is None:first_value_emission_request=name
   value['first_emission_request']=first_value_emission_request
   if case=='p32768' and out.name=='mode1':assert value['count']==1,'Missing ON selection for32K; effect unqualified'
   entry={'VALUE_marker':value,'name':name,'status':'PASS_LOCAL_GATES','usage':row['usage'],'work':ww,'native_summary_raw':SERVE_RE.search(log).group(0),'native_part_intervals':parts,'QSA_marker':marker,'matches_retained_S38_text_descriptive_only':row['text']==ref['text'],'native_prompt_tps':row['timings']['prompt_per_second'],'native_prompt_ms':row['timings']['prompt_ms'],'ttft_s':row['ttft_s'],'wall_s':row['wall_s'],'content_event_gap_ms':row['content_event_gap_ms'],'output_sha256':row['output_sha256']}
   entries.append(entry);save(dst/(name+'-verdict.json'),entry)
 save(dst/'complete.json',{'status':'PASS_SIX_REQUESTS_LOCAL_AND_AVAILABLE_CROSS_ARM_GATES','entries':entries,'fresh_actual_prompt_counts':[4059,32728],'cap':256,'QSA_marker':qsa_marker(out,0),'VALUE_marker':value_marker(out,int(out.name=='mode1')),'first_value_emission_request':first_value_emission_request,'not_independent_GPU_dispatch':True})
 print('S93 prefill phase complete; no additional warm generation.',flush=True)
p=argparse.ArgumentParser();p.add_argument('arm',choices=['0','1']);p.add_argument('phase',choices=['warm','prefill','functional']);a=p.parse_args();out=D/f'mode{a.arm}'
if a.phase=='warm':
 pair(out,0);assert not QSA_RE.findall((out/'engine.log').read_text());assert not any('prefill attention value pipeline selected' in l for l in (out/'engine.log').read_text().splitlines())
 if a.arm=='1':
  baseline=D/'mode0/coding/warm';candidate=out/'coding/warm'
  for i in range(2):row_coherence(json.loads((baseline/f'coding-{i}.json').read_text()),json.loads((candidate/f'coding-{i}.json').read_text()))
  x=json.loads((baseline/'pair.json').read_text());y=json.loads((candidate/'pair.json').read_text())
  assert all(x[k]==y[k] for k in ['work_timings','native_windows','paired','solo','native_window_rows','logged_capture_layouts'])
  save(candidate/'control-parity.json',{'status':'PASS_WARM_TEXT_SSE_USAGE_OBSERVABLE_NATIVE_WORK'})
elif a.phase=='prefill':prefill(out)
else:functional(out)
