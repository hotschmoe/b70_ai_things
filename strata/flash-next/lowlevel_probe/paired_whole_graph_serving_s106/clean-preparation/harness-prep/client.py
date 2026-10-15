#!/usr/bin/env python3
"""S106 ordered native-prompt-start pair + existing functionals; no native-token-ID claim."""
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
def pair(out,repeat):
 output_tokens=256 if repeat==0 else 1500
 parent=out/'coding';parent.mkdir(exist_ok=True);dst=parent/('warm' if repeat==0 else f'repeat{repeat}');dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
 initial=(out/'engine.log').read_bytes();save(dst/'window-start.json',{'byte_offset':len(initial),'epoch_ns':time.time_ns()})
 def request(i):
  ref=json.loads((R.parent/f'flashnext_ista_gsq_rco_iq3xxs_strata_accounting_20261014/concurrency/coding-0-{i}.json').read_text())
  messages=ref['request']['messages']
  if repeat:
   prompt=messages[0]['content'].split('] ',1)[1]
   messages=[{'role':'user','content':f'[Flash-Next concurrent coding repeat {repeat} slot {i}] '+prompt}]
  call_start=time.monotonic_ns();epoch_start=time.time_ns()
  life={'client_call_start_monotonic_ns':call_start,'client_call_start_epoch_ns':epoch_start,'model':'hotschmoe-dd','messages':messages,'max_new_tokens':output_tokens,'strata_tune':{'spec_min_p':0}}
  save(dst/f'coding-{i}-lifetime.json',life)
  try:row=send(BASE,'hotschmoe-dd',messages,output_tokens,spec_min_p=0)
  except Exception as e:
   life.update(client_call_end_monotonic_ns=time.monotonic_ns(),client_call_end_epoch_ns=time.time_ns(),error_type=type(e).__name__);save(dst/f'coding-{i}-lifetime.json',life);raise
  life.update(client_call_end_monotonic_ns=time.monotonic_ns(),client_call_end_epoch_ns=time.time_ns());save(dst/f'coding-{i}-lifetime.json',life)
  row['client_lifetime']=life
  required=['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']
  assert all(k in row['timings'] and isinstance(row['timings'][k],(int,float)) and not isinstance(row['timings'][k],bool) and math.isfinite(row['timings'][k]) and row['timings'][k]>=0 for k in required),'Missing/nonfinite/negative native work field'
  row['prefix_matches_s16']=ref['text'].startswith(row['text']) if repeat==0 else None;save(dst/f'coding-{i}.json',row)
  assert row['usage']['completion_tokens']==output_tokens and row['usage']['prompt_tokens']==107
  if repeat==0:assert row['prefix_matches_s16']
  else:assert row['timings']['cache_n']<=64,'Timed prompt not fresh'
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
 full=(out/'engine.log').read_text()
 markers=re.findall(r'^strata MTP pair graph: .*$',full,re.M)
 first_key=None
 if out.name=='mode1':
  assert len(markers)==1,'First-eligible ON host selection marker missing/duplicated'
  match=re.fullmatch(r'strata MTP pair graph: enabled=1 depth=2 rerecorded=1 T0=([1-3]) T1=([1-3]) host-selection-only',markers[0])
  assert match,'Malformed/out-of-bound first-key host marker'
  first_key=[int(v)for v in match.groups()]
 else:assert not markers,'OFF arm emitted ON marker'
 assert not re.search(r'^strata IQ2_XS multi:',full,re.M),'Unrelated IQ2 multi candidate unexpectedly ON'
 assert not re.search(r'ERR batch MTP|DEVICE_LOST|capture.*(?:failed|produced no graph)',full,re.I)
 captures=re.findall(r'^strata verify: captured (?:the \d+-token window \(upload no error, sync no error\)|the batch window over slots [0-9,]+)$',log,re.M)
 native_rows=re.findall(r'^strata batch: (\d+) windows, avg ([0-9.]+) rows,',log,re.M)
 assert len(native_rows)==len(native_windows)
 save(dst/'pair.json',{'pair_graph_host_selection_marker':markers[0]if markers else None,'first_host_selected_T0_T1':first_key,'all_replayed_keys_observed':False,'logged_capture_layouts':captures,'native_window_rows':native_rows,'status':'PASS_HTTP_TEXT_COUNTS_FINITE_AND_PAIRED','wall_s':elapsed,'output_sha256':[r['output_sha256'] for r in rows],
  'usage':[r['usage'] for r in rows],'work_timings':[{k:r['timings'][k] for k in ['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']} for r in rows],
  'native_windows':sum(map(int,native_windows)),'paired':sum(int(x) for x,y in pairs),'solo':sum(int(y) for x,y in pairs),'raw_native_token_ids_observed':False,'aggregate_output_tps':sum(r['usage']['completion_tokens'] for r in rows)/elapsed,'stream_metrics':[{'ttft_s':r['ttft_s'],'completion_wall_s':r['wall_s'],'decode_tps':r['client_decode_tps'],'whole_request_output_tps':r['usage']['completion_tokens']/r['wall_s'],'content_event_gap_ms':r['content_event_gap_ms']} for r in rows],
  'admission_gate':gate,'client_overlap_ns':overlap,'repeat':repeat,'output_tokens_per_stream':output_tokens,'exact_gate':'full UTF8 text/SSE content plus usage/native work; not raw token IDs'})
 if out.name=='mode1':require_control_pair(out,repeat)
 print('S106 pair complete; release quiet CPU lease before functional requests.',flush=True)
def require_control_pair(out,repeat):
 # Fail this ON pair before sending the next pair/functionals. Never normalize
 # per-slot work, capture layouts or admission order after seeing a mismatch.
 label='warm'if repeat==0 else f'repeat{repeat}'
 baseline=D/'mode0/coding'/label;candidate=out/'coding'/label
 x=[json.loads((baseline/f'coding-{i}.json').read_text())for i in range(2)]
 y=[json.loads((candidate/f'coding-{i}.json').read_text())for i in range(2)]
 pairs=[json.loads((z/'pair.json').read_text())for z in [baseline,candidate]]
 def emitted(z):return ''.join(c.get('choices',[{}])[0].get('delta',{}).get('content')or''for c in z['chunks']if c.get('choices'))
 checks={'request_text_SSE_usage':all(x[i]['request']==y[i]['request']and x[i]['text']==y[i]['text']==emitted(x[i])==emitted(y[i])and x[i]['usage']==y[i]['usage']for i in range(2)),
  'native_per_request_work':pairs[0]['work_timings']==pairs[1]['work_timings'],
  'native_windows_pair_solo_rows_captures':all(pairs[0][k]==pairs[1][k]for k in ['native_windows','paired','solo','native_window_rows','logged_capture_layouts'])}
 if repeat==0:
  startup=[]
  for arm in [0,1]:
   folder=D/f'mode{arm}';offset=json.loads((folder/'coding/warm/window-start.json').read_text())['byte_offset']
   startup.append(re.findall(r'^strata verify: captured .*$',(folder/'engine.log').read_bytes()[:offset].decode(),re.M))
  checks['startup_logged_captures']=startup[0]==startup[1]
 save(candidate/'control-parity.json',{'status':'PASS_EXACT_OBSERVABLE_PAIR'if all(checks.values())else'FAIL_OR_UNMATCHED_STOP_REMAINING_ARM_WORK','checks':checks,'normalization':None})
 assert all(checks.values()),'ON/control mismatch; stop remaining requests and normal-stop/health this server'
def functional(out):
 assert (out/'coding/repeat3/pair.json').exists();dst=out/'functional';dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
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
p=argparse.ArgumentParser();p.add_argument('arm',choices=['0','1']);p.add_argument('phase',choices=['warm','timed','functional']);a=p.parse_args();out=D/f'mode{a.arm}'
if a.phase=='warm':pair(out,0)
elif a.phase=='timed':
 assert json.loads((out/'coding/warm/pair.json').read_text())['status']=='PASS_HTTP_TEXT_COUNTS_FINITE_AND_PAIRED'
 for repeat in [1,2,3]:pair(out,repeat)
else:functional(out)
