#!/usr/bin/env python3
"""One ordered107prompt/64output coding pair, no generation warmup."""
import concurrent.futures,hashlib,json,math,re,sys,threading,time,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent;REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
sys.path.insert(0,str(REPO/'strata/flash-next'));from bench_serving import get,send
BASE='http://127.0.0.1:18085';ALIAS='qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata'
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def identity():
 x=get(BASE,'/v1/models');ids=[i['id']for i in x['data']];assert ids[0]=='hotschmoe-dd'and ALIAS in ids and ALIAS in(REPO/'evals/configs/models.yaml').read_text();return x
def pair(out,repeat):
 dst=out/'pair';dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
 initial=(out/'engine.log').read_bytes();save(dst/'window-start.json',{'byte_offset':len(initial),'epoch_ns':time.time_ns()})
 def request(i):
  ref=json.loads((R.parent/f'flashnext_ista_gsq_rco_iq3xxs_strata_accounting_20261014/concurrency/coding-0-{i}.json').read_text())
  messages=ref['request']['messages']
  call_start=time.monotonic_ns();epoch_start=time.time_ns()
  life={'client_call_start_monotonic_ns':call_start,'client_call_start_epoch_ns':epoch_start,'model':'hotschmoe-dd','messages':messages,'max_new_tokens':64}
  save(dst/f'coding-{i}-lifetime.json',life)
  try:row=send(BASE,'hotschmoe-dd',messages,64)
  except Exception as e:
   life.update(client_call_end_monotonic_ns=time.monotonic_ns(),client_call_end_epoch_ns=time.time_ns(),error_type=type(e).__name__);save(dst/f'coding-{i}-lifetime.json',life);raise
  life.update(client_call_end_monotonic_ns=time.monotonic_ns(),client_call_end_epoch_ns=time.time_ns());save(dst/f'coding-{i}-lifetime.json',life)
  row['client_lifetime']=life
  required=['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']
  assert all(k in row['timings'] and isinstance(row['timings'][k],(int,float)) and not isinstance(row['timings'][k],bool) and math.isfinite(row['timings'][k]) and row['timings'][k]>=0 for k in required),'Missing/nonfinite/negative native work field'
  row['prefix_matches_s16']=ref['text'].startswith(row['text']) if repeat==0 else None;save(dst/f'coding-{i}.json',row)
  assert row['usage']['completion_tokens']==64 and row['usage']['prompt_tokens']==107
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
 save(dst/'pair.json',{'instrumented_one_pair_only':True,'q6_host_selection_marker':marker,'logged_capture_layouts':captures,'native_window_rows':native_rows,'status':'PASS_HTTP_TEXT_COUNTS_FINITE_AND_PAIRED','wall_s':elapsed,'output_sha256':[r['output_sha256'] for r in rows],
  'usage':[r['usage'] for r in rows],'work_timings':[{k:r['timings'][k] for k in ['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']} for r in rows],
  'native_windows':sum(map(int,native_windows)),'paired':sum(int(x) for x,y in pairs),'solo':sum(int(y) for x,y in pairs),'raw_native_token_ids_observed':False,'aggregate_output_tps':sum(r['usage']['completion_tokens'] for r in rows)/elapsed,'stream_metrics':[{'ttft_s':r['ttft_s'],'completion_wall_s':r['wall_s'],'decode_tps':r['client_decode_tps'],'whole_request_output_tps':r['usage']['completion_tokens']/r['wall_s'],'content_event_gap_ms':r['content_event_gap_ms']} for r in rows],
  'admission_gate':gate,'client_overlap_ns':overlap,'repeat':repeat,'exact_gate':'full UTF8 text/SSE content plus usage/native work; not raw token IDs'})
 print('S95 ordered64x2 pair complete; no generation warmup.',flush=True)

if __name__=='__main__':pair(D/'run',0)
