#!/usr/bin/env python3
"""S79 ordered native-prompt-start pair + existing functionals; no native-token-ID claim."""
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
  if repeat:
   prompt=messages[0]['content'].split('] ',1)[1]
   messages=[{'role':'user','content':f'[Flash-Next concurrent coding repeat {repeat} slot {i}] '+prompt}]
  call_start=time.monotonic_ns();epoch_start=time.time_ns()
  life={'client_call_start_monotonic_ns':call_start,'client_call_start_epoch_ns':epoch_start,'model':'hotschmoe-dd','messages':messages,'max_new_tokens':1500}
  save(dst/f'coding-{i}-lifetime.json',life)
  try:row=send(BASE,'hotschmoe-dd',messages,1500)
  except Exception as e:
   life.update(client_call_end_monotonic_ns=time.monotonic_ns(),client_call_end_epoch_ns=time.time_ns(),error_type=type(e).__name__);save(dst/f'coding-{i}-lifetime.json',life);raise
  life.update(client_call_end_monotonic_ns=time.monotonic_ns(),client_call_end_epoch_ns=time.time_ns());save(dst/f'coding-{i}-lifetime.json',life)
  row['client_lifetime']=life
  row['prefix_matches_s16']=ref['text'].startswith(row['text']) if repeat==0 else None;save(dst/f'coding-{i}.json',row)
  assert row['usage']['completion_tokens']==1500
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
  commit_rows=re.findall(r'^strata batch commit overlap: asked=(\d+) deferred=(\d+) joined=(\d+)$',log,re.M)
  candidate=out.name=='mode1'
  if pairs and native_windows and (commit_rows or not candidate):break
  time.sleep(.1)
 else:raise RuntimeError('Native summary incomplete: preserve server; do not regenerate requests')
 (dst/'engine-segment.log').write_text(log,encoding='utf8')
 assert sum(int(x) for x,y in pairs)>0 and 'ERR batch MTP' not in log
 commit={'enabled':int(candidate),'asked':sum(int(a) for a,b,c in commit_rows),'deferred':sum(int(b) for a,b,c in commit_rows),'joined':sum(int(c) for a,b,c in commit_rows)}
 if candidate:assert commit['asked']==commit['deferred']==commit['joined']==sum(map(int,native_windows))>0,'Deferred commit/join activation incomplete'
 else:assert not commit_rows,'OFF unexpectedly deferred a batch commit'
 save(dst/'pair.json',{'commit_overlap':commit,'status':'PASS_HTTP_TEXT_COUNTS_FINITE_AND_PAIRED','wall_s':elapsed,'output_sha256':[r['output_sha256'] for r in rows],
  'usage':[r['usage'] for r in rows],'work_timings':[{k:r['timings'].get(k) for k in ['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']} for r in rows],
  'native_windows':sum(map(int,native_windows)),'paired':sum(int(x) for x,y in pairs),'solo':sum(int(y) for x,y in pairs),'raw_native_token_ids_observed':False,
  'admission_gate':gate,'client_overlap_ns':overlap,'repeat':repeat,'exact_gate':'full UTF8 text/SSE content plus usage/native work; not raw token IDs'})
 print('S79 pair complete; release quiet CPU lease before functional requests.',flush=True)
def functional(out):
 assert (out/'coding/repeat3/pair.json').exists();dst=out/'functional';dst.mkdir(exist_ok=False);save(dst/'models.json',identity())
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
p=argparse.ArgumentParser();p.add_argument('arm',choices=['0','1']);p.add_argument('phase',choices=['warm','timed','functional']);a=p.parse_args();out=D/f'mode{a.arm}'
if a.phase=='warm':
 pair(out,0)
 if a.arm=='1':
  baseline=D/'mode0/coding/warm';candidate=out/'coding/warm'
  for i in range(2):
   x=json.loads((baseline/f'coding-{i}.json').read_text());y=json.loads((candidate/f'coding-{i}.json').read_text())
   def emitted(z):return ''.join(c.get('choices',[{}])[0].get('delta',{}).get('content') or '' for c in z['chunks'] if c.get('choices'))
   assert x['request']==y['request'] and x['text']==y['text']==emitted(x)==emitted(y) and x['usage']==y['usage'],'Warm control/candidate numerical coherence mismatch'
  save(candidate/'control-parity.json',{'status':'PASS_FULL_REQUEST_TEXT_SSE_USAGE','native_work_gate':'retained for exact final comparison; not normalized'})
elif a.phase=='timed':
 assert json.loads((out/'coding/warm/pair.json').read_text())['status']=='PASS_HTTP_TEXT_COUNTS_FINITE_AND_PAIRED'
 for repeat in [1,2,3]:pair(out,repeat)
else:functional(out)
