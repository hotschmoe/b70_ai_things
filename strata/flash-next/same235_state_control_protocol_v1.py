"""Bounded semantic raw recollection for exact serial state controls."""
import json,re,time
from pathlib import Path
from api_owned_terminal_association_v2 import associate_events
from same235_state_control_v1 import require,contract
from api_cache_positive_contract_v2 import actual_phase_legs
PREFIXES=('ERR','SERR','FATAL','INFO ','READY','T ','BT ','DONE ','BDONE ','BADM ','SBF ','BCPUBLIC ','BCHAIN ','PCL ','BYIELD ','LP','MIRROR_USM ','SLOT_USM ')
MAX_RAW_LINE=64<<20
def events(path,allow_incomplete_tail=False):
 rows=[];size=0
 with Path(path).open('rb') as f:
  while True:
   raw=f.readline(MAX_RAW_LINE+1)
   if not raw:break
   require(len(raw)<=MAX_RAW_LINE,'Bounded raw trace line exceeded before allocation/decode')
   if not (b'"kind": "engine_' in raw or b'"kind": "native_send"' in raw or any(('"line": "'+prefix).encode() in raw for prefix in PREFIXES)):continue
   require(len(raw)<=64<<20,'Bounded raw semantic line exceeded before JSON decode')
   if not raw.endswith(b'\n'):
    if allow_incomplete_tail:break
    raise ValueError('Final semantic trace line missing newline')
   row=json.loads(raw)
   if row['kind'].startswith('engine_') or row['kind']=='native_send' or row['kind']=='native_receive' and row['line'].startswith(PREFIXES):
    rows.append(row);size+=len(raw);require(len(rows)<=100000 and size<=64<<20,'Bounded source semantic trace exceeded')
 return rows

def exact_body(row,messages,max_new,fresh):
 expected={'model':'hotschmoe-dd','messages':messages,'stream':True,'temperature':0,'max_tokens':max_new,'frequency_penalty':0,'presence_penalty':0,'user':row['request_id'],'chat_template_kwargs':{'enable_thinking':False},'reasoning_budget_tokens':0,'strata_fresh':fresh}
 require(row['request']==expected and type(row['request']['strata_fresh']) is bool and row['error'] is None and row['done_received'] is True and row['cancel_requested'] is False,'Exact natural control client body/terminal differs')

def warm_binding(rows,actual,job,target_begin=None):
 begins={r['call']:r for r in rows if r['kind']=='engine_begin'};ends={r['call']:r for r in rows if r['kind']=='engine_end'};client=actual['rows'];require(len(client)==2 and {r['client_index'] for r in client}=={0,1},'Exact resetwarm two-client roster required');fixture,_=contract.fixture_binding();joined=[]
 for row in client:
  i=row['client_index'];exact_body(row,job['warm_messages'][i],32,True);ids=fixture['fixtures']['warm'][i]['ids'];calls=[k for k,b in begins.items() if b['submitted_ids']==ids and b['rendered_prompt']['messages']==job['warm_messages'][i] and row['sent_epoch']<=b['epoch']<=row['finished_epoch']];require(len(calls)==1,'Warm client/source IDs/messages/epoch association differs');call=calls[0];require(call in ends and begins[call]['max_new']==32 and not ends[call]['cancelled'],'Actual owned resetwarm terminal missing');terminal=associate_events(rows,{call})[call];actual_phase_legs(rows,{call},True)
  if target_begin is not None:require(ends[call]['sequence']<target_begin['sequence'] and terminal['actual_native_terminal']['sequence']<target_begin['sequence'],'Target began before own resetwarm native EOF')
  joined.append({'logical_index':i,'call':call,'native_terminal_sequence':terminal['actual_native_terminal']['sequence'],'engine_end_sequence':ends[call]['sequence']})
 require(len({r['call'] for r in joined})==2,'Warm source call borrowed');return joined

def await_warm(path,actual,job,timeout=30):
 deadline=time.monotonic()+timeout;last=None
 while True:
  try:return warm_binding(events(path,allow_incomplete_tail=True),actual,job)
  except (ValueError,KeyError) as error:last=error
  require(time.monotonic()<deadline,'Owned warm terminal/source before scalar target missing: '+str(last));time.sleep(.05)

def recollect(rows,controls,stages):
 require(not any(r['kind']=='native_receive' and r['line'].startswith(('ERR','SERR','FATAL')) for r in rows),'Actual native error/refusal cannot qualify controls')
 begins={r['call']:r for r in rows if r['kind']=='engine_begin'};ends={r['call']:r for r in rows if r['kind']=='engine_end'};require(len(begins)==14 and set(begins)==set(ends),'Actual2initialwarm+8resetwarm+4serialtarget roster required');pids={b['engine_pid'] for b in begins.values()};generations={b['engine_generation'] for b in begins.values()};require(len(pids)==len(generations)==1 and all(type(v) is int and v>0 for v in pids|generations) and all(e['engine_pid'] in pids and e['engine_generation'] in generations for e in ends.values()),'One actual engine PID/incarnation across all14 calls required');terminal=associate_events(rows,set(begins));result=[];used=set();used_warm=set();previous_target_end=None
 for control in controls:
  job=control['declared_job'];client=control['actual_target_client']['rows'];require(len(client)==1,'Actual serial target client roster differs');client=client[0]
  calls=[k for k,b in begins.items() if b['submitted_ids']==job['ids'] and b['rendered_prompt']['messages']==job['messages'] and client['sent_epoch']<=b['epoch']<=client['finished_epoch']];require(len(calls)==1,'Target actual source/token/client epoch association differs');call=calls[0];require(call not in used,'Actual target call borrowed');used.add(call);begin=begins[call];end=ends[call];exact_body(client,job['messages'],64,job['request_policy']['strata_fresh']);warm=warm_binding(rows,control['actual_warm_client'],job,begin);require(control['actual_warm_native_predecessor']==warm,'Saved actual own warm predecessor differs');warm_calls={r['call'] for r in warm};require(not used_warm&warm_calls,'Own resetwarm cohort cannot be borrowed across controls');require(previous_target_end is None or all(previous_target_end<begins[k]['sequence'] for k in warm_calls),'Next resetwarm began before previous scalar target EOF');used_warm.update(warm_calls);previous_target_end=end['sequence'];require(begin['max_new']==64 and not end['cancelled'],'Natural target budget/cancel differs')
  sends=[r for r in rows if r['kind']=='native_send' and r.get('call')==call and r['line'].startswith(('GEN ','BGEN '))];require(len(sends)==1,'Serial target must have one actual native GEN leg');send=sends[0];require(send['engine_pid']==begin['engine_pid']==end['engine_pid'],'Actual target native process association differs');tokens=send['line'].split();fields=dict(re.findall(r'(\w+)=([^ ]+)',send['line']));fresh=int(job['request_policy']['strata_fresh']);require(tokens[:2]==['GEN','64'] and fields.get('seed')=='1' and fields.get('fresh')==str(fresh) and set(fields)=={'seed','fresh','rid'} and [int(x) for x in tokens[-1].split(',')]==job['ids'],'Exact actual GEN64 greedy/fresh/RID/pinabsent/235 differs');rid=int(fields['rid']);require(rid==end['rid'],'Actual source RID/end differs')
  resets=[r for r in rows if r['kind']=='native_receive' and r['line'].startswith('BCPUBLIC reset ') and re.search(r'\brid='+str(rid)+r'\b',r['line'])];require(len(resets)==1,'Actual unique allstage reset required');reset=resets[0];require(reset['engine_pid']==send['engine_pid'],'Foreign native process reset refused');f=dict(re.findall(r'(\w+)=([^ ]+)',reset['line']));require(send['sequence']<reset['sequence']<terminal[call]['actual_native_terminal']['sequence'] and f['fresh']==str(fresh) and f['pin_present']=='0' and f['read_from']==f['lookup_ceiling']=='0' and int(f['stages'])==stages,'Actual source reset identity/stage/counter chronology differs')
  last=end['engine_last'];require(last['prompt_tokens']==last['prompt_read']==235 and last['reused']==0 and last['finish'] in ('stop','length') and terminal[call]['actual_client_cancelled'] is False,'Actual full235 no-reuse natural native terminal differs')
  result.append({'control':job['control'],'actual_warm_predecessor':warm,'call':call,'RID':rid,'actual_reset_line':reset['line'],'actual_reset_sequence':reset['sequence'],'actual_terminal':terminal[call],'actual_output_ids':end['generated_ids'],'actual_full_prompt_read':235,'actual_reused':0,'reread_to_runtime_field_observed':False,'scope':'Token/reset serial control only; no native state/full49 fidelity grant'})
 warm=set(begins)-used;fixture,_=contract.fixture_binding();warm_ids=[r['ids'] for r in fixture['fixtures']['warm']]
 require(len(warm)==10 and all(begins[k]['submitted_ids'] in warm_ids and begins[k]['max_new']==32 and not ends[k]['cancelled'] for k in warm),'Actual genuine warm47/fresh32 source roster differs');actual_phase_legs(rows,warm,True)
 initial=sorted(warm,key=lambda k:begins[k]['sequence'])[:2];require(used_warm==warm-set(initial),'Exact8 per-control warm calls must cover allwarm exceptinitial2');limit=max(ends[k]['sequence'] for k in initial);first=min(begins[k]['sequence'] for k in initial)
 native_events=[r for r in rows if r['kind']=='native_receive' and r['line'].startswith('SBF batch_event ')];enginegens=set();multi=[]
 for r in native_events:
  f=dict(re.findall(r'(\w+)=([^ ]+)',r['line']));require(r['engine_pid'] in pids and int(f['pid']) in pids and int(f['enginegen'])>0 and int(f['event'])>0,'Actual native warm event PID/incarnation invalid');enginegens.add(f['enginegen'])
  if first<r['sequence']<limit and f['rows']=='2' and f['completed']=='1':require(f['active_mask']=='3','Actual two-row warm geometry mask differs');multi.append(r)
 require(len(enginegens)==1 and multi,'Actual same-engine initial two-row warm event missing')

 return result
