from pathlib import Path
import json,collections,datetime,hashlib,re,argparse
parser=argparse.ArgumentParser(description="Read-only gateway/RPC recurrence audit; never executes captured content.")
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
P=args.bundle/'current-world-5';O=args.output
O.mkdir(parents=True,exist_ok=True)
def utc(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat() if t else None
def dump(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n')
g=[];bad=[];ident=[]
for f in [P/'gateway-events.jsonl',*sorted((P/'rpc').glob('*.jsonl'))]:
 ident.append(dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 for i,l in enumerate(f.open(),1):
  try:d=json.loads(l)
  except Exception as e:bad.append([str(f),i,str(e)]);continue
  if f.name=='gateway-events.jsonl':g.append(dict(d,line=i))
starts={d['request_id']:d for d in g if d['event']=='request_start'};ends={d['request_id']:d for d in g if d['event']=='request_end'}
active=set();peak=0;load={};rejectload=collections.Counter()
for d in sorted(g,key=lambda d:d['at']):
 if d['event']=='request_start':active.add(d['request_id']);load[d['request_id']]=len(active);peak=max(peak,len(active))
 elif d['event']=='request_end':active.discard(d['request_id'])
 elif d['event']=='request_rejected':rejectload[len(active)]+=1
messages=[];types=collections.Counter();errors=collections.Counter();retries=collections.Counter();notifies=[];schemas=collections.defaultdict(collections.Counter);settingkeys=collections.Counter()
for f in sorted((P/'rpc').glob('*.jsonl')):
 for i,l in enumerate(f.open(),1):
  try:d=json.loads(l)
  except:continue
  typ=d.get('type','world_input' if 'world_input'in d else '?');types[typ]+=1;schemas[typ].update(d.keys())
  if typ=='auto_retry_start':retries[d.get('errorMessage','')]+=1
  if typ=='extension_ui_request':notifies.append(dict(file=f.name,line=i,**d))
  if typ!='message_end':continue
  m=d.get('message',{})
  if m.get('role')!='assistant':continue
  rid=m.get('responseId');gid=rid.removeprefix('chatcmpl-') if rid else None;s=starts.get(gid);e=ends.get(gid)
  txt=json.dumps(m.get('content',[]),ensure_ascii=True)
  row=dict(file=f.name,line=i,observed=d.get('_observed_at'),observed_utc=utc(d.get('_observed_at')),timestamp=m.get('timestamp'),response_id=rid,request_id=gid,stop=m.get('stopReason'),raw_stop=m.get('rawStopReason'),error=m.get('errorMessage'),usage=m.get('usage'),bang=bool(re.search('!{32,}',txt)),max_bang=max([len(x) for x in re.findall('!+',txt)]+[0]),gateway_start=s,gateway_end=e,admitted_active_at_start=load.get(gid))
  if s:
   overlaps=[k for k,v in starts.items() if k!=gid and v['at']<(e or {'at':float('inf')})['at'] and ends.get(k,{'at':float('inf')})['at']>s['at']]
   row['overlapping_admitted_ids']=overlaps
  messages.append(row)
  if row['error']:errors[row['error']]+=1
summary=dict(gateway_events=collections.Counter(d['event'] for d in g),gateway_end_status=collections.Counter(d['status'] for d in ends.values()),range_utc=[utc(min(d['at'] for d in g)),utc(max(d['at'] for d in g))],gateway_schemas={k:dict(collections.Counter(z for d in g if d['event']==k for z in d if z!='line')) for k in set(d['event'] for d in g)},upstream_request_ids=collections.Counter(str(d.get('upstream_request_id','ABSENT')) for d in ends.values()),peak_admitted_intervals=peak,rejected429_active_intervals=rejectload,start_without_end=[v for k,v in starts.items() if k not in ends],end_without_start=[v for k,v in ends.items() if k not in starts],rpc_types=types,rpc_schemas={k:dict(v) for k,v in schemas.items()},assistant_message_end_count=len(messages),assistant_id_join_count=sum(m['gateway_start'] is not None for m in messages),assistant_no_response_id=sum(not m['response_id'] for m in messages),bang_count=sum(m['bang'] for m in messages),bang_gateway_status=collections.Counter((m['gateway_end']or{}).get('status','MISSING') for m in messages if m['bang']),errors=errors,retries=retries,parse_errors=bad)
dump('summary.json',summary);dump('messages.json',messages);dump('notifications.json',notifies);dump('input_identity.json',ident)
focus=['c5b6a71c2829ae886e2769e4','c8e555fd5002947404b397e5','6f24e162e1dcf235f2a28419','ffd2631c3e71645b09aaeb4d','7764163f49e3081bba7d8d86']
dump('focus.json',[m for m in messages if any(x in (m['response_id']or'') for x in focus)])
print(json.dumps({k:v for k,v in summary.items() if k not in ['rpc_schemas','gateway_schemas','errors','retries']},ensure_ascii=True,indent=2))
print('ERRORS',json.dumps(errors,ensure_ascii=True));print('RETRIES',json.dumps(retries,ensure_ascii=True))

rejections=[d for d in g if d['event']=='request_rejected']
aborts=[]
for m in messages:
 if not m['bang']:continue
 end=m['gateway_end'];unit=end['unit'];at=end['at']
 same=[d for d in rejections if d['unit']==unit]
 nearest=min(same,key=lambda d:abs(d['at']-at)) if same else None
 after=min((d for d in same if d['at']>=at),key=lambda d:d['at'],default=None)
 before=max((d for d in same if d['at']<at),key=lambda d:d['at'],default=None)
 aborts.append(dict(response_id=m['response_id'],unit=unit,abort_release_at=at,abort_release_utc=utc(at),nearest_rejection=nearest,nearest_delta_sec=nearest['at']-at if nearest else None,previous_delta_sec=before['at']-at if before else None,next_delta_sec=after['at']-at if after else None))
windows={str(w):dict(any_nearest=sum(x['nearest_delta_sec'] is not None and abs(x['nearest_delta_sec'])<=w for x in aborts),before=sum(x['previous_delta_sec'] is not None and abs(x['previous_delta_sec'])<=w for x in aborts),after=sum(x['next_delta_sec'] is not None and x['next_delta_sec']<=w for x in aborts)) for w in [0.1,1,5,30,60]}
dump('abort_admission_timing.json',dict(sign='rejection.at minus gateway client_disconnected.at; negative means before release',windows_sec=windows,rows=aborts))
print('ABORT429_WINDOWS',json.dumps(windows))
