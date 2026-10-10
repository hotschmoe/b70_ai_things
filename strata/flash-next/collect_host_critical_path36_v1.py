#!/usr/bin/env python3
"""Read-only H36 host coverage. Never GPU/model/math/speed qualification."""
import argparse,hashlib,json,re
from pathlib import Path
KINDS={n:i+1 for i,n in enumerate(['Request','Reset','PromptSpan','Verify','StageInputs','PlePrefetch','PleGather','Capture','CommitCapture','ReplaySubmit','ReplayWait','Sample','LPReadback','LPCPU','Commit','CommitWait','TEmit','DoneEmit','GraphCreate','GraphHit','GraphPreexisting','TPrintf','StdoutFlush','GraphFinalize','CaptureWait'])}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(paths,log,binding,requests,stages):
 require(re.fullmatch('[0-9a-f]{64}',binding) is not None,'Explicit actual prerequisite binding required')
 require(0<len(paths)==len(requests)<=4 and stages==[[0,0,48]],'Bounded onecard serial trace/request/stage scope differs')
 text=Path(log).read_text();closures=re.findall(r'HOST_TRACE36_CLOSED request=(\d+) success=(\d+)',text)
 require(closures==[(str(i+1),'1') for i in range(len(paths))],'Missing/duplicate/failed actual file-close proof')
 sessions=re.findall(r'HOST_TRACE36_SESSION requests=(\d+) thread_error=(\d+) export_error=(\d+) request_limit=(\d+)',text)
 require(sessions==[(str(len(paths)),'0','0','0')],'Missing/failed terminal host tracer ownership/free boundary')
 graphs={};last_ns=0;pid=None;reports=[]
 for ordinal,(path,request) in enumerate(zip(paths,requests),1):
  rows=[json.loads(x) for x in Path(path).read_text().splitlines()];meta,events,footer=rows[0],rows[1:-1],rows[-1]
  require(meta['schema']==36 and meta['clock']=='CLOCK_MONOTONIC' and meta['device_clock']=='not_requested' and meta['binding']==binding,'Schema/clock/binding differs')
  require(meta['rid']==request.get('rid',0) and meta['slotgen']==request.get('slotgen',0),'Public request identity differs')
  require(meta['request']==ordinal and meta['pid']>0 and meta['fresh']==request['fresh']==1 and meta['prompt']==len(request['ids']),'Request/input association differs')
  if pid is None:pid=meta['pid']
  require(meta['pid']==pid and meta['complete'] is True and all(meta[k] is False for k in ['overflow','thread_error','clock_error']),'Incomplete/overflow/foreign writer/clock failure')
  require(meta['capacity']==32768 and len(events)==meta['events']<=32768 and footer=={'footer':True,'records':len(events),'write_error':False},'Raw extent/footer differs')
  opens={};closed=set();durations=[];token_events=[];counts={};stage_windows={};paired={};last_replay=None;token_pairs=[]
  for seq,e in enumerate(events):
   require(type(e['ns']) is int and e['ns']>0 and e['ns']>=last_ns and e['seq']==seq,'Clock/sequence missing or reordered')
   last_ns=e['ns'];kind=e['kind'];require(kind in KINDS.values() and e['edge'] in (0,1,2),'Unknown kind/edge')
   counts[kind]=counts.get(kind,0)+1
   if e['dev']>=0:require([e['dev'],e['lb'],e['le']] in stages and e['queue']>0,'Wrong stage/device/queue identity')
   if e['edge']==1:
    require(e['span']>0 and e['span'] not in opens and e['span'] not in closed,'Duplicate span')
    opens[e['span']]=e
   elif e['edge']==2:
    require(e['span'] in opens,'Unmatched scope end');begin=opens.pop(e['span']);closed.add(e['span'])
    require(begin['kind']==kind and (begin['dev'],begin['lb'],begin['le'],begin['queue'])==(e['dev'],e['lb'],e['le'],e['queue']),'Scope ownership differs')
    durations.append({'kind':kind,'span':e['span'],'ns':e['ns']-begin['ns']});paired[e['span']]=(begin,e)
    if kind==KINDS['ReplayWait']:last_replay=e
    if kind==KINDS['Verify']:
     key=(e['dev'],e['lb'],e['le']);stage_windows.setdefault(key,[]).append((e['pos'],e['T']))
   if kind in (KINDS['GraphCreate'],KINDS['GraphPreexisting']):
    require(e['graph']>0 and e['graph'] not in graphs,'Stale/duplicate graph generation');graphs[e['graph']]=(e['dev'],e['lb'],e['le'],e['queue'])
   elif kind in (KINDS['GraphHit'],KINDS['ReplaySubmit'],KINDS['ReplayWait'],KINDS['TEmit']):
    require(e['graph'] in graphs and graphs[e['graph']]==(e['dev'],e['lb'],e['le'],e['queue']),'Unregistered/stale graph ownership')
   if kind==KINDS['TEmit']:
    require(e['edge']==0 and last_replay is not None,'Token marker without completed replay')
    require(last_replay['T']==1 and e['pos']==last_replay['pos']+1 and (e['graph'],e['dev'],e['lb'],e['le'],e['queue'])==(last_replay['graph'],last_replay['dev'],last_replay['lb'],last_replay['le'],last_replay['queue']),'Token marker not associated with actual final replay/input position')
    require(seq+1<len(events) and events[seq+1]['kind']==KINDS['TPrintf'] and events[seq+1]['edge']==1,'Token marker missing corresponding immediate printf begin')
    token_pairs.append(events[seq+1]['span']);token_events.append(e)
  require(not opens,'Open scopes retained as invalid partial coverage')
  require(counts.get(KINDS['TPrintf'])==2*len(token_events) and len(set(token_pairs))==len(token_events) and all(span in paired and paired[span][0]['kind']==KINDS['TPrintf'] for span in token_pairs),'Actual token printf end/ownership coverage differs')
  require(counts.get(KINDS['StdoutFlush'])==2*len(token_events),'Actual one-token serial round flush coverage differs')
  require(token_events and [e['value'] for e in token_events]==request['output_ids'] and len(token_events)==meta['produced']==int(request['done'].split()[1]),'Emitted token coverage differs')
  require([e['pos'] for e in token_events]==list(range(len(request['ids']),len(request['ids'])+len(token_events))),'Absolute emitted-token positions differ')
  for kind in ['Request','Reset','DoneEmit']:require(counts.get(KINDS[kind])==2,'Missing whole-request boundary '+kind)
  require((len(request['ids'])==1 or counts.get(KINDS['PromptSpan'],0)>0) and counts.get(KINDS['ReplaySubmit'],0)==counts.get(KINDS['ReplayWait'],0),'Missing prompt/submit/wait coverage')
  require(set(stage_windows)=={tuple(x) for x in stages},'Missing actual stage windows')
  expected=list(range(len(request['ids'])+len(token_events)-1))
  for stage,windows in stage_windows.items():
   covered=[p for pos,T in windows for p in range(pos,pos+T)]
   require(covered==expected,'Actual complete serial prompt/decode row coverage missing/duplicate')
  if request.get('LP'):require(counts.get(KINDS['LPReadback'])==counts.get(KINDS['LPCPU'])==2*len(token_events),'LP20 readback/CPU scope coverage differs')
  reports.append({'request':ordinal,'trace_sha256':sha(path),'events':len(events),'scope_durations':durations,'token_ns':[e['ns'] for e in token_events]})
 return {'schema':36,'passed':True,'scope':'bounded native serial HOST coverage only; durations overlap and are not additive','device_profile_qualified':False,'actual_GPU_run_qualified':False,'full_model_math_qualified':False,'speed_qualified':False,'requests':reports,'log_sha256':sha(log)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,nargs='+',required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--requests',type=Path,required=True);p.add_argument('--binding',required=True);p.add_argument('--stages',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 require(not a.output.exists(),'Output must be new');report=collect(a.traces,a.log,a.binding,json.loads(a.requests.read_text()),json.loads(a.stages));a.output.write_text(json.dumps(report,indent=2)+'\n')
