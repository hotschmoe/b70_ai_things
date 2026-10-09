#!/usr/bin/env python3
"""Bounded real simultaneous HTTP/SSE cohort; math evidence is native raw data."""
import argparse,concurrent.futures,json,socket,threading,time,urllib.request,urllib.parse
from pathlib import Path
from c1_serve_controller_combined_v6 import leased

def cohort(url,model,messages,output,cancel_index=None,max_new=64,timeout=2400):
 n=len(messages)
 if n not in (2,4,6) or not 2<=max_new<=64 or cancel_index is not None and not 0<=cancel_index<n:raise ValueError('Bounded2/4/6 scope')
 parsed=urllib.parse.urlparse(url)
 if parsed.hostname not in ('127.0.0.1','localhost') or parsed.scheme!='http':raise ValueError('Qualification uses owned local API only')
 barrier=threading.Barrier(n);rows=[None]*n;out=Path(output);out.mkdir(parents=True,exist_ok=False)
 with urllib.request.urlopen(url+'/v1/models',timeout=30) as handle:models=json.load(handle)
 names=[r['id'] for r in models['data']]
 if not names or names[0]!='hotschmoe-dd' or model not in names:raise ValueError('Actual served primary/model identity differs')
 (out/'models.json').write_text(json.dumps(models,ensure_ascii=True,indent=2)+'\n')
 def request(index):
  rid=f'batch-numerical-{n}-{index}';body={'model':model,'messages':messages[index],'stream':True,'temperature':0,'max_tokens':max_new,'frequency_penalty':0,'presence_penalty':0,'user':rid,'chat_template_kwargs':{'enable_thinking':False},'reasoning_budget_tokens':0};record={'request_id':rid,'request':body,'client_index':index,'events':[],'cancel_requested':False,'done_received':False,'error':None};response=None
  try:
   barrier.wait(timeout=30);record['sent_epoch']=time.time();req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Request-ID':rid})
   response=urllib.request.urlopen(req,timeout=timeout);record['http_status']=response.status
   for raw in response:
    line=raw.decode('utf-8').strip()
    if not line.startswith('data:'):continue
    data=line[5:].strip();record['events'].append({'epoch':time.time(),'data':data})
    if data=='[DONE]':record['done_received']=True;break
    event=json.loads(data)
    if event.get('error'):raise ValueError('Actual API error '+str(event['error']))
    choices=event.get('choices',[]);content=choices and choices[0].get('delta',{}).get('content')
    if cancel_index==index and content:
     record['cancel_requested']=True;record['cancel_epoch']=time.time()
     # Real HTTP reader/socket close, not a synthetic Event set in an engine mock.
     response.close();response=None;break
   if not record['done_received'] and not record['cancel_requested']:raise ValueError('Truncated SSE without intentional cancellation')
  except BaseException as exc:record['error']={'type':type(exc).__name__,'message':str(exc)}
  finally:
   if response is not None:response.close()
   record['finished_epoch']=time.time();(out/(rid+'.json')).write_text(json.dumps(record,ensure_ascii=True,indent=2)+'\n');rows[index]=record
 with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:list(pool.map(request,range(n)))
 result={'client_transport_completed':all(r['error'] is None and (r['done_received'] or r['cancel_requested']) for r in rows),'rows':rows,'requested_streams':n,'real_client_cancel_requested':cancel_index is not None and rows[cancel_index]['cancel_requested'],'actual_native_multrow_overlap_qualified':False,'native_cancellation_or_migration_qualified':False,'full_model_math_qualified':False,'latency_or_API_coherence_qualified':False,'scope':'actual bounded client transport only; native INFO/identity/frames/spans/history/serial/source/owner gates required'}
 (out/'client.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--model',default='hotschmoe-dd');p.add_argument('--messages',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cancel-index',type=int);p.add_argument('--max-new',type=int,default=64);a=p.parse_args();leased([0,1]);result=cohort(a.url,a.model,json.loads(a.messages.read_text()),a.output,a.cancel_index,a.max_new);print(json.dumps({'client_transport_completed':result['client_transport_completed'],'native_or_math_qualified':False}));return 0 if result['client_transport_completed'] else 1
if __name__=='__main__':raise SystemExit(main())
