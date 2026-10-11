#!/usr/bin/env python3
"""Six exact C140 coherence cases plus two genuinely overlapping HTTP clients."""
import argparse,json,time,threading,urllib.request,concurrent.futures
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from source43_api_baseline_profile_v1 import ALIAS,require
PRIMARY='hotschmoe-dd'
CASES=[('marker',[{'role':'user','content':'Reply with exactly B70_READY and nothing else.'}],'B70_READY'),('arithmetic',[{'role':'user','content':'What is 19 plus 23? Reply only with the decimal number.'}],'42'),('history',[{'role':'user','content':'The marker is NATIVE_31415.'},{'role':'assistant','content':'I recorded the marker.'},{'role':'user','content':'Repeat the marker exactly and nothing else.'}],'NATIVE_31415'),('fact',[{'role':'user','content':'What is the capital of France? Reply only with the city name.'}],'Paris')]
CASES+=[('marker-repeat',CASES[0][1],CASES[0][2]),('history-repeat',CASES[2][1],CASES[2][2])]
def request(url,body=None):
    data=None if body is None else json.dumps(body).encode('ascii');req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300)as response:return json.load(response)
def body(index,messages):return {'model':PRIMARY if index%2==0 else ALIAS,'messages':messages,'max_tokens':64,'temperature':0,'seed':1,'stream':False,'chat_template_kwargs':{'enable_thinking':False},'reasoning_budget_tokens':0,'strata_fresh':True}
def response_gate(value,expected):
    choices=value['choices'];require(type(choices)is list and len(choices)==1 and type(choices[0]['index'])is int and choices[0]['index']==0 and choices[0]['finish_reason']in('stop','length'),'Exact one-lane API terminal required');require((choices[0]['message'].get('content')or'').strip()==expected,'Actual semantic coherence failed');usage=value['usage'];require(all(type(usage[k])is int and usage[k]>=0 for k in('prompt_tokens','completion_tokens','total_tokens'))and usage['completion_tokens']<=64 and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'],'Typed actual API usage/budget required')
def execute(base,output,send=request):
    output=Path(output);require(not output.exists(),'New API client result required');output.mkdir();rows=[];errors=[];models=None
    def call(index,case,barrier=None):
        name,messages,expected=case;packet=body(index,messages)
        if barrier:barrier.wait(timeout=30)
        (output/('request-'+str(index)+'.json')).write_text(json.dumps(packet,indent=2)+'\n',encoding='ascii');started=time.monotonic_ns();value=send(base+'/v1/chat/completions',packet);finished=time.monotonic_ns();(output/('response-'+str(index)+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='ascii');require(value['model']==packet['model'],'Actual perrequest servedmodel identity differs');response_gate(value,expected);return {'case':name,'request':packet,'response':value,'started_monotonic_ns':started,'finished_monotonic_ns':finished,'coherent':True}
    try:
        models=send(base+'/v1/models');(output/'models.json').write_text(json.dumps(models,indent=2)+'\n',encoding='ascii');require([r['id']for r in models['data']]==[PRIMARY,ALIAS],'Actual correct server IDs/order required')
        for index,case in enumerate(CASES):rows.append(call(index,case))
        require(canonical(rows[0]['response']['choices'])==canonical(rows[4]['response']['choices'])and canonical(rows[2]['response']['choices'])==canonical(rows[5]['response']['choices']),'Actual deterministic repeat choices differ')
        barrier=threading.Barrier(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
            futures=[pool.submit(call,6+i,CASES[i*2],barrier)for i in range(2)];parallel=[f.result()for f in futures]
        require(max(r['started_monotonic_ns']for r in parallel)<min(r['finished_monotonic_ns']for r in parallel),'Actual two client lifetimes did not overlap');rows+=parallel
    except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
    report={'schema':1,'passed':not errors,'errors':errors,'models':models,'results':rows,'serial_cases':6,'concurrent_clients':2,'actual_HTTP_overlap_passed':not errors and len(rows)==8,'native_computation_parallelism_qualified':False,'engine_token_provenance_qualified':False,'owned_teardown_qualified':False,'full_model_math_qualified':False};(output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='ascii');require(report['passed'],'API client failed; original output preserved');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();execute('http://127.0.0.1:'+str(a.port),a.output)
