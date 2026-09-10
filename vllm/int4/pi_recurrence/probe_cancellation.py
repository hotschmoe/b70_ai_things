#!/usr/bin/env python3
"""Bounded HTTP cancellation/reuse diagnostic; external lifecycle owns serving.

A repeated request restarts from the prompt; it is not continuation from sampled
partial tokens. The follow-up changes only the final range instruction, testing
cached-prefix reuse. No tool or generated code is executed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'fp8'))
from kv_campaign_probe import request
from probe_generation import check


def stream(base,payload,path,cancel_chunks=None,timeout=180):
    body=dict(payload,stream=True,stream_options={'include_usage':True})
    row=dict(response={'choices':[{'message':{'role':'assistant','content':''},'finish_reason':None}]},client_cancelled=False,content_chunks=0,error=None)
    started=time.monotonic();req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
        # Per-read timeout plus checked total deadline bounds a stalled stream.
        with urllib.request.urlopen(req,timeout=min(timeout,30)) as response,path.open('w') as trace:
            for line in response:
                if time.monotonic()-started>timeout:raise TimeoutError('total stream deadline')
                if not line.startswith(b'data: '):continue
                raw=line[6:].strip()
                if raw==b'[DONE]':break
                event=json.loads(raw);trace.write(json.dumps(dict(elapsed_s=time.monotonic()-started,event=event),ensure_ascii=True)+'\n');trace.flush()
                for key in ('id','model','usage'):
                    if event.get(key) is not None:row['response'][key]=event[key]
                for choice in event.get('choices',[]):
                    assert choice['index']==0;delta=choice.get('delta',{});target=row['response']['choices'][0]
                    for key in ('content','reasoning','reasoning_content'):
                        if delta.get(key):target['message'][key]=target['message'].get(key,'')+delta[key]
                    if delta.get('content'):row['content_chunks']+=1
                    if choice.get('finish_reason'):target['finish_reason']=choice['finish_reason']
                    assert not delta.get('tool_calls'),'unexpected tool delta'
                text=row['response']['choices'][0]['message']['content']
                if '!'*32 in text:raise RuntimeError('bang output')
                if cancel_chunks is not None and row['content_chunks']>=cancel_chunks:
                    assert row['response']['choices'][0]['finish_reason'] is None,'completed before cancellation point'
                    row['client_cancelled']=True
                    row['cancellation_point']=dict(content_chunks=row['content_chunks'],text_chars=len(text),text_utf8_bytes=len(text.encode()),elapsed_s=time.monotonic()-started,token_count=None,token_count_reason='SSE content chunks are not tokenizer tokens',usage_available='usage' in row['response'])
                    break
            if not row['client_cancelled']:
                assert row['response']['choices'][0]['finish_reason'] and 'usage' in row['response'],'incomplete uncancelled response'
    except Exception as exc:row['error']=ascii(exc)
    row['elapsed_s']=time.monotonic()-started;return row


def idle_gauges(metrics):
    values={}
    for name,value in re.findall(r'^vllm:num_requests_(running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)$',metrics,re.M):values.setdefault(name,[]).append(float(value))
    return set(values)=={'running','waiting'} and all(v==0 for vs in values.values() for v in vs),values


def drain(base,out,timeout=60):
    started=time.monotonic();samples=[];passed=False
    while time.monotonic()-started<timeout:
        metrics=request(base,'/metrics',timeout=5);passed,values=idle_gauges(metrics)
        n=len(samples);(out/f'drain-{n:03d}.txt').write_text(metrics);samples.append(dict(elapsed_s=time.monotonic()-started,values=values,passed=passed))
        if passed:break
        time.sleep(1)
    result=dict(passed=passed,samples=samples,scope='Observed idle gauges after HTTP closure; no worker cancellation trace')
    (out/'drain.json').write_text(json.dumps(result,indent=2)+'\n');return passed


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:18125');p.add_argument('--model',required=True);p.add_argument('--alias');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cancel-chunks',type=int,default=32);a=p.parse_args()
    assert 1<=a.cancel_chunks<=128;a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.corpus.read_text());case=next(i for i in manifest['cases'] if i['id']=='array-cap2048-repeat0');raw=Path(case['payload']).read_bytes();assert hashlib.sha256(raw).hexdigest()==case['sha256'];base=json.loads(raw)
    identity=json.loads(request(a.base,'/v1/models'));ids={i['id'] for i in identity['data']};assert a.model in ids and (not a.alias or a.alias in ids);(a.out/'models.json').write_text(json.dumps(identity,indent=2)+'\n')
    base['model']=a.model;follow=copy.deepcopy(base)
    old='every integer from 1 through 512 inclusive';new='every integer from 513 through 768 inclusive'
    assert old in follow['messages'][0]['content'];follow['messages'][0]['content']=follow['messages'][0]['content'].replace(old,new);follow['max_tokens']=2048
    matrix=[('intact',base,list(range(1,513)),'reference',False,False),('intact-followup',follow,list(range(513,769)),'reference',False,True),('cancel',base,None,'cancelled',True,False),('recovery',base,list(range(1,513)),'cancelled',False,True),('recovery-followup',follow,list(range(513,769)),'cancelled',False,True)]
    config=dict(corpus_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cancel_chunks=a.cancel_chunks,matrix=[dict(id=n,cache_group=g,cancel=c,require_hit=h,max_tokens=p['max_tokens']) for n,p,e,g,c,h in matrix],scope=__doc__)
    (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n');rows=[]
    for name,payload,expected,group,cancel,hit in matrix:
        payload=copy.deepcopy(payload);payload['cache_salt']='cancel-'+a.out.parent.name+'-'+a.out.name+'-'+group
        (a.out/(name+'-request.json')).write_text(json.dumps(payload,ensure_ascii=True)+'\n')
        row=stream(a.base,payload,a.out/(name+'-sse.jsonl'),a.cancel_chunks if cancel else None);row.update(id=name,passed=False)
        try:
            assert row['error'] is None
            if cancel:
                assert row['client_cancelled'];row['passed']=drain(a.base,a.out)
            else:row.update(check(row['response'],dict(kind='exact_array',expected=expected),hit),passed=True)
        except Exception as exc:row['quality_error']=ascii(exc)
        rows.append(row);(a.out/'results.json').write_text(json.dumps(rows,ensure_ascii=True,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='response'}),flush=True)
        # Do not enqueue recovery if cancellation/drain itself is unproven.
        if cancel and not row['passed']:break
    result=dict(passed=len(rows)==5 and all(r['passed'] for r in rows),requests=len(rows),scope=__doc__)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
