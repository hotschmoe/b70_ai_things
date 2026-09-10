#!/usr/bin/env python3
"""CPU-prepared reconstruction and exact structured generation diagnostics.

No generated tool is executed. Reconstructed Pi checks are structural only,
not exact-wire replay or semantic certification. Clean counting tasks have an
independent exact answer. Runtime invocations are owned by the external lease.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'fp8'))
from kv_campaign_agent_probe import stream_request,contains_bang
from kv_campaign_probe import request


def check(response,item,require_hit=False):
    assert not contains_bang(response),'bang output'
    choice,=response['choices'];message=choice['message'];usage=response['usage']
    assert choice['finish_reason'] in ('stop','tool_calls'),'output incomplete or cap exhausted'
    assert usage['completion_tokens']>0
    hit=usage['prompt_tokens_details']['cached_tokens']
    assert type(hit) is int and hit>=0
    if require_hit:assert hit>0,'prefix reuse not observed'
    text='\n'.join(message.get(k) or '' for k in ('content','reasoning','reasoning_content'))
    for match in re.finditer(r'(.{1,80}?)\1{15,}',text,re.S):
        assert not match[1].strip(),'repeated output unit'
    if item['kind']=='exact_array':
        assert choice['finish_reason']=='stop' and not message.get('tool_calls')
        actual=json.loads(message.get('content') or '')
        assert isinstance(actual,list) and all(type(v) is int for v in actual)
        assert actual==item['expected'],'wrong integer sequence'
    else:
        assert text.strip() or message.get('tool_calls'),'empty reconstructed completion'
        assert not any(fragment in text for fragment in ['let?_?:','mustF the the (']),'known original garbled fragment'
        for call in message.get('tool_calls',[]):
            assert call.get('id') and call['function']['name'] in ('bash','read','write','edit')
            arguments=json.loads(call['function']['arguments']);assert isinstance(arguments,dict)
            for match in re.finditer(r'(.{1,80}?)\1{15,}',json.dumps(arguments),re.S):assert not match[1].strip(),'repeated tool arguments'
    return dict(cached_tokens=hit,prompt_tokens=usage['prompt_tokens'],completion_tokens=usage['completion_tokens'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:18125');p.add_argument('--model',required=True);p.add_argument('--alias');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['all','reconstruction','length'],default='all');p.add_argument('--timeout',type=int,default=240)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    identity=json.loads(request(a.base,'/v1/models'));ids={m['id'] for m in identity['data']};assert a.model in ids and (not a.alias or a.alias in ids)
    (a.out/'models.json').write_text(json.dumps(identity,indent=2)+'\n')
    manifest=json.loads(a.corpus.read_text());(a.out/'config.json').write_text(json.dumps(dict(args=vars(a),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest()),default=str,indent=2)+'\n')
    selected=[i for i in manifest['cases'] if a.mode=='all' or (i['kind']=='exact_array')==(a.mode=='length')]
    rows=[]
    for item in selected:
        raw=Path(item['payload']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
        payload=json.loads(raw);payload['model']=a.model
        # Fresh per-arm namespace, shared only within the declared comparison group.
        payload['cache_salt']='recurrence-'+a.out.parent.name+'-'+a.out.name+'-'+item['cache_group']
        trace=a.out/(item['id']+'-sse.jsonl');(a.out/(item['id']+'-request.json')).write_text(json.dumps(payload,ensure_ascii=True)+'\n')
        row=dict(id=item['id'],kind=item['kind'],started=time.time(),passed=False,scope='Exact integer sequence' if item['kind']=='exact_array' else 'Reconstructed history structure/repetition only; manual semantic review required')
        try:
            response=stream_request(a.base,payload,a.timeout,trace,bang_limit=32);row['response']=response
            row.update(check(response,item,require_hit=item['require_hit']),passed=True)
        except Exception as exc:
            row.update(error=ascii(exc))
            if hasattr(exc,'partial_response'):row['response']=exc.partial_response
        row['elapsed_s']=time.time()-row['started'];rows.append(row)
        (a.out/'results.json').write_text(json.dumps(rows,ensure_ascii=True,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='response'}),flush=True)
    summary=dict(passed=len(rows)==len(selected) and all(r['passed'] for r in rows),requests=len(rows),planned_requests=len(selected),limitations=manifest['limitations'],manual_reconstructed_semantic_review_required=any(r['kind']!='exact_array' for r in rows))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return 0 if summary['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
