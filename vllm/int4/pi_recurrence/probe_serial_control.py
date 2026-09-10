#!/usr/bin/env python3
"""Exact frozen session0 payload, two serial requests, separate content/cache gates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from probe_generation import check, request, stream_request


def assess(response,expected,repeat):
    row={}
    try:row.update(check(response,dict(kind='exact_array',expected=expected),False));row['semantic_passed']=True
    except Exception as exc:row.update(semantic_passed=False,semantic_error=ascii(exc))
    hit=response.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens')
    row['cached_tokens']=hit;row['cache_reuse_required']=bool(repeat)
    row['cache_passed']=type(hit) is int and hit>=0 and (not repeat or hit>0)
    row['passed']=row['semantic_passed'] and row['cache_passed'];return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--model',required=True);p.add_argument('--alias',required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=int,default=240);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    identities=json.loads(request(a.base,'/v1/models'));ids={m['id'] for m in identities['data']};assert a.model in ids and a.alias in ids
    (a.out/'models.json').write_text(json.dumps(identities,indent=2)+'\n');manifest=json.loads(a.manifest.read_text());raw=Path(manifest['payload']).read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest['sha256'];payload=json.loads(raw);payload['model']=a.model;payload['cache_salt']='recurrence-serial-control-'+a.out.parent.name+'-'+a.out.name
    (a.out/'config.json').write_text(json.dumps(dict(args=vars(a),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),default=str,indent=2)+'\n');rows=[]
    for repeat in [0,1]:
        name='session0-serial-repeat'+str(repeat);row=dict(id=name,started=time.time(),passed=False)
        (a.out/(name+'-request.json')).write_text(json.dumps(payload,ensure_ascii=True)+'\n')
        try:
            response=stream_request(a.base,payload,a.timeout,a.out/(name+'-sse.jsonl'),bang_limit=32);row['response']=response;row.update(assess(response,manifest['expected'],repeat))
        except Exception as exc:
            row.update(error=ascii(exc),semantic_passed=False,cache_passed=False)
            if hasattr(exc,'partial_response'):row['response']=exc.partial_response
        row['finished']=time.time();rows.append(row);(a.out/'results.json').write_text(json.dumps(rows,ensure_ascii=True,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='response'}),flush=True)
    summary=dict(passed=all(r['passed'] for r in rows),semantic_passed=all(r['semantic_passed'] for r in rows),cache_passed=all(r['cache_passed'] for r in rows),requests=2,limitations='Same canonical payload as concurrent session0; fresh serial cache namespace. Content and cache gates independent. Serial/control ordering and backend state still differ.')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return 0 if summary['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
