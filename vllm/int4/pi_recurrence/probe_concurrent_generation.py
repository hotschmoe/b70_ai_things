#!/usr/bin/env python3
"""Bounded c4 client dispatch; exact arrays, independent salts, cold/reuse rounds.

Client interval overlap is not proof of a specific GPU batch shape. Explicit
thinking medium is a diagnostic control, not reconstructed Pi wire settings.
All content failures are retained, and the final quality gate remains nonzero.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from probe_generation import check,request,stream_request


def execute(base,model,manifest,out,timeout=240):
    rows=[]
    for phase in manifest['phases']:
        barrier=threading.Barrier(4,timeout=20)
        def one(item):
            row=dict(id=item['id'],passed=False)
            try:
                raw=Path(item['payload']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
                payload=json.loads(raw);payload['model']=model
                payload['cache_salt']='recurrence-'+out.parent.name+'-'+out.name+'-'+item['cache_group']
                (out/(item['id']+'-request.json')).write_text(json.dumps(payload,ensure_ascii=True)+'\n')
                barrier.wait();row['started']=time.time()
                response=stream_request(base,payload,timeout,out/(item['id']+'-sse.jsonl'),bang_limit=32);row['response']=response
                row.update(check(response,item,item['require_hit']),passed=True)
            except Exception as exc:
                row['error']=ascii(exc)
                if hasattr(exc,'partial_response'):row['response']=exc.partial_response
            row['finished']=time.time();return row
        with ThreadPoolExecutor(max_workers=4) as pool:batch=list(pool.map(one,phase['cases']))
        rows.extend(batch);(out/'results.json').write_text(json.dumps(rows,ensure_ascii=True,indent=2)+'\n')
        print(json.dumps(dict(phase=phase['id'],passed=sum(r['passed'] for r in batch),requests=len(batch))),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--model',required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=int,default=240);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);identity=json.loads(request(a.base,'/v1/models'));assert a.model in {m['id'] for m in identity['data']};(a.out/'models.json').write_text(json.dumps(identity,indent=2)+'\n')
    m=json.loads(a.corpus.read_text());assert len(m['phases'])==4 and all(len(p['cases'])==4 for p in m['phases'])
    (a.out/'config.json').write_text(json.dumps(dict(args=vars(a),manifest_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),default=str,indent=2)+'\n')
    rows=execute(a.base,a.model,m,a.out,a.timeout);passed=len(rows)==16 and all(r['passed'] for r in rows)
    (a.out/'summary.json').write_text(json.dumps(dict(passed=passed,requests=len(rows),limitations=__doc__),indent=2)+'\n');return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
