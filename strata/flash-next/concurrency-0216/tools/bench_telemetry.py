#!/usr/bin/env python3
"""Diagnostic concurrent requests plus server-side queue transition snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/mnt/vm_8tb/github/b70_ai_things')
sys.path.insert(0, str(ROOT/'strata/flash-next'))
import bench_serving as bench


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--concurrency',type=int,required=True)
    p.add_argument('--max-tokens',type=int,default=256)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--poll-seconds',type=float,default=.5)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    base='http://127.0.0.1:18134'
    ids=[x['id'] for x in bench.get(base,'/v1/models')['data']]
    alias='qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216'
    assert ids[0]=='hotschmoe-dd' and alias in ids and alias in (ROOT/'evals/configs/models.yaml').read_text()
    for prompt,expected in [('Return only the integer: 17 * 19.','323'),('Return only the integer: 23 * 29.','667')]:
        assert bench.send(base,'hotschmoe-dd',[{'role':'user','content':prompt}],32)['text'].strip()==expected
    initial=bench.get(base,'/v1/telemetry')
    since=initial['activity']['last_seq']
    events={}
    errors=[]
    done=threading.Event()
    def sample():
        try:
            while not done.is_set():
                snapshot=bench.get(base,'/v1/telemetry')
                for event in snapshot['activity']['events']:
                    if event['seq']>since:events[event['seq']]=event
                done.wait(a.poll_seconds)
            snapshot=bench.get(base,'/v1/telemetry')
            for event in snapshot['activity']['events']:
                if event['seq']>since:events[event['seq']]=event
        except Exception as exc:
            errors.append(repr(exc))
    thread=threading.Thread(target=sample,daemon=True)
    thread.start()
    template=json.loads(Path('/mnt/vm_8tb/b70/results/conc-0216/a-baseline-c1/coding-0.json').read_text())['request']['messages'][0]['content'].split('] ',1)[1]
    gate=threading.Barrier(a.concurrency)
    def worker(i):
        gate.wait(timeout=60)
        prompt=f'[Flash-Next telemetry N={a.concurrency} slot={i}] '+template
        try:
            row=bench.send(base,'hotschmoe-dd',[{'role':'user','content':prompt}],a.max_tokens)
            (a.output/f'response-{i}.json').write_text(json.dumps(row,indent=2,ensure_ascii=True)+'\n')
            return row
        except Exception as exc:
            (a.output/f'error-{i}.json').write_text(json.dumps({'error':repr(exc)},ensure_ascii=True)+'\n')
            raise
    rows=[]
    failure=None
    try:
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:rows=list(pool.map(worker,range(a.concurrency)))
    except Exception as exc:
        failure=repr(exc)
    finally:
        done.set();thread.join(timeout=20)
    (a.output/'responses.json').write_text(json.dumps(rows,indent=2,ensure_ascii=True)+'\n')
    records=sorted(events.values(),key=lambda e:e['seq'])
    (a.output/'events.json').write_text(json.dumps(records,indent=2,ensure_ascii=True)+'\n')
    if failure:raise RuntimeError('Client failed; partial responses and events preserved: '+failure)
    by_id={}
    for e in records:by_id.setdefault(e['request_id'],{}).setdefault(e['event'],e['at'])
    output=[]
    for request_id,x in by_id.items():
        if 'queued' not in x:continue
        def diff(a,b):
            if a not in x or b not in x:return None
            value=x[b]-x[a]
            return value if value>=0 else None
        output.append(dict(request_id=request_id, queued_to_admitted_s=diff('queued','admitted'),
                           admitted_to_reading_s=diff('admitted','reading'), queued_to_reading_s=diff('queued','reading'),
                           reading_to_first_engine_token_s=diff('reading','first_token'),
                           queued_to_first_engine_token_s=diff('queued','first_token'), events=x))
    summary=dict(concurrency=a.concurrency,requests=output,errors=errors,events_seen=len(records),
                 sequence_start=since,completed_tokens=[r['usage']['completion_tokens'] for r in rows],
                 note='Diagnostic polling perturbs scheduling. IDs are server-internal and cannot be reliably joined to SSE client stream IDs. queued starts after input preprocessing; reading marks control submission, not an isolated GPU admission.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=True)+'\n')
    print(json.dumps(summary,ensure_ascii=True))
    if errors or len(output)!=a.concurrency or any(r['queued_to_reading_s'] is None or r['reading_to_first_engine_token_s'] is None for r in output):
        raise RuntimeError('Telemetry transitions incomplete; raw events preserved')
    if any(r['usage']['completion_tokens']!=a.max_tokens for r in rows):raise RuntimeError('Incomplete outputs')


if __name__=='__main__':main()
