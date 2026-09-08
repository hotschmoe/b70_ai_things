#!/usr/bin/env python3
"""Exercise new cold prefills while a confirmed streaming decode is active."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fp8'))
from kv_campaign_probe import long_prompt, request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18125')
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--logprobs', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True)
    identity = json.loads(request(args.base, '/v1/models'))
    assert args.model in [r['id'] for r in identity['data']]
    (args.out / 'models.json').write_text(json.dumps(identity) + '\n')
    (args.out / 'config.json').write_text(json.dumps(vars(args), default=str) + '\n')
    rows = []
    def run(name, prompt, limit, ready=None):
        body = dict(model=args.model, messages=[dict(role='user', content=prompt)],
                    temperature=0, seed=42, max_tokens=limit, stream=True,
                    stream_options=dict(include_usage=True),
                    chat_template_kwargs=dict(enable_thinking=False),
                    cache_salt='mixed-' + args.out.name + '-' + name)
        if args.logprobs:
            body.update(logprobs=True, top_logprobs=5)
        (args.out / (name + '-request.json')).write_text(json.dumps(body) + '\n')
        row = dict(name=name, started=time.time(), text='', usage={}, error=None,
                   first_delta=None, last_delta=None)
        req = urllib.request.Request(args.base + '/v1/chat/completions',
              data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=180) as response, (args.out / (name + '-sse.jsonl')).open('w') as trace:
                for line in response:
                    if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':
                        continue
                    event = json.loads(line[6:])
                    now = time.time()
                    trace.write(json.dumps(dict(at=now, event=event), ensure_ascii=True) + '\n')
                    trace.flush()
                    row['usage'] = event.get('usage') or row['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {}).get('content') or ''
                        if delta:
                            row['text'] += delta
                            row['first_delta'] = row['first_delta'] or now
                            row['last_delta'] = now
                            if ready:
                                ready.set()
                    if '!' * 32 in row['text']:
                        row['error'] = 'cancelled natural bang loop'
                        break
        except Exception as exc:
            row['error'] = ascii(exc)
        row['finished'] = time.time()
        return row
    for wave in range(2):
        ready = threading.Event()
        with ThreadPoolExecutor(max_workers=4) as pool:
            anchor = pool.submit(run, f'{wave}-anchor',
                'Write a detailed practical guide to implementing and testing a Python LRU cache. Include code, complexity analysis, edge cases and concurrent access considerations. Continue until the guide is complete.',
                2048, ready)
            if not ready.wait(90):
                raise RuntimeError('anchor never emitted a token')
            prompts = [long_prompt(32000, 80 + wave * 3 + i) for i in range(3)]
            futures = [pool.submit(run, f'{wave}-prefill-{i}', prompt, 64)
                       for i, (prompt, _) in enumerate(prompts)]
            a = anchor.result()
            a['passed'] = not a['error'] and 'cache' in a['text'].lower() and '```' in a['text']
            rows.append(a)
            for future, (_, expected) in zip(futures, prompts):
                r = future.result()
                r['passed'] = not r['error'] and r['text'].strip() == expected
                r['overlaps_anchor_stream'] = bool(a['last_delta'] and a['first_delta'] <= r['started'] < a['last_delta'])
                rows.append(r)
        (args.out / 'responses.json').write_text(json.dumps(rows, indent=2) + '\n')
    summary = dict(passed=all(r['passed'] for r in rows), requests=len(rows),
        bang_requests=sum('!' * 32 in r['text'] for r in rows),
        overlaps=sum(r.get('overlaps_anchor_stream', False) for r in rows),
        scope='Client-observed overlap; does not prove the backend used a single mixed batch. Anchor coherence is a bounded heuristic.')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
