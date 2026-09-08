#!/usr/bin/env python3
"""Bounded cache campaign probes; retains responses, SSE timing, and metrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


def request(base, route, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def stream(base, model, prompt, limit=256, salt='kv-campaign', timeout=600, thinking=False):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'top_p': 1, 'seed': 42, 'max_tokens': limit,
            'chat_template_kwargs': ({'enable_thinking': True, 'reasoning_effort': 'low'} if thinking else {'enable_thinking': False}),
            'stream': True, 'stream_options': {'include_usage': True}, 'cache_salt': salt}
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic(); times = []; parts = []; reasoning = []; usage = {}; finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                if time.monotonic() - start > timeout:
                    raise TimeoutError('total request deadline exceeded')
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('content'):
                        parts.append(delta['content']); times.append(time.monotonic() - start)
                    if delta.get('reasoning') or delta.get('reasoning_content'):
                        reasoning.append(delta.get('reasoning') or delta['reasoning_content'])
                    finish = choice.get('finish_reason') or finish
        error = None
    except Exception as exc:
        error = ascii(exc)
    elapsed = time.monotonic() - start
    text = ''.join(parts)
    return {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'text': text,
            'reasoning': ''.join(reasoning), 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'usage': usage, 'elapsed_s': elapsed, 'ttft_s': times[0] if times else None,
            'chunk_times_s': times, 'max_chunk_gap_s': max([b-a for a,b in zip(times,times[1:])], default=0),
            'finish_reason': finish, 'error': error}


TASKS = [
    ('Reply with only the capital of France.', 'Paris'),
    ('Calculate 17 + 26. Reply with only the number.', '43'),
    ('Reply with only the chemical symbol for gold.', 'Au'),
    ('Return a JSON object with key sum and the integer sum of 19 and 23. No markdown.', '"sum"'),
    ('In Python, what is list(range(2, 9, 3))? Reply with only the list.', '[2, 5, 8]'),
    ('What does Python print for print("abcdef"[1::2])? Reply with only the output.', 'bdf'),
    ('A queue receives A, B, C, then removes two items. Which item remains? Reply with only that letter.', 'C'),
    ('Return only the base-10 value of binary 101101.', '45'),
    ('Sort these integers ascending: 17, -3, 0, 9. Reply with only a JSON array.', '[-3, 0, 9, 17]'),
    ('In SQL, which clause filters groups after aggregation? Reply with only its keyword.', 'HAVING'),
    ('What is the worst-case time complexity of binary search on a sorted array? Reply briefly.', 'log'),
    ('A rectangle is 7 cm by 13 cm. Give its area as an integer, no units.', '91'),
]


def long_prompt(tokens, index=0):
    # Distinct first tokens prevent accidental sharing between pressure streams.
    secret = f'B70-{index:02d}-ORANGE-4917'
    header = f'Archive {index}. Read these records and remember the access code.\n'
    record = 'Routine archive entry: the service processed a request and stored its response successfully.\n'
    # Actual token count is retained from server usage; this is an approximate target.
    n = max(1, tokens // 16)
    records = [record] * n
    records[n // 3] = f'The unique access code for archive {index} is {secret}.\n'
    prompt = header + ''.join(records) + '\nReturn only the unique access code from this archive.'
    return prompt, secret


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18125')
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['quality', 'long', 'pressure', 'decode', 'reuse'], default='quality')
    p.add_argument('--tokens', type=int, default=150000)
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--timeout', type=int, default=600)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    identity = json.loads(request(args.base, '/v1/models'))
    assert any(d['id'] == args.model for d in identity['data']), identity
    (args.out / 'models.json').write_text(json.dumps(identity, indent=2) + '\n')
    (args.out / 'metrics-before.txt').write_text(request(args.base, '/metrics'))
    rows = []; all_start = time.monotonic()
    def save(row):
        rows.append(row)
        with (args.out / 'responses.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')
    if args.mode == 'quality':
        for repeat in range(args.rounds):
            def task(item):
                i, (prompt, expected) = item
                row = stream(args.base, args.model, prompt, salt='quality-v1', timeout=args.timeout)
                row.update(task=i, repeat=repeat, expected=expected, passed=expected.lower() in row['text'].lower() and row['error'] is None)
                if i == 3:
                    try:
                        row['passed'] = json.loads(row['text']) == {'sum': 42}
                    except ValueError:
                        row['passed'] = False
                return row
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for row in pool.map(task, enumerate(TASKS)):
                    save(row)
    elif args.mode == 'reuse':
        for i in [0, 1, 2, 0]:
            prompt, expected = long_prompt(args.tokens, i)
            row = stream(args.base, args.model, prompt, 64, salt='reuse-v1', timeout=args.timeout)
            row.update(task=i, repeat=len(rows), expected=expected, passed=row['text'].strip() == expected and row['error'] is None)
            save(row)
            (args.out / f'metrics-step-{len(rows)}.txt').write_text(request(args.base, '/metrics'))
    elif args.mode in ('long', 'pressure'):
        for repeat in range(args.rounds):
            def task(i):
                prompt, expected = long_prompt(args.tokens, i)
                if args.mode == 'pressure':
                    prompt += '\nAfter the code, write a detailed guide to testing a Python dictionary-backed LRU cache, with code examples and edge cases.'
                row = stream(args.base, args.model, prompt, 1024 if args.mode == 'pressure' else 64, salt=f'long-v1-{repeat}', timeout=args.timeout)
                row.update(task=i, repeat=repeat, expected=expected, passed=(expected in row['text'] if args.mode == 'pressure' else row['text'].strip() == expected) and row['error'] is None)
                return row
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for row in pool.map(task, range(args.concurrency)):
                    save(row)
    else:
        prompt = 'Write a detailed practical guide to implementing and testing a Python LRU cache. Include code, complexity analysis, edge cases and concurrent access considerations. Continue until the guide is complete.'
        for repeat in range(args.rounds):
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for i, row in enumerate(pool.map(lambda i: stream(args.base, args.model, f'Task {i}. '+prompt, 2048, salt=f'decode-v1-{repeat}', timeout=args.timeout), range(args.concurrency))):
                    # Code/Markdown commonly contains long hyphen/equals rulers.
                    row.update(task=i, repeat=repeat, passed=row['error'] is None and len(row['text']) > 100 and not re.search(r'([!?.A-Za-z0-9])\1{50}', row['text']))
                    save(row)
    (args.out / 'metrics-after.txt').write_text(request(args.base, '/metrics'))
    repeat_exact = all(len({r['text_sha256'] for r in rows if r['task'] == i}) == 1 for i in {r['task'] for r in rows})
    summary = {'mode': args.mode, 'passed': all(r['passed'] for r in rows), 'repeat_exact': repeat_exact,
               'rows': len(rows), 'elapsed_s': time.monotonic()-all_start,
               'completion_tokens': sum(r['usage'].get('completion_tokens', 0) for r in rows),
               'prompt_tokens': [r['usage'].get('prompt_tokens') for r in rows],
               'ttft_s': [r['ttft_s'] for r in rows]}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
