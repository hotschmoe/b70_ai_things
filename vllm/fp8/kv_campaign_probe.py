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


def stream(base, model, prompt, limit=256, salt='kv-campaign', timeout=600, thinking=False, force_length=False, cancel_after_chunks=None, reasoning_effort='low'):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'top_p': 1, 'seed': 42, 'max_tokens': limit,
            'chat_template_kwargs': ({'enable_thinking': True, 'reasoning_effort': reasoning_effort} if thinking else {'enable_thinking': False}),
            'stream': True, 'stream_options': {'include_usage': True}, 'cache_salt': salt}
    if force_length:
        body['ignore_eos'] = True
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic(); times = []; parts = []; reasoning = []; usage = {}; finish = None; cancelled = False
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
                if cancel_after_chunks is not None and len(parts) >= cancel_after_chunks:
                    cancelled = True
                    break
        error = None
    except Exception as exc:
        error = ascii(exc)
    elapsed = time.monotonic() - start
    text = ''.join(parts)
    return {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'text': text,
            'reasoning': ''.join(reasoning), 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'usage': usage, 'elapsed_s': elapsed, 'ttft_s': times[0] if times else None,
            'chunk_times_s': times, 'max_chunk_gap_s': max([b-a for a,b in zip(times,times[1:])], default=0),
            'finish_reason': finish, 'error': error, 'client_cancelled': cancelled}


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


def generation_issue(row):
    text = row['text']
    if row['error']:
        return 'request error'
    if len(text) <= 100:
        return 'unexpectedly short guide'
    if re.search(r'([!?.A-Za-z0-9])\1{50}', text):
        return 'single-character loop'
    previous = None
    streak = 0
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 12:
            previous = None; streak = 0
            continue
        streak = streak + 1 if line == previous else 1
        previous = line
        if streak >= 6:
            return 'repeated-line loop'
    if row.get('finish_reason') == 'stop' and text.count('```') % 2:
        return 'unclosed code fence at EOS'
    return None


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
    p.add_argument('--mode', choices=['quality', 'long', 'pressure', 'decode', 'reuse', 'cancel'], default='quality')
    p.add_argument('--tokens', type=int, default=150000)
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--output-tokens', type=int, default=1024)
    p.add_argument('--thinking', action='store_true')
    p.add_argument('--reasoning-effort', choices=['low', 'medium', 'high', 'xhigh'], default='xhigh')
    p.add_argument('--salt', default='')
    p.add_argument('--force-length', action='store_true')
    p.add_argument('--reuse-sequence', default='0,1,2,0')
    args = p.parse_args()
    if args.thinking and args.mode != 'quality':
        p.error('--thinking currently applies only to the quality probe')
    try:
        reuse_sequence = [int(i) for i in args.reuse_sequence.split(',')]
        assert 2 <= len(reuse_sequence) <= 32 and all(0 <= i <= 99 for i in reuse_sequence)
    except (ValueError, AssertionError):
        p.error('--reuse-sequence needs 2-32 comma-separated indices in 0..99')
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
                row = stream(args.base, args.model, prompt,
                             args.output_tokens if args.thinking else 256,
                             salt='quality-v1' + args.salt, timeout=args.timeout,
                             thinking=args.thinking, reasoning_effort=args.reasoning_effort)
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
    elif args.mode == 'cancel':
        prompt, expected = long_prompt(args.tokens, 91)
        prompt += '\nAfter the code, explain how to test a cache, with detailed Python examples.'
        row = stream(args.base, args.model, prompt, 1024, salt='cancel-v1' + args.salt,
                     timeout=args.timeout, cancel_after_chunks=2)
        row.update(task='cancel', passed=row['client_cancelled'] and row['error'] is None)
        save(row)
        # Closing the HTTP stream requests cancellation; allow scheduler cleanup.
        time.sleep(2)
        for repeat in range(2):
            row = stream(args.base, args.model, prompt, 512, salt='cancel-v1' + args.salt, timeout=args.timeout)
            row.update(task='resume', repeat=repeat, expected=expected,
                       passed=expected in row['text'] and row['error'] is None)
            save(row)
    elif args.mode == 'reuse':
        for i in reuse_sequence:
            prompt, expected = long_prompt(args.tokens, i)
            row = stream(args.base, args.model, prompt, 64, salt='reuse-v1' + args.salt, timeout=args.timeout)
            row.update(task=i, repeat=len(rows), expected=expected, passed=row['text'].strip() == expected and row['error'] is None)
            save(row)
            (args.out / f'metrics-step-{len(rows)}.txt').write_text(request(args.base, '/metrics'))
    elif args.mode in ('long', 'pressure'):
        for repeat in range(args.rounds):
            def task(i):
                prompt, expected = long_prompt(args.tokens, i)
                if args.mode == 'pressure':
                    prompt += '\nAfter the code, write a detailed guide to testing a Python dictionary-backed LRU cache, with code examples and edge cases.'
                row = stream(args.base, args.model, prompt, args.output_tokens if args.mode == 'pressure' else 64, salt=f'long-v1-{repeat}' + args.salt, timeout=args.timeout, force_length=args.force_length)
                row['forced_length_diagnostic'] = args.force_length
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
                    issue = generation_issue(row)
                    row.update(task=i, repeat=repeat, passed=issue is None, coherence_issue=issue)
                    save(row)
    (args.out / 'metrics-after.txt').write_text(request(args.base, '/metrics'))
    repeat_exact = all(len({r['text_sha256'] for r in rows if r['task'] == i}) == 1 for i in {r['task'] for r in rows})
    rows_passed = all(r['passed'] for r in rows)
    repeat_required = args.mode in ('quality', 'decode', 'reuse', 'cancel')
    summary = {'mode': args.mode, 'passed': rows_passed and (repeat_exact or not repeat_required),
               'rows_passed': rows_passed, 'repeat_exact': repeat_exact,
               'rows': len(rows), 'elapsed_s': time.monotonic()-all_start,
               'completion_tokens': sum(r['usage'].get('completion_tokens', 0) for r in rows),
               'prompt_tokens': [r['usage'].get('prompt_tokens') for r in rows],
               'ttft_s': [r['ttft_s'] for r in rows]}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
