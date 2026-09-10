#!/usr/bin/env python3
"""Bounded OpenAI-compatible viability screen; never execute model tools.

Uses completion text for backend portability, not an exact token-ID oracle.
Run only as a job owned by a leased serving lifecycle.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--model', default='hotschmoe-dd')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps(dict(
        base=args.base, model=args.model, timeout=args.timeout,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Text/structure viability, not full semantic quality or token-ID equivalence.')) + '\n')
    identity = json.load(urllib.request.urlopen(args.base + '/v1/models', timeout=10))
    assert args.model in [x['id'] for x in identity['data']]
    (args.out / 'models.json').write_text(json.dumps(identity, ensure_ascii=True) + '\n')
    rows = []
    lock = threading.Lock()

    def request(name, endpoint, payload, expected=None):
        payload = dict(payload, model=args.model, temperature=0, seed=42, stream=False)
        row = dict(name=name, endpoint=endpoint, request=payload)
        start = time.monotonic()
        try:
            req = urllib.request.Request(args.base + endpoint, json.dumps(payload).encode(),
                                         {'Content-Type': 'application/json'})
            response = json.load(urllib.request.urlopen(req, timeout=args.timeout))
            row['response'] = response
            choice = response['choices'][0]
            text = choice.get('text', choice.get('message', {}).get('content')) or ''
            visible = ''.join(text.split())
            repeated = len(visible) >= 32 and Counter(visible).most_common(1)[0][1] / len(visible) >= .90
            row['bang'] = '!' * 32 in json.dumps(response, ensure_ascii=True)
            row['passed'] = bool(text.strip()) and not repeated and not row['bang']
            if expected is not None:
                row['passed'] = row['passed'] and text.strip() == expected
            row['finish_reason'] = choice.get('finish_reason')
            row['passed'] = row['passed'] and row['finish_reason'] in ('stop', 'length')
        except Exception as exc:
            row.update(passed=False, error=ascii(exc))
        row['elapsed_s'] = time.monotonic() - start
        with lock:
            rows.append(row)
            (args.out / 'results.json').write_text(json.dumps(rows, ensure_ascii=True, indent=2) + '\n')
            print(json.dumps({k: row[k] for k in ('name', 'passed', 'elapsed_s')}, ensure_ascii=True), flush=True)
        return row['passed']

    for name, prompt, answer in [('identity-canary', 'Reply with exactly READY.', 'READY'),
                                  ('arithmetic', 'What is 17 + 25? Reply with only the number.', '42')]:
        if not request(name, '/v1/chat/completions', dict(
                messages=[dict(role='user', content=prompt)], max_tokens=128,
                chat_template_kwargs=dict(enable_thinking=False)), answer):
            break
    else:
        # One/two-token contrast plus normal prose. Repeated fresh requests
        # and c4 submissions exercise state reuse without executing tools.
        cases = [('one-token', [17]), ('two-token', [17, 18]),
                 ('arithmetic', 'Calculate 17 + 25. Give the answer in a short sentence.'),
                 ('operations', 'Explain three practical checks for a slow document search service.')]
        for name, prompt in cases:
            for repeat in range(2):
                request(name + '-serial-' + str(repeat), '/v1/completions',
                        dict(prompt=prompt, max_tokens=64, add_special_tokens=False))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for repeat in range(4):
                futures = [pool.submit(request, name + '-mixed-' + str(repeat),
                           '/v1/completions', dict(prompt=prompt, max_tokens=64,
                                                  add_special_tokens=False)) for name, prompt in cases]
                for future in futures:
                    future.result()
    passed = len(rows) == 26 and all(row['passed'] for row in rows)
    (args.out / 'OUTCOME.json').write_text(json.dumps(dict(passed=passed, requests=len(rows))) + '\n')
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
