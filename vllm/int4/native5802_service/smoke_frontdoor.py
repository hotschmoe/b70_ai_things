#!/usr/bin/env python3
"""Explicit runtime smoke inside the existing service's leased job queue.

Preparation/import does not read the key or make requests. No throughput gate.
Report auth statuses, exact identity/context and concurrent answer correctness;
never record authentication headers or credential values.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from pathlib import Path
import urllib.error
import urllib.request

STABLE = 'hotschmoe-dd'
ALIAS = 'qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276phase-mrv1fix-native5802-tp2-mtp3-fp8kv-freshcal-backend-ctx200k'
KEY_FILE = Path('/mnt/vm_8tb/b70/secrets/dd_api_key')


def identity_ok(models):
    rows = models.get('data', [])
    return ([row.get('id') for row in rows] == [STABLE, ALIAS]
            and all(row.get('max_model_len') == 200000 for row in rows))


def run(base, key, request):
    """The injected request callable lets CPU tests use temporary fake responses."""
    report = {'base': base, 'model': STABLE, 'research_alias': ALIAS,
              'required_max_model_len': 200000, 'checks': [],
              'scope': 'Frontdoor identity/auth/concurrent arithmetic smoke; no performance or unrestricted quality claim'}
    # Distinct from the real key even if it equals a conventional test string.
    wrong = key + '-deliberately-invalid'
    for label, kind, token in [('missing-key', None, None),
                                ('wrong-bearer', 'bearer', wrong),
                                ('wrong-x-api-key', 'x-api-key', wrong)]:
        try:
            status, _ = request('/v1/models', kind, token, None)
            row = {'name': label, 'status': status, 'passed': status == 401}
        except Exception as error:
            row = {'name': label, 'passed': False, 'error_type': type(error).__name__}
        report['checks'].append(row)
    for kind in ['bearer', 'x-api-key']:
        try:
            status, body = request('/v1/models', kind, key, None)
            models = json.loads(body)
            row = {'name': kind + '-models', 'status': status,
                   'passed': status == 200 and identity_ok(models),
                   'ids': [r.get('id') for r in models.get('data', [])],
                   'max_model_lens': [r.get('max_model_len') for r in models.get('data', [])]}
        except Exception as error:
            row = {'name': kind + '-models', 'passed': False, 'error_type': type(error).__name__}
        report['checks'].append(row)
    # Identity/auth failures refuse inference against an unintended backend.
    if not all(row['passed'] for row in report['checks']):
        report['passed'] = False
        report['inference_skipped'] = True
        return report

    start_barrier = threading.Barrier(4)

    def complete(pair):
        n, kind = pair
        expected = str(137 + n)
        row = {'name': 'concurrent-' + str(n), 'auth_kind': kind, 'expected': expected}
        try:
            start_barrier.wait(timeout=10)
            status, body = request('/v1/chat/completions', kind, key, {
                'model': STABLE, 'messages': [{'role': 'user',
                    'content': f'What is 137 + {n}? Reply with only the integer.'}],
                'temperature': 0, 'seed': 42, 'max_tokens': 32,
                'chat_template_kwargs': {'enable_thinking': False}})
            response = json.loads(body)
            choices = response.get('choices', [])
            choice = choices[0] if len(choices) == 1 else {}
            content = choice.get('message', {}).get('content')
            row.update(status=status, model=response.get('model'),
                       content=content, finish_reason=choice.get('finish_reason'),
                       usage=response.get('usage'), response_id=response.get('id'))
            row['passed'] = (status == 200 and response.get('model') == STABLE
                             and isinstance(content, str) and content.strip() == expected
                             and choice.get('finish_reason') == 'stop')
        except Exception as error:
            row.update(passed=False, error_type=type(error).__name__)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(complete, [(19, 'bearer'), (23, 'x-api-key'),
                                        (31, 'bearer'), (47, 'x-api-key')]))
    report['checks'].extend(rows)
    report['passed'] = all(row['passed'] for row in report['checks'])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:18080')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0 or args.timeout > 300:
        raise ValueError('timeout must be in (0,300]')
    # Match the unchanged frontdoor's first-line credential semantics exactly.
    with KEY_FILE.open(encoding='ascii') as handle:
        key = handle.readline().strip()
    if not key:
        raise RuntimeError('API key file empty')

    def request(path, kind, token, payload):
        headers = {'Content-Type': 'application/json'}
        if kind == 'bearer': headers['Authorization'] = 'Bearer ' + token
        elif kind == 'x-api-key': headers['X-API-Key'] = token
        req = urllib.request.Request(args.base + path, headers=headers,
            data=None if payload is None else json.dumps(payload).encode('ascii'))
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                return response.status, response.read().decode('utf-8')
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read().decode('utf-8')

    report = run(args.base, key, request)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Defense against an unexpected response reflecting a credential.
    def redact(value):
        if isinstance(value, str): return value.replace(key, '<redacted>')
        if isinstance(value, list): return [redact(v) for v in value]
        if isinstance(value, dict): return {redact(k): redact(v) for k,v in value.items()}
        return value
    encoded = json.dumps(redact(report), indent=2, ensure_ascii=True)
    args.out.write_text(encoded + '\n')
    print(json.dumps({'passed': report['passed'], 'checks': len(report['checks'])}))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
