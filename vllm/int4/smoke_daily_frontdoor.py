#!/usr/bin/env python3
"""Run inside the serving lifecycle's leased job queue; never log API keys."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18080')
    p.add_argument('--key-file', type=Path, default=Path('/mnt/vm_8tb/b70/secrets/dd_api_key'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    key = args.key_file.read_text().strip()
    assert key
    report = {'base': args.base, 'model': 'hotschmoe-dd', 'checks': []}

    def request(path, token=None, payload=None):
        headers = {'Content-Type': 'application/json'}
        if token is not None:
            headers['Authorization'] = 'Bearer ' + token
        req = urllib.request.Request(args.base + path, headers=headers,
            data=None if payload is None else json.dumps(payload).encode())
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    try:
        for label, token in [('missing-key', None), ('wrong-key', 'invalid-campaign-key')]:
            status, _ = request('/v1/models', token)
            report['checks'].append({'name': label, 'status': status})
            assert status == 401, label
        status, body = request('/v1/models', key)
        models = json.loads(body)
        report['models'] = models
        assert status == 200 and [r['id'] for r in models['data']] == ['hotschmoe-dd']

        def complete(n):
            expected = str(137 + n)
            status, body = request('/v1/chat/completions', key, {
                'model': 'hotschmoe-dd', 'messages': [{'role': 'user',
                    'content': f'What is 137 + {n}? Reply with only the integer.'}],
                'temperature': 0, 'seed': 42, 'max_tokens': 32,
                'chat_template_kwargs': {'enable_thinking': False}})
            response = json.loads(body)
            passed = (status == 200 and response.get('model') == 'hotschmoe-dd'
                and response['choices'][0]['message']['content'].strip() == expected
                and response['choices'][0]['finish_reason'] == 'stop')
            return {'name': 'concurrent-' + str(n), 'passed': passed, 'response': response}

        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(complete, [19, 23, 31, 47]))
        report['checks'].extend(rows)
        assert all(row['passed'] for row in rows), 'concurrent arithmetic smoke'
        report['passed'] = True
    except Exception as error:
        report['passed'] = False
        report['error_type'] = type(error).__name__
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
