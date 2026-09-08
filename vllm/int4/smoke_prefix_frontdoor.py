#!/usr/bin/env python3
"""Verify long-prefix reuse through the authenticated public API, under its lease."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fp8'))
from kv_campaign_agent_probe import stream_request
from kv_campaign_probe import long_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18080')
    p.add_argument('--key-file', type=Path, default=Path('/mnt/vm_8tb/b70/secrets/dd_api_key'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    headers = {'Authorization': 'Bearer ' + args.key_file.read_text().strip()}
    req = urllib.request.Request(args.base + '/v1/models', headers=headers)
    with urllib.request.urlopen(req, timeout=10) as r:
        models = json.load(r)
    assert [m['id'] for m in models['data']] == ['hotschmoe-dd']
    report = {'base': args.base, 'models': models, 'rows': [], 'passed': False}
    prompt, expected = long_prompt(150000, 73)
    body = {'model': 'hotschmoe-dd', 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'seed': 42, 'max_tokens': 64,
            'chat_template_kwargs': {'enable_thinking': False},
            'cache_salt': 'production-prefix-smoke-v1'}
    try:
        for i in range(2):
            trace = args.out / f'request-{i}-sse.jsonl'
            start = time.monotonic()
            response = stream_request(args.base, body, 600, trace, headers=headers)
            events = [json.loads(s) for s in trace.read_text().splitlines()]
            ttft = next(e['elapsed_s'] for e in events
                        if any(c.get('delta', {}).get('content')
                               for c in e['event'].get('choices', [])))
            row = {'elapsed_s': time.monotonic() - start, 'ttft_s': ttft,
                   'response': response}
            report['rows'].append(row)
            assert response['model'] == 'hotschmoe-dd'
            assert response['choices'][0]['message']['content'].strip() == expected
            assert response['choices'][0]['finish_reason'] == 'stop'
        cold, warm = report['rows']
        usage = warm['response']['usage']
        report['warm_cached_fraction'] = usage['prompt_tokens_details']['cached_tokens'] / usage['prompt_tokens']
        report['warm_to_cold_ttft_ratio'] = warm['ttft_s'] / cold['ttft_s']
        assert cold['response']['usage']['prompt_tokens_details']['cached_tokens'] == 0
        assert report['warm_cached_fraction'] >= .9
        assert report['warm_to_cold_ttft_ratio'] <= .25
        report['passed'] = True
    finally:
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
