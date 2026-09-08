#!/usr/bin/env python3
"""One bounded profile per fresh lifecycle; run after all timing/quality jobs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fp8'))
from kv_campaign_probe import long_prompt, request, stream


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18125')
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=32700)
    p.add_argument('--warm', action='store_true', help='profile a cached repeat after an unprofiled warmup')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    models = json.loads(request(args.base, '/v1/models'))
    assert [m['id'] for m in models['data']] == [args.model]
    (args.out / 'config.json').write_text(json.dumps(dict(vars(args), models=models), default=str, indent=2) + '\n')
    prompt, expected = long_prompt(args.tokens, 42)
    prompt += '\nAfter the code, explain how to implement and test an LRU cache in Python.'
    if args.warm:
        warmup = stream(args.base, args.model, prompt, 128, salt='r276-profile-v1', timeout=600)
        (args.out / 'warmup.json').write_text(json.dumps(warmup, indent=2) + '\n')
        assert warmup['error'] is None and expected in warmup['text']
    request(args.base, '/start_profile', {}, timeout=60)
    try:
        row = stream(args.base, args.model, prompt, 128, salt='r276-profile-v1', timeout=600)
        (args.out / 'response.json').write_text(json.dumps(row, indent=2) + '\n')
    finally:
        request(args.base, '/stop_profile', {}, timeout=300)
    assert row['error'] is None and expected in row['text'], row['error']
    if args.warm:
        assert row['usage']['prompt_tokens_details']['cached_tokens'] > 0
    print('Profile captured; timings are diagnostic and excluded from speed comparisons.')


if __name__ == '__main__':
    main()
