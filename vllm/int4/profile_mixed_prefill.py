#!/usr/bin/env python3
"""Profile a cold 132K prefill alongside short decodes inside a leased server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    models = json.loads(request(args.base, '/v1/models'))
    assert args.model in [m['id'] for m in models['data']]
    (args.out / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
    rows = []
    request(args.base, '/start_profile', {}, timeout=60)
    try:
        def task(i):
            if i == 0:
                prompt, expected = long_prompt(132000, 37)
            else:
                prompt = f'Task {i}: Write a practical guide to testing a Python LRU cache with examples.'
                expected = None
            row = stream(args.base, args.model, prompt, 16 if i == 0 else 128,
                         salt=f'mixed-prefill-profile-{i}', timeout=900)
            row['task'] = i
            row['passed'] = row['error'] is None and (row['text'].strip() == expected if i == 0 else bool(row['text']))
            return row
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(task, range(4)))
    finally:
        request(args.base, '/stop_profile', {}, timeout=300)
        (args.out / 'responses.json').write_text(json.dumps(rows, indent=2) + '\n')
    assert len(rows) == 4 and all(r['passed'] for r in rows)
    print('Mixed-prefill profile complete; timings are profiled, not a speed qualification.')


if __name__ == '__main__':
    main()
