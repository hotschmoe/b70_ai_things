#!/usr/bin/env python3
"""C4 correctness control with sequential cache seeding and staggered arrivals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time

ROOT = Path('/mnt/vm_8tb/github/b70_ai_things')
HARNESS = ROOT / 'vllm/flashnext/harness/c4_compare/bench.py'
spec = importlib.util.spec_from_file_location('c4bench', HARNESS)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stagger-s', type=float, default=.25)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = 'http://127.0.0.1:18134'
    alias = 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216'
    ids = [x['id'] for x in bench.get(base, '/v1/models')['data']]
    assert ids[0] == 'hotschmoe-dd' and alias in ids
    assert alias in (ROOT / 'evals/configs/models.yaml').read_text()
    fixture = json.loads(Path('/mnt/vm_8tb/b70/results/c4_compare_20261021/strata-warm-fixture.json').read_text())
    prompts = fixture['batches'][0]['prompts']
    checks = []
    for prompt, expected in [('Return only the integer: 17 * 19.', '323'),
                             ('Return only the integer: 23 * 29.', '667')]:
        row = bench.request(base, bench.body(prompt, 16))
        assert row['text'].strip() == expected
        checks.append(row)
    seed = []
    for i, prompt in enumerate(prompts):
        row = bench.request(base, bench.body(prompt, 512))
        row['slot_index'] = i
        (args.output/f'seed-{i}.json').write_text(json.dumps(row,indent=2,ensure_ascii=True)+'\n')
        assert row['usage']['completion_tokens'] == 512 and len(row['text']) > 200
        seed.append(row)
        print(json.dumps({'phase':'seed', 'slot':i, 'sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
                          'prompt_tokens':row['usage']['prompt_tokens'], 'cached_tokens':row['cached_tokens']}), flush=True)
    (args.output/'seed.json').write_text(json.dumps(seed,indent=2,ensure_ascii=True)+'\n')
    rounds = []
    for rep in range(2):
        barrier = threading.Barrier(4)
        def worker(i):
            barrier.wait(timeout=30)
            time.sleep(i * args.stagger_s)
            row = bench.request(base, bench.body(prompts[i], 512))
            row['slot_index'] = i
            return row
        with ThreadPoolExecutor(4) as pool:
            rows = list(pool.map(worker, range(4)))
        (args.output/f'repeat-{rep}.json').write_text(json.dumps(rows,indent=2,ensure_ascii=True)+'\n')
        assert all(r['usage']['completion_tokens'] == 512 and len(r['text']) > 200 for r in rows)
        assert all(r['cached_tokens'] / r['usage']['prompt_tokens'] > .99 for r in rows)
        start_order = sorted(range(4), key=lambda i: rows[i]['start'])
        all_overlap_s = min(r['end'] for r in rows) - max(r['first'] for r in rows)
        assert start_order == [0,1,2,3] and all_overlap_s > 0
        summary = dict(repeat=rep, request_order=start_order, all_streams_overlap_s=all_overlap_s,
                       ttft_s=[r['ttft_s'] for r in rows],
                       decode_tps=[r['decode_tps'] for r in rows],
                       hashes=[hashlib.sha256(r['text'].encode()).hexdigest() for r in rows])
        rounds.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(identity=ids, stagger_s=args.stagger_s,
        seed_hashes=[hashlib.sha256(r['text'].encode()).hexdigest() for r in seed],rounds=rounds,
        note='Diagnostic schedule control; staggered clients change timing and do not reproduce simultaneous arrival.'),indent=2,ensure_ascii=True)+'\n')


if __name__ == '__main__':
    main()
