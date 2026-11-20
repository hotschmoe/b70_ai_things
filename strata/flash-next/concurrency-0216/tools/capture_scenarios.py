#!/usr/bin/env python3
"""Collect B2, B4, B3 and B6 through one already traced, pair-leased server."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/mnt/vm_8tb/github/b70_ai_things')
OUT = Path('/mnt/vm_8tb/b70/results/conc-0216')
DEST = Path(os.environ.get('B_MULTI_DEST', str(OUT)))
sys.path.insert(0, str(ROOT / 'strata/flash-next'))
import bench_serving as bench


def clocks(label):
    a = time.time_ns()
    raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    b = time.time_ns()
    return {'label': label, 'real_before_ns': a, 'real_after_ns': b,
            'raw_ns': raw, 'boot_ns': boot, 'monotonic_ns': mono}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    base = 'http://127.0.0.1:18134'
    model = 'hotschmoe-dd'
    alias = 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216'
    identity = bench.get(base, '/v1/models')
    ids = [r['id'] for r in identity['data']]
    assert ids[0] == model and alias in ids and alias in (ROOT / 'evals/configs/models.yaml').read_text()
    markers = [clocks('begin')]
    def capture(label, messages, cap):
        barrier = threading.Barrier(len(messages))
        def worker(i, message):
            barrier.wait(timeout=60)
            markers.append(clocks(label + '_request_' + str(i)))
            return bench.send(base, model, message, cap,
                              on_first_content=lambda: markers.append(clocks(label + '_first_' + str(i))))
        markers.append(clocks(label + '_begin'))
        with ThreadPoolExecutor(max_workers=len(messages)) as pool:
            rows = list(pool.map(lambda t: worker(*t), enumerate(messages)))
        markers.append(clocks(label + '_end'))
        (DEST / (label + '-responses.json')).write_text(json.dumps(rows, indent=2, ensure_ascii=True) + '\n')
        (DEST / 'b-multi-markers.json').write_text(json.dumps(markers, indent=2, ensure_ascii=True) + '\n')
        print(json.dumps({'scenario': label, 'completion_tokens': [r['usage']['completion_tokens'] for r in rows],
                          'cached_tokens': [r['usage']['prompt_tokens_details']['cached_tokens'] for r in rows],
                          'output_sha256': [r['output_sha256'] for r in rows]}, ensure_ascii=True), flush=True)
        return rows
    a = json.loads((OUT / 'a-baseline-c2/coding-0-0.json').read_text())['request']['messages']
    b = json.loads((OUT / 'a-baseline-c2/coding-0-1.json').read_text())['request']['messages']
    capture('b2', [a, b], 256)
    fixture = json.loads(Path('/mnt/vm_8tb/b70/results/c4_compare_20261021/strata-warm-fixture.json').read_text())
    prompts = fixture['batches'][0]['prompts']
    messages = [[{'role':'user','content':p}] for p in prompts]
    cold = capture('b4', messages, 512)
    assert all(r['usage']['prompt_tokens_details']['cached_tokens'] / r['usage']['prompt_tokens'] < .01 for r in cold)
    warm = capture('b3', messages, 512)
    assert all(r['usage']['prompt_tokens_details']['cached_tokens'] / r['usage']['prompt_tokens'] > .99 for r in warm)
    source = json.loads(Path('/mnt/vm_8tb/b70/results/strata-0216/candidate-bench/p32768-0.json').read_text())
    capture('b6', [source['request']['messages']], 256)
    markers.append(clocks('end'))
    (DEST / 'b-multi-markers.json').write_text(json.dumps(markers, indent=2, ensure_ascii=True) + '\n')


if __name__ == '__main__':
    main()
