#!/usr/bin/env python3
"""Record the original B2 requests and clock boundaries on a traced server."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/mnt/vm_8tb/github/b70_ai_things')
BASE = Path('/mnt/vm_8tb/b70/results/conc-0216')
OUT = BASE / 'c-attn-trace-c2'
sys.path.insert(0, str(ROOT/'strata/flash-next'))
import bench_serving as bench


def clock(label):
    a = time.time_ns()
    raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    b = time.time_ns()
    return dict(label=label,real_before_ns=a,real_after_ns=b,raw_ns=raw,boot_ns=boot,monotonic_ns=mono)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    base = 'http://127.0.0.1:18134'
    ids = [r['id'] for r in bench.get(base, '/v1/models')['data']]
    alias = 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216'
    assert ids[0] == 'hotschmoe-dd' and alias in ids and alias in (ROOT/'evals/configs/models.yaml').read_text()
    messages = [json.loads((BASE/f'a-baseline-c2/coding-0-{i}.json').read_text())['request']['messages'] for i in range(2)]
    markers = [clock('b2_begin')]
    gate = threading.Barrier(2)
    def worker(i):
        gate.wait(timeout=60)
        markers.append(clock('b2_request_'+str(i)))
        return bench.send(base,'hotschmoe-dd',messages[i],256,
                          on_first_content=lambda:markers.append(clock('b2_first_'+str(i))))
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(worker,range(2)))
    markers.append(clock('b2_end'))
    (OUT/'markers.json').write_text(json.dumps(markers,indent=2,ensure_ascii=True)+'\n')
    (OUT/'responses.json').write_text(json.dumps(rows,indent=2,ensure_ascii=True)+'\n')
    print(json.dumps(dict(completion_tokens=[r['usage']['completion_tokens'] for r in rows],
                          hashes=[r['output_sha256'] for r in rows]),ensure_ascii=True))
    assert all(r['usage']['completion_tokens'] == 256 for r in rows)


if __name__ == '__main__': main()
