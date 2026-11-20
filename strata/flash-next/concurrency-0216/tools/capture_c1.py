#!/usr/bin/env python3
"""Warm a leased server and capture one diagnostic C1 request with unitrace."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/vm_8tb/github/b70_ai_things')
OUT = Path('/mnt/vm_8tb/b70/results/conc-0216')
PREFIX = os.environ.get('B1_PREFIX', 'b1')
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


def control(action):
    tool = '/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace'
    result = subprocess.run(['docker', 'exec', 'conc-0216-b-trace', tool,
                             '--' + action, 'conc0216'], capture_output=True, text=True, timeout=30)
    return {'action': action, 'returncode': result.returncode, 'stdout': result.stdout,
            'stderr': result.stderr}


def main():
    base = 'http://127.0.0.1:18134'
    identity = bench.get(base, '/v1/models')
    ids = [x['id'] for x in identity['data']]
    alias = 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216'
    assert ids[0] == 'hotschmoe-dd' and alias in ids
    assert alias in (ROOT / 'evals/configs/models.yaml').read_text()
    warm = bench.send(base, 'hotschmoe-dd', [{'role': 'user', 'content':
                      'Return only the integer: 17 * 19.'}], 32)
    assert warm['text'].strip() == '323'
    request = json.loads((OUT / 'a-baseline-c1/coding-0.json').read_text())['request']
    markers = [clocks('before_resume')]
    resume = control('resume')
    markers.append(clocks('after_resume'))
    if resume['returncode']:
        raise RuntimeError('unitrace resume failed: ' + json.dumps(resume))
    try:
        markers.append(clocks('before_request'))
        result = bench.send(base, 'hotschmoe-dd', request['messages'], 256,
                            on_first_content=lambda: markers.append(clocks('first_content')))
        markers.append(clocks('after_request'))
    finally:
        markers.append(clocks('before_pause'))
        pause = control('pause')
        markers.append(clocks('after_pause'))
        (OUT / (PREFIX + '-markers.json')).write_text(json.dumps({'markers': markers, 'resume': resume,
                                                        'pause': pause}, indent=2, ensure_ascii=True) + '\n')
    if pause['returncode']:
        raise RuntimeError('unitrace pause failed: ' + json.dumps(pause))
    (OUT / (PREFIX + '-response.json')).write_text(json.dumps({'identity': identity, 'warm': warm,
                                                     'profiled': result}, indent=2, ensure_ascii=True) + '\n')
    print(json.dumps({'response_tokens': result['usage']['completion_tokens'],
                      'profiled_tps_diagnostic': result['client_decode_tps'],
                      'output_sha256': result['output_sha256']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
