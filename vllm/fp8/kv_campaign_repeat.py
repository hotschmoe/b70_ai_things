#!/usr/bin/env python3
"""Replay an executed job matrix on a baseline or a fresh candidate lifecycle."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from kv_campaign_server import IMAGE, REPO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kind', choices=['baseline', 'repeat'], required=True)
    p.add_argument('--jobs', help='optional comma-separated executed job stems')
    args = p.parse_args()
    deadline = time.monotonic() + 10800
    while not (args.reference / 'exit.rc').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('reference lifecycle did not finish')
        time.sleep(5)
    if (args.reference / 'exit.rc').read_text().strip() != '0':
        raise RuntimeError('reference health/teardown failed; do not launch automatically')
    reference = json.loads((args.reference / 'manifest.json').read_text())
    cfg = reference['args']
    offload = 0 if args.kind == 'baseline' else cfg['offload_gib']
    image = IMAGE if args.kind == 'baseline' else reference['image']
    model = f"qwen3.8-27b-FP8-official-W8A16-mtp{cfg['mtp']}-{cfg['kv_dtype']}kv-cpu{offload}g-kvcampaign"
    command = ['python3', str(REPO / 'vllm/fp8/kv_campaign_server.py'),
               '--preservation', cfg['preservation'], '--out', str(args.out), '--image', image,
               '--mtp', str(cfg['mtp']), '--kv-dtype', cfg['kv_dtype'],
               '--memory-gib', str(cfg['memory_gib']), '--offload-gib', str(offload)]
    if args.kind == 'repeat':
        for flag in ('offload_group_fix', 'packaged_hooks', 'trace_offload'):
            if cfg.get(flag):
                command.append('--' + flag.replace('_', '-'))
        if cfg.get('offload_group_fix'):
            model += '-gdnfix'
    for flag in ('eager', 'prefix_off'):
        if cfg.get(flag):
            command.append('--' + flag.replace('_', '-'))
    selected = set(args.jobs.split(',')) if args.jobs else None
    jobs = []
    for path in sorted((args.reference / 'jobs').glob('*.running')):
        if path.stem == '99-stop' or (selected is not None and path.stem not in selected):
            continue
        assert path.with_suffix('.done').exists(), path
        job = json.loads(path.read_text())
        job['command'] = [word.replace(str(args.reference), str(args.out)) for word in job['command']]
        if '--model' in job['command']:
            job['command'][job['command'].index('--model') + 1] = model
        jobs.append((path.stem, job))
    assert jobs and (selected is None or selected == {name for name, _ in jobs})
    print('Launching', args.kind, 'with', len(jobs), 'matched jobs', flush=True)
    server = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 660
        while not (args.out / 'jobs').is_dir():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('runner did not create its queue')
            time.sleep(1)
        for name, job in jobs:
            (args.out / 'jobs' / (name + '.json')).write_text(json.dumps(job) + '\n')
        (args.out / 'jobs/99-stop.json').write_text(json.dumps({'command': ['touch', str(args.out / 'STOP')], 'timeout': 10}) + '\n')
        rc = server.wait(timeout=10800)
    finally:
        if server.poll() is None:
            (args.out / 'STOP').touch()
            server.wait(timeout=2400)
    if rc:
        raise RuntimeError('replayed lifecycle failed')
    results = {name: int((args.out / 'jobs' / (name + '.done')).read_text()) for name, _ in jobs}
    (args.out / 'replay-job-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
