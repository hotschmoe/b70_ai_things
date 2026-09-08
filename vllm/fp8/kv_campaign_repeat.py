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
    p.add_argument('--candidate-image', help='replace repeat image with a native Python backport; no hooks')
    p.add_argument('--wait-for', type=Path, help='additional lifecycle that must finish healthy first')
    p.add_argument('--fail-fast', action='store_true', help='queue sequentially and stop on a failed job')
    args = p.parse_args()
    if args.candidate_image and args.kind != 'repeat':
        p.error('candidate image requires repeat mode')
    if args.wait_for:
        deadline = time.monotonic() + 10800
        while not (args.wait_for / 'exit.rc').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('prior lifecycle did not finish')
            time.sleep(5)
        if (args.wait_for / 'exit.rc').read_text().strip() != '0':
            raise RuntimeError('prior lifecycle health failed')
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
    if args.candidate_image:
        image = args.candidate_image
    model = f"qwen3.8-27b-FP8-official-W8A16-mtp{cfg['mtp']}-{cfg['kv_dtype']}kv-cpu{offload}g-kvcampaign"
    if args.candidate_image:
        model += '-merged'
    command = ['python3', str(REPO / 'vllm/fp8/kv_campaign_server.py'),
               '--preservation', cfg['preservation'], '--out', str(args.out), '--image', image,
               '--mtp', str(cfg['mtp']), '--kv-dtype', cfg['kv_dtype'],
               '--memory-gib', str(cfg['memory_gib']), '--offload-gib', str(offload)]
    if args.candidate_image:
        command += ['--served-model', model]
    if args.kind == 'repeat' and not args.candidate_image:
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
    if args.jobs:
        by_name = dict(jobs)
        jobs = [(name, by_name[name]) for name in args.jobs.split(',')]
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
            if args.fail_fast:
                deadline = time.monotonic() + job.get('timeout', 1800) + 1800
                done = args.out / 'jobs' / (name + '.done')
                while not done.exists():
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('sequential job did not complete: ' + name)
                    time.sleep(2)
                if done.read_text().strip() != '0':
                    print('Stopping at failed qualification job ' + name, flush=True)
                    break
        (args.out / 'jobs/99-stop.json').write_text(json.dumps({'command': ['touch', str(args.out / 'STOP')], 'timeout': 10}) + '\n')
        rc = server.wait(timeout=10800)
    finally:
        if server.poll() is None:
            (args.out / 'STOP').touch()
            server.wait(timeout=2400)
    if rc:
        raise RuntimeError('replayed lifecycle failed')
    results = {name: (int((args.out / 'jobs' / (name + '.done')).read_text())
                     if (args.out / 'jobs' / (name + '.done')).exists() else None)
               for name, _ in jobs}
    (args.out / 'replay-job-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
