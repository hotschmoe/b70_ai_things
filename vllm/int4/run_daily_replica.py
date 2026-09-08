#!/usr/bin/env python3
"""Qualify the larger plain INT4 profile only after strict parity and speed."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from prepare_replica import IMAGE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--strict', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--coding-baseline', type=Path, help='clean coding reference, separate from workload templates')
    args = p.parse_args()
    if not (args.strict / 'PASSED').exists():
        raise RuntimeError('strict replica has not passed')
    rates = [json.loads((args.strict / name / 'strict/performance.json').read_text())
             ['summary']['class_balanced_tok_s_1_100_intervals_after_ttft']['median']
             for name in ('mtp4-a', 'mtp4-b')]
    if sum(rates) / 2 < 95:
        raise RuntimeError('preregistered near-100 tok/s target was not reached')
    assert (args.reference / 'exit.rc').read_text().strip() == '0'
    identity = json.loads((args.config / 'recipe-identity.json').read_text())
    assert identity['profile'] == 'daily-prefix-off'
    repo = Path(__file__).resolve().parents[2]
    model = 'qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276-mtp4-fp16kv-daily-prefixoff'
    names = ['01-quality-c1', '02-quality-c4', '03-decode', '09-long-agent',
             '10-code', '04-reuse', '07-cancel', '08-thinking', '12-long180k', '11-pressure']
    jobs = []
    for name in names:
        job = json.loads((args.reference / 'jobs' / (name + '.running')).read_text())
        job['command'] = [s.replace(str(args.reference), str(args.out)) for s in job['command']]
        job['command'][job['command'].index('--model') + 1] = model
        if name == '11-pressure':
            job['command'][job['command'].index('--concurrency') + 1] = '4'
        jobs.append((name, job))
    near_limit = json.loads(json.dumps(dict(jobs)['12-long180k']))
    near_limit['command'] = [s.replace('12-long180k', '13-long199k') for s in near_limit['command']]
    near_limit['command'][near_limit['command'].index('--tokens') + 1] = '199000'
    jobs.insert(-1, ('13-long199k', near_limit))
    jobs.append(('98-profile', {'command': ['python3', str(repo / 'vllm/int4/profile_replica.py'),
                 '--model', model, '--out', str(args.out / '98-profile')], 'timeout': 900}))
    command = ['python3', str(repo / 'vllm/fp8/kv_campaign_server.py'),
               '--preservation', str(args.config), '--out', str(args.out), '--image', IMAGE,
               '--served-model', model, '--mtp', '4', '--memory-gib', '64',
               '--health-p2p-check', '--profile']
    server = subprocess.Popen(command)
    results = {}
    try:
        deadline = time.monotonic() + 900
        while not (args.out / 'jobs').is_dir():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('daily lifecycle did not create queue')
            time.sleep(1)
        (args.out / 'daily-plan.json').write_text(json.dumps(dict(jobs), indent=2) + '\n')
        for name, job in jobs:
            (args.out / 'jobs' / (name + '.json')).write_text(json.dumps(job) + '\n')
            deadline = time.monotonic() + job.get('timeout', 1800) + 1800
            done = args.out / 'jobs' / (name + '.done')
            while not done.exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('daily job did not finish: ' + name)
                time.sleep(2)
            results[name] = int(done.read_text())
            print(name, results[name], flush=True)
            if results[name] and name != '03-decode':
                break
            if name == '10-code':
                comparison = args.out / 'paired-code.json'
                subprocess.run(['python3', str(repo / 'vllm/int4/compare_code.py'),
                                '--baseline', str(args.coding_baseline or args.reference / '10-code'),
                                '--candidate', str(args.out / '10-code'), '--out', str(comparison)], check=True)
                if not json.loads(comparison.read_text())['score_gate_passed']:
                    break
    finally:
        if (args.out / 'jobs').is_dir():
            (args.out / 'STOP').touch()
        rc = server.wait()
    (args.out / 'daily-job-results.json').write_text(json.dumps(results, indent=2) + '\n')
    if rc:
        raise RuntimeError('daily lifecycle health failed')
    if len(results) == len(jobs) and all(value == 0 for value in results.values()):
        (args.out / 'WORKLOADS_PASSED').touch()
    print('Manual paired failure review, profile audit and fresh lifecycle remain required.', flush=True)


if __name__ == '__main__':
    main()
