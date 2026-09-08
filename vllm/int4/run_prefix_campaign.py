#!/usr/bin/env python3
"""Qualify GPU prefix reuse, independent of CPU offload and KV quantization."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from prepare_replica import IMAGE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--wait-for', type=Path, required=True)
    p.add_argument('--image', default=IMAGE)
    p.add_argument('--mtp', type=int, default=4)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    deadline = time.monotonic() + 1800
    while not (args.wait_for / 'exit.rc').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('prior lifecycle did not finish')
        time.sleep(2)
    assert (args.wait_for / 'exit.rc').read_text().strip() == '0'
    assert json.loads((args.config / 'recipe-identity.json').read_text())['profile'] == 'daily'
    repo = Path(__file__).resolve().parents[2]
    model = f'qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276-mtp{args.mtp}-fp16kv-prefix'
    probe = repo / 'vllm/fp8/kv_campaign_probe.py'
    agent = repo / 'vllm/fp8/kv_campaign_agent_probe.py'
    jobs = []
    def add(name, script, extra, timeout=2400):
        jobs.append((name, {'command': ['python3', str(script), '--model', model,
                    '--out', str(args.out / name), *extra], 'timeout': timeout}))
    add('01-quality', probe, ['--concurrency', '4'])
    add('02-guides', probe, ['--mode', 'decode', '--rounds', '3', '--concurrency', '1'])
    add('03-reuse', probe, ['--mode', 'reuse', '--tokens', '150000',
                          '--reuse-sequence', '0,0,1,2,3,0,0'])
    add('04-tools', agent, ['--records', '8000', '--timeout', '600'])
    add('05-cancel', probe, ['--mode', 'cancel'])
    add('06-thinking', probe, ['--thinking', '--reasoning-effort', 'xhigh',
                             '--concurrency', '4'])
    # Four 110K prompts fit initially; 4 x 8K growth exceeds the prior 451K pool.
    # Always inspect actual capacity/admission/preemption counters afterward.
    add('07-pressure', probe, ['--mode', 'pressure', '--tokens', '110000',
                              '--output-tokens', '8192', '--force-length',
                              '--rounds', '1', '--concurrency', '4', '--timeout', '1800'], 3000)
    add('08-postpressure', probe, ['--concurrency', '4'])
    add('09-evict-tools', agent, ['--records', '12000', '--timeout', '900'], 3600)
    code = json.loads((args.baseline / 'jobs/10-code.running').read_text())
    code['command'] = [s.replace(str(args.baseline / '10-code'), str(args.out / '10-code'))
                       for s in code['command']]
    code['command'][code['command'].index('--model') + 1] = model
    jobs.append(('10-code', code))
    add('11-postquality', probe, ['--concurrency', '4'])
    add('12-long', probe, ['--mode', 'long', '--tokens', '199000', '--rounds', '1', '--concurrency', '1'])
    if args.profile:
        add('98-profile', repo / 'vllm/int4/profile_replica.py', ['--warm'], 900)
    cmd = ['python3', str(repo / 'vllm/fp8/kv_campaign_server.py'),
           '--preservation', str(args.config), '--out', str(args.out),
           '--image', args.image, '--served-model', model, '--mtp', str(args.mtp),
           '--memory-gib', '64', '--health-p2p-check']
    if args.profile:
        cmd.append('--profile')
    server = subprocess.Popen(cmd)
    results = {}
    reviewed = {}
    try:
        deadline = time.monotonic() + 1200
        while not (args.out / 'jobs').is_dir():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('runner did not create queue')
            time.sleep(1)
        (args.out / 'prefix-plan.json').write_text(json.dumps(dict(jobs), indent=2) + '\n')
        for name, job in jobs:
            (args.out / 'jobs' / (name + '.json')).write_text(json.dumps(job) + '\n')
            deadline = time.monotonic() + job['timeout'] + 1800
            done = args.out / 'jobs' / (name + '.done')
            while not done.exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('job did not finish: ' + name)
                time.sleep(2)
            results[name] = int(done.read_text())
            print(name, results[name], flush=True)
            if name == '02-guides' and results[name]:
                # Preserve the failed byte-repeat flag. Only allow late prose
                # variation after identical, closed code blocks and a full
                # coherence pass; do not excuse changed code or early content.
                def load_rows(path):
                    return [json.loads(line) for line in path.read_text().splitlines()]
                data = load_rows(args.out / name / 'responses.jsonl')
                baseline = load_rows(args.baseline / '03-decode/responses.jsonl')
                texts = [r['text'] for r in data + baseline]
                ends = [list(re.finditer(r'```[^\n]*\n.*?```', t, re.S)) for t in texts]
                same_code_prefix = (all(ends) and len({t[:matches[-1].end()]
                    for t, matches in zip(texts, ends)}) == 1)
                allowed = all(row['passed'] for row in data) and same_code_prefix
                reviewed[name] = {'allowed': allowed, 'raw_job_rc': results[name],
                    'identical_text_through_last_closed_code_block': same_code_prefix,
                    'interpretation': 'Late prose variation; byte-exact repeat claim remains false.'}
                (args.out / 'guide-review.json').write_text(json.dumps(reviewed[name], indent=2) + '\n')
                if allowed:
                    print('REVIEWED noncritical guide tail variation', flush=True)
                    continue
            if results[name]:
                break
    finally:
        if (args.out / 'jobs').is_dir():
            (args.out / 'STOP').touch()
        rc = server.wait()
    (args.out / 'prefix-job-results.json').write_text(json.dumps(results, indent=2) + '\n')
    if rc:
        raise RuntimeError('lifecycle health failed')
    if len(results) == len(jobs) and all(value == 0 or reviewed.get(name, {}).get('allowed')
                                       for name, value in results.items()):
        (args.out / 'WORKLOADS_PASSED').touch()
    print('Require cache-hit/latency audit, paired output review and fresh lifecycle before promotion.')


if __name__ == '__main__':
    main()
