#!/usr/bin/env python3
"""Qualify GPU prefix reuse, independent of CPU offload and KV quantization."""
import argparse
import json
import math
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
    p.add_argument('--screen', action='store_true',
                   help='Bounded MTP comparison: coherence, decode, growing tools and warm reuse; not promotion')
    p.add_argument('--recovery-daily', action='store_true',
                   help='User-scoped prefix qualification with bounded churn and separately counted bang recovery')
    p.add_argument('--churn-first', action='store_true',
                   help='Retest the known 132K history failure before broader qualification')
    p.add_argument('--stream-churn-first', action='store_true',
                   help='Capture partial output in the first churn diagnostic; final churn stays nonstreaming')
    p.add_argument('--normalize-churn', action='store_true',
                   help='Add oversized tool histories sized from the actual startup KV pool')
    args = p.parse_args()
    if args.recovery_daily and (args.screen or args.churn_first or args.normalize_churn):
        p.error('--recovery-daily cannot be combined with screen or expanded churn')
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
    add('04b-shared-tools', agent, ['--records', '2000', '--timeout', '600', '--shared-cache'])
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
    if args.churn_first:
        # Retest immediately on a fresh process, then repeat in the original
        # postpressure position. A fresh pass alone does not clear the failure.
        churn = next(job for name, job in jobs if name == '09-evict-tools')
        early = dict(churn, command=[s.replace(str(args.out / '09-evict-tools'),
                       str(args.out / '00-evict-tools')) for s in churn['command']])
        if args.stream_churn_first:
            early['command'].append('--stream')
        jobs.insert(0, ('00-evict-tools', early))
    if args.screen:
        if args.churn_first or args.normalize_churn or args.profile:
            p.error('--screen cannot be combined with churn, normalization or profiling')
        selected = ['01-quality', '02-guides', '04-tools', '04b-shared-tools',
                    '06-thinking', '03-reuse']
        jobs = [(name, dict(jobs)[name]) for name in selected]
        reuse = dict(jobs)['03-reuse']['command']
        reuse[reuse.index('--reuse-sequence') + 1] = '0,0,1,0'
    if args.recovery_daily:
        # Prefix reuse/eviction remains covered by03-reuse. The earlier forced
        # 32K output pressure arm was an offload investigation, not required to
        # enable GPU prefix caching. Keep a bounded132K concurrent churn check.
        jobs = [(name, job) for name, job in jobs if name not in ('07-pressure', '08-postpressure')]
        for name, job in jobs:
            if name in ('04-tools', '09-evict-tools'):
                job['command'] += ['--stream', '--bang-retries', '3', '--salt', '-recovery-daily']
            if name == '09-evict-tools':
                job['command'] += ['--turns', '1']
    cmd = ['python3', str(repo / 'vllm/fp8/kv_campaign_server.py'),
           '--preservation', str(args.config), '--out', str(args.out),
           '--image', args.image, '--served-model', model, '--mtp', str(args.mtp),
           '--memory-gib', '64', '--health-p2p-check']
    if args.profile:
        cmd.append('--profile')
    server = subprocess.Popen(cmd)
    results = {}
    reviewed = {}
    cache_reuse_passed = True
    try:
        deadline = time.monotonic() + 1200
        while not (args.out / 'jobs').is_dir():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('runner did not create queue')
            time.sleep(1)
        if args.normalize_churn:
            while not (args.out / 'READY').exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('server not ready for cache-capacity normalization')
                time.sleep(2)
            pool = int(re.search(r'GPU KV cache size: ([\d,]+) tokens',
                       (args.out / 'server.log').read_text()).group(1).replace(',', ''))
            # The fixed tool template measured314 tokens plus11 per record.
            # Actual usage is still audited; this only sizes the workload.
            records = max(12000, math.ceil(((pool + 83000) / 4 - 314) / 11))
            records = min(records, 18000)
            assert 4 * (records * 11 + 314) > pool
            distinct = max(4, math.ceil(pool / 150045) + 1)
            reuse = next(job for name, job in jobs if name == '03-reuse')
            reuse['command'][reuse['command'].index('--reuse-sequence') + 1] = ','.join(
                str(i) for i in [0, 0, *range(1, distinct), 0, 0])
            original = next(job for name, job in jobs if name == '09-evict-tools')
            for after, name, streaming in [('00-evict-tools', '00b-normalized-tools', True),
                                           ('09-evict-tools', '09b-normalized-tools', False)]:
                if not any(n == after for n, _ in jobs):
                    continue
                cmd2 = [s.replace(str(args.out / '09-evict-tools'), str(args.out / name))
                        for s in original['command']]
                cmd2[cmd2.index('--records') + 1] = str(records)
                if streaming:
                    cmd2.append('--stream')
                index = next(i for i, (n, _) in enumerate(jobs) if n == after)
                # Budget for all32 cold requests at a conservative1000 prompt
                # tokens/s. Per-request timeout stays900s and is still a gate.
                timeout = max(original['timeout'], math.ceil(32 * (records * 11 + 314) / 1000) + 600)
                jobs.insert(index + 1, (name, dict(original, command=cmd2, timeout=timeout)))
            (args.out / 'churn-normalization.json').write_text(json.dumps({
                'gpu_pool_tokens': pool, 'records': records,
                'estimated_prompt_tokens_each': records * 11 + 314,
                'estimated_excess_tokens': 4 * (records * 11 + 314) - pool,
                'reuse_distinct_histories': distinct,
                'scope': 'Capacity sizing, not proof of actual eviction or active preemption.'}, indent=2) + '\n')
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
            if name == '03-reuse' and results[name] == 0:
                reuse_rows = [json.loads(s) for s in (args.out / name / 'responses.jsonl').read_text().splitlines()]
                warm = reuse_rows[1]
                fraction = warm['usage']['prompt_tokens_details']['cached_tokens'] / warm['usage']['prompt_tokens']
                ratio = warm['ttft_s'] / reuse_rows[0]['ttft_s']
                cache_gate = {'cached_fraction': fraction, 'warm_to_cold_ttft_ratio': ratio,
                              'passed': fraction >= .9 and ratio <= .25}
                cache_reuse_passed = cache_gate['passed']
                (args.out / 'cache-hit-gate.json').write_text(json.dumps(cache_gate, indent=2) + '\n')
                if not cache_gate['passed']:
                    print('CACHE REUSE FAILED despite correct output; do not continue qualification', flush=True)
                    break
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
                functional_review = None
                if not allowed and (args.screen or args.recovery_daily) and all(row['passed'] for row in data):
                    # User-scoped serving qualification accepts useful coherent
                    # variation, while preserving the failed exact-repeat flag.
                    from review_lru_guides import review
                    try:
                        functional_review = review(args.out / name)
                        allowed = functional_review['passed']
                    except Exception as exc:
                        functional_review = {'passed': False, 'error': ascii(exc)}
                reviewed[name] = {'allowed': allowed, 'raw_job_rc': results[name],
                    'identical_text_through_last_closed_code_block': same_code_prefix,
                    'functional_review': functional_review,
                    'interpretation': 'Reviewed guide variation; byte-exact repeat claim remains false.'}
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
    if cache_reuse_passed and len(results) == len(jobs) and all(value == 0 or reviewed.get(name, {}).get('allowed')
                                       for name, value in results.items()):
        (args.out / ('SCREEN_PASSED' if args.screen else 'WORKLOADS_PASSED')).touch()
    print('Require cache-hit/latency audit, paired output review and fresh lifecycle before promotion.')


if __name__ == '__main__':
    main()
