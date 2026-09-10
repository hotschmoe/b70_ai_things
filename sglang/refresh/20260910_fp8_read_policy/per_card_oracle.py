#!/usr/bin/env python3
"""Frozen per-card oracle plan. --run must inherit the selected GPU lease."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

MAIN = Path(__file__).resolve().parents[1] / '20260910_main'
sys.path.insert(0, str(MAIN))
import oracle_campaign as campaign
import preflight as health


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(plan):
    card = plan['card']
    lease = Path(os.environ.get('B70_GPU_LOCK', '/mnt/vm_8tb/b70/gpu.lock') + '.' + str(card))
    fd = 8 + card
    assert Path('/proc/self/fd/' + str(fd)).resolve(strict=True) == lease.resolve()
    assert os.fstat(fd).st_ino == lease.stat().st_ino
    assert Path(plan['pair_preflight_pass']).is_file(), 'candidate pair preflight must pass first'
    for path, digest in plan['frozen_files'].items():
        assert sha(Path(path)) == digest, path
    strict = Path(plan['health_probe'])
    assert sha(strict) == plan['health_sha256']
    root = Path(plan['output'])
    root.mkdir(exist_ok=False)
    health.ROOT = root
    # Reuse the existing timeout/group-kill/labeled cleanup implementation.
    wrapperdir = root / 'docker-wrapper'
    wrapperdir.mkdir()
    wrapper = wrapperdir / 'docker'
    wrapper.write_text('#!' + sys.executable + '\nimport os, sys\na=sys.argv[1:]\n'
        'if a and a[0] == "run":\n'
        ' a[1:1]=["--label", '+repr(health.LABEL+'=')+'+os.environ["B70_PREFLIGHT_OWNER"]]\n'
        'os.execv('+repr(health.DOCKER)+', ["docker", *a])\n')
    wrapper.chmod(0o755)
    inspected = health.capture([health.DOCKER, 'image', 'inspect', plan['image']])
    assert inspected.returncode == 0
    assert json.loads(inspected.stdout)[0]['Id'] == plan['image']
    (root / 'image-inspect.json').write_text(inspected.stdout)
    (root / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    result = dict(card=card, image=plan['image'], expectation=plan['expectation'],
                  scope='Synthetic nine-case attention oracle only; no model/TP/graph/MTP qualification')
    run_health = lambda phase: health.run([str(strict), '--img', plan['image'], '--card', str(card)],
        root / (phase+'-health.log'), 200, owned=True)
    pre = run_health('pre')
    result['pre_health_rc'] = pre
    if pre:
        result['recovery'] = 'deferred: no pair reset under single-card lease'
        (root / 'OUTCOME.json').write_text(json.dumps(result, indent=2)+'\n')
        return 1
    try:
        (root / ('card'+str(card)) / 'cache').mkdir(parents=True)
        rc = health.run(plan['oracle_command'], root / 'oracle.log', 420, owned=True)
        result['oracle_rc'] = rc
        # Parse even exit1: baseline must retain all nine failing/pass rows.
        parsed = campaign.parse_oracle(root / 'oracle.log')
        (root / 'oracle-result.json').write_text(json.dumps(parsed, indent=2)+'\n')
        assert parsed['source_hashes'] == plan['source_hashes']
        assert parsed['loader_source_sha256'] == plan['loader_sha256']
        expected_cases = {(mode, qlen, label) for mode, qlen in
            [('decode', 1), ('extend', 1), ('extend', 4)] for label in
            ['correct', 'K_descaling_x2', 'V_descaling_x2']}
        assert {(row['mode'], row['query_length'], row['scale_case'])
                for row in parsed['attention']} == expected_cases
        assert all(row['close_gate'] in ('pass', 'fail') for row in parsed['attention'])
        assert all(row.get('scale_sensitivity_gate') in ('pass', 'fail')
                   for row in parsed['attention'] if row['scale_case'] != 'correct')
        derived_gate = 'fail' if any(row['close_gate'] == 'fail' or
            row.get('scale_sensitivity_gate') == 'fail' for row in parsed['attention']) else 'pass'
        assert parsed['attention_gate'] == derived_gate
        expected_rc = 0 if parsed['attention_gate'] == 'pass' else 1
        assert rc == expected_rc, 'oracle process/JSON gate mismatch'
        result['attention_gate'] = parsed['attention_gate']
        result['all_nine_retained'] = True
        result['expectation_met'] = (parsed['attention_gate'] == plan['expectation'])
    except BaseException as exc:
        result['error'] = ascii(exc)
    finally:
        post = run_health('post')
        result['post_health_rc'] = post
        if post:
            result['recovery'] = 'deferred: no pair reset under single-card lease'
        result['passed'] = result.get('attention_gate') == 'pass' and post == 0 and 'error' not in result
        (root / 'OUTCOME.json').write_text(json.dumps(result, indent=2)+'\n')
    # An expected baseline numerical failure remains exit1, never relabeled pass.
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    def interrupted(signum, frame):
        if health.CLEANUP_PENDING:
            return
        raise KeyboardInterrupt('oracle interrupted by signal '+str(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    return execute(plan)


if __name__ == '__main__':
    raise SystemExit(main())
