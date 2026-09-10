#!/usr/bin/env python3
"""Frozen Mamba byte oracle lifecycle; selected inherited lease required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

MAIN = Path(__file__).resolve().parents[3] / 'sglang/refresh/20260910_main'
sys.path.insert(0, str(MAIN))
import preflight as health


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_oracle(path):
    text = path.read_text()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('execution') == 'actual installed Triton copy kernels':
            return value
    raise ValueError('No completed actual-kernel Mamba oracle report')


def execute(plan):
    card = plan['card']
    lease = Path(os.environ.get('B70_GPU_LOCK', '/mnt/vm_8tb/b70/gpu.lock') + '.' + str(card))
    fd = 8 + card
    assert Path('/proc/self/fd/' + str(fd)).resolve(strict=True) == lease.resolve()
    assert os.fstat(fd).st_ino == lease.stat().st_ino
    assert Path(plan['pair_preflight_pass']).is_file(), 'image pair preflight must pass first'
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
    result = dict(card=card, image=plan['image'],
                  scope='Synthetic padded Mamba copy only; no native GDN/publication/model qualification')
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
        # Keep the raw log on every exit and full JSON on numerical exit1.
        parsed = parse_oracle(root / 'oracle.log')
        (root / 'oracle-result.json').write_text(json.dumps(parsed, indent=2)+'\n')
        assert parsed['worker_source_sha256'] == plan['worker_source_sha256']
        assert parsed['model_source_sha256'] == plan['model_source_sha256']
        assert [row['case'] for row in parsed['cases']] == plan['cases']
        expected_pass = all(row['mismatched_bytes'] == 0 and row['source_untouched']
                            and row['all_untargeted_bytes_covered'] for row in parsed['cases'])
        assert parsed['passed'] == expected_pass
        assert all(row['passed'] == (row['mismatched_bytes'] == 0 and row['source_untouched'])
                   for row in parsed['cases'])
        assert rc == (0 if expected_pass else 1), 'process/JSON gate mismatch'
        result['numeric_passed'] = expected_pass
        result['all_cases_retained'] = True
    except BaseException as exc:
        result['error'] = ascii(exc)
    finally:
        post = run_health('post')
        result['post_health_rc'] = post
        if post:
            result['recovery'] = 'deferred: no pair reset under single-card lease'
        result['passed'] = result.get('numeric_passed') is True and post == 0 and 'error' not in result
        (root / 'OUTCOME.json').write_text(json.dumps(result, indent=2)+'\n')
    # A numerical failure remains exit1; cleanup/health success cannot mask it.
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
