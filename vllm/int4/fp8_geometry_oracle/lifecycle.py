#!/usr/bin/env python3
"""Frozen full-attention FP8 geometry oracle lifecycle; selected inherited lease required."""
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
    rows = []
    decoder = json.JSONDecoder()
    for line in path.read_text().splitlines():
        if not line.startswith('{'):
            continue
        try:
            value, _ = decoder.raw_decode(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and 'card' in value:
            rows.append(value)
    return rows


def validate_rows(rows, plan):
    assert rows and all(row['card'] == 0 for row in rows), 'one logical device required'
    exact = [r for r in rows if r.get('exact_representable_write') is True]
    attention = [r for r in rows if 'cache_stride' in r]
    final = [r for r in rows if r.get('k_and_v_read_scales_consumed') is True]
    assert len(exact) == len(attention) == len(final) == 1
    assert exact[0]['untouched_slots'] is True and exact[0]['distinct_kv_scales'] is True
    assert attention[0]['hybrid_layout'] is True
    assert attention[0]['length'] == plan['length']
    assert attention[0]['query_length'] == plan['query_length']
    assert attention[0]['cache_stride'] == [1638400, 512, 256, 1]
    result = final[0]
    assert result['length'] == plan['length'] and result['block_size'] == plan['block_size']
    assert result['block_table'] == list(range(16, 1, -1))
    assert result['q_dtype'] == 'torch.float16'
    assert result['write_k_relative_l2'] < .04 and result['write_v_relative_l2'] < .04
    assert result['attention_relative_l2'] < .015
    assert attention[0]['attention_relative_l2'] == result['attention_relative_l2']


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
                  scope='Full-attention FP8 geometry only; no model/GDN/MTP/publication qualification')
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
        rows = parse_oracle(root / 'oracle.log')
        (root / 'oracle-result.json').write_text(json.dumps(rows, indent=2)+'\n')
        validate_rows(rows, plan)
        assert rc == 0, 'oracle process failed despite complete result'
        result['numeric_passed'] = True
        result['logical_card'] = 0
        result['length'] = plan['length']
        result['query_length'] = plan['query_length']
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
