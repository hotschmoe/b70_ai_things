#!/usr/bin/env python3
"""Plan by default. Real execution requires bin/gpu-run holding both cards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys

import preflight as health

HERE = Path(__file__).resolve().parent
RAW = Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910')
ROOT = RAW / 'sglang-main-fp8-oracle-campaign'
SOURCE = RAW / 'sglang-main-refresh/sources/sglang'
CANDIDATE = HERE.parents[1] / 'cache800k'
IMAGE = 'sha256:bdc51c5f083fdcbacd59a74bfeb8389fa6d62c8478e066ed5c62f9bffcfaa4cf'
INPUTS = ('fp8_oracle/oracle.py', 'calibrated_kv/scale_loader.py',
          'calibrated_kv/scale_plan.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_command(card, root=ROOT):
    cache = root / ('card' + str(card)) / 'cache'
    return ['docker', 'run', '--name', 'b70-sglang-main-fp8-oracle-card' + str(card),
            '--network', 'none', '--device', '/dev/dri', '--cpus', '4',
            '--memory', '8g', '--memory-swap', '8g', '--ulimit', 'core=0',
            '-v', '/dev/dri/by-path:/dev/dri/by-path:ro',
            '-v', str(SOURCE) + ':/source:ro',
            '-v', str(root / 'inputs') + ':/candidate:ro',
            '-v', str(cache) + ':/cache',
            '-e', 'ZE_AFFINITY_MASK=' + str(card),
            '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu',
            '-e', 'CCL_TOPO_P2P_ACCESS=0',
            '-e', 'TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor',
            '-e', 'TRITON_CACHE_DIR=/cache/triton',
            '-e', 'NEO_CACHE_DIR=/cache/neo',
            '--entrypoint', 'python3', IMAGE,
            '/candidate/fp8_oracle/oracle.py', '--device', 'xpu',
            '--source', '/source', '--loader-root', '/candidate/calibrated_kv']


def parse_oracle(path):
    text = path.read_text()
    decoder = json.JSONDecoder()
    for start in [i for i, value in enumerate(text) if value == '{']:
        try:
            result, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and result.get('attention_execution') == 'actual Triton kernels':
            assert result['device'] == 'xpu'
            assert result['bytes'] == 'pass'
            assert result['untouched_slots'] == 'pass'
            assert result['cloned_inputs'] == 'pass'
            assert len(result['attention']) == 9
            return result
    raise ValueError('No complete actual-Triton oracle result in successful process log')


def post_health(root):
    out = root / 'post-health'; out.mkdir()
    env = dict(os.environ, XPU_COLLECTIVE_HEALTH_CACHE=str(
        root / 'sglang-main-triton-dense' / 'compiled-cache'))
    strict = str(RAW / 'health-repair/xpu-health')
    percard = health.run([strict, '--img', IMAGE], out / 'per-card.log', 200,
                         env, owned=True)
    pair = 'skipped: per-card failed'
    if percard == 0:
        pair = health.run([str(health.REPO / 'bin/xpu-collective-health'),
                           '--img', IMAGE, '--ccl-root', '/opt/venv', '--p2p', '0',
                           '--timeout', '180'], out / 'compiled-pair.log', 240,
                          env, owned=True)
    result = {'per_card_rc': percard, 'compiled_pair_rc': pair}
    if percard != 0 or pair != 0:
        # All health.run calls verify their owned containers are absent first.
        reset = health.run(['env', 'B70_XE_RESET_UNDER_LEASE=1',
                            str(health.REPO / 'bin/xe-reset'), '--method', 'rebind',
                            '--no-probe'], out / 'recovery.log', 240)
        result['reset_rc'] = reset
        if reset == 0:
            baseline_env = dict(os.environ, XPU_COLLECTIVE_HEALTH_CACHE=str(
                out / 'baseline-recovery-compiled-cache'))
            recovered = health.run([strict, '--img', health.BASELINE],
                out / 'recovery-per-card.log', 200, baseline_env, owned=True)
            result['recovery_per_card_rc'] = recovered
            if recovered == 0:
                result['recovery_pair_rc'] = health.run(
                    [str(health.REPO / 'bin/xpu-collective-health'), '--img',
                     health.BASELINE, '--ccl-root', '/opt/venv', '--p2p', '0',
                     '--timeout', '180'], out / 'recovery-pair.log', 240,
                    baseline_env, owned=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result, percard == 0 and pair == 0


def execute():
    # main() verifies both lease FDs, strict-probe hash and image identity,
    # creates the owner-label Docker wrapper, and runs pre-health exactly once.
    health.ROOT = ROOT
    health.IMAGES = [('sglang-main-triton-dense', IMAGE)]
    rc = health.main()
    if rc:
        return rc
    (ROOT / 'PASS').rename(ROOT / 'PREHEALTH_PASS')
    result = {'image': IMAGE, 'oracles': [], 'passed': False,
              'scope': 'Synthetic FP8 store/read only; no model, TP, graph or MTP qualification'}
    try:
        snapshots = ROOT / 'inputs'
        hashes = {}
        for name in INPUTS:
            dest = snapshots / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CANDIDATE / name, dest)
            hashes[name] = digest(dest)
        source_hashes = {str(p.relative_to(SOURCE)): digest(p) for p in (
            SOURCE / 'python/sglang/srt/mem_cache/memory_pool.py',
            SOURCE / 'python/sglang/srt/layers/attention/triton_backend.py',
            SOURCE / 'python/sglang/kernels/ops/attention/decode_attention.py',
            SOURCE / 'python/sglang/kernels/ops/attention/extend_attention.py')}
        (ROOT / 'input-manifest.json').write_text(json.dumps(
            dict(image=IMAGE, source=str(SOURCE), inputs=hashes,
                 source_hashes=source_hashes, runner_sha256=digest(Path(__file__)),
                 helper_sha256=digest(Path(health.__file__))), indent=2) + '\n')
        for card in (0, 1):
            out = ROOT / ('card' + str(card)); (out / 'cache').mkdir(parents=True)
            command = oracle_command(card, ROOT)
            (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            rc = health.run(command, out / 'oracle.log', 420, owned=True)
            row = dict(physical_card=card, ze_affinity_mask=str(card), rc=rc)
            result['oracles'].append(row)
            if rc:
                break
            parsed = parse_oracle(out / 'oracle.log')
            assert parsed['source_hashes'] == source_hashes
            assert parsed['loader_source_sha256'] == hashes['calibrated_kv/scale_loader.py']
            (out / 'oracle-result.json').write_text(json.dumps(parsed, indent=2) + '\n')
            row['numeric_checks'] = 'pass'
    except BaseException as exc:
        result['error'] = ascii(exc)
    finally:
        post, post_ok = post_health(ROOT)
        result['post_health'] = post
        result['passed'] = (len(result['oracles']) == 2 and
            all(row.get('numeric_checks') == 'pass' for row in result['oracles']) and
            'error' not in result and post_ok)
        (ROOT / 'OUTCOME.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 1


def interrupted(signum, frame):
    if health.CLEANUP_PENDING:
        return
    raise KeyboardInterrupt('oracle campaign interrupted by signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='requires both inherited GPU leases')
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(dict(output=str(ROOT), image=IMAGE,
            stages=['strict per-card + compiled P2P0 pre-health', 'physical card0 oracle',
                    'physical card1 oracle', 'strict per-card + compiled P2P0 post-health'],
            oracle_commands=[oracle_command(card) for card in (0, 1)],
            launch=['bin/gpu-run', 'python3', str(Path(__file__)), '--run']), indent=2))
        return 0
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    return execute()


if __name__ == '__main__':
    raise SystemExit(main())
