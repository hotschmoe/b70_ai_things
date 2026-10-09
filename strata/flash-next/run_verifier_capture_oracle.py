#!/usr/bin/env python3
"""Own bounded actual Verifier two-device queue/capture checks and health."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from parse_usm_logical_free_trace import parse_trace, negative_controls

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
HEALTH = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--leased', action='store_true')
    a = p.parse_args()
    if not a.leased:
        os.execv(str(ROOT / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output.resolve()
    r = dict(scope='Actual Verifier queue initialization and tagged mirror graph control; no model math',
             passed=False, started_epoch=int(time.time()))
    active = None
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, stop)

    def save():
        (out / 'receipt.json').write_text(json.dumps(r, indent=2) + '\n', encoding='ascii')

    def run(cmd, name, timeout=60):
        (out / (name + '.command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
        with (out / name).open('w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=True)

    def health(stage):
        run([str(ROOT / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH], stage + '-health.log', 180)
        run([str(ROOT / 'bin/xpu-collective-health'), '--img', HEALTH, '--p2p', '0', '--timeout', '180'], stage + '-collective.log', 210)
        r[stage + '_health_passed'] = True
        save()

    def faults():
        done = subprocess.run(['journalctl', '-k', '--since', '@' + str(r['started_epoch']), '--no-pager'], capture_output=True, timeout=20)
        (out / 'kernel-journal.log').write_bytes(done.stdout + done.stderr)
        assert done.returncode == 0 and not re.search(rb'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', done.stdout, re.I)

    def cleanup():
        nonlocal active
        while active:
            try:
                ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + active + '$'], timeout=20)
                if not ids.strip():
                    active = None
                    return
                obj = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]
                assert obj['Config']['Labels'].get('b70.capture.owner') == str(os.getpid())
                if obj['State']['Running']:
                    r['cleanup_error'] = 'Owned control needed termination'
                run(['docker', 'rm', '-f', active], 'remove.log')
            except Exception as e:
                r['cleanup_error'] = str(e)
                save()
                time.sleep(2)

    save()
    try:
        oracle = json.loads(a.oracle_receipt.read_text())
        assert oracle['passed'] and oracle['libraries_unchanged'] and oracle['image'] == IMAGE
        root = a.oracle_receipt.parent.resolve()
        assert sha(root / 'oracle.cpp') == oracle['oracle_source_sha256']
        assert sha(root / 'plan.snapshot.json') == oracle['plan_sha256']
        assert sha(root / 'verifier-capture-oracle') == oracle['binary_sha256']
        plan = json.loads((root / 'plan.snapshot.json').read_text())
        assert sha(ROOT / plan['oracle_source']) == oracle['oracle_source_sha256'] == plan['oracle_source_sha256']
        assert sha(oracle['engine_receipt']) == oracle['engine_receipt_sha256']
        for path, expected in oracle['library_sha256'].items():
            assert sha(path) == expected
        r.update(oracle_receipt_sha256=sha(a.oracle_receipt), engine_receipt_sha256=oracle['engine_receipt_sha256'],
                 plan_sha256=oracle['plan_sha256'], controller_sha256=sha(Path(__file__)))
        (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
        health('pre')
        assert not stopped, 'Interrupted'
        active = 'verifier-capture-' + str(os.getpid())
        cmd = ['docker', 'run', '-d', '--name', active, '--label', 'b70.capture.owner=' + str(os.getpid()),
               '--network', 'none', '--device', '/dev/dri', '--user', '1000:1000',
               '--group-add', str(os.stat('/dev/dri/renderD128').st_gid), '--group-add', str(os.stat('/dev/dri/card0').st_gid),
               '--memory', '2g', '--memory-swap', '2g', '-e', 'ZE_AFFINITY_MASK=0,1',
               '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu', '-e', 'SYCL_CACHE_PERSISTENT=0',
               '-e', 'UR_ENABLE_LAYERS=UR_LAYER_TRACING', '-e', 'UR_LOG_TRACING=level:info;flush:info;output:stderr',
               '-v', str(root) + ':/oracle:ro', '-v', str(out) + ':/results', IMAGE,
               'exec /oracle/verifier-capture-oracle /results/control.json']
        run(cmd, 'launch.log')
        deadline = time.monotonic() + 180
        while True:
            state = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]['State']
            if not state['Running']:
                break
            assert not stopped and time.monotonic() < deadline, 'Control deadline/interruption'
            faults()
            time.sleep(2)
        run(['docker', 'logs', active], 'control.log')
        assert state['ExitCode'] == 0 and not state.get('OOMKilled'), state
        report = json.loads((out / 'control.json').read_text())
        assert report['schema'] == 1 and report['passed'] is True
        assert all(report.get(k) == v for k, v in plan['expected_report'].items())
        assert report['capture_token_shapes'] == [2, 1]
        text = (out / 'control.log').read_text()
        ledger = parse_trace(text, require_owners=False)
        assert ledger['passed'], ledger['errors']
        ledger['negative_controls'] = negative_controls(text, False)
        assert all(ledger['negative_controls'].values())
        (out / 'logical-free.json').write_text(json.dumps(ledger, indent=2) + '\n')
        r.update(control_passed=True, terminal=state, report=report, logical_free_counts=ledger['counts'],
                 logical_free_scope='All traced USM allocations balanced; no per-owner destruction markers in this control')
        cleanup()
    except Exception as e:
        r['error'] = dict(type=type(e).__name__, message=str(e))
    finally:
        cleanup()
        if r.get('pre_health_passed'):
            try:
                health('post')
                faults()
            except Exception as e:
                r['post_health_error'] = str(e)
        r['passed'] = bool(r.get('control_passed') and r.get('post_health_passed') and not any(
            k in r for k in ['error', 'cleanup_error', 'post_health_error']))
        r['finished_epoch'] = int(time.time())
        save()
    print(json.dumps(dict(passed=r['passed'], receipt=str(out / 'receipt.json'))))
    return 0 if r['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
