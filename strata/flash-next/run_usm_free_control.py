#!/usr/bin/env python3
"""Run frozen allocator metadata controls on both cards, with owned cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
HEALTH = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--leased', action='store_true')
    p.add_argument('--trace', action='store_true', help='Separate UR logical allocation/free diagnostic; allocator settings unchanged')
    a = p.parse_args()
    if not a.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output.resolve()
    binary = a.binary.resolve()
    result = dict(scope='Allocator metadata control, not model/teardown qualification', cards=[], passed=False,
                  binary_sha256=sha(binary), source_sha256=sha(REPO / 'strata/flash-next/usm_free_query_control.cpp'),
                  trace=a.trace, started_epoch=int(time.time()), controller_sha256=sha(Path(__file__)))
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    active = None
    interrupted = False

    def stop(*unused):
        nonlocal interrupted
        interrupted = True

    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, stop)

    def save():
        (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(cmd, log, timeout=60, check=True):
        (out / (log + '.command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
        with (out / log).open('w') as f:
            done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        if check and done.returncode:
            raise RuntimeError(log + ' failed ' + str(done.returncode))

    def health(stage):
        run([str(REPO / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH], stage + '-health.log', 180)
        run([str(REPO / 'bin/xpu-collective-health'), '--img', HEALTH, '--p2p', '0', '--timeout', '180'], stage + '-collective.log', 210)
        result[stage + '_health_passed'] = True
        save()

    def faults():
        done = subprocess.run(['journalctl', '-k', '--since', '@' + str(result['started_epoch']), '--no-pager'], capture_output=True, timeout=20)
        (out / 'kernel-journal.log').write_bytes(done.stdout + done.stderr)
        if done.returncode or re.search(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', done.stdout.decode(errors='replace'), re.I):
            raise RuntimeError('Kernel journal unavailable or GPU fault signature')

    def cleanup():
        nonlocal active
        while active:
            try:
                ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + active + '$'], timeout=20)
                if not ids.strip():
                    active = None
                    return
                obj = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]
                assert obj['Config']['Labels']['b70.usm.owner'] == str(os.getpid())
                if obj['State']['Running']:
                    result['cleanup_error'] = 'Control needed termination'
                run(['docker', 'rm', '-f', active], active + '-remove.log')
            except Exception as e:
                result['cleanup_error'] = str(e)
                save()
                time.sleep(2)

    try:
        health('pre')
        for card in [0, 1]:
            faults()
            if interrupted:
                raise RuntimeError('Interrupted')
            active = f'usm-control-{os.getpid()}-{card}'
            cmd = ['docker', 'run', '-d', '--name', active, '--label', 'b70.usm.owner=' + str(os.getpid()),
                   '--network', 'none', '--device', '/dev/dri', '--user', '1000:1000',
                   '--group-add', str(os.stat('/dev/dri/renderD128').st_gid), '--group-add', str(os.stat('/dev/dri/card0').st_gid),
                   '--memory', '2g', '--memory-swap', '2g', '-v', str(binary.parent) + ':/control:ro',
                   '-v', str(out) + ':/results', '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu',
                   '-e', 'ZE_AFFINITY_MASK=' + str(card), '-e', 'SYCL_CACHE_PERSISTENT=0', IMAGE,
                   f'exec /control/{binary.name} --device 0 --rounds 2 --output /results/card{card}.json']
            if a.trace:
                at = cmd.index(IMAGE)
                cmd[at:at] = ['-e', 'UR_ENABLE_LAYERS=UR_LAYER_TRACING', '-e', 'UR_LOG_TRACING=level:info;flush:info;output:stderr']
            run(cmd, f'card{card}-launch.log')
            deadline = time.monotonic() + 180
            while True:
                state = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]['State']
                if not state['Running']:
                    break
                if interrupted or time.monotonic() > deadline:
                    raise RuntimeError('Control deadline/interruption')
                faults()
                time.sleep(2)
            run(['docker', 'logs', active], f'card{card}.log')
            report = json.loads((out / f'card{card}.json').read_text())
            assert state['ExitCode'] == 0 and not state.get('OOMKilled') and report['allocator_operations_passed']
            cases = report['cases']
            expected = {(r, path, n) for r in [0, 1] for path in ['sycl_malloc_device', 'strata_malloc_device_guarded', 'raw_zeMemAllocDevice']
                        for n in [8192, 40960, 163840, 3481600, 6963200, 27852800, 33554432]}
            assert len(cases) == report['completed_cases'] == 42
            assert {(c['round'], c['path'], c['requested_bytes']) for c in cases} == expected
            assert all(c['owning_free_returned'] and c['fresh_probe_passed'] and c['fresh_free_returned'] for c in cases)
            result['cards'].append(dict(card=card, state=state, report=report))
            cleanup()
            save()
        health('post')
        faults()
        result['passed'] = not result.get('cleanup_error')
    except Exception as e:
        result['error'] = str(e)
    finally:
        cleanup()
        if result.get('pre_health_passed') and not result.get('post_health_passed'):
            try:
                health('post')
            except Exception as e:
                result['post_health_error'] = str(e)
        save()
    print(json.dumps(dict(passed=result['passed'], receipt=str(out / 'receipt.json'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
