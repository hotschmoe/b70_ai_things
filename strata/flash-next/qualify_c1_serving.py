#!/usr/bin/env python3
"""Parent-owned serving screen, stop and health around a prepared C1 profile."""
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

ROOT = Path(__file__).resolve().parents[2]
HEALTH = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--leased', action='store_true')
    a = p.parse_args()
    if not a.leased:
        os.execv(str(ROOT / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    out = a.prepared.resolve()
    ctrl = ROOT / 'strata/flash-next/c1_serve_controller.py'
    started = int(time.time())
    result = dict(scope='Bounded C1 model screen, not full fidelity/latency/shelf qualification', passed=False)
    child = None
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True

    for s in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(s, stop)

    def run(cmd, name, timeout=210, check=True):
        with (out / name).open('w') as f:
            done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        if check and done.returncode:
            raise RuntimeError(name + ' failed ' + str(done.returncode))

    def health(stage):
        files = []
        for name, cmd, timeout in [
            (stage + '-health.log', [str(ROOT / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH], 180),
            (stage + '-collective.log', [str(ROOT / 'bin/xpu-collective-health'), '--img', HEALTH, '--p2p', '0', '--timeout', '180'], 210)]:
            run(cmd, name, timeout)
            files.append(dict(path=str(out / name), sha256=hashlib.sha256((out / name).read_bytes()).hexdigest(), command=cmd))
        receipt = out / (stage + '-health.json')
        receipt.write_text(json.dumps(dict(passed=True, cards=[0, 1], files=files, finished_epoch=time.time()), indent=2) + '\n')
        result[stage + '_health_passed'] = True
        return receipt

    def faults():
        done = subprocess.run(['journalctl', '-k', '--since', '@' + str(started), '--no-pager'], capture_output=True, timeout=20)
        (out / 'qualification-kernel-journal.log').write_bytes(done.stdout + done.stderr)
        if done.returncode or re.search(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', done.stdout.decode(errors='replace'), re.I):
            raise RuntimeError('GPU fault signature or unavailable journal')

    try:
        pre = health('pre')
        with (out / 'launch-supervisor.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(ctrl), 'launch', '--prepared', str(out), '--pre-health', str(pre),
                                      '--ready-deadline', '600', '--max-runtime', '1000'], stdout=log, stderr=subprocess.STDOUT, pass_fds=(8, 9))
            deadline = time.monotonic() + 620
            while True:
                faults()
                if stopped or time.monotonic() > deadline:
                    raise RuntimeError('Readiness deadline or interruption')
                launch = out / 'launch.json'
                if launch.exists() and json.loads(launch.read_text()).get('ready'):
                    break
                if child.poll() is not None:
                    raise RuntimeError('Launch supervisor exited ' + str(child.returncode))
                time.sleep(2)
            run([sys.executable, str(ctrl), 'screen', '--prepared', str(out)], 'screen-client.log', 600)
    except Exception as e:
        result['error'] = str(e)
    finally:
        if (out / 'launch.json').exists() and not (out / 'stop.json').exists():
            try:
                run([sys.executable, str(ctrl), 'stop', '--prepared', str(out)], 'stop-client.log', 120)
            except Exception as e:
                result['stop_error'] = str(e)
        if child is not None:
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                # Keep lease while the owned serving supervisor/container is live.
                child.terminate()
                child.wait()
                result['supervisor_error'] = 'Supervisor needed termination'
        if (out / 'launch.json').exists():
            name = json.loads((out / 'launch.json').read_text())['container']
            while True:
                try:
                    ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'], timeout=20)
                    if not ids.strip():
                        break
                    obj = json.loads(subprocess.check_output(['docker', 'inspect', name], timeout=20))[0]
                    assert obj['Config']['Labels'].get('b70.c1.prepared') == hashlib.sha256((out / 'prepared.json').read_bytes()).hexdigest()
                    result['cleanup_error'] = 'Owned container needed extra removal'
                    run(['docker', 'rm', '-f', name], 'forced-container-remove.log', 60)
                except Exception as e:
                    result['cleanup_error'] = str(e)
                    time.sleep(2)
        try:
            post = health('post')
            faults()
            if (out / 'screen.json').exists() and (out / 'stop.json').exists():
                run([sys.executable, str(ctrl), 'finalize', '--prepared', str(out), '--post-health', str(post)], 'finalize.log', 60)
                result['passed'] = not any(k.endswith('error') for k in result)
        except Exception as e:
            result['post_error'] = str(e)
        (out / 'parent-qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], receipt=str(out / 'parent-qualification.json'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
