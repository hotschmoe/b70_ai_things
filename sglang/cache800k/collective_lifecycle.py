#!/usr/bin/env python3
"""Bounded fresh-process diagnostic using run_arm's queued-job interface."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from server import IMAGE, REPO


def execute(command, destination, timeout):
    with destination.open('w') as log:
        try:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124


def finish(out, name, failed, run=execute):
    # docker-run client timeout does not imply container termination.
    rc = run(['docker', 'rm', '-f', name], out / 'remove.log', 45)
    remaining = out / 'remaining-container.log'
    check = run(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'], remaining, 15)
    if check or not remaining.exists() or remaining.read_text().strip():
        (out / 'teardown-failure.txt').write_text('container absence unverified\n')
        return 1  # Never reset underneath a potentially live container.
    if failed or rc:
        failed = True
        recovery = run(['env', 'B70_XE_RESET_UNDER_LEASE=1', str(REPO / 'bin/xe-reset'),
                        '--method', 'rebind', '--no-probe'], out / 'recovery.log', 180)
        if recovery:
            (out / 'recovery-failure.txt').write_text(str(recovery) + '\n')
    for tool, extra in [('xpu-health', []),
                        ('xpu-collective-health', ['--p2p', '0', '--timeout', '180'])]:
        result = run([str(REPO / 'bin' / tool), '--img', IMAGE, *extra],
                     out / ('post-' + tool + '.log'), 300)
        failed = bool(failed or result)
    return int(failed)


def docker_command(out, environment):
    name = 'b70-cache800k-collective-control'
    command = ['docker', 'run', '--name', name, '--memory=8g', '--memory-swap=8g',
               '--ulimit', 'core=0', '--device', '/dev/dri', '--ipc=host',
               '--cap-add', 'SYS_PTRACE', '--security-opt', 'label=disable',
               '-v', str(out / 'collective_control.py') + ':/control.py:ro',
               '-v', str(out / 'environment.json') + ':/environment.json:ro',
               '-v', str(out / 'cache') + ':/cache', '--entrypoint', '/opt/venv/bin/torchrun']
    for key, value in environment.items():
        if value is not None:
            command += ['-e', key + '=' + value]
    command += [IMAGE, '--standalone', '--nproc-per-node=2', '/control.py',
                '--environment-reference', '/environment.json']
    return name, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--leased', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).with_name('collective_control.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != args.source_sha256:
        raise RuntimeError('control source identity mismatch')
    environment = json.loads(args.reference.read_text())['env']
    if environment.get('CCL_TOPO_P2P_ACCESS') != '0':
        raise RuntimeError('P2P must be disabled')
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__,
                                          *sys.argv[1:], '--leased'])
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    (out / 'jobs').mkdir()
    (out / 'cache').mkdir()
    (out / 'collective_control.py').write_bytes(source)
    (out / 'environment.json').write_text(json.dumps({'env': environment}, indent=2) + '\n')
    name, command = docker_command(out, environment)
    (out / 'manifest.json').write_text(json.dumps(dict(image=IMAGE, command=command,
        source_sha256=args.source_sha256, reference=str(args.reference),
        environment=environment, pid=os.getpid(), timeout=240,
        scope='fresh synthetic subgroup control; no model or served identity'), indent=2) + '\n')
    def stop(*_):
        (out / 'STOP').touch()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    failed = False
    attempted = False
    try:
        for tool, extra in [('xpu-health', []),
                            ('xpu-collective-health', ['--p2p', '0', '--timeout', '180'])]:
            rc = execute([str(REPO / 'bin' / tool), '--img', IMAGE, *extra],
                         out / ('pre-' + tool + '.log'), 300)
            if rc:
                raise RuntimeError('pre-health failed: ' + tool)
        (out / 'READY').touch()
        deadline = time.monotonic() + 360
        while not (out / 'STOP').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('coordinator handoff timeout')
            for job in sorted((out / 'jobs').glob('*.json')):
                spec = json.loads(job.read_text())
                if attempted or spec != {'command': ['fresh-subgroup-control'], 'timeout': 240}:
                    raise RuntimeError('only one frozen diagnostic job is allowed')
                job.rename(job.with_suffix('.running'))
                attempted = True
                rc = execute(command, out / 'control.log', 240)
                job.with_suffix('.done').write_text(str(rc) + '\n')
                if rc:
                    raise RuntimeError('control failed rc=' + str(rc))
            time.sleep(1)
        if not attempted:
            raise RuntimeError('stopped before diagnostic')
    except Exception as exc:
        failed = True
        (out / 'failure.txt').write_text(ascii(exc) + '\n')
    finally:
        rc = finish(out, name, failed)
        (out / 'exit.rc').write_text(str(rc) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
