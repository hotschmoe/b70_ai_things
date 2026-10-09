#!/usr/bin/env python3
"""Run counted CPU-reference primitive comparisons on each leased B70."""
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

REPO = Path(__file__).resolve().parents[2]
BUILD = Path('/mnt/vm_8tb/b70/build/flashnext-llamacpp')
HEALTH_IMAGE = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
FAULT = re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', re.I)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(log, groups):
    clean = ANSI.sub('', log)
    summaries = [(int(a), int(b)) for a, b in re.findall(r'^\s+(\d+)/(\d+) tests passed$', clean, re.M)]
    rows = re.findall(r'^\s+(MUL_MAT_ID|MUL_MAT|GET_ROWS|GATED_DELTA_NET)\((.*?)\):[^\n]*\bOK\b', clean, re.M)
    counted = {g['name']: sum(op == g['op'] and re.fullmatch(g['params_regex'], params) is not None for op, params in rows) for g in groups}
    expected = sum(g['expected_executed'] for g in groups)
    passed = (summaries == [(expected, expected)] and all(counted[g['name']] == g['expected_executed'] for g in groups)
              and 'not supported' not in clean.lower() and re.search(r'^\s+Backend SYCL0: OK$', clean, re.M) is not None)
    return {'passed': passed, 'summaries': summaries, 'counts': counted, 'expected': expected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--groups', nargs='+', help='Diagnostic subset; cannot qualify a model launch')
    p.add_argument('--debugger', action='store_true', help='Capture CPU crash stack; diagnostic only')
    p.add_argument('--persistent-cache', type=int, choices=[0, 1], default=0)
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    assert re.fullmatch(r'sha256:[0-9a-f]{64}', args.image)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    plan_paths = [REPO / 'llamacpp/flash-next/primitive-plan.json', REPO / 'llamacpp/flash-next/primitive-extra-plan.json']
    plans = [json.loads(path.read_text()) for path in plan_paths]
    groups = [group for plan in plans for group in plan['groups']]
    all_group_names = [g['name'] for g in groups]
    if args.groups:
        if not set(args.groups) <= set(all_group_names):
            p.error('Unknown primitive group')
        groups = [g for g in groups if g['name'] in args.groups]
    pattern = '|'.join('(?:' + g['params_regex'] + ')' for g in groups)
    cache = Path('/mnt/vm_8tb/b70/cache/flashnext-neo26.22')
    cache.mkdir(parents=True, exist_ok=True)
    started = int(time.time())
    result = {'image': args.image, 'source_revision': plans[0]['source_revision'], 'binary_sha256': digest(BUILD / 'bin/test-backend-ops'),
              'controller_sha256': digest(Path(__file__)), 'groups_used': [g['name'] for g in groups],
              'coverage_complete': args.groups is None and not args.debugger,
              'debugger': args.debugger, 'persistent_device_code_cache': args.persistent_cache,
              'plan_sha256': [digest(path) for path in plan_paths], 'started_epoch': started, 'cards': [], 'passed': False, 'post_health_passed': False}
    active = None
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, stop)

    def save():
        (out / 'primitive-receipt.json').write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')

    def run(command, filename, timeout, check=True):
        with (out / filename).open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if check and completed.returncode:
            raise RuntimeError(filename + ' exit ' + str(completed.returncode))
        return completed.returncode

    def health(stage):
        run([str(REPO / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH_IMAGE, '--timeout', '60'], stage + '-health.log', 180)
        run([str(REPO / 'bin/xpu-collective-health'), '--img', HEALTH_IMAGE, '--p2p', '0', '--timeout', '180'], stage + '-collective.log', 210)

    def fault_audit():
        data = subprocess.run(['journalctl', '-k', '--since', '@' + str(started), '--no-pager'], capture_output=True, timeout=20)
        (out / 'kernel-journal.log').write_bytes(data.stdout + data.stderr)
        if data.returncode:
            raise RuntimeError('Kernel journal unavailable')
        if FAULT.search(data.stdout.decode(errors='replace')):
            raise RuntimeError('GPU fault signature; stop primitive run')

    def cleanup():
        nonlocal active
        if active is None:
            return
        try:
            run(['docker', 'logs', active], active + '.log', 30, False)
            run(['docker', 'stop', '-t', '30', active], active + '-stop.log', 50, False)
        except Exception as error:
            result['cleanup_error'] = str(error)
        while True:
            try:
                observation = subprocess.run(['docker', 'ps', '-aq', '--filter', 'name=^/' + active + '$'], capture_output=True, timeout=20)
                if observation.returncode:
                    raise RuntimeError('Cleanup cannot observe daemon')
                if not observation.stdout.strip():
                    active = None
                    return
                container = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]
                if container['Config']['Labels'].get('b70.flashnext.primitive-owner') != str(os.getpid()):
                    raise RuntimeError('Container ownership mismatch')
                if container['State']['Running']:
                    result['cleanup_error'] = 'Primitive process required forced removal'
                    run(['docker', 'rm', '-f', active], active + '-forced-remove.log', 30, False)
                else:
                    run(['docker', 'rm', active], active + '-remove.log', 30)
            except Exception as error:
                result['cleanup_error'] = str(error)
                save()
            time.sleep(2)

    save()
    try:
        health('pre')
        fault_audit()
        for card in [0, 1]:
            if stopped:
                raise RuntimeError('Primitive run interrupted before launch')
            fault_audit()
            active = 'flashnext-primitives-' + str(os.getpid()) + '-' + str(card)
            workload = ['/build/bin/test-backend-ops', 'test', '-b', 'SYCL0', '-o', 'MUL_MAT,MUL_MAT_ID,GET_ROWS,GATED_DELTA_NET', '-p', pattern, '-j', '1']
            if args.debugger:
                workload = ['/opt/intel/oneapi/debugger/2026.1/opt/debugger/bin/gdb-oneapi', '--batch',
                            '-ex', 'set pagination off', '-ex', 'set debuginfod enabled off', '-ex', 'handle SIGSEGV stop print nopass',
                            '-ex', 'run', '-ex', 'thread apply all bt 12', '-ex', 'info registers rip rdi rsi rsp',
                            '-ex', 'x/16gx $rsp', '-ex', 'info sharedlibrary', '--args', *workload]
            command = ['docker', 'run', '-d', '--name', active, '--label', 'b70.flashnext.primitive-owner=' + str(os.getpid()),
                       '--device', '/dev/dri', '--memory', '24g', '--memory-swap', '24g', '--ulimit', 'core=0',
                       '-v', '/dev/dri/by-path:/dev/dri/by-path:ro', '-v', str(BUILD) + ':/build:ro', '-v', str(cache) + ':/cache',
                       '-e', 'ZE_AFFINITY_MASK=' + str(card), '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu',
                       '-e', 'GGML_SYCL_ENABLE_GRAPH=0', '-e', 'SYCL_CACHE_PERSISTENT=' + str(args.persistent_cache), '-e', 'SYCL_CACHE_DIR=/cache/sycl',
                       '-e', 'XDG_CACHE_HOME=/cache']
            if args.debugger:
                command += ['--cap-add', 'SYS_PTRACE', '--security-opt', 'seccomp=unconfined', '-e', 'INTELGT_AUTO_ATTACH_DISABLE=1']
            command += [args.image, 'exec ' + ' '.join(__import__('shlex').quote(s) for s in workload)]
            (out / ('card' + str(card) + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
            run(command, 'card' + str(card) + '-launch.log', 60)
            deadline = time.monotonic() + 2400
            while True:
                if stopped:
                    raise RuntimeError('Primitive run interrupted')
                fault_audit()
                state = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]['State']
                if not state['Running']:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('Primitive deadline exceeded')
                time.sleep(5)
            run(['docker', 'logs', active], 'card' + str(card) + '.log', 30, False)
            analysis = analyze((out / ('card' + str(card) + '.log')).read_text(errors='replace'), groups)
            analysis.update({'card': card, 'exit_code': state['ExitCode'], 'oom_killed': state.get('OOMKilled')})
            result['cards'].append(analysis)
            save()
            cleanup()
            if 'cleanup_error' in result:
                raise RuntimeError('Primitive cleanup was not clean; no next-card launch')
            fault_audit()
            if state['ExitCode'] or state.get('OOMKilled') or not analysis['passed']:
                raise RuntimeError('Card ' + str(card) + ' primitive comparison failed or coverage incomplete')
        fault_audit()
    except Exception as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        cleanup()
        try:
            health('post')
            result['post_health_passed'] = True
            fault_audit()
        except Exception as error:
            result['post_health_error'] = {'type': type(error).__name__, 'message': str(error)}
        result['passed'] = len(result['cards']) == 2 and all(c['passed'] for c in result['cards']) and result['post_health_passed'] and not any(k in result for k in ['error', 'cleanup_error', 'post_health_error'])
        result['finished_epoch'] = int(time.time())
        save()
    print(json.dumps({'passed': result['passed'], 'receipt': str(out / 'primitive-receipt.json')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
