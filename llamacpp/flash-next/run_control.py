#!/usr/bin/env python3
"""Own one bounded dual-B70 screening run, its lease, evidence and cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
BUILD = Path('/mnt/vm_8tb/b70/build/flashnext-llamacpp')
HEALTH_IMAGE = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
FAULT = re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', re.I)
CONTROLLER_SOURCE = Path(__file__).read_bytes()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Immutable locally resolved runtime image ID')
    parser.add_argument('--output', required=True, type=Path, help='New evidence directory')
    parser.add_argument('--placement', choices=['static', 'cache'], default='static')
    parser.add_argument('--moe-cache-mib', type=int, default=32768)
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--warmup', choices=['on', 'off'], default='off')
    parser.add_argument('--log-verbosity', type=int, choices=[3, 4, 5], default=3)
    parser.add_argument('--capture-profile', action='store_true', help='Run the diagnostic request suite after screening')
    parser.add_argument('--cpu-profile', action='store_true', help='Separate instrumented run with software CPU sampling')
    parser.add_argument('--init', action='store_true', help='Use Docker init to supervise the server child')
    parser.add_argument('--sycl-opt', type=int, choices=[0, 1], default=1)
    parser.add_argument('--ctx-checkpoints', type=int, choices=[0], help='Diagnostic: disable checkpoint creation and its prompt split')
    parser.add_argument('--profile-suite', type=Path)
    parser.add_argument('--diagnostic-on-screen-failure', action='store_true', help='Preserve failed screen and continue only a labeled diagnostic capture')
    parser.add_argument('--primitive-receipt', required=True, type=Path)
    parser.add_argument('--leased', action='store_true')
    args = parser.parse_args()
    if args.cpu_profile and not args.capture_profile:
        parser.error('--cpu-profile requires --capture-profile')
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        if not os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card)):
            raise RuntimeError('Missing inherited pair lease')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', args.image):
        parser.error('--image must be an immutable SHA256 ID')
    if not 1024 <= args.port <= 65535 or not 0 < args.moe_cache_mib <= 49152:
        parser.error('Invalid port or cache budget')
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    (out / 'controller.py').write_bytes(CONTROLLER_SOURCE)
    started = int(time.time())
    name = 'flashnext-control-' + str(os.getpid()) + '-' + str(started)
    stopped = False
    client = None
    profiler = None
    receipt = {'scope': 'bounded screening, not quality authority or promotion', 'passed': False,
               'image': args.image, 'placement': args.placement, 'container': name,
               'started_epoch': started, 'normal_stop': False, 'post_health_passed': False}

    def run(argv, filename, timeout=180, check=True):
        with (out / filename).open('w') as log:
            p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if check and p.returncode:
            raise RuntimeError(filename + ' exit ' + str(p.returncode))
        return p.returncode

    def health(stage):
        run([str(REPO / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH_IMAGE, '--timeout', '60'], stage + '-xpu-health.log', 180)
        run([str(REPO / 'bin/xpu-collective-health'), '--img', HEALTH_IMAGE, '--p2p', '0', '--timeout', '180'], stage + '-collective-health.log', 210)

    def inspect():
        raw = subprocess.check_output(['docker', 'inspect', name], timeout=20)
        state = json.loads(raw)[0]['State']
        save(out / 'container-state.json', state)
        return state

    def faults():
        p = subprocess.run(['journalctl', '-k', '--since', '@' + str(started), '--no-pager'], capture_output=True, timeout=20)
        (out / 'kernel-journal.log').write_bytes(p.stdout + p.stderr)
        if p.returncode or b'No journal files were found' in p.stderr:
            raise RuntimeError('Kernel journal monitor unavailable')
        lines = [line for line in p.stdout.decode(errors='replace').splitlines() if FAULT.search(line)]
        if lines:
            save(out / 'faults.json', lines)
            raise RuntimeError('Kernel GPU fault signature observed; run halted')

    def interrupted(*unused):
        nonlocal stopped
        stopped = True

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, interrupted)
    exists = False
    try:
        primitives = json.loads(args.primitive_receipt.read_text())
        if not primitives.get('passed') or not primitives.get('coverage_complete') or primitives.get('image') != args.image or not primitives.get('post_health_passed'):
            raise RuntimeError('Passing same-runtime primitive receipt required')
        if primitives.get('binary_sha256') != sha(BUILD / 'bin/test-backend-ops'):
            raise RuntimeError('Primitive binary identity changed')
        if primitives.get('sycl_optimization', 1) != args.sycl_opt:
            raise RuntimeError('Primitive optimization mode differs from the candidate')
        if args.placement == 'cache':
            required = {'slotmap_n1', 'slotmap_n2', 'slotmap_n32'} | {
                'bank_' + kind + '_n' + str(n) for kind in ['q4_K', 'q5_K', 'q5_1', 'q8_0'] for n in [1, 2]}
            if not required <= set(primitives.get('groups_used', [])):
                raise RuntimeError('Cache slot-map and mutable-view prerequisites required')
            if any(row.get('completed_mutation_phases') != 40 for row in primitives.get('cards', [])):
                raise RuntimeError('Cache mutation phase coverage incomplete')
        receipt['primitive_receipt_sha256'] = sha(args.primitive_receipt)
        lock_path = REPO / 'strata/flash-next/model-lock.json'
        lock = json.loads(lock_path.read_text())
        model_dir = REPO / lock['destination']
        intake_path = Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/model-intake.json')
        intake = json.loads(intake_path.read_text())
        if intake.get('status') != 'verified' or intake['revision'] != lock['revision']:
            raise RuntimeError('Verified pinned model intake required')
        verified = {row['path']: row for row in intake['verified']}
        for entry in lock['files']:
            path = model_dir / entry['path']
            if path.is_symlink() or path.stat().st_size != entry['size'] or verified[entry['path']]['digest'] != entry.get('sha256', entry.get('git_blob')):
                raise RuntimeError('Model intake identity mismatch: ' + entry['path'])
        receipt.update({'model_lock_sha256': sha(lock_path), 'intake_sha256': sha(intake_path),
                        'model_revision': lock['revision'], 'server_sha256': sha(BUILD / 'bin/llama-server')})
        method = args.placement if args.placement == 'static' else 'cache' + str(args.moe_cache_mib)
        alias = lock['research_alias'] + '-llamacpp-layer2-' + method + '-fp16kv-mtp0-ctx8192'
        if args.warmup == 'on':
            alias += '-warmup1'
        if args.sycl_opt == 0:
            alias += '-opt0'
        if args.ctx_checkpoints is not None:
            alias += '-ctxcp' + str(args.ctx_checkpoints)
        receipt['context_checkpoints_override'] = args.ctx_checkpoints
        receipt['sycl_optimization'] = args.sycl_opt
        receipt['warmup'] = args.warmup
        receipt['research_alias'] = alias
        import yaml
        registry = yaml.safe_load((REPO / 'evals/configs/models.yaml').read_text())
        if alias not in [row.get('served_model_id') for row in registry['models']]:
            raise RuntimeError('Research alias missing from evaluation identity registry')
        receipt['controller_sha256'] = hashlib.sha256(CONTROLLER_SOURCE).hexdigest()
        with socket.socket() as port_check:
            port_check.bind(('127.0.0.1', args.port))
    except Exception as error:
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
        save(out / 'runtime-receipt.json', receipt)
        return 1
    try:
        health('pre')
        faults()
        def memory_sample():
            values = {line.split(':', 1)[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':', 1)[0] in ['MemAvailable', 'MemTotal']}
            sample = {'epoch': time.time(), 'host': values}
            if exists:
                p = subprocess.run(['docker', 'exec', name, 'cat', '/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.peak'], capture_output=True, timeout=15)
                if p.returncode == 0:
                    sample['container_current'], sample['container_peak'] = map(int, p.stdout.split())
            with (out / 'memory-samples.jsonl').open('a') as log:
                log.write(json.dumps(sample, ensure_ascii=True) + '\n')
            if values['MemAvailable'] < 12 * 1024**3 or sample.get('container_current', 0) > 104 * 1024**3:
                raise RuntimeError('Memory guard: stop before host or container limit')
            return values
        if memory_sample()['MemAvailable'] < 64 * 1024**3:
            raise RuntimeError('At least 64 GiB host available required for this initial placement')
        server_args = ['-m', '/model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
                       '--alias', 'hotschmoe-dd,' + alias, '--host', '0.0.0.0', '--port', '8080',
                       '--device', 'SYCL0,SYCL1', '--split-mode', 'layer', '--tensor-split', '48,52', '-ngl', '99',
                       '--load-mode', 'mmap', '--lazy-mode', 'auto', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                       '--spec-type', 'none', '--reasoning', 'off', '--reasoning-budget', '0',
                       '--cache-ram', '0', '--cache-reuse', '0', '--flash-attn', 'auto',
                       '-c', '8192', '-b', '512', '-ub', '256', '-np', '1', '-t', '16', '-lv', str(args.log_verbosity)]
        if args.warmup == 'off':
            server_args += ['--no-warmup']
        if args.ctx_checkpoints is not None:
            server_args += ['--ctx-checkpoints', str(args.ctx_checkpoints)]
        if args.placement == 'static':
            server_args += ['-ot', r'per_layer_token_embd=CPU,blk\.([0-7]|4[0-7])\.ffn_(up|gate|down)_exps\.weight=CPU']
        else:
            server_args += ['--cpu-moe', '--moe-cache-mib', str(args.moe_cache_mib), '-ot', 'per_layer_token_embd=CPU']
        command = ['docker', 'run', '-d', '--name', name, '--label', 'b70.flashnext.owner=' + str(os.getpid()),
                   '--memory', '110g', '--memory-swap', '110g', '--ulimit', 'core=0', '--device', '/dev/dri',
                   '--group-add', str(Path('/dev/dri/renderD128').stat().st_gid), '--ipc', 'host', '-p', '127.0.0.1:' + str(args.port) + ':8080',
                   '-v', '/dev/dri/by-path:/dev/dri/by-path:ro', '-v', str(BUILD) + ':/build:ro',
                   '-v', str(model_dir) + ':/model:ro', '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:0,1',
                   '-e', 'ZE_AFFINITY_MASK=0,1', '-e', 'GGML_SYCL_ENABLE_GRAPH=0', '-e', 'ZES_ENABLE_SYSMAN=1',
                   '-e', 'SYCL_CACHE_PERSISTENT=0']
        command += ['-e', 'GGML_SYCL_ENABLE_OPT=' + str(args.sycl_opt)]
        if args.init or args.cpu_profile:
            command += ['--init']
        if args.cpu_profile:
            command += ['--cap-add', 'SYS_PTRACE', '--security-opt', 'seccomp=unconfined', '-v', str(out.resolve()) + ':/results']
        command += [args.image, 'exec /build/bin/llama-server ' + ' '.join(__import__('shlex').quote(s) for s in server_args)]
        receipt['supervised_child'] = args.init or args.cpu_profile
        receipt['software_cpu_sampling'] = args.cpu_profile
        receipt['server_arguments'] = server_args
        receipt['command'] = command
        save(out / 'runtime-receipt.json', receipt)
        if stopped:
            raise RuntimeError('Interrupted before container launch')
        exists = True
        run(command, 'launch.log', 60)
        deadline = time.monotonic() + 1200
        ready = False
        while time.monotonic() < deadline and not stopped:
            faults()
            memory_sample()
            if not inspect()['Running']:
                raise RuntimeError('Candidate exited before readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:' + str(args.port) + '/health', timeout=3) as response:
                    if response.status == 200:
                        ready = True
                        break
            except OSError:
                pass
            time.sleep(5)
        if not ready:
            raise RuntimeError('Startup deadline or interruption')
        run(['docker', 'logs', name], 'startup.log', 30)
        endpoint = 'http://127.0.0.1:' + str(args.port)
        with urllib.request.urlopen(endpoint + '/v1/models', timeout=10) as response:
            identity = json.load(response)
        save(out / 'models.json', identity)
        if identity['data'][0]['id'] != 'hotschmoe-dd' or alias not in identity['data'][0].get('aliases', []):
            raise RuntimeError('Live model aliases mismatch')
        client_log = (out / 'client.log').open('w')
        client = subprocess.Popen([sys.executable, str(REPO / 'llamacpp/flash-next/check_endpoint.py'),
                                   '--endpoint', endpoint, '--output', str(out / 'screen'), '--model-lock', str(lock_path),
                                   '--runtime-receipt', str(out / 'runtime-receipt.json'), '--confirm-no-speculation', '--rounds', '2'],
                                  stdout=client_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 2400
        while client.poll() is None and time.monotonic() < deadline and not stopped:
            faults()
            memory_sample()
            if not inspect()['Running']:
                raise RuntimeError('Candidate exited during screen')
            run(['docker', 'stats', '--no-stream', '--format', '{{json .}}', name], 'last-stats.json', 20, False)
            time.sleep(5)
        if client.poll() is None:
            raise RuntimeError('Screen deadline or interruption')
        if client.returncode:
            if not args.diagnostic_on_screen_failure or not args.capture_profile:
                raise RuntimeError('Screen failed; preserve responses')
            receipt['error'] = {'type': 'ScreenFailure', 'message': 'Strict screen failed; diagnostic continuation only'}
        faults()
        screen = json.loads((out / 'screen/screening-result.json').read_text())
        receipt['screen_passed'] = screen['passed']
        authority_path = REPO / 'llamacpp/flash-next/screen-authority.json'
        authority = json.loads(authority_path.read_text())
        receipt['screen_authority_sha256'] = sha(authority_path)
        compared = {row['case']: row['output_sha256'] == authority['outputs'][row['case']] for row in screen['checks'] if row['round'] == 1}
        receipt['static_screen_authority_matches'] = compared
        receipt['static_screen_authority_all_equal'] = all(compared.values())
        save(out / 'runtime-receipt.json', receipt)
        if args.capture_profile:
            if args.cpu_profile:
                from cpu_profile import CpuProfile
                profiler = CpuProfile(name, out)
                receipt['profile_target_pid'] = profiler.start()
                deadline = time.monotonic() + 60
                while not profiler.ready() and time.monotonic() < deadline and not stopped:
                    faults()
                    memory_sample()
                    if not inspect()['Running']:
                        raise RuntimeError('Server exited during profiler attachment')
                    time.sleep(1)
                if not profiler.ready():
                    raise RuntimeError('Software profiler attachment deadline')
            profile_log = (out / 'profile-client.log').open('w')
            profile_command = [sys.executable, str(REPO / 'llamacpp/flash-next/capture_profile.py'),
                                       '--endpoint', endpoint, '--output', str(out / 'profile-requests'),
                                       '--runtime-receipt', str(out / 'runtime-receipt.json'),
                                       '--screen-receipt', str(out / 'screen/screening-result.json'),
                                       '--model-lock', str(lock_path)]
            if args.profile_suite is not None:
                profile_command += ['--suite', str(args.profile_suite)]
            client = subprocess.Popen(profile_command, stdout=profile_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 2400
            while client.poll() is None and time.monotonic() < deadline and not stopped:
                faults()
                memory_sample()
                if profiler is not None:
                    profiler.assert_active()
                if not inspect()['Running']:
                    raise RuntimeError('Candidate exited during diagnostic capture')
                time.sleep(5)
            if client.poll() is None:
                raise RuntimeError('Diagnostic capture deadline or interruption')
            if client.returncode:
                raise RuntimeError('Diagnostic capture incomplete')
            receipt['diagnostic_capture_complete'] = True
            if profiler is not None:
                profiler.stop()
                if not inspect()['Running']:
                    raise RuntimeError('Server exited during profiler detachment')
                receipt['software_cpu_profile_complete'] = True
            faults()
    except Exception as error:
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if profiler is not None and not profiler.stopped:
            try:
                profiler.stop()
            except Exception as error:
                receipt['profiler_cleanup_error'] = {'type': type(error).__name__, 'message': str(error)}
        if client is not None and client.poll() is None:
            client.terminate()
            try:
                client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                client.kill(); client.wait()
        if exists:
            try:
                run(['docker', 'logs', name], 'server.log', 30, False)
                run(['docker', 'stop', '-t', '90', name], 'stop.log', 120, False)
                # Cache counters are printed at destruction, after the stop.
                run(['docker', 'logs', name], 'server.log', 30, False)
            except Exception as error:
                receipt['cleanup_error'] = {'type': type(error).__name__, 'message': str(error)}
            # Daemon errors are not proof of absence. Retain the lease until
            # the daemon positively reports no container under our unique name.
            while True:
                try:
                    p = subprocess.run(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'], capture_output=True, timeout=20)
                    if p.returncode:
                        raise RuntimeError('Docker cleanup observation failed')
                    if not p.stdout.strip():
                        break
                    raw = json.loads(subprocess.check_output(['docker', 'inspect', name], timeout=20))[0]
                    if raw['Config']['Labels'].get('b70.flashnext.owner') != str(os.getpid()):
                        raise RuntimeError('Cleanup ownership mismatch; retaining lease')
                    state = raw['State']
                    if not state['Running']:
                        receipt['normal_stop'] = state['ExitCode'] == 0 and not state.get('OOMKilled') and 'cleanup_error' not in receipt
                        run(['docker', 'rm', name], 'remove.log', 30)
                    else:
                        receipt['cleanup_error'] = {'type': 'ForcedStop', 'message': 'Owned container remained running after stop'}
                        run(['docker', 'rm', '-f', name], 'forced-remove.log', 30, False)
                except Exception as error:
                    receipt['cleanup_error'] = {'type': type(error).__name__, 'message': str(error)}
                    save(out / 'runtime-receipt.json', receipt)
                time.sleep(2)
        try:
            health('post')
            receipt['post_health_passed'] = True
        except Exception as error:
            receipt['post_health_error'] = {'type': type(error).__name__, 'message': str(error)}
        try:
            faults()
        except Exception as error:
            receipt['final_fault_error'] = {'type': type(error).__name__, 'message': str(error)}
        receipt['finished_epoch'] = int(time.time())
        receipt['passed'] = bool(receipt.get('screen_passed') and receipt['normal_stop'] and receipt['post_health_passed'] and not any(k in receipt for k in ['error', 'cleanup_error', 'final_fault_error', 'profiler_cleanup_error']))
        save(out / 'runtime-receipt.json', receipt)
    print(json.dumps({'passed': receipt['passed'], 'result': str(out)}, ensure_ascii=True))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
