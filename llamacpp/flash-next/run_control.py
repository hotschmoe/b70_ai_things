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



def reset_capture_evidence(capture_dir, mode, diagnostic_log):
    """Count only observed bounded rows; numerical reset correctness is a separate verdict."""
    records = [line for line in diagnostic_log.splitlines() if line.startswith('FNRESET ')]
    files = sorted(capture_dir.glob('*.f32'))
    manifest = [{'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in files]
    evidence = {'reset_diagnostic_records': len(records), 'reset_numeric_files': len(files)}
    if mode == 'off':
        if records or files:
            raise RuntimeError('Default-off reset observer unexpectedly produced evidence')
        evidence['reset_numeric_coverage'] = 'off; no observation expected'
    else:
        if not all(any(line.startswith(prefix) for line in records)
                   for prefix in ['FNRESET arm ', 'FNRESET ubatch ', 'FNRESET scale_meta ']):
            raise RuntimeError('Armed reset metadata path was unexercised')
        if mode == 'metadata':
            if files or any(line.startswith('FNRESET summary ') for line in records):
                raise RuntimeError('Metadata-only lane unexpectedly captured numerical rows')
            evidence['reset_numeric_coverage'] = 'metadata only; no numerical observation'
        elif mode == 'numeric':
            pre = {path.name[:-8]: path for path in files if path.name.endswith('.pre.f32')}
            post = {path.name[:-9]: path for path in files if path.name.endswith('.post.f32')}
            if (not pre or set(pre) != set(post) or len(files) != len(pre)*2
                    or len(pre) > 18 or sum(row['bytes'] for row in manifest) > 268435456
                    or 'complete_file=0' in diagnostic_log):
                raise RuntimeError('Bounded numeric reset capture missing, incomplete or unmatched')
            for label in pre:
                size = pre[label].stat().st_size
                if (not label.startswith('FNRESET_') or size != post[label].stat().st_size
                        or not 0 < size <= 16777216 or size % 4):
                    raise RuntimeError('Numeric reset row byte bounds or pair size mismatch')
                for phase in ['pre', 'post']:
                    summaries = [line for line in records
                                 if line.startswith('FNRESET summary label=' + label + ' phase=' + phase + ' ')]
                    if (len(summaries) != 1 or 'complete_file=1 ' not in summaries[0]
                            or re.search(r' count=(\d+) ', summaries[0]) is None
                            or int(re.search(r' count=(\d+) ', summaries[0]).group(1))*4 != size):
                        raise RuntimeError('Numeric reset raw row missing complete full-row summary evidence')
            evidence['reset_numeric_coverage'] = 'observed bounded rows only; later layers and state families require explicit coverage review'
        else:
            raise ValueError('Unknown reset diagnostic mode')
    return manifest, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Immutable locally resolved runtime image ID')
    parser.add_argument('--build', type=Path, help='Isolated reset diagnostic build; baseline remains default')
    parser.add_argument('--build-receipt', type=Path, help='Supplemental qualification receipt; defaults to build parent receipt.json')
    parser.add_argument('--reset-diagnostic', choices=['off', 'metadata', 'numeric'], help='Explicit isolated-build reset diagnostic lane; absent preserves baseline')
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
    parser.add_argument('--sycl-fusion', type=int, choices=[0, 1], default=1)
    parser.add_argument('--ctx-checkpoints', type=int, choices=[0], help='Diagnostic: disable checkpoint creation and its prompt split')
    parser.add_argument('--profile-suite', type=Path)
    parser.add_argument('--diagnostic-on-screen-failure', action='store_true', help='Preserve failed screen and continue only a labeled diagnostic capture')
    parser.add_argument('--primitive-receipt', required=True, type=Path)
    parser.add_argument('--leased', action='store_true')
    args = parser.parse_args()
    if args.build_receipt is not None and args.build is None:
        parser.error('--build-receipt requires --build')
    if args.cpu_profile and not args.capture_profile:
        parser.error('--cpu-profile requires --capture-profile')
    if args.reset_diagnostic is not None:
        if (args.build is None or args.build.resolve() == BUILD.resolve() or not args.capture_profile
                or args.placement != 'static' or args.warmup != 'on' or args.cpu_profile
                or args.ctx_checkpoints is not None or args.sycl_fusion != 1):
            parser.error('Reset diagnostics require isolated --build, --capture-profile, static placement, warmup on, default checkpoints/fusion and no CPU profiler')
        history_suite = REPO / 'llamacpp/flash-next/cache_history_suite.json'
        if args.profile_suite is not None and args.profile_suite.resolve() != history_suite:
            parser.error('Reset diagnostics require the frozen cache history suite')
        args.profile_suite = history_suite
    elif args.build is not None and args.build.resolve() != BUILD.resolve():
        parser.error('Isolated --build requires an explicit --reset-diagnostic mode')
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
    build = args.build.resolve() if args.build is not None else BUILD
    capture_dir = out.resolve() / 'reset-capture' if args.reset_diagnostic is not None else None
    arm_file = capture_dir / 'ARM' if capture_dir is not None else None
    if capture_dir is not None:
        capture_dir.mkdir(exist_ok=False)
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
        import run_primitives as primitive_validator
        build_identity = primitive_validator.validate_candidate_build(build, args.build_receipt)
        if args.reset_diagnostic is not None and build_identity is None:
            raise RuntimeError('Isolated reset diagnostic build identity required')
        receipt.update({'build': str(build), 'build_identity': build_identity,
                        'reset_diagnostic_mode': args.reset_diagnostic})
        primitives = json.loads(args.primitive_receipt.read_text())
        if not primitives.get('passed') or not primitives.get('coverage_complete') or primitives.get('image') != args.image or not primitives.get('post_health_passed'):
            raise RuntimeError('Passing same-runtime primitive receipt required')
        if primitives.get('binary_sha256') != sha(build / 'bin/test-backend-ops'):
            raise RuntimeError('Primitive binary identity changed')
        receipt['build_validator_sha256'] = sha(Path(primitive_validator.__file__))
        if build_identity is not None:
            if primitives.get('controller_sha256') != receipt['build_validator_sha256']:
                raise RuntimeError('Primitive validator/controller source changed')
            if primitives.get('build_identity') != build_identity or primitives.get('build') != str(build):
                raise RuntimeError('Passing primitive receipt from this exact isolated build required')
            libraries = {str(path): sha(path) for path in sorted((build / 'bin').glob('lib*.so*'))}
            if primitives.get('runtime_library_sha256') != libraries:
                raise RuntimeError('Primitive runtime library identity changed')
        if primitives.get('sycl_optimization', 1) != args.sycl_opt:
            raise RuntimeError('Primitive optimization mode differs from the candidate')
        if primitives.get('sycl_fusion', 1) != args.sycl_fusion:
            raise RuntimeError('Primitive fusion mode differs from the candidate')
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
                        'model_revision': lock['revision'], 'server_sha256': sha(build / 'bin/llama-server')})
        method = args.placement if args.placement == 'static' else 'cache' + str(args.moe_cache_mib)
        alias = lock['research_alias'] + '-llamacpp-layer2-' + method + '-fp16kv-mtp0-ctx8192'
        if args.warmup == 'on':
            alias += '-warmup1'
        if args.sycl_opt == 0:
            alias += '-opt0'
        if args.ctx_checkpoints is not None:
            alias += '-ctxcp' + str(args.ctx_checkpoints)
        if args.sycl_fusion == 0:
            alias += '-fusion0'
        if args.reset_diagnostic is not None:
            alias += '-resetdiag-' + args.reset_diagnostic
        receipt['sycl_fusion'] = args.sycl_fusion
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
                   '-v', '/dev/dri/by-path:/dev/dri/by-path:ro', '-v', str(build) + ':/build:ro',
                   '-v', str(model_dir) + ':/model:ro', '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:0,1',
                   '-e', 'ZE_AFFINITY_MASK=0,1', '-e', 'GGML_SYCL_ENABLE_GRAPH=0', '-e', 'ZES_ENABLE_SYSMAN=1',
                   '-e', 'SYCL_CACHE_PERSISTENT=0']
        command += ['-e', 'GGML_SYCL_ENABLE_OPT=' + str(args.sycl_opt),
                    '-e', 'GGML_SYCL_ENABLE_FUSION=' + str(args.sycl_fusion)]
        if args.reset_diagnostic is not None:
            reset_env = {'FLASHNEXT_RESET_DIAG': '0' if args.reset_diagnostic == 'off' else '1',
                         'FLASHNEXT_RESET_DIAG_NUMERIC': '1' if args.reset_diagnostic == 'numeric' else '0',
                         'FLASHNEXT_RESET_DIAG_ARM': '/reset-capture/ARM',
                         'FLASHNEXT_RESET_DIAG_DIR': '/reset-capture'}
            command += ['-v', str(capture_dir) + ':/reset-capture']
            for key, value in reset_env.items():
                command += ['-e', key + '=' + value]
            receipt.update({'reset_diagnostic_environment': reset_env, 'reset_capture_directory': str(capture_dir),
                            'reset_diagnostic_armed': False, 'reset_numeric_coverage': 'unexercised'})
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
            if arm_file is not None:
                if stopped or not inspect()['Running']:
                    raise RuntimeError('Cannot arm reset diagnostic after interruption or server exit')
                # New directory and exclusive create prevent stale startup/screen arming.
                with arm_file.open('x', encoding='ascii') as handle:
                    json.dump({'armed_epoch': time.time(), 'screen_passed': receipt['screen_passed'],
                               'mode': args.reset_diagnostic, 'research_alias': alias}, handle, ensure_ascii=True)
                    handle.write('\n')
                receipt['reset_diagnostic_armed'] = True
                receipt['reset_arm_sha256'] = sha(arm_file)
                receipt['reset_arm_epoch'] = time.time()
                save(out / 'runtime-receipt.json', receipt)
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
            if args.reset_diagnostic is not None:
                run(['docker', 'logs', name], 'reset-diagnostic.log', 30)
                diagnostic_log = (out / 'reset-diagnostic.log').read_text(errors='replace')
                manifest, evidence = reset_capture_evidence(capture_dir, args.reset_diagnostic, diagnostic_log)
                save(out / 'reset-capture-manifest.json', manifest)
                receipt.update(evidence)
                receipt['reset_capture_manifest_sha256'] = sha(out / 'reset-capture-manifest.json')
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
