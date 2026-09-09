#!/usr/bin/env python3
"""Run one frozen R187 cache experiment and queued probes inside gpu-run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152'


class PreflightInfrastructureError(RuntimeError):
    """A probe could not run; this is not evidence of a hardware failure."""


class RequestedStop(Exception):
    """Normal stop during preflight or startup; still run owned cleanup."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preservation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', default='b70-kv-campaign')
    p.add_argument('--image', default=IMAGE)
    p.add_argument('--packaged-hooks', action='store_true')
    p.add_argument('--served-model')
    p.add_argument('--served-alias', action='append', default=[])
    p.add_argument('--health-p2p-check', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--offload-gib', type=int, default=0)
    p.add_argument('--kv-dtype', default='auto', choices=['auto', 'fp8_e4m3'])
    p.add_argument('--eager', action='store_true')
    p.add_argument('--prefix-off', action='store_true')
    p.add_argument('--mtp', type=int, default=3)
    p.add_argument('--memory-gib', type=int, default=64)
    p.add_argument('--memory-swap-gib', type=int)
    p.add_argument('--port', type=int, default=18125)
    p.add_argument('--hook', choices=['none', 'record', 'load'], default='none')
    p.add_argument('--scales', type=Path)
    p.add_argument('--scale-audit', action='store_true')
    p.add_argument('--trace-offload', action='store_true')
    p.add_argument('--offload-group-fix', action='store_true')
    p.add_argument('--cache-seed', type=Path)
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if args.scale_audit and args.hook != 'load':
        p.error('--scale-audit requires --hook load')
    if not args.image.startswith('sha256:') or len(args.image) != 71:
        p.error('--image must be an immutable local image ID')
    if args.packaged_hooks:
        if not args.offload_group_fix or not args.offload_gib or args.hook != 'none' or args.trace_offload:
            p.error('packaged hooks require only the offload group repair and a CPU tier')
        image_config = json.loads(subprocess.check_output(
            ['docker', 'image', 'inspect', args.image], text=True))[0]['Config']
        image_env = dict(v.split('=', 1) for v in image_config['Env'])
        hashes = {name: hashlib.sha256((REPO / 'vllm/fp8/kv_hooks' / name).read_bytes()).hexdigest()
                  for name in ('sitecustomize.py', 'kv_offload_group_fix.py')}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        if (image_env.get('B70_OFFLOAD_GROUP_FIX') != '1'
                or image_env.get('PYTHONPATH') != '/opt/b70-kv/hooks'
                or (image_config.get('Labels') or {}).get('b70.kv-offload-hooks-sha256') != fingerprint):
            p.error('packaged image does not match the tracked offload hooks')
    if not args.leased:
        import sys
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    args.out.mkdir(parents=True, exist_ok=False)
    started_epoch = int(time.time())
    if args.offload_gib and args.memory_gib < args.offload_gib + 24:
        raise ValueError('container needs CPU tier plus 24 GiB runtime/headroom')
    (args.out / 'jobs').mkdir()
    def run(cmd, filename, timeout=300):
        with (args.out / filename).open('w') as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
    def health(stage):
        checks = [('xpu-health', [], ''), ('xpu-collective-health', ['--p2p', '0', '--timeout', '180'], '')]
        if args.health_p2p_check and stage == 'pre':
            checks.append(('xpu-collective-health', ['--p2p', '1', '--timeout', '180'], '-p2p1'))
        for tool, extra, suffix in checks:
            command = [str(REPO / 'bin' / tool), '--img', args.image, *extra]
            if suffix == '-p2p1':
                # Explicitly requested, scoped recipe preflight, after P2P-off
                # health. Do not export the risk opt-in to unrelated workloads.
                command = ['env', 'I_KNOW_P2P_WEDGES=1', *command]
            rc = run(command, stage + '-' + tool + suffix + '.log')
            if rc:
                log = (args.out / (stage + '-' + tool + suffix + '.log')).read_text()
                # rc=2 can also mean a compiler/worker error after device work.
                # Only this known pre-execution guard proves no GPU attempt ran.
                refused_before_execution = (
                    rc == 2 and 'refusing P2P=1 without I_KNOW_P2P_WEDGES=1' in log
                    and '=== collective probe' not in log
                )
                error = PreflightInfrastructureError if stage == 'pre' and refused_before_execution else RuntimeError
                raise error(stage + ' ' + tool + suffix + ' failed rc=' + str(rc))
    cfg = json.loads((args.preservation / 'Config.json').read_text())
    mounts = json.loads((args.preservation / 'Mounts.json').read_text())
    cmd = cfg['Cmd'][:]
    def setarg(key, value):
        cmd[cmd.index(key) + 1] = str(value)
    model = 'qwen3.8-27b-FP8-official-W8A16-mtp%d-%skv-cpu%dg-kvcampaign' % (args.mtp, args.kv_dtype, args.offload_gib)
    if args.offload_group_fix:
        model += '-gdnfix'
    if args.served_model:
        model = args.served_model
    setarg('--served-model-name', model)
    i = cmd.index('--served-model-name') + 2
    cmd[i:i] = args.served_alias
    setarg('--kv-cache-dtype', args.kv_dtype)
    if args.mtp == 0:
        i = cmd.index('--speculative-config'); del cmd[i:i+2]
    else:
        setarg('--speculative-config', json.dumps({'method': 'qwen3_next_mtp', 'num_speculative_tokens': args.mtp}))
    if args.eager:
        setarg('--compilation-config', '{"mode":0,"cudagraph_mode":"NONE"}')
        cmd.append('--enforce-eager')
    if args.prefix_off:
        cmd = [v for v in cmd if v != '--enable-prefix-caching']
        cmd.append('--no-enable-prefix-caching')
    if args.offload_gib:
        cmd += ['--kv-transfer-config', json.dumps({'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both', 'kv_connector_extra_config': {'cpu_bytes_to_use': args.offload_gib * 2**30, 'blocks_per_chunk': 1}})]
    if args.profile:
        cmd += ['--profiler-config', json.dumps({'profiler': 'torch',
                'torch_profiler_dir': '/kv-campaign/profile', 'torch_profiler_record_shapes': True,
                'torch_profiler_with_stack': False, 'torch_profiler_with_memory': False})]
    env = dict(v.split('=', 1) for v in cfg['Env'])
    if args.eager:
        env['VLLM_XPU_ENABLE_XPU_GRAPH'] = '0'
    cache = args.out / 'cache'; cache.mkdir()
    if args.cache_seed:
        shutil.copytree(args.cache_seed, cache, dirs_exist_ok=True, symlinks=True)
    source = args.out / 'source'; source.mkdir()
    for path in (REPO / 'vllm/fp8').glob('*kv*.py'):
        shutil.copy2(path, source / path.name)
    shutil.copytree(REPO / 'vllm/fp8/kv_hooks', source / 'kv_hooks',
                    ignore=shutil.ignore_patterns('__pycache__'))
    source_hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source.rglob('*.py')}
    docker = ['docker', 'run', '--rm', '--name', args.name, '--ulimit', 'core=0', '--memory', f'{args.memory_gib}g', '--memory-swap', f'{args.memory_swap_gib or args.memory_gib}g', '--device', '/dev/dri:/dev/dri', '--group-add', 'render', '--cap-add', 'SYS_PTRACE', '--security-opt', 'label=disable', '--ipc=host', '--shm-size=8g', '-p', f'127.0.0.1:{args.port}:8000']
    for m in mounts:
        src = str(cache) if m['Destination'] == '/root/.cache/vllm' else m['Source']
        docker += ['-v', src + ':' + m['Destination'] + ('' if m['RW'] else ':ro')]
    docker += ['-v', str(args.out.resolve()) + ':/kv-campaign', '-v', str(source.resolve()) + ':/kv-source:ro']
    # Python -m model-inspection subprocesses prepend cwd independently of
    # PYTHONPATH. The image's source checkout must not shadow the wheel.
    docker += ['--workdir', '/tmp']
    use_entry = not args.packaged_hooks and (args.hook != 'none' or args.trace_offload or args.offload_group_fix)
    if args.packaged_hooks:
        for key in ('PYTHONPATH', 'B70_OFFLOAD_GROUP_FIX', 'B70_KV_MODE', 'B70_OFFLOAD_TRACE'):
            env.pop(key, None)
    if use_entry:
        # An empty PYTHONPATH component adds the image WORKDIR, whose source
        # checkout can shadow the installed, patched serving package.
        env['PYTHONPATH'] = ':'.join(['/kv-source/kv_hooks', *filter(None, env.get('PYTHONPATH', '').split(':'))])
        env['B70_KV_MODE'] = args.hook
        env['B70_KV_OUT'] = '/kv-campaign'
        if args.scale_audit:
            env['B70_KV_AUDIT'] = '1'
        if args.trace_offload:
            env['B70_OFFLOAD_TRACE'] = '1'
        if args.offload_group_fix:
            env['B70_OFFLOAD_GROUP_FIX'] = '1'
        if args.scales:
            docker += ['-v', str(args.scales.resolve()) + ':/kv-scales.json:ro']
            env['B70_KV_SCALES'] = '/kv-scales.json'
        docker += ['--entrypoint', '/opt/venv/bin/python']
    for k, v in env.items():
        docker += ['-e', k + '=' + v]
    docker += [args.image]
    if use_entry:
        docker += ['/kv-source/kv_campaign_entry.py']
    docker += cmd
    manifest = {'model': model, 'image': args.image, 'args': vars(args), 'command': docker,
                'source_sha256': source_hashes}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, default=str, indent=2) + '\n')
    server = None
    failed = False
    needs_recovery = False
    def stop_signal(*_):
        (args.out / 'STOP').touch()
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGHUP, stop_signal)
    try:
        health('pre')
        if (args.out / 'STOP').exists():
            raise RequestedStop()
        with (args.out / 'server.log').open('w') as log:
            server = subprocess.Popen(docker, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('server exited during startup')
                if (args.out / 'STOP').exists():
                    raise RequestedStop()
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=3) as r:
                        identity = json.load(r)
                    assert identity['data'][0]['id'] == model
                    (args.out / 'models.json').write_text(json.dumps(identity, indent=2) + '\n')
                    break
                except (OSError, AssertionError):
                    time.sleep(2)
            else:
                raise RuntimeError('startup timeout')
            (args.out / 'READY').touch()
            print('READY ' + str(args.out), flush=True)
            while not (args.out / 'STOP').exists():
                if server.poll() is not None:
                    raise RuntimeError('server exited while ready')
                for job in sorted((args.out / 'jobs').glob('*.json'))[:1]:
                    try:
                        raw = job.read_text()
                        job.rename(job.with_suffix('.running'))
                    except FileNotFoundError:
                        continue
                    spec = json.loads(raw)
                    try:
                        rc = run(spec['command'], job.stem + '.log', spec.get('timeout', 1800))
                    except subprocess.TimeoutExpired:
                        rc = 124
                    job.with_suffix('.done').write_text(str(rc) + '\n')
                    if rc:
                        print('JOB FAILED ' + job.stem + ' rc=' + str(rc), flush=True)
                time.sleep(1)
    except RequestedStop:
        pass
    except Exception as exc:
        failed = True
        needs_recovery = server is not None or not isinstance(exc, PreflightInfrastructureError)
        (args.out / 'failure.txt').write_text(ascii(exc) + '\n')
        print(ascii(exc), flush=True)
    finally:
        if server is not None:
            # A failed probe can race with STOP from its coordinator. Detect
            # an already-dead backend before intentional shutdown either way.
            if server.poll() is not None:
                failed = True
                needs_recovery = True
            run(['docker', 'exec', args.name, 'sh', '-c', 'cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.events'], 'memory-final.log', 20)
            run(['docker', 'stop', '-t', '60', args.name], 'stop.log', 90)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                failed = True
                needs_recovery = True
                run(['docker', 'rm', '-f', args.name], 'force-remove.log', 30)
        if needs_recovery:
            # Explicit post-health below uses the campaign image, not a retired default.
            run(['env', 'B70_XE_RESET_UNDER_LEASE=1', str(REPO / 'bin/xe-reset'), '--method', 'rebind', '--no-probe'], 'recovery.log', 180)
        try:
            health('post')
        except Exception as exc:
            failed = True
            (args.out / 'post-failure.txt').write_text(ascii(exc) + '\n')
        run(['journalctl', '-k', '--since', '@' + str(started_epoch), '--no-pager'], 'kernel-journal.log', 20)
        (args.out / 'exit.rc').write_text(str(int(failed)) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
