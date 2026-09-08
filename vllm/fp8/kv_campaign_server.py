#!/usr/bin/env python3
"""Run one frozen R187 cache experiment and queued probes inside gpu-run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preservation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', default='b70-kv-campaign')
    p.add_argument('--offload-gib', type=int, default=0)
    p.add_argument('--kv-dtype', default='auto', choices=['auto', 'fp8_e4m3'])
    p.add_argument('--eager', action='store_true')
    p.add_argument('--prefix-off', action='store_true')
    p.add_argument('--mtp', type=int, default=3)
    p.add_argument('--memory-gib', type=int, default=64)
    p.add_argument('--port', type=int, default=18125)
    p.add_argument('--hook', choices=['none', 'record', 'load'], default='none')
    p.add_argument('--scales', type=Path)
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if not args.leased:
        import sys
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    args.out.mkdir(parents=True, exist_ok=False)
    started_epoch = int(time.time())
    if args.memory_gib < args.offload_gib + 24:
        raise ValueError('container needs CPU tier plus 24 GiB runtime/headroom')
    (args.out / 'jobs').mkdir()
    def run(cmd, filename, timeout=300):
        with (args.out / filename).open('w') as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
    def health(stage):
        for tool, extra in [('xpu-health', []), ('xpu-collective-health', ['--p2p', '0', '--timeout', '180'])]:
            if run([str(REPO / 'bin' / tool), '--img', IMAGE, *extra], stage + '-' + tool + '.log'):
                raise RuntimeError(stage + ' ' + tool + ' failed')
    cfg = json.loads((args.preservation / 'Config.json').read_text())
    mounts = json.loads((args.preservation / 'Mounts.json').read_text())
    cmd = cfg['Cmd'][:]
    def setarg(key, value):
        cmd[cmd.index(key) + 1] = str(value)
    model = 'qwen3.8-27b-FP8-official-W8A16-mtp%d-%skv-cpu%dg-kvcampaign' % (args.mtp, args.kv_dtype, args.offload_gib)
    setarg('--served-model-name', model)
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
    if args.offload_gib:
        cmd += ['--kv-transfer-config', json.dumps({'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both', 'kv_connector_extra_config': {'cpu_bytes_to_use': args.offload_gib * 2**30, 'blocks_per_chunk': 1}})]
    env = dict(v.split('=', 1) for v in cfg['Env'])
    if args.eager:
        env['VLLM_XPU_ENABLE_XPU_GRAPH'] = '0'
    cache = args.out / 'cache'; cache.mkdir()
    docker = ['docker', 'run', '--rm', '--name', args.name, '--ulimit', 'core=0', '--memory', f'{args.memory_gib}g', '--memory-swap', f'{args.memory_gib}g', '--device', '/dev/dri:/dev/dri', '--group-add', 'render', '--cap-add', 'SYS_PTRACE', '--security-opt', 'label=disable', '--ipc=host', '--shm-size=8g', '-p', f'127.0.0.1:{args.port}:8000']
    for m in mounts:
        src = str(cache) if m['Destination'] == '/root/.cache/vllm' else m['Source']
        docker += ['-v', src + ':' + m['Destination'] + ('' if m['RW'] else ':ro')]
    docker += ['-v', str(args.out.resolve()) + ':/kv-campaign', '-v', str(REPO / 'vllm/fp8') + ':/kv-source:ro']
    if args.hook != 'none':
        env['PYTHONPATH'] = '/kv-source/kv_hooks:' + env.get('PYTHONPATH', '')
        env['B70_KV_MODE'] = args.hook
        env['B70_KV_OUT'] = '/kv-campaign'
        if args.scales:
            docker += ['-v', str(args.scales.resolve()) + ':/kv-scales.json:ro']
            env['B70_KV_SCALES'] = '/kv-scales.json'
        docker += ['--entrypoint', '/opt/venv/bin/python']
    for k, v in env.items():
        docker += ['-e', k + '=' + v]
    docker += [IMAGE]
    if args.hook != 'none':
        docker += ['/kv-source/kv_campaign_entry.py']
    docker += cmd
    manifest = {'model': model, 'image': IMAGE, 'args': vars(args), 'command': docker}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, default=str, indent=2) + '\n')
    server = None
    failed = False
    def stop_signal(*_):
        (args.out / 'STOP').touch()
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        health('pre')
        with (args.out / 'server.log').open('w') as log:
            server = subprocess.Popen(docker, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('server exited during startup')
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
    except Exception as exc:
        failed = True
        (args.out / 'failure.txt').write_text(ascii(exc) + '\n')
        print(ascii(exc), flush=True)
    finally:
        if server is not None:
            run(['docker', 'exec', args.name, 'sh', '-c', 'cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.events'], 'memory-final.log', 20)
            run(['docker', 'stop', '-t', '60', args.name], 'stop.log', 90)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                failed = True
                run(['docker', 'rm', '-f', args.name], 'force-remove.log', 30)
        if failed:
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
