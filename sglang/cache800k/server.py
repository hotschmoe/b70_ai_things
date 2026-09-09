#!/usr/bin/env python3
"""Leased, bounded SGLang INT4 cache campaign server with queued probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78'


def trace_mounts(directory, image):
    """Only mount the reviewed overlay into its exact installed image."""
    if image != IMAGE:
        raise ValueError('trace requires the exact inspected image')
    directory = directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = {
        'vocab_parallel_embedding.py': '1afd4ee93b7173ed221e3ed88d6a296039889583641bec8ce6c1b7e97a75c726',
        'b70_embedding_trace.py': 'ae6047be2bc44b49c620702992ef224b045a23cdd488dddbe6e516e21c732357',
    }
    if manifest['files'] != expected:
        raise ValueError('trace manifest differs from reviewed overlay')
    site = '/opt/venv/lib/python3.12/site-packages/'
    targets = {'vocab_parallel_embedding.py': site + 'sglang/srt/layers/vocab_parallel_embedding.py',
               'b70_embedding_trace.py': site + 'b70_embedding_trace.py'}
    mounts = []
    for name, digest in expected.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('trace file hash mismatch: ' + name)
        mounts += ['-v', str(directory / name) + ':' + targets[name] + ':ro']
    return mounts, dict(directory=str(directory), files=expected, targets=targets,
                        device_completion_observed=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--served-model', required=True)
    p.add_argument('--image', default=IMAGE)
    p.add_argument('--attention', choices=['intel_xpu', 'triton'], default='intel_xpu')
    # SGLang uses auto to inherit the explicit float16 model dtype.
    p.add_argument('--kv-dtype', choices=['auto', 'fp8_e4m3'], default='auto')
    p.add_argument('--prefix', action='store_true')
    p.add_argument('--context', type=int, default=200000)
    p.add_argument('--memory-fraction', type=float, default=.90)
    p.add_argument('--port', type=int, default=18125)
    p.add_argument('--prefill-size', type=int, default=8192)
    p.add_argument('--embedding-trace', type=Path)
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if args.prefill_size <= 0:
        p.error('--prefill-size must be positive')
    mounts, trace_identity = trace_mounts(args.embedding_trace, args.image) if args.embedding_trace else ([], None)
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'jobs').mkdir()
    name = 'b70-sglang-cache800k'
    cache = args.out / 'cache'; cache.mkdir()
    model = REPO / 'models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212'
    command = ['docker', 'run', '--rm', '--name', name, '--memory', '64g',
               '--memory-swap', '64g', '--ulimit', 'core=0', '--device', '/dev/dri',
               '--ipc=host', '--cap-add', 'SYS_PTRACE', '--security-opt', 'label=disable',
               '-p', f'127.0.0.1:{args.port}:8000', '-v', str(model) + ':/model:ro',
               '-v', str(cache) + ':/cache', '--entrypoint', 'python3']
    command += mounts
    env = {'CCL_ATL_TRANSPORT': 'ofi', 'CCL_ENABLE_SYCL_KERNELS': '1',
           'CCL_TOPO_P2P_ACCESS': '0', 'CCL_ZE_IPC_EXCHANGE': 'pidfd',
           'CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK': '0',
           'FI_TCP_IFACE': 'eth0', 'CCL_KVS_IFACE': 'eth0',
           'ONEAPI_DEVICE_SELECTOR': 'level_zero:0,1', 'ZE_AFFINITY_MASK': '0,1',
           'SYCL_UR_USE_LEVEL_ZERO_V2': '0', 'OMP_NUM_THREADS': '1',
           # Conv state must match FP16 inputs; temporal state stays FP32.
           'SGLANG_MAMBA_CONV_DTYPE': 'float16',
           'HF_HOME': '/cache/hf', 'XDG_CACHE_HOME': '/cache',
           'TRITON_CACHE_DIR': '/cache/triton', 'TORCHINDUCTOR_CACHE_DIR': '/cache/inductor'}
    for key, value in env.items():
        command += ['-e', key + '=' + value]
    command += [args.image, '-m', 'sglang.launch_server', '--model-path', '/model',
                '--served-model-name', args.served_model, '--device', 'xpu',
                '--dtype', 'float16', '--kv-cache-dtype', args.kv_dtype,
                '--quantization', 'gptq', '--attention-backend', args.attention,
                '--linear-attn-backend', 'triton', '--mamba-ssm-dtype', 'float32',
                '--disable-cuda-graph', '--disable-overlap-schedule',
                '--skip-server-warmup', '--enable-metrics', '--disable-custom-all-reduce', '--tp-size', '2',
                '--chunked-prefill-size', str(args.prefill_size), '--context-length', str(args.context),
                '--max-running-requests', '4', '--mem-fraction-static', str(args.memory_fraction),
                '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder',
                '--host', '0.0.0.0', '--port', '8000']
    if not args.prefix:
        command.append('--disable-radix-cache')
    (args.out / 'manifest.json').write_text(json.dumps(dict(args=vars(args), command=command, trace=trace_identity), default=str, indent=2) + '\n')
    def run(cmd, file, timeout=300):
        with (args.out / file).open('w') as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    def health(stage):
        for tool, extra in [('xpu-health', []), ('xpu-collective-health', ['--p2p', '0', '--timeout', '180'])]:
            rc = run([str(REPO / 'bin' / tool), '--img', args.image, *extra], stage + '-' + tool + '.log')
            if rc:
                raise RuntimeError(stage + ' ' + tool + ' rc=' + str(rc))
    def stop(*_):
        (args.out / 'STOP').touch()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    server = None
    failed = False
    try:
        health('pre')
        with (args.out / 'server.log').open('w') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1200
        while True:
            if server.poll() is not None or time.monotonic() > deadline or (args.out / 'STOP').exists():
                raise RuntimeError('startup did not complete')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=3) as response:
                    identity = json.load(response)
                assert any(x['id'] == args.served_model for x in identity['data'])
                (args.out / 'models.json').write_text(json.dumps(identity, indent=2) + '\n')
                break
            except (OSError, AssertionError):
                time.sleep(2)
        (args.out / 'READY').touch()
        while True:
            if server.poll() is not None:
                raise RuntimeError('server exited while ready')
            if (args.out / 'STOP').exists():
                break
            for job in sorted((args.out / 'jobs').glob('*.json'))[:1]:
                spec = json.loads(job.read_text()); job.rename(job.with_suffix('.running'))
                try:
                    rc = run(spec['command'], job.stem + '.log', spec.get('timeout', 1800))
                except subprocess.TimeoutExpired:
                    rc = 124
                job.with_suffix('.done').write_text(str(rc) + '\n')
                if rc:
                    raise RuntimeError('failed job: ' + job.stem + ' rc=' + str(rc))
            time.sleep(1)
    except Exception as exc:
        failed = True
        (args.out / 'failure.txt').write_text(ascii(exc) + '\n')
    finally:
        if server is not None:
            # Detect an exit before our intentional stop, including the STOP race.
            if server.poll() is not None:
                failed = True
                failure = args.out / 'failure.txt'
                if not failure.exists():
                    failure.write_text('server exited before owned teardown\n')
            run(['docker', 'stop', '-t', '60', name], 'stop.log', 90)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                failed = True
                run(['docker', 'rm', '-f', name], 'force-remove.log', 30)
        if failed:
            run(['env', 'B70_XE_RESET_UNDER_LEASE=1', str(REPO / 'bin/xe-reset'),
                 '--method', 'rebind', '--no-probe'], 'recovery.log', 180)
        try:
            health('post')
        except Exception as exc:
            failed = True
            (args.out / 'post-failure.txt').write_text(ascii(exc) + '\n')
        (args.out / 'exit.rc').write_text(str(int(failed)) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
