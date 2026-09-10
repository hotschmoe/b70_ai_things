#!/usr/bin/env python3
"""Bounded TP1 SGLang viability; one lease, one physical card, no pair reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[2]
IMAGE = 'intel/sglang-dev@sha256:1a13d3d2b0adc63422a643eb5c59524842577d99ca3935b4b8c2c432e739e127'


def command(args, name):
    cache = args.out / 'cache'
    model = REPO / 'models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212'
    cmd = ['docker', 'run', '--rm', '--name', name, '--memory', '64g',
           '--memory-swap', '64g', '--ulimit', 'core=0', '--device', '/dev/dri',
           '--ipc=host', '--security-opt', 'label=disable',
           '-p', f'127.0.0.1:{args.port}:8000',
           '-v', str(model) + ':/model:ro', '-v', str(cache) + ':/cache',
           '--entrypoint', 'python3']
    env = {'ZE_AFFINITY_MASK': str(args.card),
           'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu',
           'CCL_TOPO_P2P_ACCESS': '0', 'CCL_ATL_TRANSPORT': 'ofi',
           'CCL_ENABLE_SYCL_KERNELS': '1', 'CCL_ZE_IPC_EXCHANGE': 'pidfd',
           'SYCL_UR_USE_LEVEL_ZERO_V2': '0', 'OMP_NUM_THREADS': '1',
           'SGLANG_MAMBA_CONV_DTYPE': 'float16',
           'HF_HOME': '/cache/hf', 'XDG_CACHE_HOME': '/cache',
           'TRITON_CACHE_DIR': '/cache/triton',
           'TORCHINDUCTOR_CACHE_DIR': '/cache/inductor'}
    for key, value in env.items():
        cmd += ['-e', key + '=' + value]
    launch = cmd + [args.image, '-m', 'sglang.launch_server', '--model-path', '/model',
                  '--served-model-name', 'hotschmoe-dd', '--device', 'xpu',
                  '--dtype', 'float16', '--kv-cache-dtype', getattr(args, 'kv_cache_dtype', 'auto'),
                  '--quantization', 'gptq', '--attention-backend', args.attention_backend,
                  '--linear-attn-backend', 'triton', '--mamba-ssm-dtype', 'float32',
                  '--disable-cuda-graph', '--disable-overlap-schedule',
                  '--disable-radix-cache', '--disable-custom-all-reduce',
                  '--tp-size', '1', '--context-length', '8192',
                  '--chunked-prefill-size', '512', '--max-running-requests', '4',
                  '--mem-fraction-static', '0.90', '--enable-metrics',
                  '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder',
                  '--host', '0.0.0.0', '--port', '8000']
    kv_dtype = getattr(args, 'kv_cache_dtype', 'auto')
    scale_path = getattr(args, 'quantization_param_path', None)
    if kv_dtype == 'fp8_e4m3' and args.attention_backend != 'triton':
        raise ValueError('Reviewed FP8 KV arm requires explicit Triton attention')
    if scale_path:
        launch += ['--quantization-param-path', str(scale_path)]
    if getattr(args, 'skip_server_warmup', False):
        launch.append('--skip-server-warmup')
    if args.prefix_cache:
        launch.remove('--disable-radix-cache')
        strategy = getattr(args, 'mamba_cache_strategy', 'extra_buffer')
        launch += ['--mamba-radix-cache-strategy', strategy]
        if strategy == 'no_buffer':
            if args.attention_backend != 'triton':
                raise ValueError('Reviewed no_buffer arm requires Triton and page_size1')
            launch += ['--page-size', '1']
    decode_graph = getattr(args, 'decode_graph', False)
    mtp_steps = getattr(args, 'mtp_steps', 0)
    if (decode_graph or mtp_steps) and args.attention_backend != 'triton':
        raise ValueError('Reviewed graph/MTP arms require explicit Triton attention')
    if mtp_steps not in (0, 1, 3):
        raise ValueError('Reviewed MTP steps are 0, 1 or 3')
    if decode_graph:
        launch.remove('--disable-cuda-graph')
        launch += ['--cuda-graph-backend-decode', 'full',
                   '--cuda-graph-backend-prefill', 'disabled',
                   '--cuda-graph-bs-decode', '1', '2', '4']
    if mtp_steps:
        launch += ['--speculative-algorithm', 'NEXTN',
                   '--speculative-num-steps', str(mtp_steps),
                   '--speculative-eagle-topk', '1',
                   '--speculative-num-draft-tokens', str(mtp_steps + 1),
                   '--speculative-draft-model-path', '/model',
                   '--speculative-draft-attention-backend', 'triton']
    return launch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--card', type=int, choices=[0, 1], required=True)
    p.add_argument('--port', type=int, default=18127)
    p.add_argument('--image', default=IMAGE)
    p.add_argument('--attention-backend', choices=['intel_xpu', 'triton'], default='intel_xpu')
    p.add_argument('--kv-cache-dtype', choices=['auto', 'fp8_e4m3'], default='auto')
    p.add_argument('--quantization-param-path', help='Scale JSON path already available inside the selected candidate image')
    p.add_argument('--health-probe', type=Path, default=REPO / 'bin/xpu-health')
    p.add_argument('--cache-seed', type=Path)
    p.add_argument('--prefix-cache', action='store_true', help='Enable radix cache for a separate feature qualification arm')
    p.add_argument('--mamba-cache-strategy', choices=['extra_buffer', 'no_buffer'], default='extra_buffer', help='Separate current-main XPU cache arm; no_buffer requires page_size1')
    p.add_argument('--skip-server-warmup', action='store_true', help='Skip automatic image warmup for the bounded text-only diagnostic arm')
    p.add_argument('--decode-graph', action='store_true', help='Current-main Triton decode FULL graph only; prefill remains eager')
    p.add_argument('--mtp-steps', type=int, choices=[0, 1, 3], default=0, help='Separate greedy NEXTN feature arm; draft tokens are steps+1')
    p.add_argument('--job', type=Path, help='JSON {command: [...], timeout: seconds}')
    p.add_argument('--startup-timeout', type=int, default=1200)
    p.add_argument('--ready-timeout', type=int, default=900)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if (args.decode_graph or args.mtp_steps or args.kv_cache_dtype == 'fp8_e4m3') and args.attention_backend != 'triton':
        p.error('reviewed graph/MTP/FP8 arms require --attention-backend triton')
    args.out = args.out.resolve()
    args.health_probe = args.health_probe.resolve()
    cache_seed_identity = None
    if args.cache_seed:
        args.cache_seed = args.cache_seed.resolve()
        if args.out == args.cache_seed or args.out in args.cache_seed.parents or args.cache_seed in args.out.parents:
            p.error('cache seed and output must be separate trees')
        previous_manifest = args.cache_seed.parent / 'manifest.json'
        previous = json.loads(previous_manifest.read_text())
        if previous['args']['image'] != args.image:
            p.error('cache seed manifest image differs from selected image')
        cache_seed_identity = dict(path=str(args.cache_seed), image=args.image,
                                   source_manifest=str(previous_manifest),
                                   source_manifest_sha256=hashlib.sha256(previous_manifest.read_bytes()).hexdigest())
    name = 'b70-sglang-tp1-card' + str(args.card) + '-' + uuid.uuid4().hex[:10]
    cmd = command(args, name)
    if args.dry_run:
        print(json.dumps(dict(command=cmd, lease=['bin/gpu-run', '--card', str(args.card)]), indent=2))
        return 0
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'),
                 ['gpu-run', '--card', str(args.card), sys.executable,
                  __file__, *sys.argv[1:], '--leased'])
    lease = os.environ.get('B70_GPU_LOCK', '/mnt/vm_8tb/b70/gpu.lock') + '.' + str(args.card)
    if Path(os.readlink('/proc/self/fd/' + str(8 + args.card))).resolve() != Path(lease).resolve():
        raise RuntimeError('selected inherited lease descriptor does not match card')
    # Docker also binds exclusively; this early check avoids querying another
    # existing endpoint during startup while our own port bind is pending.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    args.out.mkdir(parents=True, exist_ok=False)
    if args.cache_seed:
        shutil.copytree(args.cache_seed, args.out / 'cache', symlinks=True)
    else:
        (args.out / 'cache').mkdir()
    (args.out / 'manifest.json').write_text(json.dumps(
        dict(args=vars(args), command=cmd, primary_alias='hotschmoe-dd',
             health_probe_sha256=hashlib.sha256(args.health_probe.read_bytes()).hexdigest(),
             cache_seed=cache_seed_identity,
             research_identity=('qwen3.8-27b-AutoRound-INT4-W4A16-g128-sglang-tp1-'
                 + ('fp16kv' if args.kv_cache_dtype == 'auto' else args.kv_cache_dtype + 'kv')
                 + '-mtp' + str(args.mtp_steps) + ('-decodefull' if args.decode_graph else '-eager')
                 + ('-prefixon' if args.prefix_cache else '-prefixoff') + '-ctx8192')),
        default=str, indent=2) + '\n')
    docker = shutil.which('docker')
    health_label = 'b70.tp1-health=' + name
    wrapper_dir = args.out / 'wrapper'; wrapper_dir.mkdir()
    wrapper = wrapper_dir / 'docker'
    wrapper.write_text('#!/usr/bin/env python3\nimport os, sys\n'
                       'args = sys.argv[1:]\n'
                       'if args and args[0] == "run": args[1:1] = ["--label", ' + repr(health_label) + ']\n'
                       'os.execv(' + repr(docker) + ', ["docker", *args])\n')
    wrapper.chmod(0o755)
    health_env = dict(os.environ, PATH=str(wrapper_dir) + ':' + os.environ['PATH'])
    def run(command, filename, timeout=180, env=None):
        with (args.out / filename).open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            try:
                return child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                return 124
            finally:
                # A launcher can exit before descendants. Retire its dedicated
                # group before checking container cleanup so none can start a
                # new owned container after that check.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    time.sleep(0.2)
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    def remove_health_containers():
        try:
            listed = subprocess.run([docker, 'ps', '-aq', '--filter', 'label=' + health_label],
                                    capture_output=True, text=True, timeout=20, check=True)
            ids = listed.stdout.split()
            if ids:
                run([docker, 'rm', '-f', *ids], 'health-cleanup.log', 30)
            left = subprocess.run([docker, 'ps', '-aq', '--filter', 'label=' + health_label],
                                  capture_output=True, text=True, timeout=20, check=True)
            return not left.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return False
    def retain_until_clean(check, marker):
        if not check():
            (args.out / marker).write_text('Retaining selected GPU lease until owned container cleanup is verified.\n')
            while not check():
                time.sleep(5)
    def health(stage):
        rc = run([str(args.health_probe), '--img', args.image,
                  '--card', str(args.card)], stage + '-health.log', env=health_env)
        retain_until_clean(remove_health_containers, stage + '-cleanup-required.txt')
        if rc:
            raise RuntimeError(stage + ' card health rc=' + str(rc))
    def stop(*_):
        (args.out / 'STOP').touch()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    server = None
    failed = False
    try:
        health('pre')
        with (args.out / 'server.log').open('w') as log:
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        deadline = time.monotonic() + args.startup_timeout
        while True:
            if server.poll() is not None or time.monotonic() > deadline or (args.out / 'STOP').exists():
                raise RuntimeError('startup interrupted, failed, or timed out')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=3) as r:
                    identity = json.load(r)
                assert any(m['id'] == 'hotschmoe-dd' for m in identity['data'])
                owned = subprocess.run([docker, 'inspect', name], capture_output=True,
                                       text=True, timeout=10, check=True)
                container = json.loads(owned.stdout)[0]
                assert container['State']['Running']
                bindings = container['NetworkSettings']['Ports']['8000/tcp']
                assert any(b['HostIp'] == '127.0.0.1' and b['HostPort'] == str(args.port) for b in bindings)
                (args.out / 'owned-container.json').write_text(json.dumps(container, indent=2) + '\n')
                (args.out / 'models.json').write_text(json.dumps(identity, indent=2) + '\n')
                break
            except (OSError, AssertionError, subprocess.SubprocessError, KeyError):
                time.sleep(2)
        (args.out / 'READY').touch()
        if args.job:
            job = json.loads(args.job.read_text())
            rc = run(job['command'], 'job.log', job.get('timeout', 600))
            (args.out / 'job.rc').write_text(str(rc) + '\n')
            if rc:
                raise RuntimeError('job rc=' + str(rc))
        else:
            deadline = time.monotonic() + args.ready_timeout
            while not (args.out / 'STOP').exists() and time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('server died while ready')
                time.sleep(1)
    except Exception as exc:
        failed = True
        (args.out / 'failure.txt').write_text(ascii(exc) + '\n')
    finally:
        if server is not None:
            if server.poll() is not None:
                failed = True
            stop_rc = run(['docker', 'stop', '-t', '30', name], 'stop.log', 45)
            if stop_rc:
                failed = True
                run(['docker', 'rm', '-f', name], 'remove.log', 30)
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                failed = True
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            def remove_server():
                try:
                    listed = subprocess.run([docker, 'ps', '-aq', '--filter', 'name=^/' + name + '$'],
                                            capture_output=True, text=True, timeout=20, check=True)
                    if not listed.stdout.strip():
                        return True
                    run([docker, 'rm', '-f', name], 'server-cleanup.log', 30)
                    return False
                except (OSError, subprocess.SubprocessError):
                    return False
            retain_until_clean(remove_server, 'server-cleanup-required.txt')
        try:
            health('post')
        except Exception as exc:
            failed = True
            (args.out / 'post-failure.txt').write_text(ascii(exc) + '\n')
        if failed:
            (args.out / 'recovery-required.txt').write_text(
                'No pair reset under a single-card lease. After release, acquire both leases '
                'and use the recovery ladder before further risky workloads.\n')
        (args.out / 'exit.rc').write_text(str(int(failed)) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
