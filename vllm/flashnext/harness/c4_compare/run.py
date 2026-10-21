#!/usr/bin/env python3
"""Finite comparison arms. Run under bin/cpu-run --quiet; GPU launchers lease."""
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

R = Path('/mnt/vm_8tb/github/b70_ai_things')
B = Path('/mnt/vm_8tb/b70/build/vllm-flashnext')
H = R / 'vllm/flashnext/harness'
HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/vm_8tb/b70/results/c4_compare_20261021')
ALIASES = {
    'strata': 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata',
    'flashnext': 'qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-c4-mtp1',
    'qwen27b': 'qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276phase-mrv1fix-native5802-tp2-mtp3-fp8kv-freshcal-backend-ctx200k',
}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='ascii')


def execute(cmd, log, timeout=400, check=True, env=None):
    print('EXEC ' + json.dumps(cmd), flush=True)
    with log.open('w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env)
    print(f'RETURN {log.name} {result.returncode}', flush=True)
    if check and result.returncode:
        raise RuntimeError(f'{log}: exit {result.returncode}')
    return result.returncode


def health(out, stage):
    execute([str(R / 'bin/gpu-run'), str(R / 'bin/xpu-health')], out / f'{stage}-health.log')
    execute([str(R / 'bin/gpu-run'), str(R / 'bin/xpu-collective-health'), '--p2p', '0', '--timeout', '180'],
            out / f'{stage}-collective.log')


def env_recipe(paths):
    env = {}
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if '=' in line:
                k, v = line.split('=', 1)
                env[k] = v.strip()
    return env


def flash_command(out, name, port):
    env = env_recipe([H / 'o4_recipe.env', H / 'f1_prefill_delta.env'])
    env.update(B70_MTP_GGUF='/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/mtp-q2_0.gguf',
               B70_GGUF_SO=str(B / 'o5/gguf-src/b70_gguf_xpu/_C.o5c.so'),
               B70_GDN_SPEC_DECODE='1', B70_KREG_Q8='1', B70_FA_KV_SPLITS='16',
               B70_MTP_DRAFT_VOCAB='/mnt/vm_8tb/b70/build/strata-ab/src/data/draft_vocab_en.bin',
               B70_EP_ROUTE='1', B70_DECODE_HC_Q8F='1',
               B70_GGUF='1', B70_EP='1', B70_QSA='0',
               B70_GGUF_PATH='/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf',
               B70_EP_PROFILE='/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin',
               B70_EP_Q0='138', B70_EP_SPLIT='profile', B70_PLE_NBYTES='28800138240',
               B70_EP_MEM_MIN_KB='8388608', B70_EP_MAX_T='8192', B70_XPU_CG_RECLAIM='500',
               B70_GGUF_REF='0', B70_EAGER_QK='0', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', VLLM_ENGINE_READY_TIMEOUT_S='1200',
               SYCL_CACHE_PERSISTENT='0', SYCL_CACHE_DIR=str(B / 'cache/sycl-o5'))
    env['PYTHONPATH'] = ':'.join(map(str, [B / 'o5/fa-overlay', B / 'c7b-fa-overlay', B / 'vllm',
        Path('/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py'),
        *[R / 'vllm/flashnext/plugins' / p for p in ['b70_gguf_xpu', 'b70_ep_xpu', 'b70_qsa_xpu', 'b70_o4_xpu']],
        H, HERE]))
    save(out / 'recipe.json', env)
    pins = {}
    for key in ['B70_GGUF_SO', 'B70_EP_SO', 'B70_PREFILL_WIDE_SO', 'B70_MTP_DRAFT_VOCAB']:
        path = Path(env[key]); pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(out / 'pins.json', pins)
    cmd = [str(H / 'xpu-run.sh'), '--card', 'both', '--memory', '100g', '--name', name, '--port', str(port)]
    for k, v in env.items():
        cmd += ['--env', k + '=' + v]
    cmd += ['--', '/opt/venv/bin/python', str(HERE / 'api.py'), 'serve',
            '/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext', '--host', '0.0.0.0', '--port', str(port),
            '--served-model-name', 'hotschmoe-dd', ALIASES['flashnext'],
            '--tensor-parallel-size', '1', '--dtype', 'bfloat16', '--max-model-len', '131072',
            '--max-num-batched-tokens', '12288', '--max-num-seqs', '4', '--gpu-memory-utilization', '0.98',
            '--language-model-only', '--load-format', 'b70_gguf', '--quantization', 'b70_gguf',
            '--trust-remote-code', '--attention-backend', 'FLASH_ATTN', '--no-enable-prefix-caching',
            '--mamba-cache-mode', 'align', '--shutdown-timeout', '30',
            '--compilation-config', json.dumps(dict(mode=0, cudagraph_mode='FULL_DECODE_ONLY', cudagraph_capture_sizes=[1, 2, 3, 4, 8])),
            '--speculative-config', json.dumps(dict(method='mtp', num_speculative_tokens=1, use_local_argmax_reduction=True))]
    return cmd


def main(arm, tag, fixture_path):
    out = ROOT / tag
    out.mkdir(exist_ok=False)
    os.environ['B70_AGENT'] = 'c4-compare-' + arm
    def interrupted(signum, frame):
        raise RuntimeError('Comparison interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    name, port = 'c4-compare-' + tag, 18149
    started = time.time()
    if arm == 'qwen27b':
        inputs = json.loads((R / 'vllm/int4/native5802_trial/inputs.json').read_text())
        cmd = json.loads(Path(inputs['plan200k']).read_text())['server'][:]
        for flag, value in [('--out', str(out / 'server')), ('--name', name)]:
            cmd[cmd.index(flag) + 1] = value
        cmd += ['--port', str(port)]
        # This retained launcher owns its pre/post card and compiled pair health.
    elif arm == 'strata':
        config = json.loads((R / 'strata/flash-next/night/configs/night-final-v3-c4.json').read_text())
        config['exe'] = '/mnt/vm_8tb/b70/build/strata-ab/releases/strata-night-20261016'
        config['log'] = str(out / 'engine.log')
        save(out / 'config.json', config)
        cmd = ['bash', str(R / 'strata/flash-next/night/nl_launch.sh'), name, str(out / 'config.json'), str(port)]
    else:
        cmd = flash_command(out, name, port)
    save(out / 'command.json', cmd)
    save(out / 'metadata.json', dict(arm=arm, tag=tag, started_epoch=started,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip(),
        kernel=os.uname().release, fixture=str(fixture_path), fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest()))
    if arm != 'qwen27b':
        health(out, 'pre')
    status = dict(arm=arm, benchmark_rc=None, server_rc=None, teardown=False, post_health=False)
    process = None
    try:
        with (out / 'launcher.log').open('w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Server exited before READY')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                    models = json.load(response)
                if ALIASES[arm] not in [x['id'] for x in models['data']]:
                    raise RuntimeError('Unexpected live model alias')
                if arm == 'strata':
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
                        if json.load(response).get('loaded') is not True:
                            time.sleep(2); continue
                save(out / 'models.json', models)
                break
            except OSError:
                time.sleep(2)
        else:
            raise RuntimeError('Server startup timeout')
        print('READY ' + tag, flush=True)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/server_info', timeout=10) as response:
                save(out / 'server-info.json', json.load(response))
        except OSError:
            pass
        status['benchmark_rc'] = execute([sys.executable, str(HERE / 'bench.py'), '--base', f'http://127.0.0.1:{port}',
            '--fixture', str(fixture_path), '--out', str(out / 'bench'), '--alias', ALIASES[arm]],
            out / 'bench.log', timeout=1200, check=False)
    except Exception as exc:
        status['error'] = repr(exc)
        print('ERROR ' + repr(exc), flush=True)
    finally:
        if process is not None:
            if arm == 'qwen27b':
                (out / 'server/STOP').touch()
            else:
                execute(['docker', 'stop', '-t', '45', name], out / 'stop.log', timeout=65, check=False)
            try:
                status['server_rc'] = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                status['error'] = 'Server did not close after normal stop'
                execute(['docker', 'rm', '-f', name], out / 'force-remove.log', check=False)
                process.terminate()
                status['server_rc'] = process.wait(timeout=30)
            status['teardown'] = status['server_rc'] == 0
        if arm != 'qwen27b':
            try:
                health(out, 'post')
                status['post_health'] = True
            except Exception as exc:
                status['health_error'] = repr(exc)
        else:
            status['post_health'] = (out / 'server/exit.rc').exists() and (out / 'server/exit.rc').read_text().strip() == '0'
        execute(['journalctl', '-k', '--since', '@' + str(int(started)), '--no-pager'], out / 'kernel.log', check=False)
        if arm == 'flashnext':
            pins = json.loads((out / 'pins.json').read_text())
            status['pins_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in pins.items())
            text = (out / 'launcher.log').read_text(errors='replace')
            status['engine_errors'] = [line for line in text.splitlines() if any(s in line for s in
                ['EngineDeadError', 'force killing remaining process EngineCore', 'UR_RESULT_ERROR_', 'EngineCore encountered an issue'])]
        save(out / 'status.json', status)
        print('CLOSED ' + json.dumps(status), flush=True)
    return 0 if status['benchmark_rc'] == 0 and status['teardown'] and status['post_health'] and not status.get('engine_errors') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=list(ALIASES))
    parser.add_argument('--tag', required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'fixture.json')
    args = parser.parse_args()
    raise SystemExit(main(args.arm, args.tag, args.fixture))
