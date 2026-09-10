#!/usr/bin/env python3
"""Qualification-gated candidate service. No installation or automatic promotion."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/mnt/vm_8tb/b70')
HERE = Path(__file__).resolve().parent
INPUTS = HERE / 'inputs.json'
QUAL = HERE / 'qualification.json'
CURRENT = ROOT / 'run/hotschmoe-dd-mrv1fix-current'
RESULTS = ROOT / 'results/qwen38_mrv1fix_service'
IMAGE = 'sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1'
SCALE = 'be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def value(command, flag):
    return command[command.index(flag) + 1]


def verify_inputs(inputs):
    for name, digest in inputs['sha256'].items():
        require(sha(name) == digest, 'candidate input changed: ' + name)
    for key in ['plan100k', 'plan200k']:
        root = Path(inputs[key]).parent
        nested = read(root / 'manifest.json')
        dependencies = {str(root / name): digest for name, digest in nested['files'].items()}
        dependencies.update(nested['external'])
        for name, digest in dependencies.items():
            require(sha(name) == digest, 'nested qualification input changed: ' + name)
    plan = read(inputs['plan200k'])
    s = plan['server']
    for k, v in {'--image': IMAGE, '--served-model': 'hotschmoe-dd',
                 '--tensor-parallel-size': '2', '--p2p': '0', '--mtp': '3',
                 '--kv-dtype': 'fp8_e4m3', '--hook': 'load'}.items():
        require(value(s, k) == v, 'wrong candidate flag: ' + k)
    require(not any(x in s for x in ['--eager', '--prefix-off', '--health-p2p-check']),
            'unsupported candidate feature change')
    require(sha(value(s, '--scales')) == SCALE, 'wrong fresh calibration')
    cfg = read(Path(value(s, '--preservation')) / 'Config.json')
    c = cfg['Cmd']
    require(value(c, '--max-model-len') == '200000', 'wrong context')
    require(value(c, '--dtype') == 'float16', 'wrong dtype')
    require(value(c, '--quantization') == 'gptq', 'wrong quantization')
    require('--enable-prefix-caching' in c, 'prefix missing')
    require(json.loads(value(c, '--compilation-config'))['cudagraph_mode'] == 'FULL_DECODE_ONLY', 'wrong graph')
    require(value(s, '--served-alias') in (REPO / 'evals/configs/models.yaml').read_text(), 'alias unregistered')
    return plan


def evidence(plan_path):
    plan = read(plan_path)
    root = Path(plan['out'])
    require((root / 'WORKLOADS_PASSED').is_file(), 'workloads incomplete: ' + str(root))
    require((root / 'exit.rc').read_text().strip() == '0', 'backend lifecycle failed')
    require(Path(plan_path).with_suffix('.lifecycle-rc').read_text().strip() == '0', 'parent lifecycle failed')
    manifest = read(root / 'manifest.json')
    require(manifest['image'] == IMAGE, 'evidence image mismatch')
    require(read(root / 'arm-plan.json') == plan, 'executed plan mismatch')
    cfg = read(Path(value(plan['server'], '--preservation')) / 'Config.json')['Cmd']
    for flag in ['--max-model-len', '--max-num-seqs', '--max-num-batched-tokens', '--dtype', '--quantization', '--compilation-config', '--shutdown-timeout']:
        require(value(manifest['command'], flag) == value(cfg, flag), 'executed config mismatch: ' + flag)
    a = manifest['args']
    require(a['mtp'] == 3 and a['p2p'] == 0 and a['tensor_parallel_size'] == 2 and not a['prefix_off'] and not a['eager'] and a['kv_dtype'] == 'fp8_e4m3', 'executed features mismatch')
    results = read(root / 'arm-results.json')
    for j in plan['jobs']:
        require(results.get(j['name']) == 0 and
                (root / 'jobs' / (j['name'] + '.done')).read_text().strip() == '0',
                'incomplete or failed job: ' + j['name'])
    for stage in ['pre', 'post']:
        health = (root / (stage + '-xpu-health.log')).read_text()
        require('card 0: OK' in health and 'card 1: OK' in health and 'HEALTHY' in health, 'card health missing')
        collective = (root / (stage + '-xpu-collective-health.log')).read_text()
        require('COLLECTIVE_HEALTH_OK world_size=2' in collective and 'xpu-collective-health: HEALTHY' in collective and 'FAIL' not in collective, 'collective health missing')
    log = (root / 'server.log').read_text()
    require('Graph capturing finished' in log, 'graph evidence absent')
    require('force killing remaining process EngineCore' not in log, 'forced EngineCore teardown')
    loaded = [v for p in root.glob('kv-load-*.json') for v in read(p).values()]
    require(len(loaded) == 34 and all(x['artifact_sha256'] == SCALE and not x['query_quantized'] for x in loaded), 'scale load coverage/hash mismatch')
    paths = [root / f for f in ['manifest.json', 'models.json', 'arm-results.json', 'server.log', 'exit.rc',
             'pre-xpu-health.log', 'post-xpu-health.log', 'pre-xpu-collective-health.log', 'post-xpu-collective-health.log']]
    # Bind actual gate decisions and their backing output/SSE records, not only exit markers.
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ['cache', 'source', '__pycache__']]
        for name in names:
            path = Path(directory) / name
            if path.suffix in ['.json', '.jsonl', '.log', '.txt', '.done']:
                paths.append(path)
    paths = sorted(set(paths))
    return {str(p): sha(p) for p in paths}


def verify_model(inputs):
    files = read(inputs['model_manifest'])
    root = Path(inputs['model_root'])
    require(set(p.name for p in root.glob('*.safetensors')) == set(n for n in files if n.endswith('.safetensors')), 'weight inventory changed')
    index = read(root / 'model.safetensors.index.json')
    require(set(index['weight_map'].values()) <= set(files), 'weight shards missing from manifest')
    for name, record in files.items():
        require(sha(root / name) == record['sha256'], 'model file changed: ' + name)


def qualify(inputs):
    verify_inputs(inputs)
    verify_model(inputs)
    hashes = {}
    for name in ['plan100k', 'plan200k']:
        hashes.update(evidence(inputs[name]))
    return {'qualified': True, 'inputs_sha256': sha(INPUTS), 'evidence_sha256': hashes}


def validate_qualification(inputs, qualification):
    require(qualification.get('qualified') is True, 'candidate not qualified')
    require(qualification.get('inputs_sha256') == sha(INPUTS), 'qualification input mismatch')
    verify_inputs(inputs)
    for name, digest in qualification['evidence_sha256'].items():
        require(sha(name) == digest, 'qualification evidence changed: ' + name)
    fresh = qualify(inputs)
    require(fresh['evidence_sha256'] == qualification['evidence_sha256'], 'qualification evidence incomplete')


def service_command(plan, out):
    cmd = plan['server'][:]
    for k, v in [('--out', str(out)), ('--name', 'hotschmoe-dd')]:
        cmd[cmd.index(k) + 1] = v
    if '--port' in cmd:
        cmd[cmd.index('--port') + 1] = '18124'
    else:
        cmd += ['--port', '18124']
    cmd += ['--leased']
    return cmd


def owns_frontdoor(pid):
    sockets = set()
    for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            sockets.add(os.readlink(fd))
        except FileNotFoundError:
            pass
    for row in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        fields = row.split()
        if fields[1].split(':')[1] == format(18080, '04X') and fields[3] == '0A' and 'socket:[' + fields[9] + ']' in sockets:
            return True
    return False


def stop_result(result):
    require(result.is_relative_to(RESULTS), 'invalid service pointer')
    (result / 'server/STOP').touch()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['check', 'freeze', 'start', 'stop', 'status', 'ready'])
    p.add_argument('--leased', action='store_true')
    p.add_argument('--pid', type=int)
    args = p.parse_args()
    if args.action == 'ready':
        import select
        require(args.pid is not None, 'ready requires main PID')
        fd = os.pidfd_open(args.pid)
        poll = select.poll(); poll.register(fd, select.POLLIN)
        deadline = time.monotonic() + 11500
        while time.monotonic() < deadline:
            require(not poll.poll(0), 'candidate exited before readiness')
            try:
                result = CURRENT.resolve(strict=True)
                require(result.is_relative_to(RESULTS), 'invalid readiness pointer')
                marker = read(result / 'READY.json')
                if marker['owner_pid'] != args.pid or not owns_frontdoor(marker['frontdoor_pid']):
                    raise OSError('candidate readiness not yet owned')
                with urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=3) as response:
                    if response.status == 200:
                        return
            except (OSError, FileNotFoundError):
                pass
            time.sleep(2)
        raise RuntimeError('candidate readiness timeout')
    if args.action in ['stop', 'status']:
        result = CURRENT.resolve(strict=True)
        require(result.is_relative_to(RESULTS), 'invalid pointer')
        if args.action == 'status':
            print(result)
            return
        stop_result(result)
        deadline = time.monotonic() + 850
        while not (result / 'exit.rc').exists():
            require(time.monotonic() < deadline, 'service teardown timeout')
            time.sleep(1)
        return
    inputs = read(INPUTS)
    plan = verify_inputs(inputs)
    if args.action == 'freeze':
        q = qualify(inputs)
        with QUAL.open('x') as f:
            f.write(json.dumps(q, indent=2) + '\n')
        return
    validate_qualification(inputs, read(QUAL))
    if args.action == 'check':
        print('QUALIFICATION_VALID')
        return
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, 'start', '--leased'])
    key = ROOT / 'secrets/dd_api_key'
    require(key.is_file() and key.stat().st_size > 0, 'API key file missing or empty')
    result = RESULTS / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    result.mkdir(parents=True, exist_ok=False)
    (result / 'qualification.json').write_text(json.dumps(read(QUAL), indent=2) + '\n')
    CURRENT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CURRENT.with_name(CURRENT.name + '.new')
    tmp.symlink_to(result)
    tmp.replace(CURRENT)
    backend = front = None
    stopping = False
    rc = 1
    def stop(*unused):
        nonlocal stopping
        stopping = True
        if (result / 'server').exists():
            stop_result(result)
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, stop)
    try:
        backend = subprocess.Popen(service_command(plan, result / 'server'), stdout=(result / 'runner.log').open('w'), stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1500
        while not (result / 'server/READY').exists():
            require(not stopping and backend.poll() is None and time.monotonic() < deadline, 'backend startup failed')
            time.sleep(1)
        # Verify all calibrated layers before running the startup workloads.
        loaded = [v for path in (result / 'server').glob('kv-load-*.json') for v in read(path).values()]
        require(len(loaded) == 34 and all(v['artifact_sha256'] == SCALE and not v['query_quantized'] for v in loaded), 'startup calibration mismatch')
        # Replay the exact qualified200K jobs at startup, before public access.
        for job in plan['jobs']:
            frozen = copy.deepcopy(job)
            frozen['command'] = [s.replace(plan['out'], str(result / 'server')).replace('http://127.0.0.1:18125', 'http://127.0.0.1:18124') for s in job['command']]
            if '--base' not in frozen['command'] and 'probe.py' in ' '.join(frozen['command']):
                frozen['command'] += ['--base', 'http://127.0.0.1:18124']
            path = result / 'server/jobs' / (job['name'] + '.json')
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(frozen) + '\n'); temp.replace(path)
            deadline = time.monotonic() + job['timeout'] + 60
            while not path.with_suffix('.done').exists():
                require(not stopping and backend.poll() is None and time.monotonic() < deadline, 'startup qualification failed')
                time.sleep(1)
            require(path.with_suffix('.done').read_text().strip() == '0', 'startup job failed')
        with urllib.request.urlopen('http://127.0.0.1:18124/v1/models', timeout=10) as response:
            ids = [m['id'] for m in json.load(response)['data']]
        require('hotschmoe-dd' in ids and value(plan['server'], '--served-alias') in ids, 'live identity mismatch')
        env = dict(os.environ, FRONTDOOR_HOST='0.0.0.0', FRONTDOOR_PORT='18080', FRONTDOOR_BACKEND_URL='http://127.0.0.1:18124', FRONTDOOR_API_KEY_FILE=str(key))
        env.pop('B70_PRIOR_VALIDATION', None)
        front = subprocess.Popen([sys.executable, str(REPO / 'vllm/fp8/openai_key_frontdoor.py')], env=env, stdout=(result / 'frontdoor.log').open('w'), stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 30
        while not owns_frontdoor(front.pid):
            require(not stopping and front.poll() is None and backend.poll() is None and time.monotonic() < deadline, 'frontdoor failed to bind owned socket')
            time.sleep(0.1)
        (result / 'READY.json').write_text(json.dumps({'owner_pid': os.getpid(), 'frontdoor_pid': front.pid, 'backend_pid': backend.pid}) + '\n')
        while not stopping:
            require(backend.poll() is None and front.poll() is None, 'service component exited')
            time.sleep(1)
    finally:
        stop()
        if front is not None and front.poll() is None:
            front.terminate()
            try:
                front.wait(timeout=15)
            except subprocess.TimeoutExpired:
                front.kill(); front.wait()
        if backend is not None:
            rc = backend.wait()
        (result / 'exit.rc').write_text(str(rc) + '\n')
    require(rc == 0, 'backend teardown failed')


if __name__ == '__main__':
    main()
