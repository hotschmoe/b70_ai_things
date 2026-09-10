#!/usr/bin/env python3
"""Explicit user-authorized day trial; full qualification is incomplete."""
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
CURRENT = ROOT / 'run/hotschmoe-dd-mrv1fix-day-trial-current'
RESULTS = ROOT / 'results/qwen38_mrv1fix_day_trial'
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


def readiness_owner(marker, main_pid):
    if marker.get('lease_owner_pid') != main_pid:
        return False
    try:
        status = (Path('/proc') / str(marker['owner_pid']) / 'status').read_text()
        parent = next(int(line.split()[1]) for line in status.splitlines() if line.startswith('PPid:'))
        return parent == main_pid
    except (OSError, KeyError, StopIteration, ValueError):
        return False


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


def verify_trial(inputs):
    require(inputs.get('user_authorized_day_trial') is True, 'day trial authorization absent')
    require(inputs.get('production_qualified') is False, 'trial must not claim qualification')
    for name, digest in inputs['sha256'].items():
        require(sha(name) == digest, 'trial source/config changed: ' + name)
    plan = read(inputs['plan'])
    cmd = plan['server']
    for k, v in {'--image': IMAGE, '--served-model': 'hotschmoe-dd', '--tensor-parallel-size': '2', '--p2p': '0', '--mtp': '3', '--kv-dtype': 'fp8_e4m3', '--hook': 'load'}.items():
        require(value(cmd, k) == v, 'wrong trial feature: ' + k)
    require(not any(x in cmd for x in ['--eager', '--prefix-off', '--health-p2p-check']), 'unsafe trial features')
    require(sha(value(cmd, '--scales')) == SCALE, 'wrong scales')
    require(value(cmd, '--served-alias') in (REPO / 'evals/configs/models.yaml').read_text(), 'alias missing')
    cfg = read(Path(value(cmd, '--preservation')) / 'Config.json')['Cmd']
    require(value(cfg, '--max-model-len') == str(inputs['max_model_len']), 'context mismatch')
    require(value(cfg, '--dtype') == 'float16' and value(cfg, '--quantization') == 'gptq', 'model format mismatch')
    require('--enable-prefix-caching' in cfg and json.loads(value(cfg, '--compilation-config'))['cudagraph_mode'] == 'FULL_DECODE_ONLY', 'cache/graph mismatch')
    return plan


def trial_prerequisite(inputs):
    root = Path(inputs['prerequisite'])
    results = read(root / 'arm-results.json')
    required = ['02-tiny24', '03-early-history', '03-early-history-quality']
    rounds = ['deterministic-round1', 'deterministic-round2', 'deterministic-round3', 'sampled-round1']
    for name in rounds:
        required += ['04-' + name, '04-' + name + '-quality']
        gate = read(root / name / 'strict-quality-gate.json')
        require(gate['passed'] is True and gate['checks'] == 32, 'partial prerequisite round failed')
    for name in required:
        require(results.get(name) == 0 and (root / 'jobs' / (name + '.done')).read_text().strip() == '0', 'partial prerequisite incomplete: ' + name)
    require((root / 'exit.rc').read_text().strip() == '0', 'partial prerequisite teardown incomplete')
    require((root.parent / 'plan.lifecycle-rc').read_text().strip() == '0', 'partial parent teardown incomplete')
    require(read(root / 'manifest.json')['image'] == IMAGE, 'partial prerequisite image mismatch')
    for stage in ['pre', 'post']:
        text = (root / (stage + '-xpu-health.log')).read_text()
        require('card 0: OK' in text and 'card 1: OK' in text and 'HEALTHY' in text, 'partial card health failed')
        text = (root / (stage + '-xpu-collective-health.log')).read_text()
        require('COLLECTIVE_HEALTH_OK world_size=2' in text and 'xpu-collective-health: HEALTHY' in text, 'partial pair health failed')
    require('force killing remaining process EngineCore' not in (root / 'server.log').read_text(), 'partial forced teardown')


def tiny_job(inputs, out):
    job = copy.deepcopy(inputs['tiny_job'])
    c = job['command']
    c[c.index('--base-url') + 1] = 'http://127.0.0.1:18124'
    c[c.index('--out') + 1] = str(out / 'tiny-prefill.json')
    return [job]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['check', 'start', 'stop', 'status', 'ready'])
    p.add_argument('--leased', action='store_true')
    p.add_argument('--pid', type=int)
    p.add_argument('--context', choices=['100k', '200k'], default='200k')
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
                if not readiness_owner(marker, args.pid) or not owns_frontdoor(marker['frontdoor_pid']):
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
    inputs = read(HERE / ('inputs-' + args.context + '.json'))
    plan = verify_trial(inputs)
    trial_prerequisite(inputs)
    if args.action == 'check':
        print(json.dumps({'user_authorized_day_trial': True, 'production_qualified': False, 'context': args.context, 'image': IMAGE}))
        return
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, 'start', '--context', args.context, '--leased'])
    key = ROOT / 'secrets/dd_api_key'
    require(key.is_file() and key.stat().st_size > 0, 'API key file missing or empty')
    result = RESULTS / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    result.mkdir(parents=True, exist_ok=False)
    (result / 'trial.json').write_text(json.dumps({'user_authorized_day_trial': True, 'production_qualified': False, 'full_qualification_incomplete': True, 'context': args.context, 'image': IMAGE, 'scale_sha256': SCALE, 'input_sha256': sha(HERE / ('inputs-' + args.context + '.json')), 'scope': 'User requested pause testing and serve patched vLLM for a day; research resumes next downtime.'}, indent=2) + '\n')
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
        # User-authorized trial startup checks, not full qualification.
        jobs = tiny_job(inputs, result / 'server')
        (result / 'startup-plan.json').write_text(json.dumps(jobs, indent=2) + '\n')
        for job in jobs:
            frozen = copy.deepcopy(job)
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
        ready_tmp = result / 'READY.json.tmp'
        ready_tmp.write_text(json.dumps({'owner_pid': os.getpid(), 'lease_owner_pid': os.getppid(), 'frontdoor_pid': front.pid, 'backend_pid': backend.pid}) + '\n')
        ready_tmp.replace(result / 'READY.json')
        while not stopping:
            if (result / 'server/STOP').exists():
                stopping = True
                break
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
