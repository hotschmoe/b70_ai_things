#!/usr/bin/env python3
"""Explicit user-authorized monitored200K trial; NOT long200K qualification."""
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
CURRENT = ROOT / 'run/hotschmoe-dd-native5802-monitored-trial-current'
RESULTS = ROOT / 'results/qwen38_native5802_monitored_trial'
IMAGE = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
NATIVE = '6717e3ec7fe3e7cdc26cd66b84aabdd076f211e5bed62b8711b7377fca8668d9'
MODEL_MANIFEST = 'f0f3c197466b8cc5206cf5cc005fc73e1bbcad28539aeb5f9dd378a2310d898d'
SCOPE = 'User-authorized monitored200K trial after complete100K; long200K qualification incomplete'
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


def verify_plan(plan, context):
    s = plan['server']
    expected = {'--image': IMAGE, '--served-model': 'hotschmoe-dd',
                '--tensor-parallel-size': '2', '--p2p': '0', '--mtp': '3',
                '--kv-dtype': 'fp8_e4m3', '--hook': 'load'}
    for k, v in expected.items():
        require(s.count(k) == 1 and value(s, k) == v, 'wrong native service flag: ' + k)
    require(not any(x in s for x in ['--eager', '--prefix-off', '--health-p2p-check']), 'unsupported feature override')
    require(sha(value(s, '--scales')) == SCALE, 'wrong fresh calibration')
    mounts = read(Path(value(s, '--preservation')) / 'Mounts.json')
    require(all(m['Destination'] in ['/model', '/root/.cache/vllm'] for m in mounts), 'unreviewed image overlay mount')
    models = [m for m in mounts if m['Destination'] == '/model']
    require(len(models) == 1 and models[0]['RW'] is False and models[0]['Source'] == str(REPO / 'models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212'), 'model mount mismatch')
    cfg = read(Path(value(s, '--preservation')) / 'Config.json')
    c = cfg['Cmd']
    for flag, wanted in {'--max-model-len': str(context), '--max-num-seqs': '4',
                         '--max-num-batched-tokens': '32768', '--dtype': 'float16',
                         '--quantization': 'gptq'}.items():
        require(c.count(flag) == 1 and value(c, flag) == wanted, 'wrong preserved setting: ' + flag)
    require('--enable-prefix-caching' in c and '--no-enable-prefix-caching' not in c, 'prefix mismatch')
    require(json.loads(value(c, '--compilation-config'))['cudagraph_mode'] == 'FULL_DECODE_ONLY', 'graph mismatch')
    for entry in cfg.get('Env', []):
        if entry.startswith('B70_XPU_GDN_PREFIX_CONV_COPY='):
            require(entry == 'B70_XPU_GDN_PREFIX_CONV_COPY=0', 'legacy copy adapter enabled')
    require(os.environ.get('B70_XPU_GDN_PREFIX_CONV_COPY', '0') == '0', 'legacy adapter inherited')
    require(value(s, '--served-alias') in (REPO / 'evals/configs/models.yaml').read_text(), 'alias unregistered')
    return plan


def verify_inputs(inputs):
    require(inputs.get('schema') == 1 and inputs.get('scope') == SCOPE, 'wrong trial scope')
    require(inputs.get('user_authorized_monitored_trial') is True, 'explicit trial authorization absent')
    require(inputs.get('long200k_qualified') is False and inputs.get('unrestricted_model_quality_qualified') is False, 'trial must retain qualification limits')
    require(inputs.get('inputs_finalized') is True, 'trial waits for stopped200K evidence')
    for name,digest in inputs['sha256'].items():
        require(sha(name) == digest, 'trial input changed: ' + name)
    # Reuse the separately reviewed complete100K prerequisite consumer.
    import importlib.util
    consumer_path = REPO / 'vllm/int4/native5802_qualification/prerequisite.py'
    spec = importlib.util.spec_from_file_location('trial100k_prerequisite', consumer_path)
    consumer = importlib.util.module_from_spec(spec); spec.loader.exec_module(consumer)
    consumer.validate(Path(inputs['receipt100k']), inputs['receipt100k_sha256'])
    verify_pause_recovery(inputs)
    required = required_dependencies(inputs)
    require(required <= set(inputs['sha256']), 'trial dependency coverage incomplete')
    for key in ['plan100k', 'plan200k']:
        for name, digest in read(Path(inputs[key]).parent / 'frozen.json').items():
            require(inputs['sha256'].get(name) == digest, 'frozen plan input mismatch: ' + name)
    native = read(inputs['native_receipt'])
    require(native.get('image') == IMAGE and native.get('native_sha256') == NATIVE, 'native ledger mismatch')
    plan = verify_plan(read(inputs['plan200k']), 200000)
    require(sha(inputs['model_manifest']) == MODEL_MANIFEST, 'wrong model manifest')
    verify_model(inputs)
    require(read(inputs['known_quality_negative']).get('numeric_array_quality_passed') is False, 'known EOS limitation missing')
    return plan


def required_dependencies(inputs):
    paths = {str(Path(__file__).resolve()), str(HERE / 'prepare_inputs.py'),
             str(REPO / 'vllm/int4/native5802_qualification/prerequisite.py'),
             str(REPO / 'vllm/fp8/openai_key_frontdoor.py')}
    for key in ['plan100k', 'plan200k', 'receipt100k', 'stopped200k_receipt',
                'recovery_receipt', 'native_receipt', 'model_manifest', 'known_quality_negative']:
        paths.add(inputs[key])
    for key in ['plan100k', 'plan200k']:
        p = Path(inputs[key]); frozen = p.parent / 'frozen.json'
        paths.add(str(frozen)); paths.update(read(frozen))
        server = read(p)['server']; config = Path(value(server, '--preservation'))
        paths.update(str(config / x) for x in ['Config.json', 'Mounts.json'])
        paths.add(value(server, '--scales'))
    for key in ['stopped200k_receipt', 'recovery_receipt']:
        paths.update(read(inputs[key])['evidence_sha256'])
    run = Path(read(inputs['plan200k'])['out'])
    paths.update(str(run / n) for n in ['manifest.json', 'models.json', 'READY',
                 'pre-xpu-health.log', 'pre-xpu-collective-health.log'])
    paths.update(str(p) for p in run.glob('kv-load-*.json'))
    return paths


def verify_pause_recovery(inputs):
    pause = read(inputs['stopped200k_receipt'])
    for key, expected in {'user_requested_pause': True, 'long200k_qualified': False,
            'server_lifecycle_rc': 0, 'owned_container_absent': True,
            'normal_stop': False, 'forced_engine_stop': True,
            'requires_additional_recovery': True}.items():
        require(pause.get(key) == expected, 'unexpected paused200K outcome: ' + key)
    recovery = read(inputs['recovery_receipt'])
    for key, expected in {'passed': True, 'parent_exit_rc': 0, 'image': IMAGE,
            'recovery': 'rebind', 'strict_both_cards_passed': True,
            'compiled_pair_p2p0_passed': True}.items():
        require(recovery.get(key) == expected, 'recovery incomplete: ' + key)
    for receipt in [pause, recovery]:
        require(bool(receipt.get('evidence_sha256')), 'receipt backing evidence absent')
        for name, digest in receipt['evidence_sha256'].items():
            require(sha(name) == digest, 'receipt evidence changed: ' + name)
    run = Path(read(inputs['plan200k'])['out'])
    require(read(run / 'manifest.json')['image'] == IMAGE and (run / 'READY').exists(), 'paused startup identity absent')
    ids = [m['id'] for m in read(run / 'models.json')['data']]
    require(ids[:2] == ['hotschmoe-dd', value(read(inputs['plan200k'])['server'], '--served-alias')], 'paused served identity mismatch')
    scale_receipts(run)


def write_exit_receipt(result, backend_rc, clean_loop_exit):
    wrapper_rc = 0 if clean_loop_exit and backend_rc == 0 else 1
    (result / 'backend-exit.rc').write_text(str(backend_rc) + '\n')
    (result / 'exit.rc').write_text(str(wrapper_rc) + '\n')
    return wrapper_rc


def scale_receipts(root):
    records = [(v.get('rank'), name, v) for p in root.glob('kv-load-*.json') for name, v in read(p).items()]
    wanted_layers = {'language_model.model.layers.' + str(i) + '.self_attn.attn' for i in range(3,64,4)}
    # Extra MTP attention's exact loaded name is bound in the finalized inputs.
    wanted_layers.add('mtp.layers.0.self_attn.attn')
    require(len(records) == 34 and {(rank,name) for rank,name,v in records} == {(r,n) for r in (0,1) for n in wanted_layers},
            'calibration rank/layer coverage mismatch')
    require(all(v.get('artifact_sha256') == SCALE and v.get('query_quantized') is False and v.get('kv_dtype') == 'fp8_e4m3'
                for rank,name,v in records), 'calibration artifact/format mismatch')


def require_absent(name):
    result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=15)
    message = result.stderr.lower()
    require(result.returncode != 0 and any(term in message for term in ['no such object:', 'no such container:']), 'owned container removal not verified')




def verify_model(inputs):
    files = read(inputs['model_manifest'])
    root = Path(inputs['model_root'])
    require(set(p.name for p in root.glob('*.safetensors')) == set(n for n in files if n.endswith('.safetensors')), 'weight inventory changed')
    index = read(root / 'model.safetensors.index.json')
    require(set(index['weight_map'].values()) <= set(files), 'weight shards missing from manifest')
    for name, record in files.items():
        require(sha(root / name) == record['sha256'], 'model file changed: ' + name)






def verify_pair_lease():
    for card, fd in [(0, 8), (1, 9)]:
        expected = ROOT / ('gpu.lock.' + str(card))
        try:
            opened = os.fstat(fd)
            disk = expected.stat()
            info = (Path('/proc/self/fdinfo') / str(fd)).read_text()
            require((opened.st_dev, opened.st_ino) == (disk.st_dev, disk.st_ino), 'wrong inherited lease descriptor')
            require(any('FLOCK' in line and 'WRITE' in line for line in info.splitlines() if line.startswith('lock:')), 'descriptor does not hold flock lease')
        except OSError as exc:
            raise RuntimeError('both inherited gpu-run leases required') from exc





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


def startup_jobs(inputs, out, namespace):
    """Use only pinned clean100K fixture jobs; complete100K and recovered paused200K are prerequisites."""
    clean = read(inputs['plan100k'])
    names = ['02-tiny24', '04-deterministic-round1',
             '04-deterministic-round1-quality', '04-sampled-round1',
             '04-sampled-round1-quality']
    by_name = {j['name']: j for j in clean['jobs']}
    jobs = []
    for name in names:
        require(name in by_name, 'required startup fixture missing: ' + name)
        job = copy.deepcopy(by_name[name])
        cmd = [x.replace(clean['out'], str(out)).replace(
            'http://127.0.0.1:18125', 'http://127.0.0.1:18124')
            for x in job['command']]
        if name in ['04-deterministic-round1', '04-sampled-round1']:
            require(value(cmd, '--records') == '360' and value(cmd, '--turns') == '4'
                    and '--stream' in cmd and '--shared-cache' in cmd,
                    'startup fixture scope changed')
            expected = '0' if name == '04-deterministic-round1' else '0.7'
            require(value(cmd, '--temperature') == expected, 'startup temperature changed')
            cmd[cmd.index('--cache-namespace') + 1] = namespace + '-' + name
            if '--base' in cmd:
                cmd[cmd.index('--base') + 1] = 'http://127.0.0.1:18124'
            else:
                cmd += ['--base', 'http://127.0.0.1:18124']
        job['command'] = cmd
        jobs.append(job)
    return jobs


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['check', 'start', 'stop', 'status', 'ready'])
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
    inputs = read(INPUTS)
    plan = verify_inputs(inputs)
    if args.action == 'check':
        print('USER_AUTHORIZED_MONITORED200K_TRIAL_NOT_LONG200K_QUALIFIED')
        return
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, 'start', '--leased'])
    verify_pair_lease()
    key = ROOT / 'secrets/dd_api_key'
    require(key.is_file() and key.stat().st_size > 0, 'API key file missing or empty')
    result = RESULTS / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    result.mkdir(parents=True, exist_ok=False)
    (result / 'trial.json').write_text(json.dumps({'scope': SCOPE, 'user_authorized_monitored_trial': True, 'long200k_qualified': False, 'unrestricted_model_quality_qualified': False, 'inputs_sha256': sha(INPUTS), 'image': IMAGE, 'native_sha256': NATIVE, 'scale_sha256': SCALE}, indent=2) + '\n')
    CURRENT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CURRENT.with_name(CURRENT.name + '.new')
    tmp.symlink_to(result)
    tmp.replace(CURRENT)
    backend = front = None
    stopping = False
    rc = 1
    clean_loop_exit = False
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
        scale_receipts(result / 'server')
        inspect = json.loads(subprocess.check_output(['docker', 'inspect', 'hotschmoe-dd'], text=True, timeout=15))[0]
        require(inspect['Image'] == IMAGE, 'startup image mismatch')
        # Bounded restart check; complete100K and recovered paused200K evidence was required above.
        jobs = startup_jobs(inputs, result / 'server', result.name)
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
        require(ids[:2] == ['hotschmoe-dd', value(plan['server'], '--served-alias')], 'live first identity mismatch')
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
        clean_loop_exit = True
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
        rc = write_exit_receipt(result, rc, clean_loop_exit)
    require(rc == 0, 'backend teardown failed')


if __name__ == '__main__':
    main()
