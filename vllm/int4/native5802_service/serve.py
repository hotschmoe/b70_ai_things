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
CURRENT = ROOT / 'run/hotschmoe-dd-native5802-current'
RESULTS = ROOT / 'results/qwen38_native5802_service'
IMAGE = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
NATIVE = '6717e3ec7fe3e7cdc26cd66b84aabdd076f211e5bed62b8711b7377fca8668d9'
MODEL_MANIFEST = 'f0f3c197466b8cc5206cf5cc005fc73e1bbcad28539aeb5f9dd378a2310d898d'
SCOPE = 'native5802 TP2 backend100K+200K; known numeric EOS limitation retained'
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
    require(inputs.get('schema') == 1 and inputs.get('scope') == SCOPE, 'wrong qualification scope')
    require(inputs.get('inputs_finalized') is True, 'new200K inputs not finalized')
    require(inputs.get('image') == IMAGE and inputs.get('native_sha256') == NATIVE, 'wrong native identity')
    for name, digest in inputs['sha256'].items():
        require(sha(name) == digest, 'candidate input changed: ' + name)
    receipt = read(inputs['native_receipt'])
    require(receipt.get('image') == IMAGE and receipt.get('native_sha256') == NATIVE and receipt.get('build_exit_rc') == 0, 'native build receipt mismatch')
    for key, context in [('plan100k', 100000), ('plan200k', 200000)]:
        path = Path(inputs[key])
        require(path.parent.name == ('native5802-backend100k-v2' if context == 100000 else 'native5802-backend200k'), 'old or wrong qualification plan')
        require(path.is_file(), 'new backend plan missing: ' + key)
        frozen = read(path.parent / 'frozen.json')
        require(str(path) in frozen, 'plan missing from frozen manifest')
        for name, digest in frozen.items():
            require(sha(name) == digest, 'nested input changed: ' + name)
        candidate_plan = verify_plan(read(path), context)
        names = [j['name'] for j in candidate_plan['jobs']]
        require(names == inputs['required_jobs'][key] and len(names) == len(set(names)), 'mandatory job scope changed')
        expected_names, trace_output = context_scope(context)
        require(names == expected_names, 'context mandatory workload scope mismatch')
        trace_job = candidate_plan['jobs'][-1]
        require(value(trace_job['command'], '--out') == str(Path(candidate_plan['out']) / trace_output), 'context trace output mismatch')
    negative = read(inputs['known_quality_negative'])
    require(negative.get('numeric_array_quality_passed') is False and negative.get('old_markers_unchanged') is True,
            'known quality negative must be retained')
    require(bool(negative.get('evidence')), 'quality negative evidence missing')
    for name, digest in negative['evidence'].items():
        require(sha(name) == digest, 'quality negative evidence changed')
    require(sha(inputs['model_manifest']) == MODEL_MANIFEST, 'wrong model manifest')
    tokenized = read(inputs['tokenizer200k_receipt'])
    require(tokenized.get('passed') is True and tokenized.get('image') == IMAGE and len(tokenized.get('counts', [])) == 8, 'current200K tokenizer receipt missing')
    require(len({r['id'] for r in tokenized['counts']}) == 8 and all(r['actual'] == r['expected'] and 180000 <= r['actual'] <= 190000 for r in tokenized['counts']), 'long prompt token counts mismatch')
    model_files = read(inputs['model_manifest'])
    require(all(model_files[name]['sha256'] == digest for name,digest in tokenized['files'].items()), 'tokenizer/model identity mismatch')
    return read(inputs['plan200k'])


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


def evidence(plan_path):
    plan = read(plan_path)
    root = Path(plan['out'])
    require((root / 'WORKLOADS_PASSED').is_file(), 'workloads incomplete: ' + str(root))
    require((root / 'exit.rc').read_text().strip() == '0', 'backend lifecycle failed')
    require(Path(plan_path).with_suffix('.lifecycle-rc').read_text().strip() == '0', 'parent lifecycle failed')
    manifest = read(root / 'manifest.json')
    require(manifest['image'] == IMAGE, 'evidence image mismatch')
    require(manifest['source_sha256']['kv_campaign_server.py'] == sha(REPO / 'vllm/fp8/kv_campaign_server.py'), 'executed server source mismatch')
    require(manifest['health_probe_sha256'] == sha(value(plan['server'], '--health-probe')), 'executed health source mismatch')
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
    context = int(value(cfg, '--max-model-len'))
    _, trace_output = context_scope(context)
    trace = read(root / trace_output)
    require(trace.get('passed') is True and not trace.get('issues') and bool(trace.get('profiles')), 'actual host profile not verified')
    for job in plan['jobs']:
        if job['name'].endswith('-quality') and 'gate.py' in job['command'][1]:
            gate = read(Path(job['command'][3]) / 'strict-quality-gate.json')
            require(gate.get('passed') is True, 'semantic quality gate failed')
            if job['command'][2] == 'agent':
                require(gate.get('checks') == 32, 'tool checks incomplete')
    if context == 200000:
        for mode in ['retrieval', 'reuse', 'guide', 'cancel', 'recovery']:
            summary = read(root / mode / 'summary.json')
            require(summary.get('passed') is True and summary.get('mode') == mode and summary.get('checks', 0) > 0, 'long context semantic gate failed: ' + mode)
        require(read(root / 'cancel/cancellation-drain.json').get('passed') is True, 'cancellation drain not verified')
    scale_receipts(root)
    require_absent(value(plan['server'], '--name'))
    ids = [m['id'] for m in read(root / 'models.json')['data']]
    require(ids[:2] == ['hotschmoe-dd', value(plan['server'], '--served-alias')], 'live first model identity mismatch')
    require(value(manifest['command'], '--served-model-name') == 'hotschmoe-dd', 'primary ID not first')
    paths = [root / f for f in ['manifest.json', 'models.json', 'arm-results.json', 'server.log', 'exit.rc',
             'pre-xpu-health.log', 'post-xpu-health.log', 'pre-xpu-collective-health.log', 'post-xpu-collective-health.log']]
    # Bind actual gate decisions and their backing output/SSE records, not only exit markers.
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ['cache', 'source', '__pycache__']]
        for name in names:
            path = Path(directory) / name
            if path.suffix in ['.json', '.jsonl', '.log', '.txt', '.done']:
                paths.append(path)
    paths += [Path(plan_path).with_suffix('.lifecycle-rc'), root / 'WORKLOADS_PASSED', root / 'arm-plan.json']
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


def required100k_jobs():
    expected = ['02-tiny24', '03-early-history', '03-early-history-quality']
    for kind in ['deterministic', 'sampled']:
        for n in [1,2,3]:
            expected += ['04-' + kind + '-round' + str(n), '04-' + kind + '-round' + str(n) + '-quality']
    expected += ['99-startup-host-trace-review']
    return expected


def context_scope(context):
    if context == 100000:
        return required100k_jobs(), 'startup-host-trace-review.json'
    require(context == 200000, 'unsupported context scope')
    return ['01-retrieval', '02-reuse', '03-guide', '04-cancel', '05-recovery', '06-startup-host-boundaries'], 'tp-host-review.json'


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


def audit100k(plan_path):
    path = Path(plan_path)
    require(path.parent.name == 'native5802-backend100k-v2', 'requires NEW100K-v2 plan')
    frozen_path = path.parent / 'frozen.json'
    frozen = read(frozen_path)
    require(str(path) in frozen, 'plan absent from frozen manifest')
    for name, digest in frozen.items():
        require(sha(name) == digest, 'frozen100K source changed: ' + name)
    plan = verify_plan(read(path), 100000)
    names = [j['name'] for j in plan['jobs']]
    expected = required100k_jobs()
    require(names == expected, '100K mandatory workload scope mismatch')
    native_path = REPO / 'vllm/int4/native_gdn_oracle/native5802_build/image-receipt.json'
    native = read(native_path)
    require(native.get('image') == IMAGE and native.get('native_sha256') == NATIVE, 'native receipt mismatch')
    model_manifest = ROOT / 'results/bang_isolation_20260910/sglang-calibrated-kv/current-verified-model-files.json'
    require(sha(model_manifest) == MODEL_MANIFEST, 'model manifest mismatch')
    verify_model({'model_root': str(REPO / 'models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212'), 'model_manifest': str(model_manifest)})
    negative_path = path.parent / 'known-quality-negative.json'
    negative = read(negative_path)
    require(negative.get('numeric_array_quality_passed') is False and negative.get('old_markers_unchanged') is True, 'quality negative missing')
    for name,digest in negative['evidence'].items():
        require(sha(name) == digest, 'quality negative evidence changed')
    hashes = evidence(path)
    run = Path(plan['out'])
    tiny = read(run / 'tiny-prefill.json')
    early = read(run / 'early-history/strict-quality-gate.json')
    require(tiny.get('passed') is True and tiny.get('request_count') == 24, 'tiny24 incomplete')
    require(early.get('passed') is True and early.get('requests') == 3, 'early3 incomplete')
    tool_checks = sum(read(run / (kind + '-round' + str(n)) / 'strict-quality-gate.json')['checks']
                      for kind in ['deterministic', 'sampled'] for n in [1,2,3])
    require(tool_checks == 192, 'strict tools incomplete')
    hashes.update(frozen)
    hashes.update(negative['evidence'])
    for file in [path, frozen_path, native_path, model_manifest, negative_path, Path(__file__)]:
        hashes[str(file)] = sha(file)
    return {'passed': True, 'backend_scope': 'native5802-backend100k',
            'root': str(path.parent), 'tool_checks': tool_checks, 'tiny_checks': 24, 'early_checks': 3,
            'strict_pre_post_health': True,
            'scope': 'Exact100K backend workloads; numeric EOS quality negative retained',
            'image': IMAGE, 'native_sha256': NATIVE, 'scale_sha256': SCALE,
            'model_manifest_sha256': MODEL_MANIFEST, 'lifecycle_passed': True,
            'known_numeric_array_quality_passed': False, 'evidence_hashes': hashes}


def qualify(inputs):
    verify_inputs(inputs)
    verify_model(inputs)
    hashes = {}
    for name in ['plan100k', 'plan200k']:
        hashes.update(evidence(inputs[name]))
    return {'backend_configuration_qualified': True, 'unrestricted_model_quality_qualified': False,
            'known_numeric_array_quality_passed': False, 'scope': SCOPE,
            'inputs_sha256': sha(INPUTS), 'evidence_sha256': hashes}


def validate_qualification(inputs, qualification):
    require(qualification.get('backend_configuration_qualified') is True and qualification.get('scope') == SCOPE, 'backend candidate not qualified')
    require(qualification.get('unrestricted_model_quality_qualified') is False and qualification.get('known_numeric_array_quality_passed') is False, 'quality limitation missing')
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


def startup_jobs(inputs, out, namespace):
    """Use only pinned clean100K fixture jobs; full evidence remains prerequisite."""
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
    p.add_argument('action', choices=['check', 'freeze', 'start', 'stop', 'status', 'ready', 'audit100k'])
    p.add_argument('--leased', action='store_true')
    p.add_argument('--pid', type=int)
    args = p.parse_args()
    if args.action == 'audit100k':
        path = ROOT / 'results/bang_recurrence_testing_20260910/native5802-backend100k-v2/plan.json'
        receipt = audit100k(path)
        with (path.parent / 'qualification-receipt.json').open('x') as handle:
            handle.write(json.dumps(receipt, indent=2) + '\n')
        print('BACKEND100K_RECEIPT_VALID')
        return
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
    if args.action == 'freeze':
        q = qualify(inputs)
        with QUAL.open('x') as f:
            f.write(json.dumps(q, indent=2) + '\n')
        return
    validate_qualification(inputs, read(QUAL))
    if args.action == 'check':
        print('BACKEND_CONFIGURATION_VALID_KNOWN_EOS_LIMITATION')
        return
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, 'start', '--leased'])
    verify_pair_lease()
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
        scale_receipts(result / 'server')
        inspect = json.loads(subprocess.check_output(['docker', 'inspect', 'hotschmoe-dd'], text=True, timeout=15))[0]
        require(inspect['Image'] == IMAGE, 'startup image mismatch')
        # Bounded restart check; full clean100K/200K evidence was required above.
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
