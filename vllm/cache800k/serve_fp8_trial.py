#!/usr/bin/env python3
"""Serve the user-authorized calibrated FP8 trial; parent owns both leases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

IMAGE = 'sha256:521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad'
CAMPAIGN = Path('/mnt/vm_8tb/b70/results/cache800k_20260909')
MODEL_ID = 'qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276-mtp3-fp8-e4m3-calkv-cache800k'
PUBLIC_MODEL_ID = 'hotschmoe-dd'

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/mnt/vm_8tb/b70')
CURRENT = ROOT / 'run/hotschmoe-dd-fp8-trial-current'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['start', 'stop', 'status'], nargs='?', default='start')
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if args.action in ('stop', 'status'):
        result = CURRENT.resolve(strict=True)
        if not result.is_relative_to(ROOT / 'results/qwen38_int4_fp8kv_trial'):
            raise RuntimeError('invalid lifecycle pointer')
        if args.action == 'status':
            print(result)
            print('stopped' if (result / 'server/exit.rc').exists() else 'active')
            return
        (result / 'server/STOP').touch()
        deadline = time.monotonic() + 900
        while not (result / 'server/exit.rc').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('teardown did not finish')
            time.sleep(1)
        return
    scales = CAMPAIGN / 'int4-mtp3-kv-scales.json'
    artifact = json.loads(scales.read_text())
    evidence = CAMPAIGN / 'm1-mtp3-calibration'
    if (artifact['expected_layers'] != 17 or artifact['provenance']['calibration_manifest_sha256'] !=
            hashlib.sha256((evidence / 'manifest.json').read_bytes()).hexdigest()):
        raise RuntimeError('MTP calibration provenance mismatch')
    scale_sha = hashlib.sha256(scales.read_bytes()).hexdigest()
    if not (evidence / 'WORKLOADS_PASSED').exists() or (evidence / 'exit.rc').read_text().strip() != '0':
        raise RuntimeError('initial FP8 gate evidence missing')
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, 'start', '--leased'])
    key = ROOT / 'secrets/dd_api_key'
    if not key.is_file() or not key.stat().st_size:
        raise RuntimeError('API key file is missing or empty')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    result = ROOT / 'results/qwen38_int4_fp8kv_trial' / stamp
    result.mkdir(parents=True, exist_ok=False)
    import shutil
    shutil.copytree(CAMPAIGN / 'preservation/config', result / 'config')
    shutil.copy2(scales, result / 'scales.json')
    validation_source = result / 'full_feature_validate.py'
    shutil.copy2(REPO / 'vllm/cache800k/full_feature_validate.py', validation_source)
    prior_validation = os.environ.get('B70_PRIOR_VALIDATION')
    (result / 'trial.json').write_text(json.dumps(dict(
        model=MODEL_ID, served_model=PUBLIC_MODEL_ID, alias=PUBLIC_MODEL_ID, image=IMAGE, scales_sha256=scale_sha,
        mode='user-authorized real-workload trial', production_qualified=False,
        mtp=3, prefix=True, eager=False, ram_offload=False,
        prior_evidence=str(evidence),
        validation_source_sha256=hashlib.sha256(validation_source.read_bytes()).hexdigest()), indent=2) + '\n')
    CURRENT.parent.mkdir(parents=True, exist_ok=True)
    replacement = CURRENT.with_name(CURRENT.name + '.new')
    replacement.unlink(missing_ok=True)
    replacement.symlink_to(result)
    replacement.replace(CURRENT)
    command = [sys.executable, str(REPO / 'vllm/fp8/kv_campaign_server.py'),
               '--preservation', str(result / 'config'), '--out', str(result / 'server'),
               '--name', 'hotschmoe-dd', '--image', IMAGE, '--served-model', PUBLIC_MODEL_ID,
               '--served-alias', MODEL_ID, '--mtp', '3', '--memory-gib', '64',
               '--hook', 'load', '--kv-dtype', 'fp8_e4m3',
               '--scales', str(result / 'scales.json'),
               '--port', '18124', '--health-p2p-check', '--leased']
    server = None
    frontdoor = None
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        if (result / 'server').is_dir():
            (result / 'server/STOP').touch()

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    try:
        with (result / 'runner.log').open('w') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1500
        while not (result / 'server/READY').exists():
            if stopping or server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('daily server did not become ready')
            time.sleep(1)
        # The leased server executes this gate before the public frontdoor.
        job = result / 'server/jobs/01-full-feature.json'
        validation_extra = ['--prior-validation', prior_validation] if prior_validation else []
        job.write_text(json.dumps(dict(command=['env', 'B70_REPO=' + str(REPO), sys.executable,
            str(validation_source),
            '--server-root', str(result / 'server'), '--model', PUBLIC_MODEL_ID, *validation_extra], timeout=8100)) + '\n')
        deadline = time.monotonic() + 8200
        while not job.with_suffix('.done').exists():
            if stopping or server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('full-feature validation interrupted')
            time.sleep(1)
        if job.with_suffix('.done').read_text().strip() != '0':
            raise RuntimeError('full-feature validation failed; endpoint not exposed')
        env = dict(os.environ, FRONTDOOR_HOST='0.0.0.0', FRONTDOOR_PORT='18080',
                   FRONTDOOR_BACKEND_URL='http://127.0.0.1:18124',
                   FRONTDOOR_API_KEY_FILE=str(key))
        with (result / 'frontdoor.log').open('w') as log:
            frontdoor = subprocess.Popen([sys.executable, str(REPO / 'vllm/fp8/openai_key_frontdoor.py')],
                                          env=env, stdout=log, stderr=subprocess.STDOUT)
        while not stopping and server.poll() is None and frontdoor.poll() is None:
            time.sleep(1)
        if frontdoor.poll() is not None and not stopping and server.poll() is None:
            raise RuntimeError('API frontdoor exited while backend remained active')
    finally:
        stop()
        if frontdoor is not None and frontdoor.poll() is None:
            frontdoor.terminate()
            try:
                frontdoor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                frontdoor.kill()
                frontdoor.wait()
        if server is not None:
            # Keep the parent lease until the runner has completed teardown.
            rc = server.wait()
            (result / 'exit.rc').write_text(str(rc) + '\n')
    if rc:
        raise RuntimeError('daily server lifecycle failed; inspect ' + str(result))


if __name__ == '__main__':
    main()
