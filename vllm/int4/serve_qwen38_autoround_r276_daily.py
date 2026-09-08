#!/usr/bin/env python3
"""Serve the qualified R276 daily profile; the parent holds both GPU leases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from prepare_replica import IMAGE, SOURCE_COMMIT

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/mnt/vm_8tb/b70')
QUALIFICATION = Path(__file__).resolve().with_name('r276_daily_qualification.json')
CURRENT = ROOT / 'run/hotschmoe-dd-int4-current'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['start', 'stop', 'status'], nargs='?', default='start')
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if args.action in ('stop', 'status'):
        result = CURRENT.resolve(strict=True)
        if not result.is_relative_to(ROOT / 'results/qwen38_int4_r276_daily'):
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
    qualification = json.loads(QUALIFICATION.read_text())
    if (qualification.get('qualified') is not True or qualification['image'] != IMAGE
            or qualification['source_commit'] != SOURCE_COMMIT):
        raise RuntimeError('R276 daily profile has not been qualified')
    profile = qualification['profile']
    if profile not in ('daily', 'daily-prefix-off'):
        raise RuntimeError('not a daily serving qualification')
    mtp = qualification.get('mtp', 4)
    if type(mtp) is not int or mtp not in (0, 4):
        raise RuntimeError('unsupported qualified speculative depth')
    prefill_batch = qualification.get('prefill_batch', 32768)
    if type(prefill_batch) is not int or prefill_batch not in (4096, 8192, 32768):
        raise RuntimeError('unsupported qualified prefill budget')
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, 'start', '--leased'])
    key = ROOT / 'secrets/dd_api_key'
    if not key.is_file() or not key.stat().st_size:
        raise RuntimeError('API key file is missing or empty')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    result = ROOT / 'results/qwen38_int4_r276_daily' / stamp
    result.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'steve-repro/qwen38-int4-neural-20260908'
    model = REPO / 'models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212'
    subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_replica.py')),
                    '--source', str(source), '--model', str(model), '--profile', profile,
                    '--prefill-batch', str(prefill_batch),
                    '--out', str(result / 'config')], check=True,
                   stdout=(result / 'prepare.log').open('w'))
    config_hash = hashlib.sha256((result / 'config/Config.json').read_bytes()).hexdigest()
    if config_hash != qualification['config_sha256']:
        raise RuntimeError('daily configuration differs from qualified configuration')
    (result / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    CURRENT.parent.mkdir(parents=True, exist_ok=True)
    replacement = CURRENT.with_name(CURRENT.name + '.new')
    replacement.unlink(missing_ok=True)
    replacement.symlink_to(result)
    replacement.replace(CURRENT)
    command = [sys.executable, str(REPO / 'vllm/fp8/kv_campaign_server.py'),
               '--preservation', str(result / 'config'), '--out', str(result / 'server'),
               '--name', 'hotschmoe-dd', '--image', IMAGE, '--served-model', 'hotschmoe-dd',
               '--mtp', str(mtp), '--memory-gib', str(qualification['memory_gib']),
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
