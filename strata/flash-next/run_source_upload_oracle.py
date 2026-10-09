#!/usr/bin/env python3
"""Own actual HC/PLE upload checks on both B70s and their complete lifecycle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
HEALTH = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
FAULT = re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', re.I)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle-receipt', type=Path, required=True)
    p.add_argument('--pack-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--oracle-plan', type=Path, default=REPO / 'strata/flash-next/native-source-upload-plan.json')
    p.add_argument('--leased', action='store_true')
    a = p.parse_args()
    if not a.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output.resolve()
    receipt = dict(started_epoch=int(time.time()), passed=False, cases=[], model_math_qualified=False)
    active = None
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True

    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, stop)

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def run(cmd, name, timeout=60):
        (out / (name + '.command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
        with (out / name).open('w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=True)

    def health(stage):
        run([str(REPO / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH], stage + '-health.log', 180)
        run([str(REPO / 'bin/xpu-collective-health'), '--img', HEALTH, '--p2p', '0', '--timeout', '180'], stage + '-collective.log', 210)
        receipt[stage + '_health_passed'] = True
        save()

    def faults():
        done = subprocess.run(['journalctl', '-k', '--since', '@' + str(receipt['started_epoch']), '--no-pager'], capture_output=True, timeout=20)
        (out / 'kernel-journal.log').write_bytes(done.stdout + done.stderr)
        if done.returncode or FAULT.search(done.stdout.decode(errors='replace')):
            raise RuntimeError('GPU fault signature or unavailable kernel journal')

    def cleanup():
        nonlocal active
        while active:
            try:
                ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + active + '$'], timeout=20)
                if not ids.strip():
                    active = None
                    return
                state = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]
                assert state['Config']['Labels'].get('b70.upload.owner') == str(os.getpid())
                if state['State']['Running']:
                    receipt['cleanup_error'] = 'Owned oracle needed termination'
                run(['docker', 'rm', '-f', active], active + '-remove.log')
            except Exception as e:
                receipt['cleanup_error'] = str(e)
                save()
                time.sleep(2)

    save()
    try:
        oracle = json.loads(a.oracle_receipt.read_text())
        assert oracle['passed'] and oracle['libraries_unchanged'] and oracle['image'] == IMAGE
        binary = a.oracle_receipt.parent / 'source-upload-oracle'
        assert sha(binary) == oracle['binary_sha256']
        plan = json.loads(a.oracle_plan.read_text())
        version2 = plan['schema'] == 2
        assert sha(REPO / plan['oracle_source']) == oracle['oracle_source_sha256'] == plan['oracle_source_sha256']
        engine = json.loads(Path(oracle['engine_receipt']).read_text())
        assert sha(Path(oracle['engine_receipt'])) == oracle['engine_receipt_sha256']
        assert any(x['path'].endswith('0010-sycl-native-ple-allocation-accounting.patch') for x in engine['patches'])
        for path, expected in oracle['library_sha256'].items():
            assert sha(Path(path)) == expected
        pack_receipt = json.loads(a.pack_receipt.read_text())
        assert pack_receipt['metadata_complete']
        pack = Path(pack_receipt['RESULT']['pack'])
        for rel, identity in pack_receipt['RESULT']['files'].items():
            assert sha(pack / rel) == identity['sha256']
        roster = Path(plan['source_roster']['path'])
        assert sha(roster) == plan['source_roster']['sha256']
        lock = json.loads((REPO / 'strata/flash-next/model-lock.json').read_text())
        shards = [REPO / lock['destination'] / row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')]
        assert len(shards) == 4
        receipt.update(oracle_receipt_sha256=sha(a.oracle_receipt), pack_receipt_sha256=sha(a.pack_receipt),
                       source_roster_sha256=sha(roster), image=IMAGE, controller_sha256=sha(Path(__file__)),
                       oracle_plan_sha256=sha(a.oracle_plan), oracle_schema=plan['schema'])
        (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
        health('pre')
        for label, mask, stages in [('card0', '0', ['0:1:0', '1:2:0', '47:48:0']),
                                    ('card1', '1', ['0:1:0', '1:2:0', '47:48:0']),
                                    ('pair', '0,1', ['0:1:0', '1:2:1', '47:48:1'])]:
            if stopped:
                raise RuntimeError('Interrupted before oracle launch')
            faults()
            active = f'source-upload-{os.getpid()}-{label}'
            args = ['/oracle/source-upload-oracle', '--pack', str(pack), '--source-roster', str(roster),
                    '--output', '/results/' + label + '.json']
            for shard in shards:
                args += ['--shard', str(shard)]
            for stage in stages:
                args += ['--stage', stage]
            cmd = ['docker', 'run', '-d', '--name', active, '--label', 'b70.upload.owner=' + str(os.getpid()),
                   '--network', 'none', '--device', '/dev/dri', '--user', '1000:1000',
                   '--group-add', str(os.stat('/dev/dri/renderD128').st_gid), '--group-add', str(os.stat('/dev/dri/card0').st_gid),
                   '--memory', '4g', '--memory-swap', '4g', '-e', 'ZE_AFFINITY_MASK=' + mask,
                   '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu', '-e', 'SYCL_CACHE_PERSISTENT=0',
                   '-e', 'STRATA_SYCL_NATIVE_HC=1', '-v', str(a.oracle_receipt.parent.resolve()) + ':/oracle:ro',
                   '-v', str(out) + ':/results', '-v', str(REPO) + ':' + str(REPO) + ':ro',
                   '-v', '/mnt/vm_8tb/b70:/mnt/vm_8tb/b70:ro', IMAGE, 'exec ' + shlex.join(args)]
            if version2:
                at = cmd.index(IMAGE)
                environment = plan['runtime_environment']
                extras = []
                for key, value in environment.items():
                    if key not in ['STRATA_SYCL_NATIVE_HC', 'ONEAPI_DEVICE_SELECTOR', 'SYCL_CACHE_PERSISTENT']:
                        extras += ['-e', key + '=' + value]
                cmd[at:at] = extras
            run(cmd, label + '-launch.log')
            deadline = time.monotonic() + 300
            while True:
                state = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]['State']
                if not state['Running']:
                    break
                if stopped or time.monotonic() > deadline:
                    raise RuntimeError('Oracle interrupted or deadline exceeded')
                faults()
                time.sleep(2)
            run(['docker', 'logs', active], label + '.log')
            report = json.loads((out / (label + '.json')).read_text())
            assert state['ExitCode'] == 0 and not state.get('OOMKilled')
            if version2:
                from parse_usm_logical_free_trace import parse_trace, negative_controls
                text = (out / (label + '.log')).read_text()
                trace = parse_trace(text)
                assert trace['passed'], trace['errors']
                trace['negative_controls'] = negative_controls(text, True)
                assert all(trace['negative_controls'].values())
                assert report['source_and_probe_passed'] and report['all_owners_destructor_returned']
                assert trace['counts']['owners'] == sum(s['unique_allocations'] + 1 for s in report['stages'])
                (out / (label + '-logical-free.json')).write_text(json.dumps(trace, indent=2) + '\n')
            else:
                assert report['passed'] and report['all_owners_destroyed']
            assert report['hc_images'] == 27 and report['ple_images'] == 3
            receipt['cases'].append(dict(case=label, state=state, report=report))
            save()
            cleanup()
            if receipt.get('cleanup_error'):
                raise RuntimeError('Unclean oracle teardown')
        faults()
    except Exception as e:
        receipt['error'] = dict(type=type(e).__name__, message=str(e))
    finally:
        cleanup()
        if receipt.get('pre_health_passed'):
            try:
                health('post')
                faults()
            except Exception as e:
                receipt['post_health_error'] = str(e)
        receipt['passed'] = len(receipt['cases']) == 3 and receipt.get('post_health_passed', False) and not any(
            k in receipt for k in ['error', 'cleanup_error', 'post_health_error'])
        receipt['finished_epoch'] = int(time.time())
        save()
    print(json.dumps(dict(passed=receipt['passed'], receipt=str(out / 'receipt.json'))))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
