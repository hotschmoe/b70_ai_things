#!/usr/bin/env python3
"""Qualify a bounded actual segmented expert mirror under parent-owned leases."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from c1_serve_controller import upload_gate
from parse_usm_logical_free_trace import parse_trace, negative_controls
from run_source_upload_oracle_full import verify_model_identity

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
HEALTH = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
FAULT = re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', re.I)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def source_roster(plan):
    inventory_path = Path(plan['inventory'])
    assert sha(inventory_path) == plan['inventory_sha256']
    inventory = json.loads(inventory_path.read_text())
    assert inventory['inventory_complete'] and not inventory['errors']
    tensors = {t['name']: (Path(f['path']), t) for f in inventory['files']
               if '/UD-Q4_K_XL/' in f['path'] for t in f['tensors']}
    rows = {}
    for layer in [0, 2, 11, 32, 47]:
        for expert in [0, 17]:
            digest = hashlib.sha256()
            roles, count = [], 0
            for role in ['gate', 'up', 'down']:
                path, tensor = tensors[f'blk.{layer}.ffn_{role}_exps.weight']
                shape = [640, 2560, 512] if role == 'down' else [2560, 640, 512]
                assert tensor['shape_ggml_order'] == shape and tensor['packed_bytes'] % 512 == 0
                extent = tensor['packed_bytes'] // 512
                offset = tensor['absolute_offset'] + expert * extent
                with path.open('rb') as f:
                    f.seek(offset)
                    data = f.read(extent)
                assert len(data) == extent
                digest.update(data)
                count += extent
                roles.append(dict(role=role, type=tensor['type_id'], shard=str(path), offset=offset, bytes=extent))
            assert roles[0]['type'] == roles[1]['type']
            rows[(layer, expert)] = dict(layer=layer, expert=expert, raw_bytes=count,
                source_sha256=digest.hexdigest(), gu_type=roles[0]['type'], down_type=roles[2]['type'], roles=roles)
    return rows


def report_gate(report, roster, pair):
    assert report['schema'] == 1 and report['full_serving_qualified'] is False
    for field in ['source_and_arithmetic_passed', 'all_owners_release_returned', 'partial_read_failure_cleanup_returned']:
        assert report[field] is True, field
    expected = dict(expert_cases=10, tokens_per_expert=3, graph_replays_after_source_close=10,
                    mixed_quant_layouts=3, mirrored_padded_bytes=33587200,
                    source_storage_reads=10, mirror_host_reads=20, registered_owner_records=12,
                    missing_entry_plan_negative_checks=10)
    assert all(report[k] == v for k, v in expected.items())
    stages = report['stages']
    assert len(stages) == 2
    assert [(s['lo'], s['hi'], s['device'], s['segments'], s['experts']) for s in stages] == [
        (0, 32, 0, 6, 6), (32, 48, 1 if pair else 0, 4, 4)]
    assert sum(s['bytes'] for s in stages) == expected['mirrored_padded_bytes']
    for stage in stages:
        expected_bytes = sum((r['raw_bytes'] + 255) // 256 * 256 for r in roster.values()
                             if stage['lo'] <= r['layer'] < stage['hi'])
        assert stage['bytes'] == expected_bytes
    assert len(report['expert_rows']) == 10
    assert {(r['layer'], r['expert']) for r in report['expert_rows']} == set(roster)
    assert {(r['gu_type'], r['down_type']) for r in report['expert_rows']} == {(12, 7), (12, 8), (13, 8)}
    for row in report['expert_rows']:
        original = roster[(row['layer'], row['expert'])]
        assert all(row[k] == original[k] for k in ['source_sha256', 'raw_bytes', 'gu_type', 'down_type'])
        assert row['input_q8_bytes_exact'] is True and row['hidden_q8_bytes_exact'] is True
        for key in ['gpu_vs_float_l1_relative', 'cpu_vs_float_l1_relative', 'gpu_vs_native_cpu_l1_relative']:
            assert math.isfinite(row[key]) and 0 <= row[key] <= 3e-2
        metrics = row['implementation_metrics']
        assert [m['stage'] for m in metrics] == ['gate', 'up', 'silu_hidden', 'down_matched_q8']
        assert all(math.isfinite(m['nmse']) and 0 <= m['nmse'] <= 1e-6 and
                   math.isfinite(m['normalized_linf']) and 0 <= m['normalized_linf'] <= 1e-4 for m in metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle-receipt', type=Path, required=True)
    p.add_argument('--source-upload', type=Path, required=True)
    p.add_argument('--source-oracle', type=Path, required=True)
    p.add_argument('--pack-receipt', type=Path, required=True)
    p.add_argument('--model-identity', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--leased', action='store_true')
    a = p.parse_args()
    if not a.leased:
        os.execv(str(ROOT / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output.resolve()
    receipt = dict(scope='Bounded actual expert mirror source/arithmetic/graph/lifecycle; not full serving qualification',
                   started_epoch=int(time.time()), passed=False, cases=[])
    active = None
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, stop)

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='ascii')

    def run(cmd, name, timeout=60):
        (out / (name + '.command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
        with (out / name).open('w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=True)

    def health(stage):
        run([str(ROOT / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH], stage + '-health.log', 180)
        run([str(ROOT / 'bin/xpu-collective-health'), '--img', HEALTH, '--p2p', '0', '--timeout', '180'], stage + '-collective.log', 210)
        receipt[stage + '_health_passed'] = True
        save()

    def faults():
        r = subprocess.run(['journalctl', '-k', '--since', '@' + str(receipt['started_epoch']), '--no-pager'], capture_output=True, timeout=20)
        (out / 'kernel-journal.log').write_bytes(r.stdout + r.stderr)
        assert r.returncode == 0 and not FAULT.search(r.stdout.decode(errors='replace')), 'Fault or unreadable journal'

    def cleanup():
        nonlocal active
        while active:
            try:
                ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + active + '$'], timeout=20)
                if not ids.strip():
                    active = None
                    return
                obj = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]
                assert obj['Config']['Labels'].get('b70.mirror.owner') == str(os.getpid())
                if obj['State']['Running']:
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
        root = a.oracle_receipt.parent.resolve()
        binary = root / 'stage-mirror-oracle'
        assert sha(binary) == oracle['binary_sha256']
        assert sha(root / 'plan.snapshot.json') == oracle['plan_sha256']
        plan = json.loads((root / 'plan.snapshot.json').read_text())
        assert plan['runtime_environment'].get('UR_ENABLE_LAYERS') == 'UR_LAYER_TRACING'
        assert plan['runtime_environment'].get('UR_LOG_TRACING') == 'level:info;flush:info;output:stderr'
        assert 'ZE_AFFINITY_MASK' not in plan['runtime_environment']
        assert plan['runtime_environment'].get('ONEAPI_DEVICE_SELECTOR', 'level_zero:gpu') == 'level_zero:gpu'
        assert plan['runtime_environment'].get('SYCL_CACHE_PERSISTENT', '0') == '0'
        assert sha(root / 'oracle.cpp') == oracle['oracle_source_sha256'] == plan['oracle_source_sha256']
        assert sha(root / 'stage_mirror_q8_reference.hpp') == oracle['reference_source_sha256'] == plan['reference_source_sha256']
        assert sha(ROOT / plan['oracle_source']) == oracle['oracle_source_sha256']
        assert sha(ROOT / plan['reference_source']) == oracle['reference_source_sha256']
        for path, expected in oracle['library_sha256'].items():
            assert sha(path) == expected
        engine_receipt = Path(oracle['engine_receipt'])
        assert sha(engine_receipt) == oracle['engine_receipt_sha256']
        receipt['source_upload'] = upload_gate(a.source_upload, a.source_oracle, engine_receipt, a.pack_receipt)
        lock = json.loads((ROOT / 'strata/flash-next/model-lock.json').read_text())
        shards = [ROOT / lock['destination'] / f['path'] for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')]
        receipt['model_identity'] = verify_model_identity(a.model_identity, lock, shards)
        roster = source_roster(plan)
        (out / 'source-roster.json').write_text(json.dumps(list(roster.values()), indent=2) + '\n')
        receipt.update(source_roster_sha256=sha(out / 'source-roster.json'), oracle_receipt_sha256=sha(a.oracle_receipt),
                       controller_sha256=sha(Path(__file__)), plan_sha256=oracle['plan_sha256'])
        (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
        pack = Path(json.loads(a.pack_receipt.read_text())['RESULT']['pack'])
        health('pre')
        for label, mask, mode in [('card0', '0', 'single'), ('card1', '1', 'single'), ('pair', '0,1', 'pair')]:
            assert not stopped, 'Interrupted'
            faults()
            verify_model_identity(a.model_identity, lock, shards)
            active = f'mirror-oracle-{os.getpid()}-{label}'
            args = ['/oracle/stage-mirror-oracle', '--pack', str(pack), '--output', '/results/' + label + '.json', '--mode', mode]
            for path in shards:
                args += ['--shard', str(path)]
            cmd = ['docker', 'run', '-d', '--name', active, '--label', 'b70.mirror.owner=' + str(os.getpid()),
                   '--network', 'none', '--device', '/dev/dri', '--user', '1000:1000',
                   '--group-add', str(os.stat('/dev/dri/renderD128').st_gid), '--group-add', str(os.stat('/dev/dri/card0').st_gid),
                   '--memory', '8g', '--memory-swap', '8g', '-e', 'ZE_AFFINITY_MASK=' + mask,
                   '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu', '-e', 'SYCL_CACHE_PERSISTENT=0',
                   '-v', str(root) + ':/oracle:ro', '-v', str(out) + ':/results',
                   '-v', str(ROOT) + ':' + str(ROOT) + ':ro', '-v', '/mnt/vm_8tb/b70:/mnt/vm_8tb/b70:ro']
            for key, value in plan['runtime_environment'].items():
                if key not in ['ONEAPI_DEVICE_SELECTOR', 'SYCL_CACHE_PERSISTENT']:
                    cmd += ['-e', key + '=' + value]
            cmd += [IMAGE, 'exec ' + shlex.join(args)]
            run(cmd, label + '-launch.log')
            deadline = time.monotonic() + 600
            while True:
                state = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]['State']
                if not state['Running']:
                    break
                assert not stopped and time.monotonic() < deadline, 'Oracle deadline/interruption'
                faults()
                time.sleep(2)
            run(['docker', 'logs', active], label + '.log')
            assert state['ExitCode'] == 0 and not state.get('OOMKilled'), state
            report = json.loads((out / (label + '.json')).read_text())
            report_gate(report, roster, mode == 'pair')
            text = (out / (label + '.log')).read_text()
            ledger = parse_trace(text)
            assert ledger['passed'] and ledger['counts']['owners'] == report['registered_owner_records'], ledger['errors']
            ledger['negative_controls'] = negative_controls(text, True)
            assert all(ledger['negative_controls'].values())
            (out / (label + '-logical-free.json')).write_text(json.dumps(ledger, indent=2) + '\n')
            verify_model_identity(a.model_identity, lock, shards)
            receipt['cases'].append(dict(case=label, state=state, report=report))
            save()
            cleanup()
            assert not receipt.get('cleanup_error'), 'Unclean cleanup'
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
            key in receipt for key in ['error', 'cleanup_error', 'post_health_error'])
        receipt['finished_epoch'] = int(time.time())
        save()
    print(json.dumps(dict(passed=receipt['passed'], receipt=str(out / 'receipt.json'))))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
