#!/usr/bin/env python3
"""Build and qualify native HC primitives, owning both GPU leases and cleanup."""
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
SOURCE = Path('/mnt/vm_8tb/github/strata')
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
HEALTH = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
FAULT = re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', re.I)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--composition', action='store_true')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--ple', action='store_true')
    parser.add_argument('--projection-receipt', type=Path)
    parser.add_argument('--composition-receipt', type=Path)
    parser.add_argument('--native-build-receipt', type=Path)
    parser.add_argument('--leased', action='store_true')
    args = parser.parse_args()
    extended = args.composition or args.write or args.ple
    if extended and args.projection_receipt is None:
        parser.error('Composed/write gates require --projection-receipt')
    if args.write and (args.composition or args.composition_receipt is None or args.native_build_receipt is None):
        parser.error('--write requires composition/native build receipts and excludes --composition')
    if args.ple and (args.write or args.composition or args.native_build_receipt is None):
        parser.error('--ple requires a successful full native build receipt and excludes other modes')
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    overlay = args.overlay.resolve()
    fixture = REPO / ('strata/flash-next/native_ple_source_gpu_fixture.cpp' if args.ple else
                      'strata/flash-next/native_hc_write_gpu_fixture.cpp' if args.write else
                      'strata/flash-next/native_hc_composition_gpu_fixture.cpp' if args.composition else
                      'strata/flash-next/native_q8_hc_gpu_fixture.cpp')
    primitive = overlay / 'sycl/src/kernels/hc_native_projection.cpp'
    header = overlay / 'sycl/include/strata/kernels/hc_native_projection.hpp'
    result = dict(image=IMAGE, health_image=HEALTH, started_epoch=int(time.time()),
                  controller_sha256=sha(Path(__file__)), fixture_sha256=sha(fixture),
                  primitive_sha256=sha(primitive), header_sha256=sha(header),
                  cards=[], passed=False, model_integration_qualified=False)
    (out / 'controller-source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'fixture-source.cpp').write_bytes(fixture.read_bytes())
    result['composition'] = args.composition
    result['standalone_write'] = args.write
    result['ple'] = args.ple
    if extended:
        result['projection_fixture_sha256'] = sha(REPO / 'strata/flash-next/native_q8_hc_gpu_fixture.cpp')
        (out / 'projection-fixture-source.cpp').write_bytes((REPO / 'strata/flash-next/native_q8_hc_gpu_fixture.cpp').read_bytes())
    active = None
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True

    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, stop)

    def save():
        (out / 'receipt.json').write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')

    def run(command, filename, timeout, check=True):
        (out / (filename + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
        with (out / filename).open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if check and proc.returncode:
            raise RuntimeError(filename + ' exit ' + str(proc.returncode))
        return proc.returncode

    def health(stage):
        run([str(REPO / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH, '--timeout', '60'], stage + '-health.log', 180)
        run([str(REPO / 'bin/xpu-collective-health'), '--img', HEALTH, '--p2p', '0', '--timeout', '180'], stage + '-collective.log', 210)
        result[stage + '_health_passed'] = True
        save()

    def fault_audit():
        proc = subprocess.run(['journalctl', '-k', '--since', '@' + str(result['started_epoch']), '--no-pager'], capture_output=True, timeout=20)
        (out / 'kernel-journal.log').write_bytes(proc.stdout + proc.stderr)
        if proc.returncode or FAULT.search(proc.stdout.decode(errors='replace')):
            raise RuntimeError('Kernel journal unavailable or GPU fault signature')

    def cleanup():
        nonlocal active
        if active is None:
            return
        # Do not release the lease until the owned container is authoritatively absent.
        while True:
            try:
                observed = subprocess.run(['docker', 'ps', '-aq', '--filter', 'name=^/' + active + '$'], capture_output=True, timeout=20, check=True)
                if not observed.stdout.strip():
                    active = None
                    return
                obj = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]
                if obj['Config']['Labels'].get('b70.native-hc.owner') != str(os.getpid()):
                    raise RuntimeError('Container owner mismatch')
                if obj['State']['Running']:
                    result['cleanup_error'] = 'Container needed termination'
                    run(['docker', 'stop', '-t', '10', active], active + '-stop.log', 30, False)
                run(['docker', 'rm', '-f', active], active + '-remove.log', 30)
            except Exception as error:
                result['cleanup_error'] = str(error)
                save()
                time.sleep(2)

    save()
    try:
        if extended:
            prerequisite = json.loads(args.projection_receipt.read_text())
            assert prerequisite['passed'] and prerequisite['post_health_passed']
            assert prerequisite['image'] == IMAGE and len(prerequisite['cards']) == 2
            assert prerequisite['fixture_sha256'] == result['projection_fixture_sha256']
            result['projection_receipt'] = str(args.projection_receipt.resolve())
            result['projection_receipt_sha256'] = sha(args.projection_receipt)
        if args.write:
            composed = json.loads(args.composition_receipt.read_text())
            built = json.loads(args.native_build_receipt.read_text())
            assert composed['passed'] and composed['composition'] and composed['image'] == IMAGE
            assert built['passed'] and built['image'] == IMAGE and built['devices_exposed'] is False
            assert len(built['objects']) == 14
            for rel, expected_hash in built['source_hashes'].items():
                assert sha(overlay / rel) == expected_hash
            result['native_build_receipt_sha256'] = sha(args.native_build_receipt)
            result['composition_receipt_sha256'] = sha(args.composition_receipt)
        if args.ple:
            built = json.loads(args.native_build_receipt.read_text())
            assert built['build_rc'] == 0 and built['image'] == IMAGE and built['devices_exposed'] is False
            assert built['external_source_unchanged'] and built['plan_snapshot_unchanged']
            assert Path(built['source_copy']).resolve() == overlay
            assert subprocess.check_output(['git', '-C', str(overlay), 'status', '--porcelain'], text=True).strip() == built['source_copy_status']
            for rel, expected_hash in built['patched_source_sha256'].items():
                assert sha(overlay / rel) == expected_hash
            result['native_build_receipt_sha256'] = sha(args.native_build_receipt)
            rel = 'sycl/src/kernels/cuda/native_gr_norm.dp.cpp'
            assert sha(overlay / rel) == sha(SOURCE / rel)
            result['native_gr_norm_source_sha256'] = sha(overlay / rel)
        result['source_revision'] = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        assert result['source_revision'] == 'fb58e0dbc8399662c0e47c76578c6e878b14f6cf'
        plan = json.loads((REPO / 'strata/flash-next/native-q8-hc-primitive-plan.json').read_text())
        result['patch_sha256'] = sha(REPO / plan['patch'])
        assert result['patch_sha256'] == plan['patch_sha256']
        if not extended:
            compile_receipt = json.loads((overlay / 'compile-receipt.json').read_text())
            assert compile_receipt['patch_sha256'] == result['patch_sha256']
        expected = out / 'expected-overlay'
        (expected / 'sycl').mkdir(parents=True)
        cmake = SOURCE / 'sycl/CMakeLists.txt'
        assert sha(cmake) == plan['source_file_sha256']['sycl/CMakeLists.txt']
        (expected / 'sycl/CMakeLists.txt').write_bytes(cmake.read_bytes())
        subprocess.run(['git', 'apply', str(REPO / plan['patch'])], cwd=expected, check=True)
        assert sha(expected / primitive.relative_to(overlay)) == sha(primitive)
        assert sha(expected / header.relative_to(overlay)) == sha(header)
        if extended:
            patch2 = REPO / 'strata/flash-next/patches/0002-sycl-native-hc-composition.patch'
            result['composition_patch_sha256'] = sha(patch2)
            subprocess.run(['git', 'apply', str(patch2)], cwd=expected, check=True)
            if args.write or args.ple:
                patches = [REPO / p['path'] for p in built['patches'][2:]] if args.ple else [
                    REPO / 'strata/flash-next/patches' / n for n in ['0003-sycl-native-hc-source-owner-decode.patch', '0004-sycl-native-hc-prefill-verifier-routes.patch']]
                for patch in patches:
                    for rel in re.findall(r'^\+\+\+ b/(.+)$', patch.read_text(), re.M):
                        path = expected / rel
                        if not path.exists() and (SOURCE / rel).exists():
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes((SOURCE / rel).read_bytes())
                    assert any(p['sha256'] == sha(patch) for p in built['patches'])
                    subprocess.run(['git', 'apply', str(patch)], cwd=expected, check=True)
            for rel in ['sycl/src/kernels/hc_native_composition.cpp', 'sycl/include/strata/kernels/hc_native_composition.hpp']:
                assert sha(expected / rel) == sha(overlay / rel)
        # Compile from tracked primitive source, not the previously generated object.
        compiler = ['/opt/intel/oneapi/compiler/2026.1/bin/icpx', '-fsycl', '-std=c++17', '-O2',
                    '/repo/' + str(fixture.relative_to(REPO)),
                    '/overlay/sycl/src/kernels/hc_native_projection.cpp',
                    '-I/overlay/sycl/include', '-I/src/sycl/include', '-I/src/include',
                    '-o', '/out/native-q8-hc-fixture']
        if extended:
            compiler.insert(7, '/overlay/sycl/src/kernels/hc_native_composition.cpp')
        if args.write:
            compiler[2] = '-std=c++20'
            compiler.insert(4, '-DSTRATA_SYCL_Q8_HC_BUILT=1')
            compiler.insert(4, '-I/overlay/include')
        if args.ple:
            compiler[2] = '-std=c++20'
            compiler.insert(4, '-I/overlay/include')
            compiler.insert(4, '/overlay/sycl/src/kernels/cuda/native_ple_postops.dp.cpp')
            compiler.insert(4, '/overlay/sycl/src/kernels/cuda/native_gr_norm.dp.cpp')
        mounts = ['-v', str(REPO) + ':/repo:ro', '-v', str(SOURCE) + ':/src:ro',
                  '-v', str(overlay) + ':/overlay:ro', '-v', str(out) + ':/out']
        run(['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
             *mounts, IMAGE, 'exec ' + shlex.join(compiler)], 'compile.log', 600)
        result['binary_sha256'] = sha(out / 'native-q8-hc-fixture')
        health('pre')
        fault_audit()
        for card in [0, 1]:
            if stopped:
                raise RuntimeError('Interrupted before launch')
            active = f'native-hc-{os.getpid()}-{card}'
            command = ['docker', 'run', '-d', '--name', active, '--label', 'b70.native-hc.owner=' + str(os.getpid()),
                       '--network', 'none', '--device', '/dev/dri', '--user', f'{os.getuid()}:{os.getgid()}',
                       '--group-add', str(os.stat('/dev/dri/renderD128').st_gid),
                       '--group-add', str(os.stat('/dev/dri/card0').st_gid),
                       '--memory', '4g', '--memory-swap', '4g', '--ulimit', 'core=0',
                       '-v', '/dev/dri/by-path:/dev/dri/by-path:ro', '-v', str(out) + ':/out:ro',
                       '-e', 'ZE_AFFINITY_MASK=' + str(card), '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu',
                       '-e', 'SYCL_CACHE_PERSISTENT=0', IMAGE, 'exec /out/native-q8-hc-fixture']
            run(command, f'card{card}-launch.log', 60)
            deadline = time.monotonic() + 600
            while True:
                obj = json.loads(subprocess.check_output(['docker', 'inspect', active], timeout=20))[0]
                if not obj['State']['Running']:
                    break
                if stopped or time.monotonic() > deadline:
                    raise RuntimeError('Interrupted or fixture deadline exceeded')
                fault_audit()
                time.sleep(2)
            run(['docker', 'logs', active], f'card{card}.jsonl', 30)
            rows = []
            for line in (out / f'card{card}.jsonl').read_text().splitlines():
                if line.startswith('{'):
                    rows.append(json.loads(line))
            result['cards'].append(dict(card=card, state=obj['State'], records=rows))
            save()
            cleanup()
            fault_audit()
            if obj['State']['ExitCode'] or obj['State'].get('OOMKilled'):
                raise RuntimeError(f'Card {card} fixture failed')
            summary = rows[-1] if rows else {}
            cases = [r for r in rows if r.get('kind') == 'case']
            identities = {(r.get('role'), r.get('profile'), r.get('k'), r.get('m'), r.get('tokens')) for r in cases}
            profiles = {'mixed', 'negative_extreme', 'positive_extreme', 'cancellation', 'impulse', 'scale_edges'}
            expected_ids = {(f['role'], profile, f['k'], f['m'], f['tokens'])
                            for f in plan['fixtures'] for profile in profiles}
            valid = (summary.get('kind') == 'summary' and summary.get('mode') == 'GPU_NUMERICAL'
                     and summary.get('pass') and summary.get('cases') == 72
                     and summary.get('failed') == 0 and summary.get('shape_cases') == 12
                     and summary.get('numeric_negative_controls') == 72
                     and summary.get('guard_negative_control') is True
                     and summary.get('rejection_cases') == 22 and len(cases) == 72
                     and identities == expected_ids)
            for r in cases:
                valid = valid and all(r.get(k) for k in ['pass', 'finite', 'exact_repeat',
                                                        'exact_batch_vs_single', 'guards', 'unchanged_weights_input'])
                valid = valid and isinstance(r.get('nmse'), (float, int)) and 0 <= r['nmse'] <= 1e-6
                valid = valid and isinstance(r.get('normalized_linf'), (float, int)) and 0 <= r['normalized_linf'] <= 1e-4
            if args.composition:
                wanted = {(t, pending != 0, pending == 2, inj, profile)
                          for t in [1, 2, 4, 8] for pending in [0, 1, 2]
                          for inj in [False, True] for profile in ['mixed', 'epsilon_sensitive']}
                found = {(r['tokens'], r['apply'], r['inplace'], r['injection'], r['profile']) for r in cases}
                stages = [r for r in rows if r.get('kind') == 'stage']
                expected_stages = set()
                for r in cases:
                    names = {'per_stream_rms', 'xn', 'down_pre_silu', 'lo_post_silu', 'up_raw_gate', 'mixed'}
                    if r['apply']:
                        names.add('pending_R')
                    if r['injection']:
                        names.add('injection')
                    expected_stages.update((r['case'], n) for n in names)
                valid = (summary.get('mode') == 'GPU_COMPOSITION_NUMERICAL' and summary.get('pass')
                         and summary.get('cases') == 48 and summary.get('failed') == 0
                         and summary.get('stages') == 344 and summary.get('numeric_negative_controls') == 344
                         and len(cases) == 48 and found == wanted and len(stages) == 344
                         and {(r['case'], r['stage']) for r in stages} == expected_stages
                         and {r['case'] for r in cases} == set(range(48)))
                for r in cases:
                    valid = valid and all(r.get(k) for k in ['pass', 'exact_repeat', 'exact_batch_vs_single',
                                                            'guards', 'unchanged_weights_input'])
                for r in stages:
                    valid = valid and r.get('pass') and r.get('finite')
                    valid = valid and isinstance(r.get('nmse'), (float, int)) and 0 <= r['nmse'] <= 1e-6
                    valid = valid and isinstance(r.get('normalized_linf'), (float, int)) and 0 <= r['normalized_linf'] <= 1e-4
            if args.write:
                wanted = {(t, profile) for t in [1, 2, 4, 8, 9, 16, 17] for profile in ['tiny', 'mixed', 'saturated']}
                valid = (summary.get('mode') == 'GPU_WRITE_NUMERICAL' and summary.get('pass')
                         and summary.get('cases') == 21 and summary.get('failed') == 0
                         and summary.get('rejection_cases') == 15 and summary.get('numeric_negative_controls') == 21
                         and summary.get('guard_negative_control') is True and len(cases) == 21
                         and {(r.get('rows'), r.get('profile')) for r in cases} == wanted)
                for r in cases:
                    valid = valid and all(r.get(k) for k in ['pass', 'finite', 'exact_repeat', 'exact_batch_vs_single',
                        'exact_shared_wrapper', 'exact_pending_write', 'guards', 'unchanged_inputs', 'numeric_negative_control'])
                    valid = valid and isinstance(r.get('nmse'), (float, int)) and 0 <= r['nmse'] <= 1e-6
                    valid = valid and isinstance(r.get('normalized_linf'), (float, int)) and 0 <= r['normalized_linf'] <= 1e-4
            if args.ple:
                wanted_rows = {1, 2, 4, 8, 9, 16, 17}
                names = {'key_raw', 'value_raw', 'key_norm', 'query_norm', 'gate', 'gated', 'norm_conv',
                         'conv_silu', 'result', 'history_snapshots'}
                stages = [r for r in rows if r.get('kind') == 'stage']
                valid = (summary.get('mode') == 'GPU_PLE_SOURCE_NUMERICAL' and summary.get('pass')
                         and summary.get('cases') == 7 and summary.get('failed') == 0
                         and summary.get('stages') == 70 and summary.get('history_rows') == 57
                         and summary.get('numeric_negative_controls') == 70 and summary.get('guard_negative_control') is True
                         and len(cases) == 7 and {r.get('rows') for r in cases} == wanted_rows and len(stages) == 70
                         and {(r['rows'], r['stage']) for r in stages} == {(t, n) for t in wanted_rows for n in names})
                for r in cases:
                    valid = valid and all(r.get(k) for k in ['pass', 'exact_repeat', 'exact_chunk_vs_single',
                                                            'guards', 'unchanged_weights_input'])
                    valid = valid and all(r.get(k, 0) > 0 for k in ['bf16_value_reference_difference',
                        'f16_activation_reference_difference', 'q8_activation_reference_difference', 'f16_conv_reference_difference'])
                for r in stages:
                    valid = valid and r.get('pass') and r.get('finite')
                    valid = valid and isinstance(r.get('nmse'), (float, int)) and 0 <= r['nmse'] <= 1e-6
                    valid = valid and isinstance(r.get('normalized_linf'), (float, int)) and 0 <= r['normalized_linf'] <= 1e-4
            result['cards'][-1]['coverage_passed'] = bool(valid)
            if not valid:
                raise RuntimeError(f'Card {card} missing passing summary')
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
    finally:
        cleanup()
        if result.get('pre_health_passed'):
            try:
                health('post')
                fault_audit()
            except Exception as error:
                result['post_health_error'] = str(error)
        result['passed'] = (len(result['cards']) == 2 and result.get('post_health_passed', False)
                            and not any(k in result for k in ['error', 'cleanup_error', 'post_health_error']))
        result['finished_epoch'] = int(time.time())
        save()
    print(json.dumps(dict(passed=result['passed'], receipt=str(out / 'receipt.json'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
