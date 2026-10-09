#!/usr/bin/env python3
"""Prepare and supervise exact-artifact C1 serving; CPU preparation never launches GPU.

The parent owns leases, pre/post-health and recovery. Launch stays foreground
until the owned container has stopped and been removed. No shelf promotion.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import math
import time
import urllib.request
from c1_trace_contract import trace_end_accepted

REPO = Path(__file__).resolve().parents[2]
BASE_IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
ENGINE = Path('/mnt/vm_8tb/b70/build/strata-integrated25-31-SDK-UNBUILT')
PACK = Path('/mnt/vm_8tb/b70/models/flashnext-native-source-pack-20261009-v3')
INTAKE = Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/native-source-pack-intake-v4.json')
PRIMARY = 'hotschmoe-dd'
FILES = ('vocab.json', 'merges.txt', 'token_type.json', 'tokenizer.json', 'chat_template.jinja')
PYTHON_SOURCES = ('serve/server.py', 'serve/frontend.py', 'serve/artifact_identity.py', 'serve/batch_request_identity.py', 'tools/strata_tokenizer.py', 'tools/gguf_reader.py')
PROFILES = {
    'one-card': {'cards': [0], 'context': 2048, 'prefill': 64, 'ple_rows': 65536, 'extra': [],
        'alias': 'qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-source-hc-ple-mtp0-c1-fp16kv-ctx2048'},
    'one-card-segmented': {'cards': [0], 'context': 2048, 'prefill': 64, 'ple_rows': 65536,
        'extra': ['--adapt-every', '0', '--no-prefill-borrow'],
        'env': {'STRATA_STAGE_MIRRORS': '1', 'STRATA_STAGE_MIRROR_SEGMENT_MIB': '1024'},
        'segmented': True,
        'alias': 'qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-source-hc-ple-mtp0-c1-fp16kv-ctx2048-seg1024-adapt0-borrow0'},
    'two-card-segmented': {'cards': [0, 1], 'context': 8192, 'prefill': 128, 'ple_rows': 1048576,
        'extra': ['--layer-split', '32', '--split-device', '1', '--trim-stage-weights', '--adapt-every', '0', '--no-prefill-borrow'],
        'env': {'STRATA_STAGE_MIRRORS': '1', 'STRATA_STAGE_MIRROR_SEGMENT_MIB': '1024'},
        'segmented': True, 'one_card_profile': 'one-card-segmented',
        'alias': 'qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-source-hc-ple-mtp0-c1-fp16kv-ctx8192-seg1024-adapt0-borrow0'},
    'two-card': {'cards': [0, 1], 'context': 8192, 'prefill': 128, 'ple_rows': 1048576,
        'extra': ['--layer-split', '32', '--split-device', '1', '--trim-stage-weights'],
        'alias': 'qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-source-hc-ple-mtp0-c1-fp16kv-ctx8192'},
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    temporary.replace(path)


def stat_signature(path):
    st = Path(path).stat()
    # Canonical producer/consumer source stat: retain ctime, no dict/list coercion.
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def run(command, timeout=60, **kwargs):
    return subprocess.run(command, check=True, timeout=timeout, **kwargs)



# New combined source generation. Prior C1 controllers/receipts are immutable.
COMBINED_PLAN = REPO / 'strata/flash-next/prefix-residual30-mirror31-engine-build-plan-v1.json'
COMBINED_PLAN_SHA = '82951242c998acecd0b0a138466991c607c1ab89abbf3b4501c001e0feaae96e'
PYTHON_SOURCE_SHA = {'serve/batch_request_identity.py': '2f0149108f97d31ab510cca36d7b62338f1f899e5b4ce5414a90d61e0e38b14f', 'serve/server.py': '3166f70fa013b905adf713dba404a2eb91a4f3381842dafb542af98cf978d241', 'serve/frontend.py': '095bbac7d49b19d815d2634457d14d216ba9bd4e98998a4e3b4288f20a25e923', 'serve/artifact_identity.py': '47619fcd5def60d13861773b359e9bab35499212f858ba649ce6462feb15027f', 'tools/strata_tokenizer.py': 'd2fdcd2e55b4bc47c3ee6a10a0a7efa2029128cc229fb1de18da6e22d3606ff2', 'tools/gguf_reader.py': '5ecf739bb3c2f07bc7f889f7f3a455e3dfdb40068409e10781a1a9b2be289b5e'}
KNOWN_SOURCE_PAGES = ((3857879040, '2780fef9ce50fa1acbd4bdbf6c311b847395571fcc5e6ddb55898841fbcee90e'), (39437303808, 'd69f7ffbfaab20277926926a4e542ce906b7fe9f4a4791dc7c46e945f0b1bced'))
for _profile in PROFILES.values():
    _topology = 'gpu0' if _profile['cards'] == [0] else 'gpu0-1-split32-16'
    _profile['alias'] += '-srcintegrated25-31-api22v2-prefix30-mirror31-' + _topology


def combined_generation_gate(engine):
    engine = Path(engine).resolve()
    require(sha(COMBINED_PLAN) == COMBINED_PLAN_SHA, 'Reviewed combined source plan changed')
    plan = read(COMBINED_PLAN)
    receipt = engine / 'receipt.json'
    build = read(receipt)
    require(build.get('build_rc') == 0 and build.get('external_source_unchanged') is True and
            build.get('plan_snapshot_unchanged') is True and build.get('image') == BASE_IMAGE,
            'Actual new combined SDK build prerequisite missing or failed')
    require(build.get('plan_sha256') == COMBINED_PLAN_SHA and sha(engine / 'plan.snapshot.json') == COMBINED_PLAN_SHA and
            Path(build['plan_snapshot']).resolve() == engine / 'plan.snapshot.json', 'SDK plan/source generation association differs')
    require(len(plan['patches']) == 31 and len({row['path'] for row in plan['patches']}) == 31, 'Exact ordered31 patch recipe required')
    require(build.get('source_revision') == plan['source_revision'] and build.get('ggml_revision') == plan['ggml']['revision'] and
            build.get('patches') == plan['patches'] and Path(build['source_copy']).resolve() == engine / 'source',
            'Actual combined source/dependency/patch chain differs')
    ledger = plan['expected_patched_source_sha256']
    require(len(ledger) == 62 and set(ledger) == set(plan['overlay_files']) and build.get('patched_source_sha256') == ledger,
            'Exact complete62 source ledger required; earlier-generation subset refused')
    for name, digest in ledger.items():
        require(sha(engine / 'source' / name) == digest, 'Actual combined source changed: ' + name)
    headers = plan['added_header_payloads']
    require(len(headers) == 26 and len({h['path'] for h in headers}) == 26, 'Exact26 consumed header payloads required')
    for header in headers:
        require(ledger.get(header['path']) == header['sha256'] and sha(engine / 'source' / header['path']) == header['sha256'], 'Consumed header payload differs: ' + header['path'])
    require(plan['runtime_python_sources'] == PYTHON_SOURCE_SHA, 'Integrated six-source Python plan differs')
    targets = plan['build_targets']
    require(len(targets) == 8 and len(set(targets)) == 8 and
            set(build.get('binary_sha256', {})) == {str(engine / 'build' / name) for name in targets},
            'New SDK all8 target ledger required')
    for name in targets:
        path = engine / 'build' / name
        require(sha(path) == build['binary_sha256'][str(path)], 'Combined SDK target changed: ' + name)
    require(set(PYTHON_SOURCES) == set(PYTHON_SOURCE_SHA) and len(PYTHON_SOURCES) == 6, 'Complete6 runtime Python source contract required')
    for name, digest in PYTHON_SOURCE_SHA.items():
        require(sha(engine / 'source' / name) == digest, 'Actual six-source API runtime differs: ' + name)
    return {'schema': 3, 'plan_path': str(COMBINED_PLAN), 'plan_sha256': COMBINED_PLAN_SHA,
            'source_count': 62, 'header_payloads': headers, 'controller_generation': 9, 'sdk_targets': targets, 'runtime_python_sources': dict(PYTHON_SOURCE_SHA),
            'engine_receipt': str(receipt), 'engine_receipt_sha256': sha(receipt),
            'full_model_math_qualified': False, 'concurrency_qualified': False}


def full_source_case_gate(device, roster, source_shapes):
    require(device.get('schema') == 2 and device.get('coverage') == 'whole390' and
            device.get('source_and_probe_passed') is True and device.get('all_owners_destructor_returned') is True and
            device.get('hc_images') == 387 and device.get('ple_images') == 3,
            'Complete387 HC plus3 PLE whole-source/readback/destructor gate required')
    captured = [image for stage in device['stages'] for image in stage['source_rows']]
    require(len(captured) == 390 and len({image['name'] for image in captured}) == 390 and
            {image['name'] for image in captured} == set(roster), 'Whole source image name/coverage differs')
    for image in captured:
        source = roster[image['name']]
        original_shape = source_shapes[image['name']]
        shape = original_shape + [1] if len(original_shape) == 1 else original_shape
        require(image.get('gpu_byte_equal') is True and image['gpu_sha256'] == source['sha256'] and
                image['type'] == source['type_id'] and [image['ne0'], image['ne1']] == shape and
                image['bytes'] == source['bytes'] and image['absolute_offset'] == source['absolute_offset'] and
                Path(image['shard']).name == Path(source['shard']).name,
                'Actual uploaded original source image differs: ' + image['name'])
    require(all(stage['accounting_equal'] and stage['expected_image_bytes'] == stage['reported_weight_bytes'] and
                stage['ordinary_images'] > 0 and stage['unique_allocations'] ==
                stage['hc_images'] + stage['ple_images'] + stage['ordinary_images'] for stage in device['stages']),
            'Ordinary/native source count or complete owned image byte accounting differs')
    # Every model layer and the final mixer are represented exactly once.
    layers = [layer for stage in device['stages'] for layer in range(stage['lo'], stage['hi'])]
    require(sorted(layers) == list(range(48)), 'Whole model layer ownership is incomplete or duplicated')
    return captured


def segmented_profile_gate(profile, build, arguments, env):
    if not profile.get('segmented'):
        return
    names = {Path(row['path']).name for row in build['patches']}
    require({'0011-sycl-strict-native-omit-legacy-ple-key.patch',
             '0012-sycl-segmented-stage-expert-mirrors.patch'} <= names,
            'Segmented strict-native profile requires actual0011/0012 engine generation')
    require(env.get('STRATA_STAGE_MIRRORS') == '1' and env.get('STRATA_STAGE_MIRROR_SEGMENT_MIB') == '1024' and
            env.get('STRATA_VERIFY_NO_HOST') == '1' and env.get('STRATA_VERIFY_DEVICE_PLAN') == '1',
            'Segmented profile environment/complete coverage guards differ')
    require('--stream-experts' in arguments and arguments[arguments.index('--expert-cache') + 1] == 'auto' and
            arguments[arguments.index('--adapt-every') + 1] == '0' and '--no-prefill-borrow' in arguments and
            arguments[arguments.index('--adapt-swaps') + 1] == '0', 'Immutable segmented mirror/adaptation/prefill contract differs')
    forbidden = {'--mtp', '--adapt-async', '--peer-device', '--resident-experts', '--pipeline-windows',
                 '--expert-cache-remote', '--expert-cache-device1', '--expert-cache-device2', '--expert-cache-device3'}
    require(not forbidden.intersection(arguments), 'Segmented mirrors have an unsupported MTP/adaptation/peer/remote/pipeline route')
    if '--split-device' in arguments:
        require(arguments[arguments.index('--split-device') + 1] == '1', 'Segmented serving does not support same-device split')


def original_upload_gate(path, oracle_path, engine_receipt, pack_receipt):
    report = read(path)
    require(report.get('passed') is True and report.get('pre_health_passed') is True and
            report.get('post_health_passed') is True, 'GPU upload/lifecycle did not pass')
    require(report.get('image') == BASE_IMAGE, 'Upload runtime identity differs')
    require(not any(k in report for k in ['error', 'cleanup_error', 'post_health_error']), 'Upload lifecycle has errors')
    require(report.get('oracle_receipt_sha256') == sha(oracle_path), 'Upload oracle receipt differs')
    require(report.get('pack_receipt_sha256') == sha(pack_receipt), 'Upload pack receipt differs')
    oracle = read(oracle_path)
    require(oracle.get('passed') is True and oracle.get('libraries_unchanged') is True, 'Oracle source build did not pass')
    require(oracle.get('engine_receipt_sha256') == sha(engine_receipt), 'Upload oracle targets another engine generation')
    require(report.get('oracle_schema') == 2, 'This serving controller requires version2 logical-free lifecycle evidence')
    oracle_root = Path(oracle_path).parent
    require(sha(oracle_root / 'plan.snapshot.json') == oracle['plan_sha256'] == report['oracle_plan_sha256'], 'Upload frozen plan fingerprint differs')
    require(sha(oracle_root / 'oracle.cpp') == oracle['oracle_source_sha256'], 'Upload frozen source fingerprint differs')
    require(sha(oracle_root / 'source-upload-oracle') == oracle['binary_sha256'], 'Upload executable fingerprint differs')
    plan = read(oracle_root / 'plan.snapshot.json')
    require(plan['schema'] == 2 and plan['oracle_source_sha256'] == oracle['oracle_source_sha256'], 'Upload snapshot/source contract differs')
    require(sha(plan['source_roster']['path']) == plan['source_roster']['sha256'] == report['source_roster_sha256'], 'Original source roster fingerprint differs')
    require(sha(plan['source_roster']['receipt']) == plan['source_roster']['receipt_sha256'], 'Original source roster receipt changed')
    roster = {row['name']: row for row in read(plan['source_roster']['receipt'])['RESULT']['rows']}
    require(sha(plan['inventory']) == plan['inventory_sha256'], 'Original GGUF inventory fingerprint differs')
    inventory = read(plan['inventory'])
    require(inventory['inventory_complete'] and not inventory['errors'], 'Original source inventory incomplete')
    source_shapes = {tensor['name']: tensor['shape_ggml_order'] for file in inventory['files'] if '/UD-Q4_K_XL/' in file['path'] for tensor in file['tensors']}
    require(len(roster) == 390, 'Complete387 HC plus3 PLE source roster required; narrow30 is insufficient')
    from parse_usm_logical_free_trace import parse_trace, negative_controls
    cases = report.get('cases', [])
    required_cases = {'full390_card0', 'full390_card1', 'full390_two_device24_24', 'full390_actual_model_static_bounds'}
    labels = [row.get('case') for row in cases]
    require(len(labels) == len(set(labels)) and required_cases <= set(labels), 'Incomplete per-card/pair/model-static whole-source coverage')
    for row in cases:
        state, device = row.get('state', {}), row.get('report', {})
        require(state.get('ExitCode') == 0 and not state.get('OOMKilled') and not state.get('Running'), 'Upload process has no clean terminal state')
        full_source_case_gate(device, roster, source_shapes)
        if row['case'] == 'full390_actual_model_static_bounds':
            require(device.get('bounds') == 'static', 'Actual model-call static bounds were not exercised')
        logical_path = Path(path).parent / (row['case'] + '-logical-free.json')
        logical = read(logical_path)
        text = (Path(path).parent / (row['case'] + '.log')).read_text()
        checked = parse_trace(text)
        require(checked['passed'] and all(negative_controls(text, True).values()), 'Chronological context-matched logical-free trace failed')
        require(logical.get('passed') and set(logical.get('negative_controls', {})) == {'missing_free', 'double_free', 'failed_free'} and all(logical['negative_controls'].values()), 'Logical-free negative controls incomplete')
        require(all(logical.get(k) == v for k, v in checked.items()), 'Saved logical-free evidence differs from raw trace')
        require(logical['counts']['owners'] == sum(stage['unique_allocations'] + 1 for stage in device['stages']) and not logical['live'], 'Owned allocation/scratch coverage differs or live ledger nonempty')
    return {'path': str(Path(path).resolve()), 'sha256': sha(path), 'oracle': str(Path(oracle_path).resolve()),
            'oracle_sha256': sha(oracle_path), 'oracle_schema': 2, 'source_coverage': 'whole390',
            'ordinary_payload_readback_qualified': False,
            'logical_free_parser_sha256': sha(REPO / 'strata/flash-next/parse_usm_logical_free_trace.py')}



UPLOAD_RUNNER_V2_SHA = '05f7ff786922916d83589657e0e3e20ea24198b6e84b3faa84d9611592bd32ec'
UPLOAD_WATCHDOG_SHA = '45a793c86c3af0e488a28e77648126b3a59fb9d318b6ba224fb1685c725dbcb1'


def strict_v2_upload_provenance(path, oracle_path, engine_receipt):
    path = Path(path).resolve(); report = read(path)
    require(report.get('passed') is True and report.get('runner_generation') == 2 and
            report.get('controller_sha256') == UPLOAD_RUNNER_V2_SHA and report.get('watchdog_sha256') == UPLOAD_WATCHDOG_SHA,
            'New combined generation requires genuine uploadrunner v2/watchdog provenance; legacy passed flag refused')
    require(sha(REPO / 'strata/flash-next/run_source_upload_oracle_full_v2.py') == UPLOAD_RUNNER_V2_SHA and
            sha(REPO / 'strata/flash-next/source_page_watchdog_v3.py') == UPLOAD_WATCHDOG_SHA and
            sha(path.parent / 'controller.py') == UPLOAD_RUNNER_V2_SHA and
            sha(path.parent / 'source-page-watchdog.py') == UPLOAD_WATCHDOG_SHA,
            'Upload runner/watchdog actual source snapshots changed')
    oracle = read(oracle_path)
    require(report.get('engine_receipt') == str(Path(engine_receipt).resolve()) and
            report.get('engine_receipt_sha256') == sha(engine_receipt) == oracle.get('engine_receipt_sha256') and
            report.get('oracle_receipt_sha256') == sha(oracle_path) and
            Path(oracle['engine_receipt']).resolve() == Path(engine_receipt).resolve(), 'New upload/oracle/engine receipt association differs')
    require(report.get('owned_terminal') is True and report.get('post_health_passed') is True and
            report.get('post_identity_complete4') is True and
            not any(key in report for key in ('error','cleanup_error','post_health_error','post_identity_error')),
            'New upload terminal/health/source identity gate incomplete')
    binding = report.get('post_model_identity', {})
    identity_path = path.parent / 'post-model-identity.json'
    require(binding.get('passed') is True and binding.get('path') == str(identity_path) and
            binding.get('sha256') == sha(identity_path), 'Upload new full4 identity missing, moved or changed')
    identity = read(identity_path)
    epochs = [report.get('owned_terminal_epoch'), report.get('post_health_finished_epoch'), report.get('finished_epoch'),
              identity.get('started'), identity.get('finished')]
    require(all(type(value) in (int,float) and math.isfinite(value) and value > 0 for value in epochs), 'Upload source chronology timestamps invalid')
    terminal, health, finished, started, hashed = epochs; boundary = max(terminal, health)
    require(started >= boundary and hashed >= started and finished >= hashed and
            identity.get('after_terminal_and_post_health_epoch') == boundary and
            binding.get('started') == started and binding.get('finished') == hashed,
            'New full4 scan must complete after actual terminal/post-health and before upload PASS')
    lock_path = REPO / 'strata/flash-next/model-lock.json'; lock = read(lock_path)
    expected = {str(REPO / lock['destination'] / row['path']):row for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')}
    rows = identity.get('rows', [])
    require(identity.get('passed') is True and identity.get('model_revision') == lock['revision'] and
            identity.get('lock_sha256') == sha(lock_path) and identity.get('read_mode') == 'ordinary complete buffered read; no direct IO/cache mutation' and
            len(expected) == len(rows) == 4 and {row['path'] for row in rows} == set(expected), 'New final complete4 publisher/source roster invalid')
    for row in rows:
        want = expected[row['path']]
        require(row.get('passed') is True and row['sha256'] == row['expected_sha256'] == want['sha256'] and
                row['bytes'] == want['size'] and row['stat_before'] == row['stat_after'] == stat_signature(Path(row['path'])),
                'New final publisher hash/stat changed: ' + row['path'])
    require(report.get('source_guard_between_polls_unobserved') is True and report.get('source_guard_poll_seconds') == 2,
            'Known-page polling scope must remain bounded, not continuous identity proof')
    def page_guard(name):
        value = report.get(name, {}); third = next(key for key in expected if '00003-of-00004' in key)
        require(value.get('passed') is True and value.get('path') == third and
                value.get('stat_before') == value.get('stat_after') == stat_signature(Path(third)), 'Two-page source guard path/stat differs: ' + name)
        observed = value.get('rows', [])
        require(len(observed) == 2 and [(row['offset'], row['expected_sha256']) for row in observed] == list(KNOWN_SOURCE_PAGES) and
                all(row.get('passed') is True and row['bytes'] == 4096 and row['sha256'] == row['expected_sha256'] for row in observed),
                'Both exact known-page views required: ' + name)
        epoch = value.get('epoch')
        require(type(epoch) in (int,float) and math.isfinite(epoch) and report['started_epoch'] <= epoch <= finished, 'Source guard chronology differs: ' + name)
        return epoch
    for name in ('known_pages_admission','known_pages_before_post_hash','known_pages_after_post_hash'):
        page_guard(name)
    require(report['known_pages_before_post_hash']['epoch'] <= started and report['known_pages_after_post_hash']['epoch'] >= hashed,
            'Both-page guards must bracket fresh final full4 scan')
    require(len(report.get('expected_cases', [])) == len(report.get('cases', [])) == 5, 'Exactly original five source cases required')
    for case in report.get('expected_cases', []):
        before = page_guard('known_pages_before_' + case); after = page_guard('known_pages_after_' + case)
        require(before <= after <= terminal, 'Case page guards must precede owned terminal: ' + case)
    return {'runner_generation': 2, 'runner_sha256': UPLOAD_RUNNER_V2_SHA, 'watchdog_sha256': UPLOAD_WATCHDOG_SHA,
            'post_model_identity': str(identity_path), 'post_model_identity_sha256': binding['sha256'],
            'post_full4_source_qualified': True, 'known_page_polling_continuous_identity_qualified': False}


def upload_gate(path, oracle_path, engine_receipt, pack_receipt):
    strengthened = strict_v2_upload_provenance(path, oracle_path, engine_receipt)
    return {**original_upload_gate(path, oracle_path, engine_receipt, pack_receipt), **strengthened}


def original_page_sentinel(shards):
    source = next(Path(row['path']) for row in shards if '00003-of-00004' in row['path'])
    before = stat_signature(source)
    pages = []
    with source.open('rb') as handle:
        for offset, expected in KNOWN_SOURCE_PAGES:
            handle.seek(offset); data = handle.read(4096)
            pages.append({'offset': offset, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'expected_sha256': expected})
    after = stat_signature(source)
    require(before == after and all(p['bytes'] == 4096 and p['sha256'] == p['expected_sha256'] for p in pages),
            'Known original page sentinel changed; stop dependent model work and preserve both pages, no automatic cache repair')
    return {'schema': 3, 'path': str(source), 'pages': pages, 'read_mode': 'buffered read only; no invalidation/write/repair'}


def runtime(a):
    a.output.mkdir(parents=True, exist_ok=False)
    context = REPO / 'strata/flash-next/c1_runtime'
    tag = 'b70-c1-python-' + str(int(time.time()))
    base_tag = 'b70/flashnext-llamacpp-runtime:neo26.22'
    observed_base = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', base_tag], text=True, timeout=20).strip()
    require(observed_base == BASE_IMAGE, 'Local base tag does not identify the pinned runtime image')
    with (a.output / 'build.log').open('w') as log:
        run(['docker', 'build', '--pull=false', '--build-arg', 'BASE_IMAGE=' + base_tag, '--tag', tag, str(context)], timeout=900, stdout=log, stderr=subprocess.STDOUT)
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', tag], text=True, timeout=20).strip()
    name = 'b70-c1-python-intake-' + str(os.getpid())
    try:
        run(['docker', 'create', '--name', name, '--network', 'none', image, 'true'], stdout=subprocess.DEVNULL)
        run(['docker', 'cp', name + ':/opt/b70-c1-runtime.json', str(a.output / 'packages.json')])
    finally:
        run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL)
    packages = read(a.output / 'packages.json')
    require(packages.get('passed') is True and packages.get('gpu_libraries_unchanged') is True, 'Runtime dependency/library intake failed')
    # Capture the actual inherited math identity after the same setvars command
    # used at serve startup, with no GPU devices exposed.
    code = 'import os,json;print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(("STRATA_","SYCL_","ONEAPI_"))},sort_keys=True))'
    environment = json.loads(subprocess.check_output(['docker', 'run', '--rm', '--network', 'none', image,
        'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /opt/b70-c1-python/bin/python -c ' + shlex.quote(code)], text=True, timeout=60))
    report = {'schema': 1, 'passed': True, 'base_image': BASE_IMAGE, 'image': image, 'devices_exposed': False,
        'python_versions': packages['python_versions'], 'gpu_libraries_unchanged': True,
        'packages_sha256': sha(a.output / 'packages.json'), 'packages_path': str((a.output / 'packages.json').resolve()),
        'inherited_math_env': environment, 'recipe_files': {str(p.relative_to(context)): sha(p) for p in context.iterdir() if p.is_file()},
        'controller_sha256': sha(Path(__file__))}
    write(a.output / 'receipt.json', report)
    print(json.dumps({'runtime_receipt': str(a.output / 'receipt.json'), 'image': image}))


BASELINE_OFF_FLAGS = ('STRATA_BATCH_FULL_STATE_CHAIN', 'STRATA_BATCH_PUBLIC_PREFIX',
                      'STRATA_BATCH_FIDELITY_DIAG', 'STRATA_FIDELITY_DIAG',
                      'STRATA_LAYER0_Q8_DIAG', 'STRATA_PREFIX_DIAG', 'STRATA_PREFIX_LIFECYCLE_DIAG',
                      'STRATA_SLOT_OWNER_TRACE', 'STRATA_MIRROR_OWNER_TRACE', 'STRATA_PREFIX30')


def baseline_profile_gate(arguments, env):
    require(arguments.count('--batch') == 1 and arguments[arguments.index('--batch')+1] == '0', 'Baseline C1 requires batch0')
    require(all(env.get(name, '0') == '0' for name in BASELINE_OFF_FLAGS), 'Baseline C1 flags must remain absent or0; activated pilots require separate qualification')


def metadata_admission_gate(prepared):
    """Validate SDK/upload association before any model page or payload access."""
    generation = combined_generation_gate(Path(prepared['engine_receipt']).parent)
    require(prepared.get('combined_generation') == generation and prepared.get('launch_allowed') is True, 'Genuine integrated preparation required')
    require(prepared.get('controller_sha256') == sha(Path(__file__)), 'Preparation belongs to another controller')
    require(prepared.get('engine_receipt_sha256') == generation['engine_receipt_sha256'], 'Prepared new SDK receipt association differs')
    upload = prepared['upload_lifecycle']
    upload_gate(upload['path'], upload['oracle'], prepared['engine_receipt'], prepared['pack_receipt'])
    require(sha(prepared['runtime_receipt']) == prepared['runtime_receipt_sha256'], 'Prepared runtime receipt changed')
    runtime = read(prepared['runtime_receipt'])
    require(runtime.get('passed') is True and runtime.get('base_image') == BASE_IMAGE and runtime.get('gpu_libraries_unchanged') is True and sha(runtime['packages_path']) == runtime['packages_sha256'], 'Actual isolated runtime identity required before source access')
    return generation


def prepare(a):
    require(a.verify_model_shards and a.runtime_receipt and a.upload_lifecycle and a.oracle_receipt, 'New generation preparation requires genuine runtime, matching new whole390 and current all4 source identity prerequisites')
    generation = combined_generation_gate(a.engine_root)
    # Validate actual new whole390 BEFORE any selected model payload scan.
    upload_gate(a.upload_lifecycle, a.oracle_receipt, a.engine_root.resolve() / 'receipt.json', INTAKE)
    runtime_receipt = read(a.runtime_receipt) if a.runtime_receipt else None
    if runtime_receipt:
        require(runtime_receipt.get('passed') and runtime_receipt.get('base_image') == BASE_IMAGE and runtime_receipt.get('gpu_libraries_unchanged'), 'Isolated runtime identity invalid')
        require(sha(runtime_receipt['packages_path']) == runtime_receipt['packages_sha256'], 'Runtime package census changed')
    baseline_profile_gate(['--batch', '0'], runtime_receipt['inherited_math_env'])
    a.output.mkdir(parents=True, exist_ok=False)
    profile = PROFILES[a.profile]
    engine = a.engine_root.resolve()
    engine_receipt = engine / 'receipt.json'
    build = read(engine_receipt)
    require(build.get('build_rc') == 0 and build.get('external_source_unchanged') and build.get('plan_snapshot_unchanged') and build.get('image') == BASE_IMAGE, 'Engine source/build identity incomplete')
    executable = engine / 'build/strata'
    require(sha(executable) == build['binary_sha256'][str(executable)], 'Engine executable differs from source build')
    for name, expected in build['patched_source_sha256'].items():
        require(sha(engine / 'source' / name) == expected, 'Patched source changed: ' + name)
    intake = read(INTAKE)
    require(intake.get('metadata_complete') and not intake['RESULT']['unexpected_inexact_conversions'], 'Pack intake metadata/conversions incomplete')
    for relative, identity in intake['RESULT']['files'].items():
        require(sha(PACK / relative) == identity['sha256'], 'Pack changed: ' + relative)
    lock_path = REPO / 'strata/flash-next/model-lock.json'
    lock = read(lock_path)
    shards = []
    for file in lock['files']:
        if not file['path'].startswith('UD-Q4_K_XL/'):
            continue
        path = REPO / lock['destination'] / file['path']
        require(path.stat().st_size == file['size'], 'GGUF size differs: ' + str(path))
        before_hash = stat_signature(path)
        actual = sha(path) if a.verify_model_shards or len(shards) == 0 else None
        require(stat_signature(path) == before_hash, 'Selected source changed during full-shard preparation scan: ' + str(path))
        if actual:
            require(actual == file['sha256'], 'GGUF hash differs: ' + str(path))
        shards.append({'path': str(path), 'sha256': file['sha256'], 'verified_sha256': actual, 'stat': stat_signature(path)})
    require(len(shards) == 4, 'Four selected GGUF shards required')
    sentinel = original_page_sentinel(shards)
    registry = REPO / 'evals/configs/models.yaml'
    require('served_model_id: ' + profile['alias'] in registry.read_text(), 'Detailed alias is absent from eval registry')
    launch_plan = read(REPO / 'strata/flash-next/native-hc-launch-plan.json')
    arguments = list(launch_plan['engine_arguments_common'])
    for option, value in [('--max-context', profile['context']), ('--prefill', profile['prefill']), ('--ple-row-cache', profile['ple_rows'])]:
        arguments[arguments.index(option)+1] = str(value)
    arguments += profile['extra']
    require('--mtp' not in arguments and '--no-ple' not in arguments and arguments[arguments.index('--batch')+1] == '0', 'C1/no-MTP/full-model argument contract violated')
    env = dict(runtime_receipt['inherited_math_env']) if runtime_receipt else {}
    env.update(launch_plan['runtime_environment_common'])
    env['ZE_AFFINITY_MASK'] = ','.join(map(str, profile['cards']))
    env.update(profile.get('env', {}))
    env.update({'STRATA_DEBUG': '1', 'STRATA_REQUEST_LINES': '1'})
    baseline_profile_gate(arguments, env)
    segmented_profile_gate(profile, build, arguments, env)
    manifest = {'schema': 1, 'primary_model_name': PRIMARY, 'research_alias': profile['alias'],
        'model': {'repo': lock['repo'], 'revision': lock['revision'], 'artifact': 'UD-Q4_K_XL',
                  'tokenizer_gguf': '/model/' + next(f['path'] for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')), 'tokenizer_gguf_sha256': shards[0]['sha256']},
        'tokenizer_files': {name: sha(PACK / 'tokenizer' / name) for name in FILES},
        'runtime': {'expert_profile_sha256': sha(engine / 'source/data/expert-profile.bin'), 'exe_sha256': sha(executable), 'args': arguments,
            'env': {k: v for k, v in sorted(env.items()) if k.startswith(('STRATA_', 'SYCL_', 'ONEAPI_'))},
            'python_versions': runtime_receipt['python_versions'] if runtime_receipt else {},
            'python_sources': {name: sha(engine / 'source' / name) for name in PYTHON_SOURCES}}}
    write(a.output / 'artifact-identity.json', manifest)
    cfg = {'exe': '/build/strata', 'args': arguments, 'cwd': '/src', 'tokenizer': '/pack/tokenizer',
        'model_name': PRIMARY, 'aliases': [profile['alias']], 'artifact_identity_manifest': '/results/artifact-identity.json',
        'host': '127.0.0.1', 'port': a.port, 'backend': 'sycl', 'env': env, 'log': '/results/server-engine.log',
        'parallel': 1, 'repeat_stop_tokens': 0, 'reasoning_close_retry': False, 'reasoning_budget_tokens': 0,
        'engine_silence_s': 180, 'sampling': {'temperature': 0, 'seed': 1}}
    write(a.output / 'server-config.json', cfg)
    lifecycle = upload_gate(a.upload_lifecycle, a.oracle_receipt, engine_receipt, INTAKE) if a.upload_lifecycle else None
    report = {'schema': 1, 'status': 'PREPARED unqualified C1; no GPU launch performed', 'launch_allowed': bool(runtime_receipt and lifecycle and a.verify_model_shards),
        'profile': a.profile, 'cards': profile['cards'], 'segmented_stage_mirrors': bool(profile.get('segmented')),
        'source_upload_required_coverage': 'whole390', 'combined_generation': generation, 'runtime_python_source_count': 6, 'ordinary_payload_readback_qualified': False, 'engine_receipt': str(engine_receipt), 'engine_receipt_sha256': sha(engine_receipt),
        'pack_receipt': str(INTAKE), 'pack_receipt_sha256': sha(INTAKE), 'model_lock_sha256': sha(lock_path), 'model_shards': shards, 'source_page_sentinel': sentinel,
        'executable': str(executable), 'executable_sha256': sha(executable), 'source': str(engine / 'source'), 'pack': str(PACK),
        'runtime_receipt': str(a.runtime_receipt.resolve()) if a.runtime_receipt else None,
        'runtime_receipt_sha256': sha(a.runtime_receipt) if a.runtime_receipt else None, 'runtime': runtime_receipt,
        'upload_lifecycle': lifecycle, 'registry_sha256': sha(registry), 'alias': profile['alias'], 'port': a.port,
        'artifact_manifest_sha256': sha(a.output / 'artifact-identity.json'), 'config_sha256': sha(a.output / 'server-config.json'),
        'trace_contract_sha256': sha(REPO / 'strata/flash-next/c1_trace_contract.py'), 'controller_sha256': sha(Path(__file__)), 'trace_sha256': sha(REPO / 'strata/flash-next/c1_api_trace.py'),
        'full_logits_supported_over_http': False, 'state_hash_enabled': False, 'prefix_state_qualified': False,
        'one_card_required_before_pair': True}
    write(a.output / 'prepared.json', report)
    print(json.dumps({'prepared': str(a.output / 'prepared.json'), 'launch_allowed': report['launch_allowed']}))


def validate_prepared(directory):
    m = read(directory / 'prepared.json')
    require(m.get('combined_generation') == combined_generation_gate(Path(m['engine_receipt']).parent), 'Combined generation source chain changed')
    require(m.get('runtime_python_source_count') == 6 and read(directory / 'artifact-identity.json')['runtime']['python_sources'] == PYTHON_SOURCE_SHA, 'Manifest must bind actual6 runtime Python files')
    require(m['launch_allowed'], 'Preparation has unresolved runtime/model/upload lifecycle gates')
    require(m.get('trace_contract_sha256') == sha(REPO / 'strata/flash-next/c1_trace_contract.py'), 'C1 completion contract changed or older prepared generation')
    require(m['controller_sha256'] == sha(Path(__file__)) and m['trace_sha256'] == sha(REPO / 'strata/flash-next/c1_api_trace.py'), 'Qualification controller/tracer changed')
    require(sha(m['engine_receipt']) == m['engine_receipt_sha256'] and sha(m['executable']) == m['executable_sha256'], 'Engine build generation changed')
    require(sha(m['pack_receipt']) == m['pack_receipt_sha256'], 'Pack receipt changed')
    require(sha(directory / 'artifact-identity.json') == m['artifact_manifest_sha256'] and sha(directory / 'server-config.json') == m['config_sha256'], 'Launch configuration changed')
    require(sha(m['runtime_receipt']) == m['runtime_receipt_sha256'], 'Runtime receipt changed')
    require(sha(m['runtime']['packages_path']) == m['runtime']['packages_sha256'], 'Runtime package/library census changed')
    for row in m['model_shards']:
        require(row['verified_sha256'] == row['sha256'] and stat_signature(row['path']) == row['stat'], 'Selected model was not fully hashed or changed since hashing')
    require(original_page_sentinel(m['model_shards']) == m['source_page_sentinel'], 'Original page sentinel contract changed')
    for name, expected in read(m['engine_receipt'])['patched_source_sha256'].items():
        require(sha(Path(m['source']) / name) == expected, 'Frozen source changed: ' + name)
    for name, identity in read(m['pack_receipt'])['RESULT']['files'].items():
        require(sha(Path(m['pack']) / name) == identity['sha256'], 'Pack file changed: ' + name)
    upload = m['upload_lifecycle']
    require(sha(upload['path']) == upload['sha256'] and sha(upload['oracle']) == upload['oracle_sha256'], 'Upload lifecycle evidence changed')
    require(upload['logical_free_parser_sha256'] == sha(REPO / 'strata/flash-next/parse_usm_logical_free_trace.py'), 'Logical-free evidence parser changed')
    upload_gate(upload['path'], upload['oracle'], m['engine_receipt'], m['pack_receipt'])
    cfg = read(directory / 'server-config.json')
    baseline_profile_gate(cfg['args'], cfg['env'])
    segmented_profile_gate(PROFILES[m['profile']], read(m['engine_receipt']), cfg['args'], cfg['env'])
    return m


def leased(cards):
    for card in cards:
        require(os.path.samefile('/proc/self/fd/' + str(8+card), '/mnt/vm_8tb/b70/gpu.lock.' + str(card)), 'Parent GPU lease descriptor is missing')


def inspected(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name], text=True, timeout=20))[0]


@contextlib.contextmanager
def stop_lock(directory, timeout=120):
    # The same prepared-run inode lock covers request publication, inspection,
    # stop, terminal state, removal and receipt publication across processes.
    with (directory / 'stop.lock').open('a') as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, 'Prepared-run stop lock deadline exceeded')
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def absent(name):
    return not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'], timeout=20).strip()


def native_close_proven(directory):
    path = directory / 'engine-token-trace.jsonl'
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    begins = {row['engine_pid'] for row in rows if row['kind'] == 'engine_begin'}
    closes = [row for row in rows if row['kind'] == 'engine_close']
    # A ready engine may be stopped before its first GEN. Its actual traced
    # child close can prove teardown, while a missing screen still cannot
    # qualify serving. A null/unloaded child is never sufficient evidence.
    pids = begins or {row['engine_pid'] for row in closes if type(row.get('engine_pid')) is int and row['engine_pid'] > 0}
    return bool(pids) and all(any(row['engine_pid'] == pid and row.get('clean_exit') is True and row.get('exit_code') == 0 and row.get('error') is None for row in closes) for pid in pids)


def stop_receipt_checked(directory):
    prior = read(directory / 'stop.json')
    launch_state = read(directory / 'launch.json')
    require(prior['container'] == launch_state['container'] and prior['prepared_sha256'] == launch_state['prepared_sha256'], 'Stop receipt belongs to another run')
    require(prior['removed'] and absent(prior['container']), 'Previously removed container is live again')
    return prior


def poll_owned_run(directory):
    with stop_lock(directory):
        if (directory / 'stop.json').exists():
            return {'stopped': True, 'receipt': stop_receipt_checked(directory)}
        if (directory / 'stop-request.json').exists():
            # A request alone is not proof of termination. The caller must
            # resume bounded cleanup and obtain actual terminal/native evidence.
            return {'stop_requested': True}
        state = read(directory / 'launch.json')
        info = inspected(state['container'])
        require(info['Config']['Labels'].get('b70.c1.prepared') == state['prepared_sha256'], 'Owned container label changed')
        return {'state': info['State']}


def owned_stop(directory, grace=75):
    with stop_lock(directory):
        if (directory / 'stop.json').exists():
            return stop_receipt_checked(directory)
        state = read(directory / 'launch.json')
        name = state['container']
        info = inspected(name)
        require(info['Config']['Labels'].get('b70.c1.prepared') == state['prepared_sha256'], 'Refusing to stop a container not owned by this prepared run')
        write(directory / 'stop-request.json', {'container': name, 'prepared_sha256': state['prepared_sha256'],
            'requested_epoch': time.time(), 'requester_pid': os.getpid()})
        if info['State']['Running']:
            run(['docker', 'stop', '--time', str(grace), name], timeout=grace+30, stdout=subprocess.DEVNULL)
        terminal = inspected(name)['State']
        require(not terminal['Running'], 'Owned container remained live after bounded stop')
        with (directory / 'server-container.log').open('w') as log:
            run(['docker', 'logs', name], stdout=log, stderr=subprocess.STDOUT)
        native_error = None
        try:
            native_clean = native_close_proven(directory)
        except (OSError, ValueError, KeyError) as error:
            native_clean = False
            native_error = str(error)
        run(['docker', 'rm', name], stdout=subprocess.DEVNULL)
        require(absent(name), 'Container removal could not be verified')
        result = {'container': name, 'prepared_sha256': state['prepared_sha256'], 'terminal': terminal,
            'removed': True, 'finished_epoch': time.time(),
            'clean_exit': terminal['ExitCode'] == 0 and not terminal.get('OOMKilled'),
            'engine_clean_exit_proven': native_clean, 'engine_close_trace_error': native_error, 'post_health_proven': False}
        write(directory / 'stop.json', result)
        return result


def launch(a):
    directory = a.prepared.resolve()
    m = validate_prepared(directory)
    leased(m['cards'])
    require(socket.socket().connect_ex(('127.0.0.1', m['port'])) != 0, 'API port is already listening')
    require(sha(Path(m['source']) / 'data/expert-profile.bin') == read(directory / 'artifact-identity.json')['runtime']['expert_profile_sha256'], 'Expert placement profile changed')
    health = read(a.pre_health)
    require(health.get('passed') is True and set(m['cards']) <= set(health.get('cards', [])), 'Parent pre-health receipt does not cover leased cards')
    require(0 <= time.time()-health.get('finished_epoch', 0) <= 300, 'Parent pre-health receipt is not fresh')
    require(health.get('files'), 'Parent pre-health receipt has no command/log artifacts')
    for row in health['files']:
        require(sha(row['path']) == row['sha256'], 'Pre-health command/log changed')
    if len(m['cards']) > 1:
        require(a.one_card_receipt, 'Two-card serving requires completed one-card qualification')
        c = read(a.one_card_receipt)
        require(c.get('passed') is True and c.get('profile') == PROFILES[m['profile']].get('one_card_profile', 'one-card') and c.get('engine_receipt_sha256') == m['engine_receipt_sha256'] and c.get('teardown_passed') is True and c.get('post_health_passed') is True, 'Matched one-card qualification/lifecycle is incomplete')
    name = 'b70-strata-c1-' + str(os.getpid()) + '-' + str(int(time.time()))
    command = ['docker', 'run', '-d', '--name', name, '--label', 'b70.c1.prepared=' + sha(directory / 'prepared.json'),
        '--network', 'host', '--device', '/dev/dri', '--user', '1000:1000', '--memory', '105g', '--memory-swap', '105g',
        '--group-add', str(os.stat('/dev/dri/renderD128').st_gid), '--group-add', str(os.stat('/dev/dri/card0').st_gid),
        '-e', 'B70_C1_TRACE=/results/engine-token-trace.jsonl', '-v', str(Path(m['executable']).parent) + ':/build:ro',
        '-v', m['source'] + ':/src:ro', '-v', m['pack'] + ':/pack:ro',
        '-v', str(REPO / read(REPO / 'strata/flash-next/model-lock.json')['destination']) + ':/model:ro',
        '-v', str(directory) + ':/results', '-v', str(REPO / 'strata/flash-next/c1_api_trace.py') + ':/controller/trace.py:ro',
        '-v', str(REPO / 'strata/flash-next/c1_trace_contract.py') + ':/controller/c1_trace_contract.py:ro']
    command += [m['runtime']['image'], 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /opt/b70-c1-python/bin/python /controller/trace.py --engine strata --config /results/server-config.json --host 127.0.0.1 --port ' + str(m['port'])]
    write(directory / 'launch.command.json', command)
    require(not (directory / 'launch.json').exists(), 'Prepared run was already launched; create a new result directory')
    write(directory / 'launch.json', {'container': name, 'prepared_sha256': sha(directory / 'prepared.json'), 'started_epoch': time.time(), 'owner_pid': os.getpid(), 'pre_health_sha256': sha(a.pre_health), 'ready': False})
    stopping = False
    def stop_handler(*unused):
        nonlocal stopping
        stopping = True
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, stop_handler)
    start = time.monotonic()
    try:
        run(command, stdout=subprocess.DEVNULL)
        ready = False
        while not stopping and time.monotonic()-start < a.max_runtime:
            original_page_sentinel(m['model_shards'])
            observation = poll_owned_run(directory)
            if observation.get('stop_requested'):
                owned_stop(directory)
                observation = poll_owned_run(directory)
            if observation.get('stopped'):
                require(observation['receipt']['clean_exit'] and observation['receipt']['engine_clean_exit_proven'], 'Externally requested stop lacked clean API/native termination')
                break
            state = observation['state']
            require(state['Running'] and not state.get('OOMKilled'), 'Serving container exited or was OOM killed')
            if not ready:
                require(time.monotonic()-start <= a.ready_deadline, 'Engine/API readiness deadline exceeded')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:' + str(m['port']) + '/v1/models', timeout=2) as r:
                        models = json.load(r)
                    require([x['id'] for x in models['data']] == [PRIMARY, m['alias']], 'Live served IDs/order differ')
                    require(all(x['meta']['artifact_identity']['artifact_identity_sha256'] == m['artifact_manifest_sha256'] for x in models['data']), 'Live API artifact identity differs')
                    write(directory / 'live-models.json', models)
                    s = read(directory / 'launch.json');s['ready'] = True;write(directory / 'launch.json', s);ready = True
                except (OSError, TimeoutError):
                    pass
            time.sleep(1)
        require(ready, 'Serving run ended before readiness')
    finally:
        # Always enter the serialized stop path. An unlocked absence check can
        # observe removal before another stopper publishes its terminal receipt.
        stopped_run = owned_stop(directory)
        require(stopped_run['clean_exit'] and stopped_run['engine_clean_exit_proven'], 'Serving cleanup lacked clean API/native exit')
    write(directory / 'launch-supervisor.json', {'passed': True, 'container': name, 'actual_native_exit_checked': True, 'finished_epoch': time.time()})
    print(json.dumps({'run': str(directory), 'stopped': True, 'parent_post_health_required': True}))



def request_json(url, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def screen(a):
    directory = a.prepared.resolve()
    m = validate_prepared(directory)
    require(read(directory / 'launch.json').get('ready') and not (directory / 'stop.json').exists(), 'Screen requires a currently supervised ready server')
    require(not (directory / 'screen.json').exists(), 'Screen already executed; use a new prepared run')
    base = 'http://127.0.0.1:' + str(m['port'])
    models = request_json(base + '/v1/models')
    require([row['id'] for row in models['data']] == [PRIMARY, m['alias']], 'Live model IDs/order differ before screen')
    cases = [
        ('marker', [{'role': 'user', 'content': 'Reply with exactly B70_READY and nothing else.'}], 'B70_READY'),
        ('arithmetic', [{'role': 'user', 'content': 'What is 19 plus 23? Reply only with the decimal number.'}], '42'),
        ('history', [{'role': 'user', 'content': 'The marker is NATIVE_31415.'},
            {'role': 'assistant', 'content': 'I recorded the marker.'},
            {'role': 'user', 'content': 'Repeat the marker exactly and nothing else.'}], 'NATIVE_31415'),
        ('fact', [{'role': 'user', 'content': 'What is the capital of France? Reply only with the city name.'}], 'Paris'),
    ]
    cases += [('marker-repeat', cases[0][1], cases[0][2]), ('history-repeat', cases[2][1], cases[2][2])]
    results = []
    for index, (name, messages, expected) in enumerate(cases):
        body = {'model': PRIMARY if index % 2 == 0 else m['alias'], 'messages': messages,
            'max_tokens': 64, 'temperature': 0, 'seed': 1, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_budget_tokens': 0}
        write(directory / ('request-' + str(index) + '.json'), body)
        before_sentinel = original_page_sentinel(m['model_shards'])
        started = time.time()
        response = request_json(base + '/v1/chat/completions', body, a.request_timeout)
        after_sentinel = original_page_sentinel(m['model_shards'])
        write(directory / ('response-' + str(index) + '.json'), response)
        content = response['choices'][0]['message'].get('content') or ''
        results.append({'case': name, 'request': body, 'response': response, 'elapsed_s': time.time()-started,
                        'expected': expected, 'content': content, 'coherent': content.strip() == expected,
                        'source_page_sentinel_before': before_sentinel, 'source_page_sentinel_after': after_sentinel})
    trace = [json.loads(line) for line in (directory / 'engine-token-trace.jsonl').read_text().splitlines()]
    begins = [r for r in trace if r['kind'] == 'engine_begin']
    ends = [r for r in trace if r['kind'] == 'engine_end']
    transport = len(begins) == len(ends) == 6 and {r['call'] for r in begins} == {r['call'] for r in ends}
    transport = transport and all(r['rendered_matches_submitted'] and r['embeddings'] is None for r in begins)
    transport = transport and all(trace_end_accepted(r) for r in ends)
    one_engine = len({r['engine_pid'] for r in begins + ends}) == 1
    repeat = transport and begins[0]['submitted_ids'] == begins[4]['submitted_ids'] and begins[2]['submitted_ids'] == begins[5]['submitted_ids'] and ends[0]['generated_ids'] == ends[4]['generated_ids'] and ends[2]['generated_ids'] == ends[5]['generated_ids']
    for first, again in [(0, 4), (2, 5)]:
        repeat = repeat and results[first]['response']['choices'] == results[again]['response']['choices']
    coherent = all(r['coherent'] for r in results)
    report = {'scope': 'C1 bounded API coherence/identity/consumed-token/repeat screen only', 'profile': m['profile'],
        'engine_receipt_sha256': m['engine_receipt_sha256'], 'prepared_sha256': sha(directory / 'prepared.json'),
        'api_identity_checked': True, 'token_transport_and_consumption_passed': transport, 'one_engine': one_engine,
        'repeat_passed': repeat, 'coherence_passed': coherent, 'cases': results, 'trace_sha256': sha(directory / 'engine-token-trace.jsonl'),
        'trace_rows': len(trace), 'full_logits_parity_qualified': False, 'prefix_state_qualified': False, 'concurrent_serving_qualified': False,
        'passed': bool(transport and one_engine and repeat and coherent), 'finished_epoch': time.time()}
    write(directory / 'screen.json', report)
    print(json.dumps({'screen': str(directory / 'screen.json'), 'passed': report['passed']}))
    require(report['passed'], 'C1 API/token/coherence/repeat screen failed; receipt preserves evidence')


def finalize(a):
    directory = a.prepared.resolve()
    m, test, stop, health = read(directory / 'prepared.json'), read(directory / 'screen.json'), read(directory / 'stop.json'), read(a.post_health)
    require(original_page_sentinel(m['model_shards']) == m['source_page_sentinel'], 'Known source page sentinel changed after teardown')
    supervisor = read(directory / 'launch-supervisor-exit.json')
    require(type(supervisor.get('return_code')) is int and supervisor['return_code'] == 0 and supervisor.get('passed') is True, 'Parent did not observe a zero-exit launch supervisor')
    require(read(directory / 'launch-supervisor.json').get('passed') is True, 'Launch supervisor did not publish successful owned-stop evidence')
    require(stop.get('removed') and stop.get('clean_exit') and stop.get('engine_clean_exit_proven'), 'Server/container did not stop cleanly')
    require(health.get('passed') is True and set(m['cards']) <= set(health.get('cards', [])) and health.get('finished_epoch', 0) >= stop['finished_epoch'], 'Matched post-health is absent or predates teardown')
    require(health.get('files'), 'Post-health has no command/log artifacts')
    for row in health['files']:
        require(sha(row['path']) == row['sha256'], 'Post-health command/log changed')
    trace = [json.loads(line) for line in (directory / 'engine-token-trace.jsonl').read_text().splitlines()]
    raw_lines = (directory / 'engine-token-trace.jsonl').read_bytes().splitlines(keepends=True)
    require(hashlib.sha256(b''.join(raw_lines[:test['trace_rows']])).hexdigest() == test['trace_sha256'], 'Screen token evidence changed before teardown')
    require(all(r['kind'] == 'engine_close' for r in trace[test['trace_rows']:]), 'Unscreened engine requests occurred after C1 qualification')
    pids = {r['engine_pid'] for r in trace if r['kind'] == 'engine_begin'}
    closes = [r for r in trace if r['kind'] == 'engine_close']
    destroyed = bool(pids) and all(any(c['engine_pid'] == pid and c['clean_exit'] and c['error'] is None for c in closes) for pid in pids)
    report = {'scope': 'Completed C1 screen and lifecycle; full model numerical/state/concurrent qualification outstanding',
        'profile': m['profile'], 'engine_receipt_sha256': m['engine_receipt_sha256'],
        'screen_sha256': sha(directory / 'screen.json'), 'stop_sha256': sha(directory / 'stop.json'),
        'launch_supervisor_exit_sha256': sha(directory / 'launch-supervisor-exit.json'), 'post_health_sha256': sha(a.post_health), 'teardown_passed': destroyed, 'post_health_passed': True,
        'passed': bool(test['passed'] and destroyed), 'full_model_fidelity_qualified': False,
        'concurrent_serving_qualified': False, 'shelf_promotion_allowed': False}
    write(directory / 'qualification.json', report)
    print(json.dumps({'qualification': str(directory / 'qualification.json'), 'passed': report['passed']}))
    require(report['passed'], 'C1 screen or actual engine teardown failed')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    r = sub.add_parser('prepare-runtime');r.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('prepare');p.add_argument('--output', type=Path, required=True);p.add_argument('--profile', choices=PROFILES, required=True)
    p.add_argument('--engine-root', type=Path, default=ENGINE);p.add_argument('--runtime-receipt', type=Path);p.add_argument('--upload-lifecycle', type=Path);p.add_argument('--oracle-receipt', type=Path)
    p.add_argument('--verify-model-shards', action='store_true');p.add_argument('--port', type=int, default=18082)
    l = sub.add_parser('launch');l.add_argument('--prepared', type=Path, required=True);l.add_argument('--pre-health', type=Path, required=True);l.add_argument('--one-card-receipt', type=Path)
    l.add_argument('--ready-deadline', type=int, default=600);l.add_argument('--max-runtime', type=int, default=1200)
    s = sub.add_parser('stop');s.add_argument('--prepared', type=Path, required=True)
    c = sub.add_parser('screen');c.add_argument('--prepared', type=Path, required=True);c.add_argument('--request-timeout', type=int, default=180)
    f = sub.add_parser('finalize');f.add_argument('--prepared', type=Path, required=True);f.add_argument('--post-health', type=Path, required=True)
    a = ap.parse_args()
    if a.action == 'prepare-runtime':runtime(a)
    elif a.action == 'prepare':
        require(not a.upload_lifecycle or a.oracle_receipt, 'Upload lifecycle requires its oracle build receipt')
        prepare(a)
    elif a.action == 'launch':launch(a)
    elif a.action == 'screen':screen(a)
    elif a.action == 'finalize':finalize(a)
    else:
        stopped_run = owned_stop(a.prepared.resolve())
        require(stopped_run['clean_exit'] and stopped_run['engine_clean_exit_proven'], 'Owned stop completed without a clean actual native/API exit')


if __name__ == '__main__':
    main()
