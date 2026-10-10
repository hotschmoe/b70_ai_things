# NEW mandatory immutable logical-byte port of first49_pack49_v1.py
from upload_logical_port_identity_v1 import original_producer_file, original_module_file
'One declared failed harness40 job: strict readonly cacheOFF raw49 recovery.\nOriginal failed parent and PIN0 request remain unchanged and unqualified.\n'
import argparse, json, hashlib, math
from pathlib import Path
import audit_batch_logical_ports_v1 as audit
import batch40_manifest_logical_ports_v1 as origin
from batch40_runtime_evidence_v1 import child_binding, journal_binding
from batch_proofs_logical_ports_v1 import read, write, sha, require, artifact_bindings, validate_artifacts
from batch_numerical_prefixes_v2 import compare_all
from extract_serial_cacheoff_numerical_v1 import extract
from run_batch_serial_source37_v1 import config, command_recipe
ROOT = origin.ROOT
HERE = origin.HERE
SOURCE_PLAN = HERE / 'batch-serial-source37-source-plan-v1.json'
DECLARED_ROOT = Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/batch40-pair-native2-serial0-run-v1')

def tree_binding(root):
    result = {}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'Original evidence symlink rejected')
        if p.is_file():
            s = p.stat()
            result[str(p.relative_to(root))] = {'sha256': sha(p), 'stat5': [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]}
    return result

def known_failure(parent, terminal):
    require(parent['passed'] is False and parent['parent_generation'] == 40 and (parent['child_return_code'] == 1) and (parent['errors'] == ['BATCH_NUMERICAL child failed 1']) and (parent['interrupted'] is False) and (parent['forced_cleanup'] is False) and (parent['owned_containers_terminal'] is True), 'Only exact failed V40 child permitted')
    require(terminal['passed'] is False and terminal['error'] == 'ValueError: Noncancelled live chain was not reusable' and (terminal['engine_rc'] == 0) and (terminal['removed'] is True) and (terminal['state']['ExitCode'] == 0) and (not terminal['state']['Running']) and (not terminal['state']['OOMKilled']), 'Only cacheON-extractor rejection with normal owned engine terminal permitted')
    trace = terminal['failure_traceback']
    require('run_batch_serial_controls_v40.py' in trace and 'serial_prefix_qualification_v6.py' in trace and (trace.count('ValueError: Noncancelled live chain was not reusable') == 1), 'Original known extractor traceback required')

def health_binding(root, parent):
    for stage in ('pre', 'post'):
        health = read(root / (stage + '-health.json'))
        require(health['passed'] is True and health['cards'] == [0, 1] and (health['health_image'] == 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'), 'Exact strict pair/percard health required')
        commands = [[str(ROOT / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', health['health_image']], [str(ROOT / 'bin/xpu-collective-health'), '--img', health['health_image'], '--p2p', '0', '--timeout', '180']]
        require(len(health['files']) == 2 and parent[stage + '_health_passed'] is True and (health['finished_epoch'] == parent[stage + '_health_finished_epoch']), 'Actual health rows/epoch association differs')
        for row, argv, label in zip(health['files'], commands, ('strict', 'compiled-pair')):
            log = root / (stage + '-' + label + '.log')
            cmd = root / (stage + '-' + label + '.command.json')
            require(row['return_code'] == 0 and row['error'] is None and (row['path'] == str(log)) and (row['command'] == argv == read(cmd)) and (row['command_file_sha256'] == sha(cmd)) and (row['sha256'] == row['stdout_sha256'] == sha(log)) and (parent['started_epoch'] <= health['started_epoch'] <= row['started_epoch'] <= row['finished_epoch'] <= health['finished_epoch']), 'Actual health original command/log/chronology differs')
    require(read(root / 'post-health.json')['started_epoch'] >= parent['child_terminal_epoch'], 'Actual posthealth before child terminal')

def finalized_binding(root=DECLARED_ROOT, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    root = Path(root).resolve()
    require(root == DECLARED_ROOT, 'Exactly named failed first-job evidence only')
    source = read(SOURCE_PLAN)
    for name, want in source['files'].items():
        require(sha(ROOT / name) == want, 'Frozen adjudicator dependency changed ' + name)
    before = tree_binding(root)
    for name, want in source['historical_failed_first_job'].items():
        require(sha(root / name) == want, 'Hardpinned failed original artifact changed ' + name)
    parent = read(root / 'parent-qualification.json')
    plan = read(root / 'input-plan.snapshot.json')
    directory = root / 'child/serial-0'
    terminal = read(directory / 'result.json')
    known_failure(parent, terminal)
    origin.manifest_binding(plan, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(plan['kind'] == 'serial' and plan['lane'] == 'source37' and (plan['group_index'] == 0) and (plan['cards'] == [0, 1]) and (parent['expected_stage_ranges'] == plan['stage_ranges']) and (parent['wrapper_sha256'] == sha(HERE / 'qualify_batch_numerical_v40.py')) and (parent['controller_sha256'] == sha(HERE / 'batch_numerical_execution_v40.py')), 'Exact consumed failed source37 configuration differs')
    child = {'plan_sha256': parent['plan_sha256'], 'finished_epoch': parent['child_terminal_epoch'], 'artifact_bindings': artifact_bindings(root / 'child')}
    require('post_model_identity' not in parent and 'source_guard_generation' not in parent, 'Exact failed-before-finalizable source schema required')
    view = dict(parent)
    view['post_model_identity'] = {'path': str(root / 'post-model-identity.json'), 'sha256': sha(root / 'post-model-identity.json')}
    child_binding(root, view, plan, child)
    health_binding(root, parent)
    require(parent['kernel_fault_gate_passed'] is True, 'Actual kernel fault gate absent')
    journal_binding(root, parent, 'pre')
    journal_binding(root, parent, 'post')
    audit.final_source_join(root, view, plan, child, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(read(directory / 'command.json') == command_recipe(plan, root / 'child', 0, parent['child_pid']), 'Exact historical actual serial command differs')
    collector = Path(plan['batch_parent'])
    audit.parent_arm(collector, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    jobs = read(collector / 'child/serial-jobs.json')['jobs']
    job = jobs[0]
    rawpath = directory / ('serial-' + str(job['rid']) + '-' + job['role'] + '.json')
    raw = read(rawpath)
    require(raw['ids'] == job['ids'] and raw['label'] == 'serial-' + str(job['rid']) + '-' + job['role'] and (raw['pin'] == 0), 'Declared first consumed-prefix historical PIN0 job required')
    require(not (directory / 'requests.json').exists() and (not (root / 'child/report.json').exists()) and (len(list(directory.glob('serial-*.json'))) == 1), 'Only failed-before-first-extraction one-job evidence permitted')
    args, env = config(plan)
    meta = extract(raw, directory / 'captures', args, env, [tuple(x) for x in plan['stage_ranges']], allow_legacy_pin_zero=True)
    serial = {(job['rid'], job['role']): meta}
    vectors = {((job['rid'], job['role']), -1): meta['logits'][0]['path']}
    for row in meta['residuals']:
        vectors[(job['rid'], job['role']), int(row['layer'])] = row['path']
    require(len(vectors) == 49 and {k[1] for k in vectors} == set(range(-1, 48)), 'Exact first-job complete48/head required')
    batch = audit.vectors(collector)
    comparison = compare_all({k: v for k, v in batch.items() if k in vectors}, vectors)
    require(comparison['passed'], 'Actual first-job full49 bitwise comparison differs')
    validate_artifacts(root / 'child', child['artifact_bindings'])
    audit.final_source_join(root, view, plan, child, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(tree_binding(root) == before, 'Original evidence changed during read-only work')
    return {'schema': 1, 'passed': True, 'source_plan_sha256': sha(SOURCE_PLAN), 'root': str(root), 'original_parent_sha256': sha(root / 'parent-qualification.json'), 'original_parent_passed': False, 'original_records_modified': False, 'original_tree': before, 'observed_parent_terminal_boundary': child['finished_epoch'], 'original_child_report_existed': False, 'supplemental_canonical_post_identity_binding': view['post_model_identity'], 'original_parent_post_identity_binding_existed': False, 'job': job, 'raw_sha256': sha(rawpath), 'raw_meta': meta, 'actual_first_job_full49': comparison, 'engine_receipt_sha256': plan['engine_receipt_sha256'], 'collector_parent_sha256': sha(collector / 'parent-qualification.json'), 'historical_PIN0_exception_used': True, 'new_absent_PIN_scope_qualified': False, 'complete_serial_group_qualified': False, 'cache_qualification_granted': False, 'full_model_math_qualified': False, 'latency_qualified': False}
