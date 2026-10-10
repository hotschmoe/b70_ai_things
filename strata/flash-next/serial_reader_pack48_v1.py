# NEW explicit pack epoch admission port of validate_batch_serial_source37_v3.py
"""Recompute source-qualified actual 2/4/6 scoped native/API/serial proof joins."""
import argparse, json, os, re, sys, time
from pathlib import Path
from batch_proofs_pack48_v1 import require, sha, read, write, PLAN_SHA, lane_contract, off_on_histories, owner_proofs, validate_artifacts, confined_file, source_observers_off, ROOT
from batch_numerical_prefixes_v2 import compare_all
from serial37_runtime_evidence_v3 import child_binding, journal_binding
HERE = Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/validate_batch_serial_source37_v3.py').resolve().parent

def snapshot_plan_join(directory, parent, plan, child):
    directory = Path(directory).resolve()
    input_path = directory / 'input-plan.snapshot.json'
    child_path = directory / 'child/plan.snapshot.json'
    external_path = Path(parent['plan'])
    require(read(input_path) == plan == read(child_path) == read(external_path), 'External/input/child snapshot plan content differs')
    require(parent['plan_sha256'] == child['plan_sha256'] == sha(input_path) == sha(child_path) == sha(external_path), 'Complete external/input/child plan SHA association differs')
    return parent['plan_sha256']

def final_source_join(directory, parent, plan, child, *, pack_epoch):
    from batch_proofs_pack48_v1 import genuine_baseline, engine_binding, ROOT
    from batch40_manifest_pack48_v1 import verify_model_identity
    from source_page_watchdog_v3 import guard, KNOWN_PAGES
    directory = Path(directory).resolve()
    snapshot_plan_join(directory, parent, plan, child)
    prepared, chain = genuine_baseline(Path(plan['prepared']), plan['lane'], plan.get('baseline_adjudication_receipt'), pack_epoch=pack_epoch)
    require(chain == plan['baseline_source_proof'] == parent['prepared_chain'], 'Current genuine C137/source37 preparation chain differs')
    require(prepared['engine_receipt_sha256'] == plan['engine_receipt_sha256'] and prepared['cards'] == plan['cards'] and (Path(plan['engine_root']).resolve() == Path(prepared['engine_receipt']).resolve().parent), 'Current prepared SDK/topology/path differs')
    engine_binding(Path(plan['engine_root']), plan['lane'])
    path = directory / 'post-model-identity.json'
    binding = parent['post_model_identity']
    require(Path(binding['path']).resolve() == path and binding['sha256'] == sha(path), 'Postidentity exact parent path/hash association differs')
    identity = read(path)
    require(identity['started'] >= journal_binding(directory, parent, 'post')['finished_epoch'], 'New full4 before complete postjournal')
    lock_path = HERE / 'model-lock.json'
    lock = read(lock_path)
    files = [f for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')]
    shards = [ROOT / lock['destination'] / f['path'] for f in files]
    health = read(directory / 'post-health.json')
    require(health['finished_epoch'] == parent['post_health_finished_epoch'] and health['passed'] is True, 'Posthealth chronology/parent association differs')
    boundary = max(parent['child_terminal_epoch'], child['finished_epoch'], health['finished_epoch'])
    require(identity['passed'] is True and identity['lock_sha256'] == sha(lock_path) and (identity['model_revision'] == lock['revision']) and (identity['started'] >= boundary) and (identity['finished'] >= identity['started']) and (identity['after_child_terminal_epoch'] == boundary) and (parent['finished_epoch'] >= identity['finished']), 'Actual terminal/posthealth full4 source chronology differs')
    require(len(identity['rows']) == len(files) == len(shards) == 4, 'Exact four publisher shard identity required')
    for row, want, source in zip(identity['rows'], files, shards):
        st = source.stat()
        current = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        require(Path(row['path']).resolve() == source.resolve() and row['passed'] is True and (row['bytes'] == want['size']) and (row['sha256'] == row['expected_sha256'] == want['sha256']) and (row['stat_before'] == row['stat_after'] == current), 'Current ordered shard path/publisher hash/stat5 differs')
    current = verify_model_identity(path, lock, shards)
    pages = guard(shards[2])
    for name in ('known_pages_before_hash', 'known_pages_after_hash'):
        saved = parent[name]
        require(saved['passed'] is True and Path(saved['path']).resolve() == shards[2].resolve() and (saved['stat_before'] == saved['stat_after'] == identity['rows'][2]['stat_after']), 'Stored both-page source/path/stat proof differs')
        require(len(saved['rows']) == 2 and [(row['offset'], row['expected_sha256']) for row in saved['rows']] == list(KNOWN_PAGES), 'Exact two known-page offsets/digests required')
        for row in saved['rows']:
            rawpath = Path(row['preserved_path'])
            require(rawpath.resolve().parent == directory and (not rawpath.is_symlink()) and (rawpath.stat().st_size == 4096) and (row['passed'] is True) and (row['bytes'] == 4096) and (sha(rawpath) == row['sha256'] == row['expected_sha256']), 'Preserved known-page view changed/escaped/failed')
        require([(r['offset'], r['sha256']) for r in saved['rows']] == [(r['offset'], r['sha256']) for r in pages['rows']], 'Current known-page view differs from recorded brackets')
    require(parent['known_pages_before_hash']['epoch'] <= identity['started'] <= identity['finished'] <= parent['known_pages_after_hash']['epoch'] <= parent['finished_epoch'], 'Known pages must bracket new full4 scan before parent final')
    _, after = genuine_baseline(Path(plan['prepared']), plan['lane'], plan.get('baseline_adjudication_receipt'), pack_epoch=pack_epoch)
    require(after == chain, 'Current C137/SDK source changed during readonly join')
    guard(shards[2])
    return {'snapshot_plan_sha256': parent['plan_sha256'], 'current_C137_source_chain': chain, 'current_post_model_identity': current, 'current_known_pages': pages, 'post_terminal_and_health_boundary': boundary, 'new_full4_bracketed_by_two_page_views': True}

def parent_arm(directory, *, pack_epoch):
    require(__debug__ and sys.flags.optimize == 0 and (os.environ.get('PYTHONOPTIMIZE', '0') in ('', '0')), 'Strict frozen final audit requires assertions enabled; optimized Python refused')
    directory = Path(directory).resolve()
    parent = read(directory / 'parent-qualification.json')
    plan = read(directory / 'input-plan.snapshot.json')
    child = read(directory / 'child/report.json')
    source_observers_off(plan['env'])
    child_binding(directory, parent, plan, child)
    from serial_manifest_pack48_v1 import output_budget_contract, manifest_binding
    manifest_binding(plan, pack_epoch=pack_epoch)
    require(plan.get('schema') == 4 and plan.get('harness_generation') == 40, 'OldV6/foreign finalized harness refused')
    budget = output_budget_contract(plan['kind'], plan['slots'], plan['native_diagnostic_max_new_by_request'], plan['api_max_new_by_request'], plan.get('serial_group_job_count'))
    require(plan['output_budget_contract'] == budget and plan['max_new_by_request'] == budget['actual_submission_budget'], 'Final pertransport output budget changed')
    require(parent['passed'] is True and parent['scoped_collection_or_serial_arm_qualified'] is True and (parent['child_return_code'] == 0) and (parent['owned_containers_terminal'] is True) and (parent['forced_cleanup'] is False) and (not parent['interrupted']) and (not parent['errors']), 'Actual owned source/health parent arm failed')
    require(plan.get('serial_source37_generation') == 3 and plan['kind'] == 'serial', 'Explicit new source37 serial successor required')
    for name, digest in plan['dependency_sha256'].items():
        require(sha(HERE / name) == digest, 'Final consumed source dependency changed ' + name)
    require(parent['wrapper_sha256'] == sha(HERE / 'qualify_batch_serial_source37_v3.py') and parent['controller_sha256'] == sha(HERE / 'batch_serial_source37_v3.py'), 'Actual arm parent/controller source changed')
    require(plan['source_plan_sha256'] == lane_contract(plan['lane'])['sha256'] and parent['plan_sha256'] == sha(Path(parent['plan'])) and (parent['child_report_sha256'] == sha(directory / 'child/report.json')), 'Actual source/plan/child hash join differs')
    require(parent['pre_health_passed'] and parent['post_health_passed'] and parent['kernel_fault_gate_passed'], 'Actual per-arm health/journal proof absent')
    for stage in ('pre', 'post'):
        health = read(directory / (stage + '-health.json'))
        require(health['passed'] and health['cards'] == [0, 1] and (health['health_image'] == 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'), 'Per-card/compiled pair health identity differs')
        require(len(health['files']) == 2 and all((row['return_code'] == 0 and row['error'] is None and (sha(row['path']) == row['sha256']) for row in health['files'])), 'Actual health logs/commands absent or changed')
        commands = [[str(ROOT / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', health['health_image']], [str(ROOT / 'bin/xpu-collective-health'), '--img', health['health_image'], '--p2p', '0', '--timeout', '180']]
        for row, argv, label in zip(health['files'], commands, ('strict', 'compiled-pair')):
            log = directory / (stage + '-' + label + '.log')
            cmd = directory / (stage + '-' + label + '.command.json')
            require(row['path'] == str(log) and row['command'] == argv == read(cmd) and (row['command_file_sha256'] == sha(cmd)) and (row['stdout_sha256'] == sha(log)) and (health['started_epoch'] <= row['started_epoch'] <= row['finished_epoch'] <= health['finished_epoch']), 'Actual health exactargv/log/chronology differs')
    binding = parent['post_model_identity']
    require(sha(binding['path']) == binding['sha256'], 'Actual post4 receipt changed')
    identity = read(binding['path'])
    lock = read(HERE / 'model-lock.json')
    files = [f for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')]
    require(identity['passed'] is True and identity['lock_sha256'] == sha(HERE / 'model-lock.json') and (identity['started'] >= max(child['finished_epoch'], parent['post_health_finished_epoch'])) and (len(identity['rows']) == len(files) == 4), 'Actual mandatory postterminal/posthealth full4 chronology or source lock differs')
    for row, want in zip(identity['rows'], files):
        require(row['passed'] and row['bytes'] == want['size'] and (row['sha256'] == row['expected_sha256'] == want['sha256']) and (row['stat_before'] == row['stat_after']) and (len(row['stat_before']) == 5), 'Actual complete publisher shard/stat5 proof differs')
    validate_artifacts(directory / 'child', child['artifact_bindings'])
    require(parent['source_guard_generation'] == 3, 'Both known-page source guard missing')
    final_source_join(directory, parent, plan, child, pack_epoch=pack_epoch)
    return (parent, plan, child)

def serial_vectors(directory, *, pack_epoch):
    result = {}
    child = directory / 'child'
    plan = read(directory / 'input-plan.snapshot.json')
    source = read(Path(plan['batch_parent']) / 'child/serial-jobs.json')['jobs']
    expected = __import__('serial_selection_pack48_v1').selected_jobs(plan, source, pack_epoch=pack_epoch)
    seen = []
    from extract_serial_cacheoff_numerical_v1 import extract
    from run_batch_serial_source37_v3 import config
    args, env = config(plan)
    for path in child.glob('serial-*/requests.json'):
        for row in read(path):
            job = row['job']
            require(read(path.parent / ('serial-' + str(job['rid']) + '-' + job['role'] + '.json')) == row['raw'], 'Individual serial producer raw differs from request row')
            require(job in expected and job not in seen, 'Serial source token job changed/duplicated')
            seen.append(job)
            key = (job['rid'], job['role'])
            meta = row['meta']
            require(row['raw']['ids'] == job['ids'] and row['raw']['fresh'] == 1, 'Actual serial input/fresh role differs from selected source job')
            new = extract(row['raw'], path.parent / 'captures', args, env, [tuple(x) for x in plan['stage_ranges']])
            require(json.loads(json.dumps(new)) == meta, 'Final original serial collector/hash/provenance differs')
            items = [(-1, meta['logits'][0]['path'])] + [(int(v['layer']), v['path']) for v in meta['residuals']]
            require({layer for layer, name in items} == set(range(-1, 48)), 'Actual serial fullhead/all48 absent')
            for layer, name in items:
                require((key, layer) not in result, 'Duplicate serial group identity')
                result[key, layer] = name
    require(seen == expected, 'Complete source-bound selected serial group job roster differs')
    return result

def finalized_binding(root, *, pack_epoch):
    from audit_batch_pack48_v1 import parent_arm as origin_arm, vectors
    from run_batch_serial_source37_v3 import command_recipe
    root = Path(root).resolve()
    parent, plan, child = parent_arm(root, pack_epoch=pack_epoch)
    collector = Path(plan['batch_parent']).resolve()
    origin_arm(collector, pack_epoch=pack_epoch)
    directory = root / 'child' / ('serial-' + str(plan['group_index']))
    terminal = read(directory / 'result.json')
    require(terminal['passed'] is True and terminal['engine_rc'] == 0 and (terminal['removed'] is True) and (terminal['error'] is None) and (terminal['state']['ExitCode'] == 0) and (not terminal['state']['Running']) and (not terminal['state']['OOMKilled']), 'Actual new serial engine normal owned terminal required')
    require(read(directory / 'command.json') == command_recipe(plan, root / 'child', plan['group_index'], parent['child_pid']), 'Exact new serial command differs')
    serial = serial_vectors(root, pack_epoch=pack_epoch)
    batch = vectors(collector)
    recomputed = compare_all({k: v for k, v in batch.items() if k in serial}, serial)
    require(recomputed['passed'] and recomputed['comparisons'] == child['comparisons'] and (child['actual_serial_job_count'] == plan['actual_serial_job_count']) and (child['actual_matched_full49_vector_pairs'] == len(serial) == 49 * plan['actual_serial_job_count']), 'Actual complete selected-job full49 comparison/report differs')
    parent_after, plan_after, child_after = parent_arm(root, pack_epoch=pack_epoch)
    origin_arm(collector, pack_epoch=pack_epoch)
    require(parent_after == parent and plan_after == plan and (child_after == child), 'Source/parent/child evidence changed during final recollection')
    return {'schema': 1, 'passed': True, 'serial_source37_generation': 3, 'root': str(root), 'parent_sha256': sha(root / 'parent-qualification.json'), 'engine_receipt_sha256': plan['engine_receipt_sha256'], 'collector_parent_sha256': sha(collector / 'parent-qualification.json'), 'group_index': plan['group_index'], 'actual_jobs': plan['actual_serial_job_count'], 'selected_indices': plan['actual_serial_selected_indices'], 'complete_serial_group_qualified': plan['first_job_adjudication'] is None, 'first_job_adjudication': plan['first_job_adjudication'], 'actual_raw49': recomputed, 'full_model_math_qualified': False, 'cache_qualification_granted': False, 'latency_qualified': False}
