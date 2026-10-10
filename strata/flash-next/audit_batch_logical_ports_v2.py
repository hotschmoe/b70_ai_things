# NEW mandatory immutable logical-byte port of audit_batch_pack49_v1.py
from upload_logical_port_identity_v2 import original_producer_file, original_module_file
'Recompute source-qualified actual 2/4/6 scoped native/API/serial proof joins.'
import argparse, json, os, re, sys, time
from pathlib import Path
from batch_proofs_logical_ports_v2 import require, sha, read, write, PLAN_SHA, lane_contract, off_on_histories, owner_proofs, validate_artifacts, confined_file, source_observers_off, ROOT
from batch_numerical_prefixes_v2 import compare_all
from batch40_runtime_evidence_v1 import child_binding, journal_binding
HERE = Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/audit_batch_numerical_suite_v40.py').resolve().parent

def snapshot_plan_join(directory, parent, plan, child):
    directory = Path(directory).resolve()
    input_path = directory / 'input-plan.snapshot.json'
    child_path = directory / 'child/plan.snapshot.json'
    external_path = Path(parent['plan'])
    require(read(input_path) == plan == read(child_path) == read(external_path), 'External/input/child snapshot plan content differs')
    require(parent['plan_sha256'] == child['plan_sha256'] == sha(input_path) == sha(child_path) == sha(external_path), 'Complete external/input/child plan SHA association differs')
    return parent['plan_sha256']

def final_source_join(directory, parent, plan, child, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    from batch_proofs_logical_ports_v2 import genuine_baseline, engine_binding, ROOT
    from batch40_manifest_logical_ports_v2 import verify_model_identity
    from source_page_watchdog_v3 import guard, KNOWN_PAGES
    directory = Path(directory).resolve()
    snapshot_plan_join(directory, parent, plan, child)
    prepared, chain = genuine_baseline(Path(plan['prepared']), plan['lane'], plan.get('baseline_adjudication_receipt'), pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(chain == plan['baseline_source_proof'] == parent['prepared_chain'], 'Current genuine C137/source37 preparation chain differs')
    require(prepared['engine_receipt_sha256'] == plan['engine_receipt_sha256'] and prepared['cards'] == plan['cards'] and (Path(plan['engine_root']).resolve() == Path(prepared['engine_receipt']).resolve().parent), 'Current prepared SDK/topology/path differs')
    engine_binding(Path(plan['engine_root']), plan['lane'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
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
    _, after = genuine_baseline(Path(plan['prepared']), plan['lane'], plan.get('baseline_adjudication_receipt'), pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(after == chain, 'Current C137/SDK source changed during readonly join')
    guard(shards[2])
    return {'snapshot_plan_sha256': parent['plan_sha256'], 'current_C137_source_chain': chain, 'current_post_model_identity': current, 'current_known_pages': pages, 'post_terminal_and_health_boundary': boundary, 'new_full4_bracketed_by_two_page_views': True}

def parent_arm(directory, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    require(__debug__ and sys.flags.optimize == 0 and (os.environ.get('PYTHONOPTIMIZE', '0') in ('', '0')), 'Strict frozen final audit requires assertions enabled; optimized Python refused')
    directory = Path(directory).resolve()
    parent = read(directory / 'parent-qualification.json')
    plan = read(directory / 'input-plan.snapshot.json')
    child = read(directory / 'child/report.json')
    source_observers_off(plan['env'])
    child_binding(directory, parent, plan, child)
    from batch40_manifest_logical_ports_v2 import output_budget_contract, manifest_binding
    manifest_binding(plan, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(plan.get('schema') == 4 and plan.get('harness_generation') == 40, 'OldV6/foreign finalized harness refused')
    budget = output_budget_contract(plan['kind'], plan['slots'], plan['native_diagnostic_max_new_by_request'], plan['api_max_new_by_request'], plan.get('serial_group_job_count'))
    require(plan['output_budget_contract'] == budget and plan['max_new_by_request'] == budget['actual_submission_budget'], 'Final pertransport output budget changed')
    require(parent['passed'] is True and parent['scoped_collection_or_serial_arm_qualified'] is True and (parent['child_return_code'] == 0) and (parent['owned_containers_terminal'] is True) and (parent['forced_cleanup'] is False) and (not parent['interrupted']) and (not parent['errors']), 'Actual owned source/health parent arm failed')
    require(plan['dependency_sha256']['audit_batch_numerical_suite_v40.py'] == sha(Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/audit_batch_numerical_suite_v40.py')), 'Original final auditor source differs from immutable arm plan')
    for name, digest in plan['dependency_sha256'].items():
        require(sha(HERE / name) == digest, 'Final consumed source dependency changed ' + name)
    require(parent['wrapper_sha256'] == sha(HERE / 'qualify_batch_numerical_v40.py') and parent['controller_sha256'] == sha(HERE / 'batch_numerical_execution_v40.py'), 'Actual arm parent/controller source changed')
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
    final_source_join(directory, parent, plan, child, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    return (parent, plan, child)

def native_histories(directory, plan):
    roster = read(directory / 'child/requests.json')
    rows = {}
    for index in range(plan['slots']):
        row = roster[str(2001 + index)]
        require(row['ids'] == plan['tokens']['target'][index] and row['done'], 'Actual native target submitted/terminal differs')
        words = row['done'].split()
        rows[str(index)] = {'submitted_ids': row['ids'], 'sampling': {'temperature': 0}, 'max_new': plan['max_new'], 'native_generated_ids': row['generated'], 'native_finish': words[3] if words[0] == 'BDONE' else words[5], 'native_cancel_completed': words[0] == 'BDONE' and words[3] == 'cancel'}
    return rows

def api_histories(directory, plan):
    events = [json.loads(l) for l in (directory / 'child/api-native-trace.jsonl').read_text().splitlines()]
    begins = {e['call']: e for e in events if e['kind'] == 'engine_begin'}
    ends = {e['call']: e for e in events if e['kind'] == 'engine_end'}
    prefix = read(directory / 'child/prefixes.json')
    rows = {}
    for index, messages in enumerate(plan['messages']['target']):
        matching = [call for call, e in begins.items() if e['rendered_prompt']['messages'] == messages]
        require(len(matching) == 1, 'Exact logical API input roster ambiguous')
        call = matching[0]
        begin = begins[call]
        end = ends[call]
        segments = [seg for history in prefix['native_segments'].values() for seg in history if seg['call'] == call]
        require(segments and all((seg['terminal'] for seg in segments)), 'Actual API native history/terminal absent')
        segments.sort(key=lambda x: x['send_sequence'])
        actual = [token for seg in segments for token in seg['generated']]
        rows[str(index)] = {'submitted_ids': begin['submitted_ids'], 'sampling': begin['sampling'], 'max_new': begin['max_new'], 'native_generated_ids': actual, 'native_finish': end['engine_last'].get('finish'), 'native_cancel_completed': end['cancelled'] and any((e['kind'] == 'native_send' and e.get('call') == call and e['line'].startswith(('BSTOP ', 'STOP')) for e in events))}
    return rows

def vectors(directory):
    result = {}
    plan = read(directory / 'input-plan.snapshot.json')
    report = read(directory / 'child/report.json')
    stages = [(i, lo, hi) for i, (lo, hi) in enumerate(plan['stage_ranges'])]
    trace = (directory / 'child/engine.combined.log').read_text()
    from audit_batch_fidelity_coverage_v5 import coverage as initial
    from audit_batch_fidelity_coverage_v6 import coverage as migration
    if plan['kind'] == 'native':
        old = report['report']['raw']
        new = initial(trace, old['requests'], stages, directory / 'child/captures')
    else:
        old = report['raw']['raw']
        new = migration(trace, old['initial']['requests'], old['migration_requests'], stages, directory / 'child/captures')
    require(json.loads(json.dumps(new)) == old, 'Final original coverage/hash/provenance differs from child-terminal receipt')
    for line in (directory / 'child/engine.combined.log').read_text().splitlines():
        if not line.startswith('SBF vector '):
            continue
        f = dict(re.findall('(\\w+)=([^ ]+)', line))
        role = 'admission' if f['phase'].startswith('admission_') else 'later' if f['phase'].startswith('batch_step_') else 'solo_migration'
        key = ((int(f['rid']), role), int(f['layer']))
        require(key not in result, 'Duplicate actual RID/role/layer vector')
        require(Path(f['file']).parent == Path('/results/captures'), 'Actual producer file path outside exact capture root')
        file = confined_file(directory / 'child/captures', Path(f['file']).name)
        count = 248320 if int(f['layer']) == -1 else 10240
        require(f['canonical'] == 'le_f32' and int(f['floats']) == count and (int(f['bytes']) == count * 4) and (file.stat().st_size == count * 4) and (not file.is_symlink()), 'Final actual vector geometry changed')
        result[key] = str(file)
    return result

def serial_vectors(directory, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    result = {}
    child = directory / 'child'
    plan = read(directory / 'input-plan.snapshot.json')
    source = read(Path(plan['batch_parent']) / 'child/serial-jobs.json')['jobs']
    expected = source[plan['group_index'] * 6:plan['group_index'] * 6 + 6]
    seen = []
    import serial_prefix_qualification_v6 as collector
    for path in child.glob('serial-*/requests.json'):
        for row in read(path):
            job = row['job']
            require(job in expected and job not in seen, 'Serial source token job changed/duplicated')
            seen.append(job)
            key = (job['rid'], job['role'])
            meta = row['meta']
            require(row['raw']['ids'] == job['ids'] and row['raw']['fresh'] == 1, 'Actual serial input/fresh role differs from selected source job')
            new = collector.extract(row['raw'], path.parent / 'captures', True, True, [tuple(x) for x in plan['stage_ranges']])
            require(json.loads(json.dumps(new)) == meta, 'Final original serial collector/hash/provenance differs')
            items = [(-1, meta['logits'][0]['path'])] + [(int(v['layer']), v['path']) for v in meta['residuals']]
            require({layer for layer, name in items} == set(range(-1, 48)), 'Actual serial fullhead/all48 absent')
            for layer, name in items:
                require((key, layer) not in result, 'Duplicate serial group identity')
                result[key, layer] = name
    require(seen == expected, 'Complete source-bound selected serial group job roster differs')
    return result
