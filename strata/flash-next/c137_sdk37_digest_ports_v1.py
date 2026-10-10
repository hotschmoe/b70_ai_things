"""Proposed explicit SDK37 digest ports; not wired into frozen H48."""
from c1_serve_controller_combined_v137 import *
from sdk37_witness_scope_v1 import require_engine,require_oracle
def combined_generation_gate(engine, *, sdk_epoch):
    require_engine(sdk_epoch, engine)
    engine = Path(engine).resolve()
    require(sha(COMBINED_PLAN) == COMBINED_PLAN_SHA, 'Reviewed combined source plan changed')
    plan = read(COMBINED_PLAN)
    receipt = engine / 'receipt.json'
    build = read(receipt)
    require(build.get('build_rc') == 0 and build.get('external_source_unchanged') is True and (build.get('plan_snapshot_unchanged') is True) and (build.get('image') == BASE_IMAGE), 'Actual new combined SDK build prerequisite missing or failed')
    require(build.get('plan_sha256') == COMBINED_PLAN_SHA and sha(engine / 'plan.snapshot.json') == COMBINED_PLAN_SHA and (Path(build['plan_snapshot']).resolve() == engine / 'plan.snapshot.json'), 'SDK plan/source generation association differs')
    require(len(plan['patches']) == 37 and len({row['path'] for row in plan['patches']}) == 37, 'Exact ordered37 patch recipe required')
    require(build.get('source_revision') == plan['source_revision'] and build.get('ggml_revision') == plan['ggml']['revision'] and (build.get('patches') == plan['patches']) and (Path(build['source_copy']).resolve() == engine / 'source'), 'Actual combined source/dependency/patch chain differs')
    ledger = plan['expected_patched_source_sha256']
    require(len(ledger) == 64 and set(ledger) == set(plan['overlay_files']) and (build.get('patched_source_sha256') == ledger), 'Exact complete64 source ledger required; earlier-generation subset refused')
    for name, digest in ledger.items():
        require(sha(engine / 'source' / name) == digest, 'Actual combined source changed: ' + name)
    headers = plan['added_header_payloads']
    require(len(headers) == 28 and len({h['path'] for h in headers}) == 28, 'Exact28 consumed header payloads required')
    for header in headers:
        require(ledger.get(header['path']) == header['sha256'] and sha(engine / 'source' / header['path']) == header['sha256'], 'Consumed header payload differs: ' + header['path'])
    require(plan['runtime_python_sources'] == PYTHON_SOURCE_SHA, 'Integrated six-source Python plan differs')
    targets = plan['build_targets']
    require(len(targets) == 8 and len(set(targets)) == 8 and (set(build.get('binary_sha256', {})) == {str(engine / 'build' / name) for name in targets}), 'New SDK all8 target ledger required')
    for name in targets:
        path = engine / 'build' / name
        require(sdk_epoch.digest_for('SDK37:' + name, path, build['binary_sha256'][str(path)]) == build['binary_sha256'][str(path)], 'Combined SDK target changed: ' + name)
    require(set(PYTHON_SOURCES) == set(PYTHON_SOURCE_SHA) and len(PYTHON_SOURCES) == 6, 'Complete6 runtime Python source contract required')
    for name, digest in PYTHON_SOURCE_SHA.items():
        require(sha(engine / 'source' / name) == digest, 'Actual six-source API runtime differs: ' + name)
    markers = compiled_source_markers(engine)
    return {'schema': 137, **markers, 'plan_path': str(COMBINED_PLAN), 'plan_sha256': COMBINED_PLAN_SHA, 'source_count': 64, 'header_payloads': headers, 'controller_generation': 137, 'semantic_current_PLE_gather_fix32': True, 'current_PLE_host_observer_source33': True, 'final_window_PLE_observer_fix34': True, 'prompt_verifier_P30_capture35': True, 'sdk_targets': targets, 'runtime_python_sources': dict(PYTHON_SOURCE_SHA), 'engine_receipt': str(receipt), 'engine_receipt_sha256': sha(receipt), 'full_model_math_qualified': False, 'concurrency_qualified': False}

def original_upload_gate(path, oracle_path, engine_receipt, pack_receipt, *, sdk_epoch):
    require_oracle(sdk_epoch, engine_receipt, oracle_path)
    report = read(path)
    require(report.get('passed') is True and report.get('pre_health_passed') is True and (report.get('post_health_passed') is True), 'GPU upload/lifecycle did not pass')
    require(report.get('image') == BASE_IMAGE, 'Upload runtime identity differs')
    require(not any((k in report for k in ['error', 'cleanup_error', 'post_health_error'])), 'Upload lifecycle has errors')
    require(report.get('oracle_receipt_sha256') == sha(oracle_path), 'Upload oracle receipt differs')
    require(report.get('pack_receipt_sha256') == sha(pack_receipt), 'Upload pack receipt differs')
    oracle = read(oracle_path)
    require(oracle.get('passed') is True and oracle.get('libraries_unchanged') is True, 'Oracle source build did not pass')
    require(oracle.get('engine_receipt_sha256') == sha(engine_receipt), 'Upload oracle targets another engine generation')
    require(report.get('oracle_schema') == 2, 'This serving controller requires version2 logical-free lifecycle evidence')
    oracle_root = Path(oracle_path).parent
    require(sha(oracle_root / 'plan.snapshot.json') == oracle['plan_sha256'] == report['oracle_plan_sha256'], 'Upload frozen plan fingerprint differs')
    require(sha(oracle_root / 'oracle.cpp') == oracle['oracle_source_sha256'], 'Upload frozen source fingerprint differs')
    require(sdk_epoch.digest_for('source37-upload-oracle', oracle_root / 'source-upload-oracle', oracle['binary_sha256']) == oracle['binary_sha256'], 'Upload executable fingerprint differs')
    plan = read(oracle_root / 'plan.snapshot.json')
    require(plan['schema'] == 2 and plan['oracle_source_sha256'] == oracle['oracle_source_sha256'], 'Upload snapshot/source contract differs')
    require(sha(plan['source_roster']['path']) == plan['source_roster']['sha256'] == report['source_roster_sha256'], 'Original source roster fingerprint differs')
    require(sha(plan['source_roster']['receipt']) == plan['source_roster']['receipt_sha256'], 'Original source roster receipt changed')
    roster = {row['name']: row for row in read(plan['source_roster']['receipt'])['RESULT']['rows']}
    require(sha(plan['inventory']) == plan['inventory_sha256'], 'Original GGUF inventory fingerprint differs')
    inventory = read(plan['inventory'])
    require(inventory['inventory_complete'] and (not inventory['errors']), 'Original source inventory incomplete')
    source_shapes = {tensor['name']: tensor['shape_ggml_order'] for file in inventory['files'] if '/UD-Q4_K_XL/' in file['path'] for tensor in file['tensors']}
    require(len(roster) == 390, 'Complete387 HC plus3 PLE source roster required; narrow30 is insufficient')
    from parse_usm_logical_free_trace import parse_trace, negative_controls
    cases = report.get('cases', [])
    required_cases = {'full390_card0', 'full390_card1', 'full390_two_device24_24', 'full390_actual_model_static_bounds'}
    labels = [row.get('case') for row in cases]
    require(len(labels) == len(set(labels)) and required_cases <= set(labels), 'Incomplete per-card/pair/model-static whole-source coverage')
    for row in cases:
        state, device = (row.get('state', {}), row.get('report', {}))
        require(state.get('ExitCode') == 0 and (not state.get('OOMKilled')) and (not state.get('Running')), 'Upload process has no clean terminal state')
        full_source_case_gate(device, roster, source_shapes)
        if row['case'] == 'full390_actual_model_static_bounds':
            require(device.get('bounds') == 'static', 'Actual model-call static bounds were not exercised')
        logical_path = Path(path).parent / (row['case'] + '-logical-free.json')
        logical = read(logical_path)
        text = (Path(path).parent / (row['case'] + '.log')).read_text()
        checked = parse_trace(text)
        require(checked['passed'] and all(negative_controls(text, True).values()), 'Chronological context-matched logical-free trace failed')
        require(logical.get('passed') and set(logical.get('negative_controls', {})) == {'missing_free', 'double_free', 'failed_free'} and all(logical['negative_controls'].values()), 'Logical-free negative controls incomplete')
        require(all((logical.get(k) == v for k, v in checked.items())), 'Saved logical-free evidence differs from raw trace')
        require(logical['counts']['owners'] == sum((stage['unique_allocations'] + 1 for stage in device['stages'])) and (not logical['live']), 'Owned allocation/scratch coverage differs or live ledger nonempty')
    return {'path': str(Path(path).resolve()), 'sha256': sha(path), 'oracle': str(Path(oracle_path).resolve()), 'oracle_sha256': sha(oracle_path), 'oracle_schema': 2, 'source_coverage': 'whole390', 'ordinary_payload_readback_qualified': False, 'logical_free_parser_sha256': sha(REPO / 'strata/flash-next/parse_usm_logical_free_trace.py')}

def upload_gate(path, oracle_path, engine_receipt, pack_receipt, *, sdk_epoch):
    require_oracle(sdk_epoch, engine_receipt, oracle_path)
    strengthened = strict_v2_upload_provenance(path, oracle_path, engine_receipt)
    return {**original_upload_gate(path, oracle_path, engine_receipt, pack_receipt, sdk_epoch=sdk_epoch), **strengthened}
