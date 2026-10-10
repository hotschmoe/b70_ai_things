from c137_sdk37_digest_ports_v1 import combined_generation_gate
from sdk37_witness_scope_v1 import require_engine
'Strict genuine37 SDK/C137 admission and final native/API numerical proof joins.'
import hashlib, importlib, json, math, re, struct
from pathlib import Path
from audit_batch_fidelity_coverage_v5 import coverage as initial_coverage
from audit_batch_fidelity_coverage_v6 import coverage as migration_coverage
from audit_slot_owner_trace_v1 import audit as slot_ownership
ROOT = Path(__file__).resolve().parents[2]
PLAN = 'strata/flash-next/hosttrace36-batchstamp37-engine-build-plan-v1.json'
PLAN_SHA = '2e940d51c61abe5366526b926f89dcf9f42bf778feee5d13c9b72aefc2785ace'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='ascii')

def source_observers_off(env):
    for name in ('STRATA_PLE_INPUT33', 'STRATA_PREFIX30', 'STRATA_CRITICAL_PATH_TRACE'):
        require(env.get(name, '0') == '0', 'Batch source37 baseline requires ' + name + ' OFF')
    require('STRATA_VERIFY_EAGER' not in env, 'Batch source37 normal graph requires EAGER absent, including value0')
    return True
LANES = {'source37': {'plan': PLAN, 'sha256': PLAN_SHA, 'source_files': 64, 'headers': 28, 'patches': 37, 'generation': 137, 'controller_sha256': 'ad129cf6f9a0832615df88e985437168af0c40082049dc9c2e05dac27147ac53', 'parent_sha256': '703273a654783bbbe61902c3c0c0f2471bc783ea740a01754a86fab36fdc8fc2', 'parent_module': 'qualify_c1_serving_combined_v137_v4', 'parent_generation': 1374}}

def lane_contract(lane):
    require(lane in LANES, 'Explicit source37/current-PLE/C137 only; older29/31 diagnostic lanes cannot qualify corrected-model concurrency lane required')
    return LANES[lane]

def providers(lane='source37'):
    spec = lane_contract(lane)
    require(spec['controller_sha256'] and spec['parent_sha256'], 'C137 helper pins not frozen; no genuine positive permitted')
    generation = str(spec['generation'])
    c1 = importlib.import_module('c1_serve_controller_combined_v' + generation)
    parent = importlib.import_module(spec['parent_module'])
    require(sha(c1.__file__) == spec['controller_sha256'] and sha(parent.__file__) == spec['parent_sha256'], 'Exact lane C1 source helpers changed')
    return (c1, parent)

def engine_binding(engine, lane='source37', *, sdk_epoch, pack_epoch):
    require_engine(sdk_epoch, engine)
    spec = lane_contract(lane)
    source = read(ROOT / spec['plan'])
    require(sha(ROOT / spec['plan']) == spec['sha256'], 'Explicit lane SDK source plan changed')
    r = read(engine / 'receipt.json')
    require(r.get('build_rc') == 0 and r.get('external_source_unchanged') is True and (r.get('plan_snapshot_unchanged') is True) and (r.get('plan_sha256') == spec['sha256']) and (r['image'] == source['image']), 'Actual matching whole SDK absent; foreign source lane cannot transfer')
    require(r['patched_source_sha256'] == source['expected_patched_source_sha256'] and len(r['patched_source_sha256']) == spec['source_files'], 'Actual complete source ledger differs')
    require(len(r['patches']) == spec['patches'] and [(Path(p['path']).name, p['sha256']) for p in r['patches']] == [(Path(p['path']).name, p['sha256']) for p in source['patches']], 'Actual complete ordered patch chain differs')
    for name, digest in source['expected_patched_source_sha256'].items():
        require(sha(engine / 'source' / name) == digest, 'Consumed source changed ' + name)
    require(len(source['added_header_payloads']) == spec['headers'], 'Actual complete header payload roster differs')
    for item in source['added_header_payloads']:
        require(item['sha256'] == source['expected_patched_source_sha256'][item['path']], 'Header payload contradicts final source')
    for target in source['build_targets']:
        binary = engine / 'build' / target
        require(binary.is_file() and r['binary_sha256'].get(str(binary)) == sdk_epoch.digest_for('SDK37:' + target, binary, r['binary_sha256'][str(binary)]), 'Actual ABI target missing ' + target)
        require(sdk_epoch.ELF_magic('SDK37:' + target, binary, r['binary_sha256'][str(binary)]) == b'\x7fELF', 'Mock file is not actual rebuilt ELF')
    require(len(source['build_targets']) == 8 and 'icpx --version' in r['container_script'], 'Actual full SDK command evidence absent')
    c1, _ = providers(lane)
    combined_generation_gate(engine, sdk_epoch=sdk_epoch)
    return r

def genuine_baseline(directory, lane='source37', adjudication_receipt=None, *, pack_epoch=None, sdk_epoch):
    raw = read(directory / 'prepared.json')
    engine = Path(raw['engine_receipt']).parent
    receipt = engine_binding(engine, lane, sdk_epoch=sdk_epoch, pack_epoch=pack_epoch)
    source_observers_off(read(directory / 'server-config.json')['env'])
    c1, parent = providers(lane)
    from c137_baseline_pack49_v1 import finalized_binding
    prepared, admission = finalized_binding(directory, adjudication_receipt, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
    proof = admission['adjudication']['source_proof'] if adjudication_receipt is not None else admission['source_proof']
    require(prepared.get('combined_generation', {}).get('semantic_current_PLE_gather_fix32') is True, 'Current PLE32 semantic source proof absent')
    require(prepared['combined_generation'].get('current_PLE_host_observer_source33') is True, 'Current source33 compiled host observer proof absent')
    require(prepared['combined_generation'].get('final_window_PLE_observer_fix34') is True, 'Current source34 final prompt window observer proof absent')
    require(prepared['combined_generation'].get('prompt_verifier_P30_capture35') is True, 'Current source37 normal prompt verifier capture proof absent')
    require(prepared['combined_generation'].get('bounded_host_critical_path_trace36') is True and prepared['combined_generation'].get('earlier_stage_batch_observer_stamp37') is True, 'Actual source36/37 SDK code binding absent')
    require(prepared['engine_receipt_sha256'] == sha(engine / 'receipt.json'), 'C1 proof not matching this SDK')
    upload = prepared['upload_lifecycle']
    require(upload.get('runner_generation') == 2 and upload.get('post_full4_source_qualified') is True, 'New matching390/uploadV2 final4 absent')
    spec = lane_contract(lane)
    return (prepared, {'source_lane': lane, 'source_plan_sha256': spec['sha256'], 'engine_receipt_sha256': sha(engine / 'receipt.json'), 'prepared_sha256': sha(directory / 'prepared.json'), 'c1_source_proof': proof, 'explicit_baseline_admission': admission, 'C1_controller_sha256': spec['controller_sha256'], 'C1_parent_sha256': spec['parent_sha256'], 'matching_source390_upload_post4_qualified': True, 'semantic_current_PLE_gather_fix32': True, 'current_PLE_host_observer_source33': True, 'final_window_PLE_observer_fix34': True, 'prompt_verifier_P30_capture35': True, 'source33_input_observation_runtime_qualified': False, 'historical_wrong_PLE29_31_goal_positive': False})

def topology_baselines(onecard, pair, lane='source37', onecard_adjudication=None, *, pack_epoch=None, sdk_epoch):
    """Actual finalized C137 per topology, same new SDK; no old proof views."""
    one, a = genuine_baseline(Path(onecard), lane, onecard_adjudication, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
    two, b = genuine_baseline(Path(pair), lane, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
    require(one['cards'] == [0] and two['cards'] == [0, 1] and (one['engine_receipt_sha256'] == two['engine_receipt_sha256']), 'Fresh sameSDK finalized C137 one/pair prerequisite required')
    return {'onecard_root': str(Path(onecard).resolve()), 'pair_root': str(Path(pair).resolve()), 'onecard_adjudication': str(Path(onecard_adjudication).resolve()) if onecard_adjudication is not None else None, 'onecard': a, 'pair': b, 'engine_receipt_sha256': one['engine_receipt_sha256']}

def case_source_gate(spec, engine):
    require(spec.get('source_lane') == 'source37' and spec.get('harness_generation') == 49 and (spec.get('source_plan_sha256') == PLAN_SHA), 'Explicit newsource37 case generation required')
    plan = read(ROOT / PLAN)
    require(sha(ROOT / PLAN) == PLAN_SHA, 'Frozen combined36+37 recipe changed')
    require(spec.get('source_sha256') == plan['expected_patched_source_sha256'] and len(spec['source_sha256']) == 64, 'Case mustrefresh complete actual64 consumed source hashes; old35/subset cannot transfer')
    for name, digest in spec['source_sha256'].items():
        require(sha(Path(engine) / 'source' / name) == digest, 'Case actual source changed ' + name)
    return {'source_plan_sha256': PLAN_SHA, 'source_count': 64, 'header_count': 28, 'patch_count': 37, 'fresh_abi_count': 8, 'runtime_python_count': 6}

def exact_producer_counters(trace, rids, policy):
    actual = {}
    for line in trace.splitlines():
        if not line.startswith('SBF resume '):
            continue
        f = dict(re.findall('(\\w+)=([^ ]+)', line))
        rid = int(f['rid'])
        slot = int(f['slot'])
        key = (rid, 'solo_migration' if slot == 6 else 'admission')
        if rid not in rids:
            continue
        require(key not in actual, 'Duplicate actual resume identity')
        actual[key] = {k: int(f[k]) for k in ('reused', 'read_from', 'reread_to')}
    require(set(actual) == set(policy), 'Exact declared per-RID/phase counters missing or unexpected')
    for key, want in policy.items():
        require(actual[key] == want, 'Actual unplanned source reuse/reread ' + str(key))
    return {str(k): v for k, v in actual.items()}

def native_row_event(trace, requested, after_marker='HARNESS ARM'):
    lines = trace.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(after_marker)), None)
    require(start is not None, 'Actual same-process warm->ARM transition missing')
    warm = []
    target = []
    for i, line in enumerate(lines):
        if line.startswith('SBF batch_event '):
            f = dict(re.findall('(\\w+)=([^ ]+)', line))
            require(f['completed'] == '1', 'Uncompleted batch event')
            (warm if i < start else target).append(f)
    require(any((int(e['rows']) >= 2 for e in warm)), 'No real unarmed multi-row warmup')
    require(any((int(e['rows']) == requested for e in target)), 'No real requested-N-row target body event after ARM')
    require(not any((l.startswith(('SBF request ', 'SBF replay ', 'SBF vector ', 'SBF allocation ')) for l in lines[:start])), 'Unarmed warmup emitted observer allocations/copies/frames')
    return {'warm_multirow_completed': True, 'requested_Nrow_completed': True, 'same_process_ARM_line': start + 1}

def terminal_api_join(events, expected_calls, cancel_calls):
    begins = {}
    ends = {}
    sends = {}
    engine = {}
    for e in events:
        if e['kind'] == 'engine_begin':
            call = e['call']
            require(call not in begins and e['rendered_matches_submitted'] is True and (e['embeddings'] is None), 'Actual API begin/render/source identity invalid')
            begins[call] = e
            engine[call] = (e['engine_pid'], e['engine_generation'])
        elif e['kind'] == 'engine_end':
            call = e['call']
            require(call in begins and call not in ends and ((e['engine_pid'], e['engine_generation']) == engine[call]) and (e.get('error') is None), 'API reader/stale/truncated error')
            ends[call] = e
        elif e['kind'] == 'native_send' and e.get('call') is not None:
            sends.setdefault(e['call'], []).append(e['line'])
    require(set(begins) == set(ends) == set(expected_calls), 'Missing/foreign API request-local roster')
    for call in expected_calls:
        e = ends[call]
        require(e['engine_last'].get('request_id') == e['rid'] and e['engine_last'].get('engine_generation') == e['engine_generation'], 'Request-local final native identity differs')
        if call in cancel_calls:
            require(e['cancelled'] is True and any((line == 'STOP' or line.startswith('BSTOP ') for line in sends.get(call, []))), 'HTTP cancellation did not reach actual native stop')
        else:
            require(e['cancelled'] is False and e['engine_last'].get('finish') in ('stop', 'length') and e['generated_ids'], 'Unexpected canceled/unfinished peer')
            require(not e['consumer_closed'] or e['terminal_pinned_eos'] is True, 'Non-EOS truncated API consumer')
    return {'actual_request_local_completion': True, 'real_native_cancel_calls': sorted(cancel_calls), 'calls': sorted(expected_calls)}

def passing_suite(parent, child, post_hash):
    gates = ['all_requested_2_4_6_cases', 'actual_Nrow_events', 'all_private_RID_raw_head48', 'exact_consumed_prefixes_and_counters', 'actual_serial_raw_byte_equal', 'diagnostic_off_on_output_equal', 'actual_native_cancel_and_solo_migration', 'all_observer_slot_mirror_logical_frees']
    return parent.get('child_return_code') == 0 and parent.get('owned_containers_terminal') is True and (parent.get('forced_cleanup') is False) and (parent.get('pre_health_passed') is True) and (parent.get('post_health_passed') is True) and (parent.get('kernel_fault_gate_passed') is True) and (not parent.get('errors')) and (not parent.get('interrupted')) and all((child.get(k) is True for k in gates)) and (post_hash.get('passed') is True) and (post_hash.get('started', 0) >= max(child.get('finished_epoch', float('inf')), parent.get('post_health_finished_epoch', float('inf'))))

def off_on_histories(off, on, cancelled):
    """Compare native IDs, retaining cancellation timing as a separate lane."""
    require(set(off) == set(on), 'Off/on logical request roster differs')
    rows = {}
    for identity in on:
        a, b = (off[identity], on[identity])
        require(a['submitted_ids'] == b['submitted_ids'], 'Off/on submitted prefix differs')
        require(a['sampling'] == b['sampling'] and a['max_new'] == b['max_new'], 'Off/on generation configuration differs')
        if identity in cancelled:
            n = min(len(a['native_generated_ids']), len(b['native_generated_ids']))
            require(n > 0 and a['native_generated_ids'][:n] == b['native_generated_ids'][:n], 'Canceled off/on common trajectory differs')
            require(a['native_cancel_completed'] is True and b['native_cancel_completed'] is True, 'Cancel timing lane lacks native terminal')
            rows[identity] = {'common_generated_ids': n, 'complete_count_equal_required': False, 'scope': 'real cancellation can occur at different completed native token boundaries'}
        else:
            require(a['native_generated_ids'] == b['native_generated_ids'] and a['native_finish'] == b['native_finish'], 'Uncanceled off/on IDs or finish differ')
            rows[identity] = {'exact_generated_ids_and_finish': True}
    return {'passed': True, 'rows': rows, 'latency_qualified': False}

def owner_proofs(trace, stages, slots, lane='source37', snapshots_observed=True):
    from audit_batch_capture_lifecycle_v2 import audit as snapshots
    expected = [(stage, slot) for stage, lo, hi in stages for slot in range(slots)]
    raw = snapshots(trace, stages) if snapshots_observed else {'passed': True, 'status': 'UNOBSERVED disabled snapshot; no owner allocation expected'}
    private = slot_ownership(trace, expected)
    require(raw['passed'] and private['passed'], 'Actual observer or private slot owning logical frees failed')
    mirror = None
    if lane == 'source37':
        from audit_stage_mirror_owner_trace_v1 import audit as mirror_audit
        mirror = mirror_audit(trace, [(stage, stage, lo, hi) for stage, lo, hi in stages])
        require(mirror['passed'], 'Actual model-process mirror ownership/frees failed')
    return {'mirror': mirror, 'snapshots': raw, 'private_slots': private, 'snapshot_and_slot_logical_frees': bool(snapshots_observed), 'private_slot_logical_frees': True, 'serving_mirror_owner_logical_frees': bool(mirror and mirror['passed']), 'scope': 'Actual serving31 mirror source/verifier retirements and context-matched frees' if mirror else 'Model-process StageExpertMirror owner registry unavailable in29; matching390 not substituted'}

def raw_case(trace, events, rids, migrating, stages, capture_root, counter_policy, ownership_terminal=True):
    """Recompute source producer coverage/history/identity, never trust pass flags."""
    from batch_api_prefixes_v2 import prefix_jobs
    require(len(rids) in (2, 4, 6) and len(set(rids)) == len(rids), 'Exact actual request RID roster required')
    row = native_row_event(trace, len(rids))
    raw = migration_coverage(trace, rids, migrating, stages, capture_root)
    prefixes = prefix_jobs(events)
    require(prefixes['all_segments_terminal'], 'Actual native segment terminal missing')
    expected = {(rid, role) for rid in rids for role in ('admission', 'later')} | {(rid, 'solo_migration') for rid in migrating}
    actual = {(job['rid'], job['role']) for job in prefixes['jobs']}
    require(expected == actual, 'Actual prefix jobs do not cover initial/later/migration roster')
    counters = exact_producer_counters(trace, set(rids), counter_policy)
    owners = owner_proofs(trace, stages, len(rids)) if ownership_terminal else None
    return {'rows': row, 'raw': raw, 'prefixes': prefixes, 'counters': counters, 'owners': owners, 'raw_prefix_and_owner_scope_complete': bool(ownership_terminal), 'full_cache_public_fresh_pin_qualified': False, 'full_model_math_qualified': False}

def declared_remaining_gates():
    return ['genuine source37 source390 and C137 final proof per topology', 'same-process native/API warm-to-ARM 2/4/6 body events', 'actual all48/head initial/later/solo raw comparisons to exact consumed-prefix serial controls', 'native/API asynchronous cancellation with surviving peers and session isolation', 'actual serving mirror owner registry and logical frees', 'cache25/27 enabled conversation/fresh/pin/history/fairness lane', 'own-state original full-model mathematical reference']

def confined_file(root, name):
    """Reject traversal, symlink aliases and foreign capture roots before any read."""
    root = Path(root)
    name = Path(name)
    require(not root.is_symlink() and root.is_dir(), 'Artifact root symlink/missing')
    require(not name.is_absolute() and name.parts and ('..' not in name.parts), 'Artifact path outside exact root')
    file = root / name
    require(file.resolve().is_relative_to(root.resolve()), 'Artifact path escaped exact root')
    at = file
    while at != root:
        require(not at.is_symlink(), 'Artifact symlink alias rejected')
        at = at.parent
    require(file.is_file(), 'Actual sealed artifact missing')
    return file

def artifact_bindings(root):
    root = Path(root)
    result = {}
    raw_bytes = 0
    for file in sorted(root.rglob('*')):
        if file.is_dir():
            require(not file.is_symlink(), 'Artifact directory symlink')
            continue
        rel = file.relative_to(root)
        if rel.as_posix() == 'report.json':
            continue
        file = confined_file(root, rel)
        size = file.stat().st_size
        if 'captures' in rel.parts:
            raw_bytes += size
            require(raw_bytes <= 64 << 20, 'Bounded actual raw capture quota exceeded')
        require(len(result) < 8192, 'Bounded artifact roster exceeded')
        result[rel.as_posix()] = {'sha256': sha(file), 'bytes': size}
    return result

def validate_artifacts(root, expected):
    require(expected and artifact_bindings(root) == expected, 'Actual child-terminal artifact bytes/path/roster changed')
    return True
