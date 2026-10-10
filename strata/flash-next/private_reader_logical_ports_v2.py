# NEW mandatory immutable logical-byte port of private_reader_pack49_v1.py
from upload_logical_port_identity_v2 import original_producer_file, original_module_file
'Closed harness40 paired2 OFF/ON and complete actual serial49 readonly gate.'
import copy, json, os, sys
from pathlib import Path
import audit_batch_logical_ports_v2 as audit
import batch40_manifest_logical_ports_v2 as ctrl
import private_native_protocol_recollection_v2 as protocol
from batch_numerical_prefixes_v2 import compare_all
from serial37_canonical_json_v3 import matches_saved, read_unique
from batch_proofs_logical_ports_v2 import read, write, sha, require, validate_artifacts, native_row_event, exact_producer_counters, owner_proofs
HERE = Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/validate_private_native_offon_source37_v4.py').resolve().parent
ROOT = HERE.parents[1]
SOURCE_PLAN = HERE / 'private-native-offon-source37-source-plan-v4.json'
H40_SHA = '914c2f3c745e93a84f606b4155e1636e2b447971245011a3ea8297bc2a52c62a'

def source_binding():
    plan = read(SOURCE_PLAN)
    require(sha(HERE / 'batch-current-source37-v4-source-plan.json') == H40_SHA and plan['harness_source_plan_sha256'] == H40_SHA, 'Exact frozen harness40/source37 reader required')
    for name, digest in plan['files'].items():
        require(sha(ROOT / name) == digest, 'Private reader frozen source changed ' + name)
    return {'source_plan_sha256': sha(SOURCE_PLAN), 'files': plan['files']}

def model_page_epoch_binding(plan, parent):
    import math
    from serial37_canonical_json_v3 import canonical
    identity = plan['model_identity']
    path = Path(identity['path']).resolve()
    require(sha(path) == identity['sha256'], 'Exact actual full4 identity record changed')
    record = read(path)
    epoch = identity['current_known_pages']['epoch']
    require(type(epoch) in (int, float) and math.isfinite(epoch) and (type(record['finished']) in (int, float)) and math.isfinite(record['finished']) and (type(parent['started_epoch']) in (int, float)) and math.isfinite(parent['started_epoch']) and (record['finished'] <= epoch <= parent['started_epoch']), 'Per-arm observed page-read epoch outside actual identity/prelaunch boundary')
    lock = read(HERE / 'model-lock.json')
    shards = [ROOT / lock['destination'] / row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')]
    current = ctrl.verify_model_identity(path, lock, shards)

    def without_epoch(value):
        value = copy.deepcopy(value)
        del value['current_known_pages']['epoch']
        return value
    require(canonical(without_epoch(identity)) == canonical(without_epoch(current)), 'Every actual modelidentity/path/revision/sentinel/page/stat/digest field must match current independent guard')
    return {'full4_identity_path': str(path), 'full4_identity_sha256': identity['sha256'], 'full4_finished_epoch': record['finished'], 'original_observed_page_read_epoch': epoch, 'actual_parent_started_epoch': parent['started_epoch'], 'original_epoch_recomputed_from_current_clock': False, 'all_other_fields_match_current_guard': True}

def matched_plans(off, on, off_parent, on_parent):
    from serial37_canonical_json_v3 import canonical
    require(off['harness_generation'] == on['harness_generation'] == 40 and off['slots'] == on['slots'] == 2 and (off['kind'] == on['kind'] == 'native') and (off['diagnostic'] == 0) and (on['diagnostic'] == 1) and (off['cards'] == on['cards'] == [0, 1]), 'Exact actual source37harness40 paired2 native OFF/ON required')
    epochs = {'OFF': model_page_epoch_binding(off, off_parent), 'ON': model_page_epoch_binding(on, on_parent)}
    ignored = {'diagnostic', 'research_alias', 'registry_binding'}

    def comparable(plan):
        value = copy.deepcopy(plan)
        del value['model_identity']['current_known_pages']['epoch']
        return {k: v for k, v in value.items() if k not in ignored}
    require(set(off) == set(on) and canonical(comparable(off)) == canonical(comparable(on)), 'Matched OFF/ON complete source/args/env/corpus/baseline/math differs beyond independently proven per-arm page-read epoch')
    for p in (off, on):
        alias = ctrl.api.experimental_alias({'args': p['args'], 'env': p['env']}, 2, bool(p['diagnostic']), p['lane'])
        require(p['research_alias'] == alias and p['registry_binding'] == ctrl.api.registry_gate(alias), 'Exact accepted OFF/ON source alias association differs')
    require(off['registry_binding']['sha256'] == on['registry_binding']['sha256'], 'Actual global registry changed betweenarms')
    return epochs

def native_command_binding(root, parent, plan):
    import shlex
    root = Path(root).resolve()
    child = root / 'child'
    binding = sha(child / 'plan.snapshot.json')
    name = 'b70-prefix-' + str(parent['child_pid']) + '-native2'
    env = dict(plan['env'])
    env.update(STRATA_ARTIFACT_IDENTITY_SHA256=binding, STRATA_BATCH_FIDELITY_DIAG=str(plan['diagnostic']))
    engine = Path(plan['engine_root'])
    model = ROOT / read(HERE / 'model-lock.json')['destination']
    command = ['docker', 'run', '-i', '--name', name, '--label', 'b70.prefix.plan=' + binding, '--network', 'none', '--device', '/dev/dri', '--user', '1000:1000', '--memory', '105g', '--memory-swap', '105g', '--group-add', str(os.stat('/dev/dri/renderD128').st_gid), '-v', str(engine / 'build') + ':/build:ro', '-v', str(engine / 'source') + ':/src:ro', '-v', plan['pack'] + ':/pack:ro', '-v', str(model) + ':/model:ro', '-v', str(child.resolve()) + ':/results']
    for key, value in sorted(env.items()):
        command += ['-e', key + '=' + str(value)]
    command += [plan['image'], 'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec ' + shlex.join(['/build/strata', '--serve'] + plan['args'])]
    require(read(child / 'command.json') == command, 'Actual complete native launch/PID/image/env/mount/source command differs')
    return {'container': name, 'command_sha256': sha(child / 'command.json'), 'effective_environment': env, 'actual_physical_cards': plan['cards']}

def serial_roster_path(on_root):
    on_root = Path(on_root).resolve()
    return on_root.parent / (on_root.name + '.serial49-roster-v4.json')

def serial_binding(off_root, on_root, plan, batch, roster_path, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    import first49_logical_ports_v2 as first
    import serial_reader_logical_ports_v2 as future
    off_root = Path(off_root).resolve()
    on_root = Path(on_root).resolve()
    roster_path = Path(roster_path).resolve()
    require(roster_path == serial_roster_path(on_root) and (not roster_path.is_symlink()), 'Exact NEW V2 external serial49 association required')
    r = read(roster_path)
    require(r['schema'] == 4 and r['off_root'] == str(off_root) and (r['on_root'] == str(on_root)) and (r['off_parent_sha256'] == sha(off_root / 'parent-qualification.json')) and (r['on_parent_sha256'] == sha(on_root / 'parent-qualification.json')) and (r['jobs_sha256'] == sha(on_root / 'child/serial-jobs.json')), 'Actual OFF/ON selected jobs association changed')
    jobs = read(on_root / 'child/serial-jobs.json')['jobs']
    require(len(jobs) == 4, 'Exactly declared paired2 four selected jobs required')
    receipt = Path(r['first_job_receipt']['path']).resolve()
    require(r['first_job_receipt']['sha256'] == sha(receipt), 'Exact saved historical first49 receipt changed')
    saved = read_unique(receipt)
    require(matches_saved(first.finalized_binding(pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster), receipt), 'Actual historical first49 admission changed')
    require(saved['job'] == jobs[0] and saved['engine_receipt_sha256'] == plan['engine_receipt_sha256'] and (saved['collector_parent_sha256'] == sha(on_root / 'parent-qualification.json')) and (saved['original_parent_passed'] is False) and (saved['historical_PIN0_exception_used'] is True) and (saved['new_absent_PIN_scope_qualified'] is False), 'Historical first49 source/index/exception differs')
    row = r['remaining_serial']
    root = Path(row['root']).resolve()
    require(root not in (off_root, on_root) and row['parent_sha256'] == sha(root / 'parent-qualification.json') and (row['plan_sha256'] == sha(root / 'input-plan.snapshot.json')), 'Exact new remaining147 parent/plan changed')
    binding = future.finalized_binding(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    parent, source, child = future.parent_arm(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(parent['parent_generation'] == 43 and source['serial_source37_generation'] == 3 and (source['group_index'] == 0) and (source['actual_serial_selected_indices'] == [1, 2, 3]) and (source['actual_serial_job_count'] == 3) and (source['first_job_adjudication'] == r['first_job_receipt']), 'Exact new parent42 remaining-three/first49 interface required')
    require(Path(source['batch_parent']).resolve() == on_root and source['engine_receipt_sha256'] == plan['engine_receipt_sha256'] and (source['prepared_sha256'] == plan['prepared_sha256']) and (source['cards'] == plan['cards']) and (source['args'] == plan['args']) and (source['env'] == plan['env']), 'Actual remaining3 topology/math/source/context differs')
    meta = saved['raw_meta']
    firstdata = {((jobs[0]['rid'], jobs[0]['role']), -1): meta['logits'][0]['path']}
    for field in meta['residuals']:
        firstdata[(jobs[0]['rid'], jobs[0]['role']), int(field['layer'])] = field['path']
    remaining = future.serial_vectors(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    union = complete_union(jobs, firstdata, remaining)
    require(len(batch) == 196, 'Actual ON selected-job raw196 required')
    compared = compare_all(batch, union)
    require(compared['passed'], 'Actual allselected serial196 byte equality failed; no tolerance waiver')
    require(matches_saved(first.finalized_binding(pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster), receipt) and future.finalized_binding(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster) == binding, 'Historical/new serial source/artifacts changed during comparisons')
    return {'roster_path': str(roster_path), 'roster_sha256': sha(roster_path), 'jobs': jobs, 'actual_first49_receipt': r['first_job_receipt'], 'actual_remaining147_binding': binding, 'comparisons': compared, 'actual_complete_serial49_qualified': True, 'original_failed_first_parent_passed': False, 'historical_first_PIN0_exception_used': True, 'full_group_absent_PIN_scope_qualified': False, 'actual_absent_PIN_jobs': 3, 'historical_supervisor_EOF_producer_field_observed': False, 'original_independent_model_math_qualified': False}

def complete_union(jobs, first, remaining):
    require(len(jobs) == 4 and len({(j['rid'], j['role']) for j in jobs}) == 4, 'Unique exact four source-selected job indices required')
    expected = lambda subset: {((j['rid'], j['role']), layer) for j in subset for layer in range(-1, 48)}
    require(set(first) == expected(jobs[:1]) and set(remaining) == expected(jobs[1:]) and (not set(first) & set(remaining)), 'Exact index0 first49 plus indices1/2/3 remaining147 required; no gaps/duplicates')
    return {**first, **remaining}

def finalized_binding(off_root, on_root, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    require(__debug__ and sys.flags.optimize == 0 and (os.environ.get('PYTHONOPTIMIZE', '0') in ('', '0')), 'Strict readonly source/parser assertions required')
    source = source_binding()
    off_root = Path(off_root).resolve()
    on_root = Path(on_root).resolve()
    require(off_root != on_root, 'Distinct immutable OFF/ON roots required')
    off_parent, off, off_child = audit.parent_arm(off_root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    on_parent, on, on_child = audit.parent_arm(on_root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    epochs = matched_plans(off, on, off_parent, on_parent)
    outputs = {}
    commands = {}
    for label, root, parent, plan, child in [('OFF', off_root, off_parent, off, off_child), ('ON', on_root, on_parent, on, on_child)]:
        require(child['initial_native_collection_and_teardown_passed'] is True and child['engine_rc'] == 0 and (child['state']['ExitCode'] == 0) and (child['state']['Running'] is False) and (not child['state'].get('OOMKilled')) and (not child['state'].get('Error')) and (child['removed'] is True) and (child['error'] is None) and (child.get('failure_diagnostic') is None), 'Actual native collector/EOF/normal ownedterminal failure')
        commands[label] = native_command_binding(root, parent, plan)
        validate_artifacts(root / 'child', child['artifact_bindings'])
        trace = (root / 'child/engine.combined.log').read_text()
        outputs[label] = protocol.recollect(plan, trace, read(root / 'child/requests.json'))
    histories = protocol.compare_histories(outputs['OFF'], outputs['ON'], on['cancel_index'])
    trace = (on_root / 'child/engine.combined.log').read_text()
    stages = [(i, a, b) for i, (a, b) in enumerate(on['stage_ranges'])]
    real_n = native_row_event(trace, 2)
    policy = {(2001 + i, 'admission'): {'reused': 0, 'read_from': 0, 'reread_to': -1} for i in range(2)}
    counters = exact_producer_counters(trace, {2001, 2002}, policy)
    require(outputs['ON']['actual_serial_jobs'] == read(on_root / 'child/serial-jobs.json'), 'Actual complete selected consumed-prefix serial jobs changed')
    batch = audit.vectors(on_root)
    serial = serial_binding(off_root, on_root, on, batch, serial_roster_path(on_root), pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    owners = owner_proofs(trace, stages, 2, 'source37', True)
    for root in (off_root, on_root):
        audit.parent_arm(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(source_binding() == source, 'Private current source changed during readonlywork')
    return {'schema': 4, 'passed': True, 'source_lane': 'source37', 'harness_generation': 40, 'engine_receipt_sha256': on['engine_receipt_sha256'], 'cards': [0, 1], 'off_parent_sha256': sha(off_root / 'parent-qualification.json'), 'on_parent_sha256': sha(on_root / 'parent-qualification.json'), 'source_binding': source, 'actual_launch_bindings': commands, 'actual_model_page_epoch_bindings': epochs, 'actual_complete_histories': outputs, 'matched_native_histories': histories, 'actual_ON_requested_Nrow': real_n, 'actual_ON_counters': counters, 'actual_ON_complete_raw49': {'vectors': len(batch), 'selected_jobs': len(serial['jobs'])}, 'actual_complete_serial49': serial, 'actual_ON_owner_logical_frees': owners, 'actual_OFF_raw49_observed': False, 'actual_OFF_Nrow_event_observed': False, 'actual_stdin_cancellation_bytes_logged': False, 'cancellation_proof': 'Pinned original send predicate reconstructed plus actual matching cancelterminal acknowledgement; no claimed independentlylogged sendtime', 'full_model_math_qualified': False, 'broad_quality_qualified': False, 'concurrent_cache_qualified': False, 'latency_qualified': False}
