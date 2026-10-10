"""One bounded actual native/API/serial arm; each requires its own parent gates."""
import argparse, json, os, sys, time, signal
from pathlib import Path
from types import SimpleNamespace
from operation_pack_hash_witness_v2 import for_prepared
from sdk49_operation_scope_v1 import for_plan as sdk_for_plan, for_prepare as sdk_for_prepare
from batch54_logical_scope_v1 import for_plan as logical_for_plan,for_prepare as logical_for_prepare
from serial37_canonical_json_v3 import canonical
from paired_binding_diagnosis48_v1 import binding as diagnosis_binding
import c1_serve_controller_combined_v137 as c1
import serial_prefix_qualification_v6 as base
from source_page_watchdog_v3 import guard as page_guard
from batch50_case_binding_v1 import authenticated as authenticated_case, binding as case_binding
from batch49_profile_v1 import binding as profile_binding, configuration
from batch49_plan_snapshot_v1 import Snapshot, write_snapshot
from batch_numerical_proofs_v54 import genuine_baseline, engine_binding, read, write, sha, require, PLAN_SHA, artifact_bindings, lane_contract, providers, source_observers_off, topology_baselines, case_source_gate
import run_batch_numerical_pilot_v50 as native
import run_batch_api_controls_v49 as api
import run_batch_serial_controls_v54 as serial
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
expected_stage_ranges = base.expected_stage_ranges
DEPENDENCIES = list(dict.fromkeys(['c137_sdk37_logical_ports_v2.py', 'c137_prepared_logical_ports_v2.py', 'c137_baseline_logical_ports_v2.py', 'adjudicate_c137_logical_ports_v2.py', 'batch_proofs_logical_ports_v2.py', 'batch40_manifest_logical_ports_v2.py', 'audit_batch_logical_ports_v2.py', 'first49_logical_ports_v2.py', 'serial_selection_logical_ports_v2.py', 'serial_manifest_logical_ports_v2.py', 'serial_reader_logical_ports_v2.py', 'private_reader_logical_ports_v2.py', 'paired_control_logical_ports_v2.py', 'batch_numerical_proofs_logical_ports_v2.py', 'batch50_manifest_logical_ports_v2.py', 'audit_batch50_logical_ports_v2.py', 'upload_logical_port_identity_v2.py', 'prepare_upload_logical_gate_ports_v2.py']+['logical_witness_binding_v2.py','validate_batch54_serial_v1.py','logical_free_immutable_epoch_v2.py','logical_free_immutable_worker_v2.py','logical_free_worker_owned_v2.py','upload-logical-consumer-ports-source-plan-v2.json','batch54_logical_scope_v1.py','upload_logical_epoch_factory_v2.py','logical_free_require_case_epoch_v2.py','logical_free_method_binding_v2.py','original_upload_logical_requirement_v2.py','batch54_serial_group_binding_v1.py', 'batch51_serial_jobs_v1.py', 'batch_numerical_execution_v50.py', 'audit_batch_numerical_suite_v50.py', 'batch_numerical_proofs_v50.py', 'batch50_runtime_evidence_v1.py', 'batch50_health_handshake_v1.py', 'batch50_witness_binding_v1.py', 'batch54_health_handshake_v1.py', 'batch50_byte_epochs_v1.py', 'batch54_witness_binding_v1.py', 'operation_pack_hash_witness_v2.py', 'c137_prepared_pack49_v1.py', 'prepare_pack_gate_ports49_v1.py', 'pack_witness_binding47_v1.py', 'operation_sdk37_byte_witness_v1.py', 'sdk37_witness_scope_v1.py', 'c137_sdk37_digest_ports_v1.py', 'sdk_witness_binding49_v1.py', 'sdk49_operation_scope_v1.py', 'paired_binding_diagnosis48_v1.py', 'paired-native-binding-types48-source-plan-v1.json', 'c137_baseline_pack49_v1.py', 'adjudicate_c137_pack49_v1.py', 'batch_proofs_pack49_v1.py', 'batch40_manifest_pack49_v1.py', 'audit_batch_pack49_v1.py', 'first49_pack49_v1.py', 'serial_selection_pack49_v1.py', 'serial_manifest_pack49_v1.py', 'serial_reader_pack49_v1.py', 'private_reader_pack49_v1.py', 'paired_control_pack49_v1.py', 'c137_prepared_pack49_v1.py'] + ['c137_baseline_admission_v4.py', 'adjudicate_c137_missing_started_v3_v1.py', 'c137-missing-started-adjudication-source-plan-v1.json', 'batch54_runtime_evidence_v1.py', 'audit_batch_numerical_suite_v54.py', 'parse_usm_logical_free_trace.py', 'audit_fidelity_observer_coverage.py', 'collect_fidelity_observer.py', 'batch_numerical_execution_v54.py', 'qualify_batch_numerical_v54.py', 'batch_numerical_proofs_v54.py', 'run_batch_numerical_pilot_v50.py', 'run_batch_api_controls_v49.py', 'run_batch_serial_controls_v54.py', 'batch_numerical_protocol_v2.py', 'batch_numerical_prefixes_v2.py', 'batch_api_trace_v2.py', 'batch_api_client_v2.py', 'batch_api_prefixes_v2.py', 'audit_batch_capture_lifecycle_v2.py', 'audit_slot_owner_trace_v1.py', 'audit_batch_fidelity_coverage_v2.py', 'audit_batch_fidelity_coverage_v4.py', 'audit_batch_fidelity_coverage_v5.py', 'audit_batch_fidelity_coverage_v6.py', 'audit_stage_mirror_owner_trace_v1.py', 'merged_numerical_protocol_v2.py', 'serial_prefix_qualification_v6.py', 'layer0_numerical_qualification_v9.py', 'source_page_watchdog_v3.py', 'c1_trace_contract.py', 'c1_serve_controller_combined_v137.py', 'qualify_c1_serving_combined_v137_v4.py', 'c137_journal_binding_v4.py', 'c1-combined-v137-parent-source-plan-v4.json', 'c1-combined-v137-source-plan.json', 'c1-combined-v137-parent-source-plan.json', 'hosttrace36-batchstamp37-engine-build-plan-v1.json', 'paired_source37_private_v4_control_v1.py', 'validate_private_native_offon_source37_v4.py', 'private-native-offon-source37-source-plan-v4.json', 'serial37_canonical_json_v3.py', 'serial37_stdout_capture_v2.py', 'extract_serial_cacheoff_numerical_v1.py', 'serial_prefix_qualification_v8.py', 'batch54_prelease_v1.py', 'batch49_profile_v1.py', 'batch49_plan_snapshot_v1.py', 'batch50_case_binding_v1.py', 'batch-numerical-case4-source37-v11.json', 'batch-numerical-case6-source37-v11.json']))

def verify_model_identity(path, lock, shards):
    identity = base.verify_model_identity(path, lock, shards)
    identity['current_known_pages'] = page_guard(shards[2])
    return identity

def output_budget_contract(kind, slots, native_budgets, api_budgets, serial_job_count=None):
    require(kind in ('native', 'api', 'serial') and type(slots) is int and (slots in (2, 4, 6)), 'V49 actual native/API/serial2/4/6 kind required')
    require(native_budgets == [32] * slots and all((type(v) is int for v in native_budgets)), 'Native bounded diagnostic mustdeclare actual32 perrequest')
    require(len(api_budgets) == slots and all((type(v) is int and 2 <= v <= 64 for v in api_budgets)) and (64 in api_budgets), 'Separate API longsurvivor64 declaration required')
    require(serial_job_count is None or (kind == 'serial' and type(serial_job_count) is int and (1 <= serial_job_count <= 6)), 'Serial actualselected group jobcount outside1..6')
    return {'harness_generation': 54, 'native_diagnostic_max_new_by_request': list(native_budgets), 'api_max_new_by_request': list(api_budgets), 'actual_submission_budget': list(api_budgets if kind == 'api' else [1] * serial_job_count if kind == 'serial' and serial_job_count is not None else [] if kind == 'serial' else native_budgets), 'natural_completion_qualified': False, 'native_64_claim': False, 'serial_first_head_max_new': 1, 'actual_serial_group_job_count': serial_job_count}

def manifest_binding(plan, *, pack_epoch=None, sdk_epoch=None, logical_epoch, logical_roster):
    require(__debug__ and sys.flags.optimize == 0 and (os.environ.get('PYTHONOPTIMIZE', '0') in ('', '0')), 'Strict frozen parser requires assertions enabled; optimized Python refused')
    require(plan['kind'] == 'serial', 'H51 serial-only scope; native/API require separate producers')
    require(canonical(diagnosis_binding(plan['binding_type_diagnosis']['path'])) == canonical(plan['binding_type_diagnosis']), 'Actual native-type diagnosis changed')
    spec = case_binding(plan, output_budget_contract)
    source_observers_off(plan['env'])
    require(plan['slots'] in (4, 6), 'New49 requested4/bounded6 only; historical2 must come from explicit privateV4')
    require(plan.get('schema') == 4 and plan.get('harness_generation') == 54, 'FrozenV6/older harness plan refused')
    budget = output_budget_contract(plan['kind'], plan['slots'], plan['native_diagnostic_max_new_by_request'], plan['api_max_new_by_request'], plan.get('serial_group_job_count'))
    require(plan['output_budget_contract'] == budget and plan['max_new_by_request'] == budget['actual_submission_budget'] and (plan['max_new'] == 32), 'Explicit V49 native32/API64 plan budget differs')
    require(set(plan['dependency_sha256']) == set(DEPENDENCIES), 'Complete immutable V49 dependency closure required')
    require(sha(plan['registry_binding']['path']) == plan['registry_binding']['sha256'], 'Derived eval registry changed')
    require(plan['driver_sha256'] == sha(Path(__file__)) and plan['source_plan_sha256'] == lane_contract(plan['lane'])['sha256'], 'Immutable arm source changed')
    for name, digest in plan['dependency_sha256'].items():
        require(sha(HERE / name) == digest, 'Consumed harness dependency changed ' + name)
    topology = topology_baselines(plan['topology_baselines']['onecard_root'], plan['topology_baselines']['pair_root'], plan['lane'], plan['topology_baselines']['onecard_adjudication'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(canonical(topology) == canonical(plan['topology_baselines']), 'Current genuineC137 one/pair sameSDK prerequisite changed')
    require(plan['case_source_binding'] == {'source_plan_sha256': PLAN_SHA, 'source_count': 64, 'header_count': 28, 'patch_count': 37, 'fresh_abi_count': 8, 'runtime_python_count': 6}, 'Actual newcase source generation metadata differs')
    if plan['slots'] > 2:
        from audit_batch_numerical_suite_v54 import paired_two_control
        require(canonical(paired_two_control(plan['paired_two_control'], plan['engine_receipt_sha256'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)) == canonical(plan['paired_two_control_binding']), 'Actual completed paired2 OFF/ON/full49 prerequisite changed')
    prepared, binding = genuine_baseline(Path(plan['prepared']), plan['lane'], plan.get('baseline_adjudication_receipt'), pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(binding['prepared_sha256'] == plan['prepared_sha256'] and prepared['engine_receipt_sha256'] == plan['engine_receipt_sha256'], 'Actual matching corrected current-PLE32/C137 baseline changed')
    profile_binding(plan, read(Path(plan['prepared']) / 'server-config.json'), prepared)
    require(case_source_gate(spec, Path(plan['engine_root'])) == plan['case_source_binding'], 'Authenticated case current SDK source differs')
    require(plan['spec_tokenizer_sha256'] == read(Path(plan['prepared']) / 'artifact-identity.json')['tokenizer_files'], 'Authenticated case and actual baseline tokenizer differ')
    if plan['kind'] == 'serial':
        from audit_batch50_logical_ports_v2 import parent_arm, vectors
        root = Path(plan['batch_parent'])
        parent, collector_plan, _ = parent_arm(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
        require(collector_plan['diagnostic'] == 1 and collector_plan['kind'] == 'native' and (collector_plan['slots'] == plan['slots']) and (collector_plan['engine_receipt_sha256'] == plan['engine_receipt_sha256']), 'Serial actual V49 collector/topology/source differs')
        require(vectors(root), 'Serial requires recollected full raw native48/head source corpus')
        require(parent['passed'] and parent['plan_sha256'] == plan['batch_parent_plan_sha256'] and (sha(root / 'parent-qualification.json') == plan['batch_parent_sha256']), 'Actual source-qualified collector parent changed')
        require(sha(root / 'child/serial-jobs.json') == plan['serial_jobs_sha256'], 'Actual collected consumed-prefix jobs changed')
        from batch51_serial_jobs_v1 import recollect, group
        job_value = recollect(root / 'child', plan['slots'])
        jobs = job_value['jobs']
        selected = group(job_value, plan['slots'], plan['group_index'])
        require(canonical(selected) == canonical(plan['actual_serial_group_binding']), 'Actual selected full49 job group changed')
        require(plan['serial_group_job_count'] == len(jobs[plan['group_index'] * 6:plan['group_index'] * 6 + 6]), 'Serial actualsubmitted group jobcount changed')
    return binding

def prepare(a, *, pack_epoch=None, sdk_epoch=None, logical_epoch=None, logical_roster=None):
    require(a.kind == 'serial', 'H50 native4/6 handshake source lane only; API/serial refused')
    require(a.kind in ('native', 'serial'), 'Unsupported API cache0 purpose rejected before any baseline/model admission; separate API purpose required')
    diagnosis = diagnosis_binding(a.binding_type_diagnosis)
    pack_epoch = pack_epoch or for_prepared(a.prepared, 3600)
    sdk_epoch = sdk_epoch or sdk_for_prepare(a, 3600)
    if logical_epoch is None:logical_epoch,logical_roster,logical_scope=logical_for_prepare(a)
    topology = topology_baselines(a.one_card_baseline, a.pair_baseline, a.lane, a.one_card_adjudication, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    chosen_adjudication = a.one_card_adjudication if a.prepared.resolve() == a.one_card_baseline.resolve() else None
    prepared, binding = genuine_baseline(a.prepared, a.lane, chosen_adjudication, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(prepared['engine_receipt_sha256'] == topology['engine_receipt_sha256'], 'Chosen preparation not sameSDK as finalizedone/pair')
    spec = authenticated_case(a.spec, read(a.spec)['slots'])
    source_case_binding = case_source_gate(spec, Path(prepared['engine_receipt']).parent)
    require(spec['source_lane'] == a.lane, 'Case source lane differs')
    n = spec['slots']
    require(type(n) is int and n in (4, 6), 'Requested2/4/6 scope required')
    cfg = read(a.prepared / 'server-config.json')
    args, env = configuration(cfg, n)
    require(a.kind in ('native', 'serial'), 'Unsupported API cache0 purpose rejected before preparation; separate API proof still required')
    source_observers_off(env)
    tokens = spec['tokens']
    messages = spec['messages']
    require(len(tokens['warm']) == len(tokens['target']) == len(messages['warm']) == len(messages['target']) == n, 'Exact logical native/API cohorts required')
    for ids in tokens['warm'] + tokens['target']:
        require(1 <= len(ids) <= 2048 and all((type(t) is int and 0 <= t < 248320 for t in ids)), 'Bounded actual submitted token vectors required')
    require(all((x[0] != y[0] for x in tokens['warm'] for y in tokens['target'])), 'Warm native complete prefixes must be unrelated to target')
    require(len({json.dumps(m, sort_keys=True) for m in messages['target']}) == n, 'Unique API target messages needed for actual call joins')
    require(spec.get('schema') == 2 and spec.get('harness_generation') == 50, 'New explicit V49 transport-budget case required; frozenV6 refused')
    budget = output_budget_contract(a.kind, n, spec['native_diagnostic_max_new_by_request'], spec['api_max_new_by_request'])
    require(spec['max_new_by_request'] == spec['api_max_new_by_request'], 'Case generic max_new is explicitly the API budget only')
    require(0 <= spec['cancel_index'] < n, 'Bounded real cancellation logicalindex required')
    for row in spec['actual_counter_policy']:
        require(0 <= row['logical_index'] < n and row['role'] in ('admission', 'solo_migration') and (row['values'] == {'reused': 0, 'read_from': 0, 'reread_to': -1}), 'Pilot exact source no-cache/no-checkpoint counters must0/0/-1; reuse/read_from guards unchanged')
    require({(row['logical_index'], row['role']) for row in spec['actual_counter_policy']} == {(i, 'admission') for i in range(n)} | {(1, 'solo_migration')} and len(spec['actual_counter_policy']) == n + 1, 'Exact complete admission/survivor counter roster required')
    require(len(spec['api_token_ids']['warm']) == len(spec['api_token_ids']['target']) == n and spec['api_token_ids']['target'] == tokens['target'], 'Actual API/native target input corpus identity differs')
    for name, digest in spec['source_sha256'].items():
        require(sha(Path(prepared['engine_receipt']).parent / 'source' / name) == digest, 'Case consumed source identity changed ' + name)
    require(spec['tokenizer_sha256'] == read(a.prepared / 'artifact-identity.json')['tokenizer_files'], 'Case tokenizer/template identity changed')
    pair2_binding = None
    if n > 2:
        from audit_batch_numerical_suite_v54 import paired_two_control
        require('paired_two_control' in spec, 'Actual completed paired2 OFF/ON/full49 control required before4/6')
        pair2_binding = paired_two_control(spec['paired_two_control'], prepared['engine_receipt_sha256'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    identity = verify_model_identity(a.model_identity, read(HERE / 'model-lock.json'), [Path(row['path']) for row in prepared['model_shards']])
    plan = {'binding_type_diagnosis': diagnosis, 'schema': 4, 'harness_generation': 54, 'topology_baselines': topology, 'baseline_adjudication_receipt': str(Path(chosen_adjudication).resolve()) if chosen_adjudication is not None else None, 'case_source_binding': source_case_binding, 'native_diagnostic_max_new_by_request': spec['native_diagnostic_max_new_by_request'], 'api_max_new_by_request': spec['api_max_new_by_request'], 'output_budget_contract': budget, 'lane': a.lane, 'kind': a.kind, 'diagnostic': a.diagnostic, 'prepared': str(a.prepared.resolve()), 'prepared_sha256': binding['prepared_sha256'], 'baseline_source_proof': binding, 'engine_root': str(Path(prepared['engine_receipt']).parent), 'engine_receipt_sha256': prepared['engine_receipt_sha256'], 'source_plan_sha256': lane_contract(a.lane)['sha256'], 'driver_sha256': sha(Path(__file__)), 'native_driver_sha256': sha(Path(native.__file__)), 'dependency_sha256': {name: sha(HERE / name) for name in DEPENDENCIES}, 'cards': prepared['cards'], 'args': args, 'env': env, 'image': prepared['runtime']['image'], 'pack': prepared['pack'], 'model_identity': identity, 'slots': n, 'stage_ranges': expected_stage_ranges(args), 'api_token_ids': spec['api_token_ids'], 'tokens': tokens, 'messages': messages, 'cancel_index': spec['cancel_index'], 'max_new_by_request': budget['actual_submission_budget'], 'actual_counter_policy': spec['actual_counter_policy'], 'port': spec['port'], 'spec_path': str(a.spec.resolve()), 'spec_sha256': sha(a.spec), 'spec_tokenizer_sha256': spec['tokenizer_sha256'], 'max_new': 32, 'full_model_math_qualified': False, 'public_cache_lane_required_separately': True, 'serving_mirror_owner_proof': 'Mandatory actual same-process mirror31 ownership under corrected current-PLE32'}
    if n > 2:
        plan['paired_two_control'] = spec['paired_two_control']
        plan['paired_two_control_binding'] = pair2_binding
    alias = api.experimental_alias({'args': args, 'env': env}, n, bool(a.diagnostic), a.lane)
    plan['registry_binding'] = api.registry_gate(alias)
    plan['research_alias'] = alias
    if a.kind == 'serial':
        require(a.batch_parent and a.group_index is not None, 'Actual finalized collector parent/group required')
        from audit_batch50_logical_ports_v2 import parent_arm, vectors
        parent, source, _ = parent_arm(a.batch_parent, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
        require(source['kind'] == 'native' and vectors(a.batch_parent), 'Actual finalized V49 native/API collector raw corpus required')
        require(parent['passed'], 'Collector source/lifecycle parent failed')
        source = read(a.batch_parent / 'input-plan.snapshot.json')
        require(source.get('harness_generation') == 50 and source['diagnostic'] == 1 and (source['slots'] == n) and (source['engine_receipt_sha256'] == plan['engine_receipt_sha256']), 'Actual diagnostic collector/source configuration differs')
        from batch51_serial_jobs_v1 import recollect, group
        job_value = recollect(a.batch_parent / 'child', n)
        jobs = job_value['jobs']
        selected = group(job_value, n, a.group_index)
        require(0 <= a.group_index < (len(jobs) + 5) // 6, 'Bounded actual serial group not present')
        plan.update(actual_serial_group_binding=selected, batch_parent=str(a.batch_parent.resolve()), batch_parent_sha256=sha(a.batch_parent / 'parent-qualification.json'), batch_parent_plan_sha256=parent['plan_sha256'], serial_jobs_sha256=sha(a.batch_parent / 'child/serial-jobs.json'), group_index=a.group_index, serial_group_job_count=len(jobs[a.group_index * 6:a.group_index * 6 + 6]))
        plan['output_budget_contract'] = output_budget_contract('serial', n, spec['native_diagnostic_max_new_by_request'], spec['api_max_new_by_request'], plan['serial_group_job_count'])
        plan['max_new_by_request'] = plan['output_budget_contract']['actual_submission_budget']
    manifest_binding(plan, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    logical_epoch.seal_predevice(current_roster=logical_roster)
    pack_epoch.seal_predevice()
    sdk_epoch.seal_predevice()
    pack_proof = pack_epoch.finalize()
    sdk_proof = sdk_epoch.finalize()
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / 'plan.json', plan)
    write(a.output / 'prepare-pack-operation-witness.json', pack_proof)
    write(a.output / 'prepare-sdk-operation-witness.json', sdk_proof)
    write(a.output/'prepare-logical-operation-witness.json',logical_epoch.finalize(current_roster=logical_roster))
    print('PREPARED single actual-process arm; no execution or mathematical qualification')

def run(a, *, pack_epoch=None, sdk_epoch=None, logical_epoch=None, logical_roster=None):
    c1.leased([0, 1])
    admitted = Snapshot(a.plan)
    plan = admitted.plan
    from batch50_byte_epochs_v1 import pack_for_prepared, sdk_for_plan as sdk_child_for_plan
    require(plan['kind'] == 'serial', 'H50 owned health handshake supports native4/6 only; serial separate successor required')
    pack_epoch = pack_epoch or pack_for_prepared(Path(plan['prepared']), 10800)
    sdk_epoch = sdk_epoch or sdk_child_for_plan(plan, 10800)
    if logical_epoch is None:logical_epoch,logical_roster,logical_scope=logical_for_plan(plan)
    manifest_binding(plan, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    admitted.verify(plan)

    def stop(signum, frame):
        api.ABORT.set()
        raise KeyboardInterrupt('Parent requested owned arm stop')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    result = serial.run(plan, Path(plan['batch_parent']) / 'child', a.output, a.pre_health, plan['group_index'], admitted.raw, handshake=a.handshake, admitted=admitted, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    write_snapshot(a.output / 'plan.snapshot.json', plan, admitted.raw)
    pack_proof = pack_epoch.finalize()
    sdk_proof = sdk_epoch.finalize()
    write(a.output / 'pack-operation-witness.json', pack_proof)
    write(a.output / 'sdk-operation-witness.json', sdk_proof)
    write(a.output/'logical-operation-witness.json',logical_epoch.finalize(current_roster=logical_roster))
    result.update(collection_and_teardown_passed=result['passed'], finished_epoch=time.time(), plan_sha256=sha(a.output / 'plan.snapshot.json'), actual_serial_job_count=plan['serial_group_job_count'], actual_matched_full49_vector_pairs=len(result['comparisons']), full_model_math_qualified=False)
    admitted.verify(plan)
    result['artifact_bindings'] = artifact_bindings(a.output)
    write(a.output / 'report.json', result)
    return 0 if result['passed'] else 1

def main():
    pack_epoch = None
    sdk_epoch = None
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--lane', choices=['source37'], required=True)
    a.add_argument('--kind', choices=['native', 'api', 'serial'], required=True)
    a.add_argument('--diagnostic', type=int, choices=[0, 1], required=True)
    a.add_argument('--prepared', type=Path, required=True)
    a.add_argument('--one-card-baseline', type=Path, required=True)
    a.add_argument('--pair-baseline', type=Path, required=True)
    a.add_argument('--one-card-adjudication', type=Path)
    a.add_argument('--binding-type-diagnosis', type=Path, required=True)
    a.add_argument('--spec', type=Path, required=True)
    a.add_argument('--model-identity', type=Path, required=True)
    a.add_argument('--batch-parent', type=Path)
    a.add_argument('--group-index', type=int)
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('run')
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--pre-health', type=Path, required=True)
    a.add_argument('--handshake', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    return prepare(a, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch) if a.mode == 'prepare' else run(a, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
if __name__ == '__main__':
    raise SystemExit(main())
