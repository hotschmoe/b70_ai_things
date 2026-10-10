# NEW mandatory immutable logical-byte port of batch_numerical_execution_v50.py
from upload_logical_port_identity_v1 import original_producer_file, original_module_file
from batch_numerical_execution_v50 import *
from batch_numerical_proofs_logical_ports_v1 import genuine_baseline, topology_baselines

def manifest_binding(plan, *, pack_epoch=None, sdk_epoch=None, logical_epoch, logical_roster):
    require(__debug__ and sys.flags.optimize == 0 and (os.environ.get('PYTHONOPTIMIZE', '0') in ('', '0')), 'Strict frozen parser requires assertions enabled; optimized Python refused')
    require(plan['kind'] in ('native', 'serial'), 'Unsupported API cache0 mode must not reach expensive admission or leases')
    require(canonical(diagnosis_binding(plan['binding_type_diagnosis']['path'])) == canonical(plan['binding_type_diagnosis']), 'Actual native-type diagnosis changed')
    spec = case_binding(plan, output_budget_contract)
    source_observers_off(plan['env'])
    require(plan['slots'] in (4, 6), 'New49 requested4/bounded6 only; historical2 must come from explicit privateV4')
    require(plan.get('schema') == 4 and plan.get('harness_generation') == 50, 'FrozenV6/older harness plan refused')
    budget = output_budget_contract(plan['kind'], plan['slots'], plan['native_diagnostic_max_new_by_request'], plan['api_max_new_by_request'], plan.get('serial_group_job_count'))
    require(plan['output_budget_contract'] == budget and plan['max_new_by_request'] == budget['actual_submission_budget'] and (plan['max_new'] == 32), 'Explicit V49 native32/API64 plan budget differs')
    require(set(plan['dependency_sha256']) == set(DEPENDENCIES), 'Complete immutable V49 dependency closure required')
    require(sha(plan['registry_binding']['path']) == plan['registry_binding']['sha256'], 'Derived eval registry changed')
    require(plan['driver_sha256'] == sha(Path(original_producer_file('batch_numerical_execution_v50'))) and plan['source_plan_sha256'] == lane_contract(plan['lane'])['sha256'], 'Immutable arm source changed')
    for name, digest in plan['dependency_sha256'].items():
        require(sha(HERE / name) == digest, 'Consumed harness dependency changed ' + name)
    topology = topology_baselines(plan['topology_baselines']['onecard_root'], plan['topology_baselines']['pair_root'], plan['lane'], plan['topology_baselines']['onecard_adjudication'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(canonical(topology) == canonical(plan['topology_baselines']), 'Current genuineC137 one/pair sameSDK prerequisite changed')
    require(plan['case_source_binding'] == {'source_plan_sha256': PLAN_SHA, 'source_count': 64, 'header_count': 28, 'patch_count': 37, 'fresh_abi_count': 8, 'runtime_python_count': 6}, 'Actual newcase source generation metadata differs')
    if plan['slots'] > 2:
        from audit_batch50_logical_ports_v1 import paired_two_control
        require(canonical(paired_two_control(plan['paired_two_control'], plan['engine_receipt_sha256'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)) == canonical(plan['paired_two_control_binding']), 'Actual completed paired2 OFF/ON/full49 prerequisite changed')
    prepared, binding = genuine_baseline(Path(plan['prepared']), plan['lane'], plan.get('baseline_adjudication_receipt'), pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(binding['prepared_sha256'] == plan['prepared_sha256'] and prepared['engine_receipt_sha256'] == plan['engine_receipt_sha256'], 'Actual matching corrected current-PLE32/C137 baseline changed')
    profile_binding(plan, read(Path(plan['prepared']) / 'server-config.json'), prepared)
    require(case_source_gate(spec, Path(plan['engine_root'])) == plan['case_source_binding'], 'Authenticated case current SDK source differs')
    require(plan['spec_tokenizer_sha256'] == read(Path(plan['prepared']) / 'artifact-identity.json')['tokenizer_files'], 'Authenticated case and actual baseline tokenizer differ')
    if plan['kind'] == 'serial':
        from audit_batch50_logical_ports_v1 import parent_arm, vectors
        root = Path(plan['batch_parent'])
        parent, collector_plan, _ = parent_arm(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
        require(collector_plan['diagnostic'] == 1 and collector_plan['kind'] in ('native', 'api') and (collector_plan['slots'] == plan['slots']) and (collector_plan['engine_receipt_sha256'] == plan['engine_receipt_sha256']), 'Serial actual V49 collector/topology/source differs')
        require(vectors(root), 'Serial requires recollected full raw native48/head source corpus')
        require(parent['passed'] and parent['plan_sha256'] == plan['batch_parent_plan_sha256'] and (sha(root / 'parent-qualification.json') == plan['batch_parent_sha256']), 'Actual source-qualified collector parent changed')
        require(sha(root / 'child/serial-jobs.json') == plan['serial_jobs_sha256'], 'Actual collected consumed-prefix jobs changed')
        jobs = read(root / 'child/serial-jobs.json')['jobs']
        require(plan['serial_group_job_count'] == len(jobs[plan['group_index'] * 6:plan['group_index'] * 6 + 6]), 'Serial actualsubmitted group jobcount changed')
    return binding
