# NEW explicit pack epoch admission port of batch_numerical_execution_v40.py
from batch_numerical_execution_v40 import *
from batch_proofs_pack49_v1 import genuine_baseline, topology_baselines

def manifest_binding(plan, *, pack_epoch, sdk_epoch):
    require(plan['slots'] == 2, 'Only actual historical paired2 corpus is a prerequisite; no other source scope transfer')
    require(__debug__ and sys.flags.optimize == 0 and (os.environ.get('PYTHONOPTIMIZE', '0') in ('', '0')), 'Strict frozen parser requires assertions enabled; optimized Python refused')
    source_observers_off(plan['env'])
    require(plan.get('schema') == 4 and plan.get('harness_generation') == 40, 'FrozenV6/older harness plan refused')
    budget = output_budget_contract(plan['kind'], plan['slots'], plan['native_diagnostic_max_new_by_request'], plan['api_max_new_by_request'], plan.get('serial_group_job_count'))
    require(plan['output_budget_contract'] == budget and plan['max_new_by_request'] == budget['actual_submission_budget'] and (plan['max_new'] == 32), 'Explicit V40 native32/API64 plan budget differs')
    require(set(plan['dependency_sha256']) == set(DEPENDENCIES), 'Complete immutable V40 dependency closure required')
    require(sha(plan['registry_binding']['path']) == plan['registry_binding']['sha256'], 'Derived eval registry changed')
    require(plan['driver_sha256'] == sha(Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/batch_numerical_execution_v40.py')) and plan['source_plan_sha256'] == lane_contract(plan['lane'])['sha256'], 'Immutable arm source changed')
    for name, digest in plan['dependency_sha256'].items():
        require(sha(HERE / name) == digest, 'Consumed harness dependency changed ' + name)
    topology = topology_baselines(plan['topology_baselines']['onecard_root'], plan['topology_baselines']['pair_root'], plan['lane'], plan['topology_baselines']['onecard_adjudication'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
    require(topology == plan['topology_baselines'], 'Current genuineC137 one/pair sameSDK prerequisite changed')
    require(plan['case_source_binding'] == {'source_plan_sha256': PLAN_SHA, 'source_count': 64, 'header_count': 28, 'patch_count': 37, 'fresh_abi_count': 8, 'runtime_python_count': 6}, 'Actual newcase source generation metadata differs')
    prepared, binding = genuine_baseline(Path(plan['prepared']), plan['lane'], plan.get('baseline_adjudication_receipt'), pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
    require(binding['prepared_sha256'] == plan['prepared_sha256'] and prepared['engine_receipt_sha256'] == plan['engine_receipt_sha256'], 'Actual matching corrected current-PLE32/C137 baseline changed')
    if plan['kind'] == 'serial':
        from audit_batch_pack49_v1 import parent_arm, vectors
        root = Path(plan['batch_parent'])
        parent, collector_plan, _ = parent_arm(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
        require(collector_plan['diagnostic'] == 1 and collector_plan['kind'] in ('native', 'api') and (collector_plan['slots'] == plan['slots']) and (collector_plan['engine_receipt_sha256'] == plan['engine_receipt_sha256']), 'Serial actual V40 collector/topology/source differs')
        require(vectors(root), 'Serial requires recollected full raw native48/head source corpus')
        require(parent['passed'] and parent['plan_sha256'] == plan['batch_parent_plan_sha256'] and (sha(root / 'parent-qualification.json') == plan['batch_parent_sha256']), 'Actual source-qualified collector parent changed')
        require(sha(root / 'child/serial-jobs.json') == plan['serial_jobs_sha256'], 'Actual collected consumed-prefix jobs changed')
        jobs = read(root / 'child/serial-jobs.json')['jobs']
        require(plan['serial_group_job_count'] == len(jobs[plan['group_index'] * 6:plan['group_index'] * 6 + 6]), 'Serial actualsubmitted group jobcount changed')
    return binding
