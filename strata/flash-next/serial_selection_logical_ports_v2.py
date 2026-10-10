# NEW mandatory immutable logical-byte port of serial_selection_pack49_v1.py
from upload_logical_port_identity_v2 import original_producer_file, original_module_file
'Explicit selected job roster; no silent skip or rewritten source job list.'
from pathlib import Path
from batch_proofs_logical_ports_v2 import require, read, sha
from serial37_canonical_json_v3 import matches_saved, read_unique

def selected_jobs(plan, jobs, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    group = jobs[plan['group_index'] * 6:plan['group_index'] * 6 + 6]
    indices = plan['actual_serial_selected_indices']
    require(indices == list(range(len(group))) or (indices == list(range(1, len(group))) and len(group) == 4), 'Only complete group or declared remaining-three selection permitted')
    if indices != list(range(len(group))):
        import first49_logical_ports_v2 as recovery
        path = Path(plan['first_job_adjudication']['path'])
        saved = read_unique(path)
        require(sha(path) == plan['first_job_adjudication']['sha256'] and matches_saved(recovery.finalized_binding(pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster), path), 'Actual exact readonly first-job recovery changed')
        require(saved['job'] == group[0] and saved['engine_receipt_sha256'] == plan['engine_receipt_sha256'] and (saved['collector_parent_sha256'] == plan['batch_parent_sha256']) and (saved['original_parent_passed'] is False) and (saved['new_absent_PIN_scope_qualified'] is False), 'Actual first-job source/collector/historical exception association differs')
    else:
        require(plan['first_job_adjudication'] is None, 'Complete fresh group must not borrow recovery')
    require(plan['actual_serial_job_count'] == len(indices) and plan['actual_serial_submission_budgets'] == [1] * len(indices), 'Exact actual remaining submission roster/budget differs')
    return [group[i] for i in indices]
