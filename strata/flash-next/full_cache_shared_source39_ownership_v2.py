"""Current source39 trace ownership, without importing historical runtime proofs."""
from audit_batch_capture_lifecycle_v2 import audit as snapshot_audit
from audit_stage_mirror_owner_trace_v1 import audit as mirror_audit
from batch_numerical_proofs_v40 import slot_ownership


def ownership(trace, stages, slots):
    if type(slots) is not int or slots != 2:
        raise ValueError('Preregistered source39 two-slot actor required')
    if not stages or len(stages) not in (1, 2) or any(type(v) is not int for row in stages for v in row):
        raise ValueError('Actual typed source39 stage roster required')
    if len({stage for stage, lo, hi in stages}) != len(stages) or any(not 0 <= lo < hi <= 48 for stage, lo, hi in stages):
        raise ValueError('Actual distinct source39 stage ownership required')
    if sorted(l for _, lo, hi in stages for l in range(lo, hi)) != list(range(48)):
        raise ValueError('Actual complete source39 stage roster required')
    snapshots = snapshot_audit(trace, stages)
    private = slot_ownership(trace, [(stage, slot) for stage, lo, hi in stages for slot in range(slots)])
    mirrors = mirror_audit(trace, [(stage, stage, lo, hi) for stage, lo, hi in stages])
    if not all(result['passed'] is True for result in (snapshots, private, mirrors)):
        raise ValueError('Actual source39 snapshot/slot/mirror logical retirement missing')
    return {'source_lane': 'source39', 'snapshots': snapshots, 'private_slots': private,
            'mirror': mirrors, 'actual_trace_logical_retirements_observed': True,
            'historical_source37_runtime_proof_transferred': False,
            'physical_reclamation_qualified': False, 'full_cache_runtime_qualified': False}
