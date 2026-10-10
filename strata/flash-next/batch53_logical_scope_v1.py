"""Independent per-process logical byte epoch scope and actual four child edges."""
from pathlib import Path
from upload_logical_epoch_factory_v1 import for_prepared
from sdk49_operation_scope_v1 import roots
from logical_free_require_case_epoch_v2 import RequireLogicalEpoch

def for_plan(plan):
 paths=[p.parent for p in roots(plan['topology_baselines']['onecard_root'],plan['topology_baselines']['pair_root'],plan['prepared'])];return for_prepared(paths,10800)
def for_prepare(args):
 paths=[p.parent for p in roots(args.one_card_baseline,args.pair_baseline,args.prepared)];return for_prepared(paths,10800)
def ready(epoch,roster):
 if type(epoch)is not RequireLogicalEpoch:raise ValueError('Exact current logical epoch required')
 RequireLogicalEpoch.implementation_binding(epoch);epoch.owned(roster)
 if epoch.evidence.phase!='admission' or len(epoch.evidence.boundaries)!=1:raise ValueError('Exactly one independent logical entry before READY')
 epoch.evidence._fresh('ready');return epoch.evidence.boundaries[-1]
