"""Explicit SDK37 scope for every declared baseline and selected preparation."""
from pathlib import Path
from sdk37_witness_scope_v1 import create_epoch

def roots(one,pair,selected):return list(dict.fromkeys(Path(p).resolve()/'prepared.json' for p in (one,pair,selected)))
def for_plan(plan,max_seconds=10800):return create_epoch(roots(plan['topology_baselines']['onecard_root'],plan['topology_baselines']['pair_root'],plan['prepared']),max_seconds)
def for_prepare(args,max_seconds=3600):return create_epoch(roots(args.one_card_baseline,args.pair_baseline,args.prepared),max_seconds)
