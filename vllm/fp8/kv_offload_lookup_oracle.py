#!/usr/bin/env python3
"""CPU-only sparse Mamba checkpoint lookup oracle against installed code."""
import json
from types import SimpleNamespace as NS
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.v1.kv_offload.base import LookupResult


def probe(mark_mamba_eagle):
    s = object.__new__(OffloadingConnectorScheduler)
    s.config = NS(kv_group_configs=[NS(group_idx=i, tokens_per_chunk=832, is_eagle_group=(i==0 or mark_mamba_eagle), sliding_window_size_in_chunks=None if i==0 else 1) for i in range(4)])
    s._sliding_window_groups = (1,2,3)
    s._lookup_groups = (0,1,2,3)
    s._mamba_align_size = 832
    s._chunks_being_loaded = None
    s._events_tracker = NS(record_lookup=lambda *_: None)
    stored = {(0,i) for i in range(180)} | {(g,i) for g in (1,2,3) for i in (38,77,116,155,179)}
    s.manager = NS(lookup=lambda key, _: LookupResult.HIT if key in stored else LookupResult.MISS)
    status = NS(num_locally_computed_tokens=0, req=NS(num_tokens=150045, request_id='oracle'), req_context=None,
                group_states=[NS(offload_keys=[(g,i) for i in range(180)]) for g in range(4)])
    return s._lookup_complete_chunks(status)


stock = probe(True)
attention_only = probe(False)
assert stock == 0
assert attention_only > 120000
print(json.dumps({'stock_all_groups_eagle_hit_tokens':stock, 'attention_only_eagle_hit_tokens':attention_only, 'same_stored_keys':True}))
