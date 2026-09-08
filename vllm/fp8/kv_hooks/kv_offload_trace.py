"""Read-only offload lookup diagnostics; no cache or scheduler policy changes."""
import json
import os
from pathlib import Path


def install():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
    original = OffloadingConnectorScheduler._lookup
    def lookup(self, status):
        result = original(self, status)
        policy = getattr(self.manager, '_policy', None)
        groups = []
        for cfg, state in zip(self.config.kv_group_configs, status.group_states):
            ready = []
            if policy is not None:
                for i, key in enumerate(state.offload_keys):
                    block = policy.get(key)
                    if block is not None and block.is_ready:
                        ready.append(i)
            groups.append({'group': cfg.group_idx, 'eagle': cfg.is_eagle_group, 'cow': cfg.requires_cow_source,
                           'window': cfg.sliding_window_size_in_chunks, 'tokens_per_chunk': cfg.tokens_per_chunk,
                           'num_keys': len(state.offload_keys), 'ready_indices': ready})
        row = {'request': status.req.request_id, 'tokens': status.req.num_tokens, 'locally_computed': status.num_locally_computed_tokens,
               'lookup_result': result, 'groups': groups, 'cpu_blocks': getattr(self.manager, '_num_blocks', None)}
        with (Path(os.environ['B70_KV_OUT']) / ('offload-trace-' + str(os.getpid()) + '.jsonl')).open('a') as f:
            f.write(json.dumps(row) + '\n')
        return result
    OffloadingConnectorScheduler._lookup = lookup
