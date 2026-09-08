#!/usr/bin/env python3
"""CPU regression against the installed R276 manager, adapted from PR 48375."""
import argparse
import json
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import MambaManager, FullAttentionManager
from vllm.v1.kv_cache_interface import MambaSpec, FullAttentionSpec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expect-bug', action='store_true')
    args = p.parse_args()
    rows = []
    for n in (0, 1, 5):
        pool = BlockPool(64, True, 16)
        hashes = [BlockHash(('blk%04d' % i).encode()) for i in range(n)]
        for gid in (0, 1):
            for i, (h, b) in enumerate(zip(hashes, pool.get_new_blocks(n))):
                pool._insert_block_hash(make_block_hash_with_group_id(h, gid), b, (i + 1) * 16)
        specs = [FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16),
                 MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float16,), mamba_cache_mode='align')]
        for drop in (False, True):
            hits = []
            for gid, manager in enumerate((FullAttentionManager, MambaManager)):
                blocks, hit = manager.find_longest_cache_hit(
                    block_hashes=hashes, max_length=n * 16, kv_cache_group_ids=[gid],
                    block_pool=pool, kv_cache_spec=specs[gid], drop_eagle_block=drop,
                    alignment_tokens=16)
                assert len(blocks[0]) * 16 == hit
                if hit and gid == 1:
                    assert blocks[0][-1] is not pool.null_block
                    assert all(b is pool.null_block for b in blocks[0][:-1])
                hits.append(hit)
            expected = max(0, n - int(drop)) * 16
            assert hits[0] == expected
            assert hits[1] == (n * 16 if args.expect_bug else expected), (n, drop, hits)
            rows.append(dict(blocks=n, drop=drop, full_hit=hits[0], mamba_hit=hits[1]))
    print(json.dumps(dict(expected_bug=args.expect_bug, passed=True, cases=rows), indent=2))


if __name__ == '__main__':
    main()
