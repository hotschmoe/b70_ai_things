"""Byte layout of one QSA page. No torch.

Matches flashnext_ref.qsa_cache.make_layout. The SYCL kernels recompute
the same offsets and reject a page whose byte size disagrees.

Region order: K codes int8, V codes int8, K scales fp16, V scales fp16,
then completed pooled indexer rows f32. Draft pages omit the pooled tail.
"""

from __future__ import annotations


def page_layout(
    n_kv: int,
    head_dim: int,
    idx_dim: int,
    idx_block: int,
    block_size: int,
    *,
    pooled: bool = True,
) -> dict[str, int]:
    if block_size < 1 or idx_block < 2 or block_size % idx_block:
        raise ValueError("block_size must be a positive multiple of idx_block")
    if head_dim % 64:
        raise ValueError("head_dim must be a multiple of 64")
    if n_kv < 1 or idx_dim < 1:
        raise ValueError("heads")
    groups = head_dim // 64
    k_codes = n_kv * block_size * head_dim
    v_codes = k_codes
    k_scales = n_kv * block_size * groups * 2
    v_scales = k_scales
    kv = k_codes + v_codes + k_scales + v_scales
    if kv % 4:
        raise ValueError("kv region must be 4-byte aligned")
    n_states = block_size // idx_block
    pooled_bytes = n_states * idx_dim * 4 if pooled else 0
    return {
        "block_size": int(block_size),
        "n_kv": int(n_kv),
        "head_dim": int(head_dim),
        "groups": int(groups),
        "idx_dim": int(idx_dim),
        "idx_block": int(idx_block),
        "n_states": int(n_states),
        "k_codes_off": 0,
        "v_codes_off": k_codes,
        "k_scale_off": k_codes + v_codes,
        "v_scale_off": k_codes + v_codes + k_scales,
        "pooled_off": kv,
        "kv_bytes": int(kv),
        "pooled_bytes": int(pooled_bytes),
        "page_bytes": int(kv + pooled_bytes),
        "pooled": 1 if pooled else 0,
    }
