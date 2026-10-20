"""Lazy loader for the b70 QSA SYCL extension.

Decode calls are in-place. The schema lives in the SYCL library. Fake kernels
cover Dynamo / FakeTensor. Wrappers do not sync and do not allocate when the
caller already passed contiguous tensors.
"""

from __future__ import annotations

import os

import torch

_LOADED = False
_FAKES = False

DEFAULT_SO = "/mnt/vm_8tb/b70/build/vllm-flashnext/b70_qsa_xpu/libb70_qsa_xpu.so"


def _register_fakes() -> None:
    global _FAKES
    if _FAKES:
        return
    from torch.library import register_fake

    @register_fake("b70_qsa::write_step")
    def _write_step_fake(pos, step, top_k, block):
        del pos, top_k, block
        return step.new_empty(step.shape)

    @register_fake("b70_qsa::kv_append")
    def _kv_fake(k, v, k_codes, v_codes, k_scale, v_scale, page_table, step):
        del k, v, v_codes, k_scale, v_scale, page_table, step
        return k_codes.new_empty(k_codes.shape)

    @register_fake("b70_qsa::indexer_append")
    def _idx_fake(raw, k_norm, tail, dead, pooled, block_pos, cos, sin, step, mrope, pos_base, eps):
        del raw, k_norm, tail, dead, block_pos, cos, sin, step, mrope, pos_base, eps
        return pooled.new_empty(pooled.shape)

    @register_fake("b70_qsa::index_scores")
    def _scores_fake(q_idx, pooled, dead, scores, step, bias, block):
        del q_idx, pooled, dead, step, bias, block
        return scores.new_empty(scores.shape)

    @register_fake("b70_qsa::topk")
    def _topk_fake(scores, ids, step):
        del scores, step
        return ids.new_empty(ids.shape)

    @register_fake("b70_qsa::kv_gather")
    def _gather_fake(ids, k_codes, v_codes, k_scale, v_scale, page_table, k_out, v_out, step):
        del ids, k_codes, v_codes, k_scale, v_scale, page_table, v_out, step
        return k_out.new_empty(k_out.shape)

    @register_fake("b70_qsa::attend")
    def _attend_fake(q, k_f16, v_f16, attn, step):
        del q, k_f16, v_f16, step
        return attn.new_empty(attn.shape)

    @register_fake("b70_qsa::gate")
    def _gate_fake(attn, q_full, out):
        del attn, q_full
        return out.new_empty(out.shape)

    @register_fake("b70_qsa::write_step_batch")
    def _write_step_batch_fake(pos, step, top_k, block):
        del pos, top_k, block
        return step.new_empty(step.shape)

    @register_fake("b70_qsa::kv_append_batch")
    def _kv_batch_fake(k, v, pages, block_table, step, block_size, idx_block, idx_dim, has_pooled):
        del k, v, block_table, step, block_size, idx_block, idx_dim, has_pooled
        return pages.new_empty(pages.shape)

    @register_fake("b70_qsa::indexer_append_batch")
    def _idx_batch_fake(
        raw, k_norm, tail, dead, block_pos, pages, block_table, cos, sin, step, mrope, pos_base, eps, block_size
    ):
        del raw, k_norm, tail, dead, block_pos, block_table, cos, sin, step, mrope, pos_base, eps, block_size
        return pages.new_empty(pages.shape)

    @register_fake("b70_qsa::index_scores_batch")
    def _scores_batch_fake(q_idx, pages, dead, block_table, seq_slot, scores, step, bias, block_size, idx_block):
        del q_idx, pages, dead, block_table, seq_slot, step, bias, block_size, idx_block
        return scores.new_empty(scores.shape)

    @register_fake("b70_qsa::topk_batch")
    def _topk_batch_fake(scores, ids, step):
        del scores, step
        return ids.new_empty(ids.shape)

    @register_fake("b70_qsa::kv_gather_batch")
    def _gather_batch_fake(
        ids, pages, block_table, seq_slot, k_out, v_out, step, block_size, idx_block, idx_dim, has_pooled
    ):
        del ids, pages, block_table, seq_slot, v_out, step, block_size, idx_block, idx_dim, has_pooled
        return k_out.new_empty(k_out.shape)

    @register_fake("b70_qsa::attend_batch")
    def _attend_batch_fake(q, k_f16, v_f16, attn, step):
        del q, k_f16, v_f16, step
        return attn.new_empty(attn.shape)

    @register_fake("b70_qsa::gate_batch")
    def _gate_batch_fake(attn, q_full, out):
        del attn, q_full
        return out.new_empty(out.shape)

    @register_fake("b70_qsa::union_build")
    def _union_build_fake(ids, step, ulist, ucount):
        del ids, step, ucount
        return ulist.new_empty(ulist.shape)

    @register_fake("b70_qsa::union_attend")
    def _union_attend_fake(
        q, pages, block_table, seq_slot, ulist, ucount, attn, step, block_size, idx_block, idx_dim, has_pooled
    ):
        del q, pages, block_table, seq_slot, ulist, ucount, step, block_size, idx_block, idx_dim, has_pooled
        return attn.new_empty(attn.shape)

    @register_fake("b70_qsa::union_attend_xmx")
    def _union_attend_xmx_fake(
        q, pages, block_table, seq_slot, ulist, ucount, attn, step, block_size, idx_block, idx_dim, has_pooled
    ):
        del pages, block_table, seq_slot, ulist, ucount, attn, step, block_size, idx_block, idx_dim, has_pooled
        return torch.empty(1, dtype=torch.int32, device=q.device)

    _FAKES = True


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    path = os.environ.get("B70_QSA_SO", DEFAULT_SO)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"b70 QSA SYCL library not found at {path}. "
            "Build it with plugins/b70_qsa_xpu/build.sh."
        )
    torch.ops.load_library(path)
    _register_fakes()
    _LOADED = True


def _ready(t: torch.Tensor) -> torch.Tensor:
    if t.is_contiguous():
        return t
    return t.contiguous()


def write_step(pos: torch.Tensor, step: torch.Tensor, top_k: int, block: int) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.write_step(_ready(pos), _ready(step), int(top_k), int(block))


def kv_append(k, v, k_codes, v_codes, k_scale, v_scale, page_table, step) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.kv_append(
        _ready(k),
        _ready(v),
        _ready(k_codes),
        _ready(v_codes),
        _ready(k_scale),
        _ready(v_scale),
        _ready(page_table),
        _ready(step),
    )


def indexer_append(
    raw, k_norm, tail, dead, pooled, block_pos, cos, sin, step, mrope, pos_base: int, eps: float
) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.indexer_append(
        _ready(raw),
        _ready(k_norm),
        _ready(tail),
        _ready(dead),
        _ready(pooled),
        _ready(block_pos),
        _ready(cos),
        _ready(sin),
        _ready(step),
        _ready(mrope),
        int(pos_base),
        float(eps),
    )


def index_scores(q_idx, pooled, dead, scores, step, bias, block: int) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.index_scores(
        _ready(q_idx),
        _ready(pooled),
        _ready(dead),
        _ready(scores),
        _ready(step),
        _ready(bias),
        int(block),
    )


def topk(scores, ids, step) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.topk(_ready(scores), _ready(ids), _ready(step))


def kv_gather(ids, k_codes, v_codes, k_scale, v_scale, page_table, k_out, v_out, step) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.kv_gather(
        _ready(ids),
        _ready(k_codes),
        _ready(v_codes),
        _ready(k_scale),
        _ready(v_scale),
        _ready(page_table),
        _ready(k_out),
        _ready(v_out),
        _ready(step),
    )


def attend(q, k_f16, v_f16, attn, step) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.attend(
        _ready(q), _ready(k_f16), _ready(v_f16), _ready(attn), _ready(step)
    )


def gate(attn, q_full, out) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.gate(_ready(attn), _ready(q_full), _ready(out))


def write_step_batch(pos, step, top_k: int, block: int) -> torch.Tensor:
    _load()
    return torch.ops.b70_qsa.write_step_batch(_ready(pos), _ready(step), int(top_k), int(block))


def kv_append_batch(k, v, pages, block_table, step, block_size: int, idx_block: int, idx_dim: int, has_pooled: int):
    _load()
    return torch.ops.b70_qsa.kv_append_batch(
        _ready(k),
        _ready(v),
        _ready(pages),
        _ready(block_table),
        _ready(step),
        int(block_size),
        int(idx_block),
        int(idx_dim),
        int(has_pooled),
    )


def indexer_append_batch(
    raw, k_norm, tail, dead, block_pos, pages, block_table, cos, sin, step, mrope, pos_base, eps: float, block_size: int
):
    _load()
    return torch.ops.b70_qsa.indexer_append_batch(
        _ready(raw),
        _ready(k_norm),
        _ready(tail),
        _ready(dead),
        _ready(block_pos),
        _ready(pages),
        _ready(block_table),
        _ready(cos),
        _ready(sin),
        _ready(step),
        _ready(mrope),
        _ready(pos_base),
        float(eps),
        int(block_size),
    )


def index_scores_batch(q_idx, pages, dead, block_table, seq_slot, scores, step, bias, block_size: int, idx_block: int):
    _load()
    return torch.ops.b70_qsa.index_scores_batch(
        _ready(q_idx),
        _ready(pages),
        _ready(dead),
        _ready(block_table),
        _ready(seq_slot),
        _ready(scores),
        _ready(step),
        _ready(bias),
        int(block_size),
        int(idx_block),
    )


def topk_batch(scores, ids, step):
    _load()
    return torch.ops.b70_qsa.topk_batch(_ready(scores), _ready(ids), _ready(step))


def kv_gather_batch(
    ids, pages, block_table, seq_slot, k_out, v_out, step, block_size: int, idx_block: int, idx_dim: int, has_pooled: int
):
    _load()
    return torch.ops.b70_qsa.kv_gather_batch(
        _ready(ids),
        _ready(pages),
        _ready(block_table),
        _ready(seq_slot),
        _ready(k_out),
        _ready(v_out),
        _ready(step),
        int(block_size),
        int(idx_block),
        int(idx_dim),
        int(has_pooled),
    )


def attend_batch(q, k_f16, v_f16, attn, step):
    _load()
    return torch.ops.b70_qsa.attend_batch(
        _ready(q), _ready(k_f16), _ready(v_f16), _ready(attn), _ready(step)
    )


def gate_batch(attn, q_full, out):
    _load()
    return torch.ops.b70_qsa.gate_batch(_ready(attn), _ready(q_full), _ready(out))


def union_build(ids, step, ulist, ucount):
    _load()
    return torch.ops.b70_qsa.union_build(_ready(ids), _ready(step), _ready(ulist), _ready(ucount))


def union_attend_xmx(
    q, pages, block_table, seq_slot, ulist, ucount, attn, step, block_size: int, idx_block: int, idx_dim: int, has_pooled: int
):
    """Mode-1 XMX union. Returns int32[1]: 1 if the 24/2/256 kernel launched."""
    _load()
    return torch.ops.b70_qsa.union_attend_xmx(
        _ready(q),
        _ready(pages),
        _ready(block_table),
        _ready(seq_slot),
        _ready(ulist),
        _ready(ucount),
        _ready(attn),
        _ready(step),
        int(block_size),
        int(idx_block),
        int(idx_dim),
        int(has_pooled),
    )


def union_attend(
    q, pages, block_table, seq_slot, ulist, ucount, attn, step, block_size: int, idx_block: int, idx_dim: int, has_pooled: int
):
    _load()
    return torch.ops.b70_qsa.union_attend(
        _ready(q),
        _ready(pages),
        _ready(block_table),
        _ready(seq_slot),
        _ready(ulist),
        _ready(ucount),
        _ready(attn),
        _ready(step),
        int(block_size),
        int(idx_block),
        int(idx_dim),
        int(has_pooled),
    )
