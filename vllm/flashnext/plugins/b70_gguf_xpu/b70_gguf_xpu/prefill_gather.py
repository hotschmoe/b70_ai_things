"""Gather strided packed expert rows without interpreting their byte layout."""
import os

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "RS", "CS"])
def _gather(Q, I, O, N, RS, CS, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(I + expert)
    value = tl.load(Q + source * RS + col * CS, col < N, 0)
    tl.store(O + expert * N + col, value, col < N)


def gather_packed(q, selected):
    import torch

    if os.environ.get("B70_PREFILL_PACK_GATHER", "torch") != "triton":
        return q.index_select(0, selected)
    if q.ndim != 2 or q.dtype != torch.uint8:
        raise ValueError("packed gather expects a two-dimensional byte view")
    out = torch.empty((selected.numel(), q.shape[1]), device=q.device, dtype=q.dtype)
    _gather[(selected.numel(), triton.cdiv(q.shape[1], 1024))](
        q, selected, out, q.shape[1], q.stride(0), q.stride(1), 1024
    )
    return out
