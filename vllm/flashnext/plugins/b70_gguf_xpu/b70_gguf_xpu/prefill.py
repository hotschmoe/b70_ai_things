"""Prefill GEMM wrappers. Importing this module does not touch the GPU.

C2 ops live next to the A2/C1 ops in _C.so. This module does not edit ops.py.
Set B70_GGUF_SO to load a side build (the smoke .so) instead of _C.so.
"""

import os
from pathlib import Path

_LOADED = False


def load() -> None:
    """Load the prebuilt SYCL extension. Call only inside bin/gpu-run --card 0."""
    global _LOADED
    if _LOADED:
        return
    import torch

    override = os.environ.get("B70_GGUF_SO")
    so = Path(override) if override else Path(__file__).resolve().parent / "_C.so"
    if not so.is_file():
        raise FileNotFoundError(f"missing {so}; run scripts/build.sh")
    torch.ops.load_library(str(so))
    _LOADED = True


def prefill_grouped(x, q, expert_ids, ggml_type: int, n: int, k: int, mode: int = 0):
    """x fp16/bf16 [M, K], expert_ids int32 [M] -> fp32 [M, N] in input order.

    mode 0 fp16 DPAS, 1 bf16 DPAS, 2 int8 DPAS. Invalid expert ids leave zeros.
    """
    load()
    import torch

    return torch.ops.b70_gguf.prefill_grouped(x, q, expert_ids, int(ggml_type), int(n), int(k), int(mode))


def prefill_dense(x, q, ggml_type: int, n: int, k: int, mode: int = 0):
    """x fp16/bf16 [M, K], q one matrix -> fp32 [M, N]."""
    load()
    import torch

    return torch.ops.b70_gguf.prefill_dense(x, q, int(ggml_type), int(n), int(k), int(mode))


def prefill_moe_layer(x, ids, q_gate, q_up, q_down, router, type_gu: int, type_down: int, h: int, f: int, mode: int = 0):
    """One prefill MoE layer. x [T, H] -> fp32 [T, H].

    ids int32 [T, topk]. router fp32 [T, topk], or an empty fp32 tensor for ones.
    Gate and up are separate packs with the same type. One launch covers gate+up
    (interleaved), then swiglu, then one launch for down.
    """
    load()
    import torch

    if router is None:
        router = torch.empty(0, dtype=torch.float32, device=x.device)
    return torch.ops.b70_gguf.prefill_moe_layer(
        x, ids, q_gate, q_up, q_down, router, int(type_gu), int(type_down), int(h), int(f), int(mode)
    )


def prefill_split_us():
    """Last op's stage times in microseconds: sort, gather, gemm, gemm2, swiglu, reduce, total, cols*1000+mt.

    Nonzero only when B70_PREFILL_SPLIT=1 was set for that call.
    """
    load()
    import torch

    return torch.ops.b70_gguf.prefill_split_us()
