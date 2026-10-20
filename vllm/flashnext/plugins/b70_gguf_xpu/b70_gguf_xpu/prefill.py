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


def _dq_threshold(env: str, default: int):
    """None means the fused kernel. 0 means always dequant-then-GEMM."""
    raw = os.environ.get(env, "").strip().lower()
    if raw in ("off", "none", "-1"):
        return None
    if raw in ("on",):
        return 0
    if raw == "":
        return default
    return int(raw)


def _dq_ready() -> bool:
    load()
    import torch

    return hasattr(torch.ops.b70_gguf, "prefill_dequant")


def prefill_grouped(x, q, expert_ids, ggml_type: int, n: int, k: int, mode: int = 0):
    """x fp16/bf16 [M, K], expert_ids int32 [M] -> fp32 [M, N] in input order.

    mode 0 fp16 DPAS, 1 bf16 DPAS, 2 int8 DPAS. Invalid expert ids leave zeros.
    mode 0 uses dequant + oneDNN only when B70_PREFILL_DQ_GROUP_M says so.
    The measured expert win is the full layer (prefill_moe_layer), not this op.
    """
    load()
    import torch

    from b70_gguf_xpu import prefill_dq

    m = int(x.shape[0])
    thr = _dq_threshold("B70_PREFILL_DQ_GROUP_M", prefill_dq.DQ_GROUP_M_MIN)
    if mode == 0 and thr is not None and m >= thr and _dq_ready():
        return prefill_dq.grouped_dq(x, q, expert_ids, int(ggml_type), int(n), int(k))
    return torch.ops.b70_gguf.prefill_grouped(x, q, expert_ids, int(ggml_type), int(n), int(k), int(mode))


def prefill_dense(x, q, ggml_type: int, n: int, k: int, mode: int = 0):
    """x fp16/bf16 [M, K], q one matrix -> fp32 [M, N].

    mode 0 uses dequant + torch.mm under dense_wants_dq (measured M and K).
    Other modes stay on the fused DPAS kernel. B70_PREFILL_DQ_M overrides.
    """
    load()
    import torch

    from b70_gguf_xpu import prefill_dq

    m = int(x.shape[0])
    if mode == 0 and prefill_dq.dense_wants_dq(m, int(k)) and _dq_ready():
        return prefill_dq.dense_dq(x, q, int(ggml_type), int(n), int(k))
    return torch.ops.b70_gguf.prefill_dense(x, q, int(ggml_type), int(n), int(k), int(mode))


def prefill_moe_layer(x, ids, q_gate, q_up, q_down, router, type_gu: int, type_down: int, h: int, f: int, mode: int = 0):
    """One prefill MoE layer. x [T, H] -> fp32 [T, H].

    ids int32 [T, topk]. router fp32 [T, topk], or an empty fp32 tensor for ones.
    Gate and up are separate packs with the same type. One launch covers gate+up
    (interleaved), then swiglu, then one launch for down.
    mode 0 and T >= DQ_T_MIN uses the C13 one-layer path when this .so has
    prefill_dequant_wide (B70_PREFILL_DQ_LAYER=bmm keeps the C10 bmm split).
    B70_PREFILL_DQ_T overrides the token threshold.
    """
    load()
    import torch

    if router is None:
        router = torch.empty(0, dtype=torch.float32, device=x.device)
    from b70_gguf_xpu import prefill_dq

    t = int(x.shape[0])
    if mode == 0 and prefill_dq.moe_wants_dq(t) and _dq_ready():
        from b70_gguf_xpu import prefill_fast

        if prefill_fast.wants_fast():
            return prefill_fast.moe_fast(
                x, ids, q_gate, q_up, q_down, router, int(type_gu), int(type_down), int(h), int(f)
            )
        return prefill_dq.moe_dq(
            x, ids, q_gate, q_up, q_down, router, int(type_gu), int(type_down), int(h), int(f)
        )
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
