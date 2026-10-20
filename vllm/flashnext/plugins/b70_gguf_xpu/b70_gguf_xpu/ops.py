"""torch.ops.b70_gguf wrappers. Importing this module does not touch the GPU.

Decode calls must be XPU-graph safe. The schema lives in the SYCL library.
Fake kernels cover Dynamo / FakeTensor. Wrappers do not sync, and they do
not allocate a fresh empty tensor on every call.
"""

import os
from pathlib import Path

_LOADED = False
_FAKES = False
# One empty uint8 per device for the skipped shared-expert arguments.
_EMPTY_U8: dict[str, object] = {}


def _register_fakes() -> None:
    global _FAKES
    if _FAKES:
        return
    import torch
    from torch.library import register_fake

    @register_fake("b70_gguf::dequant_bf16")
    def _dequant_fake(q, ggml_type, rows, cols):
        del ggml_type
        return torch.empty((rows, cols), dtype=torch.bfloat16, device=q.device)

    @register_fake("b70_gguf::gemv")
    def _gemv_fake(x, q, ggml_type, n, k):
        del q, ggml_type, k
        return torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)

    @register_fake("b70_gguf::moe_gemv")
    def _moe_gemv_fake(x, q, expert_ids, ggml_type, n, k):
        del q, ggml_type, k
        return torch.empty(
            (x.shape[0], expert_ids.shape[1], n),
            dtype=torch.float32,
            device=x.device,
        )

    @register_fake("b70_gguf::grouped_gemm")
    def _grouped_fake(x, q, entries, ggml_type, n, k):
        del q, entries, ggml_type, k
        return torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)

    @register_fake("b70_gguf::moe_layer")
    def _moe_layer_fake(
        x,
        gate_q,
        gate_type,
        up_q,
        up_type,
        down_q,
        down_type,
        expert_ids,
        router_w,
        shared_gate_q,
        shared_gate_type,
        shared_up_q,
        shared_up_type,
        shared_down_q,
        shared_down_type,
        n_inter,
        k_hidden,
    ):
        del gate_q, gate_type, up_q, up_type, down_q, down_type
        del expert_ids, router_w, shared_gate_q, shared_gate_type
        del shared_up_q, shared_up_type, shared_down_q, shared_down_type
        del n_inter
        return torch.empty(
            (x.shape[0], k_hidden), dtype=torch.float32, device=x.device
        )

    # reserve_decode exists only in _C.graph.so. The default _C.so must
    # still load for other agents.
    ns = getattr(torch.ops, "b70_gguf", None)
    if ns is not None and hasattr(ns, "reserve_decode"):

        @register_fake("b70_gguf::reserve_decode")
        def _reserve_fake():
            return torch.empty(0, dtype=torch.float32)

    _FAKES = True


def load() -> None:
    """Load the prebuilt SYCL extension. Call only inside a GPU lease."""
    global _LOADED
    if _LOADED:
        return
    import torch

    override = os.environ.get("B70_GGUF_SO", "")
    so = Path(override) if override else Path(__file__).resolve().parent / "_C.so"
    if not so.is_file():
        raise FileNotFoundError(f"missing {so}; run scripts/build.sh")
    torch.ops.load_library(str(so))
    _register_fakes()
    _LOADED = True


def _empty_u8(device):
    import torch

    key = str(device)
    cached = _EMPTY_U8.get(key)
    if cached is None or cached.device != device:
        cached = torch.empty(0, dtype=torch.uint8, device=device)
        _EMPTY_U8[key] = cached
    return cached


def dequant_bf16(q, ggml_type: int, rows: int, cols: int):
    """q uint8 -> bf16 [rows, cols]."""
    load()
    import torch

    return torch.ops.b70_gguf.dequant_bf16(q, int(ggml_type), int(rows), int(cols))


def gemv(x, q, ggml_type: int, n: int, k: int):
    """x bf16 [M, K], q uint8 packed rows -> fp32 [M, N]. M is chunked by 8."""
    load()
    import torch

    return torch.ops.b70_gguf.gemv(x, q, int(ggml_type), int(n), int(k))


def moe_gemv(x, q, expert_ids, ggml_type: int, n: int, k: int):
    """x bf16 [T, K], expert_ids int32 [T, topk] -> fp32 [T, topk, N]."""
    load()
    import torch

    return torch.ops.b70_gguf.moe_gemv(x, q, expert_ids, int(ggml_type), int(n), int(k))


def moe_layer(
    x,
    gate_q,
    gate_type: int,
    up_q,
    up_type: int,
    down_q,
    down_type: int,
    expert_ids,
    router_w,
    n_inter: int,
    k_hidden: int,
    shared_gate_q=None,
    shared_gate_type: int = 0,
    shared_up_q=None,
    shared_up_type: int = 0,
    shared_down_q=None,
    shared_down_type: int = 0,
):
    """One decode MoE layer. x bf16 [T, K] -> fp32 [T, K].

    expert_ids int32 [T, topk], router_w fp32 [T, topk]. Empty shared tensors
    skip the shared expert. Shared routing weight is 1.
    """
    load()
    import torch

    if shared_gate_q is None:
        shared_gate_q = _empty_u8(x.device)
    if shared_up_q is None:
        shared_up_q = _empty_u8(x.device)
    if shared_down_q is None:
        shared_down_q = _empty_u8(x.device)
    return torch.ops.b70_gguf.moe_layer(
        x,
        gate_q,
        int(gate_type),
        up_q,
        int(up_type),
        down_q,
        int(down_type),
        expert_ids,
        router_w,
        shared_gate_q,
        int(shared_gate_type),
        shared_up_q,
        int(shared_up_type),
        shared_down_q,
        int(shared_down_type),
        int(n_inter),
        int(k_hidden),
    )


def grouped_gemm(x, q, entries, ggml_type: int, n: int, k: int):
    """x fp16 or bf16 [M, K], entries int32 [n_ent, 3] = (expert, row0, nrows) -> fp32 [M, N].

    N must be a multiple of 64 and K a multiple of 32. Overlapping entries race.
    """
    load()
    import torch

    return torch.ops.b70_gguf.grouped_gemm(x, q, entries, int(ggml_type), int(n), int(k))
