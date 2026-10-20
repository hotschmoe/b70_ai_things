"""torch.ops.b70_gguf wrappers. Importing this module does not touch the GPU."""

from pathlib import Path

_LOADED = False


def load() -> None:
    """Load the prebuilt SYCL extension. Call only inside bin/gpu-run --card 0."""
    global _LOADED
    if _LOADED:
        return
    import torch

    so = Path(__file__).resolve().parent / "_C.so"
    if not so.is_file():
        raise FileNotFoundError(f"missing {so}; run scripts/build.sh")
    torch.ops.load_library(str(so))
    _LOADED = True


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

    def empty_u8():
        return torch.empty(0, dtype=torch.uint8, device=x.device)

    if shared_gate_q is None:
        shared_gate_q = empty_u8()
    if shared_up_q is None:
        shared_up_q = empty_u8()
    if shared_down_q is None:
        shared_down_q = empty_u8()
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
