"""Fuse cast, routing multiply and top-k sum without an fp32 slot cube."""
import triton
import triton.language as tl


@triton.jit
def _combine(S, W, O, H: tl.constexpr, TOPK: tl.constexpr,
             WEIGHTED: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(TOPK):
        x = tl.load(S + (token * TOPK + slot) * H + col, col < H, 0).to(tl.float32)
        if WEIGHTED:
            x = x * tl.load(W + token * TOPK + slot).to(tl.float32)
        value += x
    tl.store(O + token * H + col, value, col < H)


def reduce_slots(slots, router, t, topk, h):
    import torch

    if not slots.is_contiguous():
        raise ValueError("prefill slots must be contiguous")
    weighted = router is not None and router.numel() == t * topk
    if weighted and not router.is_contiguous():
        router = router.contiguous()
    out = torch.empty((t, h), device=slots.device, dtype=torch.float32)
    _combine[(t, triton.cdiv(h, 256))](
        slots, router if weighted else slots, out, h, topk, weighted, 256
    )
    return out
