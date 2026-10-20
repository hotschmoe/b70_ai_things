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


@triton.jit(do_not_specialize=["CAP"])
def _pad_tokens(X, ORDER, STARTS, PAD, H: tl.constexpr, CAP,
                TOPK: tl.constexpr, SX: tl.constexpr, SX1: tl.constexpr, BLOCK: tl.constexpr,
                ROWS: tl.constexpr):
    expert = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(STARTS + expert)
    stop = tl.load(STARTS + expert + 1)
    slot = tl.load(ORDER + start + row, row < stop - start, 0)
    valid = (row[:, None] < stop - start) & (col[None, :] < H)
    value = tl.load(X + (slot[:, None] // TOPK) * SX + col[None, :] * SX1, valid, 0)
    tl.store(PAD + (expert * CAP + row[:, None]) * H + col[None, :], value, valid)


@triton.jit(do_not_specialize=["CAP"])
def _inverse(ORDER, STARTS, INV, CAP, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(STARTS + expert)
    stop = tl.load(STARTS + expert + 1)
    valid = row < stop - start
    slot = tl.load(ORDER + start + row, valid, 0)
    tl.store(INV + slot, expert * CAP + row, valid)


@triton.jit
def _combine_grouped(Y, INV, W, O, H: tl.constexpr, TOPK: tl.constexpr,
                     WEIGHTED: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(TOPK):
        row = tl.load(INV + token * TOPK + slot)
        x = tl.load(Y + tl.maximum(row, 0) * H + col,
                    (row >= 0) & (col < H), 0).to(tl.float32)
        if WEIGHTED:
            x = x * tl.load(W + token * TOPK + slot).to(tl.float32)
        value += x
    tl.store(O + token * H + col, value, col < H)


def pad_tokens(x, xpad, order, starts, topk):
    e, cap, h = xpad.shape
    _pad_tokens[(e, triton.cdiv(cap, 16), triton.cdiv(h, 256))](
        x, order, starts, xpad, h, cap, topk, x.stride(0), x.stride(1), 256, 16)


def reduce_grouped(ypad, order, starts, router, t, topk, h):
    import torch

    e, cap, _ = ypad.shape
    inverse = torch.full((t * topk,), -1, device=ypad.device, dtype=torch.int32)
    _inverse[(e, triton.cdiv(cap, 256))](order, starts, inverse, cap, 256)
    weighted = router is not None and router.numel() == t * topk
    if weighted and not router.is_contiguous():
        router = router.contiguous()
    out = torch.empty((t, h), device=ypad.device, dtype=torch.float32)
    _combine_grouped[(t, triton.cdiv(h, 256))](
        ypad, inverse, router if weighted else ypad, out, h, topk, weighted, 256)
    return out
