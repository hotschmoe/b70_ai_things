"""O4 decode kernels (torch.ops.b70_o4). B70_O4_SO overrides the library."""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load() -> None:
    global _LOADED
    if _LOADED:
        return
    import torch

    so = os.environ.get("B70_O4_SO") or str(Path(__file__).resolve().parent / "_C.so")
    torch.ops.load_library(so)
    _LOADED = True


def quantize_w8(w):
    """bf16/fp32 [N, K] -> (int8 [N, K], fp16 [N, K/32]); Q8_0 rounding."""
    import torch

    n, k = w.shape
    blocks = w.detach().float().reshape(n, k // 32, 32)
    amax = blocks.abs().amax(-1)
    d = (amax / 127.0).to(torch.float16)
    inv = torch.where(d.float() > 0, 1.0 / d.float(), torch.zeros_like(amax))
    q = torch.round(blocks * inv[..., None]).clamp_(-127, 127).to(torch.int8)
    return q.reshape(n, k).contiguous(), d.contiguous()


def w8_gemv(x, q, s, ksplit: int):
    import torch

    return torch.ops.b70_o4.w8_gemv(x, q, s, int(ksplit))
