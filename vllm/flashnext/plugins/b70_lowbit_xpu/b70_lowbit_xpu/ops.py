"""Lazy loader for the b70 low-bit SYCL extension."""

from __future__ import annotations

import os

import torch

_LOADED = False

DEFAULT_SO = (
    "/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so"
)


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    path = os.environ.get("B70_LOWBIT_SO", DEFAULT_SO)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"b70 low-bit SYCL library not found at {path}. "
            "Build it with plugins/b70_lowbit_xpu/build.sh."
        )
    torch.ops.load_library(path)
    _LOADED = True


def _hi(plane_hi: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if plane_hi is None:
        return torch.empty(0, dtype=torch.uint8, device=like.device)
    return plane_hi


def gemv(
    x: torch.Tensor,
    plane2: torch.Tensor,
    plane_hi: torch.Tensor | None,
    scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Dense GEMV. x is [M, K] bf16, M=1..8. Returns [M, N] bf16."""
    _load()
    return torch.ops.b70_lowbit.gemv(
        x.contiguous(),
        plane2.contiguous(),
        _hi(plane_hi, x).contiguous(),
        scale.contiguous(),
        int(bits),
    )


def moe_gemv(
    x: torch.Tensor,
    ids: torch.Tensor,
    router_w: torch.Tensor,
    w13_plane2: torch.Tensor,
    w13_plane_hi: torch.Tensor | None,
    w13_scale: torch.Tensor,
    w2_plane2: torch.Tensor,
    w2_plane_hi: torch.Tensor | None,
    w2_scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Fused routed-expert GEMV. x is [T, H], T=1..8. Returns [T, H] bf16.

    ids is [T, topk] int32. router_w is [T, topk] float. Gate is the first
    half of w13 rows, up is the second. Shared experts are not applied.
    """
    _load()
    ids_i = ids.to(torch.int32).contiguous()
    router = router_w.to(torch.float32).contiguous()
    return torch.ops.b70_lowbit.moe_gemv(
        x.contiguous(),
        ids_i,
        router,
        w13_plane2.contiguous(),
        _hi(w13_plane_hi, x).contiguous(),
        w13_scale.contiguous(),
        w2_plane2.contiguous(),
        _hi(w2_plane_hi, x).contiguous(),
        w2_scale.contiguous(),
        int(bits),
    )


def prefill_wna16(
    x: torch.Tensor,
    plane2: torch.Tensor,
    plane_hi: torch.Tensor | None,
    scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Prefill GEMM. Unpack to bf16 and ESIMD bf16 DPAS. x is [M, K] bf16."""
    _load()
    return torch.ops.b70_lowbit.prefill_wna16(
        x.contiguous(),
        plane2.contiguous(),
        _hi(plane_hi, x).contiguous(),
        scale.contiguous(),
        int(bits),
    )


def prefill_wxa8(
    x: torch.Tensor,
    plane2: torch.Tensor,
    plane_hi: torch.Tensor | None,
    scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Prefill GEMM with per-group int8 activations. Decode does not use this."""
    _load()
    return torch.ops.b70_lowbit.prefill_wxa8(
        x.contiguous(),
        plane2.contiguous(),
        _hi(plane_hi, x).contiguous(),
        scale.contiguous(),
        int(bits),
    )
