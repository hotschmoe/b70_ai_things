"""Lazy loader for the b70 low-bit SYCL extension.

Decode calls must be XPU-graph safe: the schema lives in the SYCL library,
fake kernels cover Dynamo / FakeTensor, and the wrappers do not sync or
allocate a fresh empty plane on every call.
"""

from __future__ import annotations

import os

import torch

_LOADED = False
_FAKES = False
# One empty int2 hi-plane per device. Allocating it inside a capture is a
# new graph-pool block every replay setup; the profile run fills this cache.
_EMPTY_HI: dict[str, torch.Tensor] = {}

DEFAULT_SO = (
    "/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so"
)


def _register_fakes() -> None:
    global _FAKES
    if _FAKES:
        return
    from torch.library import register_fake

    @register_fake("b70_lowbit::gemv")
    def _gemv_fake(x, plane2, plane_hi, scale, bits):
        del plane_hi, scale, bits
        return x.new_empty((x.shape[0], plane2.shape[0]))

    @register_fake("b70_lowbit::moe_gemv")
    def _moe_fake(
        x,
        ids,
        router_w,
        w13_plane2,
        w13_plane_hi,
        w13_scale,
        w2_plane2,
        w2_plane_hi,
        w2_scale,
        bits,
    ):
        del ids, router_w, w13_plane2, w13_plane_hi, w13_scale
        del w2_plane2, w2_plane_hi, w2_scale, bits
        return x.new_empty((x.shape[0], x.shape[1]))

    @register_fake("b70_lowbit::prefill_wna16")
    def _wna_fake(x, plane2, plane_hi, scale, bits):
        del plane_hi, scale, bits
        return x.new_empty((x.shape[0], plane2.shape[0]))

    @register_fake("b70_lowbit::prefill_wxa8")
    def _wxa_fake(x, plane2, plane_hi, scale, bits):
        del plane_hi, scale, bits
        return x.new_empty((x.shape[0], plane2.shape[0]))

    # The grouped op is absent from an older .so. Skip the fake then so
    # decode still loads. B70_LOWBIT_PREFILL=grouped needs the new library.
    try:
        getattr(torch.ops.b70_lowbit, "moe_prefill")
    except AttributeError:
        _FAKES = True
        return

    @register_fake("b70_lowbit::moe_prefill")
    def _moe_prefill_fake(
        x,
        ids,
        router_w,
        w13_plane2,
        w13_plane_hi,
        w13_scale,
        w2_plane2,
        w2_plane_hi,
        w2_scale,
        bits,
    ):
        del ids, router_w, w13_plane2, w13_plane_hi, w13_scale
        del w2_plane2, w2_plane_hi, w2_scale, bits
        return x.new_empty((x.shape[0], x.shape[1]))

    _FAKES = True


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
    _register_fakes()
    _LOADED = True


def _hi(plane_hi: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if plane_hi is not None:
        return plane_hi
    key = str(like.device)
    cached = _EMPTY_HI.get(key)
    if cached is None or cached.device != like.device:
        cached = torch.empty(0, dtype=torch.uint8, device=like.device)
        _EMPTY_HI[key] = cached
    return cached


def _ready(t: torch.Tensor) -> torch.Tensor:
    # Already-contiguous parameters must not allocate during capture.
    if t.is_contiguous():
        return t
    return t.contiguous()


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
        _ready(x),
        _ready(plane2),
        _ready(_hi(plane_hi, x)),
        _ready(scale),
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
    # dtype casts are no-ops when the runner already produced int32 / fp32.
    # A real cast is a captured kernel, not a host sync.
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    if router_w.dtype != torch.float32:
        router_w = router_w.to(torch.float32)
    return torch.ops.b70_lowbit.moe_gemv(
        _ready(x),
        _ready(ids),
        _ready(router_w),
        _ready(w13_plane2),
        _ready(_hi(w13_plane_hi, x)),
        _ready(w13_scale),
        _ready(w2_plane2),
        _ready(_hi(w2_plane_hi, x)),
        _ready(w2_scale),
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
        _ready(x),
        _ready(plane2),
        _ready(_hi(plane_hi, x)),
        _ready(scale),
        int(bits),
    )


def moe_prefill(
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
    """Grouped expert prefill. One DPAS submit per projection, all experts.

    x is [T, H] bf16. ids is [T, topk] int32. router_w is [T, topk] float.
    Returns [T, H] bf16. Counts are not read back to the host.
    """
    _load()
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    if router_w.dtype != torch.float32:
        router_w = router_w.to(torch.float32)
    return torch.ops.b70_lowbit.moe_prefill(
        _ready(x),
        _ready(ids),
        _ready(router_w),
        _ready(w13_plane2),
        _ready(_hi(w13_plane_hi, x)),
        _ready(w13_scale),
        _ready(w2_plane2),
        _ready(_hi(w2_plane_hi, x)),
        _ready(w2_scale),
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
        _ready(x),
        _ready(plane2),
        _ready(_hi(plane_hi, x)),
        _ready(scale),
        int(bits),
    )
