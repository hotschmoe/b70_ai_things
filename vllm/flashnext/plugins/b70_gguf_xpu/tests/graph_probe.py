"""Capture b70_gguf gemv and moe_layer in an XPU graph and replay them.

No host sync and no data-dependent alloc on this path. Exits nonzero
on a numeric mismatch or a capture failure. Shapes are tiny on purpose.
Weights are random bytes: the check is replay versus a fresh eager call,
not versus a dequant reference.
"""

from __future__ import annotations

import sys

import torch

from b70_gguf_xpu import ops


def _check(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    torch.xpu.synchronize()
    if got.shape != ref.shape or got.dtype != ref.dtype:
        raise SystemExit(
            f"FAIL {name} shape {tuple(got.shape)} {got.dtype} "
            f"vs {tuple(ref.shape)} {ref.dtype}"
        )
    nan_got = torch.isnan(got)
    nan_ref = torch.isnan(ref)
    if not torch.equal(nan_got, nan_ref):
        raise SystemExit(f"FAIL {name} NaN mask differs")
    finite = ~nan_got
    nfin = int(finite.sum())
    if nfin == 0:
        raise SystemExit(f"FAIL {name} all NaN")
    diff = (got[finite].float() - ref[finite].float()).abs()
    amax = float(diff.max())
    # abs of the replay, so an all-zero kernel is visible in the log.
    peak = float(got[finite].float().abs().max())
    print(
        f"{name} max_abs={amax:.3e} peak={peak:.3e} "
        f"finite={nfin}/{got.numel()} shape={tuple(got.shape)}",
        flush=True,
    )
    if amax > 0.0:
        raise SystemExit(f"FAIL {name} max_abs={amax}")


def _fake_shapes() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    ops.load()
    with FakeTensorMode():
        x = torch.empty(2, 32, device="xpu", dtype=torch.bfloat16)
        q = torch.empty(4, 18, device="xpu", dtype=torch.uint8)
        y = torch.ops.b70_gguf.gemv(x, q, 20, 4, 32)
        if tuple(y.shape) != (2, 4) or y.dtype != torch.float32:
            raise SystemExit(f"FAIL fake gemv {tuple(y.shape)} {y.dtype}")
        ids = torch.empty(2, 2, device="xpu", dtype=torch.int32)
        rw = torch.empty(2, 2, device="xpu", dtype=torch.float32)
        gate = torch.empty(2 * 32 * 18, device="xpu", dtype=torch.uint8)
        empty = torch.empty(0, device="xpu", dtype=torch.uint8)
        ym = torch.ops.b70_gguf.moe_layer(
            x, gate, 20, gate, 20, gate, 20, ids, rw,
            empty, 0, empty, 0, empty, 0, 32, 32,
        )
        if tuple(ym.shape) != (2, 32) or ym.dtype != torch.float32:
            raise SystemExit(f"FAIL fake moe_layer {tuple(ym.shape)} {ym.dtype}")
    print("fake ok", flush=True)


def _iq4nl_rows(n: int, k: int, device: torch.device) -> torch.Tensor:
    """IQ4_NL rows with fp16 scale 1. Random bytes can be a NaN scale."""
    row = 18 * (k // 32)
    q = torch.full((n, row), 0x11, device=device, dtype=torch.uint8)
    q[:, 0::18] = 0x00
    q[:, 1::18] = 0x3C  # fp16 1.0, little endian
    return q


def _capture_gemv() -> None:
    device = torch.device("xpu")
    m, n, k = 2, 4, 32
    q = _iq4nl_rows(n, k, device)
    static_x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    ops.gemv(static_x, q, 20, n, k)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = ops.gemv(static_x, q, 20, n, k)
    static_x.copy_(torch.randn(m, k, device=device, dtype=torch.bfloat16))
    replay_ref = ops.gemv(static_x, q, 20, n, k)
    torch.xpu.synchronize()
    captured.fill_(torch.nan)
    graph.replay()
    torch.xpu.synchronize()
    _check("gemv replay", captured, replay_ref)


def _capture_moe() -> None:
    device = torch.device("xpu")
    t, topk, e, i_dim, k_dim = 2, 2, 4, 32, 32
    # gate/up: N=I, K=hidden. down: N=hidden, K=I. IQ4_NL, 18 bytes per 32.
    gate = _iq4nl_rows(e * i_dim, k_dim, device).reshape(-1)
    down = _iq4nl_rows(e * k_dim, i_dim, device).reshape(-1)
    static_x = torch.randn(t, k_dim, device=device, dtype=torch.bfloat16)
    static_ids = torch.randint(0, e, (t, topk), device=device, dtype=torch.int32)
    static_ids[0, 1] = -1
    static_w = torch.rand(t, topk, device=device, dtype=torch.float32)
    ops.moe_layer(
        static_x, gate, 20, gate, 20, down, 20, static_ids, static_w, i_dim, k_dim
    )
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = ops.moe_layer(
            static_x, gate, 20, gate, 20, down, 20, static_ids, static_w, i_dim, k_dim
        )
    static_x.copy_(torch.randn(t, k_dim, device=device, dtype=torch.bfloat16))
    static_ids.copy_(torch.randint(0, e, (t, topk), device=device, dtype=torch.int32))
    static_ids[1, 0] = -1
    static_w.copy_(torch.rand(t, topk, device=device, dtype=torch.float32))
    replay_ref = ops.moe_layer(
        static_x, gate, 20, gate, 20, down, 20, static_ids, static_w, i_dim, k_dim
    )
    torch.xpu.synchronize()
    captured.fill_(torch.nan)
    graph.replay()
    torch.xpu.synchronize()
    _check("moe_layer replay", captured, replay_ref)


def main() -> None:
    ops.load()
    torch.ops.b70_gguf.reserve_decode()
    torch.xpu.synchronize()
    print("reserve ok", flush=True)
    _fake_shapes()
    _capture_gemv()
    _capture_moe()
    print("GRAPH_PROBE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
