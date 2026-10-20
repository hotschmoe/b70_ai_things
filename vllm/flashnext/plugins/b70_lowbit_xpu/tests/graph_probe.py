"""Capture b70_lowbit decode ops in an XPU graph and replay them.

No host sync and no data-dependent alloc on this path. Exits 0 on a
numeric mismatch or a capture failure. Shapes are tiny on purpose.
"""

from __future__ import annotations

import sys

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import quantize_and_pack


def _check(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    torch.xpu.synchronize()
    diff = (got.float() - ref.float()).abs()
    amax = float(diff.max())
    print(f"{name} max_abs={amax:.3e} shape={tuple(got.shape)}", flush=True)
    if amax > 1e-3:
        raise SystemExit(f"FAIL {name} max_abs={amax}")


def _fake_shapes() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    ops._load()
    with FakeTensorMode():
        x = torch.empty(2, 128, device="xpu", dtype=torch.bfloat16)
        p2 = torch.empty(32, 64, device="xpu", dtype=torch.uint8)
        phi = torch.empty(32, 64, device="xpu", dtype=torch.uint8)
        sc = torch.empty(32, 1, device="xpu", dtype=torch.bfloat16)
        y = torch.ops.b70_lowbit.gemv(x, p2, phi, sc, 3)
        if tuple(y.shape) != (2, 32):
            raise SystemExit(f"FAIL fake gemv shape {tuple(y.shape)}")
        ids = torch.empty(2, 4, device="xpu", dtype=torch.int32)
        rw = torch.empty(2, 4, device="xpu", dtype=torch.float32)
        w13 = torch.empty(4, 16, 64, device="xpu", dtype=torch.uint8)
        w13s = torch.empty(4, 16, 1, device="xpu", dtype=torch.bfloat16)
        w2 = torch.empty(4, 8, 64, device="xpu", dtype=torch.uint8)
        w2s = torch.empty(4, 8, 1, device="xpu", dtype=torch.bfloat16)
        ym = torch.ops.b70_lowbit.moe_gemv(
            x, ids, rw, w13, phi, w13s, w2, phi, w2s, 3
        )
        if tuple(ym.shape) != (2, 128):
            raise SystemExit(f"FAIL fake moe shape {tuple(ym.shape)}")
    print("fake ok", flush=True)


def _pack_experts(weight: torch.Tensor, bits: int, device: torch.device):
    planes, his, scales = [], [], []
    for expert in range(weight.shape[0]):
        plane2, plane_hi, scale = quantize_and_pack(weight[expert], bits)
        planes.append(plane2)
        scales.append(scale)
        if plane_hi is None:
            his.append(torch.zeros(plane2.shape[0], 0, dtype=torch.uint8))
        else:
            his.append(plane_hi)
    plane2 = torch.stack(planes).to(device)
    scale = torch.stack(scales).to(device)
    plane_hi = None if bits == 2 else torch.stack(his).to(device)
    return plane2, plane_hi, scale


def _capture_gemv(bits: int) -> None:
    torch.manual_seed(0)
    m, n, k = 2, 64, 256
    weight = torch.randn(n, k, dtype=torch.float32)
    plane2, plane_hi, scale = quantize_and_pack(weight, bits)
    device = torch.device("xpu")
    plane2 = plane2.to(device)
    scale = scale.to(device)
    plane_hi = None if plane_hi is None else plane_hi.to(device)
    static_x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    # Warm the extension and the empty-hi cache outside the capture.
    ref = ops.gemv(static_x, plane2, plane_hi, scale, bits)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = ops.gemv(static_x, plane2, plane_hi, scale, bits)
    _check(f"gemv bits={bits} capture", captured, ref)
    static_x.copy_(torch.randn(m, k, device=device, dtype=torch.bfloat16))
    replay_ref = ops.gemv(static_x, plane2, plane_hi, scale, bits)
    torch.xpu.synchronize()
    graph.replay()
    torch.xpu.synchronize()
    _check(f"gemv bits={bits} replay", captured, replay_ref)


def _capture_moe(bits: int) -> None:
    torch.manual_seed(1)
    t, topk, e, h, i = 2, 4, 8, 256, 128
    device = torch.device("xpu")
    w13 = torch.randn(e, 2 * i, h)
    w2 = torch.randn(e, h, i)
    p13, h13, s13 = _pack_experts(w13, bits, device)
    p2, h2, s2 = _pack_experts(w2, bits, device)
    static_x = torch.randn(t, h, device=device, dtype=torch.bfloat16)
    static_ids = torch.randint(0, e, (t, topk), device=device, dtype=torch.int32)
    static_w = torch.rand(t, topk, device=device, dtype=torch.float32)
    ref = ops.moe_gemv(static_x, static_ids, static_w, p13, h13, s13, p2, h2, s2, bits)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = ops.moe_gemv(
            static_x, static_ids, static_w, p13, h13, s13, p2, h2, s2, bits
        )
    _check(f"moe bits={bits} capture", captured, ref)
    static_x.copy_(torch.randn(t, h, device=device, dtype=torch.bfloat16))
    static_ids.copy_(torch.randint(0, e, (t, topk), device=device, dtype=torch.int32))
    static_w.copy_(torch.rand(t, topk, device=device, dtype=torch.float32))
    replay_ref = ops.moe_gemv(
        static_x, static_ids, static_w, p13, h13, s13, p2, h2, s2, bits
    )
    torch.xpu.synchronize()
    graph.replay()
    torch.xpu.synchronize()
    _check(f"moe bits={bits} replay", captured, replay_ref)


def main() -> None:
    print("xpu", torch.xpu.device_count(), flush=True)
    _fake_shapes()
    for bits in (2, 3, 4):
        _capture_gemv(bits)
        _capture_moe(bits)
    print("GRAPH_PROBE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GRAPH_PROBE_FAIL {type(exc).__name__}: {exc}", flush=True)
        raise
