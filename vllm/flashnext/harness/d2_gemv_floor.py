#!/usr/bin/env python3
"""Device-time floor for Ornith decode projections. No model load.

Times fp16 aten.linear, a Triton fp16 GEMV, and b70_lowbit int4 GEMV
inside an XPU graph replay. Replaying one weight overstates L2 hits, so
the stacked graph uses distinct weights for every call in one token.

Shapes match Ornith-1.5-35B-A3B GPTQ (fp16 trunk, int4 routed experts):
  GDN in_proj_qkvz [12288, 2048] x30, in_proj_ba [64, 2048] x30,
  out_proj [2048, 4096] x30, full-attn qkv [9216, 2048] x10,
  o_proj [2048, 4096] x10, shared gate+up [1024, 2048] x40,
  shared down [2048, 512] x40, router [256, 2048] x40,
  lm_head [248320, 2048] x1.
Roofline reference is 608 GB/s. ASCII only.
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn.functional as F

ROOF = 608e9
LOG = []


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def weight_bytes(n: int, k: int, elem: int) -> int:
    return n * k * elem


def gbps(nbytes: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return (nbytes / (ms * 1e-3)) / 1e9


def replay_ms(fn, nreplay: int) -> tuple[float, str]:
    fn()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(graph):
            fn()
    except Exception as exc:
        return -1.0, f"capture_fail {exc!r}"
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(nreplay):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / nreplay, "graph"


def bench_linear(name: str, n: int, k: int, m: int, copies: int, nreplay: int) -> None:
    xs = [
        torch.randn(m, k, device="xpu", dtype=torch.float16) for _ in range(copies)
    ]
    ws = [
        torch.randn(n, k, device="xpu", dtype=torch.float16) for _ in range(copies)
    ]

    def run():
        for x, w in zip(xs, ws):
            F.linear(x, w)

    ms, how = replay_ms(run, nreplay)
    # Activation traffic is small next to the weight for these K, but count it.
    nbytes = copies * (
        weight_bytes(n, k, 2) + m * k * 2 + m * n * 2
    )
    wbytes = copies * weight_bytes(n, k, 2)
    if ms < 0:
        say(f"fp16 {name} M={m} copies={copies} {how}")
        return
    say(
        f"fp16 {name} M={m} copies={copies} ms={ms:.4f} "
        f"per={ms / copies:.4f} weight_GB/s={gbps(wbytes, ms):.1f} "
        f"moved_GB/s={gbps(nbytes, ms):.1f} "
        f"roof%={100.0 * gbps(wbytes, ms) / 608.0:.1f} via={how}"
    )
    del xs, ws
    torch.xpu.synchronize()


def bench_stack(m: int) -> None:
    """One decode token of fp16 projections. Distinct weights, one graph."""
    plan = (
        [("gdn_qkvz", 12288, 2048)] * 30
        + [("gdn_ba", 64, 2048)] * 30
        + [("gdn_out", 2048, 4096)] * 30
        + [("attn_qkv", 9216, 2048)] * 10
        + [("attn_o", 2048, 4096)] * 10
        + [("shared_gu", 1024, 2048)] * 40
        + [("shared_down", 2048, 512)] * 40
        + [("router", 256, 2048)] * 40
        + [("lm_head", 248320, 2048)]
    )
    say(f"stack alloc M={m} calls={len(plan)}")
    xs = []
    ws = []
    wbytes = 0
    moved = 0
    for _name, n, k in plan:
        xs.append(torch.randn(m, k, device="xpu", dtype=torch.float16))
        ws.append(torch.randn(n, k, device="xpu", dtype=torch.float16))
        wbytes += weight_bytes(n, k, 2)
        moved += weight_bytes(n, k, 2) + m * k * 2 + m * n * 2

    def run():
        for x, w in zip(xs, ws):
            F.linear(x, w)

    ms, how = replay_ms(run, 4)
    if ms < 0:
        say(f"fp16 stack M={m} {how}")
    else:
        say(
            f"fp16 stack M={m} calls={len(plan)} ms={ms:.3f} "
            f"weight_GiB={wbytes / 1024**3:.3f} weight_GB/s={gbps(wbytes, ms):.1f} "
            f"moved_GB/s={gbps(moved, ms):.1f} "
            f"roof%={100.0 * gbps(wbytes, ms) / 608.0:.1f} via={how}"
        )
    del xs, ws
    torch.xpu.empty_cache()


def triton_fp16(n: int, k: int, m: int) -> None:
    import triton
    import triton.language as tl

    @triton.jit
    def _gemv(
        x_ptr,
        w_ptr,
        y_ptr,
        N,
        K,
        stride_xk,
        stride_wn,
        stride_wk,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + offs_k * stride_xk, mask=offs_k < K, other=0.0)
            w = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0.0,
            )
            acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
        tl.store(y_ptr + offs_n, acc.to(tl.float16), mask=offs_n < N)

    x = torch.randn(m, k, device="xpu", dtype=torch.float16)
    w = torch.randn(n, k, device="xpu", dtype=torch.float16)
    y = torch.empty(m, n, device="xpu", dtype=torch.float16)
    configs = [(64, 128), (32, 256), (128, 128)]
    best = None
    for bn, bk in configs:
        grid = (triton.cdiv(n, bn),)

        def run(bn=bn, bk=bk):
            for row in range(m):
                _gemv[grid](
                    x[row],
                    w,
                    y[row],
                    n,
                    k,
                    x.stride(1),
                    w.stride(0),
                    w.stride(1),
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                )

        ms, how = replay_ms(run, 8)
        ref = F.linear(x, w)
        torch.xpu.synchronize()
        err = (y.float() - ref.float()).abs().max().item()
        wbytes = weight_bytes(n, k, 2)
        say(
            f"triton_fp16 N={n} K={k} M={m} BN={bn} BK={bk} "
            f"ms={ms:.4f} weight_GB/s={gbps(wbytes, ms):.1f} "
            f"max_abs={err:.3e} via={how}"
        )
        if ms > 0 and (best is None or ms < best):
            best = ms
    del x, w, y


def lowbit_int4(name: str, n: int, k: int, m: int) -> None:
    sys.path.insert(
        0,
        "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_lowbit_xpu",
    )
    from b70_lowbit_xpu import ops
    from b70_lowbit_xpu.format import dequant, quantize_and_pack

    torch.manual_seed(0)
    weight = torch.randn(n, k, device="xpu", dtype=torch.float16)
    say(f"int4 pack {name} N={n} K={k}")
    plane2, plane_hi, scale = quantize_and_pack(weight.float(), 4)
    # quantize_and_pack may have landed on CPU if the input was moved.
    plane2 = plane2.to("xpu")
    scale = scale.to("xpu")
    plane_hi = None if plane_hi is None else plane_hi.to("xpu")
    x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    # Eager reference against dequant.
    y = ops.gemv(x, plane2, plane_hi, scale, 4)
    torch.xpu.synchronize()
    # Full dequant of lm_head is a 2 GiB fp32 matrix. Check a row slice.
    rows = min(n, 128)
    w_hat = dequant(plane2[:rows], None if plane_hi is None else plane_hi[:rows], scale[:rows], 4, k)
    ref = (x.float() @ w_hat.float().T).to(torch.bfloat16)
    torch.xpu.synchronize()
    eager_err = (y[:, :rows].float() - ref.float()).abs().max().item()
    say(f"int4 eager {name} M={m} rows={rows} max_abs_vs_dequant={eager_err:.3e}")

    static_x = x.clone()
    ops.gemv(static_x, plane2, plane_hi, scale, 4)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(graph):
            captured = ops.gemv(static_x, plane2, plane_hi, scale, 4)
    except Exception as exc:
        say(f"int4 capture_fail {name} {exc!r}")
        del weight, plane2, scale, x, y
        return
    static_x.copy_(torch.randn(m, k, device="xpu", dtype=torch.bfloat16))
    replay_ref = ops.gemv(static_x, plane2, plane_hi, scale, 4)
    torch.xpu.synchronize()
    captured.fill_(float("nan"))
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    nreplay = 10
    start.record()
    for _ in range(nreplay):
        graph.replay()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / nreplay
    cap_err = (captured.float() - replay_ref.float()).abs().max().item()
    # int4 payload is N*K/2 plus fp16 scales N*(K/128)*2.
    wbytes = n * k // 2 + n * (k // 128) * 2
    say(
        f"int4 graph {name} M={m} ms={ms:.4f} weight_GB/s={gbps(wbytes, ms):.1f} "
        f"roof%={100.0 * gbps(wbytes, ms) / 608.0:.1f} "
        f"replay_max_abs={cap_err:.3e} bytes={wbytes}"
    )
    del weight, plane2, plane_hi, scale, x, y, static_x, captured, replay_ref
    torch.xpu.empty_cache()


def bench_norms() -> None:
    try:
        import vllm._xpu_ops  # noqa: F401
    except Exception as exc:
        say(f"norm skip import {exc!r}")
        return
    if not hasattr(torch.ops.vllm, "xpu_fused_add_gemma_rms_norm"):
        say("norm skip op missing")
        return
    x = torch.randn(1, 2048, device="xpu", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(2048, device="xpu", dtype=torch.float16)
    eps = 1e-6

    def one():
        torch.ops.vllm.xpu_fused_add_gemma_rms_norm(x, residual, weight, eps)

    def eighty():
        for _ in range(80):
            torch.ops.vllm.xpu_fused_add_gemma_rms_norm(x, residual, weight, eps)

    ms1, how1 = replay_ms(one, 20)
    ms80, how80 = replay_ms(eighty, 8)
    say(f"norm x1 ms={ms1:.4f} via={how1}")
    say(f"norm x80 ms={ms80:.4f} via={how80} (one token, 2 norms x 40 layers)")


def main() -> None:
    say(f"device_count={torch.xpu.device_count()} name={torch.xpu.get_device_name(0)}")
    # One copy first (L2-hot upper bound), then a 4-copy graph for the big ones.
    shapes = [
        ("gdn_qkvz", 12288, 2048, 30),
        ("gdn_ba", 64, 2048, 30),
        ("gdn_out", 2048, 4096, 30),
        ("attn_qkv", 9216, 2048, 10),
        ("attn_o", 2048, 4096, 10),
        ("shared_gu", 1024, 2048, 40),
        ("shared_down", 2048, 512, 40),
        ("router", 256, 2048, 40),
        ("lm_head", 248320, 2048, 1),
    ]
    for m in (1, 2, 4):
        say(f"== fp16 individuals M={m}")
        for name, n, k, _count in shapes:
            copies = 1 if n * k * 2 > 64 * 1024 * 1024 else 4
            nreplay = 6 if name == "lm_head" else 12
            try:
                bench_linear(name, n, k, m, copies, nreplay)
            except Exception:
                say(f"fp16 FAIL {name} M={m}")
                traceback.print_exc()
    say("== fp16 full-token stacks")
    for m in (1, 4):
        try:
            bench_stack(m)
        except Exception:
            say(f"stack FAIL M={m}")
            traceback.print_exc()
            torch.xpu.empty_cache()
    say("== triton fp16")
    try:
        triton_fp16(12288, 2048, 1)
        triton_fp16(248320, 2048, 1)
    except Exception:
        say("triton FAIL")
        traceback.print_exc()
    say("== b70_lowbit int4")
    for name, n, k in (
        ("gdn_qkvz", 12288, 2048),
        ("gdn_out", 2048, 4096),
        ("attn_qkv", 9216, 2048),
        ("lm_head", 248320, 2048),
    ):
        try:
            lowbit_int4(name, n, k, 1)
        except Exception:
            say(f"int4 FAIL {name}")
            traceback.print_exc()
            torch.xpu.empty_cache()
    say("== norms")
    try:
        bench_norms()
    except Exception:
        say("norm FAIL")
        traceback.print_exc()
    say("DONE")


if __name__ == "__main__":
    main()
