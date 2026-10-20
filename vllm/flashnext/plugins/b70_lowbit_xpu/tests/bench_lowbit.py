"""Correctness and latency for b70_lowbit XPU kernels.

Run under: bin/gpu-run --card 0, ZE_AFFINITY_MASK=0.
"""

from __future__ import annotations

import math
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import (
    dequant,
    pack_codes,
    quantize_and_pack,
    quantize_rtn,
    unpack_codes,
)

ROOFLINE = 608e9
PASS_REL = 0.02
PASS_COS = 0.999
FAILS: list[str] = []


def _hi(plane_hi):
    if plane_hi is None or plane_hi.numel() == 0:
        return None
    return plane_hi


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    g = got.float().reshape(-1)
    r = ref.float().reshape(-1)
    diff = g - r
    denom = torch.linalg.vector_norm(r).clamp_min(1e-8)
    rel = (torch.linalg.vector_norm(diff) / denom).item()
    cos = F.cosine_similarity(g.unsqueeze(0), r.unsqueeze(0)).item()
    amax = diff.abs().max().item()
    return rel, cos, amax


def check(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    rel, cos, amax = metrics(got, ref)
    ok = rel < PASS_REL and cos > PASS_COS
    flag = "PASS" if ok else "FAIL"
    print(
        f"  {flag} {name} rel_l2={rel:.4e} cos={cos:.6f} max_abs={amax:.4e}",
        flush=True,
    )
    if not ok:
        FAILS.append(name)


def ref_linear(x, plane2, plane_hi, scale, bits) -> torch.Tensor:
    k = x.shape[-1]
    w = dequant(plane2, _hi(plane_hi), scale, bits, k)
    return (x.float() @ w.float().T).to(torch.bfloat16)


def ref_wxa8(x, plane2, plane_hi, scale, bits) -> torch.Tensor:
    m, k = x.shape
    n = plane2.shape[0]
    g = k // 128
    codes = unpack_codes(plane2, _hi(plane_hi), bits, k).float()
    xf = x.float().reshape(m, g, 128)
    amax = xf.abs().amax(dim=-1)
    asc = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(xf / asc.unsqueeze(-1)).clamp(-127, 127)
    dots = torch.einsum("mgk,ngk->mng", q, codes.reshape(n, g, 128))
    y = torch.einsum("mng,ng,mg->mn", dots, scale.float(), asc)
    return y.to(torch.bfloat16)


def pack_on(weight: torch.Tensor, bits: int, device: torch.device):
    plane2, plane_hi, scale = quantize_and_pack(weight, bits)
    plane2 = plane2.to(device)
    scale = scale.to(device)
    if plane_hi is not None:
        plane_hi = plane_hi.to(device)
    return plane2, plane_hi, scale


def test_pack_roundtrip() -> None:
    print("== pack roundtrip", flush=True)
    for bits in (2, 3, 4):
        for k in (128, 512, 640, 2048, 2560):
            weight = torch.randn(17, k)
            codes, scale = quantize_rtn(weight, bits)
            plane2, plane_hi = pack_codes(codes, bits)
            back = unpack_codes(plane2, plane_hi, bits, k)
            if not torch.equal(back, codes):
                FAILS.append(f"pack b{bits} k{k}")
                print(f"  FAIL pack bits={bits} K={k}", flush=True)
            else:
                print(f"  PASS pack bits={bits} K={k}", flush=True)
            w = dequant(plane2, plane_hi, scale, bits, k)
            # bf16 scale round-trip is the only intended error.
            rel = (w - codes.float() * scale.float().repeat_interleave(128, dim=1)).abs().max().item()
            if rel > 0:
                # repeat_interleave of the stored bf16 scale, converted back,
                # must match dequant exactly.
                FAILS.append(f"dequant b{bits} k{k}")
                print(f"  FAIL dequant bits={bits} K={k} max={rel}", flush=True)


def test_gemv(device: torch.device) -> None:
    print("== gemv", flush=True)
    shapes = [
        (1, 32, 128),
        (2, 32, 128),
        (4, 32, 128),
        (8, 32, 128),
        (1, 64, 640),
        (8, 64, 640),
        (1, 128, 512),
        (1, 256, 2048),
    ]
    for bits in (2, 3, 4):
        for m, n, k in shapes:
            weight = torch.randn(n, k)
            x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            p2, hi, sc = pack_on(weight, bits, device)
            got = ops.gemv(x, p2, hi, sc, bits)
            torch.xpu.synchronize()
            ref = ref_linear(x, p2, hi, sc, bits)
            check(f"gemv b{bits} M={m} N={n} K={k}", got, ref)


def test_moe(device: torch.device) -> None:
    print("== moe gemv", flush=True)
    for bits in (2, 3, 4):
        for t_tokens, topk, e, h, i in (
            (1, 2, 4, 256, 128),
            (2, 2, 4, 256, 128),
            (1, 4, 3, 512, 256),
        ):
            gen = torch.Generator().manual_seed(bits * 100 + t_tokens)
            w13 = torch.randn(e, 2 * i, h, generator=gen)
            w2 = torch.randn(e, h, i, generator=gen)
            p13, h13, s13 = [], [], []
            p2s, h2s, s2s = [], [], []
            for expert in range(e):
                a, b, c = quantize_and_pack(w13[expert], bits)
                p13.append(a)
                h13.append(torch.zeros(a.shape[0], 0, dtype=torch.uint8) if b is None else b)
                s13.append(c)
                a, b, c = quantize_and_pack(w2[expert], bits)
                p2s.append(a)
                h2s.append(torch.zeros(a.shape[0], 0, dtype=torch.uint8) if b is None else b)
                s2s.append(c)
            p13 = torch.stack(p13).to(device)
            s13 = torch.stack(s13).to(device)
            p2t = torch.stack(p2s).to(device)
            s2t = torch.stack(s2s).to(device)
            hi13 = None if bits == 2 else torch.stack(h13).to(device)
            hi2 = None if bits == 2 else torch.stack(h2s).to(device)
            x = torch.randn(t_tokens, h, dtype=torch.bfloat16, device=device)
            ids = torch.randint(0, e, (t_tokens, topk), device=device)
            # One negative id to confirm the kernel drops it.
            ids[0, 0] = -1
            router = torch.rand(t_tokens, topk, device=device)
            got = ops.moe_gemv(x, ids, router, p13, hi13, s13, p2t, hi2, s2t, bits)
            torch.xpu.synchronize()
            y = torch.zeros(t_tokens, h, device=device)
            for token in range(t_tokens):
                acc = torch.zeros(h, device=device)
                for slot in range(topk):
                    expert = int(ids[token, slot])
                    if expert < 0:
                        continue
                    gu = ref_linear(x[token : token + 1], p13[expert], None if hi13 is None else hi13[expert], s13[expert], bits)
                    gate, up = gu[0, :i].float(), gu[0, i:].float()
                    hid = F.silu(gate) * up
                    hid_b = hid.to(torch.bfloat16)
                    down = ref_linear(hid_b.unsqueeze(0), p2t[expert], None if hi2 is None else hi2[expert], s2t[expert], bits)
                    acc = acc + float(router[token, slot]) * down[0].float()
                y[token] = acc.to(torch.bfloat16)
            check(f"moe b{bits} T={t_tokens} topk={topk} H={h} I={i}", got, y)


def test_prefill(device: torch.device) -> None:
    print("== prefill wna16 / wxa8", flush=True)
    shapes = [
        (8, 16, 128),
        (7, 32, 128),
        (16, 64, 256),
        (32, 128, 512),
    ]
    for bits in (2, 3, 4):
        for m, n, k in shapes:
            if n % 16 != 0 or k % 128 != 0:
                continue
            weight = torch.randn(n, k)
            x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            p2, hi, sc = pack_on(weight, bits, device)
            got = ops.prefill_wna16(x, p2, hi, sc, bits)
            torch.xpu.synchronize()
            # DPAS multiplies bf16(code * scale) by bf16 activations.
            w = dequant(p2, hi, sc, bits, k).to(torch.bfloat16).float()
            ref = (x.float() @ w.T).to(torch.bfloat16)
            check(f"wna16 b{bits} M={m} N={n} K={k}", got, ref)
            got8 = ops.prefill_wxa8(x, p2, hi, sc, bits)
            torch.xpu.synchronize()
            ref8 = ref_wxa8(x, p2, hi, sc, bits)
            check(f"wxa8 b{bits} M={m} N={n} K={k}", got8, ref8)


def _elapsed_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    try:
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.xpu.synchronize()
        return start.elapsed_time(end) / iters
    except Exception as exc:
        print(f"  event timing unavailable ({type(exc).__name__}: {exc}); using host clock", flush=True)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters


def _report_bw(name: str, ms: float, w_bytes: float, sc_bytes: float, act: float, out: float) -> None:
    sec = ms * 1e-3
    total = w_bytes + sc_bytes + act + out
    gbps = total / sec / 1e9
    gbps_w = (w_bytes + sc_bytes) / sec / 1e9
    print(
        f"  PERF {name} {ms:.4f} ms  "
        f"all={gbps:.1f} GB/s ({100 * gbps * 1e9 / ROOFLINE:.1f}% of 608)  "
        f"weight+scale={gbps_w:.1f} GB/s ({100 * gbps_w * 1e9 / ROOFLINE:.1f}%)",
        flush=True,
    )


def weight_payload(n: int, k: int, bits: int) -> tuple[int, int]:
    p2 = n * (k // 4)
    if bits == 2:
        hi = 0
    elif bits == 3:
        hi = n * (k // 8)
    else:
        hi = n * (k // 4)
    return p2 + hi, n * (k // 128) * 2


def bench_gemv(device: torch.device) -> None:
    print("== perf gemv / prefill", flush=True)
    cases = [
        ("ornith-square", 1, 2048, 2048),
        ("ornith-gate-up", 1, 1024, 2048),
        ("ornith-down", 1, 2048, 512),
        ("fn-square", 1, 2560, 2560),
        ("fn-gate-up", 1, 1280, 2560),
        ("fn-down", 1, 2560, 640),
        ("ornith-gate-up-m8", 8, 1024, 2048),
    ]
    for bits in (3, 2, 4):
        for name, m, n, k in cases:
            weight = torch.randn(n, k)
            x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            p2, hi, sc = pack_on(weight, bits, device)
            print(f"  RUN gemv {name} b{bits}", flush=True)
            ms = _elapsed_ms(lambda: ops.gemv(x, p2, hi, sc, bits), 5, 20)
            wb, sb = weight_payload(n, k, bits)
            _report_bw(
                f"gemv {name} b{bits}",
                ms,
                wb,
                sb,
                m * k * 2,
                m * n * 2,
            )
    # Ornith / Qwen3.6-35B-A3B lm_head. Large N is the bandwidth probe.
    print("  RUN gemv ornith-lm-head b3", flush=True)
    n_head, k_head = 248320, 2048
    head = torch.randn(n_head, k_head)
    xh = torch.randn(1, k_head, dtype=torch.bfloat16, device=device)
    p2h, hih, sch = pack_on(head, 3, device)
    del head
    ms_h = _elapsed_ms(lambda: ops.gemv(xh, p2h, hih, sch, 3), 3, 8)
    wb_h, sb_h = weight_payload(n_head, k_head, 3)
    _report_bw(
        "gemv ornith-lm-head b3",
        ms_h,
        wb_h,
        sb_h,
        k_head * 2,
        n_head * 2,
    )
    del p2h, hih, sch, xh
    # Prefill at the two requested token counts, int3 and int2, Ornith and Flash-Next.
    if any(name.startswith("wna16") or name.startswith("wxa8") for name in FAILS):
        print("  SKIP prefill perf; correctness failed", flush=True)
        return
    for bits in (3, 2):
        for name, m, n, k in (
            ("ornith-gate-up", 512, 1024, 2048),
            ("ornith-gate-up", 2048, 1024, 2048),
            ("fn-gate-up", 512, 1280, 2560),
        ):
            weight = torch.randn(n, k)
            x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            p2, hi, sc = pack_on(weight, bits, device)
            print(f"  RUN wna16 {name} M={m} b{bits}", flush=True)
            ms = _elapsed_ms(lambda: ops.prefill_wna16(x, p2, hi, sc, bits), 2, 5)
            wb, sb = weight_payload(n, k, bits)
            tiles_m = math.ceil(m / 8)
            _report_bw(
                f"wna16 {name} M={m} b{bits} (weights read ~{tiles_m}x)",
                ms,
                wb * tiles_m,
                sb * tiles_m,
                m * k * 2,
                m * n * 2,
            )
            print(f"  RUN wxa8 {name} M={m} b{bits}", flush=True)
            ms8 = _elapsed_ms(lambda: ops.prefill_wxa8(x, p2, hi, sc, bits), 2, 5)
            _report_bw(
                f"wxa8 {name} M={m} b{bits}",
                ms8,
                wb * tiles_m,
                sb * tiles_m,
                m * k * 2,
                m * n * 2,
            )


def bench_moe(device: torch.device) -> None:
    print("== perf moe", flush=True)
    specs = [
        ("ornith", 256, 8, 2048, 512),
        ("flashnext", 512, 10, 2560, 640),
    ]
    for bits in (3, 2):
        for name, e, topk, h, i in specs:
            # One expert at a time. A full fp32 table is several GB and is not the op.
            print(f"  PACK moe {name} E={e} b{bits}", flush=True)
            p13, h13, s13, p2s, h2s, s2s = [], [], [], [], [], []
            for expert in range(e):
                a, b, c = quantize_and_pack(torch.randn(2 * i, h), bits)
                p13.append(a)
                h13.append(b if b is not None else torch.zeros(a.shape[0], 0, dtype=torch.uint8))
                s13.append(c)
                a, b, c = quantize_and_pack(torch.randn(h, i), bits)
                p2s.append(a)
                h2s.append(b if b is not None else torch.zeros(a.shape[0], 0, dtype=torch.uint8))
                s2s.append(c)
            p13 = torch.stack(p13).to(device)
            s13 = torch.stack(s13).to(device)
            p2t = torch.stack(p2s).to(device)
            s2t = torch.stack(s2s).to(device)
            hi13 = None if bits == 2 else torch.stack(h13).to(device)
            hi2 = None if bits == 2 else torch.stack(h2s).to(device)
            x = torch.randn(1, h, dtype=torch.bfloat16, device=device)
            ids = torch.arange(topk, device=device).view(1, topk)
            router = torch.full((1, topk), 1.0 / topk, device=device)
            print(f"  RUN moe {name} b{bits}", flush=True)
            ms = _elapsed_ms(
                lambda: ops.moe_gemv(x, ids, router, p13, hi13, s13, p2t, hi2, s2t, bits),
                5,
                20,
            )
            w13b, s13b = weight_payload(2 * i, h, bits)
            w2b, s2b = weight_payload(h, i, bits)
            _report_bw(
                f"moe {name} T=1 topk={topk} b{bits}",
                ms,
                topk * (w13b + w2b),
                topk * (s13b + s2b),
                h * 2 + topk * i * 2,
                h * 2,
            )


def bench_int4_baseline(device: torch.device) -> None:
    print("== vllm int4_gemm_w4a16 baseline (GPTQ layout, same logical shape)", flush=True)
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401  registers torch.ops._xpu_C

        op = torch.ops._xpu_C.int4_gemm_w4a16
    except Exception as exc:
        print(f"  SKIP no int4_gemm_w4a16 ({type(exc).__name__}: {exc})", flush=True)
        return
    for name, m, n, k in (
        ("ornith-gate-up", 1, 1024, 2048),
        ("ornith-down", 1, 2048, 512),
        ("fn-gate-up", 1, 1280, 2560),
        ("fn-down", 1, 2560, 640),
    ):
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        # 8 nibbles along K, low nibble first, then the INC transpose dance.
        nib = torch.randint(0, 16, (n, k // 8, 8), dtype=torch.int32)
        shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32)
        packed = (nib << shifts).sum(dim=-1).to(torch.int32)  # [N, K/8]
        q = packed.t().contiguous()  # [K/8, N]
        qweight = q.t().contiguous().t()
        qweight = qweight.to(device)
        scales = torch.rand(k // 128, n, dtype=torch.bfloat16, device=device)
        # Symmetric INC stand-in: one int8 eight, not a per-group zero table.
        qzeros = torch.tensor([8], dtype=torch.int8, device=device)
        try:
            def run():
                return op(x, qweight, None, scales, qzeros, 128, None)

            run()
            torch.xpu.synchronize()
            ms = _elapsed_ms(run, 5, 20)
            # GPTQ int4 payload is K/2 bytes per row (2 nibbles per byte) = n * k / 2.
            _report_bw(f"xpu_int4 {name}", ms, n * k // 2, n * (k // 128) * 2, m * k * 2, m * n * 2)
        except Exception as exc:
            print(f"  ERROR xpu_int4 {name}: {type(exc).__name__}: {exc}", flush=True)


def test_plugin() -> None:
    print("== vLLM quant registration", flush=True)
    try:
        from b70_lowbit_xpu import register

        register()
        from vllm.model_executor.layers.quantization import get_quantization_config

        cls = get_quantization_config("b70_lowbit")
        cfg = cls.from_config(
            {
                "quant_method": "b70_lowbit",
                "bits": 3,
                "group_size": 128,
                "sym": True,
                "pack": "b70lb1",
                "modules_to_not_convert": ["lm_head"],
                "extra_bits": {"experts": 2},
            }
        )
        print(f"  PASS get_quantization_config -> {cfg.get_name()} bits={cfg.bits}", flush=True)
        assert cfg._bits_for("model.layers.0.mlp.experts") == 2
        assert cfg._bits_for("model.layers.0.self_attn.q_proj") == 3
        assert cfg._skipped("lm_head")
        assert not cfg._skipped("model.layers.0.mlp.experts.gate_proj")
        cfg_gate = cls.from_config(
            {
                "bits": 3,
                "group_size": 128,
                "sym": True,
                "pack": "b70lb1",
                "modules_to_not_convert": ["gate", "lm_head"],
            }
        )
        assert cfg_gate._skipped("model.layers.0.mlp.gate")
        assert not cfg_gate._skipped("model.layers.0.mlp.experts.0.gate_proj")
        print("  PASS bits map and skip list", flush=True)
        from importlib.metadata import entry_points

        names = [ep.name for ep in entry_points(group="vllm.general_plugins")]
        if "b70_lowbit_xpu" in names:
            print("  PASS entry point vllm.general_plugins b70_lowbit_xpu", flush=True)
        else:
            FAILS.append("entry-point")
            print(f"  FAIL entry points {names}", flush=True)
    except Exception:
        FAILS.append("quant-config")
        traceback.print_exc()
        return

    try:
        import torch.distributed as dist
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.layers.linear import ReplicatedLinear

        with set_current_vllm_config(VllmConfig()):
            if not dist.is_initialized():
                port = os.environ.get("B70_LOWBIT_PORT", "29671")
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                    backend="gloo",
                )
                initialize_model_parallel(1, 1)

            layer = ReplicatedLinear(
                128,
                64,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=cfg,
                prefix="model.layers.0.self_attn.o_proj",
                return_bias=False,
                disable_tp=True,
            )
            print(
                f"  PASS ReplicatedLinear quant={type(layer.quant_method).__name__} "
                f"weight={tuple(layer.weight.shape)} {layer.weight.dtype}",
                flush=True,
            )
            # The stock loader is not invoked here. Fill packed weights and run apply.
            weight = torch.randn(64, 128)
            p2, hi, sc = quantize_and_pack(weight, 3)
            with torch.no_grad():
                layer.weight.copy_(p2)
                layer.weight_plane_hi.copy_(hi)
                layer.weight_scale.copy_(sc)
            layer.to("xpu")
            x = torch.randn(1, 128, dtype=torch.bfloat16, device="xpu")
            y = layer.quant_method.apply(layer, x, None)
            torch.xpu.synchronize()
            ref = ref_linear(
                x, layer.weight, layer.weight_plane_hi, layer.weight_scale, 3
            )
            check("linear.apply M=1", y, ref)
    except Exception:
        FAILS.append("linear-construct")
        traceback.print_exc()

    try:
        from b70_lowbit_xpu.quant_config import _B70MoE
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )

        class _Moe:
            is_act_and_mul = True

        method = _B70MoE(_Moe(), 3, 128)
        assert isinstance(method, FusedMoEMethodBase)
        assert method.is_monolithic is False
        assert method.get_fused_moe_quant_config(None) is None
        layer = torch.nn.Module()
        method.create_weights(layer, 4, 256, 128, torch.bfloat16)
        needed = (
            "w13_weight",
            "w13_weight_plane_hi",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_plane_hi",
            "w2_weight_scale",
        )
        for name in needed:
            if not hasattr(layer, name):
                raise RuntimeError(f"missing {name}")
        for expert in range(4):
            w13 = torch.randn(256, 256)
            w2 = torch.randn(256, 128)
            a, b, c = quantize_and_pack(w13, 3)
            layer.w13_weight.data[expert].copy_(a)
            layer.w13_weight_plane_hi.data[expert].copy_(b)
            layer.w13_weight_scale.data[expert].copy_(c)
            a, b, c = quantize_and_pack(w2, 3)
            layer.w2_weight.data[expert].copy_(a)
            layer.w2_weight_plane_hi.data[expert].copy_(b)
            layer.w2_weight_scale.data[expert].copy_(c)
        layer.to("xpu")
        ids = torch.tensor([[0, 1], [2, -1]], device="xpu")
        router = torch.tensor([[0.25, 0.75], [1.0, 0.5]], device="xpu")
        x = torch.randn(2, 256, dtype=torch.bfloat16, device="xpu")
        y = method.apply(layer, x, router, ids)
        torch.xpu.synchronize()
        if tuple(y.shape) != (2, 256):
            raise RuntimeError(f"moe apply shape {tuple(y.shape)}")
        # T=9 takes the Python grouped prefill path.
        x9 = torch.randn(9, 256, dtype=torch.bfloat16, device="xpu")
        ids9 = torch.randint(0, 4, (9, 2), device="xpu")
        router9 = torch.rand(9, 2, device="xpu")
        y9 = method.apply(layer, x9, router9, ids9)
        torch.xpu.synchronize()
        if tuple(y9.shape) != (9, 256):
            raise RuntimeError(f"moe prefill shape {tuple(y9.shape)}")
        print(
            f"  PASS MoE method decode {tuple(y.shape)} prefill {tuple(y9.shape)} "
            f"monolithic={method.is_monolithic}",
            flush=True,
        )
    except Exception:
        FAILS.append("moe-method")
        traceback.print_exc()


def main() -> int:
    print("CONFIG torch", torch.__version__, "xpu", torch.xpu.device_count(), flush=True)
    device = torch.device("xpu")
    test_pack_roundtrip()
    if FAILS:
        print("VERDICT pack failed, skipping GPU", FAILS, flush=True)
        return 1
    test_gemv(device)
    gemv_ok = not any(name.startswith("gemv") for name in FAILS)
    if gemv_ok:
        test_moe(device)
        test_prefill(device)
    else:
        print("VERDICT gemv failed; skipping moe/prefill correctness", FAILS, flush=True)
    if gemv_ok:
        bench_gemv(device)
        if not any(name.startswith("moe") for name in FAILS):
            bench_moe(device)
        bench_int4_baseline(device)
        test_plugin()
    if FAILS:
        print("VERDICT failed checks:", FAILS, flush=True)
        return 1
    print("VERDICT all correctness checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
