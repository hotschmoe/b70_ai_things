#!/usr/bin/env python3
# B1 FUSEDQ correctness + perf test (RUN ON GPU by the orchestrator).
#
# Compares the fused op int8_gemm_w8a8_fusedq (quant-inline + s8s8 GEMM in ONE
# op) against the baseline two-step path (standalone dynamic_per_token_int8_quant
# + int8_gemm_w8a8) on random tensors at the 27B W8A8 shapes. Reports max abs
# diff and per-op timings (torch.xpu.Event). Also times the standalone quant to
# confirm the widened parallel launch fixed the ~101us M=1 latency stall.
#
# HOW TO RUN (mount the NEW fused .so over the baked kernel; NO serve needed):
#   ROOT=/mnt/vm_8tb/b70
#   PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
#   ./bin/gpu-run --card 0 docker run --rm --device /dev/dri \
#     -v $ROOT/w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so:$PKGD/_xpu_C.abi3.so:ro \
#     -v $ROOT/w8a8_kernel_v0240_fusedq/libgdn_attn_kernels_xe_2.so:$PKGD/libgdn_attn_kernels_xe_2.so:ro \
#     -v /mnt/vm_8tb/github/b70_ai_things/vllm/test_fusedq.py:/opt/test_fusedq.py:ro \
#     -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
#     --entrypoint bash vllm-xpu-env:int8g-v0240 -lc \
#     'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; python /opt/test_fusedq.py'
#
# PASS = max abs diff 0 (or a couple of ULPs) on every shape, and fused time <=
# quant+gemm sum, and standalone-quant M=1 K=17408 well under the old ~101us.

import sys
import torch

# Trigger the _xpu_C library load (defines torch.ops._xpu_C.*).
try:
    import vllm._xpu_ops  # noqa: F401
except Exception as e:
    print(f"[warn] import vllm._xpu_ops failed ({e}); trying vllm_xpu_kernels", flush=True)
    try:
        import vllm_xpu_kernels  # noqa: F401
    except Exception as e2:
        print(f"[fatal] could not load _xpu_C: {e2}", flush=True)
        sys.exit(2)

assert torch.xpu.is_available(), "XPU not available"
dev = "xpu"
ops = torch.ops._xpu_C

for name in ("dynamic_per_token_int8_quant", "int8_gemm_w8a8", "int8_gemm_w8a8_fusedq"):
    if not hasattr(ops, name):
        print(f"[fatal] op {name} missing from _xpu_C (wrong .so mounted?)", flush=True)
        sys.exit(3)
print("[ok] ops present: dynamic_per_token_int8_quant, int8_gemm_w8a8, int8_gemm_w8a8_fusedq", flush=True)


def _time_us(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    times = []
    for _ in range(iters):
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.xpu.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)  # ms -> us
    times.sort()
    return times[len(times) // 2]  # median us


def make_weight(K, N, dtype):
    # int8 weight [K, N] + per-channel scale [1, N] (same for both paths).
    w_q = torch.randint(-127, 128, (K, N), dtype=torch.int8, device=dev)
    w_s = (torch.rand(1, N, device=dev, dtype=torch.float32) * 0.02 + 0.005).to(dtype)
    return w_q, w_s


# 27B W8A8 shapes: (label, K, N). down_proj is the K=17408 hotspot.
SHAPES = [
    ("gate_up  ", 5120, 34816),
    ("down_proj", 17408, 5120),
    ("qkv      ", 5120, 7168),
    ("o_proj   ", 5120, 5120),
]
MS = [1, 2, 4, 8, 16]
DTYPE = torch.float16

print("\n=== CORRECTNESS: fused vs (ref-quant + int8_gemm_w8a8) ===", flush=True)
max_diff_overall = 0.0
for label, K, N in SHAPES:
    w_q, w_s = make_weight(K, N, DTYPE)
    for M in MS:
        x = torch.randn(M, K, device=dev, dtype=DTYPE) * 2.0
        # baseline two-step
        x_q, x_s, _ = ops.dynamic_per_token_int8_quant(x, True, 8)
        out_base = ops.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, None, DTYPE)
        # fused
        out_fused = ops.int8_gemm_w8a8_fusedq(x, w_q, w_s, None, DTYPE)
        torch.xpu.synchronize()
        d = (out_base.float() - out_fused.float()).abs().max().item()
        max_diff_overall = max(max_diff_overall, d)
        print(f"  {label} K={K:5d} N={N:5d} M={M:2d}  max|base-fused|={d:.4g}", flush=True)
print(f"  --> MAX ABS DIFF over all shapes = {max_diff_overall:.4g} "
      f"({'PASS' if max_diff_overall < 1e-2 else 'CHECK'})", flush=True)

print("\n=== PERF: standalone quant (widened parallel launch) ===", flush=True)
for label, K, N in SHAPES:
    for M in (1, 8):
        x = torch.randn(M, K, device=dev, dtype=DTYPE) * 2.0
        q_us = _time_us(lambda: ops.dynamic_per_token_int8_quant(x, True, 8))
        print(f"  quant {label} K={K:5d} M={M:2d}  {q_us:7.1f} us "
              f"(old serial ~101us on K=17408 M=1)", flush=True)

print("\n=== PERF: fused vs quant+gemm (per-op median us) ===", flush=True)
for label, K, N in SHAPES:
    w_q, w_s = make_weight(K, N, DTYPE)
    for M in (1, 8):
        x = torch.randn(M, K, device=dev, dtype=DTYPE) * 2.0
        x_q, x_s, _ = ops.dynamic_per_token_int8_quant(x, True, 8)
        q_us = _time_us(lambda: ops.dynamic_per_token_int8_quant(x, True, 8))
        g_us = _time_us(lambda: ops.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, None, DTYPE))
        f_us = _time_us(lambda: ops.int8_gemm_w8a8_fusedq(x, w_q, w_s, None, DTYPE))
        two_step = q_us + g_us
        verdict = "WIN" if f_us <= two_step else "check"
        print(f"  {label} K={K:5d} M={M:2d}  quant={q_us:7.1f}  gemm={g_us:7.1f}  "
              f"two-step={two_step:7.1f}  FUSED={f_us:7.1f} us  [{verdict}]", flush=True)

print("\n[done]", flush=True)
