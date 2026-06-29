"""Route-A go/no-go probe -- DRAFT (2026-06-29). RUN BY THE GPU DRIVER, NOT the author.

Question this answers: does sglang's IN-TREE Triton fused-MoE `use_int8_w8a8` path actually
codegen + run + stay numerically correct on triton-xpu (B70)? If YES, the whole sglang W8A8 MoE
port is just the loader in sglang/patches/quark_moe_int8.py (no custom kernel). If the int8 Triton
kernel mis-codegens on XPU, this fails here and we fall back to Route C (fused SYCL grouped GEMM).

It is OFFLINE: it builds the REAL Qwen3.6-35B-A3B expert shapes (E=256, top-8,
moe_intermediate_size=512, hidden=2048) as int8 with per-channel scales and calls sglang's
`fused_experts(..., use_int8_w8a8=True, per_channel_quant=True)` directly. No server, no model load.

Correctness oracle: run the SAME `fused_experts` twice on the SAME random experts --
  (a) int8 weights + use_int8_w8a8=True  (the path under test; activations get per-token int8 quant
      INSIDE the kernel, kernels:778-785),
  (b) bf16-dequantized weights + use_int8_w8a8=False (stock unquant MoE).
A high cosine between (a) and (b) means the int8 kernel computes the same swiglu MoE as bf16 up to
quant error -- i.e. the int8 codegen is correct. (Comparing two fused_experts calls cancels the
gate/up interleaving + swiglu layout, so we don't re-derive it by hand.)

Modeled on scripts/int8_moe_grouped_test.py (same 35B shapes / regime split), but driving sglang's
Triton MoE instead of the per-expert oneDNN loop.

  HOW TO RUN (GPU driver, inside the sglang image, ATTENDED is not required -- single card, no serve):
    docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
      -v $REPO/research/w8a8:/probe sglang-xpu:mtp \
      bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
                export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; \
                python /probe/sglang_moe_int8_probe.py'
  (Pin one card with ZE_AFFINITY_MASK=0 if you want to stay off the other GPU.)
"""

import sys
import time

import torch

print("=== ENV ===", flush=True)
print("torch", torch.__version__, "| xpu avail", torch.xpu.is_available(),
      "| count", torch.xpu.device_count(), flush=True)
try:
    import triton  # noqa: F401
    print("triton", triton.__version__, flush=True)
except Exception as e:  # pragma: no cover
    print("triton import FAILED:", e, flush=True)

DEV = "xpu:0"
torch.manual_seed(0)

# sglang in-tree fused MoE (Triton). Both import sites are valid (fused_moe_triton/__init__ re-exports).
try:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.fused_moe_triton import fused_experts
    from sglang.srt.layers.moe.topk import StandardTopKOutput
except Exception as e:  # pragma: no cover
    print("FATAL: cannot import sglang fused_experts/MoeRunnerConfig/StandardTopKOutput:", e, flush=True)
    sys.exit(2)

# sglang's fused_experts reads get_global_server_args() (only .enable_deterministic_inference on the
# config-lookup path). A standalone offline probe never launched a server, and full ServerArgs
# construction trips a GDN/FLA assertion on XPU. So set the module global directly to a permissive stub:
# __getattr__ -> False covers the boolean server flags the MoE Triton path reads. The actual SERVE sets
# real server args, so this stub is probe-only.
try:
    import sglang.srt.server_args as _samod

    class _StubServerArgs:
        enable_deterministic_inference = False

        def __getattr__(self, _k):
            return False

    _samod._global_server_args = _StubServerArgs()
    print("set stub global server args (offline probe)", flush=True)
except Exception as e:  # pragma: no cover
    print("WARN: could not set stub global server args:", type(e).__name__, e, flush=True)

# XPU-safe per-token int8 activation quant (stock kernel uses cuda.libdevice.round -> unlinked on XPU).
# Mounted at /patches in the probe container.
sys.path.insert(0, "/patches")
try:
    import int8_actquant_xpu
    int8_actquant_xpu.install()
except Exception as e:  # pragma: no cover
    print("WARN: int8_actquant_xpu install failed:", type(e).__name__, e, flush=True)

# Real 35B-A3B MoE shapes (models/files/qwen3.6-35b-a3b/quark-w8a8-int8/config.json).
E, H, I, TOPK = 256, 2048, 512, 8
DTYPE = torch.bfloat16


def quant_per_out_channel(W: torch.Tensor):
    """W [..., rows, K] bf16 -> (int8 [..., rows, K], scale [..., rows, 1] f32). Symmetric per-row."""
    s = (W.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-8)
    Wq = (W / s).round().clamp(-127, 127).to(torch.int8)
    return Wq, s.to(torch.float32)


print(f"\n=== build {E} experts: w13 [E,2I={2*I},H={H}] int8 | w2 [E,H={H},I={I}] int8 ===", flush=True)
# Small magnitude so per-channel int8 quant error is low (clean correctness signal).
w13_bf16 = (torch.randn(E, 2 * I, H, device=DEV, dtype=DTYPE) * 0.02)
w2_bf16 = (torch.randn(E, H, I, device=DEV, dtype=DTYPE) * 0.02)
w13_q, w13_s = quant_per_out_channel(w13_bf16)   # w13_s [E, 2I, 1]
w2_q, w2_s = quant_per_out_channel(w2_bf16)       # w2_s  [E, H, 1]
# bf16-dequant reference weights (what the int8 kernel should reproduce up to quant error).
w13_deq = (w13_q.to(torch.float32) * w13_s).to(DTYPE).contiguous()
w2_deq = (w2_q.to(torch.float32) * w2_s).to(DTYPE).contiguous()
print("  int8 expert VRAM ~", round((w13_q.numel() + w2_q.numel()) / 1e6, 1), "MB", flush=True)


def make_topk(T: int) -> StandardTopKOutput:
    gate = torch.randn(T, E, device=DEV, dtype=torch.float32)
    topk_w, topk_id = gate.softmax(-1).topk(TOPK, dim=-1)
    return StandardTopKOutput(
        topk_w.to(torch.float32).contiguous(),
        topk_id.to(torch.int32).contiguous(),
        gate,
    )


def runner_cfg() -> MoeRunnerConfig:
    # inplace=False -> outplace path returns a fresh tensor (we reuse x across both calls).
    return MoeRunnerConfig(
        num_experts=E,
        num_local_experts=E,        # equal -> filter_expert=False (no expert_map)
        hidden_size=H,
        intermediate_size_per_partition=I,
        top_k=TOPK,
        params_dtype=DTYPE,
        activation="silu",
        is_gated=True,
        inplace=False,
    )


def run_regime(T: int, label: str, iters: int = 20):
    print(f"\n=== {label}: T={T} tokens, top_k={TOPK} ===", flush=True)
    x = (torch.randn(T, H, device=DEV, dtype=DTYPE) * 0.1).contiguous()
    topk = make_topk(T)
    cfg = runner_cfg()

    # (a) int8 path under test
    try:
        out_i8 = fused_experts(
            hidden_states=x, w1=w13_q, w2=w2_q,
            topk_output=topk, moe_runner_config=cfg,
            use_int8_w8a8=True, per_channel_quant=True,
            w1_scale=w13_s, w2_scale=w2_s,
            a1_scale=None, a2_scale=None,   # None -> dynamic per-token int8 quant inside kernel
        )
        torch.xpu.synchronize()
    except Exception as e:
        print(f"  [NO-GO] int8 fused_experts raised on XPU: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

    # (b) bf16 reference (same experts dequantized)
    out_bf16 = fused_experts(
        hidden_states=x, w1=w13_deq, w2=w2_deq,
        topk_output=topk, moe_runner_config=runner_cfg(),
        use_int8_w8a8=False,
    )
    torch.xpu.synchronize()

    a = out_i8.reshape(-1).to(torch.float32)
    b = out_bf16.reshape(-1).to(torch.float32)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).abs() / (b.abs() + 1e-3)).mean().item()
    bad = bool(torch.isnan(a).any() or torch.isinf(a).any())
    print(f"  int8 out shape={tuple(out_i8.shape)} dtype={out_i8.dtype} nan/inf={bad}", flush=True)
    print(f"  cosine(int8, bf16)={cos:.5f}  mean_rel_err={rel:.3e}", flush=True)

    ok = (not bad) and cos > 0.99
    print(f"  -> {'PASS' if ok else 'CHECK'} (want cosine>0.99, no nan/inf)", flush=True)

    # timing (int8 vs bf16); decode is launch-bound so this is informational, not the verdict.
    for fn, nm, kw in (
        (fused_experts, "int8", dict(w1=w13_q, w2=w2_q, use_int8_w8a8=True, per_channel_quant=True,
                                     w1_scale=w13_s, w2_scale=w2_s, a1_scale=None, a2_scale=None)),
        (fused_experts, "bf16", dict(w1=w13_deq, w2=w2_deq, use_int8_w8a8=False)),
    ):
        for _ in range(3):
            fn(hidden_states=x, topk_output=topk, moe_runner_config=runner_cfg(), **kw)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(hidden_states=x, topk_output=topk, moe_runner_config=runner_cfg(), **kw)
        torch.xpu.synchronize()
        print(f"  {nm} fused_experts: {(time.perf_counter()-t0)/iters*1e3:.3f} ms/iter", flush=True)
    return ok


ok_decode = run_regime(1, "DECODE", iters=50)
ok_prefill = run_regime(256, "PREFILL", iters=20)

verdict = ok_decode and ok_prefill
print("\n=== ROUTE-A VERDICT:", "GO (int8 fused_moe_triton runs + correct on XPU)"
      if verdict else "NO-GO / NEEDS REVIEW (see output above)", "===", flush=True)
sys.exit(0 if verdict else 1)
