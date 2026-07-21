#!/usr/bin/env python3
# bench_decode_gemv.py -- W8A8 int8 DECODE fast-path A/B on the real qwen3.6-27b shapes.
#
# THE QUESTION (RESEARCH_TODO Track 1a/1b/1c): the W8A8 27B decode is kernel-bound vs fp8.
# Is the lever a new/ reordered small-M int8 GEMV kernel, OR is the int8 GEMM already at the
# weight-bandwidth roofline and the only avoidable cost the per-token ACTIVATION QUANT?
#
# This bench times, on the exact 27B decode GEMM shapes, at M in {1,2,4,6,8} (M~=6 == the
# MTP verify batch), EAGER and XPUGraph-CAPTURED (the serve runs captured):
#
#   BF16              : x @ Wt                                        (baseline, hits roofline at M=1)
#   W8A16 (proposed)  : int8_gemm_w8a16(x_f16, B_s8, wscale)         -- s8 weight, f16 act, NO act-quant
#   W8A8  (current)   : dyn_per_token_int8_quant(x) + int8_gemm_w8a8 -- s8 weight + s8 act (act-quant tax)
#   W8A8-fusedq       : int8_gemm_w8a8_fusedq(x_f16,...)             -- same, one fused op (vllm build)
#   FP8   (the bar)   : fp8_gemm_w8a16(x_f16, B_f8, fscale)          -- 1 byte/wt, no act-quant
#
# For each it reports ms, effective weight-read GB/s (weight_bytes/time), INT8 TOPS-equiv, and a
# whole-model 64-layer linear-decode t/s estimate. The A/B that matters: W8A16 vs W8A8 at each M.
# If W8A16 ~= FP8 ~= roofline and W8A8 is slower by the quant, the lever is ROUTING (skip act-quant
# for small M), NOT a new kernel. See FINDINGS.md.
#
# RUN (coordinator, W8A8 int8 image w/ the built ops; card 0 only):
#   ROOT=/mnt/vm_8tb/b70
#   ./bin/gpu-run --card 0 docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
#     --ipc=host --shm-size 16g -e ZE_AFFINITY_MASK=0 \
#     -v $ROOT/w8a8_kernel:/work/kernel:ro \
#     -v /mnt/vm_8tb/github/b70_ai_things/research/w8a8/decode_gemv:/work/bench:ro \
#     -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so sglang-xpu:woq bash -c \
#     'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
#      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; \
#      python3 /work/bench/bench_decode_gemv.py'
# (For the vllm fusedq op, use B70_XPU_C_SO=$ROOT/w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so and the
#  vllm-xpu-env:int8g-v0240 image; the bench auto-detects whichever ops are present.)
import os, sys, time, ctypes
import torch

DEV = "xpu"
SO = os.environ.get("B70_XPU_C_SO", "/work/kernel/_xpu_C.abi3.so")
BW_CEIL = 581.0e9  # measured B70 read-BW ceiling (docs/kernel/23_b70_gemv_gemm_roofline.md); spec ~608

print("torch", torch.__version__, "xpu", torch.xpu.is_available(), "SO", SO, flush=True)
try:
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
    print("CDLL OK", flush=True)
except OSError as e:
    print("CDLL FAILED:", str(e)[:300], flush=True); sys.exit(1)

ops = torch.ops._xpu_C
HAVE = {nm: hasattr(ops, nm) for nm in
        ("int8_gemm_w8a16", "int8_gemm_w8a8", "int8_gemm_w8a8_fusedq",
         "fp8_gemm_w8a16", "dynamic_per_token_int8_quant")}
for k, v in HAVE.items():
    print(f"  op {k}: {v}", flush=True)
assert HAVE["int8_gemm_w8a16"], "int8_gemm_w8a16 missing -- wrong .so/image"

# ---- exact qwen3.6-27b linear-decode GEMM shapes (hidden 5120, inter 17408,
#      24 q-heads x256 + 4 kv-heads x256 => qkv N=8192, o K=6144) ----
#   name       N       K
SHAPES = [
    ("qkv_proj",  8192,  5120),
    ("o_proj",    5120,  6144),
    ("gate_up",  34816,  5120),
    ("down_proj", 5120, 17408),
]
LAYERS = 64
MS = [1, 2, 4, 6, 8]   # 6 == MTP spec_tokens(5)+1 verify batch


def sync(): torch.xpu.synchronize()


def bench(fn, warm=25, iters=100):
    for _ in range(warm): fn()
    sync(); s = time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter() - s) / iters * 1000.0


def bench_graph(fn_build):
    """fn_build() runs the op once writing into persistent buffers; capture + time replay."""
    for _ in range(10): fn_build()
    sync()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn_build()
    sync()
    return bench(lambda: g.replay())


def q_weight_s8(W):
    amax = W.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)   # [N,1]
    wscale = (amax / 127.0)
    Wq = torch.round(W / wscale).clamp_(-127, 127).to(torch.int8)
    return Wq, wscale.reshape(-1)


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


def tops(M, N, K, ms):
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


# accumulate per-M whole-model linear-path time (sum of 4 ops x 64 layers)
model_ms = {M: {"bf16": 0.0, "w8a16": 0.0, "w8a8": 0.0, "fusedq": 0.0, "fp8": 0.0} for M in MS}
have_path = {"w8a16": True, "w8a8": HAVE["int8_gemm_w8a8"] and HAVE["dynamic_per_token_int8_quant"],
             "fusedq": HAVE["int8_gemm_w8a8_fusedq"], "fp8": HAVE["fp8_gemm_w8a16"]}

for (name, N, K) in SHAPES:
    wbytes_s8 = N * K            # 1 byte/elt
    print(f"\n================ {name}  N={N} K={K}  (s8 wt {wbytes_s8/1e6:.1f} MB, "
          f"roofline {wbytes_s8/BW_CEIL*1e3:.4f} ms @ {BW_CEIL/1e9:.0f} GB/s) ================", flush=True)
    W = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02)
    Wq, wscale = q_weight_s8(W)                  # [N,K] s8, [N]
    B_nt = Wq.t()                                # [K,N] s8 NT view (stride0==1)
    assert B_nt.stride()[0] == 1
    wsc16 = wscale.to(torch.float16)
    w_fp16 = W.to(torch.float16)
    # fp8 weight
    if have_path["fp8"]:
        fs = (W.abs().amax(1, keepdim=True).clamp_(min=1e-8) / 448.0)
        Bf8 = (W / fs).clamp_(-448, 448).to(torch.float8_e4m3fn).t()
        Bfs = fs.reshape(N).to(torch.float16)

    print(f"  {'M':>2} {'path':<12} {'eager ms':>9} {'graph ms':>9} {'GB/s(g)':>8} "
          f"{'TOPS(g)':>8} {'xBF16(g)':>8} {'relerr':>9}", flush=True)

    for M in MS:
        x = (torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.05)
        ref = (x.to(torch.float32) @ w_fp16.t().to(torch.float32))

        def rel(y):
            return ((y.to(torch.float32) - ref).norm() / ref.norm()).item()

        # ---- bf16 ----
        xb = x.clone()
        yb = torch.empty(M, N, device=DEV, dtype=torch.float16)
        e = bench(lambda: torch.matmul(x, w_fp16.t()))
        gb = bench_graph(lambda: yb.copy_(torch.matmul(xb, w_fp16.t())))
        model_ms[M]["bf16"] += gb
        bf16_g = gb
        print(f"  {M:>2} {'bf16':<12} {e:9.4f} {gb:9.4f} {gbps(2*wbytes_s8, gb):8.1f} "
              f"{tops(M,N,K,gb):8.1f} {'1.00':>8} {'0':>9}", flush=True)

        # ---- W8A16 (proposed: no act-quant) ----
        yw = torch.empty(M, N, device=DEV, dtype=torch.float16)
        e = bench(lambda: ops.int8_gemm_w8a16(x, B_nt, wsc16, None))
        gb = bench_graph(lambda: yw.copy_(ops.int8_gemm_w8a16(x, B_nt, wsc16, None)))
        model_ms[M]["w8a16"] += gb
        r = rel(ops.int8_gemm_w8a16(x, B_nt, wsc16, None))
        print(f"  {M:>2} {'W8A16*prop':<12} {e:9.4f} {gb:9.4f} {gbps(wbytes_s8, gb):8.1f} "
              f"{tops(M,N,K,gb):8.1f} {bf16_g/gb:8.2f} {r:9.2e}", flush=True)

        # ---- W8A8 (current: quant + s8s8 gemm) ----
        if have_path["w8a8"]:
            def w8a8_run(xin):
                xq, xs, _ = ops.dynamic_per_token_int8_quant(xin, True, 8)
                return ops.int8_gemm_w8a8(xq, xs, None, B_nt, wsc16, None, None, torch.float16)
            ya = torch.empty(M, N, device=DEV, dtype=torch.float16)
            e = bench(lambda: w8a8_run(x))
            gb = bench_graph(lambda: ya.copy_(w8a8_run(xb)))
            model_ms[M]["w8a8"] += gb
            r = rel(w8a8_run(x))
            print(f"  {M:>2} {'W8A8 curr':<12} {e:9.4f} {gb:9.4f} {gbps(wbytes_s8, gb):8.1f} "
                  f"{tops(M,N,K,gb):8.1f} {bf16_g/gb:8.2f} {r:9.2e}", flush=True)

        # ---- W8A8 fusedq (current vllm: one fused op) ----
        if have_path["fusedq"]:
            yf = torch.empty(M, N, device=DEV, dtype=torch.float16)
            e = bench(lambda: ops.int8_gemm_w8a8_fusedq(x, B_nt, wsc16, None, torch.float16))
            gb = bench_graph(lambda: yf.copy_(ops.int8_gemm_w8a8_fusedq(xb, B_nt, wsc16, None, torch.float16)))
            model_ms[M]["fusedq"] += gb
            r = rel(ops.int8_gemm_w8a8_fusedq(x, B_nt, wsc16, None, torch.float16))
            print(f"  {M:>2} {'W8A8 fusedq':<12} {e:9.4f} {gb:9.4f} {gbps(wbytes_s8, gb):8.1f} "
                  f"{tops(M,N,K,gb):8.1f} {bf16_g/gb:8.2f} {r:9.2e}", flush=True)

        # ---- FP8 bar ----
        if have_path["fp8"]:
            yp = torch.empty(M, N, device=DEV, dtype=torch.float16)
            e = bench(lambda: ops.fp8_gemm_w8a16(x, Bf8, Bfs, None))
            gb = bench_graph(lambda: yp.copy_(ops.fp8_gemm_w8a16(xb, Bf8, Bfs, None)))
            model_ms[M]["fp8"] += gb
            print(f"  {M:>2} {'FP8 bar':<12} {e:9.4f} {gb:9.4f} {gbps(wbytes_s8, gb):8.1f} "
                  f"{tops(M,N,K,gb):8.1f} {bf16_g/gb:8.2f} {'--':>9}", flush=True)


# ---- whole-model 64-layer linear-decode t/s estimate (captured, single card, per-token) ----
# NOTE: linear path only (attn/GDN core + sampling add fixed overhead), so an UPPER bound on t/s.
# The A/B DELTA between W8A16 and W8A8 is the real signal (pure act-quant cost).
print(f"\n================ whole-model linear-decode estimate ({LAYERS} layers, captured, 1 card) ================", flush=True)
print(f"  {'M':>2} {'W8A16 ms/tok':>12} {'W8A8 ms/tok':>12} {'fusedq ms':>10} {'FP8 ms':>10} "
      f"{'W8A16 t/s':>10} {'W8A8 t/s':>10} {'W8A16/W8A8':>11}", flush=True)
for M in MS:
    mm = model_ms[M]
    w16 = mm["w8a16"] * LAYERS
    w8 = mm["w8a8"] * LAYERS if have_path["w8a8"] else float("nan")
    wf = mm["fusedq"] * LAYERS if have_path["fusedq"] else float("nan")
    fp = mm["fp8"] * LAYERS if have_path["fp8"] else float("nan")
    # decode t/s: M tokens produced per (one weight-read pass). For M=1 that's 1 token/pass.
    # For MTP verify M=spec+1, the accepted tokens/pass depends on accept rate; here we report the
    # per-PASS t/s (1000/ms_per_pass) as the kernel throughput, and separately tokens/s if all M accept.
    tps16 = 1000.0 / w16 if w16 else 0.0
    tps8 = 1000.0 / w8 if w8 and w8 == w8 else 0.0
    speedup = (w8 / w16) if (w8 == w8 and w16) else float("nan")
    print(f"  {M:>2} {w16:12.3f} {w8:12.3f} {wf:10.3f} {fp:10.3f} "
          f"{tps16:10.2f} {tps8:10.2f} {speedup:11.3f}", flush=True)

print("\nINTERPRETATION:", flush=True)
print("  * If W8A16(graph) ~= FP8(graph) ~= roofline GB/s at every M, the int8 GEMM is BW-bound and", flush=True)
print("    ALREADY optimal -- a reordered small-M GEMV kernel has no headroom (NO-GO).", flush=True)
print("  * If W8A8 > W8A16 by ~the act-quant, the lever is ROUTING: send small-M (decode + MTP", flush=True)
print("    verify, M<=~64) through the quant-free W8A16 op; reserve W8A8/fusedq for M>~64 prefill.", flush=True)
print("  * W8A16/W8A8 column = the per-pass speedup from dropping the activation quant.", flush=True)
