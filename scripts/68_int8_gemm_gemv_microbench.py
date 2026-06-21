#!/usr/bin/env python3
"""
68_int8_gemm_gemv_microbench.py
INT8 vs BF16 GEMM (prefill, large M) and GEMV (decode, M=1..8) microbench
for the Intel Arc Pro B70 (Xe2/Battlemage, 367 INT8 TOPS, 608 GB/s).

Shape source: docs/literature/08_int8_gemm_gemv_xe2_frontier.md sections C.1 and C.2.
INT8 op: torch.ops._xpu_C.int8_gemm_w8a8 when available (contrib/vllm_int8_xpu);
         falls back to torch._int_mm (int8 x int8 -> int32) for portability.

Run via:  scripts/68_run_microbench.sh  (which gates on gpu-run)
Direct:   python 68_int8_gemm_gemv_microbench.py [STAMP]

Output CSV: results/microbench_gemm_gemv_<stamp>.csv
Columns:   shape_label,M,N,K,dtype,ms,tflops,gbps,speedup_vs_bf16
"""

import sys
import os
import csv
import time
import datetime
import traceback

import torch

# ---------------------------------------------------------------------------
# GPU assertion
# ---------------------------------------------------------------------------
assert torch.xpu.is_available(), "XPU not available -- run inside vllm-xpu-env:int8 image"

DEVICE = torch.device("xpu:0")
BW_PEAK_GBPS  = 608.0   # B70 GDDR6 peak bandwidth  (GB/s)
TOPS_PEAK_INT8 = 367.0  # B70 INT8 TOPS peak

# ---------------------------------------------------------------------------
# Detect the custom oneDNN W8A8 op
# ---------------------------------------------------------------------------
try:
    import vllm._xpu_ops  # noqa: F401  trigger .so load
except Exception:
    pass

_HAS_CUSTOM_INT8 = hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "int8_gemm_w8a8")

if _HAS_CUSTOM_INT8:
    INT8_API = "torch.ops._xpu_C.int8_gemm_w8a8"
else:
    INT8_API = "torch._int_mm"

print(f"[bench] XPU device: {torch.xpu.get_device_name(0)}", flush=True)
print(f"[bench] INT8 API: {INT8_API}", flush=True)

# ---------------------------------------------------------------------------
# Shape tables -- from doc 08 sections A.1, A.2, A.3
# Each entry: (label, K, N)
# ---------------------------------------------------------------------------

# Shape group 1: Qwen3-14B attention + MLP (verified A1-A4)
SHAPES_GROUP1 = [
    ("14B_attn_QO",       5120,  5120),   # A1 Q/K/V/O proj (square)
    ("14B_attn_KV",       5120,  1024),   # A2 K,V proj (low-N)
    ("14B_mlp_gate_up",   5120, 17408),   # A3 gate/up_proj (wide-N)
    ("14B_mlp_down",     17408,  5120),   # A4 down_proj (wide-K)
]

# Shape group 2: Qwen3.6-27B unique (verified B1, B3)
# B2 (K=5120,N=1024) is identical to A2 -- skip duplicate
SHAPES_GROUP2 = [
    ("27B_attn_Q",        5120,  6144),   # B1
    ("27B_attn_O",        6144,  5120),   # B3
]

# Shape group 3: Qwen3.6-35B-A3B MoE (verified C1-C5)
# C4 (K=2048,N=512) is identical to C1/expert shapes -- skip
SHAPES_GROUP3 = [
    ("35B_expert_gate_up", 2048,  512),   # C1
    ("35B_expert_down",     512, 2048),   # C2
    ("35B_attn_Q",         2048, 4096),   # C3
    ("35B_attn_O",         4096, 2048),   # C5
    ("35B_dense_sq",       2048, 2048),   # C6
]

# Shape group 4: square/reference for XMX calibration (doc 08 C.1)
SHAPES_GROUP4 = [
    ("ref_sq4096",    4096,  4096),
    ("ref_ffn11008",  4096, 11008),
    ("ref_sq8192",    8192,  8192),
]

ALL_KN_SHAPES = SHAPES_GROUP1 + SHAPES_GROUP2 + SHAPES_GROUP3 + SHAPES_GROUP4
# 4 + 2 + 5 + 3 = 14 distinct (K,N) shapes

# ---------------------------------------------------------------------------
# GEMM sweep M values (prefill, large M)
# ---------------------------------------------------------------------------
GEMM_M_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]

# For the large reference square shapes at large-M saturation we add M=8192
LARGE_M_EXTRA = {
    "ref_sq4096": [64, 256, 512, 1024, 2048, 4096, 8192],
    "ref_sq8192": [256, 512, 1024, 2048, 4096],
}

# Shapes that skip the largest M values to stay under memory budget
M_LIMIT = {
    "ref_sq8192": 4096,
}

# GEMV sweep M values (decode)
GEMV_M_VALUES = [1, 2, 4, 8]

# ---------------------------------------------------------------------------
# Warmup / timing config
# ---------------------------------------------------------------------------
WARMUP_ITERS  = 50
TIMED_ITERS   = 200

# ---------------------------------------------------------------------------
# BF16 GEMM helper
# ---------------------------------------------------------------------------

def run_bf16_gemm(A_bf16, B_bf16):
    """[M,K] x [K,N] -> [M,N] in bf16."""
    return torch.matmul(A_bf16, B_bf16)


# ---------------------------------------------------------------------------
# INT8 GEMM helpers
# ---------------------------------------------------------------------------

def _make_int8_scales(M, K, N, dtype=torch.float32):
    """Return trivial per-token act scale [M,1] and per-channel weight scale [1,N]."""
    a_scale = torch.ones(M, 1, dtype=dtype, device=DEVICE)
    w_scale = torch.ones(1, N, dtype=dtype, device=DEVICE)
    return a_scale, w_scale


def run_int8_gemm_custom(A_i8, A_scale, B_i8, W_scale, out_dtype=torch.bfloat16):
    """torch.ops._xpu_C.int8_gemm_w8a8: A[M,K] x B[K,N] -> [M,N] out_dtype."""
    # B is already in [K,N] layout (NT path in apply_weights transposes to [K,N]).
    return torch.ops._xpu_C.int8_gemm_w8a8(
        A_i8,     # [M, K] int8  (activations)
        A_scale,  # [M, 1] f32   (per-token)
        None,     # A_zp: symmetric -> None
        B_i8,     # [K, N] int8  (weights, NT)
        W_scale,  # [1, N] f32   (per-channel)
        None,     # azp_adj: symmetric -> None
        None,     # bias: None
        out_dtype,
    )


def run_int8_gemm_fallback(A_i8, B_i8):
    """torch._int_mm: A[M,K] x B[K,N] -> [M,N] int32 (portable fallback)."""
    return torch._int_mm(A_i8, B_i8)


# ---------------------------------------------------------------------------
# Timing kernel
# ---------------------------------------------------------------------------

def time_op(fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    """
    Returns median wall time in milliseconds.
    fn() must not return a generator.  XPU synchronize() brackets each call.
    """
    # warmup
    for _ in range(warmup):
        out = fn()
        torch.xpu.synchronize()

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)

    times_ms.sort()
    return times_ms[len(times_ms) // 2]  # median


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_tflops(M, K, N, ms):
    flops = 2.0 * M * K * N
    return flops / (ms * 1e-3) / 1e12


def compute_gbps_gemm(M, K, N, ms, weight_bytes_per_elem=2):
    """
    BW = (weight bytes + act bytes + output bytes) / time.
    For BF16: 2 bytes/elem throughout.
    For INT8 weight read: 1 byte/elem for W, 1 for A, 2 for BF16 output.
    weight_bytes_per_elem: 2 for bf16, 1 for int8.
    """
    w_bytes = K * N * weight_bytes_per_elem
    a_bytes = M * K * weight_bytes_per_elem  # act same dtype as weight approximation
    o_bytes = M * N * 2                      # output always bf16
    total   = w_bytes + a_bytes + o_bytes
    return total / (ms * 1e-3) / 1e9


def compute_gbps_gemv(M, K, N, ms, weight_bytes_per_elem=1):
    """
    GEMV BW: dominated by weight reads.  Use same formula as compute_gbps_gemm
    but the activation is tiny relative to weights at small M.
    """
    return compute_gbps_gemm(M, K, N, ms, weight_bytes_per_elem)


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

rows = []  # list of dicts matching CSV columns


def record(shape_label, M, N, K, dtype, ms, regime):
    tflops = compute_tflops(M, K, N, ms)
    w_bpe  = 1 if "int8" in dtype else 2
    gbps   = compute_gbps_gemm(M, K, N, ms, w_bpe)
    rows.append(dict(
        shape_label=shape_label,
        M=M, N=N, K=K,
        dtype=dtype,
        ms=round(ms, 4),
        tflops=round(tflops, 4),
        gbps=round(gbps, 2),
        speedup_vs_bf16=None,   # filled in after pairing
        regime=regime,
    ))


# ---------------------------------------------------------------------------
# GEMM sweep (prefill, M=64..4096)
# ---------------------------------------------------------------------------

print("\n[bench] ===== GEMM SWEEP (prefill) =====", flush=True)

for label, K, N in ALL_KN_SHAPES:
    m_list = LARGE_M_EXTRA.get(label, GEMM_M_VALUES)
    m_limit = M_LIMIT.get(label, 8192)
    m_list = [m for m in m_list if m <= m_limit]

    for M in m_list:
        # Allocate tensors
        try:
            A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
            B_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)
        except Exception as e:
            print(f"  [SKIP alloc bf16 {label} M={M}]: {e}", flush=True)
            continue

        # --- BF16 timing ---
        try:
            ms_bf16 = time_op(lambda: run_bf16_gemm(A_bf16, B_bf16))
            record(label, M, N, K, "bf16", ms_bf16, "GEMM")
            print(f"  bf16  {label:25s} M={M:5d} K={K:6d} N={N:6d}  {ms_bf16:8.3f} ms  "
                  f"{compute_tflops(M,K,N,ms_bf16):.2f} TFLOP/s", flush=True)
        except Exception:
            print(f"  [FAIL bf16 GEMM {label} M={M}]:\n{traceback.format_exc()}", flush=True)

        # --- INT8 timing ---
        try:
            A_i8 = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=DEVICE)
            B_i8 = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=DEVICE)
            a_scale, w_scale = _make_int8_scales(M, K, N)

            if _HAS_CUSTOM_INT8:
                ms_int8 = time_op(lambda: run_int8_gemm_custom(A_i8, a_scale, B_i8, w_scale))
            else:
                ms_int8 = time_op(lambda: run_int8_gemm_fallback(A_i8, B_i8))

            record(label, M, N, K, "int8", ms_int8, "GEMM")
            print(f"  int8  {label:25s} M={M:5d} K={K:6d} N={N:6d}  {ms_int8:8.3f} ms  "
                  f"{compute_tflops(M,K,N,ms_int8):.2f} TFLOP/s", flush=True)
        except Exception:
            print(f"  [FAIL int8 GEMM {label} M={M}]:\n{traceback.format_exc()}", flush=True)

        del A_bf16, B_bf16
        try:
            del A_i8, B_i8, a_scale, w_scale
        except Exception:
            pass
        torch.xpu.synchronize()


# ---------------------------------------------------------------------------
# GEMV sweep (decode, M=1..8)
# Also include BW comparison across dtypes for selected shapes.
# ---------------------------------------------------------------------------

print("\n[bench] ===== GEMV SWEEP (decode) =====", flush=True)

# Primary: all (K,N) at M in {1,2,4,8}
for label, K, N in ALL_KN_SHAPES:
    for M in GEMV_M_VALUES:
        # BF16
        try:
            A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
            B_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)
            ms_bf16 = time_op(lambda: run_bf16_gemm(A_bf16, B_bf16))
            record(label, M, N, K, "bf16", ms_bf16, "GEMV")
            print(f"  bf16  {label:25s} M={M:2d} K={K:6d} N={N:6d}  {ms_bf16:8.3f} ms  "
                  f"{compute_gbps_gemv(M,K,N,ms_bf16,2):.1f} GB/s", flush=True)
            del A_bf16, B_bf16
        except Exception:
            print(f"  [FAIL bf16 GEMV {label} M={M}]:\n{traceback.format_exc()}", flush=True)

        # INT8
        try:
            A_i8 = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=DEVICE)
            B_i8 = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=DEVICE)
            a_scale, w_scale = _make_int8_scales(M, K, N)

            if _HAS_CUSTOM_INT8:
                ms_int8 = time_op(lambda: run_int8_gemm_custom(A_i8, a_scale, B_i8, w_scale))
            else:
                ms_int8 = time_op(lambda: run_int8_gemm_fallback(A_i8, B_i8))

            record(label, M, N, K, "int8", ms_int8, "GEMV")
            print(f"  int8  {label:25s} M={M:2d} K={K:6d} N={N:6d}  {ms_int8:8.3f} ms  "
                  f"{compute_gbps_gemv(M,K,N,ms_int8,1):.1f} GB/s", flush=True)
            del A_i8, B_i8, a_scale, w_scale
        except Exception:
            print(f"  [FAIL int8 GEMV {label} M={M}]:\n{traceback.format_exc()}", flush=True)

        torch.xpu.synchronize()

# ---------------------------------------------------------------------------
# Column-reorder layout variant for GEMV (P4 reorder path, M=1 only)
# Simulates W_col = W.t().contiguous() layout and re-measures GEMV to expose
# any coalescing difference. Uses the fallback _int_mm path since the custom
# op is always called with NT weight; for the reorder variant we use matmul
# on BF16 with transposed weight as a proxy, and int8 torch._int_mm with
# col-major B as the int8 variant.
# ---------------------------------------------------------------------------

print("\n[bench] ===== GEMV COL-REORDER LAYOUT (P4, M=1) =====", flush=True)

for label, K, N in ALL_KN_SHAPES:
    M = 1
    # BF16 col-reorder: A[1,K] x B_col[N,K].t() -- simulate N-first weight
    try:
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
        B_col  = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)  # col-major layout
        # matmul A [1,K] x B_col.t() [K,N]
        ms = time_op(lambda: torch.matmul(A_bf16, B_col.t()))
        record(label + "_colorder", M, N, K, "bf16_colorder", ms, "GEMV_COLORDER")
        print(f"  bf16_col {label:25s} M={M} K={K:6d} N={N:6d}  {ms:8.3f} ms  "
              f"{compute_gbps_gemv(M,K,N,ms,2):.1f} GB/s", flush=True)
        del A_bf16, B_col
    except Exception:
        print(f"  [FAIL bf16_col GEMV {label}]:\n{traceback.format_exc()}", flush=True)

    # INT8 col-reorder: A[1,K] x B_col[N,K].t()
    try:
        A_i8   = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=DEVICE)
        B_col_i8 = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=DEVICE)
        B_col_t  = B_col_i8.t().contiguous()   # [K, N] col-major values (simulated reorder)
        ms = time_op(lambda: torch._int_mm(A_i8, B_col_t))
        record(label + "_colorder", M, N, K, "int8_colorder", ms, "GEMV_COLORDER")
        print(f"  int8_col {label:25s} M={M} K={K:6d} N={N:6d}  {ms:8.3f} ms  "
              f"{compute_gbps_gemv(M,K,N,ms,1):.1f} GB/s", flush=True)
        del A_i8, B_col_i8, B_col_t
    except Exception:
        print(f"  [FAIL int8_col GEMV {label}]:\n{traceback.format_exc()}", flush=True)

    torch.xpu.synchronize()

# ---------------------------------------------------------------------------
# Per-quant BW comparison at M=1 and M=8 for selected shapes (doc 08 C.2)
# Shapes: 14B MLP gate/up, 14B MLP down, 35B expert gate/up, 35B expert down
# Dtypes: bf16, int8 (already covered above), plus int4 weight simulation
# ---------------------------------------------------------------------------

print("\n[bench] ===== BW COMPARISON (int4 weight sim, selected shapes) =====", flush=True)

BW_SHAPES = [
    ("14B_mlp_gate_up",   5120, 17408),
    ("14B_mlp_down",     17408,  5120),
    ("35B_expert_gate_up", 2048,  512),
    ("35B_expert_down",     512, 2048),
]

for label, K, N in BW_SHAPES:
    for M in [1, 8]:
        # Simulate int4 weight BW cost: store weights as int8 but only K*N/2 bytes
        # (int4 packs 2 values per byte). We time a [M, K] x [K, N//2] int8 matmul
        # where N//2 accounts for the packed dimension, then report BW as if N weights
        # were read at 0.5 bytes/elem.
        # NOTE: this does NOT produce a correct output -- it is a BW proxy only.
        N_packed = max(N // 2, 1)
        try:
            A_i8 = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=DEVICE)
            B_i4_packed = torch.randint(-127, 127, (K, N_packed), dtype=torch.int8,
                                        device=DEVICE)
            ms = time_op(lambda: torch._int_mm(A_i8, B_i4_packed))
            # Report BW as if we read K*N*0.5 weight bytes (int4)
            w_bytes_i4 = K * N * 0.5
            a_bytes    = M * K * 2   # bf16 activations
            o_bytes    = M * N * 2
            total_bytes = w_bytes_i4 + a_bytes + o_bytes
            gbps = total_bytes / (ms * 1e-3) / 1e9
            rows.append(dict(
                shape_label=label + "_bwcmp",
                M=M, N=N, K=K,
                dtype="int4_weight_sim",
                ms=round(ms, 4),
                tflops=round(compute_tflops(M, K, N_packed, ms), 4),
                gbps=round(gbps, 2),
                speedup_vs_bf16=None,
                regime="GEMV_BWCMP",
            ))
            print(f"  int4sim {label:25s} M={M} K={K:6d} N={N:6d}  {ms:8.3f} ms  "
                  f"{gbps:.1f} GB/s (int4-weight BW proxy)", flush=True)
            del A_i8, B_i4_packed
        except Exception:
            print(f"  [FAIL int4sim {label} M={M}]:\n{traceback.format_exc()}", flush=True)

        torch.xpu.synchronize()

# ---------------------------------------------------------------------------
# Compute speedup_vs_bf16 for each row by pairing with its bf16 counterpart
# ---------------------------------------------------------------------------

# Build lookup: (shape_label, M, N, K, regime) -> bf16 ms
bf16_map = {}
for r in rows:
    if r["dtype"] in ("bf16",):
        key = (r["shape_label"], r["M"], r["N"], r["K"], r["regime"])
        bf16_map[key] = r["ms"]

for r in rows:
    if r["dtype"] not in ("bf16",):
        key = (r["shape_label"], r["M"], r["N"], r["K"], r["regime"])
        bf16_ms = bf16_map.get(key)
        if bf16_ms and r["ms"] > 0:
            r["speedup_vs_bf16"] = round(bf16_ms / r["ms"], 4)

# ---------------------------------------------------------------------------
# Emit CSV
# ---------------------------------------------------------------------------

stamp = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# results/ sits NEXT TO this script (host is FLAT layout: script at /mnt/vm_8tb/b70/, results at .../results).
# Honor B70_RESULTS_DIR override; else <script_dir>/results (mounted), NOT dirname(dirname) which escaped the mount.
results_dir = os.environ.get("B70_RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, f"microbench_gemm_gemv_{stamp}.csv")

FIELDS = ["shape_label", "M", "N", "K", "dtype", "regime",
          "ms", "tflops", "gbps", "speedup_vs_bf16"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[bench] CSV written -> {csv_path}  ({len(rows)} rows)", flush=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n[bench] ===== SUMMARY: int8 vs bf16 speedup (GEMM, median M=512) =====")
print(f"  {'shape':30s}  {'K':>6}  {'N':>6}  {'bf16_ms':>8}  {'int8_ms':>8}  {'speedup':>7}")
for label, K, N in ALL_KN_SHAPES:
    M = 512
    k_bf16 = (label, M, N, K, "GEMM")
    k_int8 = k_bf16
    bf16_ms = bf16_map.get(k_bf16)
    int8_row = next((r for r in rows
                     if r["shape_label"] == label and r["M"] == M
                     and r["dtype"] == "int8" and r["regime"] == "GEMM"), None)
    if bf16_ms and int8_row:
        spd = round(bf16_ms / int8_row["ms"], 3)
        print(f"  {label:30s}  {K:6d}  {N:6d}  {bf16_ms:8.3f}  {int8_row['ms']:8.3f}  {spd:7.3f}x")

print("\n[bench] ===== SUMMARY: int8 vs bf16 BW (GEMV, M=1) =====")
print(f"  {'shape':30s}  {'K':>6}  {'N':>6}  {'bf16_GB/s':>10}  {'int8_GB/s':>10}  {'speedup':>7}")
for label, K, N in ALL_KN_SHAPES:
    M = 1
    bf16_ms = bf16_map.get((label, M, N, K, "GEMV"))
    int8_row = next((r for r in rows
                     if r["shape_label"] == label and r["M"] == M
                     and r["dtype"] == "int8" and r["regime"] == "GEMV"), None)
    if bf16_ms and int8_row:
        gbps_bf16 = compute_gbps_gemv(M, K, N, bf16_ms, 2)
        gbps_int8 = compute_gbps_gemv(M, K, N, int8_row["ms"], 1)
        spd = round(bf16_ms / int8_row["ms"], 3)
        print(f"  {label:30s}  {K:6d}  {N:6d}  {gbps_bf16:10.1f}  {gbps_int8:10.1f}  {spd:7.3f}x")

print(f"\n[bench] INT8 API used: {INT8_API}")
print(f"[bench] done.  Results: {csv_path}")
