#!/usr/bin/env python3
# bench_w4a8_shapes.py -- K1 isolated kernel matrix on Qwen3.8-27B GEMM shapes.
#
# Copy/extend of sglang/int4_gemm_w4a8_probe.py (that file is 3.6 down_proj-only).
# Campaign: docs/20260820_qwen38_w4a8_campaign.md section 8.2 / loop K1.
# 3.8 W4A8 file is NOT required. Packed int4 stand-in = 3.6 w4a8-sqgptq where the
# shape matches; synthetic pack for GDN (BF16 on 3.6) and any miss.
#
# Paths timed (skip if the op is missing from the image/.so):
#   bf16            F.linear roof control
#   w4a16           int4_gemm_w4a16          Path H decode (fp16 act, no act-quant)
#   w4a8_full       eager act-quant + int4_gemm_w4a8   Path X unfused
#   w4a8_op         int4_gemm_w4a8 op-only   (quant excluded -- honesty split)
#   w8a16           int8_gemm_w8a16          if present
#   w8a8_full       dyn quant + int8_gemm_w8a8
#   w8a8_fusedq     int8_gemm_w8a8_fusedq    if present
#
# Reports ms, weight-read GB/s vs 581, INT8-TOPS-equiv, vs-bf16, relerr.
# ASCII only. Card 1. Do not torch.compile the act-quant (D05).
import os
import sys
import time
import csv
import ctypes

import torch
import torch.nn.functional as F

DEV = "xpu"
GROUP = int(os.environ.get("GROUP", "128"))
BW_CEIL = 581.0e9
SO = os.environ.get("B70_XPU_C_SO", "")
CKPT = os.environ.get("CKPT", "/models/qwen3.6-27b/w4a8-sqgptq")
OUT_CSV = os.environ.get("OUT_CSV", "")
INCLUDE_LMHEAD = os.environ.get("INCLUDE_LMHEAD", "0") == "1"

# Campaign 8.2 shapes: (name, K, N). F.linear is x[M,K] @ W[N,K].T
SHAPES = [
    ("gate_up", 5120, 34816),     # fused gate+up
    ("down_proj", 17408, 5120),   # 101 us quant lives here
    ("qkv_gate", 5120, 14336),    # 16 full-attn layers; q already has output-gate
    ("o_proj", 6144, 5120),
    ("gdn_qkvz", 5120, 16384),    # 3.6 is BF16; 151 -> INT8
    ("gdn_out", 6144, 5120),
    ("gdn_ba", 5120, 96),         # trap; do not XMX-hero this
]
if INCLUDE_LMHEAD:
    SHAPES.append(("lm_head", 5120, 248320))
_EXTRA = os.environ.get("EXTRA_SHAPES", "").strip()
if _EXTRA:
    for _item in _EXTRA.split(","):
        _n, _k, _N = _item.split(":")
        SHAPES.append((_n.strip(), int(_k), int(_N)))

MS = [1, 2, 4, 8, 16, 32, 64, 256, 2048]
_ONLY_MS = os.environ.get("ONLY_MS", "").strip()
if _ONLY_MS:
    MS = [int(x) for x in _ONLY_MS.split(",") if x.strip()]
_ONLY_SHAPES = os.environ.get("ONLY_SHAPES", "").strip()
if _ONLY_SHAPES:
    _want = {x.strip() for x in _ONLY_SHAPES.split(",") if x.strip()}
    SHAPES = [s for s in SHAPES if s[0] in _want]


def sync():
    torch.xpu.synchronize()


def niters(M):
    if M <= 4:
        return 40, 100
    if M <= 32:
        return 20, 50
    if M <= 256:
        return 10, 24
    return 8, 16


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


def tops(M, N, K, ms):
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def pack_int4(w_i8):
    # [N,K] int8 in [-8,7] -> [N, K/8] int32. Matches XPUW4A8IntLinearKernel.
    assert w_i8.dtype == torch.int8 and w_i8.shape[1] % 8 == 0
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=w_i8.device)
    u = (w_i8.to(torch.int32) + 8).reshape(w_i8.shape[0], w_i8.shape[1] // 8, 8)
    return ((u & 0xF) << shifts[None, None, :]).sum(dim=2).to(torch.int32)


def unpack_int4_fp16(wq_i32, ws, group=GROUP):
    n, k8 = wq_i32.shape
    shifts = torch.arange(0, 32, 4, device=wq_i32.device, dtype=torch.int32)
    nib = (wq_i32.unsqueeze(-1) >> shifts) & 0xF
    vals = (nib.to(torch.int32) - 8).reshape(n, k8 * 8)
    ws_full = ws.to(torch.float32).repeat_interleave(group, dim=1)
    return (vals.to(torch.float32) * ws_full).to(torch.float16)


def q_weight_s8(w_fp16):
    w = w_fp16.to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
    wscale = amax / 127.0
    wq = torch.round(w / wscale).clamp_(-127, 127).to(torch.int8)
    return wq, wscale.reshape(-1).to(torch.float16)


def eager_act_quant(x):
    amax = x.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
    xs = (amax / 127.0).to(x.dtype).contiguous()
    xq = (x / xs).round().clamp_(-127, 127).to(torch.int8).contiguous()
    xz = torch.zeros(x.shape[0], 1, dtype=torch.int32, device=x.device).contiguous()
    return xq, xs, xz


def load_ops():
    print("torch", torch.__version__, "xpu", torch.xpu.is_available(),
          "device0", torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
          "SO", SO or "(image)", flush=True)
    if not torch.xpu.is_available():
        print("FAIL: no XPU", flush=True)
        sys.exit(2)
    if SO:
        try:
            ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
            print("CDLL OK", SO, flush=True)
        except OSError as e:
            print("CDLL FAILED:", str(e)[:300], flush=True)
    if not hasattr(torch.ops, "_xpu_C") or not hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):
        try:
            import vllm_xpu_kernels._xpu_C  # noqa: F401
            print("imported vllm_xpu_kernels._xpu_C", flush=True)
        except Exception as e:
            print("vllm_xpu_kernels import failed:", type(e).__name__, str(e)[:200], flush=True)
    ops = torch.ops._xpu_C
    names = [
        "int4_gemm_w4a8", "int4_gemm_w4a16",
        "int8_gemm_w8a8", "int8_gemm_w8a16", "int8_gemm_w8a8_fusedq",
        "dynamic_per_token_int8_quant",
    ]
    have = {nm: hasattr(ops, nm) for nm in names}
    for k, v in have.items():
        print(f"  op {k}: {v}", flush=True)
    if not have["int4_gemm_w4a8"]:
        print("FAIL: int4_gemm_w4a8 missing -- wrong image/.so", flush=True)
        sys.exit(3)
    quant_fn = None
    if have["dynamic_per_token_int8_quant"]:
        def quant_fn(x):
            return ops.dynamic_per_token_int8_quant(x, True, 8)
        print("act-quant: _xpu_C.dynamic_per_token_int8_quant", flush=True)
    else:
        try:
            from vllm._xpu_ops import xpu_ops as vops
            def quant_fn(x):
                return vops.dynamic_per_token_int8_quant_ref(x, True, 8)
            print("act-quant: vllm._xpu_ops.dynamic_per_token_int8_quant_ref", flush=True)
        except Exception as e:
            quant_fn = eager_act_quant
            print("act-quant: eager fallback", type(e).__name__, flush=True)
    return ops, have, quant_fn


def load_slice(path, key):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def try_real_packed(name):
    """Return (wq [N,K/8] cpu i32, ws [N,K/g] cpu) or None."""
    shard = os.path.join(CKPT, "model.safetensors")
    if not os.path.isfile(shard):
        return None
    L0 = "model.language_model.layers.0.mlp"
    L3 = "model.language_model.layers.3.self_attn"
    try:
        if name == "down_proj":
            w = load_slice(shard, f"{L0}.down_proj.weight")
            s = load_slice(shard, f"{L0}.down_proj.weight_scale")
            return w, s
        if name == "gate_up":
            wg = load_slice(shard, f"{L0}.gate_proj.weight")
            sg = load_slice(shard, f"{L0}.gate_proj.weight_scale")
            wu = load_slice(shard, f"{L0}.up_proj.weight")
            su = load_slice(shard, f"{L0}.up_proj.weight_scale")
            return torch.cat([wg, wu], dim=0), torch.cat([sg, su], dim=0)
        if name == "qkv_gate":
            # 3.6/3.8 q_proj already includes attn_output_gate (N=12288); +k +v = 14336
            wq = load_slice(shard, f"{L3}.q_proj.weight")
            sq = load_slice(shard, f"{L3}.q_proj.weight_scale")
            wk = load_slice(shard, f"{L3}.k_proj.weight")
            sk = load_slice(shard, f"{L3}.k_proj.weight_scale")
            wv = load_slice(shard, f"{L3}.v_proj.weight")
            sv = load_slice(shard, f"{L3}.v_proj.weight_scale")
            return torch.cat([wq, wk, wv], dim=0), torch.cat([sq, sk, sv], dim=0)
        if name == "o_proj":
            w = load_slice(shard, f"{L3}.o_proj.weight")
            s = load_slice(shard, f"{L3}.o_proj.weight_scale")
            return w, s
        if name == "gdn_out":
            # 3.6 GDN out is BF16; o_proj is the same K,N packed int4 stand-in
            w = load_slice(shard, f"{L3}.o_proj.weight")
            s = load_slice(shard, f"{L3}.o_proj.weight_scale")
            return w, s
    except Exception as e:
        print(f"  real-ckpt {name} miss: {type(e).__name__}: {e}", flush=True)
        return None
    return None


def make_packed(name, K, N, device):
    real = try_real_packed(name)
    src = "synthetic"
    if real is not None:
        wq, ws = real
        if tuple(wq.shape) == (N, K // 8) and tuple(ws.shape) == (N, K // GROUP):
            src = "3.6-sqgptq"
            wq = wq.to(device)
            ws = ws.to(device)
            print(f"  weights: {src} {tuple(wq.shape)} {wq.dtype} scale {tuple(ws.shape)} {ws.dtype}",
                  flush=True)
            return wq, ws, src
        print(f"  real shape mismatch {name}: w={tuple(wq.shape)} s={tuple(ws.shape)} "
              f"want {(N, K // 8)} / {(N, K // GROUP)} -- synthesizing", flush=True)
    w_i8 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
    ws = (torch.rand(N, K // GROUP, device=device, dtype=torch.float16) * 0.05 + 0.001)
    wq = pack_int4(w_i8)
    print(f"  weights: {src} packed {tuple(wq.shape)} {wq.dtype} scale {tuple(ws.shape)}", flush=True)
    return wq, ws, src


def relerr(y, ref):
    yf = y.to(torch.float32)
    rf = ref.to(torch.float32)
    n = rf.norm()
    if float(n) == 0.0:
        return float("nan")
    return float((yf - rf).norm() / n)


def main():
    ops, have, act_quant = load_ops()
    wzp = torch.tensor([8], dtype=torch.int8, device=DEV)
    rows = []
    print(f"CKPT={CKPT} GROUP={GROUP} MS={MS}", flush=True)
    print(f"BW_CEIL={BW_CEIL/1e9:.0f} GB/s", flush=True)

    for name, K, N in SHAPES:
        wbytes_i4 = N * K * 0.5
        wbytes_i8 = N * K
        wbytes_bf = N * K * 2
        roof_i4_ms = wbytes_i4 / BW_CEIL * 1e3
        print(f"\n================ {name}  K={K} N={N}  "
              f"i4 {wbytes_i4/1e6:.1f} MB  roof {roof_i4_ms:.4f} ms @ {BW_CEIL/1e9:.0f} GB/s "
              f"================", flush=True)
        try:
            wq, ws, src = make_packed(name, K, N, DEV)
        except Exception as e:
            print(f"  SKIP make_packed: {type(e).__name__}: {e}", flush=True)
            continue
        # B MUST be NT view, stride[0]==1. Do NOT .contiguous() the view away.
        qweight = wq.t()
        if qweight.stride()[0] != 1:
            print(f"  FAIL NT: qweight.stride={qweight.stride()} (need stride[0]==1)", flush=True)
            continue
        wscale = ws.t().contiguous()
        w_fp16 = unpack_int4_fp16(wq, ws)
        w_bf16 = w_fp16.to(torch.bfloat16)
        B_s8, wsc16 = q_weight_s8(w_fp16)
        B_nt = B_s8.t()
        assert B_nt.stride()[0] == 1

        hdr = (f"  {'M':>5} {'path':<12} {'ms':>9} {'GB/s':>8} {'TOPS':>8} "
               f"{'%roof':>7} {'xBF16':>7} {'relerr':>10} {'src':<11}")
        print(hdr, flush=True)

        for M in MS:
            warm, iters = niters(M)
            x16 = torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.05
            xbf = x16.to(torch.bfloat16)
            try:
                xq, xs, xz = act_quant(x16)
                if xs.dtype != torch.float16:
                    xs = xs.to(torch.float16)
                xz = xz.to(torch.int32)
            except Exception as e:
                print(f"  {M:>5} act-quant FAIL {type(e).__name__}: {e}", flush=True)
                continue

            ref_bf = F.linear(xbf, w_bf16)

            def run_path(path, wbytes, fn, ref):
                try:
                    y = fn()
                    sync()
                    re = relerr(y, ref) if ref is not None else float("nan")
                    ms = bench(fn, warm, iters)
                    g = gbps(wbytes, ms)
                    t = tops(M, N, K, ms)
                    pct = 100.0 * g / (BW_CEIL / 1e9)
                    row = {
                        "shape": name, "M": M, "K": K, "N": N, "path": path,
                        "ms": ms, "gbps": g, "tops": t, "pct_roof": pct,
                        "relerr": re, "src": src, "wbytes": wbytes,
                    }
                    rows.append(row)
                    return row, y
                except Exception as e:
                    print(f"  {M:>5} {path:<12} FAIL {type(e).__name__}: {str(e)[:180]}",
                          flush=True)
                    rows.append({
                        "shape": name, "M": M, "K": K, "N": N, "path": path,
                        "ms": float("nan"), "gbps": float("nan"), "tops": float("nan"),
                        "pct_roof": float("nan"), "relerr": float("nan"),
                        "src": f"FAIL:{type(e).__name__}", "wbytes": wbytes,
                    })
                    return None, None

            def emit(row, vs):
                if row is None:
                    return
                xbf_s = f"{vs:.2f}" if vs == vs else "  n/a"
                re_s = f"{row['relerr']:.2e}" if row["relerr"] == row["relerr"] else "n/a"
                print(f"  {M:>5} {row['path']:<12} {row['ms']:9.4f} {row['gbps']:8.1f} "
                      f"{row['tops']:8.2f} {row['pct_roof']:6.1f}% {xbf_s:>7} {re_s:>10} "
                      f"{row['src']:<11}", flush=True)

            r_bf, _ = run_path("bf16", wbytes_bf,
                               lambda: F.linear(xbf, w_bf16), ref_bf)
            bf_ms = r_bf["ms"] if r_bf else float("nan")
            emit(r_bf, 1.0)

            if have["int4_gemm_w4a16"]:
                r, _ = run_path(
                    "w4a16", wbytes_i4,
                    lambda: ops.int4_gemm_w4a16(x16, qweight, None, wscale, wzp, GROUP, None),
                    ref_bf)
                emit(r, (bf_ms / r["ms"]) if r and r["ms"] else float("nan"))

            def w4a8_full():
                q, s, z = act_quant(x16)
                if s.dtype != torch.float16:
                    s = s.to(torch.float16)
                return ops.int4_gemm_w4a8(q, s, z.to(torch.int32), qweight, wscale,
                                          wzp, GROUP, None, None)

            r, _ = run_path("w4a8_full", wbytes_i4, w4a8_full, ref_bf)
            emit(r, (bf_ms / r["ms"]) if r and r["ms"] else float("nan"))

            r, _ = run_path(
                "w4a8_op", wbytes_i4,
                lambda: ops.int4_gemm_w4a8(xq, xs.to(torch.float16), xz.to(torch.int32),
                                           qweight, wscale, wzp, GROUP, None, None),
                (xq.to(torch.float32) * xs.to(torch.float32)) @ w_fp16.to(torch.float32).t())
            emit(r, (bf_ms / r["ms"]) if r and r["ms"] else float("nan"))

            if have["int8_gemm_w8a16"]:
                r, _ = run_path(
                    "w8a16", wbytes_i8,
                    lambda: ops.int8_gemm_w8a16(x16, B_nt, wsc16, None),
                    ref_bf)
                emit(r, (bf_ms / r["ms"]) if r and r["ms"] else float("nan"))

            if have["int8_gemm_w8a8"]:
                def w8a8_full():
                    q, s, z = act_quant(x16)
                    return ops.int8_gemm_w8a8(q, s, None, B_nt, wsc16, None, None, torch.float16)
                r, _ = run_path("w8a8_full", wbytes_i8, w8a8_full, ref_bf)
                emit(r, (bf_ms / r["ms"]) if r and r["ms"] else float("nan"))

            if have["int8_gemm_w8a8_fusedq"]:
                r, _ = run_path(
                    "w8a8_fusedq", wbytes_i8,
                    lambda: ops.int8_gemm_w8a8_fusedq(x16, B_nt, wsc16, None, torch.float16),
                    ref_bf)
                emit(r, (bf_ms / r["ms"]) if r and r["ms"] else float("nan"))

        del wq, ws, qweight, wscale, w_fp16, w_bf16, B_s8, B_nt
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

    # ---- highlighted K1 gate: M=1 and M=8 on gate_up + down_proj ----
    print("\n=== K1 GATE (M=1,8 x gate_up,down_proj) ===", flush=True)
    print(f"{'shape':<10} {'M':>5} {'path':<12} {'ms':>9} {'GB/s':>8} {'TOPS':>8} "
          f"{'%roof':>7} {'xBF16':>7}", flush=True)
    bf_lookup = {(r["shape"], r["M"]): r["ms"] for r in rows if r["path"] == "bf16"}
    for r in rows:
        if r["shape"] not in ("gate_up", "down_proj") or r["M"] not in (1, 8):
            continue
        bf = bf_lookup.get((r["shape"], r["M"]), float("nan"))
        vs = (bf / r["ms"]) if r["ms"] == r["ms"] and bf == bf and r["ms"] else float("nan")
        print(f"{r['shape']:<10} {r['M']:>5} {r['path']:<12} "
              f"{r['ms']:9.4f} {r['gbps']:8.1f} {r['tops']:8.2f} "
              f"{r['pct_roof']:6.1f}% {vs:7.2f}", flush=True)

    if OUT_CSV:
        os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
        fields = ["shape", "M", "K", "N", "path", "ms", "gbps", "tops",
                  "pct_roof", "relerr", "src", "wbytes"]
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"\nCSV {OUT_CSV}  n={len(rows)}", flush=True)
    print("DONE_K1_MATRIX", flush=True)


if __name__ == "__main__":
    main()
