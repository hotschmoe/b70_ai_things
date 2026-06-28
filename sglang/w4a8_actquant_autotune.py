#!/usr/bin/env python3
# w4a8_actquant_autotune.py -- autotune the per-token SYMMETRIC int8 activation-quant Triton kernel
# (sglang/patches/w4a8_actquant_triton.py, gate B70_W4A8_TRITON_AQ) on the REAL Lorbus W4A8 linear K's.
#
# The shipped kernel is grid=(M,), TWO streaming passes over K (pass1 amax-reduce, pass2 quantize) ->
# it reads x from global memory TWICE. The act-quant is bandwidth-bound, so the main lever is a
# SINGLE-PASS full-row variant (load the whole row once, amax + quantize in-register -> read x ONCE).
# This sweep also covers BLOCK_K / num_warps / num_stages for the two-pass form.
#
# Shapes (real Lorbus int4 27B linears): K in {5120 (gate/up/qkv in), 17408 (down/intermediate)}.
# M in {1 (decode -- NOTE: decode uses int4_gemm_w4a16, NO act-quant, so M=1 here is informational),
#       512, 2048 (prefill -- the path this kernel actually serves)}.
#
# Gate (numerics vs the eager reference): q within <=1 LSB on >99% of elements AND scale bit-exact.
# Win bar: a config must beat the shipped (two-pass BLOCK_K=2048 num_warps=8) by >=5% at M=2048 to ship.
#
# Run (card 0, microbench, NO serve):
#   ./bin/gpu-run --card 0 docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
#     --ipc=host --shm-size 16g -e ZE_AFFINITY_MASK=0 \
#     -v /mnt/vm_8tb/b70/w4a8_kernel:/build/w4a8_kernel:ro \
#     -v /mnt/vm_8tb/b70/models:/models:ro \
#     -v /mnt/vm_8tb/github/b70_ai_things/sglang:/work \
#     sglang-xpu:woq bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
#       export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; \
#       python3 /work/w4a8_actquant_autotune.py'
import os, sys, time, ctypes
import torch

DEV = "xpu"
WARM = 30
ITERS = 80
SHIPPED = ("2pass", 2048, 8, 1)   # strategy, BLOCK_K, num_warps, num_stages  (the current kernel)
KS = [5120, 17408]
MS = [1, 512, 2048]

try:
    import triton
    import triton.language as tl
except Exception as e:  # noqa: BLE001
    print("TRITON IMPORT FAILED:", repr(e)); sys.exit(1)
print("torch", torch.__version__, "| triton", getattr(triton, "__version__", "?"))


# ---------------- kernels ----------------
@triton.jit
def _ptq_2pass(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    q_row = q_ptr + row * stride_qm
    amax = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    amax = tl.maximum(amax, 1e-5)
    inv = 127.0 / amax
    tl.store(s_ptr + row, (amax / 127.0).to(tl.float16))
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        v = x * inv
        r = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
        r = tl.minimum(tl.maximum(r, -127.0), 127.0)
        tl.store(q_row + offs, r.to(tl.int8), mask=mask)


@triton.jit
def _ptq_1pass(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK_K: tl.constexpr):
    # single pass: load the WHOLE row once (BLOCK_K >= K), amax + quantize in-register (x read ONCE)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-5)
    inv = 127.0 / amax
    tl.store(s_ptr + row, (amax / 127.0).to(tl.float16))
    v = x * inv
    r = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    tl.store(q_ptr + row * stride_qm + offs, r.to(tl.int8), mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def run_cfg(x, q, s, cfg):
    strat, BLOCK_K, nw, ns = cfg
    M, K = x.shape
    if strat == "1pass":
        _ptq_1pass[(M,)](x, q, s, K, x.stride(0), q.stride(0), BLOCK_K=BLOCK_K, num_warps=nw, num_stages=ns)
    else:
        _ptq_2pass[(M,)](x, q, s, K, x.stride(0), q.stride(0), BLOCK_K=BLOCK_K, num_warps=nw, num_stages=ns)


# ---------------- reference / numerics ----------------
def eager_ref(x):
    amax = x.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
    xs = (amax / 127.0).to(torch.float16)
    xq = (x / xs).round().clamp_(-127, 127).to(torch.int8).contiguous()
    return xq, xs.contiguous()


def sync():
    torch.xpu.synchronize()


def bench(fn):
    for _ in range(WARM):
        fn()
    sync(); t = time.time()
    for _ in range(ITERS):
        fn()
    sync(); return (time.time() - t) / ITERS * 1000.0


# ---------------- optional: int4_gemm_w4a8 output relerr on the real down_proj (K=17408) ----------------
def load_op():
    SO = "/build/w4a8_kernel/_xpu_C.abi3.so"
    if not os.path.exists(SO):
        return None
    try:
        ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
        if hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):
            return True
    except Exception as e:  # noqa: BLE001
        print("  (op load skipped:", repr(e)[:80], ")")
    return None


def op_relerr_check(best_cfg):
    """Confirm the best config keeps int4_gemm_w4a8 output ~= eager-act-quant output (real weight)."""
    if load_op() is None:
        print("\n[op-relerr] _xpu_C / int4_gemm_w4a8 unavailable -> skipped")
        return
    CKPT = "/models/Qwen3.6-27B-W4A8-sqgptq-prepacked/model.safetensors"
    if not os.path.exists(CKPT):
        print("\n[op-relerr] sqgptq ckpt absent -> skipped")
        return
    try:
        import safetensors.torch as stt
        PFX = "model.language_model.layers.20.mlp.down_proj"
        t = stt.load_file(CKPT)
        wq = t[f"{PFX}.weight"].to(DEV); ws = t[f"{PFX}.weight_scale"].to(DEV)
        N, K8 = wq.shape; K = K8 * 8; G = 128
        qweight = wq.t(); wscale = ws.t().contiguous(); wzp = torch.tensor([8], dtype=torch.int8, device=DEV)
        print(f"\n[op-relerr] real down_proj N={N} K={K}, best_cfg={best_cfg}")
        for M in (512, 2048):
            x = (torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.1).contiguous()
            eq, es = eager_ref(x); ez = torch.zeros((M, 1), dtype=torch.int32, device=DEV)
            q = torch.empty((M, K), dtype=torch.int8, device=DEV); s = torch.empty((M, 1), dtype=torch.float16, device=DEV)
            run_cfg(x, q, s, best_cfg); sync()
            ye = torch.ops._xpu_C.int4_gemm_w4a8(eq, es, ez, qweight, wscale, wzp, G, None, None)
            yt = torch.ops._xpu_C.int4_gemm_w4a8(q, s, ez, qweight, wscale, wzp, G, None, None)
            rel = ((ye.float() - yt.float()).norm() / ye.float().norm().clamp_min(1e-9)).item()
            print(f"  M={M:>4} int4_gemm_w4a8 out relerr(eager vs best) = {rel:.3e}  finite={torch.isfinite(yt).all().item()}")
    except Exception as e:  # noqa: BLE001
        print("[op-relerr] FAILED:", repr(e)[:160])


# ---------------- sweep ----------------
def candidate_cfgs(K):
    cfgs = []
    for BK in (512, 1024, 2048, 4096):
        for nw in (4, 8, 16, 32):
            for ns in (1, 2, 3):
                cfgs.append(("2pass", BK, nw, ns))
    bk1 = _next_pow2(K)
    for nw in (8, 16, 32):
        for ns in (1, 2):
            cfgs.append(("1pass", bk1, nw, ns))
    return cfgs


def numerics_ok(x, cfg):
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.int8, device=DEV)
    s = torch.empty((M, 1), dtype=torch.float16, device=DEV)
    run_cfg(x, q, s, cfg); sync()
    rq, rs = eager_ref(x)
    qdiff = (q.to(torch.int16) - rq.to(torch.int16)).abs()
    maxd = qdiff.max().item()
    mis = (qdiff > 0).float().mean().item()
    sdiff = (s.float() - rs.float()).abs().max().item()
    ok = (maxd <= 1) and (mis < 0.01) and (sdiff == 0.0)
    return ok, maxd, mis, sdiff


def main():
    torch.manual_seed(0)
    # validate the SHIPPED config numerics once per (K,M)
    print("\n==== numerics: SHIPPED config (2pass BK=2048 nw=8) vs eager ====")
    for K in KS:
        for M in MS:
            x = (torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.1).contiguous()
            ok, maxd, mis, sdiff = numerics_ok(x, SHIPPED)
            print(f"  K={K:>5} M={M:>4}: ok={ok} max|dq|={maxd} mism={mis*100:.3f}% s_maxdiff={sdiff:.1e}")

    results = {}  # (K,M) -> list of (ms, cfg, ok)
    for K in KS:
        cfgs = candidate_cfgs(K)
        for M in MS:
            x = (torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.1).contiguous()
            q = torch.empty((M, K), dtype=torch.int8, device=DEV)
            s = torch.empty((M, 1), dtype=torch.float16, device=DEV)
            rows = []
            for cfg in cfgs:
                try:
                    ok, maxd, mis, sdiff = numerics_ok(x, cfg)
                    ms = bench(lambda c=cfg: run_cfg(x, q, s, c))
                    rows.append((ms, cfg, ok, maxd, mis))
                except Exception as e:  # noqa: BLE001
                    rows.append((float("inf"), cfg, False, -1, -1))
            results[(K, M)] = rows

    # report
    overall_best = {}  # (K,M) -> best valid cfg
    for K in KS:
        for M in MS:
            rows = results[(K, M)]
            ship = next((r for r in rows if r[1] == SHIPPED), None)
            ship_ms = ship[0] if ship else float("nan")
            valid = sorted([r for r in rows if r[2] and r[0] != float("inf")], key=lambda r: r[0])
            print(f"\n==== K={K} M={M}  (SHIPPED 2pass/2048/8 = {ship_ms:.4f} ms) ====")
            for ms, cfg, ok, maxd, mis in valid[:6]:
                spd = ship_ms / ms if ms > 0 else 0
                print(f"  {ms:.4f} ms  {spd:5.2f}x  {cfg}  (max|dq|={maxd} mism={mis*100:.3f}%)")
            if valid:
                overall_best[(K, M)] = (valid[0][0], valid[0][1], ship_ms)

    # summary: best @M=2048 per K (the prefill path), with speedup vs shipped
    print("\n==== SUMMARY: best valid config @ M=2048 (the served prefill path) ====")
    best_2048 = []
    for K in KS:
        if (K, 2048) in overall_best:
            bms, bcfg, sms = overall_best[(K, 2048)]
            spd = sms / bms if bms > 0 else 0
            print(f"  K={K:>5}: best {bms:.4f} ms {bcfg}  vs shipped {sms:.4f} ms = {spd:.2f}x ({(spd-1)*100:+.1f}%)")
            best_2048.append((K, bcfg, spd))

    # pick a SINGLE config to recommend (must be valid + fastest-on-average across BOTH K at M=2048)
    print("\n==== unified recommendation (one config for the shim, scored at M=2048) ====")
    cand = {}
    for K in KS:
        for ms, cfg, ok, maxd, mis in results[(K, 2048)]:
            if ok and ms != float("inf"):
                cand.setdefault(cfg, []).append((K, ms))
    # only configs valid for BOTH K
    full = {c: v for c, v in cand.items() if len(v) == len(KS)}
    ship_ref = {K: next((r[0] for r in results[(K, 2048)] if r[1] == SHIPPED), float("nan")) for K in KS}
    scored = []
    for c, v in full.items():
        # geomean speedup vs shipped across K
        import math
        spd = math.exp(sum(math.log(ship_ref[K] / ms) for K, ms in v) / len(v))
        scored.append((spd, c, v))
    scored.sort(reverse=True)
    for spd, c, v in scored[:8]:
        det = " ".join(f"K{K}={ms:.4f}" for K, ms in v)
        print(f"  geomean {spd:.3f}x  {c}   [{det}]")
    if scored:
        best_unified = scored[0][1]
        print(f"\nRECOMMEND: {best_unified}  (geomean {scored[0][0]:.3f}x vs shipped @M=2048)")
        op_relerr_check(best_unified)
        print(f"\nGATE: ship only if geomean >= 1.05 AND op-relerr stays < 1e-2. Shipped = {SHIPPED}.")


if __name__ == "__main__":
    main()
