"""C10 dequant-then-oneDNN prefill bench. Run via scripts/c10_run.sh under gpu-run --card 0.

Correctness:
- prefill_dequant bf16 bits match CPU round-to-nearest-even (and dequant_bf16).
- dense_dq matches oneDNN bf16 torch.mm of those weights within 1 bf16 ulp.
- The same output against an fp32 matmul of the bf16 weights must sit within
  8 bf16 ulps. The old atol=rtol=5e-2 check against the fp32 dequant is logged
  only: one bf16 ulp of a Q4 output near 2e4 is 128, so abs error ~70 fails
  5e-2 on the smaller lanes while the bf16 matmul itself is right.
Bench prints event-min and wall-min microseconds. event is GPU time between
XPU events. wall is host time until the result is synced, which is the TTFT
contribution when the CPU launch sits on the critical path.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gguf import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

import torch  # noqa: E402

from b70_gguf_xpu.prefill import load  # noqa: E402
from b70_gguf_xpu.prefill_dq import LAST, dense_dq, grouped_dq, moe_dq  # noqa: E402
from b70_gguf_xpu.types import BLOCK, NAMES, row_bytes  # noqa: E402
from gpu_check import f32_to_bf16_bits  # noqa: E402
from prefill_check import close_enough, cpu_dequant, load_cpu, report  # noqa: E402

Q4, Q5, Q6 = 12, 13, 14
IQ4XS, IQ3S, IQ3, Q2 = 23, 21, 18, 42
ROOF = 608.0
ALL_TYPES = (12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 10, 11, 42)


def fast_pack(typ: int, rows: int, k: int, seed: int) -> np.ndarray:
    rb = row_bytes(typ, k)
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(rows, rb), dtype=np.uint8)
    _be, bb = BLOCK[typ]
    view = raw.reshape(rows, -1, bb)
    # fp16 scale 1.0 in the usual header slot. Extra slots keep dmin / Q6 scale finite.
    view[:, :, 0] = 0x00
    view[:, :, 1] = 0x3C
    if typ in (Q4, Q5) and bb >= 4:
        view[:, :, 2] = 0
        view[:, :, 3] = 0
    if typ == Q6 and bb >= 210:
        view[:, :, 208] = 0x00
        view[:, :, 209] = 0x3C
    if typ == 10 and bb >= 84:
        view[:, :, 80] = 0x00
        view[:, :, 81] = 0x3C
        view[:, :, 82] = 0
        view[:, :, 83] = 0
    if typ == 11 and bb >= 110:
        view[:, :, 108] = 0x00
        view[:, :, 109] = 0x3C
    return np.ascontiguousarray(raw)


def ref_f32(lib, tid: int, raw: np.ndarray, k: int) -> np.ndarray:
    if tid == Q2:
        return cpu_dequant(lib, tid, raw, k)
    return np.asarray(dequantize(raw, GGMLQuantizationType(tid)), dtype=np.float32)


def elapsed_both(fn, warmup: int = 1, iters: int = 3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    events, walls = [], []
    for _ in range(iters):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        walls.append((time.perf_counter() - t0) * 1e6)
        events.append(float(start.elapsed_time(end)) * 1000.0)
    return min(events), min(walls)


def gbps(nbytes: float, us: float) -> float:
    if us <= 0:
        return 0.0
    return (nbytes / 1e9) / (us * 1e-6)


def tflops(m, n, k, us: float) -> float:
    if us <= 0:
        return 0.0
    return (2.0 * m * n * k) / us / 1e6


def check_dequant(lib, failures: int) -> int:
    load()
    shapes = [(tid, 8, 256) for tid in ALL_TYPES]
    shapes += [(Q2, 4, 640), (IQ3, 4, 2560), (Q4, 4, 2560)]
    for tid, rows, k in shapes:
        raw = fast_pack(tid, rows, k, 1000 + tid + k)
        ref = ref_f32(lib, tid, raw, k)
        q = torch.from_numpy(raw).to(device="xpu")
        got = torch.ops.b70_gguf.prefill_dequant(
            q, tid, rows, k, torch.empty((rows, k), dtype=torch.bfloat16, device="xpu")
        )
        torch.xpu.synchronize()
        bits = got.view(torch.int16).cpu().numpy().view(np.uint16)
        exp = f32_to_bf16_bits(ref)
        bad = int(np.count_nonzero(bits.reshape(-1) != exp.reshape(-1)))
        failures = report(
            failures,
            "OK" if bad == 0 else "FAIL",
            f"dequant {NAMES[tid]} rows={rows} k={k} bf16_mismatches={bad}",
        )
        old = torch.ops.b70_gguf.dequant_bf16(q, tid, rows, k)
        torch.xpu.synchronize()
        ob = old.view(torch.int16).cpu().numpy().view(np.uint16)
        same = int(np.count_nonzero(ob.reshape(-1) != bits.reshape(-1)))
        failures = report(
            failures,
            "OK" if same == 0 else "FAIL",
            f"dequant_match_old {NAMES[tid]} rows={rows} k={k} mismatches={same}",
        )
        del q, got, old
    return failures


def bf16_ulp_ok(got, ref, nulp: float = 8.0):
    """True when got matches a bf16-rounded matmul within nulp bf16 ulps.

    oneDNN's bf16 torch.mm stores bf16. A 5e-2 check against the fp32
    dequant fails once Q4 outputs reach 1e3-1e4, where one bf16 ulp is
    64-128. A wrong layout or a swapped expert is thousands of ulps.
    Non-finite lanes fail. NaN comparisons would otherwise count as a pass.
    """
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    if g.size == 0:
        return True, 0.0, 0
    if not (np.isfinite(g).all() and np.isfinite(r).all()):
        return False, float("inf"), int(g.size)
    d = np.abs(g - r)
    mag = np.maximum(np.maximum(np.abs(g), np.abs(r)), 1e-6)
    ulp = np.exp2(np.floor(np.log2(mag)) - 7.0)
    bad = int(np.count_nonzero(d > (nulp * ulp)))
    mx = float(d.max()) if d.size else 0.0
    return bad == 0, mx, bad


def check_dense_num(lib, failures: int) -> int:
    raw = fast_pack(Q4, 128, 256, 7)
    refw = ref_f32(lib, Q4, raw, 256)
    q = torch.from_numpy(raw).to(device="xpu")
    x = torch.randn(48, 256, device="xpu", dtype=torch.float16)
    y = dense_dq(x, q, Q4, 128, 256)
    torch.xpu.synchronize()
    xf = x.to(torch.bfloat16).float().cpu().numpy()
    ref = xf @ refw.T
    wbf_t = torch.ops.b70_gguf.prefill_dequant(
        q, Q4, 128, 256, torch.empty((128, 256), dtype=torch.bfloat16, device="xpu")
    )
    mm = torch.mm(x.to(torch.bfloat16), wbf_t.t())
    torch.xpu.synchronize()
    wbf = wbf_t.float().cpu().numpy()
    refb = xf @ wbf.T
    yn = y.cpu().numpy()
    ok_mm, mx_mm, bad_mm = bf16_ulp_ok(yn, mm.float().cpu().numpy(), nulp=1.0)
    ok, mx, bad = bf16_ulp_ok(yn, refb)
    _, mx32, mr32 = close_enough(yn, ref, 5e-2, 5e-2)
    # Pass bit is the bf16 matmul, not the 5e-2 fp32-weight check.
    failures = report(
        failures,
        "OK" if (ok_mm and ok) else "FAIL",
        f"dense_dq Q4_K M=48 N=128 K=256 mm_ulps_bad={bad_mm} mm_max_abs={mx_mm:.3e} "
        f"bf16w_ulps_bad={bad} max_abs_vs_bf16w={mx:.3e} "
        f"vs_f32_max_abs={mx32:.3e} vs_f32_max_rel={mr32:.3e}",
    )
    # Wrapper path, forced on. Same bf16-weight reference.
    os.environ["B70_PREFILL_DQ_M"] = "0"
    from b70_gguf_xpu.prefill import prefill_dense

    y2 = prefill_dense(x, q, Q4, 128, 256, 0)
    torch.xpu.synchronize()
    ok_mm2, mx_mm2, bad_mm2 = bf16_ulp_ok(y2.float().cpu().numpy(), mm.float().cpu().numpy(), nulp=1.0)
    ok, mx, bad = bf16_ulp_ok(y2.float().cpu().numpy(), refb)
    failures = report(
        failures,
        "OK" if (ok_mm2 and ok) else "FAIL",
        f"dispatch dense M=48 mm_ulps_bad={bad_mm2} mm_max_abs={mx_mm2:.3e} "
        f"bf16w_ulps_bad={bad} max_abs_vs_bf16w={mx:.3e}",
    )
    os.environ["B70_PREFILL_DQ_M"] = "off"
    return failures


def check_group_and_moe(lib, failures: int) -> int:
    # Small layer: E=4, T=5, topk=2, H=256, F=64. One id invalid.
    h, f, e, topk, t = 256, 64, 4, 2, 5
    g = fast_pack(IQ3, e * f, h, 11)
    u = fast_pack(IQ3, e * f, h, 12)
    d = fast_pack(Q2, e * h, f, 13)
    gates = [ref_f32(lib, IQ3, np.ascontiguousarray(g.reshape(e * f, -1)[i * f : (i + 1) * f]), h) for i in range(e)]
    ups = [ref_f32(lib, IQ3, np.ascontiguousarray(u.reshape(e * f, -1)[i * f : (i + 1) * f]), h) for i in range(e)]
    dns = [ref_f32(lib, Q2, np.ascontiguousarray(d.reshape(e * h, -1)[i * h : (i + 1) * h]), f) for i in range(e)]
    rng = np.random.default_rng(3)
    ids_np = rng.integers(0, e, size=(t, topk), dtype=np.int32)
    ids_np[0, 1] = -1
    x_np = (rng.standard_normal((t, h)).astype(np.float32) * 0.02).astype(np.float16).astype(np.float32)
    rw = rng.random((t, topk)).astype(np.float32)
    ref = np.zeros((t, h), np.float32)
    for ti in range(t):
        for kk in range(topk):
            eid = int(ids_np[ti, kk])
            if eid < 0:
                continue
            xv = x_np[ti]
            gv = xv @ gates[eid].T
            uv = xv @ ups[eid].T
            hv = (gv / (1.0 + np.exp(-gv))) * uv
            ref[ti] += rw[ti, kk] * (hv @ dns[eid].T)
    x = torch.from_numpy(x_np.astype(np.float16)).to(device="xpu")
    ids = torch.from_numpy(ids_np).to(device="xpu")
    router = torch.from_numpy(rw).to(device="xpu")
    gq = torch.from_numpy(g).to(device="xpu")
    uq = torch.from_numpy(u).to(device="xpu")
    dq = torch.from_numpy(d).to(device="xpu")

    def gpu_rows(qt, tid, rows, k):
        got = torch.ops.b70_gguf.prefill_dequant(
            qt, tid, rows, k, torch.empty((rows, k), dtype=torch.bfloat16, device="xpu")
        )
        torch.xpu.synchronize()
        return got.float().cpu().numpy()

    gb = gpu_rows(gq, IQ3, e * f, h).reshape(e, f, h)
    ub = gpu_rows(uq, IQ3, e * f, h).reshape(e, f, h)
    db = gpu_rows(dq, Q2, e * h, f).reshape(e, h, f)
    # Match moe_dq: bf16 GEMMs, silu in fp32, hidden rounded back to bf16.
    refb = np.zeros((t, h), np.float32)
    for ti in range(t):
        acc = np.zeros(h, np.float32)
        for kk in range(topk):
            eid = int(ids_np[ti, kk])
            if eid < 0:
                continue
            xv = torch.from_numpy(x_np[ti].astype(np.float32)).to(torch.bfloat16).float().numpy()
            gv = torch.from_numpy(xv @ gb[eid].T).to(torch.bfloat16)
            uv = torch.from_numpy(xv @ ub[eid].T).to(torch.bfloat16)
            # Same op as moe_dq: bf16 silu, then the product is the down input.
            hv = (torch.nn.functional.silu(gv) * uv).to(torch.bfloat16).float().numpy()
            down = torch.from_numpy(hv @ db[eid].T).to(torch.bfloat16).float().numpy()
            acc += rw[ti, kk] * down
        refb[ti] = acc
    layer_ys = {}
    for how in ("bmm", "loop"):
        y = moe_dq(x, ids, gq, uq, dq, router, IQ3, Q2, h, f, how=how)
        torch.xpu.synchronize()
        yn = y.cpu().numpy()
        layer_ys[how] = yn
        ok, mx, bad = bf16_ulp_ok(yn, refb, nulp=8.0)
        _, mx32, mr32 = close_enough(yn, ref, 5e-2, 5e-2)
        failures = report(
            failures,
            "OK" if ok else "FAIL",
            f"moe_dq {how} T={t} E={e} H={h} F={f} bf16_ulps_bad={bad} max_abs_vs_bf16={mx:.3e} "
            f"vs_f32_max_abs={mx32:.3e} vs_f32_max_rel={mr32:.3e} mv={LAST.get('mv')} mmax={LAST.get('mmax')}",
        )
    ok, mx, bad = bf16_ulp_ok(layer_ys["bmm"], layer_ys["loop"], nulp=8.0)
    failures = report(
        failures,
        "OK" if ok else "FAIL",
        f"moe_dq bmm_vs_loop bf16_ulps_bad={bad} max_abs={mx:.3e}",
    )
    # grouped, one column of ids, including an invalid.
    flat = ids_np.reshape(-1)
    xg = torch.randn(flat.shape[0], h, device="xpu", dtype=torch.float16) * 0.02
    yg = grouped_dq(xg, gq, torch.from_numpy(flat).to(device="xpu"), IQ3, f, h, how="bmm")
    torch.xpu.synchronize()
    xf = xg.to(torch.bfloat16).float().cpu().numpy()
    refg = np.zeros((flat.shape[0], f), np.float32)
    for i, eid in enumerate(flat.tolist()):
        if int(eid) >= 0:
            refg[i] = xf[i] @ gb[int(eid)].T
    ok, mx, bad = bf16_ulp_ok(yg.cpu().numpy(), refg)
    failures = report(
        failures,
        "OK" if ok else "FAIL",
        f"grouped_dq bmm bf16_ulps_bad={bad} max_abs_vs_bf16w={mx:.3e}",
    )
    return failures


def check_gemv_link(failures: int) -> int:
    # Decode symbol still linked. Not a speed number.
    q = torch.from_numpy(fast_pack(Q4, 64, 256, 1)).to(device="xpu")
    x = torch.randn(1, 256, device="xpu", dtype=torch.bfloat16)
    y = torch.ops.b70_gguf.gemv(x, q, Q4, 64, 256)
    torch.xpu.synchronize()
    ok = bool(torch.isfinite(y).all())
    failures = report(failures, "OK" if ok else "FAIL", f"gemv_link finite={ok}")
    return failures


def time_dequant(q, tid, rows, k, iters=3):
    out = torch.empty((rows, k), dtype=torch.bfloat16, device="xpu")

    def run():
        torch.ops.b70_gguf.prefill_dequant(q, tid, rows, k, out)

    ev, wall = elapsed_both(run, warmup=1, iters=iters)
    packed = rows * row_bytes(tid, k)
    traffic = packed + rows * k * 2
    print(
        f"BENCH dequant {NAMES[tid]} rows={rows} K={k} event_us={ev:.1f} wall_us={wall:.1f} "
        f"GBps={gbps(traffic, ev):.1f} pct_roof={100.0 * gbps(traffic, ev) / ROOF:.2f} packed={packed}",
        flush=True,
    )
    return ev, wall


def time_dense(tag, q, tid, m, n, k, xb, iters=3):
    fl = 2.0 * m * n * k

    def fused():
        torch.ops.b70_gguf.prefill_dense(xb, q, tid, n, k, 0)

    def dq():
        dense_dq(xb, q, tid, n, k)

    w = torch.ops.b70_gguf.prefill_dequant(q, tid, n, k, torch.empty((n, k), dtype=torch.bfloat16, device="xpu"))
    torch.xpu.synchronize()

    def mm():
        torch.mm(xb if xb.dtype == torch.bfloat16 else xb.to(torch.bfloat16), w.t())

    ev_f, wall_f = elapsed_both(fused, 1, iters)
    ev_d, wall_d = elapsed_both(dq, 1, iters)
    ev_m, wall_m = elapsed_both(mm, 1, iters)
    print(
        f"BENCH dense {tag} {NAMES[tid]} M={m} N={n} K={k} "
        f"fused_event={ev_f:.1f} fused_wall={wall_f:.1f} fused_tf={tflops(m, n, k, ev_f):.2f} "
        f"dq_event={ev_d:.1f} dq_wall={wall_d:.1f} dq_tf={tflops(m, n, k, ev_d):.2f} "
        f"mm_event={ev_m:.1f} mm_wall={wall_m:.1f} mm_tf={tflops(m, n, k, ev_m):.2f}",
        flush=True,
    )
    return ev_f, ev_d, ev_m


def release_scratch():
    """Drop reusable dequant buffers between bench stages. The pool keys on shape, so a head and a full expert pack would otherwise stay live together."""
    from b70_gguf_xpu import prefill_dq

    prefill_dq._SCRATCH.clear()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()


def probe_grouped_mm():
    import torch.nn.functional as F

    try:
        a = torch.randn(16, 32, device="xpu", dtype=torch.bfloat16)
        b = torch.randn(2, 32, 16, device="xpu", dtype=torch.bfloat16)
        offs = torch.tensor([8, 16], device="xpu", dtype=torch.int32)
        y = F.grouped_mm(a, b, offs=offs)
        torch.xpu.synchronize()
        print(f"PROBE grouped_mm ok shape={tuple(y.shape)}", flush=True)
    except Exception as ex:
        print(f"PROBE grouped_mm unsupported {type(ex).__name__}: {ex}", flush=True)


def main() -> int:
    assert torch.xpu.is_available()
    assert torch.xpu.device_count() == 1, torch.xpu.device_count()
    print(f"DEVICE {torch.xpu.get_device_name(0)} so={os.environ.get('B70_GGUF_SO', '')}", flush=True)
    (torch.randn(64, 64, device="xpu", dtype=torch.bfloat16) @ torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)).sum().item()
    torch.xpu.synchronize()
    print("PROBE matmul ok", flush=True)
    lib = load_cpu()
    failures = 0
    if "--bench-only" not in sys.argv:
        failures = check_dequant(lib, failures)
        failures = check_dense_num(lib, failures)
        failures = check_group_and_moe(lib, failures)
        failures = check_gemv_link(failures)
        if failures:
            print(f"VERDICT failures={failures} skip_bench=1", flush=True)
            return 1
    if "--check-only" in sys.argv:
        print(f"VERDICT failures={failures}", flush=True)
        return 0

    # Dense crossover. Q4_K is the C8 shape. IQ4_XS is the real attn_q type.
    print("PACK dense Q4", flush=True)
    n, k = 12288, 2560
    q4 = torch.from_numpy(fast_pack(Q4, n, k, 7)).to(device="xpu")
    for m in (512, 1024, 2048, 4096, 8192, 12288):
        xb = torch.randn(m, k, device="xpu", dtype=torch.float16)
        iters = 2 if m >= 8192 else 3
        time_dense("crossover", q4, Q4, m, n, k, xb, iters=iters)
        del xb
        torch.xpu.empty_cache()
    # fp16 mm once, at M=4096, weight dequanted to fp16.
    m = 4096
    xb = torch.randn(m, k, device="xpu", dtype=torch.float16)
    w16 = torch.empty((n, k), dtype=torch.float16, device="xpu")

    def dq16():
        torch.ops.b70_gguf.prefill_dequant(q4, Q4, n, k, w16)
        return torch.mm(xb, w16.t())

    ev, wall = elapsed_both(dq16, 1, 3)
    print(
        f"BENCH dense fp16_out Q4_K M={m} N={n} K={k} event_us={ev:.1f} wall_us={wall:.1f} tf={tflops(m, n, k, ev):.2f}",
        flush=True,
    )
    del q4, xb, w16
    torch.xpu.empty_cache()

    print("PACK dense IQ4_XS", flush=True)
    qxs = torch.from_numpy(fast_pack(IQ4XS, n, k, 8)).to(device="xpu")
    for m in (512, 4096, 12288):
        xb = torch.randn(m, k, device="xpu", dtype=torch.float16)
        time_dense("attn_q", qxs, IQ4XS, m, n, k, xb, iters=2 if m >= 12288 else 3)
        del xb
    del qxs
    torch.xpu.empty_cache()

    # K=640 down shape as one dense GEMM (single matrix, not grouped).
    print("PACK down-like Q2_0", flush=True)
    nd, kd = 2560, 640
    qd = torch.from_numpy(fast_pack(Q2, nd, kd, 9)).to(device="xpu")
    time_dequant(qd, Q2, nd, kd)
    for m in (512, 4096, 12288):
        xb = torch.randn(m, kd, device="xpu", dtype=torch.float16)
        time_dense("k640", qd, Q2, m, nd, kd, xb, iters=2 if m >= 12288 else 3)
        del xb
    del qd
    torch.xpu.empty_cache()

    # Head. M=12288 output is ~6 GiB bf16; measure M=512 and M=4096.
    release_scratch()
    print("PACK head Q5_K", flush=True)
    nh, kh = 248320, 2560
    qh = torch.from_numpy(fast_pack(Q5, nh, kh, 10)).to(device="xpu")
    time_dequant(qh, Q5, nh, kh, iters=2)
    for m in (512, 4096):
        xb = torch.randn(m, kh, device="xpu", dtype=torch.float16)
        time_dense("head", qh, Q5, m, nh, kh, xb, iters=2)
        del xb
        torch.xpu.empty_cache()
    del qh
    torch.xpu.empty_cache()

    # A few more dequant shapes used by the budget, GEMM scaled from the curves above.
    print("PACK dequant shapes", flush=True)
    extra = (
        (IQ4XS, 10240, 2560, "gdn_qkv"),
        (IQ3S, 6144, 2560, "gdn_gate"),
        (IQ4XS, 2560, 6144, "gdn_out"),
        (Q6, 2560, 6144, "attn_out"),
        (IQ3, 512 * 640, 2560, "expert_gate_all"),
        (Q2, 512 * 2560, 640, "expert_down_all"),
    )
    for tid, rows, kk, name in extra:
        print(f"PACK {name}", flush=True)
        raw = fast_pack(tid, rows, kk, 20 + tid + rows)
        qt = torch.from_numpy(raw).to(device="xpu")
        del raw
        time_dequant(qt, tid, rows, kk, iters=2)
        del qt
        torch.xpu.empty_cache()

    probe_grouped_mm()

    release_scratch()
    print("PACK expert layer", flush=True)
    h, ffn, e, topk = 2560, 640, 512, 10
    g = fast_pack(IQ3, e * ffn, h, 7)
    u = fast_pack(IQ3, e * ffn, h, 8)
    d = fast_pack(Q2, e * h, ffn, 9)
    gq = torch.from_numpy(g).to(device="xpu")
    uq = torch.from_numpy(u).to(device="xpu")
    dq = torch.from_numpy(d).to(device="xpu")
    del g, u, d
    nbytes = e * ffn * row_bytes(IQ3, h) * 2 + e * h * row_bytes(Q2, ffn)
    for t in (512, 4096, 12288):
        ids = torch.randint(0, e, (t, topk), device="xpu", dtype=torch.int32)
        x = torch.randn(t, h, device="xpu", dtype=torch.float16)
        router = torch.rand(t, topk, device="xpu", dtype=torch.float32)
        iters = 1 if t >= 12288 else 2

        def fused(x=x, ids=ids, router=router):
            torch.ops.b70_gguf.prefill_moe_layer(x, ids, gq, uq, dq, router, IQ3, Q2, h, ffn, 0)

        ev, wall = elapsed_both(fused, 1, iters)
        os.environ["B70_PREFILL_SPLIT"] = "1"
        torch.ops.b70_gguf.prefill_moe_layer(x, ids, gq, uq, dq, router, IQ3, Q2, h, ffn, 0)
        torch.xpu.synchronize()
        sp = torch.ops.b70_gguf.prefill_split_us().tolist()
        os.environ["B70_PREFILL_SPLIT"] = "0"
        print(
            f"BENCH layer fused T={t} event_us={ev:.1f} wall_us={wall:.1f} "
            f"expert48_tok/s={t / (ev * 1e-6) / 48:.1f} "
            f"split_gu={sp[2]:.1f} split_down={sp[3]:.1f} split_total={sp[6]:.1f}",
            flush=True,
        )
        for how in ("bmm", "loop", "hybrid"):
            if how == "loop" and t >= 12288:
                # Host launch of 512*3 mm is measured, but one iter is enough.
                pass

            def run(how=how, x=x, ids=ids, router=router):
                moe_dq(x, ids, gq, uq, dq, router, IQ3, Q2, h, ffn, how=how)

            ev, wall = elapsed_both(run, 1, iters)
            print(
                f"BENCH layer {how} T={t} event_us={ev:.1f} wall_us={wall:.1f} "
                f"expert48_event_tok/s={t / (ev * 1e-6) / 48:.1f} "
                f"expert48_wall_tok/s={t / (wall * 1e-6) / 48:.1f} "
                f"mv={LAST.get('mv')} mmax={LAST.get('mmax')} unique_bytes={nbytes}",
                flush=True,
            )
        del x, ids, router
        torch.xpu.empty_cache()

    print(f"VERDICT failures={failures}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
