"""C11 expert prefill bench. Card 0, via scripts/c11_run.sh.

Correctness: wide SGN matches SGN=4 (same fp16 DPAS order) and a bf16-weight
reference on a small layer. Speed: full expert layer at T=512/4096/12288 for
SGN=4/8/16/40, then oneDNN grouped/bmm ceilings and dequant GB/s.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b70_gguf_xpu.prefill import load  # noqa: E402
from b70_gguf_xpu.types import row_bytes  # noqa: E402

IQ3, Q2 = 18, 42
H, F, E, TOPK = 2560, 640, 512, 10
# C10 winner event ms and the GEMM-only model sums (reports/1C10_prefill_dq.md).
C10_LAYER_MS = {512: 10.82, 4096: 26.76, 12288: 66.97}
C10_EXPERT_SUM = {512: 519.38, 4096: 1284.53, 12288: 3214.53}
C10_GEMM_SUM = {512: 604.71, 4096: 1641.90, 12288: 4255.71}


def fast_pack(typ: int, rows: int, k: int, seed: int) -> np.ndarray:
    rb = row_bytes(typ, k)
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(rows, rb), dtype=np.uint8)
    bb = {18: 98, 42: 18, 12: 144}[typ]
    be = {18: 256, 42: 64, 12: 256}[typ]
    view = raw.reshape(rows, -1, bb)
    view[:, :, 0] = 0x00
    view[:, :, 1] = 0x3C
    if typ == 12 and bb >= 4:
        view[:, :, 2] = 0
        view[:, :, 3] = 0
    return np.ascontiguousarray(raw.reshape(rows, (k // be) * bb))


def elapsed_both(fn, warmup: int = 1, iters: int = 2):
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


def tflops_layer(t: int, us: float) -> float:
    # gate + up + down: 3 * 2 * T * topk * H * F
    flops = 6.0 * t * TOPK * H * F
    return flops / us / 1e6 if us > 0 else 0.0


def report(failures: int, ok: bool, msg: str) -> int:
    print(("OK " if ok else "FAIL ") + msg, flush=True)
    return failures if ok else failures + 1


def layer_call(x, ids, gq, uq, dq, router):
    return torch.ops.b70_gguf.prefill_moe_layer(x, ids, gq, uq, dq, router, IQ3, Q2, H, F, 0)


def check_small(failures: int) -> tuple[int, bool]:
    """Return (failures, proceed). SGN=8 and SGN=16 must match SGN=4.

    SGN=40 can miss the device work-group limit. That is recorded and the
    speed sweep still runs for the widths that matched.
    """
    e, t, topk = 4, 8, 2
    gq = torch.from_numpy(fast_pack(IQ3, e * F, H, 3)).to("xpu")
    uq = torch.from_numpy(fast_pack(IQ3, e * F, H, 4)).to("xpu")
    dq = torch.from_numpy(fast_pack(Q2, e * H, F, 5)).to("xpu")
    ids = torch.randint(0, e, (t, topk), device="xpu", dtype=torch.int32)
    # Scale-1 IQ3 packs peak near 480. fp16 swiglu then 0*inf in the down
    # GEMM, which is NaN on _C.c10.so too (c11-nan.log). 1e-3 keeps the
    # product in fp16. The schedule does not depend on the values.
    x = (torch.randn(t, H, device="xpu", dtype=torch.float16) * 0.001).contiguous()
    router = torch.rand(t, topk, device="xpu", dtype=torch.float32)
    os.environ["B70_PREFILL_SPLIT"] = "0"
    outs = {}
    cols = {}
    for sgn in (4, 8, 16, 40):
        os.environ["B70_PREFILL_SGN"] = str(sgn)
        try:
            y = layer_call(x, ids, gq, uq, dq, router)
            torch.xpu.synchronize()
            outs[sgn] = y.float().cpu()
            os.environ["B70_PREFILL_SPLIT"] = "1"
            layer_call(x, ids, gq, uq, dq, router)
            torch.xpu.synchronize()
            sp = torch.ops.b70_gguf.prefill_split_us().tolist()
            os.environ["B70_PREFILL_SPLIT"] = "0"
            cols[sgn] = int(sp[7]) // 1000
            finite = bool(torch.isfinite(outs[sgn]).all())
            mx = float(outs[sgn].abs().max()) if finite else float("nan")
            print(f"CHECK sgn={sgn} cols={cols[sgn]} finite={finite} max_abs={mx:.4e}", flush=True)
        except Exception as ex:
            os.environ["B70_PREFILL_SPLIT"] = "0"
            outs[sgn] = None
            cols[sgn] = -1
            print(f"FAIL check sgn={sgn} {type(ex).__name__}: {ex}", flush=True)
            failures += 1
    base = outs.get(4)
    hard_ok = base is not None and bool(torch.isfinite(base).all())
    for sgn in (8, 16, 40):
        got = outs.get(sgn)
        if got is None or base is None:
            failures = report(failures, False, f"sgn={sgn} vs sgn=4 missing output cols={cols.get(sgn)}")
            if sgn != 40:
                hard_ok = False
            continue
        d = (got - base).abs()
        mx = float(d.max()) if d.numel() else 0.0
        want = sgn * 16
        ok = bool(torch.isfinite(got).all()) and mx <= 1e-3 and cols[sgn] == want
        failures = report(failures, ok, f"sgn={sgn} vs sgn=4 max_abs={mx:.3e} cols={cols[sgn]} want={want}")
        if sgn != 40 and not ok:
            hard_ok = False
    # bf16-weight reference, one token-slot at a time. fp16 DPAS vs bf16 mm.
    wg = torch.ops.b70_gguf.prefill_dequant(
        gq, IQ3, e * F, H, torch.empty((e * F, H), dtype=torch.bfloat16, device="xpu")
    ).view(e, F, H)
    wu = torch.ops.b70_gguf.prefill_dequant(
        uq, IQ3, e * F, H, torch.empty((e * F, H), dtype=torch.bfloat16, device="xpu")
    ).view(e, F, H)
    wd = torch.ops.b70_gguf.prefill_dequant(
        dq, Q2, e * H, F, torch.empty((e * H, F), dtype=torch.bfloat16, device="xpu")
    ).view(e, H, F)
    xb = x.to(torch.bfloat16)
    ref = torch.zeros(t, H, device="xpu", dtype=torch.float32)
    idc = ids.cpu()
    rc = router.cpu()
    for ti in range(t):
        acc = torch.zeros(H, device="xpu", dtype=torch.float32)
        for k in range(topk):
            ee = int(idc[ti, k])
            g = torch.mv(wg[ee], xb[ti])
            u = torch.mv(wu[ee], xb[ti])
            hid = torch.nn.functional.silu(g.float()) * u.float()
            acc = acc + torch.mv(wd[ee], hid.to(torch.bfloat16)).float() * float(rc[ti, k])
        ref[ti] = acc
    torch.xpu.synchronize()
    ref_sgn = 40 if outs.get(40) is not None else (16 if outs.get(16) is not None else 4)
    got_t = outs.get(ref_sgn)
    if got_t is None:
        return failures, False
    got = got_t.numpy()
    r = ref.cpu().numpy()
    if not (np.isfinite(got).all() and np.isfinite(r).all()):
        failures = report(failures, False, f"bf16 ref nonfinite sgn={ref_sgn}")
        del gq, uq, dq, wg, wu, wd, x, xb
        torch.xpu.empty_cache()
        return failures, hard_ok
    absd = np.abs(got - r)
    rel = absd / np.maximum(np.maximum(np.abs(got), np.abs(r)), 1e-6)
    mx = float(absd.max())
    mr = float(rel.max())
    failures = report(failures, mr <= 5e-2, f"sgn={ref_sgn} vs bf16-weight max_abs={mx:.3e} max_rel={mr:.3e}")
    del gq, uq, dq, wg, wu, wd, x, xb
    torch.xpu.empty_cache()
    # 8 and 16 matching SGN=4 is enough to time the layer. A 40 miss or a
    # bf16-tolerance miss stays in failures and still prints in VERDICT.
    return failures, hard_ok


def time_sgn(gq, uq, dq, best: dict) -> None:
    os.environ["B70_PREFILL_SPLIT"] = "0"
    for t in (512, 4096, 12288):
        ids = torch.randint(0, E, (t, TOPK), device="xpu", dtype=torch.int32)
        x = (torch.randn(t, H, device="xpu", dtype=torch.float16) * 0.001).contiguous()
        router = torch.rand(t, TOPK, device="xpu", dtype=torch.float32)
        flat = ids.reshape(-1)
        bc = torch.bincount(flat, minlength=E)
        mmax = int(bc.max())
        empty = int((bc == 0).sum())
        print(f"ROUTE T={t} mmax={mmax} empty={empty} mean={float(bc.float().mean()):.2f}", flush=True)
        iters = 1 if t >= 12288 else 2
        for sgn in (4, 8, 16, 40):
            os.environ["B70_PREFILL_SGN"] = str(sgn)
            print(f"WARM sgn={sgn} T={t}", flush=True)
            try:
                ev, wall = elapsed_both(lambda: layer_call(x, ids, gq, uq, dq, router), 1, iters)
            except Exception as ex:
                print(f"FAIL layer sgn={sgn} T={t} {type(ex).__name__}: {ex}", flush=True)
                continue
            os.environ["B70_PREFILL_SPLIT"] = "1"
            try:
                layer_call(x, ids, gq, uq, dq, router)
                torch.xpu.synchronize()
                sp = torch.ops.b70_gguf.prefill_split_us().tolist()
            except Exception as ex:
                sp = [0] * 8
                print(f"FAIL split sgn={sgn} T={t} {type(ex).__name__}: {ex}", flush=True)
            os.environ["B70_PREFILL_SPLIT"] = "0"
            cols = int(sp[7]) // 1000
            mt = int(sp[7]) % 1000
            print(
                f"BENCH layer sgn={sgn} T={t} event_us={ev:.1f} wall_us={wall:.1f} "
                f"tf={tflops_layer(t, ev):.2f} cols={cols} mt={mt} "
                f"gu_us={sp[2]:.1f} down_us={sp[3]:.1f} "
                f"c10_ms={C10_LAYER_MS[t]:.2f}",
                flush=True,
            )
            prev = best.get(t)
            if cols == sgn * 16 and (prev is None or ev < prev[0]):
                best[t] = (ev, sgn, wall)
        del x, ids, router
        torch.xpu.empty_cache()


def dequant_gbps(gq, uq, dq) -> None:
    jobs = (
        ("gate", gq, IQ3, E * F, H),
        ("up", uq, IQ3, E * F, H),
        ("down", dq, Q2, E * H, F),
    )
    for name, q, typ, rows, k in jobs:
        out = torch.empty((rows, k), dtype=torch.bfloat16, device="xpu")
        nbytes = rows * k * 2 + rows * row_bytes(typ, k)

        def run(q=q, typ=typ, rows=rows, k=k, out=out):
            torch.ops.b70_gguf.prefill_dequant(q, typ, rows, k, out)

        ev, wall = elapsed_both(run, 1, 2)
        print(f"BENCH dequant {name} event_us={ev:.1f} gbps={gbps(nbytes, ev):.1f} wall_us={wall:.1f}", flush=True)
        del out
    # Copy ceiling on the gate bf16 footprint.
    n = E * F * H
    src = torch.empty(n, dtype=torch.bfloat16, device="xpu")
    dst = torch.empty(n, dtype=torch.bfloat16, device="xpu")
    ev, wall = elapsed_both(lambda: dst.copy_(src), 1, 2)
    print(f"BENCH copy gate_bf16 event_us={ev:.1f} gbps={gbps(n * 2 * 2, ev):.1f} wall_us={wall:.1f}", flush=True)
    del src, dst


def onednn_ceilings() -> None:
    import torch.nn.functional as F

    print("BENCH onednn start", flush=True)
    for t, rows_per in ((4096, 80), (12288, 240)):
        m = t * TOPK
        x = torch.randn(m, H, device="xpu", dtype=torch.bfloat16)
        w = torch.randn(H, F, device="xpu", dtype=torch.bfloat16)
        flops = 2.0 * m * F * H

        def mm(x=x, w=w):
            torch.mm(x, w)

        ev, wall = elapsed_both(mm, 1, 2 if t < 12288 else 1)
        print(
            f"BENCH dense_ceil gate T={t} event_us={ev:.1f} tf={flops / ev / 1e6:.2f} wall_us={wall:.1f}",
            flush=True,
        )
        # Uniform grouped: every expert has the same row count, so bmm has no pad.
        x3 = x.view(E, rows_per, H)
        w3 = torch.randn(E, H, F, device="xpu", dtype=torch.bfloat16)

        def bmm(x3=x3, w3=w3):
            torch.bmm(x3, w3)

        ev, wall = elapsed_both(bmm, 1, 2 if t < 12288 else 1)
        print(
            f"BENCH bmm_uniform gate T={t} event_us={ev:.1f} tf={flops / ev / 1e6:.2f} wall_us={wall:.1f}",
            flush=True,
        )
        offs = torch.arange(1, E + 1, device="xpu", dtype=torch.int32) * rows_per
        try:

            def gmm(x=x, w3=w3, offs=offs):
                F.grouped_mm(x, w3, offs=offs)

            ev, wall = elapsed_both(gmm, 1, 1)
            print(
                f"BENCH grouped_mm gate T={t} event_us={ev:.1f} tf={flops / ev / 1e6:.2f} wall_us={wall:.1f}",
                flush=True,
            )
        except Exception as ex:
            print(f"PROBE grouped_mm T={t} {type(ex).__name__}: {ex}", flush=True)
        # Down shape, uniform, one bmm. K=F N=H.
        xd = torch.randn(E, rows_per, F, device="xpu", dtype=torch.bfloat16)
        wd = torch.randn(E, F, H, device="xpu", dtype=torch.bfloat16)
        flops_d = 2.0 * m * H * F

        def bmm_d(xd=xd, wd=wd):
            torch.bmm(xd, wd)

        ev, wall = elapsed_both(bmm_d, 1, 1)
        print(
            f"BENCH bmm_uniform down T={t} event_us={ev:.1f} tf={flops_d / ev / 1e6:.2f} wall_us={wall:.1f}",
            flush=True,
        )
        # Fused gate+up N: one bmm with N=2F.
        wgu = torch.randn(E, H, 2 * F, device="xpu", dtype=torch.bfloat16)
        flops_gu = 2.0 * flops

        def bmm_gu(x3=x3, wgu=wgu):
            torch.bmm(x3, wgu)

        ev, wall = elapsed_both(bmm_gu, 1, 1)
        print(
            f"BENCH bmm_uniform gateup T={t} event_us={ev:.1f} tf={flops_gu / ev / 1e6:.2f} wall_us={wall:.1f}",
            flush=True,
        )
        del x, w, x3, w3, xd, wd, wgu
        torch.xpu.empty_cache()
    # Two streams: a large mm against a same-size copy.
    a = torch.randn(4096, H, device="xpu", dtype=torch.bfloat16)
    b = torch.randn(H, 4096, device="xpu", dtype=torch.bfloat16)
    c = torch.empty_like(a)
    s1 = torch.xpu.Stream()
    s2 = torch.xpu.Stream()

    def only_mm():
        torch.mm(a, b)

    def only_copy():
        c.copy_(a)

    def both():
        with torch.xpu.stream(s1):
            torch.mm(a, b)
        with torch.xpu.stream(s2):
            c.copy_(a)

    ev_m, wall_m = elapsed_both(only_mm, 1, 2)
    ev_c, wall_c = elapsed_both(only_copy, 1, 2)
    ev_b, wall_b = elapsed_both(both, 1, 2)
    print(
        f"BENCH overlap mm_us={ev_m:.1f} copy_us={ev_c:.1f} both_us={ev_b:.1f} "
        f"wall_mm={wall_m:.1f} wall_copy={wall_c:.1f} wall_both={wall_b:.1f}",
        flush=True,
    )


def main() -> int:
    assert torch.xpu.is_available()
    print(f"DEVICE {torch.xpu.get_device_name(0)} so={os.environ.get('B70_GGUF_SO', '')}", flush=True)
    load()
    wide = int(torch.ops.b70_gguf.prefill_wide_sgn().item())
    print(f"WIDE_SGN {wide}", flush=True)
    (torch.randn(64, 64, device="xpu", dtype=torch.bfloat16) @ torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)).sum().item()
    torch.xpu.synchronize()
    failures = 0
    if "--bench-only" not in sys.argv:
        failures, proceed = check_small(failures)
        if not proceed:
            print(f"VERDICT failures={failures} skip_bench=1", flush=True)
            return 1
    if "--check-only" in sys.argv:
        print(f"VERDICT failures={failures}", flush=True)
        return 0
    print("PACK expert layer", flush=True)
    gq = torch.from_numpy(fast_pack(IQ3, E * F, H, 7)).to("xpu")
    uq = torch.from_numpy(fast_pack(IQ3, E * F, H, 8)).to("xpu")
    dq = torch.from_numpy(fast_pack(Q2, E * H, F, 9)).to("xpu")
    best: dict = {}
    if "--dnn-only" not in sys.argv:
        time_sgn(gq, uq, dq, best)
    if "--sgn-only" not in sys.argv:
        try:
            dequant_gbps(gq, uq, dq)
        except Exception as ex:
            print(f"FAIL dequant {type(ex).__name__}: {ex}", flush=True)
        del gq, uq, dq
        torch.xpu.empty_cache()
        try:
            onednn_ceilings()
        except Exception as ex:
            print(f"FAIL onednn {type(ex).__name__}: {ex}", flush=True)
    else:
        del gq, uq, dq
    for t, item in sorted(best.items()):
        ev, sgn, wall = item
        layer_ms = ev / 1000.0
        expert_sum = layer_ms * 48.0
        gemm_sum = C10_GEMM_SUM[t] - C10_EXPERT_SUM[t] + expert_sum
        toks = t / (gemm_sum / 1000.0)
        print(
            f"BUDGET T={t} best_sgn={sgn} layer_ms={layer_ms:.2f} expert48_ms={expert_sum:.1f} "
            f"gemm_sum_ms={gemm_sum:.1f} gemm_tok/s={toks:.0f} wall_us={wall:.1f}",
            flush=True,
        )
    print(f"VERDICT failures={failures}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
