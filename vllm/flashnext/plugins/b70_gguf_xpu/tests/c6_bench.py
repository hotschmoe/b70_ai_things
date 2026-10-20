"""C6 prefill schedule check and bench. Run via scripts/c6_run.sh under gpu-run --card 0.

Correctness matches prefill_check.py: fp16 vs fp32 dequant at atol=rtol=5e-2.
Bench prints event-min microseconds. torch.mm bf16 on the same MxNxK is the XMX ceiling.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
sys.path.insert(0, str(ROOT))

from gguf import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

import torch  # noqa: E402

from b70_gguf_xpu.prefill import prefill_dense, prefill_grouped, prefill_moe_layer, prefill_split_us  # noqa: E402
from b70_gguf_xpu.types import BLOCK, row_bytes  # noqa: E402
from prefill_check import close_enough, cpu_dequant, elapsed_us, load_cpu, report, synth_pack  # noqa: E402

Q4, IQ3, Q2 = 12, 18, 42


def ref_w(lib, tid: int, raw: np.ndarray, k: int) -> np.ndarray:
    if tid == Q2:
        return cpu_dequant(lib, tid, raw, k)
    return np.asarray(dequantize(raw, GGMLQuantizationType(tid)), dtype=np.float32)


def check_dense_one(lib, tid, n, k, m, failures):
    raw = synth_pack(tid, n, k, 100 + tid + m + k)
    w = ref_w(lib, tid, raw, k)
    q = torch.from_numpy(raw).to(device="xpu")
    x = torch.randn(m, k, device="xpu", dtype=torch.float16)
    y = prefill_dense(x, q, tid, n, k, 0)
    torch.xpu.synchronize()
    ref = x.float().cpu().numpy() @ w.T
    ok, mx, mr = close_enough(y.float().cpu().numpy(), ref, 5e-2, 5e-2)
    return report(failures, "OK" if ok else "FAIL", f"dense tid={tid} M={m} N={n} K={k} max_abs={mx:.3e} max_rel={mr:.3e}")


def check_grouped_one(lib, tid, n, k, failures):
    raw = synth_pack(tid, n, k, 200 + tid + k)
    parts = [raw, raw, np.ascontiguousarray(raw[::-1])]
    blob = np.ascontiguousarray(np.concatenate(parts, axis=0))
    weights = [ref_w(lib, tid, p, k) for p in parts]
    q = torch.from_numpy(blob).to(device="xpu")
    ids_np = np.array([0] * 40 + [2] * 7 + [-1, 1, 1], np.int32)
    rng = np.random.default_rng(1)
    rng.shuffle(ids_np)
    x = torch.randn(ids_np.shape[0], k, device="xpu", dtype=torch.float16)
    y = prefill_grouped(x, q, torch.from_numpy(ids_np).to(device="xpu"), tid, n, k, 0)
    torch.xpu.synchronize()
    xf = x.float().cpu().numpy()
    ref = np.zeros((ids_np.shape[0], n), np.float32)
    for i, e in enumerate(ids_np.tolist()):
        if e >= 0:
            ref[i] = xf[i] @ weights[e].T
    ok, mx, mr = close_enough(y.float().cpu().numpy(), ref, 5e-2, 5e-2)
    return report(failures, "OK" if ok else "FAIL", f"grouped tid={tid} M={ids_np.shape[0]} E=3 N={n} K={k} max_abs={mx:.3e} max_rel={mr:.3e}")


def check_layer_small(lib, failures):
    # H multiple of the IQ3 block, F multiple of the Q2_0 block, 2F multiple of 64.
    # Reference rounds x and the swiglu output through fp16, matching the kernel.
    h, f, e, topk, t = 256, 64, 4, 2, 5
    g = synth_pack(IQ3, e * f, h, 11)
    u = synth_pack(IQ3, e * f, h, 12)
    d = synth_pack(Q2, e * h, f, 13)
    gates = [ref_w(lib, IQ3, np.ascontiguousarray(g.reshape(e * f, -1)[i * f : (i + 1) * f]), h) for i in range(e)]
    ups = [ref_w(lib, IQ3, np.ascontiguousarray(u.reshape(e * f, -1)[i * f : (i + 1) * f]), h) for i in range(e)]
    dns = [ref_w(lib, Q2, np.ascontiguousarray(d.reshape(e * h, -1)[i * h : (i + 1) * h]), f) for i in range(e)]
    rng = np.random.default_rng(3)
    ids_np = rng.integers(0, e, size=(t, topk), dtype=np.int32)
    ids_np[0, 1] = -1
    x_np = rng.standard_normal((t, h)).astype(np.float16).astype(np.float32)
    rw = rng.random((t, topk)).astype(np.float32)
    ref = np.zeros((t, h), np.float32)
    for ti in range(t):
        for k in range(topk):
            eid = int(ids_np[ti, k])
            if eid < 0:
                continue
            xv = x_np[ti]
            gv = xv @ gates[eid].T
            uv = xv @ ups[eid].T
            hv = (gv / (1.0 + np.exp(-gv))) * uv
            hv = hv.astype(np.float16).astype(np.float32)
            ref[ti] += rw[ti, k] * (hv @ dns[eid].T)
    x = torch.from_numpy(x_np.astype(np.float16)).to(device="xpu")
    ids = torch.from_numpy(ids_np).to(device="xpu")
    router = torch.from_numpy(rw).to(device="xpu")
    y = prefill_moe_layer(
        x,
        ids,
        torch.from_numpy(g).to(device="xpu"),
        torch.from_numpy(u).to(device="xpu"),
        torch.from_numpy(d).to(device="xpu"),
        router,
        IQ3,
        Q2,
        h,
        f,
        0,
    )
    torch.xpu.synchronize()
    ok, mx, mr = close_enough(y.float().cpu().numpy(), ref, 5e-2, 5e-2)
    return report(failures, "OK" if ok else "FAIL", f"layer small T={t} E={e} H={h} F={f} max_abs={mx:.3e} max_rel={mr:.3e}")


# (sched, mt or None, rm or None). Both instantiations are already in the smoke .so.
ARMS = (
    ("s64", "4", None),
    ("s64", "16", None),
    ("reuse", None, "32"),
    ("reuse", None, "64"),
)


def apply_arm(sched: str, mt, rm) -> str:
    os.environ["B70_PREFILL_SCHED"] = sched
    if mt is None:
        os.environ.pop("B70_PREFILL_MT", None)
    else:
        os.environ["B70_PREFILL_MT"] = mt
    if rm is None:
        os.environ.pop("B70_PREFILL_RM", None)
    else:
        os.environ["B70_PREFILL_RM"] = rm
    return f"{sched}-mt{mt}" if sched == "s64" else f"{sched}-rm{rm}"


def guard(failures: int, label: str, fn) -> int:
    try:
        return fn(failures)
    except Exception as ex:
        print(f"FAIL {label} {type(ex).__name__}: {ex}", flush=True)
        try:
            torch.xpu.synchronize()
        except Exception:
            pass
        return failures + 1


def split_off(kind: str) -> None:
    torch.xpu.synchronize()
    sp = prefill_split_us().tolist()
    os.environ["B70_PREFILL_SPLIT"] = "0"
    cols = int(sp[7]) // 1000
    mt = int(sp[7]) % 1000
    if kind == "dense":
        print(f"SPLIT dense gemm={sp[2]:.1f} cols={cols} mt={mt}", flush=True)
    else:
        print(
            f"SPLIT layer sort={sp[0]:.1f} gather={sp[1]:.1f} gu={sp[2]:.1f} down={sp[3]:.1f} "
            f"swiglu={sp[4]:.1f} reduce={sp[5]:.1f} total={sp[6]:.1f} cols={cols} mt={mt}",
            flush=True,
        )


def main() -> int:
    quick = "--full" not in sys.argv
    assert torch.xpu.is_available()
    assert torch.xpu.device_count() == 1, torch.xpu.device_count()
    print(
        f"DEVICE {torch.xpu.get_device_name(0)} quick={int(quick)} so={os.environ.get('B70_GGUF_SO', '')}",
        flush=True,
    )
    (torch.randn(64, 64, device="xpu", dtype=torch.bfloat16) @ torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)).sum().item()
    torch.xpu.synchronize()
    print("PROBE matmul ok", flush=True)
    lib = load_cpu()
    failures = 0
    # Q4 hits every instantiation. IQ3 and Q2 hit both schedules (s64-mt4, reuse-rm64).
    if "--bench-only" not in sys.argv:
        for sched, mt, rm in ARMS:
            name = apply_arm(sched, mt, rm)
            types = ((Q4, 128, 2560), (IQ3, 128, 2560), (Q2, 128, 640)) if name in ("s64-mt4", "reuse-rm64") else ((Q4, 128, 2560),)
            for tid, n, k in types:
                for m in (7, 130):
                    failures = guard(
                        failures,
                        f"dense {name} tid={tid} M={m}",
                        lambda f, tid=tid, n=n, k=k, m=m: check_dense_one(lib, tid, n, k, m, f),
                    )
                failures = guard(
                    failures,
                    f"grouped {name} tid={tid}",
                    lambda f, tid=tid, n=n, k=k: check_grouped_one(lib, tid, n, k, f),
                )
        apply_arm("s64", "4", None)
        failures = guard(failures, "layer small", lambda f: check_layer_small(lib, f))
        if failures:
            print(f"VERDICT failures={failures} skip_bench=1", flush=True)
            return 1
    if "--check-only" not in sys.argv:
        shapes = [(512, 12288, 2560)] if quick else [(512, 12288, 2560), (4096, 12288, 2560), (12288, 12288, 2560)]
        for m, n, k in shapes:
            print(f"PACK dense M={m}", flush=True)
            raw = synth_pack(Q4, n, k, 7)
            q = torch.from_numpy(raw).to(device="xpu")
            x = torch.randn(m, k, device="xpu", dtype=torch.float16)
            flops = 2.0 * m * n * k
            for sched, mt, rm in ARMS:
                name = apply_arm(sched, mt, rm)

                def run_dense(name=name, m=m, n=n, k=k, flops=flops):
                    us = elapsed_us(lambda: prefill_dense(x, q, Q4, n, k, 0), warmup=1, iters=3)
                    print(
                        f"BENCH dense {name} Q4_K M={m} N={n} K={k} us={us:.1f} tflops={flops / us / 1e6:.2f}",
                        flush=True,
                    )
                    os.environ["B70_PREFILL_SPLIT"] = "1"
                    prefill_dense(x, q, Q4, n, k, 0)
                    split_off("dense")
                    return 0

                failures = guard(failures, f"bench dense {name} M={m}", run_dense)
            xb = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
            wb = torch.randn(n, k, device="xpu", dtype=torch.bfloat16)
            us_mm = elapsed_us(lambda: torch.mm(xb, wb.t()), warmup=2, iters=4)
            print(f"BENCH torch.mm bf16 M={m} N={n} K={k} us={us_mm:.1f} tflops={flops / us_mm / 1e6:.2f}", flush=True)
            del x, q, raw, xb, wb
            torch.xpu.empty_cache()
        h, ffn, e, topk = 2560, 640, 512, 10
        print("PACK layer", flush=True)
        g = synth_pack(IQ3, e * ffn, h, 7)
        u = synth_pack(IQ3, e * ffn, h, 8)
        d = synth_pack(Q2, e * h, ffn, 9)
        gq = torch.from_numpy(g).to(device="xpu")
        uq = torch.from_numpy(u).to(device="xpu")
        dq = torch.from_numpy(d).to(device="xpu")
        nbytes = e * ffn * row_bytes(IQ3, h) * 2 + e * h * row_bytes(Q2, ffn)
        ts = [512] if quick else [512, 4096, 12288]
        for t in ts:
            ids = torch.randint(0, e, (t, topk), device="xpu", dtype=torch.int32)
            x = torch.randn(t, h, device="xpu", dtype=torch.float16)
            router = torch.rand(t, topk, device="xpu", dtype=torch.float32)
            iters = 2 if t >= 12288 else 3
            for sched, mt, rm in ARMS:
                name = apply_arm(sched, mt, rm)

                def run_layer(name=name, t=t, iters=iters):
                    us = elapsed_us(
                        lambda: prefill_moe_layer(x, ids, gq, uq, dq, router, IQ3, Q2, h, ffn, 0),
                        warmup=1,
                        iters=iters,
                    )
                    tok = t / (us * 1e-6)
                    gbps = (nbytes / 1e9) / (us * 1e-6)
                    print(
                        f"BENCH layer {name} T={t} E={e} us={us:.1f} unique_GBps={gbps:.2f} "
                        f"one_layer_tok/s={tok:.1f} expert48_tok/s={tok / 48:.1f}",
                        flush=True,
                    )
                    os.environ["B70_PREFILL_SPLIT"] = "1"
                    prefill_moe_layer(x, ids, gq, uq, dq, router, IQ3, Q2, h, ffn, 0)
                    split_off("layer")
                    return 0

                failures = guard(failures, f"bench layer {name} T={t}", run_layer)
            del x, ids, router
            torch.xpu.empty_cache()
    print(f"VERDICT failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
