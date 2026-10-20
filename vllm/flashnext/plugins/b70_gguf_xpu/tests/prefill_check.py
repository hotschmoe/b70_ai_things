"""Prefill correctness and per-layer latency. Run via scripts/prefill_gpu.sh under gpu-run --card 0.

fp16/bf16 must match a fp32 dequant reference at atol=rtol=5e-2.
int8 is an evaluation: fail only on NaN/Inf or a large fraction of lanes far from the reference.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
sys.path.insert(0, str(ROOT))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

import torch  # noqa: E402

from b70_gguf_xpu.prefill import prefill_dense, prefill_grouped, prefill_moe_layer, prefill_split_us  # noqa: E402
from b70_gguf_xpu.types import BLOCK, NAMES, row_bytes  # noqa: E402

SHARD = (
    "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)
ROOF_GBPS = 608.0
# Q2_K/Q3_K are MTP-only and are not in the prefill instantiation set.
WANTED = [12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 42]
SMOKE_TYPES = {18, 42}
EVAL_MODES = (1, 2)  # bf16, int8. Built for a subset of types.
EVAL_TYPES = {12, 13, 14, 18, 20, 21, 23, 42}


def load_cpu():
    lib = ctypes.CDLL(str(ROOT / "b70_gguf_xpu" / "libdequant_cpu.so"))
    lib.b70_dequant_f32_cpu.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


def cpu_dequant(lib, tid: int, raw: np.ndarray, k: int) -> np.ndarray:
    rows = raw.shape[0]
    out = np.empty((rows, k), np.float32)
    lib.b70_dequant_f32_cpu(
        tid,
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        rows,
        k,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def ref_dequant(tid: int, raw: np.ndarray, k: int, lib) -> np.ndarray:
    if tid == 42:
        return cpu_dequant(lib, tid, raw, k)
    return np.asarray(dequantize(raw, GGMLQuantizationType(tid)), dtype=np.float32)


def close_enough(got: np.ndarray, ref: np.ndarray, atol: float, rtol: float):
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    scale = np.maximum(np.abs(ref.astype(np.float64)), 1e-6)
    rel = d / scale
    mx = float(d.max()) if d.size else 0.0
    mr = float(rel.max()) if rel.size else 0.0
    ok = bool(np.all(d <= (atol + rtol * scale)))
    return ok, mx, mr


def int8_sane(got: np.ndarray, ref: np.ndarray):
    if not np.all(np.isfinite(got)):
        return False, float("nan"), 1.0
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    scale = np.maximum(np.abs(ref.astype(np.float64)), 1e-6)
    rel = d / scale
    frac = float(np.mean(rel > 0.5)) if rel.size else 0.0
    mx = float(d.max()) if d.size else 0.0
    # A swapped lane misses a large fraction of outputs. Quant error does not.
    return frac < 0.30, mx, frac


def elapsed_us(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return min(samples)


def flat_matrix(tensor):
    data = np.asarray(tensor.data)
    k = int(tensor.shape[0])
    flat = np.ascontiguousarray(data.reshape(-1, data.shape[-1]))
    return flat, k


def take_rows(flat, k, tid, n_want):
    if row_bytes(tid, k) <= 0 or k % 32 != 0:
        return None
    n = min(n_want, flat.shape[0])
    n -= n % 64
    if n < 64:
        return None
    return np.ascontiguousarray(flat[:n])


def synth_pack(typ: int, rows: int, k: int, seed: int) -> np.ndarray:
    rb = row_bytes(typ, k)
    if rb <= 0:
        raise ValueError(f"bad pack type={typ} k={k}")
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(rows, rb), dtype=np.uint8)
    be, bb = BLOCK[typ]
    view = raw.reshape(rows, k // be, bb)
    view[:, :, 0] = 0x00
    view[:, :, 1] = 0x3C
    return np.ascontiguousarray(raw)


def report(failures: int, status: str, msg: str) -> int:
    print(f"{status} {msg}")
    return failures + (0 if status == "OK" else 1)


def check_dense(lib, tid, raw, k, failures: int) -> int:
    n = raw.shape[0]
    name = NAMES[tid]
    w = ref_dequant(tid, raw, k, lib)
    q = torch.from_numpy(np.array(raw, dtype=np.uint8, copy=True)).to(device="xpu")
    for m in (7, 130):
        x = torch.randn(m, k, device="xpu", dtype=torch.float16)
        y = prefill_dense(x, q, tid, n, k, 0)
        torch.xpu.synchronize()
        ref = x.float().cpu().numpy() @ w.T
        ok, mx, mr = close_enough(y.cpu().numpy(), ref, 5e-2, 5e-2)
        failures = report(failures, "OK" if ok else "FAIL", f"dense {name} M={m} N={n} K={k} mode=0 max_abs={mx:.3e} max_rel={mr:.3e}")
    return failures


def context_alive() -> bool:
    try:
        a = torch.randn(16, 16, device="xpu", dtype=torch.float16)
        (a @ a).float().sum().item()
        torch.xpu.synchronize()
        return True
    except RuntimeError:
        return False


def check_eval(lib, tid, raw, k, failures: int, dead_modes: set) -> int:
    """bf16 and int8 after every fp16 check. A rejected combination must not
    be retried, and a poisoned context must not be fed more launches."""
    if tid not in EVAL_TYPES:
        return failures
    n = raw.shape[0]
    name = NAMES[tid]
    w = ref_dequant(tid, raw, k, lib)
    q = torch.from_numpy(np.array(raw, dtype=np.uint8, copy=True)).to(device="xpu")
    x = torch.randn(17, k, device="xpu", dtype=torch.bfloat16)
    ref = x.float().cpu().numpy() @ w.T
    if 1 not in dead_modes:
        try:
            y = prefill_dense(x, q, tid, n, k, 1)
            torch.xpu.synchronize()
            ok, mx, mr = close_enough(y.cpu().numpy(), ref, 5e-2, 5e-2)
            failures = report(failures, "OK" if ok else "FAIL", f"dense {name} M=17 N={n} K={k} mode=1 max_abs={mx:.3e} max_rel={mr:.3e}")
        except RuntimeError as exc:
            dead_modes.add(1)
            failures = report(failures, "FAIL", f"dense {name} M=17 N={n} K={k} mode=1 EXC {exc}")
            if not context_alive():
                dead_modes.add(2)
                print("CONTEXT dead after mode=1; skipping further eval launches")
    x16 = x.to(torch.float16)
    ref8 = x16.float().cpu().numpy() @ w.T
    if 2 not in dead_modes:
        try:
            y8 = prefill_dense(x16, q, tid, n, k, 2)
            torch.xpu.synchronize()
            ok, mx, frac = int8_sane(y8.cpu().numpy(), ref8)
            failures = report(failures, "OK" if ok else "FAIL", f"dense {name} M=17 N={n} K={k} mode=2 max_abs={mx:.3e} far_frac={frac:.3f}")
        except RuntimeError as exc:
            dead_modes.add(2)
            failures = report(failures, "FAIL", f"dense {name} M=17 N={n} K={k} mode=2 EXC {exc}")
            if not context_alive():
                print("CONTEXT dead after mode=2; skipping further eval launches")
    return failures


def check_grouped(lib, tid, raw, k, failures: int) -> int:
    n = raw.shape[0]
    name = NAMES[tid]
    # Three copies. ids are shuffled so the device sort is on the critical path.
    parts = [raw, raw, np.ascontiguousarray(raw[::-1])]
    blob = np.ascontiguousarray(np.concatenate(parts, axis=0))
    weights = [ref_dequant(tid, p, k, lib) for p in parts]
    q = torch.from_numpy(blob).to(device="xpu")
    rng = np.random.default_rng(1)
    ids_np = np.array([0] * 40 + [2] * 7 + [-1, 1, 1], np.int32)
    rng.shuffle(ids_np)
    m = ids_np.shape[0]
    x = torch.randn(m, k, device="xpu", dtype=torch.float16)
    ids = torch.from_numpy(ids_np).to(device="xpu")
    y = prefill_grouped(x, q, ids, tid, n, k, 0)
    torch.xpu.synchronize()
    xf = x.float().cpu().numpy()
    ref = np.zeros((m, n), np.float32)
    for i, e in enumerate(ids_np.tolist()):
        if e < 0:
            continue
        ref[i] = xf[i] @ weights[e].T
    ok, mx, mr = close_enough(y.cpu().numpy(), ref, 5e-2, 5e-2)
    failures = report(
        failures, "OK" if ok else "FAIL", f"grouped {name} M={m} E=3 N={n} K={k} mode=0 max_abs={mx:.3e} max_rel={mr:.3e}"
    )
    # One expert, more rows than an MT=16 tile, so the kernel re-reads weights.
    ids2 = torch.zeros(130, dtype=torch.int32, device="xpu")
    x2 = torch.randn(130, k, device="xpu", dtype=torch.float16)
    q1 = torch.from_numpy(np.ascontiguousarray(raw)).to(device="xpu")
    y2 = prefill_grouped(x2, q1, ids2, tid, n, k, 0)
    torch.xpu.synchronize()
    ref2 = x2.float().cpu().numpy() @ weights[0].T
    ok, mx, mr = close_enough(y2.cpu().numpy(), ref2, 5e-2, 5e-2)
    failures = report(
        failures, "OK" if ok else "FAIL", f"grouped {name} M=130 E=1 N={n} K={k} mode=0 max_abs={mx:.3e} max_rel={mr:.3e}"
    )
    return failures


def check_layer(lib, gate, up, down, failures: int) -> int:
    # gate/up: [E, F, H] in packed rows. down: [E, H, F].
    gt, gk = int(gate.tensor_type), int(gate.shape[0])
    ut, uk = int(up.tensor_type), int(up.shape[0])
    dt, dk = int(down.tensor_type), int(down.shape[0])
    gflat, _ = flat_matrix(gate)
    uflat, _ = flat_matrix(up)
    dflat, _ = flat_matrix(down)
    if gflat.shape != uflat.shape or gt != ut or gk != uk:
        print(f"SKIP layer gate/up mismatch gate={gate.name} up={up.name}")
        return failures
    # gguf-py shape[0] is K. Row count of one expert is the middle dim.
    gdata = np.asarray(gate.data)
    ddata = np.asarray(down.data)
    if gdata.ndim < 3 or ddata.ndim < 3:
        print("SKIP layer not 3D")
        return failures
    e_all = gdata.shape[-3] if gdata.ndim >= 3 else 0
    # reshape to [E, N, rb]
    grow = gdata.reshape(-1, gdata.shape[-2], gdata.shape[-1])
    urow = np.asarray(up.data).reshape(-1, np.asarray(up.data).shape[-2], np.asarray(up.data).shape[-1])
    drow = ddata.reshape(-1, ddata.shape[-2], ddata.shape[-1])
    f = grow.shape[1]
    h = drow.shape[1]
    if gk != h or dk != f or drow.shape[0] < 2 or grow.shape[0] < 2:
        print(f"SKIP layer shape H={h} F={f} Kg={gk} Kd={dk} Eg={grow.shape[0]} Ed={drow.shape[0]}")
        return failures
    if h % 64 or f % 32 or (2 * f) % 64:
        print(f"SKIP layer alignment H={h} F={f}")
        return failures
    e = 2
    gpack = np.ascontiguousarray(grow[:e].reshape(-1))
    upack = np.ascontiguousarray(urow[:e].reshape(-1))
    dpack = np.ascontiguousarray(drow[:e].reshape(-1))
    # Reference weights per expert: gate [F, H], up [F, H], down [H, F].
    gw, uw, dw = [], [], []
    for i in range(e):
        gw.append(ref_dequant(gt, np.ascontiguousarray(grow[i]), h, lib))
        uw.append(ref_dequant(ut, np.ascontiguousarray(urow[i]), h, lib))
        dw.append(ref_dequant(dt, np.ascontiguousarray(drow[i]), f, lib))
    t, topk = 3, 2
    rng = np.random.default_rng(2)
    ids_np = rng.integers(0, e, size=(t, topk), dtype=np.int32)
    ids_np[0, 1] = -1
    rw_np = rng.random((t, topk), dtype=np.float32) * 0.3
    x = torch.randn(t, h, device="xpu", dtype=torch.float16)
    ids = torch.from_numpy(ids_np).to(device="xpu")
    router = torch.from_numpy(rw_np).to(device="xpu")
    y = prefill_moe_layer(
        x,
        ids,
        torch.from_numpy(gpack).to(device="xpu"),
        torch.from_numpy(upack).to(device="xpu"),
        torch.from_numpy(dpack).to(device="xpu"),
        router,
        gt,
        dt,
        h,
        f,
        0,
    )
    torch.xpu.synchronize()
    xf = x.float().cpu().numpy()
    ref = np.zeros((t, h), np.float32)
    for ti in range(t):
        acc = np.zeros(h, np.float32)
        for s in range(topk):
            eid = int(ids_np[ti, s])
            if eid < 0:
                continue
            g = xf[ti] @ gw[eid].T
            u = xf[ti] @ uw[eid].T
            hid = g / (1.0 + np.exp(-g)) * u
            hid = hid.astype(np.float16).astype(np.float32)
            acc += rw_np[ti, s] * (hid @ dw[eid].T)
        ref[ti] = acc
    ok, mx, mr = close_enough(y.cpu().numpy(), ref, 5e-2, 5e-2)
    failures = report(
        failures,
        "OK" if ok else "FAIL",
        f"layer {NAMES[gt]}->{NAMES[dt]} T={t} E={e} H={h} F={f} max_abs={mx:.3e} max_rel={mr:.3e} gate={gate.name}",
    )
    return failures


def gbps(nbytes: int, us: float) -> float:
    if us <= 0:
        return 0.0
    return (nbytes / 1e9) / (us * 1e-6)


def bench_one(label: str, fn, nbytes: int, warmup: int = 2, iters: int = 8) -> str:
    us = elapsed_us(fn, warmup=warmup, iters=iters)
    rate = gbps(nbytes, us)
    return f"{label}\t{us:.1f}\t{rate:.2f}\t{100.0 * rate / ROOF_GBPS:.2f}"


def run_bench(smoke: bool) -> int:
    failures = 0
    print("BENCH op\tshape\tus\tGB/s\tpct_of_608")
    # A2 single-expert shapes. One expert, so this is the old launch's occupancy with the new kernel.
    cases = [
        ("grouped_a2", 20, 128, 2560, 640, 1),
        ("grouped_a2", 18, 128, 640, 2560, 1),
    ]
    if not smoke:
        cases += [
            ("dense", 12, 512, 12288, 2560, 1),
            ("dense", 12, 4096, 12288, 2560, 1),
            ("dense", 12, 12288, 12288, 2560, 1),
            ("dense", 13, 512, 12288, 2560, 1),
            ("dense", 13, 4096, 12288, 2560, 1),
            ("dense", 14, 512, 12288, 2560, 1),
            ("dense", 23, 512, 12288, 2560, 1),
            ("dense", 21, 512, 12288, 2560, 1),
            ("head", 13, 512, 248320, 2560, 1),
        ]
    for kind, tid, m, n, k, e in cases:
        try:
            raw = synth_pack(tid, e * n, k, 1000 + tid + m)
            q = torch.from_numpy(raw).to(device="xpu")
            x = torch.randn(m, k, device="xpu", dtype=torch.float16)
            if kind == "grouped_a2":
                ids = torch.zeros(m, dtype=torch.int32, device="xpu")
                fn = lambda: prefill_grouped(x, q, ids, tid, n, k, 0)
            else:
                fn = lambda: prefill_dense(x, q, tid, n, k, 0)
            nbytes = e * n * row_bytes(tid, k)
            line = bench_one(f"{kind}\t{NAMES[tid]}\tM={m} N={n} K={k} E={e}", fn, nbytes)
            print(line)
            os.environ["B70_PREFILL_SPLIT"] = "1"
            fn()
            torch.xpu.synchronize()
            sp = prefill_split_us().tolist()
            os.environ["B70_PREFILL_SPLIT"] = "0"
            print(
                f"SPLIT {kind} {NAMES[tid]} M={m} sort={sp[0]:.1f} gather={sp[1]:.1f} gemm={sp[2]:.1f} "
                f"gemm2={sp[3]:.1f} swiglu={sp[4]:.1f} reduce={sp[5]:.1f} total={sp[6]:.1f} cols={int(sp[7]) // 1000} mt={int(sp[7]) % 1000}"
            )
            del x, q, raw
            torch.xpu.empty_cache()
        except RuntimeError as exc:
            failures += 1
            print(f"FAIL {kind} {NAMES.get(tid, tid)} M={m} N={n} K={k} {exc}")
    # Full expert layer. Weight bytes are gate+up+down, read once per column-group.
    layer_ts = [512] if smoke else [512, 4096, 12288]
    h, f, e, topk = 2560, 640, 512, 10
    gu_t, dn_t = 18, 42
    try:
        g = synth_pack(gu_t, e * f, h, 7)
        u = synth_pack(gu_t, e * f, h, 8)
        d = synth_pack(dn_t, e * h, f, 9)
        gq = torch.from_numpy(g).to(device="xpu")
        uq = torch.from_numpy(u).to(device="xpu")
        dq = torch.from_numpy(d).to(device="xpu")
        nbytes = e * f * row_bytes(gu_t, h) * 2 + e * h * row_bytes(dn_t, f)
        for t in layer_ts:
            ids = torch.randint(0, e, (t, topk), device="xpu", dtype=torch.int32)
            x = torch.randn(t, h, device="xpu", dtype=torch.float16)
            router = torch.rand(t, topk, device="xpu", dtype=torch.float32)
            fn = lambda: prefill_moe_layer(x, ids, gq, uq, dq, router, gu_t, dn_t, h, f, 0)
            iters = 4 if t >= 12288 else 8
            line = bench_one(f"layer\tIQ3_XXS+Q2_0\tT={t} E={e} topk={topk} H={h} F={f}", fn, nbytes, warmup=1, iters=iters)
            us = float(line.split("\t")[-3])
            tok_s = t / (us * 1e-6) if us > 0 else 0.0
            # 48 layers at this expert time, attention not included.
            layer48_s = us * 48 * 1e-6
            e2e = t / layer48_s if layer48_s > 0 else 0.0
            print(f"{line}\ttok/s_one_layer={tok_s:.1f}\texpert_only_48layer_tok/s={e2e:.1f}")
            os.environ["B70_PREFILL_SPLIT"] = "1"
            fn()
            torch.xpu.synchronize()
            sp = prefill_split_us().tolist()
            os.environ["B70_PREFILL_SPLIT"] = "0"
            print(
                f"SPLIT layer T={t} sort={sp[0]:.1f} gather={sp[1]:.1f} gu={sp[2]:.1f} swiglu={sp[4]:.1f} "
                f"down={sp[3]:.1f} reduce={sp[5]:.1f} total={sp[6]:.1f} cols={int(sp[7]) // 1000} mt={int(sp[7]) % 1000}"
            )
            del x, ids, router
            torch.xpu.empty_cache()
        if not smoke and context_alive():
            # bf16 and int8 at T=512 only, same weights. One rejected mode
            # must not erase the fp16 numbers already printed.
            t = 512
            ids = torch.randint(0, e, (t, topk), device="xpu", dtype=torch.int32)
            xb = torch.randn(t, h, device="xpu", dtype=torch.bfloat16)
            x16 = xb.to(torch.float16)
            router = torch.rand(t, topk, device="xpu", dtype=torch.float32)
            for mode, xm, tag in ((1, xb, "bf16"), (2, x16, "int8")):
                try:
                    fn = lambda xm=xm, mode=mode: prefill_moe_layer(xm, ids, gq, uq, dq, router, gu_t, dn_t, h, f, mode)
                    line = bench_one(f"layer_{tag}\tIQ3_XXS+Q2_0\tT={t} E={e} H={h} F={f}", fn, nbytes, warmup=1, iters=4)
                    print(line)
                except RuntimeError as exc:
                    failures += 1
                    print(f"FAIL layer_{tag} {exc}")
                    if not context_alive():
                        print(f"CONTEXT dead after layer_{tag}")
                        break
    except RuntimeError as exc:
        failures += 1
        print(f"FAIL layer bench {exc}")
    return failures


def main() -> int:
    smoke = os.environ.get("B70_PREFILL_SMOKE") == "1"
    bench = "--bench" in sys.argv
    bench_only = "--bench-only" in sys.argv
    assert torch.xpu.is_available(), "XPU not available"
    assert torch.xpu.device_count() == 1, f"expected 1 visible device, got {torch.xpu.device_count()}"
    print(f"DEVICE {torch.xpu.get_device_name(0)} count={torch.xpu.device_count()} smoke={int(smoke)}")
    a = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
    (a @ b).float().sum().item()
    torch.xpu.synchronize()
    print("PROBE matmul ok")

    failures = 0
    if not bench_only:
        lib = load_cpu()
        if not Path(SHARD).is_file():
            print(f"MISSING {SHARD}")
            return 1
        print(f"OPEN {SHARD}")
        reader = gguf.GGUFReader(SHARD)
        by_type = {}
        named = {}
        for tensor in reader.tensors:
            tid = int(tensor.tensor_type)
            named.setdefault(tensor.name, tensor)
            if tid not in by_type and tid in WANTED:
                by_type[tid] = tensor
        wanted = [t for t in WANTED if not smoke or t in SMOKE_TYPES]
        kept = []
        for tid in wanted:
            tensor = by_type.get(tid)
            if tensor is None:
                print(f"MISSING_TYPE {NAMES[tid]}")
                failures += 1
                continue
            flat, k = flat_matrix(tensor)
            raw = take_rows(flat, k, tid, 128)
            if raw is None:
                print(f"SKIP {NAMES[tid]} {tensor.name} shape={tensor.shape} k={k} rows={flat.shape[0]}")
                failures += 1
                continue
            print(f"USE {NAMES[tid]} {tensor.name} rows={raw.shape[0]} k={k}")
            try:
                failures = check_dense(lib, tid, raw, k, failures)
                failures = check_grouped(lib, tid, raw, k, failures)
            except RuntimeError as exc:
                failures += 1
                print(f"FAIL {NAMES[tid]} fp16 EXC {exc}")
            kept.append((tid, raw, k))
        gate = up = down = None
        for tensor in reader.tensors:
            if "ffn_gate_exps" in tensor.name and gate is None:
                gate = tensor
            elif "ffn_up_exps" in tensor.name and up is None:
                up = tensor
            elif "ffn_down_exps" in tensor.name and down is None:
                down = tensor
        if gate is not None and up is not None and down is not None and not smoke:
            try:
                failures = check_layer(lib, gate, up, down, failures)
            except RuntimeError as exc:
                failures += 1
                print(f"FAIL layer EXC {exc}")
        elif not smoke:
            print("SKIP layer (no ffn_*_exps in shard 1)")
            shown = 0
            for tensor in reader.tensors:
                if tensor.data.ndim >= 3:
                    print(f"TENSOR3 {tensor.name} type={int(tensor.tensor_type)} shape={tensor.shape}")
                    shown += 1
                    if shown >= 12:
                        break
        del reader
        if not smoke and kept and context_alive():
            dead_modes = set()
            for tid, raw, k in kept:
                if dead_modes >= {1, 2}:
                    break
                failures = check_eval(lib, tid, raw, k, failures, dead_modes)
        elif not smoke and kept:
            failures += 1
            print("FAIL eval skipped; context dead after fp16")
    if bench or bench_only:
        failures += run_bench(smoke)
    print(f"VERDICT failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
