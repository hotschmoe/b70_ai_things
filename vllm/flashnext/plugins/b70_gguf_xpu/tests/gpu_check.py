"""XPU correctness and latency for b70_gguf. Run via scripts/gpu_test.sh under gpu-run --card 0."""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
sys.path.insert(0, str(ROOT))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

import torch  # noqa: E402

from b70_gguf_xpu import dequant_bf16, gemv, grouped_gemm, moe_gemv  # noqa: E402
from b70_gguf_xpu.types import NAMES  # noqa: E402

SHARD = (
    "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)
ROOF_GBPS = 608.0
WANTED = [10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 42]


def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r = u + 0x7FFF + ((u >> 16) & 1)
    bits = (r >> 16).astype(np.uint16)
    bits = np.where(nan, ((u >> 16) | 64).astype(np.uint16), bits)
    return bits


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


def flat_matrix(tensor):
    data = np.asarray(tensor.data)
    k = int(tensor.shape[0])
    flat = np.ascontiguousarray(data.reshape(-1, data.shape[-1]))
    return flat, k


def expert_pack(tensor):
    data = np.asarray(tensor.data)
    k = int(tensor.shape[0])
    if data.ndim < 3:
        return None
    packed = np.ascontiguousarray(data.reshape(-1, data.shape[-2], data.shape[-1]))
    return packed, k


def close_enough(got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> tuple[bool, float, float]:
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    scale = np.maximum(np.abs(ref.astype(np.float64)), 1e-6)
    rel = d / scale
    mx = float(d.max()) if d.size else 0.0
    mr = float(rel.max()) if rel.size else 0.0
    ok = bool(np.all(d <= (atol + rtol * scale)))
    return ok, mx, mr


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


def elapsed_behind(fn, iters: int = 4) -> float:
    # Device-side span with a matmul already queued, so the tag is not the
    # idle host-submit gap. Same clock as elapsed_us (submit_profiling_tag).
    burn_a = torch.randn(1024, 1024, device="xpu")
    burn_b = torch.randn(1024, 1024, device="xpu")
    for _ in range(2):
        (burn_a @ burn_b).sum()
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        (burn_a @ burn_b).sum()
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return min(samples)


def main() -> int:
    assert torch.xpu.is_available(), "XPU not available"
    assert torch.xpu.device_count() == 1, f"expected 1 visible device, got {torch.xpu.device_count()}"
    print(f"DEVICE {torch.xpu.get_device_name(0)} count={torch.xpu.device_count()}")
    a = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
    (a @ b).float().sum().item()
    torch.xpu.synchronize()
    print("PROBE matmul ok")

    lib = load_cpu()
    paths = [SHARD]
    dev = Path("/mnt/cache/b70/devmodel")
    if dev.is_dir():
        paths.extend(str(p) for p in sorted(dev.rglob("*IQ3_XXS*.gguf")) if ".cache" not in str(p))

    failures = 0
    bench_rows = []
    seen = set()
    for path in paths:
        if not Path(path).is_file():
            print(f"MISSING {path}")
            continue
        print(f"OPEN {path}")
        reader = gguf.GGUFReader(path)
        chosen = {}
        for tensor in reader.tensors:
            tid = int(tensor.tensor_type)
            if tid not in NAMES or tid in chosen:
                continue
            chosen[tid] = tensor
        for tid in WANTED:
            if tid in seen or tid not in chosen:
                continue
            tensor = chosen[tid]
            name = NAMES[tid]
            flat, k = flat_matrix(tensor)
            n_take = min(64, flat.shape[0])
            n_take = n_take - (n_take % 64) or min(64, flat.shape[0])
            if n_take < 4:
                print(f"SKIP {name} only {flat.shape[0]} rows")
                continue
            # grouped wants a multiple of 64; gemv can use the same tile
            if n_take % 64 != 0:
                n_gemv = min(32, flat.shape[0])
            else:
                n_gemv = n_take
            raw = np.array(flat[:n_gemv], dtype=np.uint8, copy=True)
            ref = ref_dequant(tid, raw, k, lib)
            q = torch.from_numpy(raw).to(device="xpu")
            got_bf = dequant_bf16(q, tid, n_gemv, k)
            torch.xpu.synchronize()
            got_bits = got_bf.view(torch.int16).cpu().numpy().view(np.uint16)
            exp_bits = f32_to_bf16_bits(ref)
            bad = int(np.count_nonzero(got_bits.reshape(-1) != exp_bits.reshape(-1)))
            status = "OK" if bad == 0 else "FAIL"
            if bad:
                failures += 1
            print(f"{status} dequant {name} {tensor.name} rows={n_gemv} k={k} bf16_mismatches={bad}")

            rng = np.random.default_rng(0)
            for m in (1, 3, 8):
                x_np = rng.standard_normal((m, k)).astype(np.float32)
                x = torch.from_numpy(x_np).to(device="xpu", dtype=torch.bfloat16)
                y = gemv(x, q, tid, n_gemv, k)
                torch.xpu.synchronize()
                y_np = y.cpu().numpy()
                x_ref = x.float().cpu().numpy()
                ref_y = x_ref @ ref.T
                # Every GEMV quantizes x to Q8_1 (KREG or the row DP4A). A
                # swapped lane still misses by a large fraction of y.
                ok, mx, mr = close_enough(y_np, ref_y, atol=5e-2, rtol=5e-2)
                status = "OK" if ok else "FAIL"
                if not ok:
                    failures += 1
                print(f"{status} gemv {name} M={m} N={n_gemv} K={k} max_abs={mx:.3e} max_rel={mr:.3e}")

            if n_gemv % 64 == 0:
                xg = torch.randn(16, k, device="xpu", dtype=torch.float16)
                entries = torch.tensor([[0, 0, 16]], dtype=torch.int32, device="xpu")
                yg = grouped_gemm(xg, q, entries, tid, n_gemv, k)
                torch.xpu.synchronize()
                w = ref
                ref_y = xg.float().cpu().numpy() @ w.T
                ok, mx, mr = close_enough(yg.cpu().numpy(), ref_y, atol=5e-2, rtol=5e-2)
                status = "OK" if ok else "FAIL"
                if not ok:
                    failures += 1
                print(f"{status} grouped {name} M=16 N={n_gemv} K={k} max_abs={mx:.3e} max_rel={mr:.3e}")

            pack = expert_pack(tensor)
            if pack is not None:
                packed, k2 = pack
                e_take = min(4, packed.shape[0])
                n_exp = min(64, packed.shape[1])
                n_exp -= n_exp % 64
                if e_take >= 2 and n_exp >= 64:
                    blob = np.ascontiguousarray(packed[:e_take, :n_exp].reshape(-1))
                    qe = torch.from_numpy(blob).to(device="xpu")
                    ids = np.array([[0, 2 % e_take], [1 % e_take, 0]], np.int32)
                    ide = torch.from_numpy(ids).to(device="xpu")
                    x = torch.randn(2, k2, device="xpu", dtype=torch.bfloat16)
                    y = moe_gemv(x, qe, ide, tid, n_exp, k2)
                    torch.xpu.synchronize()
                    y_np = y.cpu().numpy()
                    x_ref = x.float().cpu().numpy()
                    moe_ok = True
                    worst = 0.0
                    for t in range(2):
                        for s in range(2):
                            rows = np.ascontiguousarray(packed[ids[t, s], :n_exp])
                            w = cpu_dequant(lib, tid, rows, k2)
                            ref_y = x_ref[t : t + 1] @ w.T
                            ok, mx, mr = close_enough(y_np[t, s : s + 1], ref_y, atol=5e-2, rtol=5e-2)
                            moe_ok = moe_ok and ok
                            worst = max(worst, mx)
                    status = "OK" if moe_ok else "FAIL"
                    if not moe_ok:
                        failures += 1
                    print(f"{status} moe {name} E={e_take} N={n_exp} K={k2} max_abs={worst:.3e}")

                    xg = torch.randn(32, k2, device="xpu", dtype=torch.float16)
                    entries = torch.tensor([[0, 0, 16], [1 % e_take, 16, 16]], dtype=torch.int32, device="xpu")
                    yg = grouped_gemm(xg, qe, entries, tid, n_exp, k2)
                    torch.xpu.synchronize()
                    yg_np = yg.cpu().numpy()
                    xg_ref = xg.float().cpu().numpy()
                    g_ok = True
                    g_worst = 0.0
                    for expert, row0, nrows in ((0, 0, 16), (1 % e_take, 16, 16)):
                        w = cpu_dequant(lib, tid, np.ascontiguousarray(packed[expert, :n_exp]), k2)
                        ref_y = xg_ref[row0 : row0 + nrows] @ w.T
                        ok, mx, mr = close_enough(yg_np[row0 : row0 + nrows], ref_y, atol=5e-2, rtol=5e-2)
                        g_ok = g_ok and ok
                        g_worst = max(g_worst, mx)
                    status = "OK" if g_ok else "FAIL"
                    if not g_ok:
                        failures += 1
                    print(f"{status} grouped {name} M=32 N={n_exp} K={k2} max_abs={g_worst:.3e}")
            seen.add(tid)
        # Bench the big tensors once, from the Flash-Next shard only.
        if path == SHARD:
            bench_rows.extend(run_bench(reader, lib))
        del reader

    missing = [NAMES[t] for t in WANTED if t not in seen]
    if missing:
        print("MISSING_TYPES " + ",".join(missing))
        failures += len(missing)
    print("BENCH")
    print("op\ttype\tshape\tus\tGB/s\tpct_of_608")
    for row in bench_rows:
        print(row)
    print(f"VERDICT failures={failures}")
    return 1 if failures else 0


def run_bench(reader, lib) -> list[str]:
    rows = []
    # Pick the heaviest 2D tensor per type plus one expert pack per type.
    best2 = {}
    best3 = {}
    for tensor in reader.tensors:
        tid = int(tensor.tensor_type)
        if tid not in NAMES:
            continue
        data = tensor.data
        nbytes = int(np.asarray(data).size)
        if data.ndim == 2 and nbytes > best2.get(tid, (0, None))[0]:
            best2[tid] = (nbytes, tensor)
        if data.ndim >= 3 and nbytes > best3.get(tid, (0, None))[0]:
            best3[tid] = (nbytes, tensor)

    def time_gemv(tid, tensor, m):
        flat, k = flat_matrix(tensor)
        n = flat.shape[0]
        # Cap the output head-sized GEMV at the real tensor; it is the point of the bench.
        raw = np.ascontiguousarray(flat)
        q = torch.from_numpy(raw).to(device="xpu")
        x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
        us = elapsed_us(lambda: gemv(x, q, tid, n, k))
        weight = n * raw.shape[1]
        gbps = (weight / 1e9) / (us * 1e-6)
        rows.append(
            f"gemv\t{NAMES[tid]}\tM={m} N={n} K={k} bytes={weight}\t{us:.1f}\t{gbps:.1f}\t{100.0 * gbps / ROOF_GBPS:.1f}"
        )
        if m == 1 and tid in (12, 14, 21, 23):
            bus = elapsed_behind(lambda: gemv(x, q, tid, n, k))
            bgbps = (weight / 1e9) / (bus * 1e-6)
            rows.append(
                f"gemv_behind\t{NAMES[tid]}\tM={m} N={n} K={k} bytes={weight}\t{bus:.1f}\t{bgbps:.1f}\t{100.0 * bgbps / ROOF_GBPS:.1f}"
            )
        del q, x

    for tid, (_n, tensor) in best2.items():
        # Skip tiny rows; bench tensors of at least 8 MiB, always include output head.
        flat, _k = flat_matrix(tensor)
        if flat.nbytes < 8 << 20 and "output" not in tensor.name:
            continue
        for m in (1, 8):
            print(f"BENCH gemv {NAMES[tid]} {tensor.name} M={m} rows={flat.shape[0]}")
            time_gemv(tid, tensor, m)

    for tid, (_n, tensor) in best3.items():
        packed, k = expert_pack(tensor)
        e, n, rb = packed.shape
        # One token, top-10, full expert rows. Cap E used by the ids, weights stay the full pack
        # only if it fits; use the first 32 experts so the copy stays under a few hundred MB
        # while the indexed experts are real.
        e_use = min(e, 32)
        blob = np.ascontiguousarray(packed[:e_use])
        q = torch.from_numpy(blob.reshape(-1)).to(device="xpu")
        ids = torch.randint(0, e_use, (1, 10), device="xpu", dtype=torch.int32)
        x = torch.randn(1, k, device="xpu", dtype=torch.bfloat16)
        us = elapsed_us(lambda: moe_gemv(x, q, ids, tid, n, k))
        weight = 10 * n * rb
        gbps = (weight / 1e9) / (us * 1e-6)
        rows.append(
            f"moe_gemv\t{NAMES[tid]}\tT=1 topk=10 E={e_use} N={n} K={k} bytes={weight}\t{us:.1f}\t{gbps:.1f}\t{100.0 * gbps / ROOF_GBPS:.1f}"
        )
        # Prefill grouped: 128 rows, one expert, full N (must be % 64).
        if n % 64 == 0:
            xg = torch.randn(128, k, device="xpu", dtype=torch.float16)
            entries = torch.tensor([[0, 0, 128]], dtype=torch.int32, device="xpu")
            us = elapsed_us(lambda: grouped_gemm(xg, q, entries, tid, n, k))
            # launch_grouped kMt=16 covers 128 activation rows per weight read.
            unique = n * rb
            issued = unique * ((128 + 127) // 128)
            gbps_u = (unique / 1e9) / (us * 1e-6)
            gbps_i = (issued / 1e9) / (us * 1e-6)
            rows.append(
                f"grouped\t{NAMES[tid]}\tM=128 N={n} K={k} unique={unique} issued={issued}\t{us:.1f}\t{gbps_u:.1f} unique / {gbps_i:.1f} issued\t{100.0 * gbps_i / ROOF_GBPS:.1f} issued"
            )
            # Fallback probe: dequant expert 0 to bf16, then the XPU matmul.
            # mm is the GEMM alone; dequant_mm repeats the dequant inside the timed call.
            q0 = q[: n * rb].contiguous()
            w = dequant_bf16(q0, tid, n, k)
            wt = w.transpose(0, 1).contiguous()
            xbf = xg.to(torch.bfloat16)
            us_mm = elapsed_us(lambda: torch.mm(xbf, wt))
            us_both = elapsed_us(lambda: torch.mm(xbf, dequant_bf16(q0, tid, n, k).transpose(0, 1)))
            dense = n * k * 2
            rows.append(
                f"xpu_mm\t{NAMES[tid]}\tM=128 N={n} K={k} bf16_bytes={dense}\t{us_mm:.1f}\t{(dense / 1e9) / (us_mm * 1e-6):.1f}\t{100.0 * (dense / 1e9) / (us_mm * 1e-6) / ROOF_GBPS:.1f}"
            )
            rows.append(
                f"dequant_mm\t{NAMES[tid]}\tM=128 N={n} K={k} packed={unique}\t{us_both:.1f}\t{(unique / 1e9) / (us_both * 1e-6):.1f}\t{100.0 * (unique / 1e9) / (us_both * 1e-6) / ROOF_GBPS:.1f}"
            )
        del q
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
