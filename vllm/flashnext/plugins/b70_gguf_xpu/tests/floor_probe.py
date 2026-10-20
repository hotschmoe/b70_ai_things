"""Is the ~122 us moe_gemv number a host/timer floor?

A2 times one call between two torch.xpu events and syncs every iteration.
On an idle in-order queue that interval includes the host gap after the
start tag. This probe reports:

- a2: that method (min of 10)
- batch: one event pair around N back-to-back calls, divided by N
- wall: host perf_counter around N calls, divided by N
- behind: the same pair, but queued behind a long matmul so the tags
  sit in the queue before the GPU reaches them (device time)

Run only via scripts/gpu_test.sh style env, under bin/gpu-run --card 0.
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

import gguf  # noqa: E402
import torch  # noqa: E402

from b70_gguf_xpu import gemv, moe_gemv  # noqa: E402
from b70_gguf_xpu.types import NAMES  # noqa: E402

SHARD = (
    "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)


def a2_us(fn, warmup=3, iters=10) -> float:
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


def batch_us(fn, n=32, warmup=2) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.xpu.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / n


def wall_us(fn, n=32, warmup=2) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n


def behind_us(fn, n=4, preamble=2) -> float:
    """Queue fn behind a fat matmul so the event tags are already submitted."""
    a = torch.randn(4096, 4096, device="xpu")
    b = torch.randn(4096, 4096, device="xpu")
    for _ in range(1):
        (a @ b).sum().item()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    for _ in range(preamble):
        _ = a @ b
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.xpu.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / n


def report(label, fn, bytes_touched=0):
    u_a2 = a2_us(fn)
    u_batch = batch_us(fn)
    u_wall = wall_us(fn)
    u_behind = behind_us(fn)
    gbps = (bytes_touched / 1e9) / (u_behind * 1e-6) if bytes_touched and u_behind > 0 else 0.0
    print(
        f"{label}\ta2={u_a2:.1f}\tbatch32={u_batch:.1f}\twall32={u_wall:.1f}\t"
        f"behind={u_behind:.1f}\tbehind_GBs={gbps:.1f}\tbytes={bytes_touched}",
        flush=True,
    )


def main() -> int:
    assert torch.xpu.is_available()
    print(f"DEVICE {torch.xpu.get_device_name(0)} count={torch.xpu.device_count()}", flush=True)
    print(f"ENV immediate={os.environ.get('SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS', '')}", flush=True)
    x = torch.zeros(1, device="xpu")
    report("empty_add1", lambda: x.add(1))
    a = torch.randn(64, 64, device="xpu")
    b = torch.randn(64, 64, device="xpu")
    report("mm_64", lambda: a @ b)
    big = torch.randn(2048, 2048, device="xpu")
    report("mm_2048", lambda: big @ big)

    reader = gguf.GGUFReader(SHARD, "r")
    # One expert tensor per (type, kind) plus the big dense rows. K is the
    # ggml ne[0] (tensor.shape[0]), matching gpu_check.
    seen = set()
    jobs = []
    for tensor in reader.tensors:
        name = tensor.name
        tid = int(tensor.tensor_type)
        if tid not in NAMES:
            continue
        data = np.asarray(tensor.data)
        k = int(tensor.shape[0])
        if data.ndim >= 3 and any(s in name for s in ("ffn_gate_exps", "ffn_down_exps", "ffn_up_exps")):
            kind = "gate" if "gate" in name else "down" if "down" in name else "up"
            key = (tid, kind)
            if key in seen:
                continue
            seen.add(key)
            jobs.append((tid, kind, name, k, data))
        elif data.ndim == 2 and ("output.weight" in name or "token_embd" in name):
            jobs.append((tid, "dense", name, k, data))
        if len(jobs) >= 8:
            break

    for tid, kind, name, k, data in jobs:
        if kind == "dense":
            flat = np.ascontiguousarray(data.reshape(-1, data.shape[-1]))
            n = flat.shape[0]
            n_use = n if n <= 32768 else 32768
            raw = np.ascontiguousarray(flat[:n_use])
            q = torch.from_numpy(raw).to(device="xpu")
            xbf = torch.randn(1, k, device="xpu", dtype=torch.bfloat16)
            weight = n_use * raw.shape[1]
            report(
                f"gemv {NAMES[tid]} {name} N={n_use} K={k}",
                lambda q=q, xbf=xbf, tid=tid, n_use=n_use, k=k: gemv(xbf, q, tid, n_use, k),
                weight,
            )
            del q, xbf
            continue
        packed = np.ascontiguousarray(data.reshape(-1, data.shape[-2], data.shape[-1]))
        e, n, rb = packed.shape
        e_use = min(int(e), 32)
        blob = np.ascontiguousarray(packed[:e_use]).reshape(-1)
        q = torch.from_numpy(blob).to(device="xpu")
        ids = torch.randint(0, e_use, (1, 10), device="xpu", dtype=torch.int32)
        xbf = torch.randn(1, k, device="xpu", dtype=torch.bfloat16)
        weight = 10 * n * rb
        report(
            f"moe {NAMES[tid]} {kind} {name} E={e_use} N={n} K={k}",
            lambda q=q, xbf=xbf, ids=ids, tid=tid, n=n, k=k: moe_gemv(xbf, q, ids, tid, n, k),
            weight,
        )
        del q, xbf, ids
    print("FLOOR_PROBE_OK", flush=True)
    return 0


if __name__ == "__main__":
    # Re-bind K from the GGUF tensor, not the ndarray, inside main's loop.
    raise SystemExit(main())
