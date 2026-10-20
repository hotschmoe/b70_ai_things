"""Device time of moe_layer on a Strata fair dump. One process, one B70_MOE_SCHED value.

The .so latches B70_MOE_SCHED at first use. Graph replay is 48 calls cycling 8
weight copies, matching the Strata bench's layer count. us_graph is per call.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b70_gguf_xpu import moe_layer  # noqa: E402
from b70_gguf_xpu import ops  # noqa: E402


def load_dump(path: Path) -> dict:
    b = path.read_bytes()
    o = 0

    def u32():
        nonlocal o
        v = struct.unpack_from("<I", b, o)[0]
        o += 4
        return v

    def i32():
        nonlocal o
        v = struct.unpack_from("<i", b, o)[0]
        o += 4
        return v

    def take(n):
        nonlocal o
        sl = b[o : o + n]
        o += n
        return sl

    magic = u32()
    if magic != 0x31463943:
        raise RuntimeError(f"bad magic {magic:#x} in {path}")
    T, I, K, topk, gu, dt, E, ng, nent, shared, grb, urb, drb = [i32() for _ in range(13)]
    ids = np.frombuffer(take(T * topk * 4), dtype=np.int32).copy()
    router = np.frombuffer(take(T * topk * 4), dtype=np.float32).copy()
    x = np.frombuffer(take(T * K * 4), dtype=np.float32).copy()
    gate = np.frombuffer(take(E * I * grb), dtype=np.uint8).copy()
    up = np.frombuffer(take(E * I * urb), dtype=np.uint8).copy()
    down = np.frombuffer(take(E * K * drb), dtype=np.uint8).copy()
    sgate = np.frombuffer(take(I * grb), dtype=np.uint8).copy() if shared else None
    sup = np.frombuffer(take(I * urb), dtype=np.uint8).copy() if shared else None
    sdown = np.frombuffer(take(K * drb), dtype=np.uint8).copy() if shared else None
    y = np.frombuffer(take(T * K * 4), dtype=np.float32).copy()
    if o != len(b):
        raise RuntimeError(f"trailing {len(b) - o} bytes in {path}")
    return dict(T=T, I=I, K=K, topk=topk, gu=gu, dt=dt, E=E, ng=ng, nent=nent, shared=shared, grb=grb, urb=urb,
                drb=drb, ids=ids, router=router, x=x, gate=gate, up=up, down=down, sgate=sgate, sup=sup, sdown=sdown, y=y)


def to_xpu(a: np.ndarray, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device="xpu")


def graph_us(fn, calls: int, reps: int) -> float:
    for _ in range(2):
        fn()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(3):
        graph.replay()
    torch.xpu.synchronize()
    samples = []
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    for _ in range(5):
        start.record()
        for _r in range(reps):
            graph.replay()
        end.record()
        torch.xpu.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / (reps * calls))
    samples.sort()
    return samples[len(samples) // 2]


def run_one(d: dict, reps: int) -> None:
    T, I, K = d["T"], d["I"], d["K"]
    x = to_xpu(d["x"].reshape(T, K), torch.bfloat16)
    ids = to_xpu(d["ids"].reshape(T, d["topk"]))
    router = to_xpu(d["router"].reshape(T, d["topk"]))
    gates = [to_xpu(d["gate"]) for _ in range(8)]
    ups = [to_xpu(d["up"]) for _ in range(8)]
    downs = [to_xpu(d["down"]) for _ in range(8)]
    sg = to_xpu(d["sgate"]) if d["sgate"] is not None else None
    su = to_xpu(d["sup"]) if d["sup"] is not None else None
    sd = to_xpu(d["sdown"]) if d["sdown"] is not None else None
    # Eight distinct allocations. clone() after the first H2D so the copies differ in address.
    for i in range(1, 8):
        gates[i] = gates[0].clone()
        ups[i] = ups[0].clone()
        downs[i] = downs[0].clone()
    step = {"i": 0}

    def call():
        i = step["i"]
        step["i"] = i + 1
        return moe_layer(
            x, gates[i % 8], d["gu"], ups[i % 8], d["gu"], downs[i % 8], d["dt"], ids, router, I, K, sg, d["gu"], su,
            d["gu"], sd, d["dt"],
        )

    y = call()
    torch.xpu.synchronize()
    got = y.float().cpu().numpy()
    ref = d["y"]
    abs_err = np.max(np.abs(got - ref))
    denom = np.maximum(np.abs(ref), 1e-8)
    rel = float(np.max(np.abs(got - ref) / denom))
    step["i"] = 0
    held: list = []

    def call_hold():
        held.append(call())

    us = graph_us(call_hold, 48, reps)
    sched = os.environ.get("B70_MOE_SCHED", "1")
    print(
        f"OURS T={T} gu={d['gu']} dt={d['dt']} sched={sched} us_graph={us:.2f} us_per_tok={us / T:.2f} "
        f"max_abs_vs_strata={abs_err:.4e} max_rel={rel:.4e}",
        flush=True,
    )


def main() -> int:
    dump = Path(os.environ.get("FAIR_OUT", "/mnt/vm_8tb/b70/build/vllm-flashnext/c9/fair_dump"))
    reps = int(os.environ.get("C9_REPS", "4"))
    ops.load()
    if hasattr(torch.ops.b70_gguf, "reserve_decode"):
        torch.ops.b70_gguf.reserve_decode()
        torch.xpu.synchronize()
    files = sorted(dump.glob("fair_T*_gu*_dt*.bin"))
    if not files:
        print("NO_DUMPS", dump)
        return 1
    for f in files:
        run_one(load_dump(f), reps)
    print("OURS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
