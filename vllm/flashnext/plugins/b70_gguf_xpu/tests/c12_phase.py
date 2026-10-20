"""Eager per-phase times for the fair dumps. Not a graph measurement.

B70_MOE_PHASE=1 makes moe_layer print C12_PHASE (submit+wait, serialized).
Do not point this at a graph capture. us_a2 is a separate synchronized event.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from c9_fair import load_dump, to_xpu  # noqa: E402
from b70_gguf_xpu import moe_layer  # noqa: E402
from b70_gguf_xpu import ops  # noqa: E402


def one(d: dict) -> None:
    T, I, K = d["T"], d["I"], d["K"]
    x = to_xpu(d["x"].reshape(T, K), torch.bfloat16)
    ids = to_xpu(d["ids"].reshape(T, d["topk"]))
    router = to_xpu(d["router"].reshape(T, d["topk"]))
    gate = to_xpu(d["gate"])
    up = to_xpu(d["up"])
    down = to_xpu(d["down"])
    sg = to_xpu(d["sgate"]) if d["sgate"] is not None else None
    su = to_xpu(d["sup"]) if d["sup"] is not None else None
    sd = to_xpu(d["sdown"]) if d["sdown"] is not None else None

    def call():
        return moe_layer(
            x, gate, d["gu"], up, d["gu"], down, d["dt"], ids, router, I, K,
            sg, d["gu"], su, d["gu"], sd, d["dt"],
        )

    for _ in range(2):
        call()
    torch.xpu.synchronize()
    samples = []
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    for _ in range(3):
        start.record()
        call()
        end.record()
        torch.xpu.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    samples.sort()
    print(
        f"PHASE_A2 T={T} gu={d['gu']} dt={d['dt']} us={samples[0]:.2f}",
        flush=True,
    )


def main() -> int:
    os.environ.setdefault("B70_MOE_PHASE", "1")
    dump = Path(os.environ.get("FAIR_OUT", "/mnt/vm_8tb/b70/build/vllm-flashnext/c9/fair_dump"))
    ops.load()
    if hasattr(torch.ops.b70_gguf, "reserve_decode"):
        torch.ops.b70_gguf.reserve_decode()
        torch.xpu.synchronize()
    files = sorted(dump.glob("fair_T*_gu*_dt*.bin"))
    if not files:
        print("NO_DUMPS", dump)
        return 1
    only = os.environ.get("C12_PHASE_GLOB", "")
    for f in files:
        if only and only not in f.name:
            continue
        one(load_dump(f))
    print("PHASE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
