"""O3 stage 3: moe_layer_sh (routed + shared expert + sigmoid shared gate, one op).

Correctness against moe_layer on the C9 fair dumps:
  routed  = moe_layer(no shared)
  shared  = moe_layer(with shared, weight 1) - routed
  expect  y2[0] ~ routed, y2[1] ~ sigmoid(x . w_sg) * shared   (bf16 outputs)
Then XPUGraph time per layer (48 calls, 8 weight copies) of
  before: moe_layer(no shared) + unfused shared MLP (gemv gate, gemv up, silu*up,
          gemv down, F.linear gate, sigmoid, mul) + add, all in one graph
  after:  moe_layer_sh + add
Run under bin/gpu-run --card 0.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from b70_gguf_xpu import ops  # noqa: E402
from c9_fair import graph_us, load_dump, to_xpu  # noqa: E402


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def run(path: Path, reps: int) -> int:
    d = load_dump(path)
    T, I, K, gu, dt = d["T"], d["I"], d["K"], d["gu"], d["dt"]
    x = to_xpu(d["x"].reshape(T, K), torch.bfloat16)
    ids = to_xpu(d["ids"].reshape(T, d["topk"]))
    rw = to_xpu(d["router"].reshape(T, d["topk"]))
    gate, up, down = to_xpu(d["gate"]), to_xpu(d["up"]), to_xpu(d["down"])
    sg, su, sd = to_xpu(d["sgate"]), to_xpu(d["sup"]), to_xpu(d["sdown"])
    g = torch.Generator().manual_seed(7)
    fails = 0
    for wdt in (torch.bfloat16, torch.float32):
        wsg = (torch.randn(K, generator=g) * 0.02).to(device="xpu", dtype=wdt)
        routed = ops.moe_layer(x, gate, gu, up, gu, down, dt, ids, rw, I, K)
        full = ops.moe_layer(x, gate, gu, up, gu, down, dt, ids, rw, I, K, sg, gu, su, gu, sd, dt)
        shared = full - routed
        sw = torch.sigmoid(x.float() @ wsg.float())  # [T]
        y2 = ops.moe_layer_sh(x, gate, gu, up, gu, down, dt, ids, rw, I, K, sg, gu, su, gu, sd, dt, wsg)
        y1 = ops.moe_layer_sh(x, gate, gu, up, gu, down, dt, ids, rw, I, K, sg, gu, su, gu, sd, dt, None)
        torch.xpu.synchronize()
        c0 = cos(y2[0], routed)
        c1 = cos(y2[1], sw[:, None] * shared)
        c2 = cos(y1[1], shared)
        e0 = float((y2[0].float() - routed).abs().max() / routed.abs().max())
        e1 = float((y2[1].float() - sw[:, None] * shared).abs().max() / shared.abs().max())
        ok = min(c0, c1, c2) >= 0.9999 and max(e0, e1) < 1e-2
        fails += 0 if ok else 1
        print(f"{'OK' if ok else 'FAIL'} T={T} gu={gu} dt={dt} wsg={str(wdt).split('.')[-1]} "
              f"cos_routed={c0:.6f} cos_shared={c1:.6f} cos_shared_w1={c2:.6f} relmax_r={e0:.2e} relmax_s={e1:.2e}",
              flush=True)

    # timing, bf16 gate input like the 35B / Flash-Next model weight after load
    wsg = (torch.randn(K, generator=g) * 0.02).to(device="xpu", dtype=torch.bfloat16)
    gates = [gate.clone() for _ in range(8)]
    ups = [up.clone() for _ in range(8)]
    downs = [down.clone() for _ in range(8)]
    sgs = [sg.clone() for _ in range(8)]
    sus = [su.clone() for _ in range(8)]
    sds = [sd.clone() for _ in range(8)]
    st = {"i": 0}
    held: list = []

    def before():
        i = st["i"] % 8
        st["i"] += 1
        r = ops.moe_layer(x, gates[i], gu, ups[i], gu, downs[i], dt, ids, rw, I, K).to(torch.bfloat16)
        hg = ops.gemv(x, sgs[i], gu, I, K)
        hu = ops.gemv(x, sus[i], gu, I, K)
        h = (F.silu(hg) * hu).to(torch.bfloat16)
        o = ops.gemv(h, sds[i], dt, K, I).to(torch.bfloat16)
        gt = F.linear(x, wsg[None, :])
        held.append(r + torch.sigmoid(gt) * o)

    def after():
        i = st["i"] % 8
        st["i"] += 1
        y = ops.moe_layer_sh(x, gates[i], gu, ups[i], gu, downs[i], dt, ids, rw, I, K, sgs[i], gu, sus[i], gu,
                             sds[i], dt, wsg)
        held.append(y[0] + y[1])

    ub = graph_us(before, 48, reps)
    held.clear()
    st["i"] = 0
    ua = graph_us(after, 48, reps)
    print(f"TIME T={T} gu={gu} dt={dt} us_before={ub:.2f} us_after={ua:.2f} saved={ub - ua:.2f}", flush=True)
    return fails


def main() -> int:
    ops.load()
    if hasattr(torch.ops.b70_gguf, "reserve_decode"):
        torch.ops.b70_gguf.reserve_decode()
    dump = Path(os.environ.get("FAIR_OUT", "/mnt/vm_8tb/b70/build/vllm-flashnext/c9/fair_dump"))
    fails = 0
    for f in sorted(dump.glob("fair_T*_gu*_dt*.bin")):
        if "_T2_" in f.name or "_T3_" in f.name:
            continue
        fails += run(f, int(os.environ.get("C9_REPS", "4")))
    print(f"VERDICT failures={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
