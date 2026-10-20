"""O3 stage 3: batch invariance of the decode MoE layer.

Token 0 (fixed x, ids, router weights) is run alone (T=1) and as row 0 of
T=2..8 batches whose other tokens share some of its experts, so the expert
grouping and the entry order change. Its output row must be bit-identical in
every batch, for moe_layer (with and without the shared expert) and
moe_layer_sh (both outputs). Also checks an invalid (-1) slot in token 0.
Run under bin/gpu-run --card 0.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from b70_gguf_xpu import ops  # noqa: E402
from c9_fair import load_dump, to_xpu  # noqa: E402


def main() -> int:
    ops.load()
    if hasattr(torch.ops.b70_gguf, "reserve_decode"):
        torch.ops.b70_gguf.reserve_decode()
    dump = Path(os.environ.get("FAIR_OUT", "/mnt/vm_8tb/b70/build/vllm-flashnext/c9/fair_dump"))
    fails = 0
    for name in ("fair_T4_gu21_dt42.bin", "fair_T4_gu17_dt20.bin"):
        d = load_dump(dump / name)
        I, K, topk, E, gu, dt = d["I"], d["K"], d["topk"], d["E"], d["gu"], d["dt"]
        gate, up, down = to_xpu(d["gate"]), to_xpu(d["up"]), to_xpu(d["down"])
        sg, su, sd = to_xpu(d["sgate"]), to_xpu(d["sup"]), to_xpu(d["sdown"])
        rng = np.random.default_rng(11)
        wsg = torch.from_numpy((rng.standard_normal(K) * 0.02).astype(np.float32)).to("xpu", torch.bfloat16)
        for invalid in (False, True):
            ids0 = rng.choice(E, size=topk, replace=False).astype(np.int32)
            if invalid:
                ids0[3] = -1
            w0 = rng.random(topk).astype(np.float32)
            x0 = rng.standard_normal(K).astype(np.float32)
            ref = None
            for T in range(1, 9):
                ids = np.empty((T, topk), np.int32)
                ids[0] = ids0
                for t in range(1, T):
                    # half of the slots reuse token 0's experts, in a different order
                    mix = np.concatenate([rng.permutation(ids0[ids0 >= 0])[: topk // 2],
                                          rng.choice(E, size=topk, replace=False)])
                    _, first = np.unique(mix, return_index=True)
                    uniq = mix[np.sort(first)][:topk]
                    ids[t] = uniq if len(uniq) == topk else rng.choice(E, size=topk, replace=False)
                rw = rng.random((T, topk)).astype(np.float32)
                rw[0] = w0
                x = rng.standard_normal((T, K)).astype(np.float32)
                x[0] = x0
                xt = torch.from_numpy(x).to("xpu", torch.bfloat16)
                it = torch.from_numpy(ids).to("xpu")
                wt = torch.from_numpy(rw).to("xpu")
                a = ops.moe_layer(xt, gate, gu, up, gu, down, dt, it, wt, I, K)
                b = ops.moe_layer(xt, gate, gu, up, gu, down, dt, it, wt, I, K, sg, gu, su, gu, sd, dt)
                c = ops.moe_layer_sh(xt, gate, gu, up, gu, down, dt, it, wt, I, K, sg, gu, su, gu, sd, dt, wsg)
                torch.xpu.synchronize()
                row = (a[0].cpu().numpy().view(np.uint32), b[0].cpu().numpy().view(np.uint32),
                       c[0, 0].cpu().view(torch.int16).numpy(), c[1, 0].cpu().view(torch.int16).numpy())
                if ref is None:
                    ref = row
                    continue
                diff = [int(np.count_nonzero(r != q)) for r, q in zip(row, ref)]
                ok = sum(diff) == 0
                fails += 0 if ok else 1
                print(f"{'OK' if ok else 'FAIL'} {name} invalid={int(invalid)} T={T} differing elements "
                      f"routed={diff[0]} with_shared={diff[1]} sh_routed={diff[2]} sh_shared={diff[3]}", flush=True)
    print(f"VERDICT failures={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
