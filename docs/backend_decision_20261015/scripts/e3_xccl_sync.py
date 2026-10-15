#!/usr/bin/env python3
"""E3(b): per-sync latency of small XCCL all-reduces on two B70s (torchrun, 2 ranks).

Mimics TP decode: SYNCS all-reduces per "token" on a [rows, 2560] bf16 tensor
(rows=1 -> 5 KB, C1; rows=4 -> 20 KB, C4) with a tiny dependent op between them,
one host synchronize per token. Reports p50/p95 microseconds per sync and per token.
Also tries capturing one token's chain in torch.xpu.XPUGraph and replaying it.
"""
from datetime import timedelta
import json, os, statistics, time
import torch
import torch.distributed as dist

SYNCS = int(os.environ.get("E3_SYNCS", "96"))
TOKENS = int(os.environ.get("E3_TOKENS", "200"))
WARM = 20


def chain(x):
    for _ in range(SYNCS):
        x.mul_(0.5)
        dist.all_reduce(x)
    return x


def timed(fn, x):
    per = []
    for i in range(WARM + TOKENS):
        torch.xpu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn(x)
        torch.xpu.synchronize()
        if i >= WARM:
            per.append((time.perf_counter() - t0) * 1e6)
    per.sort()
    return {"token_p50_us": round(statistics.median(per), 1),
            "token_p95_us": round(per[int(0.95 * len(per)) - 1], 1),
            "sync_p50_us": round(statistics.median(per) / SYNCS, 2)}


def main():
    rank = int(os.environ["RANK"]); lr = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(lr)
    dist.init_process_group(backend="xccl", timeout=timedelta(seconds=60))
    out = {"p2p": os.environ.get("CCL_TOPO_P2P_ACCESS"), "syncs_per_token": SYNCS}
    for rows in (1, 4):
        x = torch.ones(rows, 2560, dtype=torch.bfloat16, device=f"xpu:{lr}")
        out[f"eager_rows{rows}"] = timed(chain, x)
        torch.xpu.synchronize()
        out[f"eager_rows{rows}"]["value_check"] = bool(torch.allclose(x.float(), torch.ones_like(x, dtype=torch.float32)))
        if os.environ.get("E3_GRAPH") != "1":
            continue
        # Capture attempt: one token's chain in a graph.
        try:
            g = torch.xpu.XPUGraph()
            y = torch.ones_like(x)
            chain(y); torch.xpu.synchronize()
            with torch.xpu.graph(g):
                chain(y)
            y.fill_(1.0); g.replay(); torch.xpu.synchronize()
            ok1 = bool(torch.allclose(y.float(), torch.ones_like(y, dtype=torch.float32)))
            y.fill_(3.0); g.replay(); torch.xpu.synchronize()
            ok3 = bool(torch.allclose(y.float(), torch.full_like(y, 3.0, dtype=torch.float32)))
            r = timed(lambda _t: g.replay(), y)
            r["replay_value_check"] = ok1 and ok3
            out[f"graph_rows{rows}"] = r
        except Exception as e:  # record, do not fail the eager result
            out[f"graph_rows{rows}"] = f"capture failed: {type(e).__name__}: {str(e)[:160]}"
    dist.barrier()
    if rank == 0:
        print("E3_XCCL_RESULT " + json.dumps(out))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
