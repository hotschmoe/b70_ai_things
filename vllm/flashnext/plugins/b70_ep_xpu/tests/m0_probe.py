"""M0: S28 flag oracle in torch's XPU context, eager then graph, plus probes 1-5.

m0_setup faults in every peer page, runs one dummy push/return, and resets
flag epochs before this process scores a sample. Exit 0 only when eager and
graph are 65/65 (first use + 64 reps) with push_bad = ret_bad = guard_bad =
err = 0. The mirror arm is a sensitivity control. Probe 1 prints one-way
flag latency; a value above 5 us is reported and does not by itself fail.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b70_ep_xpu.ops import load  # noqa: E402


def _alarm(signum, frame):
    del signum, frame
    print("FAIL TIMEOUT host alarm; exiting without teardown", flush=True)
    raise SystemExit(7)


def _bad(row) -> bool:
    vals = [int(v) for v in row.tolist()]
    return any(v != 0 for v in vals)


def _run_eager(reps: int, mirror: int) -> int:
    fails = 0
    for s in range(reps + 1):
        torch.ops.b70_ep.m0_prep(s)
        torch.ops.b70_ep.m0_submit_card1()
        torch.ops.b70_ep.m0_submit_card0(mirror)
        torch.xpu.synchronize()
        row = torch.ops.b70_ep.m0_check(s)
        if _bad(row):
            fails += 1
            if fails >= 3:
                break
    return fails


def main() -> int:
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(900)
    load()
    torch.xpu.set_device(0)
    # Create both contexts before the probe allocates peer regions.
    torch.empty(1, device="xpu:0")
    torch.empty(1, device="xpu:1")
    torch.xpu.synchronize()
    info = torch.ops.b70_ep.m0_setup(200000)
    print("SETUP", [int(v) for v in info.tolist()], flush=True)

    eager_fail = _run_eager(64, 0)
    print(f"EAGER fails={eager_fail}", flush=True)

    torch.xpu.set_device(0)
    torch.ops.b70_ep.m0_prep(0)
    torch.ops.b70_ep.m0_card1_capture_begin()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        torch.ops.b70_ep.m0_submit_card1()
        torch.ops.b70_ep.m0_submit_card0(0)
    torch.ops.b70_ep.m0_card1_capture_end()
    torch.xpu.synchronize()
    capture_row = torch.ops.b70_ep.m0_check(0)
    capture_ran = not _bad(capture_row)
    print(f"GRAPH capture_executed={int(capture_ran)}", flush=True)
    # 65 oracle samples. If capture executed sample 0, replay 1..64.
    # If capture only recorded, sample 0 is still the prepped payload and
    # the first replay is first use.
    graph_fail = 0
    start = 1 if capture_ran else 0
    for s in range(start, 65):
        if not (s == 0 and not capture_ran):
            torch.ops.b70_ep.m0_prep(s)
        torch.ops.b70_ep.m0_card1_replay()
        graph.replay()
        torch.xpu.synchronize()
        row = torch.ops.b70_ep.m0_check(s)
        if _bad(row):
            graph_fail += 1
            if graph_fail >= 3:
                break
    print(f"GRAPH fails={graph_fail}", flush=True)

    print("MIRROR begin (not a gate)", flush=True)
    mirror_fail = _run_eager(7, 1)
    print(f"MIRROR fails={mirror_fail}", flush=True)

    torch.ops.b70_ep.m0_latency()
    torch.ops.b70_ep.m0_teardown()
    signal.alarm(0)
    if eager_fail or graph_fail:
        print(f"FAIL eager={eager_fail} graph={graph_fail}", flush=True)
        return 1
    print("PASS eager 65/65 graph 65/65", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL exception {type(exc).__name__}: {exc}", flush=True)
        raise
