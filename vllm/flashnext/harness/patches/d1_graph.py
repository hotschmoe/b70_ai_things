# XPU graph replay reclaim.
# B70_XPU_CG_RECLAIM=N (N>0) forces keep_graph=True and re-instantiates
# each XPUGraph every N replays. Re-instantiation drops the Level Zero
# immediate-command-list growth that queue.ext_oneapi_graph leaves
# behind (torch 2.14 still submits with an event). A full torch rebuild
# for xpugraph_submit_without_event is not required for this overlay.
# See vllm/patches/xpugraph_submit_without_event.patch and
# vllm/nvfp4/patches/sitecustomize.py block (9).

import os
import sys
import time


def install_reclaim() -> None:
    n = int(os.environ.get("B70_XPU_CG_RECLAIM", "0"))
    if n <= 0:
        return
    import torch
    import torch.xpu.graphs as tgraphs

    base = torch.xpu.XPUGraph
    if getattr(base, "_d1_reclaim", False):
        return
    orig_replay = base.replay
    counts: dict[int, int] = {}
    # B70_REPLAY_LOG=1 records host time around replay(). Replay already
    # waits on the submit event, so this is the step's device-plus-submit
    # cost, not a pure launch. One sample also records an XPU event.
    log_replay = os.environ.get("B70_REPLAY_LOG", "0") not in ("", "0")
    trace_reclaim = os.environ.get("B70_RECLAIM_TRACE", "0") == "1"
    log_path = os.environ.get(
        "B70_STEP_LOG",
        "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-replay.log",
    )
    samples: list[float] = []
    device_done = {"ok": False}

    def _replay_line(msg: str) -> None:
        line = f"pid={os.getpid()} {msg}\n"
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            sys.stderr.write(line)

    class _XPUGraphReclaim(base):
        _d1_reclaim = True

        def __new__(cls, *args, **kwargs):
            return base.__new__(cls, keep_graph=True)

        # Plugin sets _b70_ep_before_replay. This flag tells it the call site
        # between instantiate and the Level Zero submit is present.
        _b70_ep_replay_hook = True

        def __init__(self, *args, **kwargs):
            super().__init__(keep_graph=True)

        def replay(self):
            k = id(self)
            seen = counts.get(k, 0) + 1
            counts[k] = seen
            if seen % n == 0:
                try:
                    if trace_reclaim:
                        _replay_line(f"instantiate_begin graph={k} replays={seen}")
                    self.instantiate()
                    if trace_reclaim:
                        _replay_line(f"instantiate_done graph={k} replays={seen}")
                except Exception as exc:
                    sys.stderr.write(
                        f"[d1-graph] instantiate failed: {exc!r}\n"
                    )
            # EP sidecar submits card 1 after instantiate and before card 0.
            # Absent unless b70_ep_xpu installed the callback. One-card replay
            # never sets it.
            before = getattr(self, "_b70_ep_before_replay", None)
            if before is not None:
                before()
            if not log_replay:
                return orig_replay(self)
            measure_device = (
                (not device_done["ok"]) and sum(counts.values()) >= 40
            )
            if measure_device:
                import torch

                torch.xpu.synchronize()
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
                out = orig_replay(self)
                end.record()
                torch.xpu.synchronize()
                device_done["ok"] = True
                try:
                    dev_ms = float(start.elapsed_time(end))
                except Exception as exc:
                    dev_ms = -1.0
                    _replay_line(f"replay_device_fail {exc!r}")
                _replay_line(f"replay_device_ms={dev_ms:.3f}")
                return out
            t0 = time.perf_counter()
            out = orig_replay(self)
            samples.append((time.perf_counter() - t0) * 1000.0)
            if len(samples) == 8 or len(samples) % 32 == 0:
                ordered = sorted(samples)
                mid = ordered[len(ordered) // 2]
                _replay_line(
                    f"replay_host n={len(samples)} "
                    f"p50_ms={mid:.3f} last_ms={samples[-1]:.3f}"
                )
            return out

    torch.xpu.XPUGraph = _XPUGraphReclaim
    tgraphs.XPUGraph = _XPUGraphReclaim
    # Model runner init copies torch.xpu.XPUGraph onto torch.cuda.CUDAGraph
    # later. If that copy already happened, refresh it.
    if getattr(torch.cuda, "CUDAGraph", None) is base:
        torch.cuda.CUDAGraph = _XPUGraphReclaim
    sys.stderr.write(
        f"[d1-graph] XPUGraph reclaim on: keep_graph=True, "
        f"re-instantiate every {n} replays\n"
    )
