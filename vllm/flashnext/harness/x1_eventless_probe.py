"""Eventless graph replay and RNG gate; no serving or latency claim."""

import ctypes
import os
from pathlib import Path

import torch

torch.xpu.set_device(0)


def rss_kb():
    return int(
        next(
            s.split()[1]
            for s in Path("/proc/self/status").read_text().splitlines()
            if s.startswith("VmRSS:")
        )
    )


shim = ctypes.CDLL(os.environ["X1_EVENTLESS_SHIM"])
shim.b70_x1_eventless_replay_count.restype = ctypes.c_uint64
x = torch.zeros(1024, device="xpu")
graph = torch.xpu.XPUGraph()
with torch.xpu.graph(graph):
    y = x * 3 + 1
for i in range(64):
    x.fill_(i % 17)
    graph.replay()
torch.xpu.synchronize()
before = rss_kb()
for i in range(2000):
    x.fill_(i % 17)
    graph.replay()
    if i % 250 == 249:
        torch.testing.assert_close(
            y.cpu(), torch.full((1024,), 3 * (i % 17) + 1.0), rtol=0, atol=0
        )
torch.xpu.synchronize()
after = rss_kb()
random_graph = torch.xpu.XPUGraph()
with torch.xpu.graph(random_graph):
    random = torch.rand(128, device="xpu")
random_graph.replay()
first = random.cpu()
random_graph.replay()
second = random.cpu()
assert not torch.equal(first, second), "captured RNG did not advance"
count = shim.b70_x1_eventless_replay_count()
assert count >= 2066, count
print(
    "X1_EVENTLESS_REPLAY_AND_RNG_OK",
    count,
    "rss_kb",
    before,
    after,
    "delta",
    after - before,
    flush=True,
)
