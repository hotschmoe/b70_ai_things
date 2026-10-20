"""One-card PLE ring graph gate, real HC width, no model load."""

from types import SimpleNamespace

import torch
from vllm.models.qwen4_exp.gguf.mtp_ple import PLEHistory

device = torch.device("xpu:0")
d = 10240
weight = torch.arange(d * 4).reshape(d, 1, 4).remainder(7).float() / 16
p = SimpleNamespace(
    hist=9,
    hc_dim=d,
    kern=4,
    dil=3,
    eos=1,
    conv1d=SimpleNamespace(weight=weight.to(device)),
)
p._ensure_state = lambda slots, device: None
history = PLEHistory(p, 3)
history.ensure(2, 6, device)
x = torch.zeros(6, d, dtype=torch.bfloat16, device=device)
history.slots[:6].copy_(torch.tensor([0, 0, 0, 1, 1, 1], device=device))
history.starts[:6].copy_(torch.tensor([0, 0, 0, 3, 3, 3], device=device))
history.ends[:6].copy_(torch.tensor([3, 3, 3, 6, 6, 6], device=device))
history.positions[:6].copy_(torch.tensor([0, 1, 2, 0, 1, 2], device=device))
history.conv(x)
torch.xpu.synchronize()
graph = torch.xpu.XPUGraph()
with torch.xpu.graph(graph):
    output = history.conv(x)
torch.xpu.synchronize()
history.ring.zero_()
reference = {0: {}, 1: {}}
generator = torch.Generator().manual_seed(19)
for groups in (
    [(0, 0), (1, 0)],
    [(0, 3), (1, 3)],
    [(1, 4), (0, 5)],
    [(0, 6), (1, 5)],
):
    values = torch.randint(-16, 16, (6, d), generator=generator).bfloat16() / 16
    slots, positions, expected = [], [], []
    for s, (slot, first) in enumerate(groups):
        for old in list(reference[slot]):
            if old >= first:
                del reference[slot][old]
        for j in range(3):
            position = first + j
            current = values[s * 3 + j]
            reference[slot][position] = current
            acc = torch.zeros(d)
            for k in range(4):
                past = reference[slot].get(position - (3 - k) * 3, torch.zeros(d))
                acc += past.float() * weight[:, 0, k]
            expected.append(torch.nn.functional.silu(acc).bfloat16())
            slots.append(slot)
            positions.append(position)
    history.slots[:6].copy_(torch.tensor(slots, device=device))
    history.positions[:6].copy_(torch.tensor(positions, device=device))
    x.copy_(values.to(device))
    graph.replay()
    torch.xpu.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.stack(expected), atol=2e-3, rtol=1e-2
    )
print("X1_PLE_XPU_GRAPH_REJECTION_REORDER_OK", flush=True)
start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
start.record()
for _ in range(50):
    graph.replay()
end.record()
end.synchronize()
print("X1_PLE_CONV_GRAPH_MS", start.elapsed_time(end) / 50, flush=True)
