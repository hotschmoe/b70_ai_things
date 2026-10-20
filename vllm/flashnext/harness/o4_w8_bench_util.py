import torch


def gtime(fn, n=20):
    fn()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    torch.xpu.synchronize()
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / n
