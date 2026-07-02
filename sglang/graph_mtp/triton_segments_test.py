#!/usr/bin/env python3
# triton_segments_test.py -- scale probe for the breakable-capture segfault: 200 pooled
# XPUGraph segments each containing a triton kernel + eltwise, eager ops between segments.
import torch
import triton
import triton.language as tl


@triton.jit
def addk(X, n, K, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(X + i, mask=m)
    tl.store(X + i, x + K, mask=m)


def main():
    torch.xpu.init()
    dev = "xpu"
    x = torch.zeros(4096, device=dev)
    grid = (4096 // 256,)
    addk[grid](x, 4096, 1.0, BLOCK=256)
    torch.xpu.synchronize()
    print("triton eager ok", float(x[0]), flush=True)
    pool = torch.xpu.graph_pool_handle()
    s = torch.xpu.Stream()
    segs = []
    with torch.xpu.stream(s):
        for i in range(200):
            g = torch.xpu.XPUGraph()
            g.capture_begin(pool=pool)
            addk[grid](x, 4096, 1.0, BLOCK=256)
            y = x * 1.0001
            g.capture_end()
            segs.append((g, y))
            x.add_(0.0)  # eager between segments
        print("200 triton+eltwise segments captured OK", flush=True)
        for g, _ in segs:
            g.replay()
        torch.xpu.synchronize()
        print("replay 200 OK", float(x[0]), flush=True)


if __name__ == "__main__":
    main()
