#!/usr/bin/env python3
# alloc_between_segments_test.py -- does ALLOCATING eager work between pooled XPUGraph segment
# captures (like eager_on_graph attention allocating fresh outputs) corrupt the next capture?
# Mirrors: seg(pool) -> eager alloc+kernel (regular allocator) -> seg(pool) -> ... x 100 -> replay.
import torch


def main():
    torch.xpu.init()
    dev = "xpu"
    x = torch.ones(8192, device=dev)
    pool = torch.xpu.graph_pool_handle()
    s = torch.xpu.Stream()
    segs = []
    keep = []
    with torch.xpu.stream(s):
        for i in range(100):
            g = torch.xpu.XPUGraph()
            g.capture_begin(pool=pool)
            y = x * 1.001 + i  # pool alloc inside capture
            g.capture_end()
            segs.append((g, y))
            # eager ALLOCATING work between segments (attention-like: fresh output + free)
            out = torch.empty(4096 * (1 + i % 7), device=dev)
            out.normal_()
            z = out.sum()
            if i % 10 == 0:
                keep.append(out)  # some survive, some get freed -> allocator churn
            del out
        print("100 pooled segments with allocating eager breaks: captured OK", flush=True)
        for g, _ in segs:
            g.replay()
        torch.xpu.synchronize()
        print("replay OK", flush=True)


if __name__ == "__main__":
    main()
