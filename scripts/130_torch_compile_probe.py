#!/usr/bin/env python3
# 130_torch_compile_probe.py -- #12 Phase 1: does torch.compile (Inductor XPU backend) FUSE + speed up a
# representative decode op-chain on the B70, WITHOUT cudagraphs (so no L0/NEO accumulation)? The journal's
# "torch-compile is a no-op on XPU" was about sglang's --enable-torch-compile flag (decode stays EagerRunner),
# NOT torch.compile itself. If Inductor-XPU speeds this synthetic decode chain, it's worth wiring into the
# decode/spec forward; if it falls back / no speedup, manual fusion (#8) is the launch-reduction path instead.
import torch, time

dev = "xpu"
H, I, L, M = 5120, 17408, 32, 1   # hidden, intermediate, layers, decode token
torch.manual_seed(0)

def rmsnorm(x, w):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * w

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gw = torch.nn.Parameter(torch.randn(H, 2 * I, dtype=torch.bfloat16, device=dev) * 0.02)
        self.dw = torch.nn.Parameter(torch.randn(I, H, dtype=torch.bfloat16, device=dev) * 0.02)
        self.n1 = torch.nn.Parameter(torch.ones(H, dtype=torch.bfloat16, device=dev))
        self.n2 = torch.nn.Parameter(torch.ones(H, dtype=torch.bfloat16, device=dev))
    def forward(self, x):
        h = rmsnorm(x, self.n1)
        g, u = (h @ self.gw).chunk(2, -1)
        x = x + (torch.nn.functional.silu(g) * u) @ self.dw
        # a cheap norm to mimic the post-attn norm launch
        return rmsnorm(x, self.n2) * 0 + x

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(L)])
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

net = Net().to(dev).eval()
x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)

def bench(fn, it=60):
    with torch.no_grad():
        for _ in range(8):
            fn(x)
        torch.xpu.synchronize()
        t = time.time()
        for _ in range(it):
            fn(x)
        torch.xpu.synchronize()
    return (time.time() - t) / it * 1000

eager_ms = bench(net)
print(f"eager:    {eager_ms:.3f} ms/token  ({1000/eager_ms:.1f} tok/s, {L} synthetic layers)")
for mode in ("default", "max-autotune-no-cudagraphs"):
    try:
        cn = torch.compile(net, backend="inductor", mode=mode)
        comp_ms = bench(cn)
        with torch.no_grad():
            rel = ((net(x) - cn(x)).norm() / net(x).norm().clamp(min=1e-6)).item()
        print(f"compile[{mode}]: {comp_ms:.3f} ms/token  speedup {eager_ms/comp_ms:.2f}x  rel-err {rel:.2e}")
    except Exception as e:
        print(f"compile[{mode}]: FAILED {type(e).__name__}: {str(e)[:160]}")
