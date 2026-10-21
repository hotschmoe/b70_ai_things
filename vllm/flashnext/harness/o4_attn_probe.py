import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness")
from o4_w8_bench_util import gtime
from vllm.models.qwen4_exp.xpu import attn_decode as ad

@triton.jit
def _ld(KV, OUT, s_kn, L, S: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    sp = tl.program_id(0)
    chunk = (L + S - 1) // S
    lo = sp * chunk
    hi = tl.minimum(lo + chunk, L)
    n = tl.arange(0, BN); d = tl.arange(0, D)
    acc = tl.zeros((D,), tl.float32)
    for st in range(lo, hi, BN):
        j = st + n
        k = tl.load(KV + j[:, None].to(tl.int64) * s_kn + d[None, :], (j < hi)[:, None], 0.0).to(tl.float32)
        acc += tl.sum(k, 0)
    tl.store(OUT + sp * D + d, acc)

NQH, KVH, D, P = 24, 2, 256, 1600
for L in (300, 800, 1600):
    nb = 4
    kv = (torch.randn(nb, KVH, P, 2 * D, device="xpu") * 0.5).to(torch.bfloat16)
    out = torch.empty(64, D, device="xpu")
    for S in (8, 32):
        t = gtime(lambda: [_ld[(S,)](kv, out, kv.stride(2), L, S, 32, D) for _ in range(12)]) / 12 * 1e3
        print(f"L={L} load-only S={S} {t:.2f} us", flush=True)
    bt = torch.arange(1, dtype=torch.int32, device="xpu").view(1, 1)
    sl = torch.tensor([L], dtype=torch.int32, device="xpu")
    q = torch.randn(1, NQH, D, device="xpu").to(torch.bfloat16)
    o = torch.empty_like(q)
    for S in (4, 8, 16, 32):
        t = gtime(lambda: [ad.decode(q, kv, bt, sl, o, 0.06, KVH, S) for _ in range(12)]) / 12 * 1e3
        print(f"L={L} decode S={S} {t:.2f} us", flush=True)
