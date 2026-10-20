#!/usr/bin/env python3
"""O4: Triton topk_softmax vs vLLM XPU topk_softmax (bitwise), timing."""
import sys
import torch
import vllm._custom_ops as ops  # noqa: F401  (loads _moe_C)
from vllm.models.qwen4_exp.xpu.topk_decode import topk_softmax
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness")
from o4_w8_bench_util import gtime

E, K = 512, 10
for recip, precise in ((False, True), (True, True), (False, False)):
    for renorm in (True,):
        bad_w = bad_i = 0
        for trial in range(200):
            m = 1 + trial % 8
            g = (torch.randn(m, E, device="xpu") * (0.5 + trial % 5)).to(torch.bfloat16)
            w1 = torch.empty(m, K, dtype=torch.float32, device="xpu"); i1 = torch.empty(m, K, dtype=torch.int32, device="xpu"); t1 = torch.empty_like(i1)
            torch.ops._moe_C.topk_softmax(w1, i1, t1, g, renorm, None, None)
            w2 = torch.empty_like(w1); i2 = torch.empty_like(i1); t2 = torch.empty_like(i1)
            topk_softmax(w2, i2, t2, g, renorm, recip, precise)
            bad_i += int(not torch.equal(i1, i2))
            bad_w += int(not torch.equal(w1, w2))
            if trial < 3 and not torch.equal(w1, w2):
                d = (w1.view(torch.int32) - w2.view(torch.int32)).abs().max().item()
                print("  max ulp", d, flush=True)
            if trial == 0:
                print("tei equal", torch.equal(t1, t2), t1[0, :4].tolist(), t2[0, :4].tolist())
        print(f"recip={recip} precise={precise} renorm={renorm} id_mismatch={bad_i}/200 w_mismatch={bad_w}/200", flush=True)
g = torch.randn(1, E, device="xpu").to(torch.bfloat16)
w = torch.empty(1, K, dtype=torch.float32, device="xpu"); i = torch.empty(1, K, dtype=torch.int32, device="xpu"); t = torch.empty_like(i)
def a():
    for _ in range(48):
        torch.ops._moe_C.topk_softmax(w, i, t, g, True, None, None)
def b():
    for _ in range(48):
        topk_softmax(w, i, t, g, True)
print(f"48 calls: xpu {gtime(a):.3f} ms triton {gtime(b):.3f} ms")
