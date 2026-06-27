#!/usr/bin/env python3
# woq_probe.py -- decisive test of the auto_round_kernel WOQ XPU path (int4 weight + int8 XMX compute,
# the kernel behind vLLM's proven 30 t/s). Instantiate QuantLinearGPTQ on a real Lorbus int4 layer and
# time its forward (decode GEMV, M=1) vs a bf16 nn.Linear of the same shape. Run in vllm-xpu-env:v0230.
import torch, time, glob
from safetensors import safe_open
from auto_round_kernel.qlinear import QuantLinearGPTQ

d = "/models/Lorbus_Qwen3.6-27B-int4-AutoRound"
sf = sorted(glob.glob(d + "/*.safetensors"))
base = "model.language_model.layers.0.mlp.down_proj"
qw = qz = sc = None
for p in sf:
    with safe_open(p, "pt") as f:
        if base + ".qweight" in f.keys():
            qw = f.get_tensor(base + ".qweight"); qz = f.get_tensor(base + ".qzeros"); sc = f.get_tensor(base + ".scales"); break
IN, OUT = qw.shape[0] * 8, qw.shape[1]
print(f"layer={base}  in={IN} out={OUT}  (bits=4 g=128 sym)")

ql = QuantLinearGPTQ(4, 128, True, IN, OUT, False, weight_dtype=torch.bfloat16)
ql.qweight.data = qw; ql.qzeros.data = qz; ql.scales.data = sc
ql = ql.to("xpu"); ql.post_init()
print(f"post_init OK  cdt={ql.cdt} wdt={ql.wdt} sdt={ql.sdt}  packed_qweight={tuple(ql.qweight.shape)} {ql.qweight.dtype}")

x = torch.randn(1, IN, dtype=torch.bfloat16, device="xpu")
y = ql(x); torch.xpu.synchronize()
print(f"woq forward -> {tuple(y.shape)} {y.dtype}  finite={torch.isfinite(y).all().item()}")

lin = torch.nn.Linear(IN, OUT, bias=False).to("xpu").to(torch.bfloat16)

def bench(fn, n=200):
    fn(); torch.xpu.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.xpu.synchronize(); return (time.time() - t) / n * 1e3

t_woq = bench(lambda: ql(x)); t_bf16 = bench(lambda: lin(x))
print(f"per-GEMV ms:  woq(int4 wt / int8 XMX)={t_woq:.3f}   bf16(XMX)={t_bf16:.3f}   speedup={t_bf16/t_woq:.2f}x")
print(">1x => the WOQ int4 kernel BEATS bf16 -> this is the path to bring into sglang.")
# also a prefill-ish batch to see GEMM behaviour
xb = torch.randn(512, IN, dtype=torch.bfloat16, device="xpu")
t_woq_b = bench(lambda: ql(xb), 50); t_bf16_b = bench(lambda: lin(xb), 50)
print(f"per-GEMM(M=512) ms:  woq={t_woq_b:.3f}   bf16={t_bf16_b:.3f}   speedup={t_bf16_b/t_woq_b:.2f}x")
