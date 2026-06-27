#!/usr/bin/env python3
# Compare decode-GEMV paths on XPU: non-fused (awq_dequantize+matmul) vs FUSED (awq_gemm_triton)
# vs bf16 matmul. Correctness + timing. The fused kernel reads 4-bit weights with NO fp16
# materialization -> should beat bf16 on bandwidth-bound decode if the win is real.
import torch, time
from safetensors import safe_open
from sgl_kernel import awq_dequantize
from sglang.srt.layers.quantization.awq.awq_triton import awq_gemm_triton
PATH = "/models/Qwen3.6-27B-W4A16-awq-repack/model.safetensors"
dev = "xpu"
with safe_open(PATH, "pt") as f:
    qkeys = [k for k in f.keys() if k.endswith(".qweight")]
    base = next((k[:-8] for k in qkeys if "down_proj" in k), qkeys[0])
    qw = f.get_tensor(base+".qweight").to(dev); qz = f.get_tensor(base+".qzeros").to(dev); sc = f.get_tensor(base+".scales").to(dev)
IN, OUT = qw.shape[0], qw.shape[1]*8
print(f"layer={base}  in={IN} out={OUT}")

def bench(fn, n=100):
    fn(); torch.xpu.synchronize(); t=time.time()
    for _ in range(n): r=fn()
    torch.xpu.synchronize(); return (time.time()-t)/n*1e3, r

x16 = torch.randn(1, IN, dtype=torch.float16, device=dev)
xbf = x16.to(torch.bfloat16)

# reference (non-fused)
w = awq_dequantize(qw, sc, qz)              # [in,out] fp16
y_ref = torch.matmul(x16, w)

# fused
y_fused = awq_gemm_triton(x16, qw, sc, qz, 1)
err = (y_fused.float() - y_ref.float()).abs().max().item() / (y_ref.float().abs().max().item()+1e-6)
print(f"fused vs non-fused rel-err: {err:.3e}  (expect small)")

# bf16 dense weight for baseline
wbf = w.to(torch.bfloat16)

t_nonfused,_ = bench(lambda: torch.matmul(x16, awq_dequantize(qw, sc, qz)))
t_fused1,_   = bench(lambda: awq_gemm_triton(x16, qw, sc, qz, 1))
t_fused8,_   = bench(lambda: awq_gemm_triton(x16, qw, sc, qz, 8))
t_bf16,_     = bench(lambda: torch.matmul(xbf, wbf))
print(f"per-GEMV ms:  nonfused(dequant+mm)={t_nonfused:.3f}   fused_sk1={t_fused1:.3f}   fused_sk8={t_fused8:.3f}   bf16_mm={t_bf16:.3f}")
print(f"  fused_sk1 vs bf16: {t_bf16/t_fused1:.2f}x   (>1 = fused FASTER than bf16 -> decode win is real)")
