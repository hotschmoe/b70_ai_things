#!/usr/bin/env python3
# awq_kernel_probe.py -- validate the bf16-AWQ path on XPU directly (no serve plumbing):
# awq_dequantize (XPU kernel) -> cast bf16 -> matmul. Checks finite + correctness vs CPU
# reference (dequantize_gemm), and times a decode-style GEMV to gauge the 4-bit speed win.
import torch, time
from safetensors import safe_open
PATH = "/models/Qwen3.6-27B-W4A16-awq-repack/model.safetensors"
with safe_open(PATH, "pt") as f:
    qkeys = [k for k in f.keys() if k.endswith(".qweight")]
    # pick a big mlp.down_proj (in=17408) to make the GEMV meaningful
    base = next((k[:-8] for k in qkeys if "down_proj" in k), qkeys[0])
    qw = f.get_tensor(base + ".qweight"); qz = f.get_tensor(base + ".qzeros"); sc = f.get_tensor(base + ".scales")
print(f"layer={base}  qweight{tuple(qw.shape)} {qw.dtype}  scales{tuple(sc.shape)} {sc.dtype}")

from sgl_kernel import awq_dequantize
dev = "xpu"
qw_x, qz_x, sc_x = qw.to(dev), qz.to(dev), sc.to(dev)
w = awq_dequantize(qw_x, sc_x, qz_x)            # [in, out], scales.dtype (fp16)
torch.xpu.synchronize()
print(f"awq_dequantize -> {tuple(w.shape)} {w.dtype}  finite={torch.isfinite(w).all().item()}")

# correctness vs CPU reference (optional -- repack already validated round-trip 0.00e+00)
try:
    from auto_round.export.export_to_awq.utils import dequantize_gemm
    w_ref = dequantize_gemm(qw, qz, sc, 4, 128).float()
    err = (w.float().cpu() - w_ref).abs().max().item()
    print(f"max|kernel - cpu_ref| = {err:.4e}  (expect ~0)")
except Exception as e:
    print(f"(skip cpu-ref: {e})")

# the bf16-AWQ apply path: cast to bf16, matmul a batch=1 activation (decode GEMV)
wb = w.to(torch.bfloat16)
x = torch.randn(1, w.shape[0], dtype=torch.bfloat16, device=dev)
y = torch.matmul(x, wb)
torch.xpu.synchronize()
print(f"bf16 matmul -> {tuple(y.shape)} {y.dtype}  finite={torch.isfinite(y).all().item()}")

# timing: 4-bit AWQ GEMV (dequant+matmul) vs a pure bf16 GEMV of the same shape
N = 100
torch.xpu.synchronize(); t = time.time()
for _ in range(N):
    y = torch.matmul(x, awq_dequantize(qw_x, sc_x, qz_x).to(torch.bfloat16))
torch.xpu.synchronize(); awq_t = (time.time() - t) / N * 1e3
wfull = wb.clone()
torch.xpu.synchronize(); t = time.time()
for _ in range(N):
    y = torch.matmul(x, wfull)
torch.xpu.synchronize(); bf16_t = (time.time() - t) / N * 1e3
print(f"per-GEMV: awq(dequant+mm)={awq_t:.3f} ms   bf16(mm only)={bf16_t:.3f} ms")
print("NOTE: awq includes dequant overhead; in serve the win is 4x less WEIGHT BANDWIDTH read from VRAM.")
