# Fused silu_and_mul + per-token symmetric int8 quant (Triton) -- the down_proj-input fusion that vLLM-XPU
# is MISSING (kernel/11: the existing silu_and_mul_quant is FP8/per-block only, not per-token-symmetric s8).
#
# Replaces the standalone path  y = silu(gate)*up ; xq,xs = dynamic_per_token_int8_quant(y)  (2 launches +
# a full f16 `y` round-trip) with ONE kernel that emits int8 directly + the per-token scale. Measured on B70
# card0 (:int8g, I=17408): separate 105us -> fused 70us = 1.51x @M=1 (1.73x @M=64). CORRECT (max int8 diff 1,
# scale rel-err 3e-3). NOTE: Triton on XPU has a ~60-70us fixed DISPATCH floor (see kernel/23) -> this proves
# the fusion but a NATIVE (SYCL, in vllm-xpu-kernels csrc) version would beat it (lower dispatch + the win
# survives graph capture, where only the parallel-reduction WORK remains). Productionization = native kernel
# + wire into XPUInt8ScaledMMLinearKernel for the down_proj (and the existing native rms_norm_dynamic_per_token_quant
# for the qkv/gate_up inputs -- 2.06x, EXISTS, just needs wiring). DRAFT / reference, NOT wired.
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_quant_kernel(g_ptr, u_ptr, xq_ptr, s_ptr, I: tl.constexpr, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    base = m * I
    amax = 0.0
    for i0 in range(0, I, BLOCK):  # pass 1: silu(g)*u, per-row absmax (parallel over the work-group)
        o = i0 + tl.arange(0, BLOCK); mask = o < I
        g = tl.load(g_ptr + base + o, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + base + o, mask=mask, other=0.0).to(tl.float32)
        y = (g / (1.0 + tl.exp(-g))) * u
        amax = tl.maximum(amax, tl.max(tl.abs(y)))
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    tl.store(s_ptr + m, amax / 127.0)
    for i0 in range(0, I, BLOCK):  # pass 2: quantize (recompute silu*u; the f16 y is never materialized)
        o = i0 + tl.arange(0, BLOCK); mask = o < I
        g = tl.load(g_ptr + base + o, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + base + o, mask=mask, other=0.0).to(tl.float32)
        xs = ((g / (1.0 + tl.exp(-g))) * u) * inv
        xi = tl.maximum(tl.minimum((xs + tl.where(xs >= 0.0, 0.5, -0.5)).to(tl.int32), 127), -127)
        tl.store(xq_ptr + base + o, xi.to(tl.int8), mask=mask)


def silu_mul_quant_int8(gate: torch.Tensor, up: torch.Tensor, block: int = 1024, num_warps: int = 16):
    """gate, up: [M, I] (the split gate_up_proj output). Returns (xq[M,I] int8, scale[M] fp32)."""
    M, I = gate.shape
    xq = torch.empty((M, I), dtype=torch.int8, device=gate.device)
    s = torch.empty((M,), dtype=torch.float32, device=gate.device)
    _silu_mul_quant_kernel[(M,)](gate, up, xq, s, I=I, BLOCK=min(block, I), num_warps=num_warps)
    return xq, s
