# int8_actquant_xpu.py -- XPU-safe drop-in for sglang's per_token_quant_int8.
#
# WHY: sglang's stock dynamic per-token int8 activation-quant kernel
#   sglang/srt/layers/quantization/int8_kernel.py:50
#     x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)
# uses the CUDA libdevice `round` intrinsic, which does NOT link on the Intel triton-xpu backend ->
# the int8 fused-MoE kernel dies at launch with
#   triton.runtime.errors.IntelGPUError: ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED
# (proven by research/w8a8/sglang_moe_int8_probe.py on the B70). This was the ONLY tl.extra.cuda.*
# use on the int8 MoE path (the main int8 tl.dot GEMM is platform-agnostic and codegens fine on XPU).
#
# FIX: replace the round with round-half-away-from-zero via tl.floor/tl.ceil -- the exact XPU-safe
# rounding already proven for the W4A8 dense path (sglang/patches/w4a8_actquant_triton.py:46). Numerics
# match the stock kernel to <=1 LSB (round-half-away vs round-half-even), negligible for int8 activations.
# Same signature + returns as the stock per_token_quant_int8, so install() is a transparent monkeypatch
# of BOTH the int8_kernel module attr AND the by-name import in fused_moe_triton_kernels.
#
# This module imports triton at top level -> import it ONLY inside the sglang container (never on the
# host). quark_moe_int8.install() imports + installs it lazily; the offline probe imports it directly.
import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_quant_int8_xpu(
    x_ptr,
    xq_ptr,
    scale_ptr,
    x_sum_ptr,
    stride_x,
    stride_xq,
    N,
    CAL_SUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Byte-for-byte the stock _per_token_quant_int8 (int8_kernel.py) EXCEPT the round line.
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    # XPU-safe round-half-away-from-zero (replaces tl.extra.cuda.libdevice.round, which does not link
    # on triton-xpu). Mirrors w4a8_actquant_triton.py:46.
    x_q = tl.where(x_q >= 0, tl.floor(x_q + 0.5), tl.ceil(x_q - 0.5)).to(tl.int8)
    if CAL_SUM:
        x_sum = tl.sum(x, axis=0)
        tl.store(x_sum_ptr + row_id, x_sum.to(x_sum_ptr.dtype.element_ty))
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def per_token_quant_int8_xpu(x, scale_dtype=torch.float32, cal_sum=False):
    """Drop-in for sglang int8_kernel.per_token_quant_int8 (same args + returns)."""
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=scale_dtype)
    x_sum = torch.empty(x.shape[:-1], device=x.device, dtype=x.dtype) if cal_sum else None
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    assert x.is_contiguous()
    _per_token_quant_int8_xpu[(M,)](
        x, x_q, scales, x_sum,
        stride_x=x.stride(-2), stride_xq=x_q.stride(-2), N=N,
        CAL_SUM=cal_sum, BLOCK=BLOCK, num_warps=num_warps, num_stages=1,
    )
    if cal_sum:
        return x_q, scales, x_sum
    return x_q, scales


_installed = False


def install():
    """Monkeypatch sglang's per_token_quant_int8 -> the XPU-safe version, in every module that uses it."""
    global _installed
    if _installed:
        return
    from sglang.srt.layers.quantization import int8_kernel
    int8_kernel.per_token_quant_int8 = per_token_quant_int8_xpu
    # fused_moe_triton_kernels.py does `from ...int8_kernel import per_token_quant_int8` (by-name), so
    # it holds its own reference -> patch that module attr too (this is the one the MoE kernel calls).
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_kernels as _fk
        if hasattr(_fk, "per_token_quant_int8"):
            _fk.per_token_quant_int8 = per_token_quant_int8_xpu
    except Exception as e:  # pragma: no cover
        print(f"[int8_actquant_xpu] WARN: could not patch fused_moe_triton_kernels: {e}", flush=True)
    _installed = True
    print("[int8_actquant_xpu] patched per_token_quant_int8 -> XPU-safe round (floor/ceil)", flush=True)
