# b70 task #12: make DENSE CompressedTensorsW8A8Int8 use a CAPTURE-SAFE true-int8 kernel on vllm-xpu-env:v0230.
#
# v0230 stock has no XPU entry in _POSSIBLE_INT8_KERNELS -> dense W8A8 fails/dequants. We register a kernel that does
# int8 weights -> triton_scaled_mm (tl.dot int8->int32 lowers to Intel DPAS/XMX). The task-c version called the triton
# kernel DIRECTLY in apply_weights; that works EAGER but inductor's torch-triton-wrap can't trace it under PIECEWISE
# graph capture (`triton.compiler.errors.CompilationError: Function argument index out of range`, x142). FIX: wrap the
# act-quant + triton_scaled_mm in an OPAQUE torch custom op (direct_register_custom_op + a fake/meta impl) -- inductor
# captures the op call and never traces into the triton kernel. Same pattern vLLM uses for its own cutlass ops.
#
# Registration is deferred to the first init_int8_linear_kernel() call (model load; vLLM fully imported) -> covers BOTH
# Quark and CompressedTensors W8A8, and avoids import-time cycles. Self-contained (no quark.py mount needed).
import os
os.environ.setdefault("B70_INT8_LINEAR", "triton")

_DONE = {"v": False}


def _b70_register():
    if _DONE["v"]:
        return
    import torch
    from vllm.model_executor.kernels.linear import _POSSIBLE_INT8_KERNELS
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )
    from vllm.platforms import PlatformEnum, current_platform
    from vllm.utils.torch_utils import direct_register_custom_op

    # --- opaque custom op: dynamic per-token symmetric int8 act-quant + int8 triton GEMM (DPAS) ---
    def _b70_int8_scaled_mm(
        x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        xc = x.contiguous()
        amax = xc.abs().amax(dim=-1, keepdim=True).to(torch.float32).clamp(min=1e-12)
        x_s = amax / 127.0
        x_q = (xc.to(torch.float32) / x_s).round().clamp_(-128, 127).to(torch.int8)
        return triton_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias)

    def _b70_int8_scaled_mm_fake(
        x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        # x is [M, K] (vLLM flattens before linear); w_q is [K, N] (cutlass-transposed) -> out [M, N]
        return torch.empty((x.shape[0], w_q.shape[1]), dtype=x.dtype, device=x.device)

    try:
        direct_register_custom_op(
            "b70_xpu_int8_scaled_mm", _b70_int8_scaled_mm, fake_impl=_b70_int8_scaled_mm_fake
        )
    except Exception as e:  # already registered (re-entrant) is fine
        print("[b70 int8 CAPTURE] custom-op register note:", repr(e), flush=True)

    class XPUInt8TritonScaledMMLinearKernel(TritonInt8ScaledMMLinearKernel):
        @classmethod
        def is_supported(cls, compute_capability=None):
            if not current_platform.is_xpu():
                return False, "requires XPU."
            return True, None

        @classmethod
        def can_implement(cls, c):
            return True, None

        def apply_weights(self, layer, x, bias=None):
            w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)
            return torch.ops.vllm.b70_xpu_int8_scaled_mm(x, w_q, w_s, bias)

    lst = _POSSIBLE_INT8_KERNELS.setdefault(PlatformEnum.XPU, [])
    if not any(k.__name__ == "XPUInt8TritonScaledMMLinearKernel" for k in lst):
        lst.insert(0, XPUInt8TritonScaledMMLinearKernel)
    _DONE["v"] = True
    print("[b70 int8 CAPTURE] registered capture-safe XPU int8 custom-op kernel (opaque -> PIECEWISE-safe)", flush=True)


try:
    import vllm.model_executor.kernels.linear as _L

    _orig = _L.init_int8_linear_kernel

    def _wrapped(*a, **k):
        _b70_register()
        return _orig(*a, **k)

    _L.init_int8_linear_kernel = _wrapped
    print("[b70 int8 CAPTURE] hooked init_int8_linear_kernel", flush=True)
except Exception as e:
    print("[b70 int8 CAPTURE] hook setup FAILED:", repr(e), flush=True)
