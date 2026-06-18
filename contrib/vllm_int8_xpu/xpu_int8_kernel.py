# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
XPUInt8ScaledMMLinearKernel -- INT8 W8A8 scaled-MM linear kernel for Intel XPU
(Battlemage / Xe2, e.g. Arc Pro B70).

This is a vLLM-side patch that pairs with the native `int8_gemm_w8a8` op added
to vllm-xpu-kernels (csrc/xpu/onednn/int8_gemm_w8a8.h + the s8_s8 joint dtype).

Place this file at:
    vllm/model_executor/kernels/linear/scaled_mm/xpu_int8.py
(beside scaled_mm/xpu.py which holds XPUFP8ScaledMMLinearKernel), or paste the
class into scaled_mm/xpu.py directly. Then wire it into the registry per
registry_patch.md.

Scope (phase 1): per-token DYNAMIC symmetric int8 activations x per-channel
(or per-tensor) symmetric int8 weights -> f16/bf16. Static and asymmetric (AZP)
schemes are rejected in can_implement() and are a follow-up.

The native op contract (must match torch_bindings.cpp / onednn_matmul.cpp):
    torch.ops._xpu_C.int8_gemm_w8a8(
        A,         # [M, K] int8   (pre-quantized activations)
        A_scale,   # [M, 1] f32    (per-token act scale)
        A_zp,      # None          (symmetric)
        B,         # [K, N] int8   (weight, transposed to NT in
                   #                process_weights_after_loading)
        B_scale,   # [1, N] f32    (per-channel weight scale)
        azp_adj,   # None          (symmetric)
        bias,      # [N] or None
        out_dtype, # torch.float16 / torch.bfloat16
    ) -> Tensor[M, N]
"""

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """INT8 W8A8 dynamic-symmetric scaled-MM via oneDNN s8s8s32 on XPU."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUInt8ScaledMM is only supported on XPU."
        if not hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):
            return False, (
                "int8_gemm_w8a8 op not present in the installed "
                "vllm-xpu-kernels wheel."
            )
        return True, None

    @classmethod
    def can_implement(
        cls, c: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if c.is_static_input_scheme:
            return False, (
                "XPU int8 kernel supports dynamic activation "
                "quantization only (no static input scale)."
            )
        if not c.input_symmetric:
            return False, (
                "XPU int8 kernel supports symmetric activations only "
                "(asymmetric / AZP not yet implemented)."
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, _, _, _ = self.layer_param_names

        # WEIGHT: stored [N, K]; the oneDNN op consumes [K, N] (NT). Transpose
        # and make contiguous so the weight strides are k-major (is_nt path).
        weight = getattr(layer, w_q_name)
        replace_parameter(
            layer,
            w_q_name,
            Parameter(weight.t().contiguous().data, requires_grad=False),
        )

        # WEIGHT SCALE: oneDNN supports per-tensor and per-channel only.
        # For a fused module (QKV / gate_up) carrying per-tensor scales, expand
        # to per-channel. Final layout fed to the op is [1, N].
        weight_scale = getattr(layer, w_s_name)
        is_fused_module = len(layer.logical_widths) > 1
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(
                weight_scale, layer.logical_widths
            )
        replace_parameter(
            layer,
            w_s_name,
            Parameter(
                weight_scale.reshape(1, -1).contiguous().data,
                requires_grad=False,
            ),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm._xpu_ops import xpu_ops as ops

        w_q, w_s, _i_s, _i_zp, _azp_adj = self._get_layer_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic per-token symmetric int8 quant of activations.
        # Prefer the fused native SYCL op; fall back to the slow @torch.compile
        # reference if the installed wheel predates it.
        # Returns (int8 [M, K], scale [M, 1], zero_point [M, 1] == 0 for sym).
        if hasattr(torch.ops._xpu_C, "dynamic_per_token_int8_quant"):
            x_q, x_s, _x_zp = torch.ops._xpu_C.dynamic_per_token_int8_quant(
                x_2d, True, 8
            )
        else:
            x_q, x_s, _x_zp = ops.dynamic_per_token_int8_quant_ref(
                x_2d, True, 8
            )

        out = torch.ops._xpu_C.int8_gemm_w8a8(
            x_q,        # A: int8 [M, K]
            x_s,        # A_scale: f32 [M, 1]
            None,       # A_zp: symmetric
            w_q,        # B: int8 [K, N]
            w_s,        # B_scale: f32 [1, N]
            None,       # azp_adj: symmetric
            bias,       # bias: [N] or None
            x.dtype,    # out_dtype: f16 / bf16
        )

        return out.reshape(x.shape[:-1] + (out.size(-1),))
