"""Narrow XPU compatibility layer for Qwen3.8 ModelOpt NVFP4 on vLLM 0.28."""

from __future__ import annotations

import os
import sys


if os.environ.get("B70_NVFP4_V028") == "1":
    import torch

    import vllm_xpu_kernels._xpu_C  # noqa: F401
    from vllm.model_executor.kernels import linear as linear_kernels
    from vllm.model_executor.kernels.linear.nvfp4.emulation import (
        EmulationNvFp4LinearKernel,
    )
    from vllm.platforms.interface import PlatformEnum
    from vllm.platforms.xpu import XPUPlatform

    if "modelopt_mixed" not in XPUPlatform.supported_quantization:
        XPUPlatform.supported_quantization.append("modelopt_mixed")

    class B70XpuNvFp4W4A16LinearKernel(EmulationNvFp4LinearKernel):
        """Keep packed E2M1 weights resident and use the custom oneDNN GEMM."""

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            packed_k = layer.weight.shape[1]
            scale_k = layer.weight_scale.shape[1]
            layer._b70_nvfp4_group_size = packed_k * 2 // scale_k

            global_scale = (
                layer.weight_global_scale.data.to(torch.float32)
                .reshape(-1)
                .contiguous()
            )
            if global_scale.numel() != 1:
                raise RuntimeError("NVFP4 global weight scale must be scalar")

            layer._b70_nvfp4_scale_nt = (
                layer.weight_scale.data.to(torch.float32)
                .mul(global_scale)
                .to(torch.bfloat16)
                .t()
                .contiguous()
            )
            layer._b70_nvfp4_scale_f8_nt = (
                layer.weight_scale.data.t().contiguous()
            )
            layer._b70_nvfp4_global_scale = global_scale

            del layer.weight_scale
            del layer.weight_global_scale

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            original_shape = x.shape
            x_2d = x.reshape(-1, original_shape[-1]).to(torch.bfloat16)
            threshold = int(os.environ.get("B70_NVFP4_F8_SCALE_M_MAX", "8"))
            if x_2d.shape[0] <= threshold:
                output = torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(
                    x_2d,
                    layer.weight.data.t(),
                    bias,
                    layer._b70_nvfp4_scale_f8_nt,
                    layer._b70_nvfp4_global_scale,
                    layer._b70_nvfp4_group_size,
                )
            else:
                output = torch.ops._xpu_C.nvfp4_gemm_w4a16(
                    x_2d,
                    layer.weight.data.t(),
                    bias,
                    layer._b70_nvfp4_scale_nt,
                    layer._b70_nvfp4_group_size,
                )
            return output.reshape(*original_shape[:-1], layer.weight.shape[0])

    linear_kernels._POSSIBLE_NVFP4_KERNELS[PlatformEnum.XPU] = [
        B70XpuNvFp4W4A16LinearKernel
    ]

    def _fake_nvfp4(
        activations: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        weight_scale: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        del bias, weight_scale, group_size
        return activations.new_empty((activations.shape[0], weight.shape[1]))

    def _fake_nvfp4_f8scale(
        activations: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        weight_scale: torch.Tensor,
        global_scale: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        del bias, weight_scale, global_scale, group_size
        return activations.new_empty((activations.shape[0], weight.shape[1]))

    for op_name, fake in (
        ("_xpu_C::nvfp4_gemm_w4a16", _fake_nvfp4),
        ("_xpu_C::nvfp4_gemm_w4a16_f8scale", _fake_nvfp4_f8scale),
    ):
        try:
            torch.library.register_fake(op_name, fake)
        except (RuntimeError, ValueError):
            pass

    print(
        "[b70-v028-nvfp4] enabled modelopt_mixed and oneDNN NVFP4 W4A16",
        file=sys.stderr,
        flush=True,
    )
