"""Current-stack ModelOpt NVFP4 execution overlay for Intel XPU.

The Qwen3.8 RadixArk checkpoint mixes NVFP4 dense MLP/lm-head weights with
FP8 attention and Gated DeltaNet projections.  SGLang loads the format, but
its stock NVFP4 dense path dispatches to CUDA-only Marlin/CUTLASS kernels.

This overlay keeps packed E2M1 weights resident and calls the tracked
torch-2.13 XPU oneDNN operators.  The checkpoint's FP8 projections use the
stock static-W8A8 path unless the separately qualified decode-only W8A16
route is enabled.  Everything is gated by B70_XPU_NVFP4=1 so other serves
remain untouched.
"""

from __future__ import annotations

import os
import sys


_INSTALLED = False


def _load_xpu_extension():
    import torch

    if hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16_f8scale"):
        return

    import vllm_xpu_kernels._xpu_C  # noqa: F401

    required = (
        "nvfp4_gemm_w4a16",
        "nvfp4_gemm_w4a16_f8scale",
    )
    missing = [name for name in required if not hasattr(torch.ops._xpu_C, name)]
    if missing:
        raise RuntimeError(f"source-built XPU extension is missing ops: {missing}")


def _register_fake_ops():
    import torch

    def folded_fake(activations, weight, bias, weight_scale, group_size):
        del bias, weight_scale, group_size
        return activations.new_empty((activations.shape[0], weight.shape[1]))

    def native_f8_fake(
        activations,
        weight,
        bias,
        weight_scale,
        global_scale,
        group_size,
    ):
        del bias, weight_scale, global_scale, group_size
        return activations.new_empty((activations.shape[0], weight.shape[1]))

    def fp8_w8a16_fake(activations, weight, weight_scale, bias):
        del weight_scale, bias
        return activations.new_empty((activations.shape[0], weight.shape[1]))

    for name, fake in (
        ("_xpu_C::nvfp4_gemm_w4a16", folded_fake),
        ("_xpu_C::nvfp4_gemm_w4a16_f8scale", native_f8_fake),
        ("_xpu_C::fp8_gemm_w8a16", fp8_w8a16_fake),
    ):
        try:
            torch.library.register_fake(name, fake)
        except (RuntimeError, ValueError):
            # The image or another tracked overlay may already own the fake.
            pass


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import torch
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
        ModelOptFp4LinearMethod,
        ModelOptFp8LinearMethod,
        ModelOptMixedPrecisionConfig,
        ModelOptNvFp4A16LinearMethod,
    )
    from sglang.srt.utils import is_xpu

    if not is_xpu():
        return

    _load_xpu_extension()
    _register_fake_ops()

    # Capability numbers describe NVIDIA SM generations and are not meaningful
    # for the separately qualified XPU operator below.
    ModelOptFp4Config.get_min_capability = classmethod(lambda cls: 0)
    ModelOptMixedPrecisionConfig.get_min_capability = classmethod(lambda cls: 0)

    f8_m_max = int(os.environ.get("B70_NVFP4_F8_SCALE_M_MAX", "8"))
    if f8_m_max < 0:
        raise ValueError("B70_NVFP4_F8_SCALE_M_MAX must be nonnegative")
    fp8_w8a16_m_max = int(os.environ.get("B70_FP8_W8A16_M_MAX", "0"))
    if fp8_w8a16_m_max < 0:
        raise ValueError("B70_FP8_W8A16_M_MAX must be nonnegative")

    def nvfp4_process_weights(self, layer):
        group_size = int(self.quant_config.group_size)
        if group_size != 16:
            raise ValueError(f"XPU NVFP4 requires group_size=16, got {group_size}")

        if torch.unique(layer.weight_scale_2.data).numel() != 1:
            raise RuntimeError(
                "fused NVFP4 partitions have different global weight scales"
            )

        global_scale = (
            layer.weight_scale_2.data.max()
            .to(torch.float32)
            .reshape(-1)
            .contiguous()
        )
        if global_scale.numel() != 1:
            raise RuntimeError("NVFP4 global weight scale must be scalar")

        scale_f8_nt = layer.weight_scale.data.t().contiguous()
        scale_bf16_nt = (
            layer.weight_scale.data.to(torch.float32)
            .mul(global_scale)
            .to(torch.bfloat16)
            .t()
            .contiguous()
        )

        layer.b70_nvfp4_group_size = group_size
        layer.b70_nvfp4_output_size = int(layer.weight.shape[0])
        layer.b70_nvfp4_global_scale = torch.nn.Parameter(
            global_scale, requires_grad=False
        )
        layer.b70_nvfp4_scale_f8_nt = torch.nn.Parameter(
            scale_f8_nt, requires_grad=False
        )
        layer.b70_nvfp4_scale_bf16_nt = torch.nn.Parameter(
            scale_bf16_nt, requires_grad=False
        )

        del layer.weight_scale
        del layer.weight_scale_2
        if hasattr(layer, "input_scale"):
            del layer.input_scale

    def nvfp4_apply(self, layer, x, bias=None):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).to(torch.bfloat16).contiguous()
        weight_nt = layer.weight.data.t()

        if x_2d.shape[0] <= f8_m_max:
            output = torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(
                x_2d,
                weight_nt,
                bias,
                layer.b70_nvfp4_scale_f8_nt,
                layer.b70_nvfp4_global_scale,
                layer.b70_nvfp4_group_size,
            )
        else:
            output = torch.ops._xpu_C.nvfp4_gemm_w4a16(
                x_2d,
                weight_nt,
                bias,
                layer.b70_nvfp4_scale_bf16_nt,
                layer.b70_nvfp4_group_size,
            )

        return output.reshape(
            *original_shape[:-1], layer.b70_nvfp4_output_size
        ).to(x.dtype)

    ModelOptFp4LinearMethod.process_weights_after_loading = nvfp4_process_weights
    ModelOptFp4LinearMethod.apply = nvfp4_apply
    ModelOptNvFp4A16LinearMethod.process_weights_after_loading = (
        nvfp4_process_weights
    )
    ModelOptNvFp4A16LinearMethod.apply = nvfp4_apply

    original_fp8_apply = ModelOptFp8LinearMethod.apply

    def fp8_apply(self, layer, x, bias=None):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        if (
            fp8_w8a16_m_max > 0
            and x_2d.shape[0] <= fp8_w8a16_m_max
            and layer.weight.dtype == torch.float8_e4m3fn
            and layer.weight_scale.numel() == 1
        ):
            output = torch.ops._xpu_C.fp8_gemm_w8a16(
                x_2d.to(torch.bfloat16),
                layer.weight,
                layer.weight_scale,
                bias,
            )
            return output.reshape(*original_shape[:-1], layer.weight.shape[1]).to(
                x.dtype
            )
        return original_fp8_apply(self, layer, x, bias)

    ModelOptFp8LinearMethod.apply = fp8_apply

    # LogitsProcessor has a defensive runtime-state allowlist for the stock
    # Marlin/CUTLASS ModelOpt layouts.  Admit only this overlay's explicit
    # packed XPU state so the NVFP4 lm_head uses quant_method.apply instead of
    # falling through to a raw matmul on its K/2-byte packed dimension.
    import sglang.srt.layers.logits_processor as logits_processor

    original_lm_head_gate = logits_processor.should_apply_lm_head_quant_method

    def should_apply_xpu_nvfp4_lm_head(lm_head, quant_method):
        if isinstance(
            quant_method,
            (ModelOptFp4LinearMethod, ModelOptNvFp4A16LinearMethod),
        ) and all(
            hasattr(lm_head, name)
            for name in (
                "b70_nvfp4_group_size",
                "b70_nvfp4_output_size",
                "b70_nvfp4_global_scale",
                "b70_nvfp4_scale_f8_nt",
                "b70_nvfp4_scale_bf16_nt",
            )
        ):
            return lm_head.weight.dtype == torch.uint8
        return original_lm_head_gate(lm_head, quant_method)

    logits_processor.should_apply_lm_head_quant_method = (
        should_apply_xpu_nvfp4_lm_head
    )

    _INSTALLED = True
    print(
        "[b70-xpu-nvfp4] installed packed NVFP4 oneDNN path; "
        f"FP8 W8A16 M max={fp8_w8a16_m_max}",
        file=sys.stderr,
        flush=True,
    )


if os.environ.get("B70_XPU_NVFP4") == "1":
    install()
