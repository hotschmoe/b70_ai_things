"""Register the B70 native XPU W8A8 linear kernel before vLLM loads models."""

from __future__ import annotations

import os
import sys

import torch


def _register_fake_once(op_name: str, implementation) -> None:
    try:
        torch.library.register_fake(op_name)(implementation)
    except RuntimeError as exc:
        message = str(exc)
        if "already has" not in message and "already registered" not in message:
            raise


def _install() -> None:
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    from vllm.model_executor.kernels.linear import _POSSIBLE_INT8_KERNELS
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearKernel,
        Int8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.platforms import PlatformEnum, current_platform

    def has_op(name: str) -> bool:
        try:
            torch._C._dispatch_find_schema_or_throw(f"_xpu_C::{name}", "")
            return True
        except RuntimeError:
            return False

    def fake_int8_gemm(a, a_scale, b, b_scale, out_dtype, bias):
        del a_scale, b_scale, bias
        return torch.empty(
            (a.shape[0], b.shape[1]),
            device=a.device,
            dtype=out_dtype or torch.bfloat16,
        )

    def fake_per_token_quant(x):
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(
            (*x.shape[:-1], 1), device=x.device, dtype=torch.float32
        )
        return q, scales

    if has_op("int8_gemm_w8a8"):
        _register_fake_once("_xpu_C::int8_gemm_w8a8", fake_int8_gemm)
    if has_op("per_token_quant_int8_xpu"):
        _register_fake_once(
            "_xpu_C::per_token_quant_int8_xpu", fake_per_token_quant
        )

    class B70XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
        """Dynamic-symmetric A8 and per-channel W8 through oneDNN XMX."""

        @classmethod
        def is_supported(cls, compute_capability=None):
            del compute_capability
            if not current_platform.is_xpu():
                return False, "B70XPUInt8ScaledMM only supports XPU"
            if not has_op("int8_gemm_w8a8"):
                return False, "vllm-xpu-kernels lacks int8_gemm_w8a8"
            return True, None

        @classmethod
        def can_implement(cls, config: Int8ScaledMMLinearLayerConfig):
            if config.is_static_input_scheme:
                return False, "only dynamic activation scales are supported"
            if not config.input_symmetric:
                return False, "only symmetric activation quantization is supported"
            if not config.is_channelwise:
                return False, "only per-channel weight scales are supported"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            weight_name, scale_name, input_scale_name, input_zp_name, azp_name = (
                self.layer_param_names
            )
            weight = getattr(layer, weight_name)
            if weight.ndim != 2:
                raise ValueError(
                    f"B70 XPU INT8 expects a 2D weight, got {tuple(weight.shape)}"
                )

            replace_parameter(layer, weight_name, weight.t().contiguous())
            weight_scale = getattr(layer, scale_name)
            replace_parameter(
                layer,
                scale_name,
                weight_scale.to(dtype=torch.float32).flatten().contiguous(),
            )
            setattr(layer, input_scale_name, None)
            setattr(layer, input_zp_name, None)
            setattr(layer, azp_name, None)
            layer.b70_native_int8_quant = (
                has_op("per_token_quant_int8_xpu")
                and os.environ.get("B70_XPU_NATIVE_INT8_QUANT", "0") == "1"
            )

        def apply_weights(self, layer, x, bias=None):
            weight, weight_scale, _, _, _ = self._get_layer_params(layer)
            x_2d = x.view(-1, x.shape[-1]).contiguous()
            if layer.b70_native_int8_quant:
                x_q, x_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x_2d)
            else:
                x_q, x_scale = per_token_quant_int8(x_2d)
            output = torch.ops._xpu_C.int8_gemm_w8a8(
                x_q,
                x_scale,
                weight,
                weight_scale,
                x.dtype,
                bias,
            )
            return output.view(*x.shape[:-1], weight.shape[1])

    kernels = _POSSIBLE_INT8_KERNELS.setdefault(PlatformEnum.XPU, [])
    kernels[:] = [
        kernel
        for kernel in kernels
        if kernel.__name__ != "B70XPUInt8ScaledMMLinearKernel"
    ]
    kernels.insert(0, B70XPUInt8ScaledMMLinearKernel)
    print(
        "[qwen38-w8a8] registered B70 native INT8 XMX linear kernel",
        file=sys.stderr,
        flush=True,
    )


try:
    _install()
except Exception as exc:
    print(
        f"[qwen38-w8a8] native INT8 registration failed: {exc!r}",
        file=sys.stderr,
        flush=True,
    )
    raise
