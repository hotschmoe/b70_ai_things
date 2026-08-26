"""Narrow compressed-tensors W8A8 INT8 enablement for SGLang on XPU."""

import os


def install() -> None:
    import torch

    import sglang.srt.layers.quantization.compressed_tensors.compressed_tensors as ct
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    config_cls = ct.CompressedTensorsConfig
    original_get_linear_scheme = config_cls.get_linear_scheme
    original_process = CompressedTensorsW8A8Int8.process_weights_after_loading

    def get_linear_scheme(self, *args, **kwargs):
        try:
            return original_get_linear_scheme(self, *args, **kwargs)
        except RuntimeError as error:
            expected = (
                "CompressedTensorsW8A8Int8 is not supported on XPU "
                "(no XPU kernel implementation)."
            )
            if str(error) != expected:
                raise

            # Re-enter only the already-selected W8A8 INT8 arm with the generic
            # capability gate disabled. Restore both globals before returning.
            original_is_xpu = ct._is_xpu
            original_check = config_cls._check_scheme_supported
            ct._is_xpu = False
            config_cls._check_scheme_supported = lambda self, capability, error=True: True
            try:
                scheme = original_get_linear_scheme(self, *args, **kwargs)
            finally:
                config_cls._check_scheme_supported = original_check
                ct._is_xpu = original_is_xpu

            if not isinstance(scheme, CompressedTensorsW8A8Int8):
                raise RuntimeError(
                    "B70 XPU W8A8 retry selected an unexpected quantization scheme: "
                    f"{type(scheme).__name__}"
                )
            return scheme

    def process_weights_after_loading(self, layer) -> None:
        original_process(self, layer)
        weight = layer.weight.data
        layer.b70_weight_t = weight.contiguous()
        layer.b70_weight_scale = layer.weight_scale.data.reshape(1, -1).to(
            torch.float32
        )
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=weight.dtype, device=weight.device),
            requires_grad=False,
        )

    def apply_weights(self, layer, x, bias=None):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        absmax = x_2d.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-5)
        x_scale = absmax * (1.0 / 127.0)
        x_int8 = torch.round(x_2d / x_scale).clamp_(-127, 127).to(torch.int8)
        accumulator = torch._int_mm(x_int8, layer.b70_weight_t)
        output = (
            accumulator.to(torch.float32)
            * x_scale.to(torch.float32)
            * layer.b70_weight_scale
        ).to(x.dtype)
        if bias is not None:
            output = output + bias
        return output.reshape(*original_shape[:-1], -1)

    config_cls.get_linear_scheme = get_linear_scheme
    CompressedTensorsW8A8Int8.process_weights_after_loading = (
        process_weights_after_loading
    )
    CompressedTensorsW8A8Int8.apply_weights = apply_weights
    print(
        "[b70-xpu-w8a8] installed compressed-tensors W8A8 via torch._int_mm",
        flush=True,
    )


if os.environ.get("B70_XPU_W8A8") == "1":
    install()
