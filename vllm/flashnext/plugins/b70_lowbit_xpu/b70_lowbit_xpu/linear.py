"""LinearMethod for symmetric int2/int3/int4 b70lb1 weights. TP=1."""

from __future__ import annotations

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride


class B70LowbitLinearMethod:
    """Weight-only linear. M<=8 uses GEMV; larger M uses bf16 DPAS prefill.

    This is a LinearMethodBase by duck typing plus the real base class when
    vLLM is imported. The quant config returns an instance of the subclass
    defined in quant_config.py, which inherits LinearMethodBase.
    """

    def __init__(self, bits: int, group_size: int = 128) -> None:
        if bits not in (2, 3, 4):
            raise ValueError(bits)
        if group_size != 128:
            raise ValueError("b70_lowbit group_size is 128")
        self.bits = bits
        self.group_size = group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.parameter import (
            GroupQuantScaleParameter,
            ModelWeightParameter,
        )
        from vllm.model_executor.utils import set_weight_attrs

        del input_size, output_size
        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"K={input_size_per_partition} is not a multiple of "
                f"{self.group_size}"
            )
        attrs = dict(extra_weight_attrs)
        weight_loader = attrs.pop("weight_loader")
        n = sum(output_partition_sizes)
        k = input_size_per_partition
        # output_dim shards N (column parallel, QKV, merged gate/up).
        # input_dim shards the packed row. That is correct at TP=1.
        # Row-parallel TP>1 must not slice a packed K row.
        weight = ModelWeightParameter(
            data=torch.empty(n, plane2_stride(k), dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, attrs)

        if self.bits != 2:
            plane_hi = ModelWeightParameter(
                data=torch.empty(
                    n, plane_hi_stride(k, self.bits), dtype=torch.uint8
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_plane_hi", plane_hi)
            set_weight_attrs(plane_hi, attrs)

        scale = GroupQuantScaleParameter(
            data=torch.empty(n, k // self.group_size, dtype=params_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)
        set_weight_attrs(scale, attrs)
        layer.b70_bits = self.bits

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = layer.weight.data.contiguous()
        plane_hi = getattr(layer, "weight_plane_hi", None)
        if plane_hi is not None:
            plane_hi.data = plane_hi.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.contiguous()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_features = layer.weight.shape[0]
        x2 = x.reshape(-1, x.shape[-1])
        plane_hi = getattr(layer, "weight_plane_hi", None)
        if x2.shape[0] <= 8:
            y = ops.gemv(x2, layer.weight, plane_hi, layer.weight_scale, self.bits)
        else:
            y = ops.prefill_wna16(
                x2, layer.weight, plane_hi, layer.weight_scale, self.bits
            )
        if bias is not None:
            y = y + bias
        return y.view(*x.shape[:-1], out_features)
