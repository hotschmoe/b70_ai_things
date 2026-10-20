"""LinearMethod for symmetric int2/int3/int4 b70lb1 weights. TP=1."""

from __future__ import annotations

import os

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import gptq_to_b70lb1, plane2_stride, plane_hi_stride

# gemv SLM is 8 * 2560 bf16 (csrc kSlmBytes). o_proj and linear out_proj
# are K=4096, so M=6..8 does not fit and must take the DPAS prefill kernel.
_GEMV_MAX_ELEMENTS = 8 * 2560


def _vendor_dense_prefill() -> bool:
    """True when dense prefill should unpack to bf16 and call F.linear.

    The MoE file/env switch is the source of truth. Env "bmm" used to
    fall through to prefill_wna16 here, so every GDN and attention
    projection took the slow DPAS kernel while experts used the vendor
    GEMM.
    """
    from b70_lowbit_xpu.moe import _prefill_flag

    return _prefill_flag() in ("matmul", "bmm")


def _ignore_loaded(param, loaded_weight, loaded_shard_id=None, **kwargs):
    """Checkpoint tensors the symmetric kernel does not read (qzeros, g_idx)."""
    del param, loaded_weight, loaded_shard_id, kwargs
    return None


def _drop_param(layer: torch.nn.Module, name: str) -> None:
    if not hasattr(layer, name):
        return
    old = getattr(layer, name)
    if isinstance(old, torch.nn.Parameter):
        old.data = torch.empty(0, dtype=old.dtype, device=old.device)
    delattr(layer, name)


def _install_param(layer: torch.nn.Module, name: str, data: torch.Tensor) -> None:
    _drop_param(layer, name)
    layer.register_parameter(
        name, torch.nn.Parameter(data.contiguous(), requires_grad=False)
    )


class B70LowbitLinearMethod:
    """Weight-only linear. M<=8 uses GEMV; larger M uses bf16 DPAS prefill.

    This is a LinearMethodBase by duck typing plus the real base class when
    vLLM is imported. The quant config returns an instance of the subclass
    defined in quant_config.py, which inherits LinearMethodBase.

    pack=b70lb1 registers plane2/plane_hi/scale directly (the bench path).
    pack=gptq registers qweight/scales plus ignored qzeros/g_idx, then
    process_weights_after_loading repacks into the same plane names.
    """

    def __init__(self, bits: int, group_size: int = 128, pack: str = "b70lb1") -> None:
        if bits not in (2, 3, 4):
            raise ValueError(bits)
        if group_size != 128:
            raise ValueError("b70_lowbit group_size is 128")
        if pack not in ("b70lb1", "gptq"):
            raise ValueError(pack)
        self.bits = bits
        self.group_size = group_size
        self.pack = pack

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
        del input_size, output_size
        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"K={input_size_per_partition} is not a multiple of "
                f"{self.group_size}"
            )
        tp = int(getattr(layer, "tp_size", 1) or 1)
        if self.pack == "gptq" and tp > 1:
            raise RuntimeError(
                "b70_lowbit GPTQ linear is TP=1 only; packed-K row split is wrong"
            )
        if self.pack == "gptq":
            self._create_gptq(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                extra_weight_attrs,
            )
            return
        self._create_planes(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
            extra_weight_attrs,
        )

    def _create_planes(
        self,
        layer: torch.nn.Module,
        k: int,
        output_partition_sizes: list[int],
        params_dtype: torch.dtype,
        extra_weight_attrs: dict,
    ) -> None:
        from vllm.model_executor.parameter import (
            GroupQuantScaleParameter,
            ModelWeightParameter,
        )
        from vllm.model_executor.utils import set_weight_attrs

        attrs = dict(extra_weight_attrs)
        weight_loader = attrs.pop("weight_loader")
        n = sum(output_partition_sizes)
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
                data=torch.empty(n, plane_hi_stride(k, self.bits), dtype=torch.uint8),
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
        layer.b70_k = k

    def _create_gptq(
        self,
        layer: torch.nn.Module,
        k: int,
        output_partition_sizes: list[int],
        extra_weight_attrs: dict,
    ) -> None:
        from vllm.model_executor.parameter import (
            GroupQuantScaleParameter,
            ModelWeightParameter,
        )
        from vllm.model_executor.utils import set_weight_attrs

        attrs = dict(extra_weight_attrs)
        weight_loader = attrs.pop("weight_loader")
        n = sum(output_partition_sizes)
        rows = k * self.bits // 32
        if rows * 32 != k * self.bits:
            raise ValueError(f"K={k} bits={self.bits} is not an int32 bitstream")
        # GPTQ qweight is [K*bits/32, N]. output_dim=1 narrows logical N.
        # Do not set packed_dim: int3 pack_factor 32//bits is not an integer
        # divisor of N, and the stock packed path would slice the wrong axis.
        qweight = ModelWeightParameter(
            data=torch.empty(rows, n, dtype=torch.int32),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, attrs)

        # Checkpoint scales are fp16 [K/128, N]. Keep that dtype so copy_
        # matches the file. Repack casts to bf16 for the kernel.
        scales = GroupQuantScaleParameter(
            data=torch.empty(k // self.group_size, n, dtype=torch.float16),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, attrs)

        # Symmetric dequant subtracts maxq, not the qzeros lane. g_idx is
        # identity when desc_act is false. Both still have to exist so the
        # loader does not treat the checkpoint keys as unexpected.
        for name in ("qzeros", "g_idx"):
            ignored = torch.nn.Parameter(
                torch.empty(0, dtype=torch.int32), requires_grad=False
            )
            layer.register_parameter(name, ignored)
            set_weight_attrs(ignored, {"weight_loader": _ignore_loaded})
        layer.b70_bits = self.bits
        layer.b70_k = k

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "qweight", None) is not None:
            self._repack_gptq(layer)
            return
        layer.weight.data = layer.weight.data.contiguous()
        plane_hi = getattr(layer, "weight_plane_hi", None)
        if plane_hi is not None:
            plane_hi.data = plane_hi.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.contiguous()

    def _repack_gptq(self, layer: torch.nn.Module) -> None:
        k = int(layer.b70_k)
        device = layer.qweight.device
        plane2, plane_hi, scale = gptq_to_b70lb1(
            layer.qweight.detach().to("cpu"),
            layer.scales.detach().to("cpu"),
            self.bits,
            k,
        )
        _install_param(layer, "weight", plane2.to(device))
        if plane_hi is not None:
            _install_param(layer, "weight_plane_hi", plane_hi.to(device))
        _install_param(layer, "weight_scale", scale.to(device))
        for name in ("qweight", "scales", "qzeros", "g_idx"):
            _drop_param(layer, name)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_features = layer.weight.shape[0]
        out_dtype = x.dtype
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        plane_hi = getattr(layer, "weight_plane_hi", None)
        m_tok = x2.shape[0]
        if m_tok <= 8 and m_tok * x2.shape[1] <= _GEMV_MAX_ELEMENTS:
            y = ops.gemv(x2, layer.weight, plane_hi, layer.weight_scale, self.bits)
        elif _vendor_dense_prefill():
            weight = ops.unpack_bf16(
                layer.weight, plane_hi, layer.weight_scale, self.bits
            )
            y = torch.nn.functional.linear(x2, weight)
        else:
            y = ops.prefill_wna16(
                x2, layer.weight, plane_hi, layer.weight_scale, self.bits
            )
        if bias is not None:
            y = y + bias.to(y.dtype)
        y = y.view(*x.shape[:-1], out_features)
        if y.dtype != out_dtype:
            y = y.to(out_dtype)
        return y
