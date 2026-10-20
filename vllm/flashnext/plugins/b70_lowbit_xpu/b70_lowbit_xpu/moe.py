"""FusedMoE method for symmetric int2/int3/int4 b70lb1 experts. TP=1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride


class B70LowbitMoEMethod:
    """Routed experts. T<=8 uses the fused GEMV. Larger T gathers per expert.

    apply() returns only the routed hidden states. The MoE runner applies
    shared experts itself.
    """

    def __init__(self, moe, bits: int, group_size: int = 128) -> None:
        if bits not in (2, 3, 4):
            raise ValueError(bits)
        if group_size != 128:
            raise ValueError("b70_lowbit group_size is 128")
        self.moe = moe
        self.bits = bits
        self.group_size = group_size
        self.moe_quant_config = None
        self.moe_kernel = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            FusedMoeWeightScaleSupported,
        )
        from vllm.model_executor.utils import set_weight_attrs

        if hidden_size % self.group_size != 0:
            raise ValueError(f"hidden {hidden_size} is not a multiple of 128")
        if intermediate_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"intermediate {intermediate_size_per_partition} "
                "is not a multiple of 128"
            )
        gated = bool(getattr(self.moe, "is_act_and_mul", True))
        n13 = (2 if gated else 1) * intermediate_size_per_partition
        attrs = dict(extra_weight_attrs)
        attrs["is_transposed"] = False
        scale_attrs = dict(attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value

        def _plane(name: str, n: int, k: int, hi: bool) -> None:
            stride = plane_hi_stride(k, self.bits) if hi else plane2_stride(k)
            param = torch.nn.Parameter(
                torch.empty(num_experts, n, stride, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, attrs)

        def _scale(name: str, n: int, k: int) -> None:
            param = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    n,
                    k // self.group_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, scale_attrs)

        _plane("w13_weight", n13, hidden_size, hi=False)
        if self.bits != 2:
            _plane("w13_weight_plane_hi", n13, hidden_size, hi=True)
        _scale("w13_weight_scale", n13, hidden_size)
        _plane("w2_weight", hidden_size, intermediate_size_per_partition, hi=False)
        if self.bits != 2:
            _plane(
                "w2_weight_plane_hi",
                hidden_size,
                intermediate_size_per_partition,
                hi=True,
            )
        _scale("w2_weight_scale", hidden_size, intermediate_size_per_partition)
        layer.b70_bits = self.bits

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for name, param in layer.named_parameters(recurse=False):
            if name.startswith("w13_") or name.startswith("w2_"):
                param.data = param.data.contiguous()

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        # fused_moe/config.py allows a method to hardcode this to None.
        del layer
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        in_shape = x.shape
        x2 = x.reshape(-1, in_shape[-1])
        if x2.shape[0] <= 8:
            y = ops.moe_gemv(
                x2,
                topk_ids,
                topk_weights,
                layer.w13_weight,
                getattr(layer, "w13_weight_plane_hi", None),
                layer.w13_weight_scale,
                layer.w2_weight,
                getattr(layer, "w2_weight_plane_hi", None),
                layer.w2_weight_scale,
                self.bits,
            )
        else:
            y = self._prefill(layer, x2, topk_weights, topk_ids)
        return y.view(*in_shape[:-1], y.shape[-1])

    def _prefill(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        t_tokens, hidden = x.shape
        n13 = layer.w13_weight.shape[1]
        intermediate = n13 // 2
        num_experts = layer.w13_weight.shape[0]
        ids = topk_ids.to(torch.int32)
        router = topk_weights.float()
        y = torch.zeros(t_tokens, hidden, dtype=torch.float32, device=x.device)
        hi13 = getattr(layer, "w13_weight_plane_hi", None)
        hi2 = getattr(layer, "w2_weight_plane_hi", None)
        present = ids.unique()
        for expert in present.tolist():
            if expert < 0 or expert >= num_experts:
                continue
            sel = (ids == expert).nonzero(as_tuple=False)
            tok = sel[:, 0]
            which = sel[:, 1]
            xe = x.index_select(0, tok)
            gu = ops.prefill_wna16(
                xe,
                layer.w13_weight[expert],
                None if hi13 is None else hi13[expert],
                layer.w13_weight_scale[expert],
                self.bits,
            )
            gate = gu[:, :intermediate]
            up = gu[:, intermediate:]
            hid = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
            down = ops.prefill_wna16(
                hid,
                layer.w2_weight[expert],
                None if hi2 is None else hi2[expert],
                layer.w2_weight_scale[expert],
                self.bits,
            )
            contrib = down.float() * router[tok, which].unsqueeze(-1)
            y.index_add_(0, tok, contrib)
        return y.to(torch.bfloat16)
