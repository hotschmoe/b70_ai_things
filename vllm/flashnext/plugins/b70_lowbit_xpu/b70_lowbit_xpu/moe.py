"""FusedMoE method for symmetric int2/int3/int4 b70lb1 experts. TP=1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import (
    gptq_experts_to_b70lb1,
    plane2_stride,
    plane_hi_stride,
)


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


class B70LowbitMoEMethod:
    """Routed experts. T<=8 uses the fused GEMV. Larger T gathers per expert.

    apply() returns only the routed hidden states. The MoE runner applies
    shared experts itself.

    pack=b70lb1 registers plane tensors (the bench path). pack=gptq registers
    the checkpoint names (w13_qweight, w13_scales, ...) and repacks in
    process_weights_after_loading. The RoutedExperts weight_loader is left
    in place; is_transposed selects the GPTQ K-first shard axis.
    """

    def __init__(
        self, moe, bits: int, group_size: int = 128, pack: str = "b70lb1"
    ) -> None:
        if bits not in (2, 3, 4):
            raise ValueError(bits)
        if group_size != 128:
            raise ValueError("b70_lowbit group_size is 128")
        if pack not in ("b70lb1", "gptq"):
            raise ValueError(pack)
        self.moe = moe
        self.bits = bits
        self.group_size = group_size
        self.pack = pack
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
        mpc = getattr(self.moe, "moe_parallel_config", None)
        if mpc is not None and int(getattr(mpc, "tp_size", 1) or 1) > 1:
            raise RuntimeError(
                "b70_lowbit MoE is TP=1 only; packed-K row split is wrong"
            )
        gated = bool(getattr(self.moe, "is_act_and_mul", True))
        n13 = (2 if gated else 1) * intermediate_size_per_partition
        layer.b70_bits = self.bits
        layer.b70_k13 = hidden_size
        layer.b70_k2 = intermediate_size_per_partition
        if self.pack == "gptq":
            self._create_gptq(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                n13,
                extra_weight_attrs,
                FusedMoeWeightScaleSupported.GROUP.value,
            )
            return
        self._create_planes(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            n13,
            params_dtype,
            extra_weight_attrs,
            FusedMoeWeightScaleSupported.GROUP.value,
        )

    def _create_planes(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate: int,
        n13: int,
        params_dtype: torch.dtype,
        extra_weight_attrs: dict,
        group_quant: str,
    ) -> None:
        from vllm.model_executor.utils import set_weight_attrs

        attrs = dict(extra_weight_attrs)
        attrs["is_transposed"] = False
        scale_attrs = dict(attrs)
        scale_attrs["quant_method"] = group_quant

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
        _plane("w2_weight", hidden_size, intermediate, hi=False)
        if self.bits != 2:
            _plane("w2_weight_plane_hi", hidden_size, intermediate, hi=True)
        _scale("w2_weight_scale", hidden_size, intermediate)

    def _create_gptq(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate: int,
        n13: int,
        extra_weight_attrs: dict,
        group_quant: str,
    ) -> None:
        from vllm.model_executor.utils import set_weight_attrs

        # Keep the RoutedExperts.weight_loader passed in extra_weight_attrs.
        # is_transposed flips the shard axis so GPTQ [K_packed, N] lines up
        # with w1/w3 along N and w2 along K. At TP=1 the K slice is the
        # full packed row.
        attrs = dict(extra_weight_attrs)
        attrs["is_transposed"] = True
        group_attrs = dict(attrs)
        group_attrs["quant_method"] = group_quant

        def _qweight(name: str, k: int, n: int) -> None:
            rows = k * self.bits // 32
            param = torch.nn.Parameter(
                torch.empty(num_experts, rows, n, dtype=torch.int32),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, attrs)

        def _scales(name: str, k: int, n: int) -> None:
            param = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    k // self.group_size,
                    n,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, group_attrs)

        def _qzeros(name: str, k: int, n: int) -> None:
            # Packed along N. Gate and up each occupy half of dim 2.
            param = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    k // self.group_size,
                    n * self.bits // 32,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, group_attrs)

        def _gidx(name: str) -> None:
            # The loader indexes [expert] and then finds no scale/weight
            # token in the name, so the copy is skipped. The parameter has
            # to exist or a missing-key error is raised.
            param = torch.nn.Parameter(
                torch.empty(num_experts, 1, dtype=torch.int32),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, attrs)

        _qweight("w13_qweight", hidden_size, n13)
        _scales("w13_scales", hidden_size, n13)
        _qzeros("w13_qzeros", hidden_size, n13)
        _gidx("w13_g_idx")
        _qweight("w2_qweight", intermediate, hidden_size)
        _scales("w2_scales", intermediate, hidden_size)
        _qzeros("w2_qzeros", intermediate, hidden_size)
        _gidx("w2_g_idx")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "w13_qweight", None) is not None:
            self._repack_gptq(layer)
            return
        for name, param in list(layer.named_parameters(recurse=False)):
            if name.startswith("w13_") or name.startswith("w2_"):
                param.data = param.data.contiguous()

    def _repack_one(
        self,
        layer: torch.nn.Module,
        stem: str,
        k: int,
        device: torch.device,
    ) -> None:
        qweight = getattr(layer, stem + "_qweight").detach().to("cpu")
        scales = getattr(layer, stem + "_scales").detach().to("cpu")
        plane2, plane_hi, scale = gptq_experts_to_b70lb1(
            qweight, scales, self.bits, k
        )
        print(
            f"b70_lowbit repack {stem} bits={self.bits} "
            f"qweight={tuple(qweight.shape)} plane2={tuple(plane2.shape)}",
            flush=True,
        )
        del qweight, scales
        _install_param(layer, stem + "_weight", plane2.to(device))
        if plane_hi is not None:
            _install_param(layer, stem + "_weight_plane_hi", plane_hi.to(device))
        _install_param(layer, stem + "_weight_scale", scale.to(device))
        for suffix in ("_qweight", "_scales", "_qzeros", "_g_idx"):
            _drop_param(layer, stem + suffix)

    def _repack_gptq(self, layer: torch.nn.Module) -> None:
        device = layer.w13_qweight.device
        self._repack_one(layer, "w13", int(layer.b70_k13), device)
        self._repack_one(layer, "w2", int(layer.b70_k2), device)
        if device.type == "xpu":
            torch.xpu.empty_cache()

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
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
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
        topk = topk_ids.shape[1]
        ids = topk_ids.to(torch.int32)
        router = topk_weights.to(torch.float32)
        y = torch.zeros(t_tokens, hidden, dtype=torch.float32, device=x.device)
        hi13 = getattr(layer, "w13_weight_plane_hi", None)
        hi2 = getattr(layer, "w2_weight_plane_hi", None)
        # One host read of the expert histogram per layer. Gemm launches
        # after that stay on the device until the next layer's histogram.
        flat = ids.reshape(-1)
        valid = flat >= 0
        key = torch.where(
            valid, flat, torch.full_like(flat, num_experts)
        )
        counts = torch.bincount(key, minlength=num_experts + 1)
        order = torch.argsort(key)
        counts_cpu = counts.tolist()
        cursor = 0
        for expert, n_tok in enumerate(counts_cpu[:num_experts]):
            n_hit = int(n_tok)
            if n_hit == 0:
                continue
            sel = order[cursor : cursor + n_hit]
            cursor += n_hit
            tok = torch.div(sel, topk, rounding_mode="floor")
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
            weight = router.reshape(-1)[sel].unsqueeze(-1)
            contrib = down.float() * weight
            y.index_add_(0, tok, contrib)
        return y.to(torch.bfloat16)
