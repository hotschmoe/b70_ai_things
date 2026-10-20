"""FusedMoE method for symmetric int2/int3/int4 b70lb1 experts. TP=1."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import (
    gptq_experts_mixed_to_b70lb1,
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


# Reused across layers. Keyed by device, dtype, and shape so a prefill
# chunk does not allocate a fresh expert scratch on every layer.
_SCRATCH: dict = {}
# (name, start_event, end_event) recorded when B70_LOWBIT_FWD_SPLIT=1.
_TIMER_SPANS: list = []


def _prefill_flag() -> str:
    """B70_LOWBIT_PREFILL, or one line from B70_LOWBIT_PREFILL_FILE.

    The file is read on each apply so a long serve can switch matmul/bmm
    without reloading weights. Set B70_LOWBIT_PREFILL_SWITCH=1 to enable
    that read. Unset, the env value is fixed for the process.
    """
    if os.environ.get("B70_LOWBIT_PREFILL_SWITCH") == "1":
        path = os.environ.get(
            "B70_LOWBIT_PREFILL_FILE",
            "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/e2-prefill-mode.txt",
        )
        try:
            with open(path, encoding="ascii") as fh:
                line = fh.read().strip()
        except OSError:
            line = ""
        if line:
            return line
    return os.environ.get("B70_LOWBIT_PREFILL", "")


def _span(name: str):
    class _Bracket:
        def __enter__(self):
            self.on = os.environ.get("B70_LOWBIT_FWD_SPLIT") == "1"
            if self.on:
                import torch

                self.a = torch.xpu.Event(enable_timing=True)
                self.b = torch.xpu.Event(enable_timing=True)
                self.a.record()
            return self

        def __exit__(self, *exc):
            if self.on:
                self.b.record()
                _TIMER_SPANS.append((name, self.a, self.b))
            return False

    return _Bracket()


def drain_timer_ms() -> dict:
    """Sum recorded XPU-event spans. Caller synchronizes first."""
    acc: dict = {}
    spans = list(_TIMER_SPANS)
    _TIMER_SPANS.clear()
    for name, start, end in spans:
        acc[name] = acc.get(name, 0.0) + float(start.elapsed_time(end))
    return acc


def _scratch(key, shape, dtype, device):
    import torch

    buf = _SCRATCH.get(key)
    if (
        buf is None
        or buf.dtype != dtype
        or buf.device != device
        or tuple(buf.shape) != tuple(shape)
    ):
        buf = torch.empty(shape, dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf


def _group_slots(ids: torch.Tensor, num_experts: int):
    """Sort slots by expert. One D2H of the max count, for the bmm pad."""
    topk = int(ids.shape[1])
    flat = ids.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    slots = int(flat.shape[0])
    key = torch.where(flat >= 0, flat, torch.full_like(flat, num_experts))
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key, minlength=num_experts + 1)
    active = counts[:num_experts]
    if num_experts <= 1:
        starts = torch.zeros(num_experts, dtype=torch.int32, device=ids.device)
    else:
        tail = torch.cumsum(active[:-1], dim=0).to(torch.int32)
        starts = torch.cat(
            (torch.zeros(1, dtype=torch.int32, device=ids.device), tail), 0
        )
    mmax = int(active.max().item()) if slots and num_experts else 0
    return order, key, starts, mmax, topk


def _scatter_rows(src, expert, local, num_experts: int, mmax: int, k: int):
    """src [S, K] -> padded [E, M, K] at (expert, local). One index_copy."""
    import torch

    dest = _scratch(
        ("pad", num_experts, mmax, k, src.dtype, src.device.type, src.device.index),
        (num_experts, mmax, k),
        src.dtype,
        src.device,
    )
    if src.numel() == 0:
        return dest
    flat_i = expert.to(torch.int64) * mmax + local.to(torch.int64)
    dest.view(num_experts * mmax, k).index_copy_(0, flat_i, src)
    return dest


def expert_prefill_bmm(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_plane2: torch.Tensor,
    w13_plane_hi: torch.Tensor | None,
    w13_scale: torch.Tensor,
    w2_plane2: torch.Tensor,
    w2_plane_hi: torch.Tensor | None,
    w2_scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """One unpack of every expert and one vendor bmm per projection.

    Same math as the per-expert F.linear loop: bf16 GEMM, fp32 SiLU, fp32
    router index_add. Empty experts are still in the batch. A pad past
    8 GB falls back to the loop so a one-expert hotspot cannot allocate
    the full [E, T*topk, K] cube.
    """
    t_tokens, hidden = x.shape
    num_experts = int(w13_plane2.shape[0])
    n13 = int(w13_plane2.shape[1])
    intermediate = n13 // 2
    with _span("unpack_w13"):
        w13 = ops.unpack_bf16(
            w13_plane2.reshape(num_experts * n13, -1),
            None
            if w13_plane_hi is None
            else w13_plane_hi.reshape(num_experts * n13, -1),
            w13_scale.reshape(num_experts * n13, -1),
            bits,
        ).view(num_experts, n13, hidden)
    with _span("unpack_w2"):
        w2 = ops.unpack_bf16(
            w2_plane2.reshape(num_experts * hidden, -1),
            None
            if w2_plane_hi is None
            else w2_plane_hi.reshape(num_experts * hidden, -1),
            w2_scale.reshape(num_experts * hidden, -1),
            bits,
        ).view(num_experts, hidden, intermediate)
    with _span("group"):
        order, key, starts, mmax, topk = _group_slots(topk_ids, num_experts)
    y = torch.zeros(t_tokens, hidden, dtype=torch.float32, device=x.device)
    if mmax <= 0:
        return y.to(torch.bfloat16)
    mmax_pad = (mmax + 7) & ~7
    if num_experts * mmax_pad * hidden * 2 > 8_000_000_000:
        raise RuntimeError(
            f"bmm pad {num_experts}*{mmax_pad}*{hidden} exceeds 8GB"
        )
    slots = int(order.shape[0])
    pos = torch.arange(slots, device=x.device, dtype=torch.int32)
    sorted_e = key[order]
    valid = sorted_e < num_experts
    local = pos - starts[sorted_e.clamp(0, max(num_experts - 1, 0))]
    tok = torch.div(order, topk, rounding_mode="floor")
    expert = sorted_e[valid]
    local_v = local[valid]
    tok_v = tok[valid]
    flat_i = expert.to(torch.int64) * mmax_pad + local_v.to(torch.int64)
    with _span("pad_x"):
        gathered = x.index_select(0, tok_v)
        xpad = _scatter_rows(gathered, expert, local_v, num_experts, mmax_pad, hidden)
    with _span("bmm_w13"):
        # w13 is [E, N, K]. bmm(W, X^T) keeps the large weight contiguous.
        gu_nm = torch.bmm(w13, xpad.transpose(1, 2).contiguous())
        gu = gu_nm.transpose(1, 2).contiguous().view(num_experts * mmax_pad, n13)
        gu = gu.index_select(0, flat_i)
    with _span("silu"):
        gate = gu[:, :intermediate]
        up = gu[:, intermediate:]
        hid = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    with _span("pad_hid"):
        hpad = _scatter_rows(hid, expert, local_v, num_experts, mmax_pad, intermediate)
    with _span("bmm_w2"):
        down_nm = torch.bmm(w2, hpad.transpose(1, 2).contiguous())
        down = down_nm.transpose(1, 2).contiguous().view(num_experts * mmax_pad, hidden)
        down = down.index_select(0, flat_i)
    with _span("reduce"):
        weight = topk_weights.reshape(-1).to(torch.float32)[order[valid]].unsqueeze(-1)
        y.index_add_(0, tok_v.to(torch.int64), down.float() * weight)
    return y.to(torch.bfloat16)


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
        self,
        moe,
        bits: int,
        group_size: int = 128,
        pack: str = "b70lb1",
        bit_lookup=None,
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
        self.bit_lookup = bit_lookup
        # None means every expert uses self.bits. A list is per-expert bits
        # and self.bits is the storage width (the max), not the traffic.
        self.per_expert: list[int] | None = None
        self.moe_quant_config = None
        self.moe_kernel = None

    def _resolve_expert_bits(self, num_experts: int) -> None:
        if self.bit_lookup is None:
            return
        found: list[int] = []
        for expert in range(num_experts):
            vals = [
                int(self.bit_lookup(expert, name))
                for name in ("gate_proj", "up_proj", "down_proj")
            ]
            if any(v not in (2, 3, 4) for v in vals):
                raise ValueError(f"expert {expert} bits {vals}")
            if len(set(vals)) != 1:
                raise ValueError(f"expert {expert} proj bits differ: {vals}")
            found.append(vals[0])
        if len(set(found)) == 1:
            self.bits = found[0]
            self.per_expert = None
            return
        self.per_expert = found
        self.bits = max(found)

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
        self._resolve_expert_bits(num_experts)
        layer.b70_bits = self.bits
        layer.b70_per_expert = self.per_expert
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
        if self.per_expert is None:
            plane2, plane_hi, scale = gptq_experts_to_b70lb1(
                qweight, scales, self.bits, k
            )
        else:
            plane2, plane_hi, scale = gptq_experts_mixed_to_b70lb1(
                qweight, scales, self.per_expert, k, self.bits
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
        if self.per_expert is not None:
            layer.b70_expert_bits = torch.tensor(
                self.per_expert, dtype=torch.int32, device=device
            )
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
        out_dtype = x.dtype
        x2 = x.reshape(-1, in_shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if x2.shape[0] <= 8 and self.per_expert is not None:
            y = ops.moe_gemv_mixed(
                x2,
                topk_ids,
                topk_weights,
                layer.w13_weight,
                getattr(layer, "w13_weight_plane_hi", None),
                layer.w13_weight_scale,
                layer.w2_weight,
                getattr(layer, "w2_weight_plane_hi", None),
                layer.w2_weight_scale,
                layer.b70_expert_bits,
                self.bits,
            )
        elif x2.shape[0] <= 8:
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
        y = y.view(*in_shape[:-1], y.shape[-1])
        if y.dtype != out_dtype:
            y = y.to(out_dtype)
        return y

    def _prefill_matmul(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Unpack each expert to bf16 and use the vendor GEMM.

        Uniform bits only. One host read of the expert histogram per layer.
        """
        t_tokens, hidden = x.shape
        n13 = layer.w13_weight.shape[1]
        intermediate = n13 // 2
        num_experts = layer.w13_weight.shape[0]
        topk = topk_ids.shape[1]
        bits = self.bits
        hi13 = getattr(layer, "w13_weight_plane_hi", None)
        hi2 = getattr(layer, "w2_weight_plane_hi", None)
        w13 = ops.unpack_bf16(
            layer.w13_weight.reshape(num_experts * n13, -1),
            None if hi13 is None else hi13.reshape(num_experts * n13, -1),
            layer.w13_weight_scale.reshape(num_experts * n13, -1),
            bits,
        ).view(num_experts, n13, hidden)
        w2 = ops.unpack_bf16(
            layer.w2_weight.reshape(num_experts * hidden, -1),
            None if hi2 is None else hi2.reshape(num_experts * hidden, -1),
            layer.w2_weight_scale.reshape(num_experts * hidden, -1),
            bits,
        ).view(num_experts, hidden, intermediate)
        ids = topk_ids.to(torch.int32)
        router = topk_weights.to(torch.float32)
        flat = ids.reshape(-1)
        key = torch.where(flat >= 0, flat, torch.full_like(flat, num_experts))
        counts = torch.bincount(key, minlength=num_experts + 1)
        order = torch.argsort(key, stable=True)
        counts_cpu = counts.tolist()
        y = torch.zeros(t_tokens, hidden, dtype=torch.float32, device=x.device)
        cursor = 0
        for expert, n_tok in enumerate(counts_cpu[:num_experts]):
            n_hit = int(n_tok)
            if n_hit == 0:
                continue
            sel = order[cursor : cursor + n_hit]
            cursor += n_hit
            tok = torch.div(sel, topk, rounding_mode="floor")
            xe = x.index_select(0, tok)
            gu = torch.nn.functional.linear(xe, w13[expert])
            gate = gu[:, :intermediate]
            up = gu[:, intermediate:]
            hid = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                torch.bfloat16
            )
            down = torch.nn.functional.linear(hid, w2[expert])
            weight = router.reshape(-1)[sel].unsqueeze(-1)
            y.index_add_(0, tok, down.float() * weight)
        return y.to(torch.bfloat16)

    def _prefill(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # bmm: one grouped unpack and one vendor bmm per projection.
        # matmul: the same unpack, then one GEMM per occupied expert.
        # grouped: one ESIMD DPAS submit per projection.
        # Unset stays the per-expert WNA16 loop.
        flag = _prefill_flag()
        if flag == "bmm" and self.per_expert is None:
            return expert_prefill_bmm(
                x,
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
        if flag == "matmul" and self.per_expert is None:
            return self._prefill_matmul(layer, x, topk_weights, topk_ids)
        if flag == "grouped" and self.per_expert is None:
            y = ops.moe_prefill(
                x,
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
            return y
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
            bits = self.bits if self.per_expert is None else int(self.per_expert[expert])
            gu = ops.prefill_wna16(
                xe,
                layer.w13_weight[expert],
                None if bits == 2 or hi13 is None else hi13[expert],
                layer.w13_weight_scale[expert],
                bits,
            )
            gate = gu[:, :intermediate]
            up = gu[:, intermediate:]
            hid = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
            down = ops.prefill_wna16(
                hid,
                layer.w2_weight[expert],
                None if bits == 2 or hi2 is None else hi2[expert],
                layer.w2_weight_scale[expert],
                bits,
            )
            weight = router.reshape(-1)[sel].unsqueeze(-1)
            contrib = down.float() * weight
            y.index_add_(0, tok, contrib)
        return y.to(torch.bfloat16)
