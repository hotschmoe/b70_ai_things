"""Split a GGUF MoE layer across card 0 and card 1.

Subclasses B70GGUFMoEMethod so the loader's isinstance check still passes.
Card 1 tensors are attributes of this method, not module parameters, so
vLLM does not move them onto card 0.
"""

from __future__ import annotations

import os

_INSTALLED = False
_INIT = False
_NEXT = 0
_SHARED_LOGGED = False
_SHARED_OVERLAP = os.environ.get("B70_EP_SHARED_OVERLAP") == "1"
# O4: B70_EP_FUSE_SHARED=1 fuses the shared expert into card 0's expert call
# (T <= 8; needs B70_EP_SO with moe_ep_sh); larger T takes the overlap path.
_FUSE_SHARED = os.environ.get("B70_EP_FUSE_SHARED") == "1"
# F1: B70_EP_PREFILL=1 routes T > 8 through prefill_ep (two-card prefill).
_PF_ON = os.environ.get("B70_EP_PREFILL") == "1"
_ZERO: dict = {}
# O4: B70_EP_BF16_OUT=1 has the fused call write bf16 (moe_ep_shb, needs o4c+).
_BF16_OUT = os.environ.get("B70_EP_BF16_OUT") == "1"
_ROUTE = os.environ.get("B70_EP_ROUTE") == "1"


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.models.qwen4_exp.gguf.config import B70GGUFConfig
    from vllm.models.qwen4_exp.gguf.moe import B70GGUFMoEMethod

    orig = B70GGUFConfig.get_quant_method

    def get_quant_method(self, layer, prefix):
        method = orig(self, layer, prefix)
        if type(method) is B70GGUFMoEMethod:
            if prefix and "mtp" in prefix.lower():
                return method
            return B70EPMoEMethod(method.moe)
        return method

    B70GGUFConfig.get_quant_method = get_quant_method
    if _ROUTE:
        _install_route()
    _INSTALLED = True
    print("b70_ep: B70EPMoEMethod installed", flush=True)


def _install_route() -> None:
    """O4 B70_EP_ROUTE=1: for decode (T <= 8) the EP layer skips the vLLM
    router and runs softmax top-k + bump + dispatch in one card-0 kernel
    (moe_ep_route). Needs B70_EP_FUSE_SHARED semantics (shared expert fused,
    zero shared output) and a b70_ep library with moe_ep_route."""
    import torch
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    orig = MoERunner._apply_quant_method

    def _apply_quant_method(self, hidden_states, router_logits, shared_experts_input, input_ids=None,
                            shared_experts_overlapping=False):
        qm = getattr(self.routed_experts, "quant_method", None)
        r = self.router
        if (
            getattr(qm, "_o4_route_ok", False)
            and not shared_experts_overlapping
            and self._shared_experts is not None
            and type(r) is FusedTopKRouter
            and r.scoring_func == "softmax"
            and getattr(r, "eplb_state", None) is None
            and getattr(r, "_routing_replay_out", None) is None
            and 0 < hidden_states.shape[0] <= 8
            and hidden_states.dtype == torch.bfloat16
            and router_logits.dtype == torch.bfloat16
        ):
            y = qm.apply_route(self.routed_experts, hidden_states, router_logits, int(r.top_k),
                               bool(r.renormalize), self._shared_experts)
            if y is not None:
                return self._shared_experts.output, y
        return orig(self, hidden_states, router_logits, shared_experts_input, input_ids, shared_experts_overlapping)

    MoERunner._apply_quant_method = _apply_quant_method
    print("b70_ep: B70_EP_ROUTE runner hook installed", flush=True)


class B70EPMoEMethod:
    """Filled in after vLLM imports. The real class is created in install()."""


def _make_class():
    import torch
    from torch import nn

    from vllm.models.qwen4_exp.gguf.moe import B70GGUFMoEMethod
    from vllm.models.qwen4_exp.gguf.types import packed_row_bytes

    class _Method(B70GGUFMoEMethod):
        def __init__(self, moe):
            super().__init__(moe)
            self._layer_idx = -1
            self._split = False

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            global _INIT, _NEXT
            if getattr(layer, "_ep_split", False) or self._split:
                return
            from . import ops
            from .placement import ROUTED_TYPES, flashnext_plans, split_mode

            gate_info = layer.b70_expert.get("gate")
            up_info = layer.b70_expert.get("up")
            down_info = layer.b70_expert.get("down")
            if gate_info is None or up_info is None or down_info is None:
                raise RuntimeError("b70_ep: layer is missing gate/up/down blobs")
            e, inter, hidden = (int(v) for v in gate_info["logical_shape"])
            for info, name in ((up_info, "up"), (down_info, "down")):
                if int(info["logical_shape"][0]) != e:
                    raise RuntimeError(f"b70_ep: {name} expert count != gate count {e}")
            mode = split_mode(e)
            if mode == "linear" and e % 2 != 0:
                raise RuntimeError(f"b70_ep: expert count {e} is odd")
            gt = int(layer.b70_gate_type_id)
            ut = int(layer.b70_up_type_id)
            dt = int(layer.b70_down_type_id)
            plan = None
            row_note = ""
            if mode == "profile":
                for name, qtype in (("gate", gt), ("up", ut), ("down", dt)):
                    if qtype not in ROUTED_TYPES:
                        raise RuntimeError(
                            f"b70_ep: {name} ggml type {qtype} is not a routed expert type"
                        )
                n_layers = int(os.environ.get("B70_EP_NLAYERS", "48"))
                plans = flashnext_plans(n_layers=n_layers, n_expert=e)
                if _NEXT >= len(plans):
                    raise RuntimeError(
                        f"b70_ep: profile covers {len(plans)} layers; MoE index {_NEXT} is past it"
                    )
                plan = plans[_NEXT]
                n0 = len(plan.card0)
                n1 = len(plan.card1)
                g0, g1, g_rb = _take(
                    layer.b70_w_gate, gate_info, packed_row_bytes, plan.card0, plan.card1
                )
                u0, u1, u_rb = _take(
                    layer.b70_w_up, up_info, packed_row_bytes, plan.card0, plan.card1
                )
                d0, d1, d_rb = _take(
                    layer.b70_w_down, down_info, packed_row_bytes, plan.card0, plan.card1
                )
                row_note = f" rows={g_rb},{u_rb},{d_rb}"
            else:
                n0 = e // 2
                n1 = e - n0
                g0, g1 = _halves(layer.b70_w_gate, gate_info, packed_row_bytes)
                u0, u1 = _halves(layer.b70_w_up, up_info, packed_row_bytes)
                d0, d1 = _halves(layer.b70_w_down, down_info, packed_row_bytes)
            if not _INIT:
                max_t = int(os.environ.get("B70_EP_MAX_T", "8192"))
                ops.init(max_t, hidden, 16, 64)
                _INIT = True
            idx = _NEXT
            _NEXT += 1
            self._layer_idx = idx
            # Card-0 pack replaces the CPU parameter so the loader copy can drop.
            layer.register_parameter("b70_w_gate", nn.Parameter(g0, requires_grad=False))
            layer.register_parameter("b70_w_up", nn.Parameter(u0, requires_grad=False))
            layer.register_parameter("b70_w_down", nn.Parameter(d0, requires_grad=False))
            self._c1_gate = g1
            self._c1_up = u1
            self._c1_down = d1
            ops.register0(idx, g0, gt, u0, ut, d0, dt, n0, inter, hidden)
            ops.register1(idx, g1, u1, d1, n1)
            if plan is not None:
                m0 = torch.tensor(plan.res0, dtype=torch.int32, device="xpu:0")
                m1 = torch.tensor(plan.res1, dtype=torch.int32, device="xpu:0")
                self._map0 = m0
                self._map1 = m1
                ops.register_map(idx, m0, m1)
                print(
                    f"b70_ep: split layer {idx} mode=profile E={e} n0={n0} n1={n1} "
                    f"I={inter} K={hidden} types={gt},{ut},{dt}{row_note}",
                    flush=True,
                )
            else:
                print(
                    f"b70_ep: split layer {idx} E={e} n0={n0} I={inter} K={hidden} "
                    f"types={gt},{ut},{dt}",
                    flush=True,
                )
            if _PF_ON:
                # F1: global expert id -> local slot per card, for ep_prefill_dispatch.
                from . import prefill_ep

                if plan is not None:
                    pm0, pm1 = self._map0, self._map1
                else:
                    pm0 = torch.full((e,), -1, dtype=torch.int32, device="xpu:0")
                    pm1 = torch.full((e,), -1, dtype=torch.int32, device="xpu:0")
                    pm0[:n0] = torch.arange(n0, dtype=torch.int32, device="xpu:0")
                    pm1[n0:] = torch.arange(n1, dtype=torch.int32, device="xpu:0")
                    self._map0, self._map1 = pm0, pm1
                self._pf_maps = (pm0, pm1, n0, n1)
                prefill_ep.init_if_on(hidden, 16, int(os.environ.get("B70_EP_NLAYERS", "48")))
            self._split = True
            object.__setattr__(layer, "_ep_split", True)

        @property
        def mk_can_overlap_shared_experts(self) -> bool:
            # O4: B70_EP_SHARED_OVERLAP=1 runs the shared expert inside
            # apply(), between moe_ep_begin and moe_ep_end, so it overlaps
            # card 1's experts instead of preceding the dispatch.
            return _SHARED_OVERLAP or _FUSE_SHARED

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts=None,
            shared_experts_input: torch.Tensor | None = None,
        ):
            global _SHARED_LOGGED
            # F1: long inputs (prefill) take the two-card prefill path.
            if _PF_ON and x.shape[0] > 8:
                from . import prefill_ep

                return prefill_ep.apply(
                    self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
                )
            if _FUSE_SHARED and shared_experts is not None and 0 < x.shape[0] <= 8:
                y = self._apply_fused_shared(layer, x, topk_weights, topk_ids, shared_experts)
                if y is not None:
                    return y
            if (_SHARED_OVERLAP or _FUSE_SHARED) and shared_experts is not None:
                return self._apply_overlap(
                    layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
                )
            # The runner owns the shared expert and adds it after this call.
            # moe_ep does not see those weights. They stay on card 0.
            if shared_experts is not None and not _SHARED_LOGGED:
                _SHARED_LOGGED = True
                print(
                    "b70_ep: shared expert stays on card 0; the runner adds it after moe_ep",
                    flush=True,
                )
            del shared_experts, shared_experts_input
            if getattr(layer, "expert_map", None) is not None:
                raise RuntimeError("b70_ep: expert_map is set; sidecar expects world size 1")
            xb = x if x.dtype == torch.bfloat16 else x.to(dtype=torch.bfloat16)
            if not xb.is_contiguous():
                xb = xb.contiguous()
            w = topk_weights
            if w.dtype != torch.float32:
                w = w.to(dtype=torch.float32)
            if not w.is_contiguous():
                w = w.contiguous()
            ids = topk_ids
            if ids.dtype != torch.int32:
                ids = ids.to(dtype=torch.int32)
            if not ids.is_contiguous():
                ids = ids.contiguous()
            y = torch.ops.b70_ep.moe_ep(xb, w, ids, int(self._layer_idx))
            return y.to(dtype=x.dtype)

        def _apply_fused_shared(self, layer, x, topk_weights, topk_ids, shared_experts):
            """O4: card 0 computes its routed experts + the gated shared expert in
            one b70_moe_layer2 call inside moe_ep (overlapping card 1). The
            runner's shared output is a cached zero."""
            state = getattr(self, "_o4_shared", None)
            if state is None:
                from vllm.models.qwen4_exp.gguf.moe import _shared_pack

                pack = _shared_pack(shared_experts)
                if pack is None or getattr(layer, "expert_map", None) is not None:
                    state = False
                else:
                    sg, sgt, su, sut, sd, sdt, gate_inp = pack
                    gi = gate_inp if gate_inp is not None else torch.empty(0, dtype=torch.bfloat16, device=x.device)
                    torch.ops.b70_ep.ep_register_shared(
                        int(self._layer_idx), sg, int(sgt), su, int(sut), sd, int(sdt), gi
                    )
                    state = True
                    print(f"b70_ep: layer {self._layer_idx} shared expert fused into moe_ep_sh", flush=True)
                self._o4_shared = state
                self._o4_zero = {}
            if not state:
                return None
            xb = x if x.dtype == torch.bfloat16 else x.to(dtype=torch.bfloat16)
            if not xb.is_contiguous():
                xb = xb.contiguous()
            w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(dtype=torch.float32)
            if not w.is_contiguous():
                w = w.contiguous()
            ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(dtype=torch.int32)
            if not ids.is_contiguous():
                ids = ids.contiguous()
            if _BF16_OUT and x.dtype == torch.bfloat16:
                y = torch.ops.b70_ep.moe_ep_shb(xb, w, ids, int(self._layer_idx))
            else:
                y = torch.ops.b70_ep.moe_ep_sh(xb, w, ids, int(self._layer_idx))
            key = (tuple(x.shape), x.dtype)
            zero = _ZERO.get(key)
            if zero is None:
                zero = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
                _ZERO[key] = zero
            shared_experts._output[shared_experts._output_idx] = zero
            return y.to(dtype=x.dtype)

        @property
        def _o4_route_ok(self) -> bool:
            return _ROUTE and getattr(self, "_o4_shared", None) is True

        def apply_route(self, layer, x, logits, topk, renorm, shared_experts):
            if getattr(layer, "expert_map", None) is not None:
                return None
            xb = x if x.is_contiguous() else x.contiguous()
            lg = logits if logits.is_contiguous() else logits.contiguous()
            y = torch.ops.b70_ep.moe_ep_route(xb, lg, int(self._layer_idx), int(topk), 1 if renorm else 0)
            key = (tuple(x.shape), x.dtype)
            zero = _ZERO.get(key)
            if zero is None:
                zero = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
                _ZERO[key] = zero
            shared_experts._output[shared_experts._output_idx] = zero
            return y

        def _apply_overlap(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            if getattr(layer, "expert_map", None) is not None:
                raise RuntimeError("b70_ep: expert_map is set; sidecar expects world size 1")
            xb = x if x.dtype == torch.bfloat16 else x.to(dtype=torch.bfloat16)
            if not xb.is_contiguous():
                xb = xb.contiguous()
            w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(dtype=torch.float32)
            if not w.is_contiguous():
                w = w.contiguous()
            ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(dtype=torch.int32)
            if not ids.is_contiguous():
                ids = ids.contiguous()
            y = torch.ops.b70_ep.moe_ep_begin(xb, w, ids, int(self._layer_idx))
            src = shared_experts_input if shared_experts_input is not None else x
            shared_experts._output[shared_experts._output_idx] = shared_experts._layer(src)
            torch.ops.b70_ep.moe_ep_end(y, int(self._layer_idx))
            return y.to(dtype=x.dtype)

    def _halves(param: torch.Tensor, info: dict, row_bytes_fn) -> tuple[torch.Tensor, torch.Tensor]:
        logical = tuple(int(v) for v in info["logical_shape"])
        qtype = int(info["qtype"])
        e, n, last = logical
        rb = row_bytes_fn(last, qtype)
        view = param.detach().reshape(e, n, rb)
        n0 = e // 2
        a = view[:n0].contiguous().to(device="xpu:0")
        b = view[n0:].contiguous().to(device="xpu:1")
        return a.reshape(-1), b.reshape(-1)

    def _take(param: torch.Tensor, info: dict, row_bytes_fn, card0, card1):
        """Copy one card's experts. index_select, not a contiguous() of the full pack."""
        logical = tuple(int(v) for v in info["logical_shape"])
        qtype = int(info["qtype"])
        e, n, last = logical
        rb = int(row_bytes_fn(last, qtype))
        if rb <= 0:
            raise RuntimeError(f"b70_ep: packed row is {rb} bytes for ggml type {qtype}")
        if param.device.type != "cpu":
            raise RuntimeError(f"b70_ep: expert blob is on {param.device}, expected CPU")
        view = param.detach().reshape(e, n * rb)
        i0 = torch.tensor(tuple(card0), dtype=torch.long)
        i1 = torch.tensor(tuple(card1), dtype=torch.long)
        a = view.index_select(0, i0).contiguous().to(device="xpu:0")
        b = view.index_select(0, i1).contiguous().to(device="xpu:1")
        return a.reshape(-1), b.reshape(-1), rb

    return _Method


# install() replaces this name's class after vLLM is importable.
def _bind():
    global B70EPMoEMethod
    B70EPMoEMethod = _make_class()


_orig_install = install


def install() -> None:  # noqa: F811
    _bind()
    _orig_install()
