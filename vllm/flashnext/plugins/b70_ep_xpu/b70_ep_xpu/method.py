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
    _INSTALLED = True
    print("b70_ep: B70EPMoEMethod installed", flush=True)


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
            self._split = True
            object.__setattr__(layer, "_ep_split", True)

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
