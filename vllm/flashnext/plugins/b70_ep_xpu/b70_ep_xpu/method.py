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

            gate_info = layer.b70_expert.get("gate")
            up_info = layer.b70_expert.get("up")
            down_info = layer.b70_expert.get("down")
            if gate_info is None or up_info is None or down_info is None:
                raise RuntimeError("b70_ep: layer is missing gate/up/down blobs")
            e, inter, hidden = (int(v) for v in gate_info["logical_shape"])
            if e % 2 != 0:
                raise RuntimeError(f"b70_ep: expert count {e} is odd")
            n0 = e // 2
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
            gt = int(layer.b70_gate_type_id)
            ut = int(layer.b70_up_type_id)
            dt = int(layer.b70_down_type_id)
            layer.register_parameter("b70_w_gate", nn.Parameter(g0, requires_grad=False))
            layer.register_parameter("b70_w_up", nn.Parameter(u0, requires_grad=False))
            layer.register_parameter("b70_w_down", nn.Parameter(d0, requires_grad=False))
            self._c1_gate = g1
            self._c1_up = u1
            self._c1_down = d1
            ops.register0(idx, g0, gt, u0, ut, d0, dt, n0, inter, hidden)
            ops.register1(idx, g1, u1, d1, e - n0)
            self._split = True
            object.__setattr__(layer, "_ep_split", True)
            print(
                f"b70_ep: split layer {idx} E={e} n0={n0} I={inter} K={hidden} "
                f"types={gt},{ut},{dt}",
                flush=True,
            )

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts=None,
            shared_experts_input: torch.Tensor | None = None,
        ):
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

    return _Method


# install() replaces this name's class after vLLM is importable.
def _bind():
    global B70EPMoEMethod
    B70EPMoEMethod = _make_class()


_orig_install = install


def install() -> None:  # noqa: F811
    _bind()
    _orig_install()
