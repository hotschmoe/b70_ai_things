"""Opt-in deterministic GDN b|a projection; no general plugin registration."""

from __future__ import annotations

import copy
import os

_LOADED = False


def load() -> None:
    global _LOADED
    if not _LOADED:
        import torch

        so = os.environ.get("B70_N1_SO", "/mnt/vm_8tb/b70/build/vllm-flashnext/n1/_C.so")
        torch.ops.load_library(so)
        _LOADED = True


def install_gdn_ba(projection) -> None:
    """Intercept one projection's quant-method apply, preserving loader type.

    A private shallow copy retains isinstance checks and all weight-loading
    methods. Its apply delegates unsupported dtypes/devices to the original.
    GGUF b/a shards are joined in the same order as B70_LINEAR_CAT. Cache only
    after weights have reached their final device (first model forward).
    """
    import torch

    method = copy.copy(projection.quant_method)
    original = method.apply

    def apply(layer, x, bias=None):
        if x.device.type != "xpu" or x.dtype != torch.bfloat16:
            return original(layer, x, bias)
        weight = getattr(layer, "_n1_ba_weight", None)
        if weight is None:
            shards = getattr(layer, "b70_shards", {})
            if shards:
                if set(shards) != {"b", "a"}:
                    raise RuntimeError("N1 GDN b|a expects exactly b/a dense GGUF shards")
                ordered = [shards[key] for key in ("b", "a")]
                if any(sh.dense is None or sh.packed is not None for sh in ordered):
                    raise RuntimeError("N1 GDN b|a requires dense BF16 weights")
                weight = torch.cat([sh.dense for sh in ordered], dim=0).contiguous()
            else:
                weight = layer.weight
            if weight.dtype != torch.bfloat16 or weight.dim() != 2:
                return original(layer, x, bias)
            if weight.device != x.device or weight.shape[1] % 8:
                raise RuntimeError("N1 GDN b|a requires same-device weights and K divisible by 8")
            layer._n1_ba_weight = weight.contiguous()
            weight = layer._n1_ba_weight
            load()
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        y = torch.ops.b70_n1.bf_gemm(flat, weight)
        y = y.reshape(*x.shape[:-1], weight.shape[0])
        return y if bias is None else y + bias

    method.apply = apply
    projection.quant_method = method
