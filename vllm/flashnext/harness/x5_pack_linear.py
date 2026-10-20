"""Opt-in merge of compatible split projections before GPU weight placement.

Keeps the existing quant kernels and math. It concatenates output rows in
the same order as B70GGUFLinearMethod.apply, saving repeated input quant,
output casts, concatenations, and small BA GEMMs. Full token parity must
be checked against an unmerged arm before retaining this option.
"""
import sys


def install():
    import torch
    from vllm.models.qwen4_exp.gguf.linear import B70GGUFLinearMethod, PackedShard
    from vllm.models.qwen4_exp.gguf.types import is_unquantized

    cls = B70GGUFLinearMethod
    if getattr(cls, "_x5_pack_linear", False):
        return
    original = cls.process_weights_after_loading

    def process(self, layer):
        shards = getattr(layer, "b70_shards", {})
        order = next((o for o in [("qkv", "z"), ("q", "k", "v"), ("b", "a")]
                      if set(shards) == set(o)), None)
        if order:
            parts = [shards[key] for key in order]
            same_type = len({p.qtype for p in parts}) == 1
            same_k = len({p.logical_shape[-1] for p in parts}) == 1
            if same_type and same_k:
                typ = parts[0].qtype
                dense = is_unquantized(typ)
                payloads = [p.dense if dense else p.packed for p in parts]
                if (all(p is not None and p.ndim == 2 for p in payloads)
                        and len({p.dtype for p in payloads}) == 1
                        and len({p.device for p in payloads}) == 1):
                    merged = torch.cat(payloads, dim=0).contiguous()
                    n = sum(p.out_features for p in parts)
                    k = parts[0].logical_shape[-1]
                    assert merged.shape[0] == n
                    layer.b70_shards = {"full": PackedShard(
                        qtype=typ, packed=None if dense else merged,
                        dense=merged if dense else None,
                        out_features=n, logical_shape=(n, k))}
                    print(f"X5_PACK_LINEAR order={order} qtype={typ} N={n} K={k}",
                          file=sys.stderr, flush=True)
        return original(self, layer)

    cls.process_weights_after_loading = process
    cls._x5_pack_linear = True
