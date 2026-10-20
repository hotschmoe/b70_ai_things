"""Serving dispatch for packed experts; decode remains in ops_interface."""

import os


def moe_prefill(x, ids, gate, up, down, weights, gt, ut, dt, h, f):
    import torch

    from . import prefill, prefill_dq

    # The fused expert kernel has one gate/up type. Mixed GGUF layers must
    # decode each pack with its own type instead of reinterpreting up bytes.
    if gt == ut:
        return prefill.prefill_moe_layer(
            x, ids, gate, up, down, weights, gt, dt, h, f
        )
    if any(typ in (8, 10, 11) for typ in (gt, ut, dt)) or prefill_dq.moe_wants_dq(int(x.shape[0])):
        return prefill_dq.moe_dq(
            x, ids, gate, up, down, weights, gt, dt, h, f, type_up=ut
        )
    t, topk = ids.shape
    # Invalid/nonlocal IDs are handled by prefill_grouped itself.
    act = x[:, None, :].expand(t, topk, h).reshape(-1, h).contiguous()
    flat_ids = ids.reshape(-1).contiguous()
    g = prefill.prefill_grouped(act, gate, flat_ids, gt, f, h)
    u = prefill.prefill_grouped(act, up, flat_ids, ut, f, h)
    mid = (torch.nn.functional.silu(g) * u).to(x.dtype)
    slots = prefill.prefill_grouped(mid, down, flat_ids, dt, h, f)
    return prefill_dq._reduce(slots, weights, t, topk, h)
