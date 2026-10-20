"""Serving dispatch for packed experts; decode remains in ops_interface."""

import os

LAST = {}


def moe_prefill(x, ids, gate, up, down, weights, gt, ut, dt, h, f):
    import torch

    from .types import row_bytes
    from .prefill_dq import moe_wants_dq

    LAST.clear()
    tile = int(os.environ.get("B70_PREFILL_EXPERT_TILE", "0"))
    e = int(gate.numel() // (f * row_bytes(gt, h)))
    dequant_path = moe_wants_dq(int(x.shape[0])) or any(typ in (8,10,11) for typ in (gt,ut,dt))
    if dequant_path and os.environ.get("B70_PREFILL_BUCKET") == "1":
        return _bucketed(x,ids,gate,up,down,weights,gt,ut,dt,h,f,e,tile)
    if tile <= 0 or e <= tile or not dequant_path:
        return _moe_once(x, ids, gate, up, down, weights, gt, ut, dt, h, f)
    g = gate.reshape(e, -1)
    u = up.reshape(e, -1)
    d = down.reshape(e, -1)
    out = torch.zeros_like(x, dtype=torch.float32)
    for start in range(0, e, tile):
        stop = min(start + tile, e)
        local = torch.where((ids >= start) & (ids < stop), ids - start, -1)
        part = _moe_once(x, local, g[start:stop], u[start:stop], d[start:stop],
                         weights, gt, ut, dt, h, f)
        out.add_(part)
    return out


def _bucketed(x,ids,gate,up,down,weights,gt,ut,dt,h,f,e,tile):
    import torch
    from . import prefill_dq

    flat=prefill_dq._sanitize(ids,e)
    group=prefill_dq._Group(flat,e)
    counts=[group.starts_cpu[i+1]-group.starts_cpu[i] for i in range(e)]
    buckets={}
    for expert,n in enumerate(counts):
        if n:
            width=max(8,1<<(n-1).bit_length())
            buckets.setdefault(width,[]).append(expert)
    packs=[q.reshape(e,-1) for q in (gate,up,down)]
    out=torch.zeros_like(x,dtype=torch.float32)
    index=ids.clamp(min=0,max=e-1).long()
    valid=(ids>=0)&(ids<e)
    calls=0
    stages={}
    for width,experts in sorted(buckets.items()):
        step=tile if tile>0 else len(experts)
        for start in range(0,len(experts),step):
            selected=torch.tensor(experts[start:start+step],device=x.device,dtype=torch.long)
            mapping=torch.full((e,),-1,device=x.device,dtype=torch.int32)
            mapping[selected]=torch.arange(selected.numel(),device=x.device,dtype=torch.int32)
            local=torch.where(valid,mapping[index],-1)
            g,u,d=[q.index_select(0,selected) for q in packs]
            out.add_(_moe_once(x,local,g,u,d,weights,gt,ut,dt,h,f))
            if os.environ.get("B70_PREFILL_DQ_PROFILE") == "1" and gt == ut:
                from .prefill_fast import LAST as inner
                for name,ms in inner.get("profile_ms",{}).items():
                    stages[name]=stages.get(name,0)+ms
            calls+=1
    LAST.update(E=e,T=int(x.shape[0]),slots=group.mv,max_rows=max(counts,default=0),
                buckets=len(buckets),calls=calls,tile=tile)
    if stages:
        LAST["profile_ms"]=stages
    return out


def _moe_once(x, ids, gate, up, down, weights, gt, ut, dt, h, f):
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
