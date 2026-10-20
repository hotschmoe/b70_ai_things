"""Numeric check for moe_prefill against the per-expert prefill_wna16 loop.

The grouped kernel must match that loop: same unpack, same bf16 DPAS,
same SiLU, same router sum. A large max_abs means the SLM VNNI layout
or the slot index is wrong. Timing is printed for one realistic shape
and is not a serve number.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import quantize_and_pack

PASS_REL = 0.02
PASS_COS = 0.999
FAILS: list[str] = []


def _stack(rows, device):
    return torch.stack(rows).to(device)


def pack_experts(weights, bits, device):
    p2s, his, scs = [], [], []
    for weight in weights:
        plane2, plane_hi, scale = quantize_and_pack(weight, bits)
        p2s.append(plane2)
        scs.append(scale)
        if plane_hi is None:
            his.append(torch.zeros(plane2.shape[0], 0, dtype=torch.uint8))
        else:
            his.append(plane_hi)
    p2 = _stack(p2s, device)
    sc = _stack(scs, device)
    hi = None if bits == 2 else _stack(his, device)
    return p2, hi, sc


def ref_loop(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits):
    """Same reduction as B70LowbitMoEMethod._prefill, without the env switch."""
    t_tokens, hidden = x.shape
    n13 = p13.shape[1]
    intermediate = n13 // 2
    num_experts = p13.shape[0]
    topk = ids.shape[1]
    ids = ids.to(torch.int32)
    router = router.to(torch.float32)
    y = torch.zeros(t_tokens, hidden, dtype=torch.float32, device=x.device)
    flat = ids.reshape(-1)
    key = torch.where(flat >= 0, flat, torch.full_like(flat, num_experts))
    counts = torch.bincount(key, minlength=num_experts + 1)
    order = torch.argsort(key, stable=True)
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
            p13[expert],
            None if hi13 is None else hi13[expert],
            s13[expert],
            bits,
        )
        gate = gu[:, :intermediate]
        up = gu[:, intermediate:]
        hid = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        down = ops.prefill_wna16(
            hid,
            p2[expert],
            None if hi2 is None else hi2[expert],
            s2[expert],
            bits,
        )
        weight = router.reshape(-1)[sel].unsqueeze(-1)
        y.index_add_(0, tok, down.float() * weight)
    return y.to(torch.bfloat16)


def check(name, got, ref) -> None:
    torch.xpu.synchronize()
    g = got.float().reshape(-1)
    r = ref.float().reshape(-1)
    diff = g - r
    denom = torch.linalg.vector_norm(r).clamp_min(1e-8)
    rel = float(torch.linalg.vector_norm(diff) / denom)
    cos = float(torch.nn.functional.cosine_similarity(g.unsqueeze(0), r.unsqueeze(0)))
    amax = float(diff.abs().max())
    ok = rel < PASS_REL and cos > PASS_COS
    print(
        f"  {'PASS' if ok else 'FAIL'} {name} rel_l2={rel:.4e} "
        f"cos={cos:.6f} max_abs={amax:.4e}",
        flush=True,
    )
    if not ok:
        FAILS.append(name)


def one_case(device, bits, t_tokens, topk, experts, hidden, intermediate, seed):
    gen = torch.Generator().manual_seed(seed)
    w13 = [
        torch.randn(2 * intermediate, hidden, generator=gen) for _ in range(experts)
    ]
    w2 = [torch.randn(hidden, intermediate, generator=gen) for _ in range(experts)]
    p13, hi13, s13 = pack_experts(w13, bits, device)
    p2, hi2, s2 = pack_experts(w2, bits, device)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    x = torch.randn(t_tokens, hidden, dtype=torch.bfloat16, generator=g).to(device)
    ids = torch.randint(0, experts, (t_tokens, topk), generator=g)
    # Drop one slot and leave expert 0 unused when there is more than one.
    ids[0, 0] = -1
    if experts > 1:
        ids = torch.where(ids == 0, torch.ones_like(ids), ids)
        ids[0, 0] = -1
    ids = ids.to(device)
    router = torch.rand(t_tokens, topk, generator=g).to(device)
    got = ops.moe_prefill(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    ref = ref_loop(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    check(
        f"b{bits} T={t_tokens} topk={topk} E={experts} H={hidden} I={intermediate}",
        got,
        ref,
    )


def time_case(device):
    bits = 3
    experts, t_tokens, topk, hidden, intermediate = 32, 128, 8, 2048, 512
    print(
        f"== time b{bits} T={t_tokens} E={experts} H={hidden} I={intermediate}",
        flush=True,
    )
    gen = torch.Generator().manual_seed(1)
    w13 = [
        torch.randn(2 * intermediate, hidden, generator=gen) for _ in range(experts)
    ]
    w2 = [torch.randn(hidden, intermediate, generator=gen) for _ in range(experts)]
    p13, hi13, s13 = pack_experts(w13, bits, device)
    p2, hi2, s2 = pack_experts(w2, bits, device)
    x = torch.randn(t_tokens, hidden, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, experts, (t_tokens, topk), device=device)
    router = torch.rand(t_tokens, topk, device=device)
    for _ in range(2):
        ops.moe_prefill(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    torch.xpu.synchronize()
    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.moe_prefill(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    torch.xpu.synchronize()
    grouped_ms = (time.perf_counter() - t0) * 1000 / iters
    for _ in range(1):
        ref_loop(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    ref_loop(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    torch.xpu.synchronize()
    loop_ms = (time.perf_counter() - t0) * 1000
    print(
        f"  grouped {grouped_ms:.2f} ms/call  per-expert {loop_ms:.2f} ms/call  "
        f"tokens={t_tokens}",
        flush=True,
    )


def main() -> None:
    device = torch.device("xpu")
    print("so", __import__("os").environ.get("B70_LOWBIT_SO"), flush=True)
    print("== grouped vs per-expert", flush=True)
    one_case(device, 3, 13, 4, 4, 256, 128, 11)
    for bits in (2, 3, 4):
        one_case(device, bits, 17, 8, 4, 2048, 512, 20 + bits)
    if FAILS:
        raise SystemExit("GROUPED_PROBE_FAIL " + ",".join(FAILS))
    time_case(device)
    print("GROUPED_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
