"""Numeric check for moe_gemv_mixed. Not a serve number.

Uniform bits must match templated moe_gemv. Mixed bits must match the
per-expert prefill_wna16 loop. Storage width is the max bits. int2 hi
planes are zeros inside an int3 hi allocation.
"""

from __future__ import annotations

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane_hi_stride, quantize_and_pack

PASS_REL = 0.02
PASS_COS = 0.999
FAILS: list[str] = []


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


def pack_mixed(weights, expert_bits, storage, device):
    """weights is a list of [N, K] float. plane2 stride is shared."""
    planes = []
    his = []
    scales = []
    for weight, bits in zip(weights, expert_bits):
        p2, hi, sc = quantize_and_pack(weight, bits)
        planes.append(p2)
        scales.append(sc)
        if storage == 2:
            his.append(None)
            continue
        k = weight.shape[1]
        wide = torch.zeros(p2.shape[0], plane_hi_stride(k, storage), dtype=torch.uint8)
        if hi is not None:
            wide[:, : hi.shape[1]] = hi
        his.append(wide)
    p2 = torch.stack(planes).to(device)
    sc = torch.stack(scales).to(device)
    hi = None if storage == 2 else torch.stack(his).to(device)
    bits_t = torch.tensor(expert_bits, dtype=torch.int32, device=device)
    return p2, hi, sc, bits_t


def _native_hi(hi_row, bits, k):
    """prefill_wna16 requires the stride for this bit width.

    Storage can be wider (int3 payload in an int4 row). The bytes that
    the kernel reads sit at the start of the row.
    """
    if bits == 2 or hi_row is None:
        return None
    need = plane_hi_stride(k, bits)
    if hi_row.shape[-1] == need:
        return hi_row
    return hi_row[:, :need].contiguous()


def ref_loop(x, ids, router, p13, hi13, s13, p2, hi2, s2, expert_bits):
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
    cursor = 0
    counts_cpu = counts.tolist()
    for expert, n_tok in enumerate(counts_cpu[:num_experts]):
        n_hit = int(n_tok)
        if n_hit == 0:
            continue
        sel = order[cursor : cursor + n_hit]
        cursor += n_hit
        tok = torch.div(sel, topk, rounding_mode="floor")
        xe = x.index_select(0, tok)
        bits = int(expert_bits[expert])
        k13 = int(s13.shape[-1]) * 128
        k2 = int(s2.shape[-1]) * 128
        hi_up = _native_hi(None if hi13 is None else hi13[expert], bits, k13)
        gu = ops.prefill_wna16(xe, p13[expert], hi_up, s13[expert], bits)
        gate = gu[:, :intermediate]
        up = gu[:, intermediate:]
        hid = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
        hi_dn = _native_hi(None if hi2 is None else hi2[expert], bits, k2)
        down = ops.prefill_wna16(hid, p2[expert], hi_dn, s2[expert], bits)
        weight = router.reshape(-1)[sel].unsqueeze(-1)
        y.index_add_(0, tok, down.float() * weight)
    return y.to(torch.bfloat16)


def one_case(device, expert_bits, t_tokens, topk, hidden, intermediate, seed, label):
    experts = len(expert_bits)
    storage = max(expert_bits)
    gen = torch.Generator().manual_seed(seed)
    w13 = [
        torch.randn(2 * intermediate, hidden, generator=gen) for _ in range(experts)
    ]
    w2 = [torch.randn(hidden, intermediate, generator=gen) for _ in range(experts)]
    p13, hi13, s13, bits_t = pack_mixed(w13, expert_bits, storage, device)
    p2, hi2, s2, _bits2 = pack_mixed(w2, expert_bits, storage, device)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randn(t_tokens, hidden, dtype=torch.bfloat16)
    ids = torch.randint(0, experts, (t_tokens, topk), generator=g)
    ids[0, 0] = -1
    x = x.to(device)
    ids = ids.to(device)
    router = torch.rand(t_tokens, topk).to(device)
    got = ops.moe_gemv_mixed(
        x, ids, router, p13, hi13, s13, p2, hi2, s2, bits_t, storage
    )
    ref = ref_loop(x, ids, router, p13, hi13, s13, p2, hi2, s2, expert_bits)
    check(
        f"{label} T={t_tokens} E={experts} H={hidden} I={intermediate}",
        got,
        ref,
    )
    if len(set(expert_bits)) == 1:
        uniform = ops.moe_gemv(
            x, ids, router, p13, hi13, s13, p2, hi2, s2, expert_bits[0]
        )
        check(f"{label} vs templated moe_gemv", got, uniform)


def main() -> None:
    device = torch.device("xpu")
    print("so", __import__("os").environ.get("B70_LOWBIT_SO"), flush=True)
    print("== mixed gemv", flush=True)
    one_case(device, [3, 3, 3, 3], 2, 4, 256, 128, 11, "uniform3")
    one_case(device, [2, 3, 2, 3], 2, 4, 256, 128, 12, "mix-small")
    # Checkpoint shape: experts are only int2 and int3, K=2048.
    one_case(device, [2, 3, 2, 3, 3, 2, 2, 3], 2, 8, 2048, 512, 13, "mix-23")
    one_case(device, [2, 3, 4, 2, 3, 4, 3, 2], 2, 8, 2048, 512, 14, "mix-234")
    if FAILS:
        raise SystemExit("MIXED_PROBE_FAIL " + ",".join(FAILS))
    print("MIXED_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
