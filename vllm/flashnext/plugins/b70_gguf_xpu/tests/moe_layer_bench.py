"""Fused MoE layer correctness and device time. Run under bin/gpu-run --card 0.

Prints a2 (sync each call) and behind (calls queued behind a matmul) so the
host gap is not counted as kernel time. Reference is a fp32 dequant matmul.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
sys.path.insert(0, str(ROOT))

import gguf  # noqa: E402

import torch  # noqa: E402

from b70_gguf_xpu import moe_gemv, moe_layer  # noqa: E402
from b70_gguf_xpu.types import NAMES, row_bytes  # noqa: E402
from gpu_check import SHARD, close_enough, cpu_dequant, load_cpu  # noqa: E402

ROOF = 608.0


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def pack_experts(tensor, e_use: int):
    data = np.asarray(tensor.data)
    k = int(tensor.shape[0])
    packed = np.ascontiguousarray(data.reshape(-1, data.shape[-2], data.shape[-1]))
    e_use = min(e_use, packed.shape[0])
    blob = np.ascontiguousarray(packed[:e_use])
    return blob, k, e_use


def find_experts(reader):
    found = {}
    for tensor in reader.tensors:
        tid = int(tensor.tensor_type)
        if tid not in NAMES:
            continue
        data = np.asarray(tensor.data)
        if data.ndim < 3:
            continue
        n = data.shape[-2]
        k = int(tensor.shape[0])
        key = (tid, n, k, "gate" if n < k else "down")
        if key not in found:
            found[key] = tensor
    return found


def ref_layer(lib, gate, gty, up, uty, down, dty, x, ids, router, sg, sgty, su, suty, sd, sdty):
    t, k = x.shape
    topk = ids.shape[1]
    i = gate.shape[1]
    y = np.zeros((t, k), np.float32)
    slots_g = [cpu_dequant(lib, gty, gate[e], k) for e in range(gate.shape[0])]
    slots_u = [cpu_dequant(lib, uty, up[e], k) for e in range(up.shape[0])]
    slots_d = [cpu_dequant(lib, dty, down[e], i) for e in range(down.shape[0])]
    sg_w = cpu_dequant(lib, sgty, sg, k) if sg is not None else None
    su_w = cpu_dequant(lib, suty, su, k) if su is not None else None
    sd_w = cpu_dequant(lib, sdty, sd, i) if sd is not None else None
    for ti in range(t):
        acc = np.zeros((k,), np.float32)
        for s in range(topk):
            e = int(ids[ti, s])
            h = silu(slots_g[e] @ x[ti]) * (slots_u[e] @ x[ti])
            acc += router[ti, s] * (slots_d[e] @ h)
        if sg_w is not None:
            h = silu(sg_w @ x[ti]) * (su_w @ x[ti])
            acc += sd_w @ h
        y[ti] = acc
    return y


def time_a2(fn, iters=8) -> float:
    for _ in range(2):
        fn()
    torch.xpu.synchronize()
    samples = []
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return min(samples)


def time_behind(fn, iters=4) -> float:
    burn_a = torch.randn(2048, 2048, device="xpu")
    burn_b = torch.randn(2048, 2048, device="xpu")
    for _ in range(2):
        (burn_a @ burn_b).sum()
        fn()
    torch.xpu.synchronize()
    samples = []
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    for _ in range(iters):
        (burn_a @ burn_b).sum()
        start.record()
        fn()
        end.record()
    torch.xpu.synchronize()
    # The events above are not paired per iter after the fact; retime one-by-one
    # but keep the burn in the same queue ahead of the start tag.
    for _ in range(iters):
        (burn_a @ burn_b).sum()
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return min(samples)


def main() -> int:
    assert torch.xpu.device_count() == 1
    lib = load_cpu()
    reader = gguf.GGUFReader(SHARD)
    found = find_experts(reader)
    # Prefer an IQ gate (N=640, K=2560) and a Q2_0 or IQ4_NL down.
    gate_key = None
    down_key = None
    for key in found:
        tid, n, k, kind = key
        if kind == "gate" and n == 640 and k == 2560 and gate_key is None:
            gate_key = key
        if kind == "down" and n == 2560 and k == 640 and down_key is None:
            down_key = key
    if gate_key is None or down_key is None:
        print("FAIL missing flash-next expert shapes", list(found))
        return 1
    g_tid = gate_key[0]
    d_tid = down_key[0]
    # Up may be a different IQ type with the same shape. Use the gate tensor
    # for both projections when a second type is not required for the op
    # (the kernel still takes two blobs). Also pick a different type if one exists.
    up_key = gate_key
    for key in found:
        if key[1] == 640 and key[2] == 2560 and key[0] != g_tid:
            up_key = key
            break
    u_tid = up_key[0]
    print(f"LAYER gate={NAMES[g_tid]} up={NAMES[u_tid]} down={NAMES[d_tid]}")

    def load_pack(key, e_use):
        blob, k, e = pack_experts(found[key], e_use)
        return blob, k, e

    failures = 0
    # Correctness on a small expert slice, with and without a shared expert.
    gate_b, _, e = load_pack(gate_key, 4)
    up_b, _, _ = load_pack(up_key, 4)
    down_b, _, _ = load_pack(down_key, 4)
    i_dim = gate_b.shape[1]
    k_dim = int(found[gate_key].shape[0])
    rng = np.random.default_rng(1)
    for t_tok, topk, shared in ((1, 4, False), (2, 4, True), (8, 4, False)):
        if topk > e:
            continue
        x_np = rng.standard_normal((t_tok, k_dim)).astype(np.float32)
        x_np = torch.from_numpy(x_np).to(dtype=torch.bfloat16).float().numpy()
        ids = rng.integers(0, e, size=(t_tok, topk), dtype=np.int32)
        router = rng.random((t_tok, topk)).astype(np.float32)
        sg = su = sd = None
        sgty = suty = sdty = 0
        if shared:
            sg = gate_b[0]
            su = up_b[1 % e]
            sd = down_b[0]
            sgty, suty, sdty = g_tid, u_tid, d_tid
        ref = ref_layer(lib, gate_b, g_tid, up_b, u_tid, down_b, d_tid, x_np, ids, router, sg, sgty, su, suty, sd, sdty)
        x = torch.from_numpy(x_np).to(device="xpu", dtype=torch.bfloat16)
        y = moe_layer(
            x,
            torch.from_numpy(gate_b.reshape(-1)).to(device="xpu"),
            g_tid,
            torch.from_numpy(up_b.reshape(-1)).to(device="xpu"),
            u_tid,
            torch.from_numpy(down_b.reshape(-1)).to(device="xpu"),
            d_tid,
            torch.from_numpy(ids).to(device="xpu"),
            torch.from_numpy(router).to(device="xpu"),
            i_dim,
            k_dim,
            None if sg is None else torch.from_numpy(np.ascontiguousarray(sg)).to(device="xpu"),
            sgty,
            None if su is None else torch.from_numpy(np.ascontiguousarray(su)).to(device="xpu"),
            suty,
            None if sd is None else torch.from_numpy(np.ascontiguousarray(sd)).to(device="xpu"),
            sdty,
        )
        torch.xpu.synchronize()
        ok, mx, mr = close_enough(y.cpu().numpy(), ref, atol=8e-2, rtol=8e-2)
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"{status} moe_layer T={t_tok} topk={topk} shared={int(shared)} max_abs={mx:.3e} max_rel={mr:.3e}")

    # Device time at the real shape. E=32, topk=10. T=1 and T=2..8.
    gate_b, _, e = load_pack(gate_key, 32)
    up_b, _, _ = load_pack(up_key, 32)
    down_b, _, _ = load_pack(down_key, 32)
    gq = torch.from_numpy(gate_b.reshape(-1)).to(device="xpu")
    uq = torch.from_numpy(up_b.reshape(-1)).to(device="xpu")
    dq = torch.from_numpy(down_b.reshape(-1)).to(device="xpu")
    # Shared expert: one real row-matrix of the same types.
    sg = torch.from_numpy(np.ascontiguousarray(gate_b[0])).to(device="xpu")
    su = torch.from_numpy(np.ascontiguousarray(up_b[0])).to(device="xpu")
    sd = torch.from_numpy(np.ascontiguousarray(down_b[0])).to(device="xpu")
    gate_bytes = 10 * gate_b.shape[1] * gate_b.shape[2]
    up_bytes = 10 * up_b.shape[1] * up_b.shape[2]
    down_bytes = 10 * down_b.shape[1] * down_b.shape[2]
    shared_bytes = gate_b.shape[2] * gate_b.shape[1] + up_b.shape[2] * up_b.shape[1] + down_b.shape[2] * down_b.shape[1]
    # row bytes are the last dim. Weight bytes touched: topk * (gate+up+down row bytes * N) + shared.
    weight = gate_bytes + up_bytes + down_bytes + shared_bytes
    print(f"WEIGHT_BYTES routed+shared={weight} gate_rb={gate_b.shape[2]} down_rb={down_b.shape[2]}")
    for t_tok in (1, 2, 4, 8):
        x = torch.randn(t_tok, k_dim, device="xpu", dtype=torch.bfloat16)
        ids = torch.randint(0, e, (t_tok, 10), device="xpu", dtype=torch.int32)
        router = torch.rand(t_tok, 10, device="xpu")

        def run(x=x, ids=ids, router=router):
            return moe_layer(x, gq, g_tid, uq, u_tid, dq, d_tid, ids, router, i_dim, k_dim, sg, g_tid, su, u_tid, sd, d_tid)

        a2 = time_a2(run)
        behind = time_behind(run)
        # Single projection, same timer, so the fused call can be compared
        # with three moe_gemv launches. Gate only, T tokens, top-10.
        def run_gate(x=x, ids=ids):
            return moe_gemv(x, gq, ids, g_tid, i_dim, k_dim)

        gate_a2 = time_a2(run_gate)
        gate_behind = time_behind(run_gate)
        print(
            f"BENCH moe_gemv {NAMES[g_tid]} T={t_tok} topk=10 N={i_dim} K={k_dim} "
            f"a2_us={gate_a2:.1f} behind_us={gate_behind:.1f}"
        )
        gbps = (weight / 1e9) / (behind * 1e-6)
        print(
            f"BENCH moe_layer T={t_tok} topk=10 E={e} I={i_dim} K={k_dim} "
            f"a2_us={a2:.1f} behind_us={behind:.1f} GB/s={gbps:.1f} pct={100.0 * gbps / ROOF:.1f}"
        )
    # Touch row_bytes so an import mismatch fails loudly.
    if row_bytes(g_tid, k_dim) != gate_b.shape[2]:
        print("FAIL row_bytes")
        failures += 1
    print(f"VERDICT failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
