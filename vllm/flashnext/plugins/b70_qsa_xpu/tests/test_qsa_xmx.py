"""Mode-1 XMX union attention against the SIMT union and the CPU page.

24/2/256 launches. Any other head shape returns ok=0 and does not write
attn. Q, K, V, and P are FP16 in the joint_matrix products, so this is
not a bitwise compare. A NaN fails. The SIMT union remains the
correctness path when ok is 0.
"""

from __future__ import annotations

import numpy as np
import torch

from b70_qsa_xpu import ops
from flashnext_ref.qsa_cache import QsaPaged
from flashnext_ref.qsa_layer import QsaGeom
from test_qsa_batch import (
    _bind,
    _check_close,
    _compare_cache,
    _gamma,
    _load,
    _make_batch,
    _make_tok,
    _pair,
    _table,
    _to,
)
from flashnext_ref.qsa_layer import gate_fp16


def test_refuse(device) -> None:
    q = torch.zeros(2, 4, 64, dtype=torch.float32, device=device)
    pages = torch.zeros(1, 16, dtype=torch.uint8, device=device)
    table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    slot = torch.zeros(2, dtype=torch.int32, device=device)
    ulist = torch.zeros(1, 8, dtype=torch.int32, device=device)
    ucount = torch.zeros(1, dtype=torch.int32, device=device)
    attn = torch.zeros_like(q)
    step = torch.zeros(2, 4, dtype=torch.int32, device=device)
    ok = ops.union_attend_xmx(q, pages, table, slot, ulist, ucount, attn, step, 4, 4, 32, 0)
    torch.xpu.synchronize()
    if int(ok.item()) != 0:
        raise SystemExit(f"FAIL xmx refuse ok={int(ok.item())}")
    print("PASS xmx refuse", flush=True)


def test_match(device) -> None:
    print("=== xmx 24/2/256 ===", flush=True)
    cpu_geom, gpu_geom = _pair(
        QsaGeom(
            n_head=24,
            n_kv=2,
            head_dim=256,
            n_rot=64,
            idx_n_head=4,
            idx_dim=128,
            idx_block=4,
            idx_top_k=32,
            rope_base=1.0e7,
            eps=1.0e-6,
            page_size=4,
            hidden=64,
        )
    )
    rng = np.random.default_rng(31)
    n_tok = 64
    tokens = [_make_tok(rng, cpu_geom) for _ in range(n_tok)]
    table = _table(1, 4, rng)
    k_norm = _gamma(cpu_geom.idx_dim)
    cpu = QsaPaged.create(cpu_geom, table, 16, pooled=True)
    gpu = _make_batch(gpu_geom, table, 16, device, pooled=True)
    _bind(gpu, k_norm, None)
    for pos, tok in enumerate(tokens):
        cpu.append_token(0, tok["k"], tok["v"], tok["raw"], k_norm, pos)
        _load(gpu, 0, tok, pos)
        gpu.run_decode()
    # Staggered selections so the 8-query union crosses a 32-cell chunk.
    n_q = 8
    width = 16
    dev = device
    q = torch.zeros(n_q, cpu_geom.n_head, cpu_geom.head_dim, dtype=torch.float32, device=dev)
    cap = gpu_geom.width_cap()
    ids_np = np.zeros((n_q, cap), dtype=np.int32)
    for i in range(n_q):
        q[i].copy_(_to(rng.normal(size=(cpu_geom.n_head, cpu_geom.head_dim)).astype(np.float32), dev))
        start = i * 6
        ids_np[i, :width] = np.arange(start, start + width, dtype=np.int32)
    ids = _to(ids_np, dev)
    seq_slot = torch.zeros(n_q, dtype=torch.int32, device=dev)
    step = torch.zeros(n_q, 4, dtype=torch.int32, device=dev)
    step[:, 0] = n_tok - 1
    step[:, 1] = n_tok
    step[:, 2] = n_tok // cpu_geom.idx_block
    step[:, 3] = width
    tiles = 1
    ulist = torch.empty(tiles, k_tile_stride(cap), dtype=torch.int32, device=dev)
    ucount = torch.empty(tiles, dtype=torch.int32, device=dev)
    attn = torch.zeros(n_q, cpu_geom.n_head, cpu_geom.head_dim, dtype=torch.float32, device=dev)
    gpu.union(q, ids, step, seq_slot, ulist, ucount, attn)
    attn_x = torch.zeros_like(attn)
    ok = ops.union_attend_xmx(
        q,
        gpu.pages,
        gpu.block_table,
        seq_slot,
        ulist,
        ucount,
        attn_x,
        step,
        gpu.block_size,
        cpu_geom.idx_block,
        cpu_geom.idx_dim,
        1,
    )
    torch.xpu.synchronize()
    launched = int(ok.item())
    count = int(ucount.item())
    print(f"  xmx ok={launched} union_cells={count}", flush=True)
    if launched != 1:
        raise SystemExit("FAIL xmx did not launch")
    if count < 32:
        raise SystemExit(f"FAIL xmx union shorter than one chunk ({count})")
    got = attn_x.detach().cpu().numpy()
    sim = attn.detach().cpu().numpy()
    if not np.isfinite(got).all():
        raise SystemExit("FAIL xmx nan")
    _check_close("xmx vs simt", got, sim, 2.0e-2)
    worst = 0.0
    for i in range(n_q):
        ref = cpu.attend_ids(0, q[i].detach().cpu().numpy(), ids_np[i, :width])
        err = float(np.max(np.abs(got[i].astype(np.float64) - ref.astype(np.float64))))
        worst = max(worst, err)
    print(f"  xmx vs cpu max_abs={worst:.3e} atol=5.0e-02", flush=True)
    if worst > 5.0e-2:
        raise SystemExit(f"FAIL xmx vs cpu max_abs={worst}")
    print("PASS xmx", flush=True)


def k_tile_stride(cap: int) -> int:
    return 8 * cap


def test_prefill_xmx(device) -> None:
    """10 queries over 16 appended cells. Tiles are 8 then 2. XMX must launch.

    Query i is scored at n_kv=i+1, so a tile that ignored the mask would
    attend later cells and miss the CPU gated output.
    """
    print("=== prefill xmx tiles ===", flush=True)
    cpu_geom, gpu_geom = _pair(
        QsaGeom(
            n_head=24,
            n_kv=2,
            head_dim=256,
            n_rot=64,
            idx_n_head=4,
            idx_dim=128,
            idx_block=4,
            idx_top_k=32,
            rope_base=1.0e7,
            eps=1.0e-6,
            page_size=4,
            hidden=64,
        )
    )
    rng = np.random.default_rng(32)
    n_tok = 16
    n_q = 10
    tokens = [_make_tok(rng, cpu_geom) for _ in range(n_tok)]
    table = _table(1, 4, rng)
    k_norm = _gamma(cpu_geom.idx_dim)
    cpu = QsaPaged.create(cpu_geom, table, 16, pooled=True)
    gpu = _make_batch(gpu_geom, table, 16, device, pooled=True)
    _bind(gpu, k_norm, None)
    for pos, tok in enumerate(tokens):
        cpu.append_token(0, tok["k"], tok["v"], tok["raw"], k_norm, pos)
        gpu.pos.fill_(-1)
        _load(gpu, 0, tok, pos)
        gpu.append_cells()
    _compare_cache(cpu, gpu, 0, n_tok)
    dev = device
    q = torch.zeros(n_q, cpu_geom.n_head, cpu_geom.head_dim, dtype=torch.float32, device=dev)
    q_idx = torch.zeros(n_q, cpu_geom.idx_n_head, cpu_geom.idx_dim, dtype=torch.float32, device=dev)
    q_full = torch.zeros(n_q, cpu_geom.n_head, cpu_geom.head_dim * 2, dtype=torch.float32, device=dev)
    for i in range(n_q):
        q[i].copy_(_to(tokens[i]["q"], dev))
        q_idx[i].copy_(_to(tokens[i]["q_idx"], dev))
        q_full[i].copy_(_to(tokens[i]["q_full"], dev))
    pos = torch.arange(n_q, dtype=torch.int32, device=dev)
    gated = gpu.prefill_chunk(0, q, q_idx, q_full, pos)
    torch.xpu.synchronize()
    if not gpu.union_xmx:
        raise SystemExit("FAIL prefill xmx did not launch")
    got = gated.detach().cpu().numpy()
    if not np.isfinite(got).all():
        raise SystemExit("FAIL prefill xmx nan")
    worst = 0.0
    for i in range(n_q):
        ids = cpu.select(0, tokens[i]["q_idx"], i + 1)
        attn = cpu.attend_ids(0, tokens[i]["q"], ids)
        ref = gate_fp16(attn, tokens[i]["q_full"])
        err = float(np.max(np.abs(got[i].astype(np.float64) - ref.astype(np.float64))))
        worst = max(worst, err)
    print(f"  prefill xmx gated vs cpu max_abs={worst:.3e} atol=5.0e-02", flush=True)
    if worst > 5.0e-2:
        raise SystemExit(f"FAIL prefill xmx vs cpu max_abs={worst}")
    print("PASS prefill xmx", flush=True)


def main() -> None:
    print("xpu", torch.xpu.device_count(), flush=True)
    device = torch.device("xpu")
    test_refuse(device)
    test_match(device)
    test_prefill_xmx(device)
    print("QSA_XMX_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"QSA_XMX_FAIL {type(exc).__name__}: {exc}", flush=True)
        raise
