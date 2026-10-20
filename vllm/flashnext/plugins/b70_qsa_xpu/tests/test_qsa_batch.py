"""Paged batch QSA against flashnext_ref.qsa_cache.QsaPaged.

Eager decode must match the CPU page (ids and int8 codes exact). A
captured decode replays a later cell, including one inactive slot.
mRoPE tests keep pos_base at 0: the device indexes the triple by
absolute position and the CPU reference indexes it by cell.
"""

from __future__ import annotations

import numpy as np
import torch

from b70_qsa_xpu import ops
from b70_qsa_xpu.batch import QsaBatch
from b70_qsa_xpu.state import Geom
from flashnext_ref.qsa_cache import (
    QsaPaged,
    union_cells,
    union_pack,
    write_step_rows,
)
from flashnext_ref.qsa_layer import QsaGeom, gate_fp16


def _pair(cpu: QsaGeom) -> tuple[QsaGeom, Geom]:
    gpu = Geom(
        n_head=cpu.n_head,
        n_kv=cpu.n_kv,
        head_dim=cpu.head_dim,
        n_rot=cpu.n_rot,
        idx_n_head=cpu.idx_n_head,
        idx_dim=cpu.idx_dim,
        idx_block=cpu.idx_block,
        idx_top_k=cpu.idx_top_k,
        rope_base=cpu.rope_base,
        eps=cpu.eps,
        page_size=cpu.page_size,
    )
    return cpu, gpu


def _small():
    return _pair(
        QsaGeom(
            n_head=4,
            n_kv=2,
            head_dim=64,
            n_rot=8,
            idx_n_head=2,
            idx_dim=32,
            idx_block=4,
            idx_top_k=4,
            rope_base=1.0e7,
            eps=1.0e-6,
            page_size=4,
            hidden=16,
        )
    )


def _wide():
    return _pair(
        QsaGeom(
            n_head=4,
            n_kv=2,
            head_dim=256,
            n_rot=64,
            idx_n_head=4,
            idx_dim=128,
            idx_block=4,
            idx_top_k=4,
            rope_base=1.0e7,
            eps=1.0e-6,
            page_size=4,
            hidden=32,
        )
    )


def _to(x, device, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device=device)


def _max_abs(a, b) -> float:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.size == 0:
        return 0.0
    return float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64))))


def _check_close(name: str, got, ref, atol: float) -> None:
    err = _max_abs(got, ref)
    print(f"  {name} max_abs={err:.3e} atol={atol:.1e}", flush=True)
    if err > atol:
        raise SystemExit(f"FAIL {name} max_abs={err}")


def _check_equal(name: str, got, ref) -> None:
    g = np.asarray(got)
    r = np.asarray(ref)
    if g.shape != r.shape or not np.array_equal(g, r):
        print(f"  {name} MISMATCH got={g.shape} ref={r.shape}", flush=True)
        if g.shape == r.shape:
            bad = np.argwhere(g != r)
            print(f"  {name} n_diff={bad.shape[0]} first={bad[:6].tolist()}", flush=True)
            for idx in bad[:3]:
                i = tuple(int(v) for v in idx)
                print(f"  {name} at {i} got={g[i]!r} ref={r[i]!r}", flush=True)
        raise SystemExit(f"FAIL {name}")
    print(f"  {name} exact {g.shape}", flush=True)


def _table(batch: int, max_blocks: int, rng: np.random.Generator, slack: int = 3) -> np.ndarray:
    n = batch * max_blocks
    perm = rng.permutation(n + slack).astype(np.int32)[:n]
    return perm.reshape(batch, max_blocks)


def _gamma(dim: int) -> np.ndarray:
    return (0.8 + 0.4 * np.linspace(0.0, 1.0, dim)).astype(np.float32)


def _make_tok(rng: np.random.Generator, geom: QsaGeom) -> dict:
    def n(*shape):
        return rng.normal(size=shape).astype(np.float32)

    q = n(geom.n_head, geom.head_dim)
    gate = n(geom.n_head, geom.head_dim)
    return {
        "q": q,
        "k": n(geom.n_kv, geom.head_dim),
        "v": n(geom.n_kv, geom.head_dim),
        "q_idx": n(geom.idx_n_head, geom.idx_dim),
        "raw": n(geom.idx_dim),
        "q_full": np.concatenate([q, gate], axis=-1),
    }


def _make_batch(gpu_geom, table, block_size, device, *, pooled=True, pos_base=None) -> QsaBatch:
    batch = QsaBatch(
        gpu_geom,
        table,
        block_size,
        device,
        pooled=pooled,
        pos_base=pos_base,
    )
    return batch


def _bind(batch: QsaBatch, k_norm: np.ndarray, mrope: np.ndarray | None) -> None:
    batch.k_norm.copy_(_to(k_norm, batch.device))
    if mrope is not None:
        batch.set_mrope(mrope)


def _load(batch: QsaBatch, seq: int, tok: dict, pos: int) -> None:
    dev = batch.device
    batch.q[seq].copy_(_to(tok["q"], dev))
    batch.k[seq].copy_(_to(tok["k"], dev))
    batch.v[seq].copy_(_to(tok["v"], dev))
    batch.q_idx[seq].copy_(_to(tok["q_idx"], dev))
    batch.raw[seq].copy_(_to(tok["raw"], dev))
    batch.q_full[seq].copy_(_to(tok["q_full"], dev))
    batch.pos[seq] = int(pos)


def _mirror(cpu: QsaPaged, gpu: QsaBatch) -> QsaPaged:
    torch.xpu.synchronize()
    pg = QsaPaged.create(
        cpu.geom,
        cpu.block_table,
        cpu.layout.block_size,
        pooled=cpu.layout.pooled,
        pos_base=cpu.pos_base,
        mrope=cpu.mrope,
    )
    src = gpu.pages.detach().cpu().numpy()
    if src.shape != pg.pages.shape:
        raise SystemExit(f"FAIL page shape gpu {src.shape} cpu {pg.pages.shape}")
    pg.pages[:] = src
    pg.tail[:] = gpu.tail.detach().cpu().numpy()
    pg.dead[:] = gpu.dead.detach().cpu().numpy()
    pg.block_pos[:] = gpu.block_pos.detach().cpu().numpy()
    pg.n_kv[:] = gpu.length.detach().cpu().numpy()
    pg.pos[:] = gpu.pos.detach().cpu().numpy()
    pg.have_dead[:] = pg.n_kv > 0
    return pg


def _check_spare(cpu: QsaPaged, mirror: QsaPaged, seq: int, n_kv: int) -> None:
    g = cpu.geom
    r = g.idx_block
    if n_kv <= 0 or n_kv % r != 0:
        return
    b = n_kv // r - 1
    lay = cpu.layout
    logical, _state = lay.indexer_addr(b)
    logical_s, _state_s = lay.indexer_addr(b + 1)
    if logical_s >= cpu.max_blocks:
        return
    got = mirror.read_pooled(seq, b + 1)
    ref = cpu.read_pooled(seq, b + 1)
    if logical_s == logical:
        _check_close(f"seq {seq} n_kv {n_kv} spare", got, ref, 1.0e-5)
        _check_close(f"seq {seq} n_kv {n_kv} spare==dead", ref, cpu.dead[seq], 1.0e-5)
        return
    if float(np.max(np.abs(got))) != 0.0 or float(np.max(np.abs(ref))) != 0.0:
        raise SystemExit(
            f"FAIL seq {seq} n_kv {n_kv} spare spilled "
            f"gpu {float(np.max(np.abs(got))):.3e} cpu {float(np.max(np.abs(ref))):.3e}"
        )
    print(f"  seq {seq} n_kv {n_kv} spare stays off the next page", flush=True)


def _compare_cache(cpu: QsaPaged, gpu: QsaBatch, seq: int, n_kv: int) -> None:
    mirror = _mirror(cpu, gpu)
    g = cpu.geom
    for pos in range(n_kv):
        gk, gks, gv, gvs = mirror.read_kv_cell(seq, pos)
        ck, cks, cv, cvs = cpu.read_kv_cell(seq, pos)
        _check_equal(f"seq {seq} pos {pos} k_codes", gk, ck)
        _check_equal(f"seq {seq} pos {pos} v_codes", gv, cv)
        _check_close(f"seq {seq} pos {pos} k_scale", gks, cks, 0.0)
        _check_close(f"seq {seq} pos {pos} v_scale", gvs, cvs, 0.0)
    _check_close(f"seq {seq} tail", mirror.tail[seq], cpu.tail[seq], 0.0)
    _check_close(f"seq {seq} dead", mirror.dead[seq], cpu.dead[seq], 1.0e-5)
    if int(mirror.block_pos[seq]) != int(cpu.block_pos[seq]):
        raise SystemExit(
            f"FAIL seq {seq} block_pos gpu {int(mirror.block_pos[seq])} cpu {int(cpu.block_pos[seq])}"
        )
    print(f"  seq {seq} block_pos {int(cpu.block_pos[seq])}", flush=True)
    if int(mirror.n_kv[seq]) != n_kv:
        raise SystemExit(f"FAIL seq {seq} length gpu {int(mirror.n_kv[seq])} ref {n_kv}")
    if cpu.layout.pooled:
        n_bid = n_kv // g.idx_block
        for b in range(n_bid):
            _check_close(
                f"seq {seq} pooled {b}",
                mirror.read_pooled(seq, b),
                cpu.read_pooled(seq, b),
                1.0e-5,
            )
        _check_spare(cpu, mirror, seq, n_kv)


def _compare_seq(cpu: QsaPaged, gpu: QsaBatch, seq: int, tok: dict, n_kv: int) -> None:
    _compare_cache(cpu, gpu, seq, n_kv)
    g = cpu.geom
    width = g.width_at(n_kv)
    ids = gpu.ids[seq, :width].detach().cpu().numpy()
    _check_equal(f"seq {seq} ids", ids, cpu.select(seq, tok["q_idx"], n_kv))
    _check_close(
        f"seq {seq} scores",
        gpu.scores[seq, :n_kv].detach().cpu().numpy(),
        cpu.scores(seq, tok["q_idx"], n_kv),
        1.0e-4,
    )
    attn = cpu.attend_ids(seq, tok["q"], ids)
    gated = gate_fp16(attn, tok["q_full"])
    _check_close(f"seq {seq} attn", gpu.attn[seq].detach().cpu().numpy(), attn, 2.0e-3)
    _check_close(
        f"seq {seq} gated",
        gpu.gated[seq].detach().float().cpu().numpy(),
        gated,
        2.0e-3,
    )
    k_ref, v_ref = cpu.gather(seq, ids)
    _check_close(
        f"seq {seq} k_f16",
        gpu.k_f16[seq, :width].detach().float().cpu().numpy(),
        k_ref,
        1.0e-3,
    )
    _check_close(
        f"seq {seq} v_f16",
        gpu.v_f16[seq, :width].detach().float().cpu().numpy(),
        v_ref,
        1.0e-3,
    )


def _side(gpu: QsaBatch, seq: int) -> dict:
    torch.xpu.synchronize()
    table = gpu.block_table[seq].detach().cpu().numpy()
    pages = gpu.pages.detach().cpu().numpy()
    return {
        "pages": np.stack([pages[int(p)] for p in table.tolist()]),
        "tail": gpu.tail[seq].detach().cpu().numpy().copy(),
        "dead": gpu.dead[seq].detach().cpu().numpy().copy(),
        "block_pos": int(gpu.block_pos[seq].detach().cpu()),
        "length": int(gpu.length[seq].detach().cpu()),
    }


def _same_side(name: str, got: dict, ref: dict) -> None:
    _check_equal(f"{name} pages", got["pages"], ref["pages"])
    _check_close(f"{name} tail", got["tail"], ref["tail"], 0.0)
    _check_close(f"{name} dead", got["dead"], ref["dead"], 0.0)
    if got["block_pos"] != ref["block_pos"] or got["length"] != ref["length"]:
        raise SystemExit(
            f"FAIL {name} block_pos {got['block_pos']}/{ref['block_pos']} "
            f"length {got['length']}/{ref['length']}"
        )
    print(f"  {name} block_pos {got['block_pos']} length {got['length']}", flush=True)


def test_union_words(device) -> None:
    print("=== union words ===", flush=True)
    ids = np.zeros((8, 4), dtype=np.int32)
    widths = np.array([3, 2, 1, 2, 2, 1, 3, 2], dtype=np.int32)
    ids[0, :3] = (1, 2, 2)
    ids[1, :2] = (4, 5)
    ids[2, :1] = (7,)
    ids[3, :2] = (2, 9)
    ids[4, :2] = (0, 4)
    ids[5, :1] = (6,)
    ids[6, :3] = (9, 3, 1)
    ids[7, :2] = (8, 1)
    ref, count = union_pack(ids, widths)
    step = np.zeros((8, 4), dtype=np.int32)
    step[:, 1] = 32
    step[:, 3] = widths
    ulist = torch.zeros(1, 8 * 4, dtype=torch.int32, device=device)
    ucount = torch.zeros(1, dtype=torch.int32, device=device)
    ops.union_build(_to(ids, device), _to(step, device), ulist, ucount)
    torch.xpu.synchronize()
    got_n = int(ucount.detach().cpu())
    if got_n != count:
        raise SystemExit(f"FAIL union count gpu {got_n} cpu {count}")
    got = ulist[0, :count].detach().cpu().numpy()
    _check_equal("union words", got, ref)
    cells, masks = union_cells(got)
    print(f"  union cells {cells.tolist()} masks {masks.tolist()}", flush=True)
    if int(masks[np.where(cells == 1)[0][0]]) & 0x80 == 0:
        raise SystemExit("FAIL union query-7 bit")
    print("PASS union words", flush=True)


def _run_decode(
    cpu_geom,
    gpu_geom,
    tokens,
    *,
    block_size: int,
    table: np.ndarray,
    pos_base=None,
    mrope=None,
    label: str,
    device,
) -> tuple[QsaPaged, QsaBatch]:
    print(f"=== {label} T={len(tokens[0])} block={block_size} ===", flush=True)
    if mrope is not None and pos_base is not None and int(np.max(pos_base)) != 0:
        raise SystemExit("mrope test requires pos_base 0")
    k_norm = _gamma(cpu_geom.idx_dim)
    cpu = QsaPaged.create(
        cpu_geom,
        table,
        block_size,
        pooled=True,
        pos_base=pos_base,
        mrope=mrope,
    )
    gpu = _make_batch(gpu_geom, table, block_size, device, pooled=True, pos_base=pos_base)
    _bind(gpu, k_norm, mrope)
    batch = table.shape[0]
    for pos in range(len(tokens[0])):
        for seq in range(batch):
            tok = tokens[seq][pos]
            cpu.append_token(seq, tok["k"], tok["v"], tok["raw"], k_norm, pos)
            _load(gpu, seq, tok, pos)
        gpu.run_decode()
        for seq in range(batch):
            _compare_seq(cpu, gpu, seq, tokens[seq][pos], pos + 1)
    print(f"PASS {label}", flush=True)
    return cpu, gpu


def test_inactive(cpu_geom, gpu_geom, device) -> None:
    print("=== inactive slot ===", flush=True)
    rng = np.random.default_rng(11)
    table = _table(2, 4, rng)
    tokens = [[_make_tok(rng, cpu_geom) for _ in range(5)] for _ in range(2)]
    tokens[0][2]["k"][:] = 0
    tokens[0][2]["v"][:] = 0
    k_norm = _gamma(cpu_geom.idx_dim)
    cpu = QsaPaged.create(cpu_geom, table, 4, pooled=True)
    gpu = _make_batch(gpu_geom, table, 4, device, pooled=True)
    _bind(gpu, k_norm, None)
    for pos in range(4):
        for seq in range(2):
            tok = tokens[seq][pos]
            cpu.append_token(seq, tok["k"], tok["v"], tok["raw"], k_norm, pos)
            _load(gpu, seq, tok, pos)
        gpu.run_decode()
    held = _side(gpu, 1)
    tok = tokens[0][4]
    cpu.append_token(0, tok["k"], tok["v"], tok["raw"], k_norm, 4)
    _load(gpu, 0, tok, 4)
    gpu.pos[1] = -1
    gpu.run_decode()
    _compare_seq(cpu, gpu, 0, tok, 5)
    _same_side("inactive neighbor", _side(gpu, 1), held)
    if int(gpu.length[1].detach().cpu()) != 4:
        raise SystemExit("FAIL inactive length")
    print("PASS inactive slot", flush=True)


def test_chunk(cpu_geom, gpu_geom, device) -> None:
    print("=== chunked prefill ===", flush=True)
    rng = np.random.default_rng(12)
    table = _table(1, 4, rng)
    tokens = [_make_tok(rng, cpu_geom) for _ in range(8)]
    k_norm = _gamma(cpu_geom.idx_dim)
    cpu = QsaPaged.create(cpu_geom, table, 8, pooled=True)
    gpu = _make_batch(gpu_geom, table, 8, device, pooled=True)
    _bind(gpu, k_norm, None)
    for pos, tok in enumerate(tokens):
        cpu.append_token(0, tok["k"], tok["v"], tok["raw"], k_norm, pos)
        _load(gpu, 0, tok, pos)
        gpu.run_decode()
    n = len(tokens)
    dev = device
    q_idx = torch.zeros(n, cpu_geom.idx_n_head, cpu_geom.idx_dim, dtype=torch.float32, device=dev)
    q = torch.zeros(n, cpu_geom.n_head, cpu_geom.head_dim, dtype=torch.float32, device=dev)
    pos = torch.arange(n, dtype=torch.int32, device=dev)
    seq_slot = torch.zeros(n, dtype=torch.int32, device=dev)
    scores = torch.zeros(n, gpu.max_cells, dtype=torch.float32, device=dev)
    cap = gpu_geom.width_cap()
    ids = torch.zeros(n, cap, dtype=torch.int32, device=dev)
    step = torch.zeros(n, 4, dtype=torch.int32, device=dev)
    for i, tok in enumerate(tokens):
        q_idx[i].copy_(_to(tok["q_idx"], dev))
        q[i].copy_(_to(tok["q"], dev))
    gpu.score_queries(q_idx, pos, seq_slot, scores, ids, step)
    k_f16 = torch.zeros(n, cap, cpu_geom.n_kv, cpu_geom.head_dim, dtype=torch.float16, device=dev)
    v_f16 = torch.zeros_like(k_f16)
    attn = torch.zeros(n, cpu_geom.n_head, cpu_geom.head_dim, dtype=torch.float32, device=dev)
    gpu.attend_ids(q, ids, seq_slot, step, k_f16, v_f16, attn)
    torch.xpu.synchronize()
    for i, tok in enumerate(tokens):
        n_kv = i + 1
        width = cpu_geom.width_at(n_kv)
        ref_ids = cpu.select(0, tok["q_idx"], n_kv)
        _check_equal(f"chunk {i} ids", ids[i, :width].detach().cpu().numpy(), ref_ids)
        ref_attn = cpu.attend_ids(0, tok["q"], ref_ids)
        _check_close(f"chunk {i} attn", attn[i].detach().cpu().numpy(), ref_attn, 2.0e-3)
    print("PASS chunked prefill", flush=True)
    return cpu, gpu, tokens


def test_union_attend(cpu, gpu, tokens, device) -> None:
    print("=== union attend ===", flush=True)
    cpu_geom = cpu.geom
    n_kv = len(tokens)
    # Three queries over one sequence, all at the final n_kv. Reuse the
    # stored queries plus one fresh vector so the masks differ.
    rng = np.random.default_rng(13)
    qs = [tokens[0]["q"], tokens[3]["q"], rng.normal(size=tokens[0]["q"].shape).astype(np.float32)]
    q_idx = [
        tokens[0]["q_idx"],
        tokens[3]["q_idx"],
        rng.normal(size=tokens[0]["q_idx"].shape).astype(np.float32),
    ]
    n_q = 3
    cap = cpu_geom.width_cap()
    width = cpu_geom.width_at(n_kv)
    ids_np = np.zeros((n_q, cap), dtype=np.int32)
    for i in range(n_q):
        ids_np[i, :width] = cpu.select(0, q_idx[i], n_kv)
    step_np = write_step_rows(np.full((n_q,), n_kv - 1, dtype=np.int32), cpu_geom.idx_top_k, cpu_geom.idx_block)
    ref_words, count = union_pack(ids_np, step_np[:, 3])
    dev = device
    q = torch.zeros(n_q, cpu_geom.n_head, cpu_geom.head_dim, dtype=torch.float32, device=dev)
    for i in range(n_q):
        q[i].copy_(_to(qs[i], dev))
    ids = _to(ids_np, dev)
    step = _to(step_np, dev)
    seq_slot = torch.zeros(n_q, dtype=torch.int32, device=dev)
    ulist = torch.zeros(1, 8 * cap, dtype=torch.int32, device=dev)
    ucount = torch.zeros(1, dtype=torch.int32, device=dev)
    attn = torch.zeros(n_q, cpu_geom.n_head, cpu_geom.head_dim, dtype=torch.float32, device=dev)
    gpu.union(q, ids, step, seq_slot, ulist, ucount, attn)
    torch.xpu.synchronize()
    got_n = int(ucount.detach().cpu())
    if got_n != count:
        raise SystemExit(f"FAIL union attend count gpu {got_n} cpu {count}")
    _check_equal("union attend words", ulist[0, :count].detach().cpu().numpy(), ref_words)
    for i in range(n_q):
        ref = cpu.attend_ids(0, qs[i], ids_np[i, :width])
        _check_close(f"union q{i} attn", attn[i].detach().cpu().numpy(), ref, 2.0e-3)
    print("PASS union attend", flush=True)


def test_draft(cpu_geom, gpu_geom, device) -> None:
    print("=== draft ===", flush=True)
    rng = np.random.default_rng(14)
    table = _table(1, 4, rng)
    tokens = [_make_tok(rng, cpu_geom) for _ in range(6)]
    k_norm = _gamma(cpu_geom.idx_dim)
    ref = QsaPaged.create(cpu_geom, table, 4, pooled=True)
    draft = QsaPaged.create(cpu_geom, table, 4, pooled=False)
    gpu = _make_batch(gpu_geom, table, 4, device, pooled=False)
    _bind(gpu, k_norm, None)
    for pos, tok in enumerate(tokens):
        ref.append_token(0, tok["k"], tok["v"], tok["raw"], k_norm, pos)
        draft.append_token(0, tok["k"], tok["v"], None, None, pos)
        n_kv = pos + 1
        width = cpu_geom.width_at(n_kv)
        ids = ref.select(0, tok["q_idx"], n_kv)
        gpu.ids[0, :width] = _to(ids, device)
        _load(gpu, 0, tok, pos)
        gpu.run_decode()
        _compare_cache(draft, gpu, 0, n_kv)
        attn = ref.attend_ids(0, tok["q"], ids)
        gated = gate_fp16(attn, tok["q_full"])
        _check_close(f"draft pos {pos} attn", gpu.attn[0].detach().cpu().numpy(), attn, 2.0e-3)
        _check_close(
            f"draft pos {pos} gated",
            gpu.gated[0].detach().float().cpu().numpy(),
            gated,
            2.0e-3,
        )
        if gpu.layout["pooled_bytes"] != 0 or gpu.layout["page_bytes"] != gpu.layout["kv_bytes"]:
            raise SystemExit("FAIL draft page layout")
    print("PASS draft", flush=True)


def test_export(cpu_geom, gpu_geom, device) -> None:
    print("=== export import ===", flush=True)
    rng = np.random.default_rng(15)
    table_a = _table(2, 4, rng)
    table_b = _table(2, 4, rng)
    tokens = [[_make_tok(rng, cpu_geom) for _ in range(6)] for _ in range(2)]
    k_norm = _gamma(cpu_geom.idx_dim)
    cpu = QsaPaged.create(cpu_geom, table_a, 4, pooled=True)
    gpu = _make_batch(gpu_geom, table_a, 4, device, pooled=True)
    _bind(gpu, k_norm, None)
    for pos in range(6):
        for seq in range(2):
            tok = tokens[seq][pos]
            cpu.append_token(seq, tok["k"], tok["v"], tok["raw"], k_norm, pos)
            _load(gpu, seq, tok, pos)
        gpu.run_decode()
    blob = gpu.export_seq(0)
    cpu_blob = cpu.export_seq(0)
    if int(blob["n_kv"]) != int(cpu_blob["n_kv"]) or int(blob["pos"]) != int(cpu_blob["pos"]):
        raise SystemExit("FAIL export counts")
    if not np.array_equal(blob["logical_blocks"], cpu_blob["logical_blocks"]):
        raise SystemExit("FAIL export logical blocks")
    _check_close("export tail", blob["tail"], cpu_blob["tail"], 0.0)
    _check_close("export dead", blob["dead"], cpu_blob["dead"], 1.0e-5)
    if int(blob["block_pos"]) != int(cpu_blob["block_pos"]):
        raise SystemExit("FAIL export block_pos")
    # Codes survive the byte copy. Pooled rows are compared after import.
    gpu2 = _make_batch(gpu_geom, table_b, 4, device, pooled=True)
    _bind(gpu2, k_norm, None)
    for pos in range(3):
        tok = tokens[1][pos]
        _load(gpu2, 1, tok, pos)
        gpu2.pos[0] = -1
        gpu2.run_decode()
    neighbor = _side(gpu2, 1)
    gpu2.import_seq(0, blob)
    cpu2 = QsaPaged.create(cpu_geom, table_b, 4, pooled=True)
    cpu2.import_seq(0, cpu_blob)
    _compare_cache(cpu2, gpu2, 0, 6)
    _same_side("import neighbor", _side(gpu2, 1), neighbor)
    nxt = _make_tok(rng, cpu_geom)
    cpu2.append_token(0, nxt["k"], nxt["v"], nxt["raw"], k_norm, 6)
    _load(gpu2, 0, nxt, 6)
    gpu2.pos[1] = -1
    gpu2.run_decode()
    _compare_seq(cpu2, gpu2, 0, nxt, 7)
    _same_side("import neighbor after", _side(gpu2, 1), neighbor)
    print("PASS export import", flush=True)


def _snap(gpu: QsaBatch) -> dict:
    torch.xpu.synchronize()
    return {
        "pages": gpu.pages.detach().cpu().clone(),
        "tail": gpu.tail.detach().cpu().clone(),
        "dead": gpu.dead.detach().cpu().clone(),
        "block_pos": gpu.block_pos.detach().cpu().clone(),
        "length": gpu.length.detach().cpu().clone(),
        "ids": gpu.ids.detach().cpu().clone(),
        "attn": gpu.attn.detach().cpu().clone(),
        "gated": gpu.gated.detach().cpu().clone(),
    }


def _restore(gpu: QsaBatch, snap: dict, names) -> None:
    for name in names:
        getattr(gpu, name).copy_(snap[name].to(device=gpu.device))


def _equal_names(gpu: QsaBatch, snap: dict, names) -> None:
    torch.xpu.synchronize()
    for name in names:
        got = getattr(gpu, name).detach().cpu()
        ref = snap[name]
        if not torch.equal(got, ref):
            if got.dtype.is_floating_point:
                err = float((got.float() - ref.float()).abs().max())
                raise SystemExit(f"FAIL graph {name} max_abs={err}")
            raise SystemExit(f"FAIL graph {name} mismatch")
        print(f"  graph {name} exact", flush=True)


def _equal_ids(gpu: QsaBatch, snap: dict, width: int, label: str) -> None:
    """Compare the live prefix. topk does not write ids past width.

    The snap tail is the zero fill from allocation. A replay that first
    fills the buffer with -1 leaves that tail at -1, so a full-tensor
    compare fails even when the selected cells match.
    """
    torch.xpu.synchronize()
    got = gpu.ids[:, :width].detach().cpu()
    ref = snap["ids"][:, :width]
    if not torch.equal(got, ref):
        raise SystemExit(
            "FAIL graph " + label + " ids\n got " + str(got.tolist()) + "\n ref " + str(ref.tolist())
        )
    print("  graph " + label + " ids exact", flush=True)


def test_graph(cpu_geom, gpu_geom, device) -> None:
    print("=== graph ===", flush=True)
    rng = np.random.default_rng(16)
    table = _table(2, 4, rng)
    tokens = [[_make_tok(rng, cpu_geom) for _ in range(8)] for _ in range(2)]
    k_norm = _gamma(cpu_geom.idx_dim)
    cpu = QsaPaged.create(cpu_geom, table, 4, pooled=True)
    gpu = _make_batch(gpu_geom, table, 4, device, pooled=True)
    _bind(gpu, k_norm, None)
    snaps = {}
    inputs = {}
    cache_names = ("pages", "tail", "dead", "block_pos", "length")
    for pos in range(8):
        for seq in range(2):
            tok = tokens[seq][pos]
            cpu.append_token(seq, tok["k"], tok["v"], tok["raw"], k_norm, pos)
            _load(gpu, seq, tok, pos)
        gpu.run_decode()
        for seq in range(2):
            _compare_seq(cpu, gpu, seq, tokens[seq][pos], pos + 1)
        if pos in (3, 4, 5, 6, 7):
            snaps[pos] = _snap(gpu)
            inputs[pos] = {
                seq: {key: tokens[seq][pos][key].copy() for key in ("q", "k", "v", "q_idx", "raw", "q_full")}
                for seq in range(2)
            }
    _restore(gpu, snaps[3], cache_names)

    def _load_pos(pos: int, seqs) -> None:
        for seq in range(2):
            if seq not in seqs:
                gpu.pos[seq] = -1
                continue
            tok = inputs[pos][seq]
            dev = gpu.device
            gpu.q[seq].copy_(_to(tok["q"], dev))
            gpu.k[seq].copy_(_to(tok["k"], dev))
            gpu.v[seq].copy_(_to(tok["v"], dev))
            gpu.q_idx[seq].copy_(_to(tok["q_idx"], dev))
            gpu.raw[seq].copy_(_to(tok["raw"], dev))
            gpu.q_full[seq].copy_(_to(tok["q_full"], dev))
            gpu.pos[seq] = int(pos)

    _load_pos(4, (0, 1))
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        gpu.run_decode()
    torch.xpu.synchronize()
    _restore(gpu, snaps[3], cache_names)
    _load_pos(4, (0, 1))
    gpu.ids.fill_(-1)
    gpu.attn.fill_(float("nan"))
    graph.replay()
    torch.xpu.synchronize()
    _equal_names(gpu, snaps[4], cache_names + ("attn", "gated"))
    _equal_ids(gpu, snaps[4], cpu_geom.width_at(5), "pos 4")
    _load_pos(5, (0, 1))
    gpu.ids.fill_(-1)
    graph.replay()
    torch.xpu.synchronize()
    _equal_names(gpu, snaps[5], cache_names + ("attn", "gated"))
    _equal_ids(gpu, snaps[5], cpu_geom.width_at(6), "pos 5")
    # Captured on slot 0. Replay a slot-3 completion on seq 0 and an
    # inactive seq 1, which must keep the post-pos-6 cache.
    _restore(gpu, snaps[6], cache_names)
    _load_pos(7, (0,))
    gpu.ids.fill_(-1)
    graph.replay()
    torch.xpu.synchronize()
    seq0_pages = gpu.block_table[0].detach().cpu().numpy()
    got_pages = gpu.pages.detach().cpu()
    ref_pages = snaps[7]["pages"]
    for phys in seq0_pages.tolist():
        if not torch.equal(got_pages[phys], ref_pages[phys]):
            raise SystemExit(f"FAIL graph seq0 page {phys}")
    print("  graph seq0 pages exact", flush=True)
    _check_close("graph seq0 tail", gpu.tail[0].detach().cpu().numpy(), snaps[7]["tail"][0].numpy(), 0.0)
    _check_close("graph seq0 dead", gpu.dead[0].detach().cpu().numpy(), snaps[7]["dead"][0].numpy(), 0.0)
    if int(gpu.block_pos[0].detach().cpu()) != int(snaps[7]["block_pos"][0]):
        raise SystemExit("FAIL graph seq0 block_pos")
    if int(gpu.length[0].detach().cpu()) != int(snaps[7]["length"][0]):
        raise SystemExit("FAIL graph seq0 length")
    width = cpu_geom.width_at(8)
    _check_equal("graph seq0 ids", gpu.ids[0, :width].detach().cpu().numpy(), snaps[7]["ids"][0, :width].numpy())
    _check_close("graph seq0 attn", gpu.attn[0].detach().cpu().numpy(), snaps[7]["attn"][0].numpy(), 0.0)
    _check_close("graph seq0 gated", gpu.gated[0].detach().cpu().numpy(), snaps[7]["gated"][0].numpy(), 0.0)
    seq1_pages = gpu.block_table[1].detach().cpu().numpy()
    ref6 = snaps[6]["pages"]
    for phys in seq1_pages.tolist():
        if not torch.equal(got_pages[phys], ref6[phys]):
            raise SystemExit(f"FAIL graph inactive page {phys}")
    print("  graph inactive pages exact", flush=True)
    _check_close("graph inactive tail", gpu.tail[1].detach().cpu().numpy(), snaps[6]["tail"][1].numpy(), 0.0)
    _check_close("graph inactive dead", gpu.dead[1].detach().cpu().numpy(), snaps[6]["dead"][1].numpy(), 0.0)
    if int(gpu.block_pos[1].detach().cpu()) != int(snaps[6]["block_pos"][1]):
        raise SystemExit("FAIL graph inactive block_pos")
    if int(gpu.length[1].detach().cpu()) != int(snaps[6]["length"][1]):
        raise SystemExit("FAIL graph inactive length")
    print("PASS graph", flush=True)


def main() -> None:
    print("xpu", torch.xpu.device_count(), flush=True)
    device = torch.device("xpu")
    cpu_geom, gpu_geom = _small()
    test_union_words(device)
    rng = np.random.default_rng(21)
    tokens = [[_make_tok(rng, cpu_geom) for _ in range(8)] for _ in range(2)]
    tokens[1][1]["k"][:] = 0
    tokens[1][1]["v"][:] = 0
    table4 = _table(2, 4, rng)
    _run_decode(
        cpu_geom,
        gpu_geom,
        tokens,
        block_size=4,
        table=table4,
        label="batch block 4",
        device=device,
    )
    table8 = _table(2, 4, rng)
    _run_decode(
        cpu_geom,
        gpu_geom,
        tokens,
        block_size=8,
        table=table8,
        label="batch block 8",
        device=device,
    )
    base = np.array([100, 40], dtype=np.int32)
    _run_decode(
        cpu_geom,
        gpu_geom,
        [[row[i] for i in range(6)] for row in tokens],
        block_size=4,
        table=_table(2, 4, rng),
        pos_base=base,
        label="pos_base",
        device=device,
    )
    mrope = np.zeros((2, 32, 3), dtype=np.int32)
    for s in range(2):
        for i in range(32):
            mrope[s, i] = ((i + s) % 5, (i + 1) % 7, (i * 3) % 11)
    _run_decode(
        cpu_geom,
        gpu_geom,
        tokens,
        block_size=4,
        table=_table(2, 4, rng),
        pos_base=np.zeros((2,), dtype=np.int32),
        mrope=mrope,
        label="mrope",
        device=device,
    )
    test_inactive(cpu_geom, gpu_geom, device)
    cpu_chunk, gpu_chunk, chunk_tokens = test_chunk(cpu_geom, gpu_geom, device)
    test_union_attend(cpu_chunk, gpu_chunk, chunk_tokens, device)
    test_draft(cpu_geom, gpu_geom, device)
    test_export(cpu_geom, gpu_geom, device)
    test_graph(cpu_geom, gpu_geom, device)
    wide_cpu, wide_gpu = _wide()
    wrng = np.random.default_rng(22)
    wide_tokens = [[_make_tok(wrng, wide_cpu) for _ in range(4)] for _ in range(2)]
    _run_decode(
        wide_cpu,
        wide_gpu,
        wide_tokens,
        block_size=4,
        table=_table(2, 3, wrng),
        label="head_dim 256",
        device=device,
    )
    print("QSA_BATCH_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"QSA_BATCH_FAIL {type(exc).__name__}: {exc}", flush=True)
        raise
