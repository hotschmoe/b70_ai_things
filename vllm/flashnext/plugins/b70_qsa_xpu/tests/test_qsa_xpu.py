"""One-card QSA check against flashnext_ref.qsa_layer.

Eager tokens must match the CPU reference (ids and int8 codes exact).
An XPU graph captured at one position must replay a later position.
"""

from __future__ import annotations

import numpy as np
import torch

from b70_qsa_xpu import ops
from b70_qsa_xpu.state import Geom, QsaBuffers
from flashnext_ref.qsa_layer import (
    QsaGeom,
    QsaState,
    attend_ids,
    gate_fp16,
    topk_radix,
)


def _small():
    cpu = QsaGeom(
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


def _wide():
    cpu = QsaGeom(
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


def _to(x, device, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device=device)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _check_close(name: str, got, ref, atol: float) -> None:
    err = _max_abs(np.asarray(got), np.asarray(ref))
    print(f"  {name} max_abs={err:.3e} atol={atol:.1e}", flush=True)
    if err > atol:
        raise SystemExit(f"FAIL {name} max_abs={err}")


def _check_equal(name: str, got, ref) -> None:
    g = np.asarray(got)
    r = np.asarray(ref)
    if g.shape != r.shape or not np.array_equal(g, r):
        print(
            f"  {name} MISMATCH shape got={g.shape} ref={r.shape} dtype {g.dtype} {r.dtype}",
            flush=True,
        )
        if g.shape == r.shape:
            bad = np.argwhere(g != r)
            print(f"  {name} n_diff={bad.shape[0]} first={bad[:8].tolist()}", flush=True)
            for idx in bad[:4]:
                i = tuple(int(v) for v in idx)
                print(f"  {name} at {i} got={g[i]} ref={r[i]}", flush=True)
        else:
            print(f"  {name} flat got={g.reshape(-1)[:8]} ref={r.reshape(-1)[:8]}", flush=True)
        raise SystemExit(f"FAIL {name}")
    print(f"  {name} exact {g.shape}", flush=True)


class _Static:
    def __init__(self, geom: Geom, device) -> None:
        g = geom
        self.q = torch.zeros(g.n_head, g.head_dim, dtype=torch.float32, device=device)
        self.k = torch.zeros(g.n_kv, g.head_dim, dtype=torch.float32, device=device)
        self.v = torch.zeros(g.n_kv, g.head_dim, dtype=torch.float32, device=device)
        self.q_idx = torch.zeros(g.idx_n_head, g.idx_dim, dtype=torch.float32, device=device)
        self.raw = torch.zeros(g.idx_dim, dtype=torch.float32, device=device)
        self.q_full = torch.zeros(g.n_head, g.head_dim * 2, dtype=torch.float32, device=device)

    def load(self, tok: dict, pos: int, buf: QsaBuffers) -> None:
        dev = buf.device
        self.q.copy_(_to(tok["q"], dev))
        self.k.copy_(_to(tok["k"], dev))
        self.v.copy_(_to(tok["v"], dev))
        self.q_idx.copy_(_to(tok["q_idx"], dev))
        self.raw.copy_(_to(tok["raw"], dev))
        self.q_full.copy_(_to(tok["q_full"], dev))
        buf.pos.fill_(int(pos))


def _make_tok(rng, geom: QsaGeom) -> dict:
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


def _bind(cpu: QsaState, buf: QsaBuffers, k_norm: np.ndarray) -> None:
    dev = buf.device
    buf.cos.copy_(_to(cpu.cos_tab, dev))
    buf.sin.copy_(_to(cpu.sin_tab, dev))
    buf.k_norm.copy_(_to(k_norm, dev))
    buf.page_table.copy_(_to(cpu.page_table, dev))
    if cpu.mrope is not None:
        buf.set_mrope(torch.from_numpy(np.ascontiguousarray(cpu.mrope)))


def _compare_token(pos: int, cpu: QsaState, buf: QsaBuffers, tok: dict, n_kv: int) -> None:
    torch.xpu.synchronize()
    g = cpu.geom
    width = g.width_at(n_kv)
    ids = buf.ids[:width].detach().cpu().numpy()
    _check_equal(f"pos {pos} ids", ids, cpu_ids_of(cpu, tok, n_kv))
    _check_equal(f"pos {pos} k_codes", buf.k_codes.detach().cpu().numpy(), cpu.k_q)
    _check_equal(f"pos {pos} v_codes", buf.v_codes.detach().cpu().numpy(), cpu.v_q)
    _check_close(f"pos {pos} k_scale", buf.k_scale.detach().cpu().numpy(), cpu.k_scale, 0.0)
    _check_close(f"pos {pos} v_scale", buf.v_scale.detach().cpu().numpy(), cpu.v_scale, 0.0)
    _check_close(f"pos {pos} tail", buf.tail.detach().cpu().numpy(), cpu.tail, 0.0)
    _check_close(f"pos {pos} dead", buf.dead.detach().cpu().numpy(), cpu.dead, 1.0e-5)
    _check_close(f"pos {pos} pooled", buf.pooled.detach().cpu().numpy(), cpu.pooled, 1.0e-5)
    bpos = int(buf.block_pos.detach().cpu().numpy()[0])
    if int(cpu.block_pos) != bpos:
        raise SystemExit(f"FAIL pos {pos} block_pos gpu {bpos} cpu {int(cpu.block_pos)}")
    scores = cpu.scores(tok["q_idx"], n_kv)
    _check_close(f"pos {pos} scores", buf.scores[:n_kv].detach().cpu().numpy(), scores, 1.0e-4)
    attn, _w = attend_ids(tok["q"], cpu, cpu.select(tok["q_idx"], n_kv))
    gated = gate_fp16(attn, tok["q_full"])
    _check_close(f"pos {pos} attn", buf.attn.detach().cpu().numpy(), attn, 2.0e-3)
    got_g = buf.gated[:].detach().cpu().numpy().astype(np.float32)
    _check_close(f"pos {pos} gated", got_g, gated, 2.0e-3)
    k_ref, v_ref = cpu.gather(cpu.select(tok["q_idx"], n_kv))
    _check_close(f"pos {pos} k_f16", buf.k_f16[:width].detach().float().cpu().numpy(), k_ref, 1.0e-3)
    _check_close(f"pos {pos} v_f16", buf.v_f16[:width].detach().float().cpu().numpy(), v_ref, 1.0e-3)


def cpu_ids_of(cpu: QsaState, tok: dict, n_kv: int):
    return cpu.select(tok["q_idx"], n_kv)


def _run_seq(cpu_geom, gpu_geom, tokens, *, pos_base=0, mrope=None, page_perm=None, label="seq"):
    print(f"=== {label} T={len(tokens)} pos_base={pos_base} ===", flush=True)
    device = torch.device("xpu")
    cpu = QsaState(max(32, len(tokens) + 4), cpu_geom, pos_base=pos_base)
    if mrope is not None:
        cpu.set_mrope(mrope)
    if page_perm is not None:
        cpu.page_table[:] = page_perm
    buf = QsaBuffers(cpu.max_cells, gpu_geom, device, pos_base=pos_base)
    k_norm = np.ones((cpu_geom.idx_dim,), dtype=np.float32)
    # Non-trivial gamma so RMS is not a no-op on the scale.
    k_norm = (0.8 + 0.4 * np.linspace(0, 1, cpu_geom.idx_dim)).astype(np.float32)
    _bind(cpu, buf, k_norm)
    static = _Static(gpu_geom, device)
    saved = []
    for pos, tok in enumerate(tokens):
        cpu.append_kv(tok["k"], tok["v"], pos)
        cpu.append_indexer(tok["raw"], k_norm, pos)
        static.load(tok, pos, buf)
        buf.run(static.q, static.k, static.v, static.q_idx, static.raw, static.q_full)
        _compare_token(pos, cpu, buf, tok, pos + 1)
        saved.append(tok)
    print(f"PASS {label}", flush=True)
    return buf, static, saved


def test_topk(device) -> None:
    print("=== topk ===", flush=True)
    rng = np.random.default_rng(1)
    cases = []
    scores = rng.normal(size=64).astype(np.float32)
    scores[3] = scores[10] = scores[20] = 4.0
    cases.append((scores, 5))
    scores = rng.normal(size=32).astype(np.float32)
    scores[4:] = scores[4] 
    scores[8] = 1.0e9
    scores[9] = 1.0e9
    cases.append((scores, 7))
    big = rng.normal(size=2100).astype(np.float32)
    big[-4:] += 1.0e9
    cases.append((big, 2051))
    for scores, width in cases:
        n = scores.shape[0]
        ref = topk_radix(scores, width)
        st = torch.tensor([n - 1, n, 0, width], dtype=torch.int32, device=device)
        sc = _to(scores, device)
        ids = torch.empty(width, dtype=torch.int32, device=device)
        ops.topk(sc, ids, st)
        torch.xpu.synchronize()
        got = ids.detach().cpu().numpy()
        _check_equal(f"topk n={n} width={width}", got, ref)
    print("PASS topk", flush=True)


def test_zero_group(device, gpu_geom: Geom) -> None:
    print("=== zero group ===", flush=True)
    buf = QsaBuffers(8, gpu_geom, device)
    g = gpu_geom
    k = torch.zeros(g.n_kv, g.head_dim, dtype=torch.float32, device=device)
    v = torch.zeros_like(k)
    buf.pos.fill_(0)
    ops.write_step(buf.pos, buf.step, g.idx_top_k, g.idx_block)
    ops.kv_append(k, v, buf.k_codes, buf.v_codes, buf.k_scale, buf.v_scale, buf.page_table, buf.step)
    torch.xpu.synchronize()
    if int(buf.k_codes.abs().sum()) != 0 or int(buf.v_codes.abs().sum()) != 0:
        raise SystemExit("FAIL zero group codes")
    if float(buf.k_scale.abs().sum()) != 0.0 or float(buf.v_scale.abs().sum()) != 0.0:
        raise SystemExit("FAIL zero group scales")
    print("PASS zero group", flush=True)


CACHE = ("tail", "dead", "pooled", "block_pos", "k_codes", "v_codes", "k_scale", "v_scale")


def _snap(buf: QsaBuffers, names) -> dict:
    torch.xpu.synchronize()
    return {name: getattr(buf, name).detach().to("cpu").clone() for name in names}


def _restore(buf: QsaBuffers, snap: dict) -> None:
    for name, val in snap.items():
        getattr(buf, name).copy_(val.to(device=buf.device))


def _equal_snap(got: dict, ref: dict, names, width: int | None = None) -> None:
    for name in names:
        g = got[name]
        r = ref[name]
        if name == "ids" and width is not None:
            g = g[:width]
            r = r[:width]
        if name in ("k_f16", "v_f16") and width is not None:
            g = g[:width]
            r = r[:width]
        if not torch.equal(g, r):
            if g.dtype.is_floating_point:
                err = float((g.float() - r.float()).abs().max())
                raise SystemExit(f"FAIL graph {name} max_abs={err}")
            raise SystemExit(f"FAIL graph {name} mismatch")
        print(f"  graph {name} exact", flush=True)


def test_graph(cpu_geom, gpu_geom, tokens) -> None:
    print("=== graph ===", flush=True)
    device = torch.device("xpu")
    cpu = QsaState(32, cpu_geom)
    buf = QsaBuffers(cpu.max_cells, gpu_geom, device)
    k_norm = (0.8 + 0.4 * np.linspace(0, 1, cpu_geom.idx_dim)).astype(np.float32)
    _bind(cpu, buf, k_norm)
    static = _Static(gpu_geom, device)
    snaps = {}
    outs = {}
    for pos, tok in enumerate(tokens):
        cpu.append_kv(tok["k"], tok["v"], pos)
        cpu.append_indexer(tok["raw"], k_norm, pos)
        static.load(tok, pos, buf)
        buf.run(static.q, static.k, static.v, static.q_idx, static.raw, static.q_full)
        _compare_token(pos, cpu, buf, tok, pos + 1)
        if pos in (3, 4, 5, 6, 7):
            snaps[pos] = _snap(buf, CACHE)
            width = cpu_geom.width_at(pos + 1)
            outs[pos] = {
                "ids": buf.ids[:width].detach().to("cpu").clone(),
                "attn": buf.attn.detach().to("cpu").clone(),
                "gated": buf.gated.detach().to("cpu").clone(),
                "width": width,
            }
            # Inputs at this pos, so replay can restore them.
            outs[pos]["q"] = static.q.detach().to("cpu").clone()
            outs[pos]["k"] = static.k.detach().to("cpu").clone()
            outs[pos]["v"] = static.v.detach().to("cpu").clone()
            outs[pos]["q_idx"] = static.q_idx.detach().to("cpu").clone()
            outs[pos]["raw"] = static.raw.detach().to("cpu").clone()
            outs[pos]["q_full"] = static.q_full.detach().to("cpu").clone()

    _restore(buf, snaps[3])
    def _load_out(pos: int) -> None:
        o = outs[pos]
        static.q.copy_(o["q"])
        static.k.copy_(o["k"])
        static.v.copy_(o["v"])
        static.q_idx.copy_(o["q_idx"])
        static.raw.copy_(o["raw"])
        static.q_full.copy_(o["q_full"])
        buf.pos.fill_(pos)

    _load_out(4)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        buf.run(static.q, static.k, static.v, static.q_idx, static.raw, static.q_full)
    torch.xpu.synchronize()
    # Capture consumed the token-4 update. Put the cache back, then replay.
    _restore(buf, snaps[3])
    _load_out(4)
    buf.ids.fill_(-1)
    buf.attn.fill_(float("nan"))
    graph.replay()
    torch.xpu.synchronize()
    width = outs[4]["width"]
    got = _snap(buf, CACHE)
    _equal_snap(got, snaps[4], CACHE)
    _check_equal("graph replay pos 4 ids", buf.ids[:width].detach().cpu().numpy(), outs[4]["ids"].numpy())
    _check_close("graph replay pos 4 attn", buf.attn.detach().cpu().numpy(), outs[4]["attn"].numpy(), 0.0)
    _check_close("graph replay pos 4 gated", buf.gated.detach().cpu().numpy(), outs[4]["gated"].numpy(), 0.0)
    # Same graph, next cell. step/pos are device tensors, not baked args.
    _load_out(5)
    buf.ids.fill_(-1)
    graph.replay()
    torch.xpu.synchronize()
    width = outs[5]["width"]
    got = _snap(buf, CACHE)
    _equal_snap(got, snaps[5], CACHE)
    _check_equal("graph replay pos 5 ids", buf.ids[:width].detach().cpu().numpy(), outs[5]["ids"].numpy())
    _check_close("graph replay pos 5 attn", buf.attn.detach().cpu().numpy(), outs[5]["attn"].numpy(), 0.0)
    _check_close("graph replay pos 5 gated", buf.gated.detach().cpu().numpy(), outs[5]["gated"].numpy(), 0.0)
    # Captured on a non-completing cell (pos 4, slot 0). Replay a completion
    # (pos 7, slot 3) from the eager cache so the in-kernel branch is not
    # the one capture executed.
    _restore(buf, snaps[6])
    _load_out(7)
    buf.ids.fill_(-1)
    graph.replay()
    torch.xpu.synchronize()
    width = outs[7]["width"]
    got = _snap(buf, CACHE)
    _equal_snap(got, snaps[7], CACHE)
    _check_equal("graph replay pos 7 ids", buf.ids[:width].detach().cpu().numpy(), outs[7]["ids"].numpy())
    _check_close("graph replay pos 7 attn", buf.attn.detach().cpu().numpy(), outs[7]["attn"].numpy(), 0.0)
    _check_close("graph replay pos 7 gated", buf.gated.detach().cpu().numpy(), outs[7]["gated"].numpy(), 0.0)
    print("PASS graph", flush=True)


def main() -> None:
    print("xpu", torch.xpu.device_count(), flush=True)
    device = torch.device("xpu")
    cpu_geom, gpu_geom = _small()
    test_topk(device)
    test_zero_group(device, gpu_geom)
    rng = np.random.default_rng(2)
    tokens = [_make_tok(rng, cpu_geom) for _ in range(8)]
    tokens[2]["k"][:] = 0
    tokens[2]["v"][:] = 0
    _run_seq(cpu_geom, gpu_geom, tokens, label="small")
    perm = np.arange((32 + 3) // 4, dtype=np.int32)
    perm = perm[::-1].copy()
    _run_seq(cpu_geom, gpu_geom, tokens[:6], page_perm=perm, label="page perm")
    _run_seq(cpu_geom, gpu_geom, tokens[:6], pos_base=100, label="pos_base 100")
    # Triple values are cos/sin table indices. Keep them inside the table.
    mrope = np.zeros((32, 3), dtype=np.int32)
    for i in range(32):
        mrope[i] = (i % 5, (i + 1) % 7, (i * 3) % 11)
    _run_seq(cpu_geom, gpu_geom, tokens[:8], mrope=mrope, label="mrope")
    wide_cpu, wide_gpu = _wide()
    wrng = np.random.default_rng(3)
    wide_tokens = [_make_tok(wrng, wide_cpu) for _ in range(10)]
    _run_seq(wide_cpu, wide_gpu, wide_tokens, label="head_dim 256")
    test_graph(cpu_geom, gpu_geom, tokens)
    print("QSA_XPU_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"QSA_XPU_FAIL {type(exc).__name__}: {exc}", flush=True)
        raise
