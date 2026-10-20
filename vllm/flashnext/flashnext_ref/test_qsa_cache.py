"""Paged QSA cache against QsaState. CPU only.

Run from vllm/flashnext:
  python3 -m flashnext_ref.test_qsa_cache
"""

from __future__ import annotations

import numpy as np

from flashnext_ref.qsa import attend
from flashnext_ref.qsa_cache import (
    QsaPaged,
    attend_masked,
    make_layout,
    union_cells,
    union_pack,
    write_step_rows,
)
from flashnext_ref.qsa_layer import (
    QsaGeom,
    QsaState,
    QsaWeights,
    attend_ids,
    gate_fp16,
    prepare_token,
    prefill,
    prefill_chunk,
)


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    print(f"PASS {name} {detail}", flush=True)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def _table(batch: int, max_blocks: int, rng: np.random.Generator, slack: int = 3) -> np.ndarray:
    n = batch * max_blocks
    perm = rng.permutation(n + slack).astype(np.int32)[:n]
    return perm.reshape(batch, max_blocks)


def _small() -> QsaGeom:
    return QsaGeom(
        n_head=4,
        n_kv=2,
        head_dim=64,
        n_rot=8,
        idx_n_head=2,
        idx_dim=8,
        idx_block=4,
        idx_top_k=4,
        hidden=16,
        rope_base=1.0e7,
    )


def _gamma(rng: np.random.Generator, dim: int) -> np.ndarray:
    return (0.8 + 0.4 * np.linspace(0.0, 1.0, dim)).astype(np.float32)


def _compare_cell(st: QsaState, pg: QsaPaged, seq: int, pos: int) -> None:
    g = st.geom
    page_i = pos // g.page_size
    phys = int(st.page_table[page_i])
    slot = pos % g.page_size
    kc, ks, vc, vs = pg.read_kv_cell(seq, pos)
    if not np.array_equal(kc, st.k_q[phys, :, slot, :]):
        raise SystemExit(f"FAIL k codes seq {seq} pos {pos}")
    if not np.array_equal(vc, st.v_q[phys, :, slot, :]):
        raise SystemExit(f"FAIL v codes seq {seq} pos {pos}")
    if _max_abs(ks, st.k_scale[phys, :, slot, :]) != 0.0:
        raise SystemExit(f"FAIL k scales seq {seq} pos {pos}")
    if _max_abs(vs, st.v_scale[phys, :, slot, :]) != 0.0:
        raise SystemExit(f"FAIL v scales seq {seq} pos {pos}")


def _compare_indexer(st: QsaState, pg: QsaPaged, seq: int) -> None:
    if not np.array_equal(pg.tail[seq], st.tail):
        raise SystemExit(f"FAIL tail seq {seq}")
    if int(pg.block_pos[seq]) != int(st.block_pos):
        raise SystemExit(f"FAIL block_pos seq {seq} {pg.block_pos[seq]} vs {st.block_pos}")
    if bool(pg.have_dead[seq]) != bool(st.have_dead):
        raise SystemExit(f"FAIL have_dead seq {seq}")
    if st.have_dead and _max_abs(pg.dead[seq], st.dead) != 0.0:
        raise SystemExit(f"FAIL dead seq {seq} {_max_abs(pg.dead[seq], st.dead)}")
    n_bid = int(st.n_kv) // st.geom.idx_block
    for b in range(n_bid):
        err = _max_abs(pg.read_pooled(seq, b), st.pooled[b])
        if err != 0.0:
            raise SystemExit(f"FAIL pooled block {b} seq {seq} {err}")


def _feed_pair(
    st: QsaState,
    pg: QsaPaged,
    seq: int,
    weights: QsaWeights,
    xs: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    outs = []
    for i in range(xs.shape[0]):
        thw = st.mrope[i] if st.mrope is not None else None
        tok = prepare_token(
            xs[i], weights, st.geom, st.pos_base + i, st.cos_tab, st.sin_tab, thw=thw
        )
        st.append_kv(tok["k"], tok["v"], i)
        st.append_indexer(tok["raw"], weights.idx_k_norm, i)
        got = pg.decode_prepared(
            seq,
            tok["k"],
            tok["v"],
            tok["raw"],
            weights.idx_k_norm,
            tok["q"],
            tok["q_idx"],
            tok["q_full"],
            i,
        )
        ref_ids = st.select(tok["q_idx"], i + 1)
        ref_attn, _w = attend_ids(tok["q"], st, ref_ids)
        ref_gated = gate_fp16(ref_attn, tok["q_full"])
        if not np.array_equal(got["ids"], ref_ids):
            raise SystemExit(f"FAIL ids seq {seq} pos {i}\n{got['ids']}\n{ref_ids}")
        aerr = _max_abs(got["attn"], ref_attn)
        gerr = _max_abs(got["gated"], ref_gated)
        if aerr > 2e-3 or gerr > 2e-3:
            raise SystemExit(f"FAIL attn seq {seq} pos {i} attn {aerr} gated {gerr}")
        _compare_cell(st, pg, seq, i)
        outs.append(got)
    _compare_indexer(st, pg, seq)
    return outs


def test_plugin_layout_matches() -> None:
    import sys
    from pathlib import Path

    plugin = Path(__file__).resolve().parents[1] / "plugins" / "b70_qsa_xpu"
    sys.path.insert(0, str(plugin))
    from b70_qsa_xpu.layout import page_layout

    keys = (
        "block_size",
        "n_kv",
        "head_dim",
        "groups",
        "idx_dim",
        "idx_block",
        "n_states",
        "k_codes_off",
        "v_codes_off",
        "k_scale_off",
        "v_scale_off",
        "pooled_off",
        "kv_bytes",
        "pooled_bytes",
        "page_bytes",
    )
    geoms = (QsaGeom(), _small())
    for geom in geoms:
        for block in (4, 8, 16):
            for pooled in (True, False):
                a = make_layout(geom, block, pooled=pooled)
                b = page_layout(
                    geom.n_kv,
                    geom.head_dim,
                    geom.idx_dim,
                    geom.idx_block,
                    block,
                    pooled=pooled,
                )
                for key in keys:
                    _check(
                        f"plugin {key} hd{geom.head_dim} b{block} p{int(pooled)}",
                        getattr(a, key) == b[key],
                        f"{getattr(a, key)} vs {b[key]}",
                    )
                _check(
                    f"plugin pooled hd{geom.head_dim} b{block} p{int(pooled)}",
                    int(a.pooled) == b["pooled"],
                )


def test_layout_flashnext() -> None:
    lay = make_layout(QsaGeom(), 16, pooled=True)
    _check("kv region", lay.kv_bytes == 1056 * 16, str(lay.kv_bytes))
    _check("pooled region", lay.pooled_bytes == (16 // 4) * 128 * 4, str(lay.pooled_bytes))
    _check("page bytes", lay.page_bytes == 1056 * 16 + 2048, str(lay.page_bytes))
    per = lay.page_bytes // 16
    _check("per token", per == 1184, str(per))
    draft = make_layout(QsaGeom(), 16, pooled=False)
    _check("draft page", draft.page_bytes == 1056 * 16 and draft.pooled_bytes == 0)
    _check("offsets", lay.v_codes_off == lay.k_codes_off + 2 * 16 * 256)
    _check("scale aligns", lay.k_scale_off % 2 == 0 and lay.pooled_off % 4 == 0)
    # block 4 never has room for the spare row of the next indexer block.
    lay4 = make_layout(QsaGeom(), 4, pooled=True)
    _check("block4 states", lay4.n_states == 1)
    logical, state = lay4.indexer_addr(0)
    logical_s, state_s = lay4.indexer_addr(1)
    _check("block4 spare spills", logical == 0 and logical_s == 1 and state == 0 and state_s == 0)
    lay8 = make_layout(QsaGeom(), 8, pooled=True)
    l0, s0 = lay8.indexer_addr(0)
    l1, s1 = lay8.indexer_addr(1)
    l2, s2 = lay8.indexer_addr(2)
    _check("block8 spare stays", l0 == l1 == 0 and s0 == 0 and s1 == 1 and l2 == 1 and s2 == 0)


def test_write_step_inactive() -> None:
    step = write_step_rows(np.array([7, -1, 0], dtype=np.int32), 4, 4)
    _check("step live", np.array_equal(step[0], [7, 8, 2, 7]), str(step[0]))
    _check("step inactive", np.array_equal(step[1], [0, 0, 0, 0]), str(step[1]))
    _check("step zero", np.array_equal(step[2], [0, 1, 0, 1]), str(step[2]))


def test_batch_matches_state(block_size: int) -> None:
    rng = np.random.default_rng(10 + block_size)
    geom = _small()
    weights = QsaWeights.random(geom, rng, scale=0.15)
    weights.idx_k_norm = _gamma(rng, geom.idx_dim)
    lengths = (10, 7)
    max_blocks = (max(lengths) + block_size - 1) // block_size + 1
    table = _table(2, max_blocks, rng)
    pg = QsaPaged.create(geom, table, block_size, pooled=True)
    states = []
    for seq, n in enumerate(lengths):
        st = QsaState(32, geom)
        xs = rng.normal(size=(n, geom.hidden)).astype(np.float32)
        _feed_pair(st, pg, seq, weights, xs)
        states.append(st)
    # The other sequence's pages are a different permutation. Re-check both.
    for seq, st in enumerate(states):
        _compare_indexer(st, pg, seq)
        for pos in range(int(st.n_kv)):
            _compare_cell(st, pg, seq, pos)
    _check(f"batch block {block_size}", True, f"lens {lengths} pages {pg.num_pages}")


def test_spare_stays_in_page() -> None:
    rng = np.random.default_rng(3)
    geom = _small()
    weights = QsaWeights.random(geom, rng, scale=0.15)
    weights.idx_k_norm = _gamma(rng, geom.idx_dim)
    # block 8: completion of indexer 0 writes the spare into state 1.
    table = _table(1, 4, rng)
    pg = QsaPaged.create(geom, table, 8, pooled=True)
    st = QsaState(32, geom)
    xs = rng.normal(size=(8, geom.hidden)).astype(np.float32)
    prepared = []
    for i in range(4):
        tok = prepare_token(xs[i], weights, geom, i, st.cos_tab, st.sin_tab)
        st.append_kv(tok["k"], tok["v"], i)
        st.append_indexer(tok["raw"], weights.idx_k_norm, i)
        pg.append_token(0, tok["k"], tok["v"], tok["raw"], weights.idx_k_norm, i)
        prepared.append(tok)
    err = _max_abs(pg.read_pooled(0, 1), st.dead)
    _check("spare in same page", err == 0.0, f"{err}")
    nxt = pg.pages[pg._phys(0, 1)]
    _check("next vllm block untouched", bool(np.all(nxt == 0)))
    # Finish the block. State 1 becomes the real key. The following page stays empty.
    for i in range(4, 8):
        tok = prepare_token(xs[i], weights, geom, i, st.cos_tab, st.sin_tab)
        st.append_kv(tok["k"], tok["v"], i)
        st.append_indexer(tok["raw"], weights.idx_k_norm, i)
        pg.append_token(0, tok["k"], tok["v"], tok["raw"], weights.idx_k_norm, i)
    err = _max_abs(pg.read_pooled(0, 1), st.pooled[1])
    _check("completion overwrites spare", err == 0.0 and _max_abs(st.pooled[1], st.dead) > 0.0)
    nxt = pg.pages[pg._phys(0, 1)]
    _check("no spare past the block", bool(np.all(nxt == 0)))

    # block 4: the spare would be the next vLLM block, which stays zero
    # through the open tail of that next block's first tokens... at n_kv=4
    # the next page is untouched even though QsaState.pooled[1] is dead.
    pg4 = QsaPaged.create(geom, _table(1, 4, rng), 4, pooled=True)
    st4 = QsaState(32, geom)
    for i in range(4):
        tok = prepare_token(xs[i], weights, geom, i, st4.cos_tab, st4.sin_tab)
        st4.append_kv(tok["k"], tok["v"], i)
        st4.append_indexer(tok["raw"], weights.idx_k_norm, i)
        pg4.append_token(0, tok["k"], tok["v"], tok["raw"], weights.idx_k_norm, i)
    _check("state spare alias", _max_abs(st4.pooled[1], st4.dead) == 0.0)
    _check("page omits spare", bool(np.all(pg4.pages[pg4._phys(0, 1)] == 0)))
    _check("completed row kept", _max_abs(pg4.read_pooled(0, 0), st4.pooled[0]) == 0.0)


def test_chunk_prefill() -> None:
    rng = np.random.default_rng(4)
    geom = _small()
    weights = QsaWeights.random(geom, rng, scale=0.12)
    weights.idx_k_norm = _gamma(rng, geom.idx_dim)
    xs = rng.normal(size=(10, geom.hidden)).astype(np.float32)
    st = QsaState(32, geom)
    ref = prefill_chunk(st, weights, xs)
    tok_state = QsaState(32, geom)
    token_ref = prefill(tok_state, weights, xs)
    pg = QsaPaged.create(geom, _table(1, 4, rng), 8, pooled=True)
    prepared = []
    for i in range(xs.shape[0]):
        tok = prepare_token(xs[i], weights, geom, i, pg.cos_tab, pg.sin_tab)
        tok["pos"] = i
        prepared.append(tok)
    got = pg.prefill_prepared(0, prepared, weights.idx_k_norm)
    for i, (row, exp) in enumerate(zip(got, ref)):
        _check(
            f"chunk ids {i}",
            np.array_equal(row["ids"], exp["ids"]) and np.array_equal(row["ids"], token_ref[i]["ids"]),
        )
        _check(f"chunk attn {i}", _max_abs(row["attn"], exp["attn"]) == 0.0)
        _check(f"chunk gated {i}", _max_abs(row["gated"], exp["gated"]) == 0.0)
    _compare_indexer(st, pg, 0)


def test_export_roundtrip() -> None:
    rng = np.random.default_rng(5)
    geom = _small()
    weights = QsaWeights.random(geom, rng, scale=0.12)
    weights.idx_k_norm = _gamma(rng, geom.idx_dim)
    table = _table(2, 4, rng)
    pg = QsaPaged.create(geom, table, 8, pooled=True)
    states = [QsaState(32, geom), QsaState(32, geom)]
    xs = [
        rng.normal(size=(6, geom.hidden)).astype(np.float32),
        rng.normal(size=(9, geom.hidden)).astype(np.float32),
    ]
    for seq in range(2):
        _feed_pair(states[seq], pg, seq, weights, xs[seq])
    blob = pg.export_seq(0)
    other_pages = pg.pages[pg.block_table[1]].copy()
    other_tail = pg.tail[1].copy()
    pg.zero_seq(0)
    _check("zeroed tail", bool(np.all(pg.tail[0] == 0)) and int(pg.n_kv[0]) == 0)
    _check("zero keeps neighbor", np.array_equal(pg.pages[pg.block_table[1]], other_pages))
    # Remap seq 0 onto a fresh permutation and import by logical block.
    fresh = _table(2, 4, np.random.default_rng(6))
    # Keep seq 1's physical pages so the neighbor check still applies.
    fresh[1] = table[1]
    # Seq 0 must not collide with seq 1.
    used = set(int(x) for x in fresh[1].tolist())
    pool = [i for i in range(int(fresh.max()) + 8) if i not in used]
    fresh[0] = np.asarray(pool[: fresh.shape[1]], dtype=np.int32)
    restored = QsaPaged.create(geom, fresh, 8, pooled=True)
    # Copy seq 1 pages across so the neighbor is present, then import seq 0.
    for logical in range(table.shape[1]):
        restored.pages[int(fresh[1, logical])] = pg.pages[int(table[1, logical])]
    restored.tail[1] = other_tail
    restored.n_kv[1] = states[1].n_kv
    restored.pos[1] = states[1].n_kv - 1
    restored.import_seq(0, blob)
    _compare_indexer(states[0], restored, 0)
    for pos in range(int(states[0].n_kv)):
        _compare_cell(states[0], restored, 0, pos)
    _check("import keeps neighbor tail", np.array_equal(restored.tail[1], other_tail))
    # Next token matches a clone that was never exported.
    nxt = rng.normal(size=(geom.hidden,)).astype(np.float32)
    tok = prepare_token(nxt, weights, geom, int(states[0].n_kv), states[0].cos_tab, states[0].sin_tab)
    pos = int(states[0].n_kv)
    states[0].append_kv(tok["k"], tok["v"], pos)
    states[0].append_indexer(tok["raw"], weights.idx_k_norm, pos)
    got = restored.decode_prepared(
        0, tok["k"], tok["v"], tok["raw"], weights.idx_k_norm, tok["q"], tok["q_idx"], tok["q_full"], pos
    )
    ref = states[0].select(tok["q_idx"], pos + 1)
    _check("resume ids", np.array_equal(got["ids"], ref))
    ref_attn, _w = attend_ids(tok["q"], states[0], ref)
    _check("resume attn", _max_abs(got["attn"], ref_attn) == 0.0)
    _compare_indexer(states[0], restored, 0)


def test_draft_reuses_ids() -> None:
    rng = np.random.default_rng(7)
    geom = _small()
    weights = QsaWeights.random(geom, rng, scale=0.12)
    weights.idx_k_norm = _gamma(rng, geom.idx_dim)
    st = QsaState(32, geom)
    pg = QsaPaged.create(geom, _table(1, 4, rng), 4, pooled=False)
    _check("draft bytes", pg.layout.page_bytes == pg.layout.kv_bytes and pg.layout.pooled_bytes == 0)
    xs = rng.normal(size=(9, geom.hidden)).astype(np.float32)
    last = None
    for i in range(xs.shape[0]):
        tok = prepare_token(xs[i], weights, geom, i, st.cos_tab, st.sin_tab)
        st.append_kv(tok["k"], tok["v"], i)
        st.append_indexer(tok["raw"], weights.idx_k_norm, i)
        pg.append_kv(0, tok["k"], tok["v"], i)
        last = tok
        _compare_cell(st, pg, 0, i)
    assert last is not None
    ids = st.select(last["q_idx"], xs.shape[0])
    attn, _w = attend_ids(last["q"], st, ids)
    got = pg.attend_ids(0, last["q"], ids)
    gated = gate_fp16(got, last["q_full"])
    ref_g = gate_fp16(attn, last["q_full"])
    _check("draft attend", _max_abs(got, attn) == 0.0 and _max_abs(gated, ref_g) == 0.0)


def test_pos_base_and_mrope() -> None:
    rng = np.random.default_rng(8)
    geom = _small()
    weights = QsaWeights.random(geom, rng, scale=0.1)
    weights.idx_k_norm = _gamma(rng, geom.idx_dim)
    xs = rng.normal(size=(4, geom.hidden)).astype(np.float32)
    st = QsaState(16, geom, pos_base=100)
    pg = QsaPaged.create(
        geom, _table(1, 2, rng), 4, pooled=True, pos_base=np.array([100], dtype=np.int32)
    )
    _feed_pair(st, pg, 0, weights, xs)
    _check("pos_base block_pos", int(pg.block_pos[0]) == 100)

    st1 = QsaState(16, geom, pos_base=0)
    cells = 8
    mrope = np.zeros((1, cells, 3), dtype=np.int32)
    mrope[0, :, 0] = np.arange(cells, dtype=np.int32)
    mrope[0, :, 1] = np.arange(cells, dtype=np.int32) + 3
    mrope[0, :, 2] = 1
    st1.set_mrope(mrope[0])
    pg1 = QsaPaged.create(geom, _table(1, 2, rng), 4, pooled=True, mrope=mrope)
    xs1 = rng.normal(size=(4, geom.hidden)).astype(np.float32)
    _feed_pair(st1, pg1, 0, weights, xs1)
    _check("mrope pooled", _max_abs(pg1.read_pooled(0, 0), st1.pooled[0]) == 0.0)


def test_union_and_masked_attend() -> None:
    ids = np.array(
        [
            [4, 5, 6, 0],
            [6, 7, 1, 0],
            [4, 9, 0, 0],
            [2, 2, 5, 0],
        ],
        dtype=np.int32,
    )
    widths = np.array([3, 3, 2, 3], dtype=np.int32)
    packed, count = union_pack(ids, widths)
    _check("union count", count == 7, str(count))
    cells, masks = union_cells(packed[:count])
    _check("union order", np.array_equal(cells, [1, 2, 4, 5, 6, 7, 9]), str(cells))
    # query 3 selected 2 twice; bit 3 once. query 1 also selected cell 1.
    expect = {
        1: 1 << 1,
        2: 1 << 3,
        4: (1 << 0) | (1 << 2),
        5: (1 << 0) | (1 << 3),
        6: (1 << 0) | (1 << 1),
        7: 1 << 1,
        9: 1 << 2,
    }
    for cell, mask in zip(cells.tolist(), masks.tolist()):
        if int(mask) != expect[int(cell)]:
            raise SystemExit(f"FAIL union mask cell {cell} {mask} vs {expect[int(cell)]}")
    _check("union masks", True)

    rng = np.random.default_rng(9)
    q = rng.normal(size=(4, 8)).astype(np.float32)
    k = rng.normal(size=(count, 2, 8)).astype(np.float32)
    v = rng.normal(size=(count, 2, 8)).astype(np.float32)
    bit0 = (masks & np.uint8(1 << 0)).astype(bool)
    got = attend_masked(q, k, v, bit0, n_kv_heads=2)
    ref, _w = attend(q, k[bit0], v[bit0], n_kv_heads=2)
    _check("masked attend", _max_abs(got, ref) == 0.0)
    empty = attend_masked(q, k, v, np.zeros((count,), dtype=bool), n_kv_heads=2)
    _check("empty union", bool(np.all(empty == 0)))


def test_flashnext_shape() -> None:
    rng = np.random.default_rng(11)
    geom = QsaGeom(idx_top_k=4)
    weights = QsaWeights.random(geom, rng, scale=0.02)
    weights.idx_k_norm = _gamma(rng, geom.idx_dim)
    pg = QsaPaged.create(geom, _table(1, 3, rng), 16, pooled=True)
    st = QsaState(32, geom)
    xs = rng.normal(size=(10, geom.hidden)).astype(np.float32)
    _feed_pair(st, pg, 0, weights, xs)
    _check("flashnext shape", pg.layout.kv_bytes == 1056 * 16)


def main() -> None:
    test_layout_flashnext()
    test_plugin_layout_matches()
    test_write_step_inactive()
    test_batch_matches_state(4)
    test_batch_matches_state(8)
    test_spare_stays_in_page()
    test_chunk_prefill()
    test_export_roundtrip()
    test_draft_reuses_ids()
    test_pos_base_and_mrope()
    test_union_and_masked_attend()
    test_flashnext_shape()
    print("QSA_CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
