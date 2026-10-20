"""Strata semantic checks for the full QSA layer. CPU only.

Run from vllm/flashnext:
  python3 -m flashnext_ref.test_qsa_layer
"""

from __future__ import annotations

import numpy as np

from flashnext_ref.geometry import FLASHNEXT
from flashnext_ref.qsa import pool_indexer_keys, rope_neox, topk_ids
from flashnext_ref.qsa_layer import (
    QsaGeom,
    QsaState,
    QsaWeights,
    apply_rope_table,
    dequant_q8,
    gate_fp16,
    prefill,
    prefill_chunk,
    quantize_q8,
    step_from_pos,
    topk_radix,
    vllm_block_expand_ids,
)


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    print(f"PASS {name} {detail}", flush=True)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_radix_matches_lexsort() -> None:
    rng = np.random.default_rng(0)
    for n, width in ((1, 1), (5, 3), (8, 8), (17, 7), (64, 11), (2051, 2051), (2100, 2051)):
        scores = rng.normal(size=n).astype(np.float32)
        # Plant ties, a negative, and a tail-sized bonus.
        scores[0] = scores[min(1, n - 1)]
        scores[-1] = np.float32(1e9)
        got = topk_radix(scores, width)
        exp = topk_ids(scores, width)
        _check(f"radix n={n} w={width}", np.array_equal(got, exp), f"got {got[:8]} exp {exp[:8]}")


def test_step_and_bytes() -> None:
    g = QsaGeom()
    step = step_from_pos(0, g)
    _check("step0", np.array_equal(step, [0, 1, 0, 1]))
    step = step_from_pos(2050, g)
    _check("step2050", np.array_equal(step, [2050, 2051, 512, 2051]), str(step))
    step = step_from_pos(199999, g)
    _check("width200k", int(step[3]) == 2051 and int(step[2]) == 50000)
    _check("kv bytes", g.kv_bytes_per_cell() == 1056, str(g.kv_bytes_per_cell()))
    # 12 layers * (1056 + 128) + draft 1056 = 15264, G4's per-token slope.
    per = 12 * (g.kv_bytes_per_cell() + g.idx_dim * 4 // g.idx_block) + g.kv_bytes_per_cell()
    _check("park bytes per token", per == 15264, str(per))
    _check("fixed indexer", g.fixed_indexer_bytes() == (3 * 128 * 4) + (128 * 4) + 4)


def test_q8_roundtrip() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 256)).astype(np.float32)
    x[0, :64] = 0
    codes, scale = quantize_q8(x)
    back = dequant_q8(codes, scale)
    _check("zero group", bool(np.all(codes[0, :64] == 0) and scale[0, 0] == 0))
    err = np.max(np.abs(x - back))
    # One fp16 scale and one fp16 product. Well under a step of the scale.
    _check("q8 err", float(err) < 0.05, f"{err}")
    # rint ties-to-even: 0.5 -> 0, 1.5 -> 2, against scale 1.
    tiny = np.zeros((64,), dtype=np.float32)
    tiny[0] = 0.5
    tiny[1] = 1.5
    tiny[2] = -1.5
    # Force scale 1 by planting 127.
    tiny[3] = 127
    codes, scale = quantize_q8(tiny)
    _check("scale one", float(scale[0]) == 1.0, str(scale[0]))
    _check("rint", codes[0] == 0 and codes[1] == 2 and codes[2] == -2, str(codes[:3]))


def test_indexer_matches_pool_and_pos_base() -> None:
    rng = np.random.default_rng(2)
    geom = QsaGeom(
        n_head=4, n_kv=2, head_dim=64, n_rot=8, idx_n_head=2, idx_dim=8,
        idx_block=4, idx_top_k=4, hidden=16, rope_base=1e7,
    )
    raw = rng.normal(size=(8, geom.idx_dim)).astype(np.float32)
    gamma = rng.normal(size=(geom.idx_dim,)).astype(np.float32) * 0.1 + 1
    st = QsaState(32, geom)
    for i in range(8):
        st.append_indexer(raw[i], gamma, i)
    pooled, n_bid = pool_indexer_keys(raw, gamma, block=4, rotary_dim=geom.n_rot, pos_base=0)
    # G3's pool is fp32; the state machine sums in float64 like qsa.cu.
    err = _max_abs(st.pooled[: n_bid + 1], pooled)
    _check("pool vs fp32 reference", err < 1e-4, f"max_abs {err}")
    _check("n_bid", n_bid == 2 and st.n_kv == 8)

    st100 = QsaState(32, geom, pos_base=100)
    for i in range(4):
        st100.append_indexer(raw[i], gamma, i)
    # Same magnitudes, different angle. Not equal to pos_base 0.
    err_base = _max_abs(st100.pooled[0], st.pooled[0])
    _check("pos_base rotates the block", err_base > 1e-3, f"max_abs {err_base}")
    # Spare is angle 0 in both, so dead matches.
    _check("spare ignores pos_base", _max_abs(st100.dead, st.dead) == 0.0)

    # Text mRoPE (t=h=w) matches 1D. A split triple does not.
    vec = rng.normal(size=(16,)).astype(np.float32)
    cos, sin = st.cos_tab, st.sin_tab
    one = apply_rope_table(vec, 5, cos, sin)
    triple = apply_rope_table(vec, 5, cos, sin, thw=np.array([5, 5, 5]))
    split = apply_rope_table(vec, 5, cos, sin, thw=np.array([5, 6, 7]))
    _check("text mrope is 1d", _max_abs(one, triple) == 0.0)
    _check("split mrope differs", _max_abs(one, split) > 1e-4)
    # Position 0 is the identity, including the unrotated tail.
    ident = apply_rope_table(vec, 0, cos, sin)
    _check("rope0", _max_abs(ident, vec) == 0.0)
    # And it agrees with G3's rope_neox.
    g3 = rope_neox(vec, np.array(3), rotary_dim=geom.n_rot, base=geom.rope_base)
    tab = apply_rope_table(vec, 3, cos, sin)
    _check("table matches rope_neox", _max_abs(g3, tab) < 1e-6, f"{_max_abs(g3, tab)}")


def test_hand_block() -> None:
    geom = QsaGeom(
        n_head=2, n_kv=2, head_dim=64, n_rot=0, idx_n_head=1, idx_dim=2,
        idx_block=4, idx_top_k=4, hidden=4, rope_base=10000.0, eps=1e-6,
    )
    raws = np.array(
        [[1, 0], [0, 1], [1, 1], [0, 0]],
        dtype=np.float32,
    )
    gamma = np.ones(2, dtype=np.float32)
    st = QsaState(8, geom)
    for i in range(4):
        st.append_indexer(raws[i], gamma, i)
    mean = raws.astype(np.float64).mean(axis=0)
    ss = float(np.sum(mean * mean))
    inv = 1.0 / np.sqrt(ss / 2 + 1e-6)
    expect = (mean * inv).astype(np.float32)
    _check("hand mean", _max_abs(st.pooled[0], expect) == 0.0, str(st.pooled[0]))
    spare_ss = 1.0
    spare_inv = 1.0 / np.sqrt(spare_ss / 2 + 1e-6)
    spare = np.array([spare_inv, 0], dtype=np.float32)
    _check("hand spare", _max_abs(st.dead, spare) == 0.0, str(st.dead))
    _check("spare moved", _max_abs(st.pooled[1], spare) == 0.0)


def test_layer_oracle() -> None:
    rng = np.random.default_rng(3)
    geom = QsaGeom(
        n_head=4, n_kv=2, head_dim=64, n_rot=8, idx_n_head=2, idx_dim=32,
        idx_block=4, idx_top_k=4, hidden=16, rope_base=10000.0,
    )
    # width cap = 7. 16 tokens forces a partial marginal block.
    weights = QsaWeights.random(geom, rng)
    xs = rng.normal(size=(16, geom.hidden)).astype(np.float32)
    online = QsaState(32, geom)
    got = prefill(online, weights, xs)
    chunk_state = QsaState(32, geom)
    chunk = prefill_chunk(chunk_state, weights, xs)
    for i in range(16):
        _check(
            f"chunk ids {i}",
            np.array_equal(got[i]["ids"], chunk[i]["ids"]),
            f"online {got[i]['ids']} chunk {chunk[i]['ids']}",
        )
        err = _max_abs(got[i]["gated"], chunk[i]["gated"])
        _check(f"chunk gated {i}", err == 0.0, f"{err}")
        _check(f"causal {i}", bool(np.all(got[i]["ids"] <= i)))
    # Below the cap the selection is every cell.
    _check("identity short", np.array_equal(got[5]["ids"], np.arange(6)))
    # Above the cap it is a proper subset, and it cuts a block.
    ids = got[15]["ids"]
    _check("long width", ids.shape[0] == 7, str(ids))
    _check("long subset", ids.shape[0] < 16)
    # Strata vs whole-block expand.
    scores = online.scores(got[15]["q_idx"], 16)
    whole = vllm_block_expand_ids(scores, 16, geom)
    _check(
        "marginal block differs from vllm expand",
        not np.array_equal(ids, whole),
        f"strata {ids.tolist()} vllm {whole.tolist()}",
    )
    # Page permutation does not change the gather.
    perm = online.page_table.copy()
    perm[0], perm[1] = perm[1], perm[0]
    moved = QsaState(32, geom)
    moved.page_table = perm
    prefill(moved, weights, xs)
    # Compare dequant of logical cell 0.
    a = online.gather(np.array([0, 5, 9], dtype=np.int32))
    b = moved.gather(np.array([0, 5, 9], dtype=np.int32))
    _check("paged gather", _max_abs(a[0], b[0]) == 0.0 and _max_abs(a[1], b[1]) == 0.0)
    # Physical pages actually differ.
    _check("page swapped", int(perm[0]) != 0)

    # GQA: head 2 of 4 reads kv head 1 (integer division, group=2).
    q = rng.normal(size=(4, 64)).astype(np.float32)
    k = np.zeros((3, 2, 64), dtype=np.float32)
    v = np.zeros_like(k)
    v[:, 0, :] = 1
    v[:, 1, :] = -1
    from flashnext_ref.qsa import attend

    attn, _w = attend(q, k, v)
    _check("gqa head0", float(attn[0].mean()) > 0)
    _check("gqa head2", float(attn[2].mean()) < 0)

    # Gate half, then the fp16 round.
    q_full = np.zeros((4, 128), dtype=np.float32)
    q_full[:, 64:] = 0  # sigmoid 0.5
    gated = gate_fp16(attn, q_full)
    expect = np.float32(np.float16(0.5 * attn))
    _check("gate fp16", _max_abs(gated, expect) == 0.0)

    # Tail bonus is visible on a 5-cell prefix: cell 4 outscores a finished block
    # that was given a large negative... the bonus dominates any finite dot.
    st = QsaState(16, geom)
    prefill(st, weights, xs[:5])
    sc = st.scores(got[4]["q_idx"], 5)
    _check("tail bonus", float(sc[4]) > 1e8 and float(sc[0]) < 1e8, str(sc))

    # pos_base 100: online y differs from pos_base 0 (rope), ids may also move.
    st_b = QsaState(32, geom, pos_base=100)
    alt = prefill(st_b, weights, xs[:4])
    err = _max_abs(alt[-1]["y"], got[3]["y"])
    _check("pos_base changes the layer", err > 1e-4, f"{err}")


def test_real_geom_short() -> None:
    """Three tokens at the artifact shape. Selection is the identity."""
    rng = np.random.default_rng(4)
    geom = QsaGeom(hidden=32)
    weights = QsaWeights.random(geom, rng, scale=0.02)
    xs = rng.normal(size=(3, 32)).astype(np.float32)
    st = QsaState(8, geom)
    got = prefill(st, weights, xs)
    _check("real identity", np.array_equal(got[-1]["ids"], np.arange(3)))
    _check("real y finite", bool(np.all(np.isfinite(got[-1]["y"]))))
    _check("flashnext width cap", FLASHNEXT.qsa_width_extra == 2051)


def main() -> None:
    test_radix_matches_lexsort()
    test_step_and_bytes()
    test_q8_roundtrip()
    test_indexer_matches_pool_and_pos_base()
    test_hand_block()
    test_layer_oracle()
    test_real_geom_short()
    print("QSA_LAYER_OK", flush=True)


if __name__ == "__main__":
    main()
