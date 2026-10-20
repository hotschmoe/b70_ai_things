"""Full QSA attention layer. Strata semantics, fp32 reference.

The piece kernels are in qsa.py. This module owns the per-request state,
the int8 KV pool, prefill (a query sees only cells up to itself), and the
projection-to-gated-output layer.

Sources (night-20261016, /mnt/vm_8tb/b70/build/strata-ab/src-ep):
- include/strata/kernels/qsa.hpp
- include/strata/kernels/kv_q8.hpp and src/kernels/cuda/kv_q8.cu
- include/strata/kernels/qsa_select.hpp (tail scores the dead key)
- include/strata/kernels/mrope.hpp (text t=h=w; spare rotates at 0)
- src/kernels/cuda/qsa.cu indexer_key_append_kernel

vLLM's CUDA indexer selects whole compressed blocks
(nvidia/ops/qsa_indexer.py expand). Strata selects cells. Those sets
differ on the marginal block. This reference follows Strata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import FLASHNEXT
from .qsa import (
    attend,
    gate_apply,
    indexer_scores,
    rms_norm_rows,
    selection_width,
)

KV_Q8_GROUP = 64


@dataclass(frozen=True)
class QsaGeom:
    """One QSA layer. Defaults are the Flash-Next artifact."""

    n_head: int = FLASHNEXT.n_q_heads
    n_kv: int = FLASHNEXT.n_kv_heads
    head_dim: int = FLASHNEXT.head_dim
    n_rot: int = FLASHNEXT.rope_dim
    idx_n_head: int = FLASHNEXT.idx_n_head
    idx_dim: int = FLASHNEXT.idx_dim
    idx_block: int = FLASHNEXT.idx_block
    idx_top_k: int = FLASHNEXT.idx_top_k
    rope_base: float = FLASHNEXT.rope_base
    eps: float = FLASHNEXT.rms_eps
    page_size: int = FLASHNEXT.idx_block
    hidden: int = FLASHNEXT.n_embd

    def width_cap(self) -> int:
        return int(self.idx_top_k + self.idx_block - 1)

    def width_at(self, n_kv: int) -> int:
        return selection_width(n_kv, self.idx_top_k, self.idx_block)

    def kv_bytes_per_cell(self) -> int:
        """int8 K and V codes plus fp16 scales. Real artifact: 1056."""
        groups = self.head_dim // KV_Q8_GROUP
        codes = self.n_kv * self.head_dim * 2
        scales = self.n_kv * groups * 2 * 2
        return int(codes + scales)

    def pooled_bytes(self, n_kv: int) -> int:
        rows = (n_kv // self.idx_block) + (1 if n_kv else 0)
        return int(rows * self.idx_dim * 4)

    def fixed_indexer_bytes(self) -> int:
        """Tail, dead key, and block_pos. The growing pool is separate."""
        tail = (self.idx_block - 1) * self.idx_dim * 4
        dead = self.idx_dim * 4
        return int(tail + dead + 4)


def step_from_pos(pos: int, geom: QsaGeom) -> np.ndarray:
    """qsa.hpp QsaStep: pos, n_kv, n_bid, width. Host-side filler."""
    if pos < 0:
        raise ValueError("pos < 0")
    n_kv = pos + 1
    n_bid = n_kv // geom.idx_block
    width = geom.width_at(n_kv)
    return np.array([pos, n_kv, n_bid, width], dtype=np.int32)


def rope_tables(max_pos: int, half: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin [max_pos, half], the same angles rope_neox uses."""
    if half <= 0:
        return (
            np.zeros((max_pos, 0), dtype=np.float32),
            np.zeros((max_pos, 0), dtype=np.float32),
        )
    inv = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))
    pos = np.arange(max_pos, dtype=np.float64)[:, None]
    ang = pos * inv
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope_table(
    x: np.ndarray,
    pos: int,
    cos_tab: np.ndarray,
    sin_tab: np.ndarray,
    *,
    thw: np.ndarray | None = None,
) -> np.ndarray:
    """NeoX rotate using a table. thw is (t, h, w) for interleaved mRoPE.

    Pair d reads thw[d % 3] when thw is set, else `pos` (mrope.hpp: text
    t=h=w, so one position matches). Pair d pairs with d+half.
    """
    vec = np.asarray(x, dtype=np.float32).copy()
    half = int(cos_tab.shape[1])
    if half == 0:
        return vec
    if thw is None:
        positions = np.full(half, pos, dtype=np.int64)
    else:
        triple = np.asarray(thw, dtype=np.int64).reshape(3)
        positions = triple[np.arange(half) % 3]
    c = cos_tab[positions, np.arange(half)]
    s = sin_tab[positions, np.arange(half)]
    # Copy: these slices alias `vec`, and the second write must see the
    # original partner, not the rotated one.
    x1 = np.array(vec[..., :half], copy=True)
    x2 = np.array(vec[..., half : 2 * half], copy=True)
    vec[..., :half] = x1 * c - x2 * s
    vec[..., half : 2 * half] = x2 * c + x1 * s
    return vec


def quantize_q8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group-64 int8. scale = fp16(max|x|/127), code = rint(x/scale).

    Matches kv_q8.cu: quantize against the stored fp16 scale, clamp
    [-127, 127], and a zero group stores scale 0 and codes 0.
    """
    vec = np.asarray(x, dtype=np.float32)
    if vec.shape[-1] % KV_Q8_GROUP:
        raise ValueError("last dim must be a multiple of 64")
    grouped = vec.reshape(*vec.shape[:-1], -1, KV_Q8_GROUP)
    amax = np.max(np.abs(grouped), axis=-1, keepdims=True)
    scale = np.float32(np.float16(amax / np.float32(127.0)))
    codes = np.zeros(grouped.shape, dtype=np.int32)
    live = scale > 0
    quot = np.divide(grouped, scale, out=np.zeros_like(grouped), where=live)
    codes = np.clip(np.rint(quot), -127, 127).astype(np.int32)
    codes = np.where(live, codes, 0).astype(np.int8)
    return codes.reshape(vec.shape), scale.reshape(*vec.shape[:-1], -1)


def dequant_q8(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """fp16(code * scale), promoted back to fp32. qsa_attend reads fp16."""
    c = np.asarray(codes, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)
    s = np.repeat(s, KV_Q8_GROUP, axis=-1)
    return np.float32(np.float16(c * s))


def _ordered_keys(scores: np.ndarray) -> np.ndarray:
    """uint64 rank: higher score first, lower index wins a tie."""
    vec = np.ascontiguousarray(scores, dtype=np.float32)
    bits = vec.view(np.uint32)
    sign = bits >> np.uint32(31)
    mask = np.where(sign != 0, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    ordered = (bits ^ mask).astype(np.uint64)
    inv = np.uint64(0xFFFFFFFF) ^ np.arange(vec.shape[0], dtype=np.uint64)
    return (ordered << np.uint64(32)) | inv


def topk_radix(scores: np.ndarray, width: int) -> np.ndarray:
    """Same ids as topk_ids. This is the device kernel's algorithm.

    64-bit radix select, MSB first. The key packs an ordered float and
    the inverted index, so ties resolve to the smaller index and the
    scan that keeps key >= marginal emits cells in ascending order.
    """
    keys = _ordered_keys(scores)
    n = int(keys.shape[0])
    width = int(min(max(width, 0), n))
    if width == 0:
        return np.zeros((0,), dtype=np.int32)
    prefix = np.uint64(0)
    decided = np.uint64(0)
    need = width
    for bit in range(63, -1, -1):
        bitv = np.uint64(1) << np.uint64(bit)
        match = ((keys & decided) == prefix) & ((keys & bitv) != 0)
        cnt = int(match.sum())
        decided = decided | bitv
        if cnt >= need:
            prefix = prefix | bitv
        else:
            need -= cnt
    chosen = np.flatnonzero(keys >= prefix)
    if chosen.size != width:
        raise RuntimeError(f"radix select kept {chosen.size}, width {width}")
    return np.sort(chosen.astype(np.int32))


class QsaState:
    """One request, one layer. Layout matches qsa.hpp QsaIndexerBuffers
    plus kv_q8.hpp pools.

    tail     (block-1, idx_dim) f32 raw keys of the open block
    dead     (idx_dim,) f32      rms(raw of cell 0), rotated at position 0
    pooled   (max_blocks, idx_dim) f32  row b is completed block b;
             row n_bid holds dead after a quiescent append
    block_pos scalar i32, first-cell position of the last completion
    k_q, v_q     int8 [pages, kv_heads, page, head_dim]
    k_scale, v_scale  f32 [pages, kv_heads, page, groups]
    page_table   i32 [pages], logical page -> physical page
    """

    def __init__(self, max_cells: int, geom: QsaGeom, *, pos_base: int = 0) -> None:
        if max_cells < 1:
            raise ValueError("max_cells")
        if geom.head_dim % KV_Q8_GROUP:
            raise ValueError("head_dim must be a multiple of 64")
        self.geom = geom
        self.max_cells = int(max_cells)
        self.pos_base = int(pos_base)
        self.n_kv = 0
        r = geom.idx_block
        self.tail = np.zeros((r - 1, geom.idx_dim), dtype=np.float32)
        self.dead = np.zeros((geom.idx_dim,), dtype=np.float32)
        self.have_dead = False
        self.max_blocks = max_cells // r + 1
        self.pooled = np.zeros((self.max_blocks, geom.idx_dim), dtype=np.float32)
        self.block_pos = np.int32(0)
        pages = (max_cells + geom.page_size - 1) // geom.page_size
        groups = geom.head_dim // KV_Q8_GROUP
        shape = (pages, geom.n_kv, geom.page_size, geom.head_dim)
        self.k_q = np.zeros(shape, dtype=np.int8)
        self.v_q = np.zeros(shape, dtype=np.int8)
        self.k_scale = np.zeros(shape[:-1] + (groups,), dtype=np.float32)
        self.v_scale = np.zeros(shape[:-1] + (groups,), dtype=np.float32)
        self.page_table = np.arange(pages, dtype=np.int32)
        half = geom.n_rot // 2
        # +1 so a position equal to max_cells-1 is in range, and pos_base shifts it.
        span = max_cells + max(pos_base, 0) + 1
        self.cos_tab, self.sin_tab = rope_tables(span, half, geom.rope_base)
        self.mrope: np.ndarray | None = None

    def set_mrope(self, table: np.ndarray | None) -> None:
        """int32 [cells, 3] of (t, h, w). None keeps 1D positions."""
        if table is None:
            self.mrope = None
            return
        tab = np.asarray(table, dtype=np.int32)
        if tab.ndim != 2 or tab.shape[1] != 3:
            raise ValueError("mrope table is [cells, 3]")
        self.mrope = tab

    def _rot_thw(self, cell: int) -> np.ndarray | None:
        if self.mrope is None:
            return None
        return self.mrope[cell]

    def _rot_pos(self, cell: int) -> int:
        return self.pos_base + cell

    def append_indexer(self, raw: np.ndarray, k_norm: np.ndarray, pos: int) -> None:
        """One raw indexer key. Double sum, matching qsa.cu.

        The completing cell is not stored in the tail. The spare key is
        rms(raw of cell 0) rotated at angle 0, copied to pooled[n_bid]
        after every completion. Completed block b rotates at the block's
        first cell (pos_base + b*r), or at that cell's mRoPE triple.
        """
        g = self.geom
        raw64 = np.asarray(raw, dtype=np.float64).reshape(g.idx_dim)
        gamma = np.asarray(k_norm, dtype=np.float64).reshape(g.idx_dim)
        r = g.idx_block
        slot = pos % r
        if slot < r - 1:
            self.tail[slot] = raw64.astype(np.float32)
        if pos == 0:
            y = _rms_double(raw64, gamma, g.eps)
            # Angle 0 even when pos_base or mRoPE says otherwise.
            y = apply_rope_table(y, 0, self.cos_tab, self.sin_tab)
            self.dead = y
            self.pooled[0] = y
            self.have_dead = True
        if slot == r - 1:
            acc = np.zeros((g.idx_dim,), dtype=np.float64)
            for j in range(r - 1):
                acc += self.tail[j].astype(np.float64)
            acc += raw64
            mean = acc / np.float64(r)
            y = _rms_double(mean, gamma, g.eps)
            b = pos // r
            cell0 = b * r
            y = apply_rope_table(
                y,
                self._rot_pos(cell0),
                self.cos_tab,
                self.sin_tab,
                thw=self._rot_thw(cell0),
            )
            self.pooled[b] = y
            self.pooled[b + 1] = self.dead
            self.block_pos = np.int32(self._rot_pos(cell0))
        self.n_kv = pos + 1

    def append_kv(self, k: np.ndarray, v: np.ndarray, pos: int) -> None:
        """k, v are (n_kv, head_dim) f32, already normed. K is already rotated."""
        g = self.geom
        k = np.asarray(k, dtype=np.float32).reshape(g.n_kv, g.head_dim)
        v = np.asarray(v, dtype=np.float32).reshape(g.n_kv, g.head_dim)
        kc, ks = quantize_q8(k)
        vc, vs = quantize_q8(v)
        page_i = pos // g.page_size
        phys = int(self.page_table[page_i])
        slot = pos % g.page_size
        self.k_q[phys, :, slot, :] = kc
        self.v_q[phys, :, slot, :] = vc
        self.k_scale[phys, :, slot, :] = ks
        self.v_scale[phys, :, slot, :] = vs

    def scores(self, q_idx: np.ndarray, n_kv: int, bias: np.ndarray | None = None) -> np.ndarray:
        """Per-cell scores for a query that can see n_kv cells.

        Completed blocks read pooled[b]. The open block reads `dead`,
        not pooled[n_bid]: a chunk append may already have overwritten
        that row with a later block (qsa_select.hpp). At a decode
        boundary the two aliases are equal. Tail cells add +1e9.
        """
        g = self.geom
        q = np.asarray(q_idx, dtype=np.float32).reshape(g.idx_n_head, g.idx_dim)
        n_bid = n_kv // g.idx_block
        if not self.have_dead and n_kv > n_bid * g.idx_block:
            raise RuntimeError("tail score needs the dead key")
        # Build the view indexer_scores expects: completed rows, spare last.
        view = np.zeros((n_bid + 1, g.idx_dim), dtype=np.float32)
        if n_bid:
            view[:n_bid] = self.pooled[:n_bid]
        view[n_bid] = self.dead
        return indexer_scores(
            view, q, n_kv, n_bid=n_bid, block=g.idx_block, bias=bias
        )

    def select(self, q_idx: np.ndarray, n_kv: int) -> np.ndarray:
        width = g_width(self.geom, n_kv)
        return topk_radix(self.scores(q_idx, n_kv), width)

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dequant selected cells to fp16-rounded f32. (n_ids, n_kv, dim)."""
        g = self.geom
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        k = np.zeros((ids.shape[0], g.n_kv, g.head_dim), dtype=np.float32)
        v = np.zeros_like(k)
        for i, cell in enumerate(ids.tolist()):
            phys = int(self.page_table[cell // g.page_size])
            slot = cell % g.page_size
            k[i] = dequant_q8(self.k_q[phys, :, slot, :], self.k_scale[phys, :, slot, :])
            v[i] = dequant_q8(self.v_q[phys, :, slot, :], self.v_scale[phys, :, slot, :])
        return k, v


def g_width(geom: QsaGeom, n_kv: int) -> int:
    return geom.width_at(n_kv)


def _rms_double(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    """y = x * rsqrt(mean(x^2)+eps) * gamma, sum in float64, store f32."""
    xx = np.asarray(x, dtype=np.float64)
    ss = float(np.sum(xx * xx))
    inv = 1.0 / np.sqrt(ss / xx.shape[0] + float(eps))
    y = xx * inv * np.asarray(gamma, dtype=np.float64)
    return y.astype(np.float32)


def project(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """x (..., in) @ weight (in, out). Strata's gemv is outside qsa.hpp."""
    return np.matmul(np.asarray(x, dtype=np.float32), np.asarray(weight, dtype=np.float32))


@dataclass
class QsaWeights:
    """Kernel gammas are the multiply weights (no extra 1+w).

    The GGUF stores (1+w). The fork loader subtracts 1
    (mapping.py TRANSFORM_DEVIATION) and GemmaRMSNorm adds it back.
    Pass the stored GGUF value here, or (1 + module weight) after that
    load. q is packed [query | gate] per head.
    """

    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray
    w_idx_q: np.ndarray
    w_idx_k: np.ndarray
    idx_q_norm: np.ndarray
    idx_k_norm: np.ndarray

    @staticmethod
    def random(geom: QsaGeom, rng: np.random.Generator, scale: float = 0.05) -> "QsaWeights":
        h = geom.hidden
        def m(*shape: int) -> np.ndarray:
            return rng.normal(size=shape).astype(np.float32) * np.float32(scale)

        return QsaWeights(
            w_q=m(h, geom.n_head * 2 * geom.head_dim),
            w_k=m(h, geom.n_kv * geom.head_dim),
            w_v=m(h, geom.n_kv * geom.head_dim),
            w_o=m(geom.n_head * geom.head_dim, h),
            q_norm=np.ones((geom.head_dim,), dtype=np.float32),
            k_norm=np.ones((geom.head_dim,), dtype=np.float32),
            w_idx_q=m(h, geom.idx_n_head * geom.idx_dim),
            w_idx_k=m(h, geom.idx_dim),
            idx_q_norm=np.ones((geom.idx_dim,), dtype=np.float32),
            idx_k_norm=np.ones((geom.idx_dim,), dtype=np.float32),
        )


def _norm_heads(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    return rms_norm_rows(x, gamma, eps)


def prepare_token(
    x: np.ndarray,
    weights: QsaWeights,
    geom: QsaGeom,
    pos: int,
    cos_tab: np.ndarray,
    sin_tab: np.ndarray,
    thw: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Project, RMSNorm (y * gamma), and RoPE. Returns the kernel inputs."""
    q_full = project(x, weights.w_q).reshape(geom.n_head, 2 * geom.head_dim)
    q = _norm_heads(q_full[:, : geom.head_dim], weights.q_norm, geom.eps)
    k = project(x, weights.w_k).reshape(geom.n_kv, geom.head_dim)
    k = _norm_heads(k, weights.k_norm, geom.eps)
    v = project(x, weights.w_v).reshape(geom.n_kv, geom.head_dim)
    q = apply_rope_table(q, pos, cos_tab, sin_tab, thw=thw)
    k = apply_rope_table(k, pos, cos_tab, sin_tab, thw=thw)
    iq = project(x, weights.w_idx_q).reshape(geom.idx_n_head, geom.idx_dim)
    iq = _norm_heads(iq, weights.idx_q_norm, geom.eps)
    iq = apply_rope_table(iq, pos, cos_tab, sin_tab, thw=thw)
    raw = project(x, weights.w_idx_k).reshape(geom.idx_dim)
    return {"q_full": q_full, "q": q, "k": k, "v": v, "q_idx": iq, "raw": raw}


def attend_ids(
    q: np.ndarray,
    state: QsaState,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k, v = state.gather(ids)
    # attend() wants fp32 and does the softmax. The gather already rounded
    # through fp16, which is what qsa_attend reads from its scratch.
    return attend(q, k, v, n_kv_heads=state.geom.n_kv)


def gate_fp16(attn: np.ndarray, q_full: np.ndarray) -> np.ndarray:
    """sigmoid gate, then one round to fp16 (qsa_gate_apply)."""
    y = gate_apply(attn, q_full)
    return np.float32(np.float16(y))


def decode_token(
    state: QsaState,
    weights: QsaWeights,
    x: np.ndarray,
    pos: int,
    *,
    thw: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Append this cell and attend. pos is the cell index for a sequence
    whose positions are pos_base + cell (the state holds pos_base)."""
    geom = state.geom
    abs_pos = state.pos_base + pos
    if thw is None and state.mrope is not None:
        thw = state.mrope[pos]
    tok = prepare_token(
        x, weights, geom, abs_pos, state.cos_tab, state.sin_tab, thw=thw
    )
    state.append_kv(tok["k"], tok["v"], pos)
    state.append_indexer(tok["raw"], weights.idx_k_norm, pos)
    ids = state.select(tok["q_idx"], pos + 1)
    attn, weights_out = attend_ids(tok["q"], state, ids)
    gated = gate_fp16(attn, tok["q_full"])
    y = project(gated.reshape(-1), weights.w_o)
    return {
        "ids": ids,
        "attn": attn,
        "gated": gated,
        "y": y,
        "q": tok["q"],
        "k": tok["k"],
        "v": tok["v"],
        "q_idx": tok["q_idx"],
        "raw": tok["raw"],
        "q_full": tok["q_full"],
    }


def prefill(
    state: QsaState,
    weights: QsaWeights,
    xs: np.ndarray,
    *,
    pos0: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Token-by-token. This is the oracle the chunk path must match."""
    xs = np.asarray(xs, dtype=np.float32)
    return [decode_token(state, weights, xs[i], pos0 + i) for i in range(xs.shape[0])]


def prefill_chunk(
    state: QsaState,
    weights: QsaWeights,
    xs: np.ndarray,
    *,
    pos0: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Append the whole chunk, then score each query with its own n_kv.

    Tail cells use `dead` rather than pooled[n_bid], because later
    completions have moved the spare. Outputs match prefill().
    """
    xs = np.asarray(xs, dtype=np.float32)
    geom = state.geom
    prepared = []
    for i in range(xs.shape[0]):
        pos = pos0 + i
        thw = state.mrope[pos] if state.mrope is not None else None
        tok = prepare_token(
            xs[i], weights, geom, state.pos_base + pos, state.cos_tab, state.sin_tab, thw=thw
        )
        state.append_kv(tok["k"], tok["v"], pos)
        state.append_indexer(tok["raw"], weights.idx_k_norm, pos)
        prepared.append(tok)
    out = []
    for i, tok in enumerate(prepared):
        n_kv = pos0 + i + 1
        ids = state.select(tok["q_idx"], n_kv)
        attn, _w = attend_ids(tok["q"], state, ids)
        gated = gate_fp16(attn, tok["q_full"])
        y = project(gated.reshape(-1), weights.w_o)
        out.append({"ids": ids, "attn": attn, "gated": gated, "y": y, "q_idx": tok["q_idx"]})
    return out


def vllm_block_expand_ids(scores: np.ndarray, n_kv: int, geom: QsaGeom) -> np.ndarray:
    """Whole-block expand used by vLLM's QSA indexer, for the contrast test.

    Keeps the top (token_topk / compress_ratio) completed blocks, all of
    their cells, plus the open tail. Strata instead keeps
    min(n_kv, top_k+block-1) cells, so it can cut the marginal block.
    """
    r = geom.idx_block
    n_bid = n_kv // r
    n_tail = n_kv - n_bid * r
    keep_blocks = geom.idx_top_k // r
    if n_bid == 0:
        return np.arange(n_kv, dtype=np.int32)
    # One score per completed block (cell 0 of the block).
    block_scores = np.asarray(scores[: n_bid * r], dtype=np.float32).reshape(n_bid, r)[:, 0]
    order = np.lexsort((np.arange(n_bid), -block_scores))
    chosen = order[: min(keep_blocks, n_bid)]
    cells = []
    for b in chosen.tolist():
        cells.extend(range(b * r, (b + 1) * r))
    cells.extend(range(n_bid * r, n_bid * r + n_tail))
    return np.sort(np.asarray(cells, dtype=np.int32))
