"""QSA indexer, selection, attention, and the per-head sigmoid gate.

Decode formulas from include/strata/kernels/qsa.hpp. Projections are not
here: the caller hands in the projected q/k/v and the raw indexer key.

Selection width is min(n_kv, idx_top_k + idx_block - 1). Below that bound
the selection is the identity, so dense attention matches QSA.
"""

from __future__ import annotations

import numpy as np

from .geometry import FLASHNEXT

G = FLASHNEXT


def selection_width(n_kv: int, top_k: int = G.idx_top_k, block: int = G.idx_block) -> int:
    if n_kv < 0:
        raise ValueError("n_kv < 0")
    return int(min(n_kv, top_k + block - 1))


def rms_norm_rows(x: np.ndarray, gamma: np.ndarray, eps: float = G.rms_eps) -> np.ndarray:
    """RMS over the last axis. gamma broadcasts on that axis. No (1+w)."""
    vec = np.asarray(x, dtype=np.float32)
    g = np.asarray(gamma, dtype=np.float32)
    var = np.mean(np.square(vec), axis=-1, keepdims=True)
    return vec * np.reciprocal(np.sqrt(var + np.float32(eps))) * g


def rope_neox(
    x: np.ndarray,
    positions: np.ndarray,
    *,
    rotary_dim: int,
    base: float = G.rope_base,
) -> np.ndarray:
    """NeoX rotary on the first rotary_dim axes. Position 0 is the identity.

    x: (..., dim). positions broadcasts against the leading axes.
    """
    vec = np.asarray(x, dtype=np.float32).copy()
    half = rotary_dim // 2
    if rotary_dim > vec.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and <= dim")
    inv = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))
    pos = np.asarray(positions, dtype=np.float64)
    # Broadcast positions onto every axis except the last.
    while pos.ndim < vec.ndim - 1:
        pos = pos[..., None]
    ang = pos[..., None] * inv
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    x1 = vec[..., :half]
    x2 = vec[..., half:rotary_dim]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    vec[..., :half] = out1
    vec[..., half:rotary_dim] = out2
    return vec


def pool_indexer_keys(
    raw: np.ndarray,
    k_norm: np.ndarray,
    *,
    block: int = G.idx_block,
    rotary_dim: int = G.rope_dim,
    pos_base: int = 0,
) -> tuple[np.ndarray, int]:
    """raw (n_kv, idx_dim) -> pooled (n_bid+1, idx_dim), spare slot last.

    Completed block b is rope(rms(mean(raw[b*r:b*r+r])), pos of cell b*r).
    The spare slot is rope(rms(raw[0]), 0), which is rms(raw[0]).
    """
    keys = np.asarray(raw, dtype=np.float32)
    n_kv = keys.shape[0]
    n_bid = n_kv // block
    dim = keys.shape[-1]
    pooled = np.zeros((n_bid + 1, dim), dtype=np.float32)
    gamma = np.asarray(k_norm, dtype=np.float32).reshape(dim)
    for b in range(n_bid):
        mean = keys[b * block:(b + 1) * block].mean(axis=0)
        normed = rms_norm_rows(mean, gamma)
        pos = np.array(pos_base + b * block)
        pooled[b] = rope_neox(normed, pos, rotary_dim=min(rotary_dim, dim))
    spare = rms_norm_rows(keys[0], gamma)
    pooled[n_bid] = rope_neox(spare, np.array(0), rotary_dim=min(rotary_dim, dim))
    return pooled, n_bid


def cell_block(cell: int, n_bid: int, block: int) -> int:
    return min(cell // block, n_bid)


def indexer_scores(
    pooled: np.ndarray,
    q_idx: np.ndarray,
    n_kv: int,
    *,
    n_bid: int | None = None,
    block: int = G.idx_block,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """score[j] = bias[cb] + sum_h relu(dot(pooled[cb], q_idx[h])).

    Incomplete tail cells (j >= n_bid * block) get +1e9. Relu is per head.
    q_idx is (n_head, dim), already normed and rotated.
    """
    keys = np.asarray(pooled, dtype=np.float32)
    query = np.asarray(q_idx, dtype=np.float32)
    if n_bid is None:
        n_bid = keys.shape[0] - 1
    dots = np.maximum(query @ keys.T, 0.0)  # (heads, n_bid+1)
    summed = dots.sum(axis=0)
    scores = np.empty(n_kv, dtype=np.float32)
    for j in range(n_kv):
        cb = cell_block(j, n_bid, block)
        scores[j] = summed[cb]
        if bias is not None:
            scores[j] += np.float32(bias[cb])
        if j >= n_bid * block:
            scores[j] += np.float32(1e9)
    return scores


def topk_ids(scores: np.ndarray, width: int) -> np.ndarray:
    """Largest `width` cells, ties by ascending index, returned ascending."""
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.shape[0]
    width = min(width, n)
    order = np.lexsort((np.arange(n), -scores))
    chosen = np.sort(order[:width])
    return chosen.astype(np.int32)


def attend(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    n_kv_heads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax GQA. q (n_q, dim), k/v (n_ids, n_kv, dim).

    KV head = q_head // (n_q / n_kv), integer division, not modulo.
    Returns attn (n_q, dim) and weights (n_q, n_ids).
    """
    query = np.asarray(q, dtype=np.float32)
    key = np.asarray(k, dtype=np.float32)
    val = np.asarray(v, dtype=np.float32)
    n_q, dim = query.shape
    if key.ndim == 2:
        key = key[:, None, :]
        val = val[:, None, :]
    n_kv = key.shape[1] if n_kv_heads is None else n_kv_heads
    if n_q % n_kv:
        raise ValueError(f"n_q {n_q} not divisible by n_kv {n_kv}")
    group = n_q // n_kv
    scale = 1.0 / np.sqrt(np.float32(dim))
    attn = np.zeros((n_q, dim), dtype=np.float32)
    weights = np.zeros((n_q, key.shape[0]), dtype=np.float32)
    if key.shape[0] == 0:
        return attn, weights
    for h in range(n_q):
        kv_h = h // group
        logits = (key[:, kv_h, :] @ query[h]) * scale
        logits = logits - logits.max()
        w = np.exp(logits)
        w /= w.sum()
        weights[h] = w
        attn[h] = w @ val[:, kv_h, :]
    return attn, weights


def gate_apply(attn: np.ndarray, q_full: np.ndarray) -> np.ndarray:
    """attn * sigmoid(gate). q_full is (n_head, 2*head_dim), gate is the
    second half of each head. Stays in fp32; the fp16 round is the caller's."""
    query = np.asarray(q_full, dtype=np.float32)
    head = query.shape[-1] // 2
    gate = query[..., head:]
    sig = 1.0 / (1.0 + np.exp(-gate))
    return np.asarray(attn, dtype=np.float32) * sig
