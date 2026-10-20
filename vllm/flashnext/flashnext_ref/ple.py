"""PLE hash, IQ4_NL row dequant, and the ple_block residual update.

Strata: src/kernels/ngram.cpp ngram_rows / iq4nl_dequant_row, and
include/strata/kernels/ple.hpp. The XPU transcription is
vllm/models/qwen4_exp/xpu/model.py Qwen4ExpPLE._compute_rows.
"""

from __future__ import annotations

import numpy as np

from .geometry import FLASHNEXT, IQ4NL_CODES, PleConsts

MASK64 = (1 << 64) - 1
G = FLASHNEXT


def mul_u64(token: int, multiplier: int) -> int:
    """uint64 wrapping product. A negative token is two's-complement."""
    return (int(token) * int(multiplier)) & MASK64


def ngram_mixed(ctx: list[int], mult: tuple[int, ...], n: int) -> int:
    mixed = mul_u64(ctx[0], mult[0])
    for j in range(1, n):
        mixed ^= mul_u64(ctx[j], mult[j])
    return mixed


def ngram_rows(
    tokens: np.ndarray,
    prev: np.ndarray,
    consts: PleConsts,
    *,
    eos: int = G.ple_eos,
) -> np.ndarray:
    """Sixteen row ids per token.

    tokens: (T,) int
    prev:   (T, 2) int, OLDEST first. TOKEN_NULL (-1) means missing.
    An EOS or a negative predecessor replaces that slot and every older
    slot with eos. The current token is never cut.
    """
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    prev = np.asarray(prev, dtype=np.int64)
    if prev.shape != (tokens.shape[0], G.ngram - 1):
        raise ValueError(f"prev shape {prev.shape} != {(tokens.shape[0], G.ngram - 1)}")
    out = np.empty((tokens.shape[0], G.ple_n_heads), dtype=np.uint32)
    n_prev = G.ngram - 1
    for i, tok in enumerate(tokens.tolist()):
        ctx = [0] * G.ngram
        ctx[0] = int(tok)
        cut = False
        for s in range(1, G.ngram):
            raw = G.token_null if cut else int(prev[i, n_prev - s])
            cut = cut or raw < 0 or raw == eos
            ctx[s] = eos if cut else raw
        for n in range(2, G.ngram + 1):
            mixed = ngram_mixed(ctx, consts.mult, n)
            base = (n - 2) * G.heads_per_ngram
            for g in range(G.heads_per_ngram):
                h = base + g
                out[i, h] = (mixed % int(consts.vocab[h]) + int(consts.offset[h])) & 0xFFFFFFFF
    return out


def strata_sequence(
    tokens: list[int],
    consts: PleConsts,
    *,
    oldest: int = G.token_null,
    newest: int = G.token_null,
    eos: int = G.ple_eos,
) -> np.ndarray:
    """Walk a sequence the way the engine does: prev is the two real
    predecessors, missing ones stay TOKEN_NULL until the cut rewrites them."""
    rows = []
    old, new = int(oldest), int(newest)
    for tok in tokens:
        prev = np.array([[old, new]], dtype=np.int64)
        rows.append(ngram_rows(np.array([tok]), prev, consts, eos=eos)[0])
        old, new = new, int(tok)
    return np.stack(rows, axis=0)


def xpu_sequence(
    tokens: list[int],
    consts: PleConsts,
    *,
    has_init: bool = False,
    prev_newest: int | None = None,
    prev_older: int | None = None,
    eos: int = G.ple_eos,
) -> np.ndarray:
    """Transcription of Qwen4ExpPLE._compute_rows (xpu/model.py).

    A fresh sequence starts from eos, not TOKEN_NULL. c2 is forced to eos
    only when the newest predecessor is eos. A negative id is multiplied,
    not rewritten to eos.
    """
    if not has_init:
        prev1 = eos
        prev2 = eos
    else:
        if prev_newest is None or prev_older is None:
            raise ValueError("has_init requires both predecessors")
        prev1 = int(prev_newest)
        prev2 = int(prev_older)
    m = consts.mult
    rows = np.empty((len(tokens), G.ple_n_heads), dtype=np.uint32)
    for i, c0 in enumerate(tokens):
        c1 = prev1
        c2 = eos if prev1 == eos else prev2
        mixed2 = mul_u64(c0, m[0]) ^ mul_u64(c1, m[1])
        mixed3 = mixed2 ^ mul_u64(c2, m[2])
        for h in range(G.heads_per_ngram):
            rows[i, h] = (mixed2 % int(consts.vocab[h]) + int(consts.offset[h])) & 0xFFFFFFFF
        for h in range(G.heads_per_ngram, G.ple_n_heads):
            rows[i, h] = (mixed3 % int(consts.vocab[h]) + int(consts.offset[h])) & 0xFFFFFFFF
        prev2 = prev1
        prev1 = int(c0)
    return rows


def iq4nl_dequant_row(row90: np.ndarray) -> np.ndarray:
    """One 90-byte IQ4_NL row -> 160 float32. Split-half nibbles, no -8.

    Byte j holds elements j and j+16, matching dequantize_row_iq4_nl.
    """
    row = np.ascontiguousarray(row90, dtype=np.uint8).reshape(-1)
    if row.shape[0] != G.ple_row_bytes_iq4nl:
        raise ValueError(f"IQ4_NL row is {row.shape[0]} bytes, want {G.ple_row_bytes_iq4nl}")
    out = np.empty(G.ple_head_dim, dtype=np.float32)
    for b in range(G.ple_head_dim // 32):
        blk = row[b * 18:(b + 1) * 18]
        scale = np.frombuffer(blk[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        qs = blk[2:]
        for j in range(16):
            out[b * 32 + j] = scale * IQ4NL_CODES[int(qs[j]) & 0x0F]
            out[b * 32 + j + 16] = scale * IQ4NL_CODES[int(qs[j]) >> 4]
    return out


def iq4nl_dequant_rows(raw: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(raw, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed.reshape(1, -1)
    return np.stack([iq4nl_dequant_row(packed[i]) for i in range(packed.shape[0])], axis=0)


def grouped_rms(x: np.ndarray, gamma: np.ndarray, eps: float = G.rms_eps) -> np.ndarray:
    """Per-stream RMSNorm. x is (..., hc, n_embd), gamma is (hc * n_embd,).

    gamma is the value that multiplies the normalized stream. Pass the
    GGUF bytes widened to fp32. Do not add 1 here: the caller decides
    whether the file already stores (1+w).
    """
    hc = x.shape[-2]
    n_embd = x.shape[-1]
    flat_g = np.asarray(gamma, dtype=np.float32).reshape(hc, n_embd)
    var = np.mean(np.square(x.astype(np.float32)), axis=-1, keepdims=True)
    normed = x.astype(np.float32) * np.reciprocal(np.sqrt(var + eps))
    return normed * flat_g


def ple_gate(score: np.ndarray) -> np.ndarray:
    """sigmoid(sign(s) * sqrt(max(|s|, 1e-6))). sign(0) is 0."""
    s = np.asarray(score, dtype=np.float32)
    sign = np.sign(s)
    mag = np.sqrt(np.maximum(np.abs(s), 1e-6))
    return (1.0 / (1.0 + np.exp(-sign * mag))).astype(np.float32)


def ple_conv(
    normalized: np.ndarray,
    kernel: np.ndarray,
    hist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dilated causal conv. kernel is (hc_dim, kern) in ggml order k-fast.

    normalized: (T, hc, n_embd). Returns silu(conv) in the same layout and
    the new history of normalized rows, oldest first, length ple_hist.
    """
    x = np.asarray(normalized, dtype=np.float32)
    t = x.shape[0]
    flat = x.reshape(t, G.hc_dim)
    weight = np.asarray(kernel, dtype=np.float32).reshape(G.hc_dim, G.ple_conv_kernel)
    hist_rows = G.ple_hist
    if hist is None:
        state = np.zeros((hist_rows, G.hc_dim), dtype=np.float32)
    else:
        state = np.asarray(hist, dtype=np.float32).reshape(hist_rows, G.hc_dim)
    padded = np.concatenate([state, flat], axis=0)
    acc = np.zeros((t, G.hc_dim), dtype=np.float32)
    for k in range(G.ple_conv_kernel):
        offset = hist_rows - (G.ple_conv_kernel - 1 - k) * G.ngram
        acc += padded[offset:offset + t] * weight[:, k]
    conv = acc / (1.0 + np.exp(-acc))  # silu
    new_hist = padded[-hist_rows:]
    return conv.reshape(x.shape), new_hist


def ple_block(
    emb: np.ndarray,
    hidden: np.ndarray,
    hist: np.ndarray | None,
    key_w: np.ndarray,
    value_w: np.ndarray,
    norm_key: np.ndarray,
    norm_query: np.ndarray,
    norm_conv: np.ndarray,
    conv_w: np.ndarray,
) -> dict[str, np.ndarray]:
    """One token or a short run. Weights are fp32 in PyTorch layout.

    key_w:   (hc_dim, n_embd) so emb @ key_w.T is the key projection.
    value_w: (n_embd, n_embd)
    conv_w:  (hc_dim, kern)
    Norms are multiplied as given (already the gamma the kernel uses).
    """
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim == 2:
        hidden = hidden.reshape(hidden.shape[0], G.hc, G.n_embd)
    key = (emb @ np.asarray(key_w, dtype=np.float32).T).reshape(-1, G.hc, G.n_embd)
    value = emb @ np.asarray(value_w, dtype=np.float32).T
    key_n = grouped_rms(key, norm_key)
    query_n = grouped_rms(hidden, norm_query)
    score = np.sum(key_n * query_n, axis=-1) / np.sqrt(np.float32(G.n_embd))
    gate = ple_gate(score)
    gated = value[:, None, :] * gate[..., None]
    normalized = grouped_rms(gated, norm_conv)
    conv, new_hist = ple_conv(normalized, conv_w, hist)
    out = hidden + gated + conv
    return {
        "key": key,
        "value": value,
        "score": score,
        "gate": gate,
        "gated": gated,
        "normalized": normalized,
        "conv": conv,
        "hist": new_hist,
        "out": out,
    }
