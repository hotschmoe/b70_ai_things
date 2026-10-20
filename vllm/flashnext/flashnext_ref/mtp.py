"""MTP fc-stage layouts.

Strata include/strata/core/mtp.hpp, default (DD does not pass
--mtp-hnorm stream):

  e = fc_embedding(rms(embed) * gamma_e)          # gamma length H
  h = rms over the full hc*H vector * gamma_h     # gamma length hc*H
  h_c = fc_hidden(h_c) for each stream c
  R = h + e                                        # e broadcast onto streams

The opt-in stream norm is one RMS per stream. The GGUF sidecar is a
different head: one eh_proj over concat(norm(embed), mean_collapse(norm(wide))).
That is vllm/.../gguf/mtp_proj.py try_gguf_fc_stage. The daily driver
loads mtp-bf16/rt, which is the two-matrix wide path, not eh_proj.
"""

from __future__ import annotations

import numpy as np

from .geometry import FLASHNEXT

G = FLASHNEXT


def _rms(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    vec = np.asarray(x, dtype=np.float32)
    g = np.asarray(gamma, dtype=np.float32)
    var = np.mean(np.square(vec), axis=-1, keepdims=True)
    return vec * np.reciprocal(np.sqrt(var + np.float32(eps))) * g


def fc_wide(
    embed: np.ndarray,
    wide: np.ndarray,
    gamma_e: np.ndarray,
    gamma_h: np.ndarray,
    w_emb: np.ndarray,
    w_hid: np.ndarray,
    *,
    per_stream_norm: bool = False,
    eps: float = G.rms_eps,
) -> np.ndarray:
    """Return the wide residual (T, hc, H).

    w_emb and w_hid are (H, H), torch linear layout (out, in).
    per_stream_norm False is the DD default.
    """
    tok = np.asarray(embed, dtype=np.float32)
    residual = np.asarray(wide, dtype=np.float32)
    if tok.ndim == 1:
        tok = tok.reshape(1, -1)
    if residual.ndim == 2:
        residual = residual.reshape(tok.shape[0], G.hc, G.n_embd)
    e = _rms(tok, gamma_e, eps) @ np.asarray(w_emb, dtype=np.float32).T
    if per_stream_norm:
        # gamma_h is (hc, H). One RMS per stream.
        g = np.asarray(gamma_h, dtype=np.float32).reshape(G.hc, G.n_embd)
        h = _rms(residual, g, eps)
    else:
        flat = residual.reshape(residual.shape[0], G.hc * G.n_embd)
        g = np.asarray(gamma_h, dtype=np.float32).reshape(G.hc * G.n_embd)
        h = _rms(flat, g, eps).reshape(residual.shape)
    proj = h @ np.asarray(w_hid, dtype=np.float32).T
    return proj + e[:, None, :]


def fc_gguf_eh(
    embed: np.ndarray,
    wide: np.ndarray,
    gamma_e: np.ndarray,
    gamma_h: np.ndarray,
    eh_weight: np.ndarray,
    *,
    eps: float = G.rms_eps,
) -> np.ndarray:
    """XPU GGUF path: mean-collapse the wide norm, one proj, broadcast.

    eh_weight is (H, 2H): columns are concat(embed_norm, collapsed_hidden).
    gamma_h is (hc*H,) and the collapse is the XPU
    _GroupedRMSNormCollapse.forward (per-stream RMS, then mean), which is
    not the same as one RMS over hc*H.
    """
    tok = np.asarray(embed, dtype=np.float32)
    residual = np.asarray(wide, dtype=np.float32)
    if tok.ndim == 1:
        tok = tok.reshape(1, -1)
    if residual.ndim == 2:
        residual = residual.reshape(tok.shape[0], G.hc, G.n_embd)
    e = _rms(tok, gamma_e, eps)
    g = np.asarray(gamma_h, dtype=np.float32).reshape(G.hc, G.n_embd)
    hidden = _rms(residual, g, eps).mean(axis=1)
    fused = np.concatenate([e, hidden], axis=-1)
    proj = fused @ np.asarray(eh_weight, dtype=np.float32).T
    return np.broadcast_to(proj[:, None, :], residual.shape).copy()
