"""Hyper-connection mix and combine.

Strata gr.hpp:
  lo = silu((xn @ w_down.T) / hc)     # divide inside silu
  mixed = mean_c( xn_c * sigmoid(lo @ w_up.T) )
  inject = xn @ w_inject.T
  R_out = R + block_out * 2 * sigmoid(inject / hc)

w_norm is the gamma that multiplies the per-stream RMS. The GGUF converter
stores (1+w). The XPU module stores the deviation and adds 1 in forward,
so loader-then-forward multiplies by the same stored gamma.
"""

from __future__ import annotations

import numpy as np

from .geometry import FLASHNEXT

G = FLASHNEXT


def _as_streams(residual: np.ndarray, hc: int, n_embd: int) -> np.ndarray:
    x = np.asarray(residual, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, hc, n_embd)
    elif x.ndim == 2 and x.shape[-1] == hc * n_embd:
        x = x.reshape(x.shape[0], hc, n_embd)
    if x.shape[-2:] != (hc, n_embd):
        raise ValueError(f"residual {x.shape} is not (T, {hc}, {n_embd})")
    return x


def grouped_rms(x: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(var + np.float32(eps)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def gr_read(
    residual: np.ndarray,
    w_norm: np.ndarray,
    w_down: np.ndarray,
    w_up: np.ndarray,
    w_inject: np.ndarray | None,
    *,
    hc: int = G.hc,
    n_embd: int = G.n_embd,
    eps: float = G.rms_eps,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (mixed [T, n_embd], inject [T, hc] or None).

    w_down is (hc_lr, hc*n_embd), w_up is (hc*n_embd, hc_lr),
    w_inject is (hc, hc*n_embd). These match ggml ne0-fast rows and
    torch.nn.functional.linear weights.
    """
    streams = _as_streams(residual, hc, n_embd)
    gamma = np.asarray(w_norm, dtype=np.float32).reshape(hc, n_embd)
    xn = grouped_rms(streams, eps) * gamma
    flat = xn.reshape(xn.shape[0], hc * n_embd)
    down = np.asarray(w_down, dtype=np.float32)
    up = np.asarray(w_up, dtype=np.float32)
    proj = flat @ down.T
    lo = silu(proj / np.float32(hc))
    gate = 1.0 / (1.0 + np.exp(-(lo @ up.T)))
    gated = (flat * gate).reshape(xn.shape)
    mixed = gated.mean(axis=1)
    inject = None
    if w_inject is not None:
        inject = flat @ np.asarray(w_inject, dtype=np.float32).T
    return mixed, inject


def gr_write(
    residual: np.ndarray,
    block_out: np.ndarray,
    inject: np.ndarray,
    *,
    hc: int = G.hc,
    n_embd: int = G.n_embd,
) -> np.ndarray:
    streams = _as_streams(residual, hc, n_embd)
    block = np.asarray(block_out, dtype=np.float32)
    if block.ndim == 1:
        block = block.reshape(1, n_embd)
    inj = np.asarray(inject, dtype=np.float32).reshape(streams.shape[0], hc)
    scale = 2.0 / (1.0 + np.exp(-(inj / np.float32(hc))))
    out = streams + block[:, None, :] * scale[:, :, None]
    return out


def vllm_xpu_mix(
    residual: np.ndarray,
    w_deviation: np.ndarray,
    w_down: np.ndarray,
    w_up: np.ndarray,
    w_inject: np.ndarray | None,
    *,
    hc: int = G.hc,
    n_embd: int = G.n_embd,
    eps: float = G.rms_eps,
) -> tuple[np.ndarray, np.ndarray | None]:
    """XPU HyperConnection.mix: gamma = 1 + stored deviation."""
    gamma = 1.0 + np.asarray(w_deviation, dtype=np.float32)
    return gr_read(
        residual, gamma, w_down, w_up, w_inject, hc=hc, n_embd=n_embd, eps=eps
    )
