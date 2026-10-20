"""QSA path for the fork's Qwen4ExpDecoderLayer. Default off (B70_QSA=1).

The runner owns the int8 KV pool and the indexer tail. It does not call
vLLM's Attention, so the paged block manager is not the QSA cache. TP=1,
one sequence, positions are the cell indices. The first forward allocates;
do not capture that forward.

Indexer K gamma passed into the kernel is (1 + GemmaRMSNorm.weight). The
GGUF loader stores w after subtracting 1, and GemmaRMSNorm adds it back.
The kernel multiplies by gamma directly (qsa.cu, y * w_k_norm).

Main q and k come from Qwen3NextAttention._project_qkv_gate, already
normed and rotated. Indexer queries are normed and rotated here. Indexer
keys stay raw; indexer_append pools them.
"""

from __future__ import annotations

import os

import torch

from b70_qsa_xpu.state import Geom, QsaBuffers


def _tp_is_one() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return int(get_tensor_model_parallel_world_size()) == 1
    except Exception:
        return True


def _capturing() -> bool:
    try:
        return bool(torch.xpu.is_current_stream_capturing())
    except Exception:
        return False


def attach_qsa(layer):
    """Return a runner, or None when this layer has no indexer weights."""
    attn = getattr(layer, "self_attn", None)
    indexer = getattr(attn, "indexer", None) if attn is not None else None
    if attn is None or indexer is None:
        return None
    return QsaRunner(attn, indexer)


class QsaRunner:
    def __init__(self, attn, indexer) -> None:
        self.attn = attn
        self.indexer = indexer
        self.buf: QsaBuffers | None = None
        self._ready = False

    def _geom(self) -> Geom:
        attn = self.attn
        cfg = attn.config
        rope = getattr(cfg, "rope_parameters", None) or {}
        if not isinstance(rope, dict):
            rope = {}
        n_rot = int(getattr(attn.rotary_emb, "rotary_dim", 0) or 0)
        if n_rot <= 0:
            n_rot = int(round(float(rope.get("partial_rotary_factor", 0.25)) * attn.head_dim))
        base = float(rope.get("rope_theta", 1.0e7))
        return Geom(
            n_head=int(attn.num_heads),
            n_kv=int(attn.num_kv_heads),
            head_dim=int(attn.head_dim),
            n_rot=n_rot,
            idx_n_head=int(self.indexer.index_n_heads),
            idx_dim=int(self.indexer.index_head_dim),
            idx_block=int(getattr(cfg, "indexer_compress_ratio", 4) or 4),
            idx_top_k=int(getattr(cfg, "indexer_budget", 2048) or 2048),
            rope_base=base,
            eps=float(getattr(cfg, "rms_norm_eps", 1.0e-6)),
            page_size=int(getattr(cfg, "indexer_compress_ratio", 4) or 4),
        )

    def _ensure(self, device: torch.device) -> QsaBuffers:
        if self.buf is None:
            if _capturing():
                raise RuntimeError("B70_QSA allocates on the first forward; warmup before capture")
            if int(self.indexer.index_kv_heads) != 1:
                raise RuntimeError("B70_QSA indexer kv heads must be 1")
            max_cells = int(os.environ.get("B70_QSA_MAX_CELLS", "8192"))
            self.buf = QsaBuffers(max_cells, self._geom(), device)
            self._fill_tables()
            gamma = self.indexer.k_layernorm.weight.detach().float() + 1
            self.buf.k_norm.copy_(gamma)
            g = self.buf.geom
            dev = self.buf.device
            self.q = torch.zeros(g.n_head, g.head_dim, dtype=torch.float32, device=dev)
            self.k = torch.zeros(g.n_kv, g.head_dim, dtype=torch.float32, device=dev)
            self.v = torch.zeros(g.n_kv, g.head_dim, dtype=torch.float32, device=dev)
            self.q_idx = torch.zeros(g.idx_n_head, g.idx_dim, dtype=torch.float32, device=dev)
            self.raw = torch.zeros(g.idx_dim, dtype=torch.float32, device=dev)
            self.q_full = torch.zeros(g.n_head, g.head_dim * 2, dtype=torch.float32, device=dev)
            self._q_gamma = (self.indexer.q_layernorm.weight.detach().float() + 1).contiguous()
            self._ready = True
        return self.buf

    def _fill_tables(self) -> None:
        buf = self.buf
        assert buf is not None
        half = buf.cos.shape[1]
        if half == 0:
            return
        span = buf.cos.shape[0]
        inv = 1.0 / (buf.geom.rope_base ** (torch.arange(half, dtype=torch.float64) / half))
        pos = torch.arange(span, dtype=torch.float64)[:, None]
        ang = pos * inv
        buf.cos.copy_(torch.cos(ang).float())
        buf.sin.copy_(torch.sin(ang).float())

    def _rope_(self, x: torch.Tensor, pos: torch.Tensor, thw: torch.Tensor | None) -> None:
        half = self.buf.cos.shape[1]
        if half == 0:
            return
        pair = torch.arange(half, device=x.device)
        if thw is None:
            c = self.buf.cos[pos]
            s = self.buf.sin[pos]
        else:
            p = thw[pair % 3]
            c = self.buf.cos[p, pair]
            s = self.buf.sin[p, pair]
        # Both halves from the pre-rotate values. The second write must not
        # observe the first.
        x1 = x[..., :half].clone()
        x2 = x[..., half : 2 * half].clone()
        x[..., :half] = x1 * c - x2 * s
        x[..., half : 2 * half] = x2 * c + x1 * s

    def _one(self, i: int, q, k, v, gate, iq, raw, positions) -> None:
        buf = self.buf
        g = buf.geom
        assert buf is not None
        pos_rows = positions if positions.ndim == 1 else positions[0]
        pos_i = pos_rows[i].to(torch.int32)
        buf.pos.copy_(pos_i.reshape(1))
        self.q.copy_(q[i].reshape(g.n_head, g.head_dim).float())
        self.k.copy_(k[i].reshape(g.n_kv, g.head_dim).float())
        self.v.copy_(v[i].reshape(g.n_kv, g.head_dim).float())
        gate_i = gate[i].reshape(g.n_head, g.head_dim).float()
        self.q_full[:, : g.head_dim].copy_(self.q)
        self.q_full[:, g.head_dim :].copy_(gate_i)
        iq_i = iq[i].reshape(self.indexer.index_n_heads, self.indexer.index_head_dim).float()
        var = iq_i.pow(2).mean(dim=-1, keepdim=True)
        iq_i = iq_i * torch.rsqrt(var + g.eps) * self._q_gamma
        thw = None
        if positions.ndim == 2:
            thw = positions[:, i].to(torch.int32)
            if buf.mrope is None:
                if _capturing():
                    raise RuntimeError("B70_QSA mRoPE table allocates on first image token")
                buf.set_mrope(torch.zeros(buf.max_cells, 3, dtype=torch.int32, device=buf.device))
            buf.mrope[pos_i.long()] = thw
        self._rope_(iq_i, pos_i.long(), thw)
        self.q_idx.copy_(iq_i)
        raw_i = raw[i].reshape(self.indexer.index_kv_heads, self.indexer.index_head_dim)[0]
        self.raw.copy_(raw_i.float())
        buf.run(self.q, self.k, self.v, self.q_idx, self.raw, self.q_full)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not _tp_is_one():
            raise RuntimeError("B70_QSA is TP=1 only")
        if positions is None:
            raise RuntimeError("B70_QSA needs positions")
        buf = self._ensure(hidden.device)
        attn = self.attn
        qkv, _ = attn.qkv_proj(hidden)
        q, k, v, gate = attn._project_qkv_gate(qkv, positions)
        if gate is None:
            raise RuntimeError("QSA requires the attention output gate")
        idx = self.indexer
        qk, _ = idx.index_qk_proj(hidden)
        q_out = idx.index_n_heads * idx.index_head_dim
        iq = qk[:, :q_out]
        raw = qk[:, q_out:]
        parts = []
        n = hidden.shape[0]
        for i in range(n):
            self._one(i, q, k, v, gate, iq, raw, positions)
            parts.append(buf.gated.reshape(1, -1))
        y = torch.cat(parts, dim=0).to(dtype=hidden.dtype)
        out, _ = attn.o_proj(y)
        return out
