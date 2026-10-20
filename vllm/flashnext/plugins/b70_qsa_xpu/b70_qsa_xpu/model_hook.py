"""QSA path for the fork's Qwen4ExpDecoderLayer. Default off (B70_QSA=1).

When the attention layer's KV cache is a uint8 QSA page, the runner
writes that cache through the vLLM block table. Indexer tail, dead key,
block_pos, and the token count live beside the page, keyed by the
physical id of logical block 0. Otherwise the runner keeps the private
one-sequence pool and logs once.

TP=1. The first forward allocates. Do not capture that forward.
Decode (one token per request) is graph-capturable. A chunked prefill
appends on the host, one token at a time, then the open block is the
dead key. Draft pages have no pooled rows; the caller passes ids.

Indexer K gamma passed into the kernel is (1 + GemmaRMSNorm.weight). The
GGUF loader stores w after subtracting 1, and GemmaRMSNorm adds it back.
The kernel multiplies by gamma directly (qsa.cu, y * w_k_norm).

Main q and k come from Qwen3NextAttention._project_qkv_gate, already
normed and rotated. Indexer queries are normed and rotated here. Indexer
keys stay raw; indexer_append pools them.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from b70_qsa_xpu.batch import QsaBatch
from b70_qsa_xpu.layout import page_layout
from b70_qsa_xpu.serve import (
    as_pages,
    export_request_state,
    import_request_state,
    reset_open_rows,
)
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


def attach_qsa(layer, pooled: bool = True):
    """Return a runner, or None when this layer has no indexer weights."""
    attn = getattr(layer, "self_attn", None)
    indexer = getattr(attn, "indexer", None) if attn is not None else None
    if attn is None or indexer is None:
        return None
    return QsaRunner(attn, indexer, pooled=pooled)


def attach_draft(layer):
    """Draft runner: int8 KV, no pooled indexer rows. The caller supplies ids."""
    return attach_qsa(layer, pooled=False)


def mark_draft(layer):
    """Retarget a runner the decoder hook already attached.

    Call this from the MTP module before vLLM collects KV specs. The fork
    hook calls attach_qsa, which is the pooled target path.
    """
    runner = getattr(layer, "qsa_runner", None)
    if runner is None:
        return None
    runner.pooled = False
    runner._drop_batch()
    runner._install_spec()
    return runner


class QsaRunner:
    def __init__(self, attn, indexer, pooled: bool = True) -> None:
        self.attn = attn
        self.indexer = indexer
        self.pooled = bool(pooled)
        self.buf: QsaBuffers | None = None
        self.batch: QsaBatch | None = None
        self._ready = False
        self._logged = False
        self._cache_token = None
        self._draft_ids: torch.Tensor | None = None
        self._q_gamma: torch.Tensor | None = None
        self.side_tail = None
        self.side_dead = None
        self.side_block_pos = None
        self.side_length = None
        self._index = None
        self._sink = 0
        self._install_spec()

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

    def _log_once(self, text: str) -> None:
        if self._logged:
            return
        self._logged = True
        print(f"b70_qsa_xpu: {text}", file=sys.stderr)

    def _drop_batch(self) -> None:
        self.batch = None
        self._cache_token = None
        self.side_tail = None
        self.side_dead = None
        self.side_block_pos = None
        self.side_length = None
        self._index = None

    def _install_spec(self) -> None:
        """Patch this attention instance so the KV manager allocates one QSA page per block."""
        try:
            import b70_qsa_xpu.kv_spec as kv_spec
            from b70_qsa_xpu.kv_spec import make_qsa_spec

            kv_spec.register_specs()
        except Exception as exc:
            self._log_once(f"KV spec not installed ({type(exc).__name__}: {exc})")
            return
        runner = self

        def get_kv_cache_spec(vllm_config):
            g = runner._geom()
            return make_qsa_spec(
                n_kv=g.n_kv,
                head_dim=g.head_dim,
                idx_dim=g.idx_dim,
                idx_block=g.idx_block,
                block_size=int(vllm_config.cache_config.block_size),
                pooled=runner.pooled,
            )

        def get_attn_backend():
            return kv_spec.B70QsaBackend

        self.attn.get_kv_cache_spec = get_kv_cache_spec
        self.attn.get_attn_backend = get_attn_backend

    def _limits(self) -> tuple[int, int, int]:
        block_size = int(os.environ.get("B70_QSA_BLOCK", "0") or 0)
        max_seqs = int(os.environ.get("B70_QSA_MAX_SEQS", "0") or 0)
        max_len = int(os.environ.get("B70_QSA_MAX_LEN", "0") or 0)
        cfg = None
        try:
            from vllm.config import get_current_vllm_config_or_none

            cfg = get_current_vllm_config_or_none()
        except Exception:
            cfg = None
        if cfg is not None:
            if block_size <= 0:
                block_size = int(cfg.cache_config.block_size)
            if max_seqs <= 0:
                max_seqs = int(cfg.scheduler_config.max_num_seqs)
            if max_len <= 0:
                max_len = int(cfg.model_config.max_model_len)
        if block_size <= 0:
            block_size = 16
        if max_seqs <= 0:
            max_seqs = 8
        if max_len <= 0:
            max_len = int(os.environ.get("B70_QSA_MAX_CELLS", "8192"))
        max_blocks = max(1, (max_len + block_size - 1) // block_size)
        return block_size, max_seqs, max_blocks

    def _cache_id(self):
        cache = getattr(self.attn, "kv_cache", None)
        if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
            return None
        return (int(cache.data_ptr()), tuple(int(n) for n in cache.shape), str(cache.dtype))

    def _apply_gamma(self) -> None:
        k_gamma = self.indexer.k_layernorm.weight.detach().float() + 1
        if self.batch is not None:
            self.batch.k_norm.copy_(k_gamma)
        if self.buf is not None:
            self.buf.k_norm.copy_(k_gamma)
        self._q_gamma = (self.indexer.q_layernorm.weight.detach().float() + 1).contiguous()

    def _alloc_side(self) -> None:
        b = self.batch
        assert b is not None
        # One private sink slot per batch row. Inactive rows must not share
        # an index: index_copy_ is undefined when the index repeats.
        n = b.num_pages + b.batch
        r = b.geom.idx_block
        dev = b.device
        self.side_tail = torch.zeros(n, r - 1, b.geom.idx_dim, dtype=torch.float32, device=dev)
        self.side_dead = torch.zeros(n, b.geom.idx_dim, dtype=torch.float32, device=dev)
        self.side_block_pos = torch.zeros(n, dtype=torch.int32, device=dev)
        self.side_length = torch.zeros(n, dtype=torch.int32, device=dev)
        self._index = torch.zeros(b.batch, dtype=torch.int32, device=dev)
        self._row_sink = torch.arange(b.batch, device=dev, dtype=torch.int32) + b.num_pages
        self._sink = b.num_pages

    def _try_bind(self, device: torch.device) -> bool:
        g = self._geom()
        block_size, max_seqs, max_blocks = self._limits()
        lay = page_layout(
            g.n_kv,
            g.head_dim,
            g.idx_dim,
            g.idx_block,
            block_size,
            pooled=self.pooled,
        )
        pages = as_pages(getattr(self.attn, "kv_cache", None), lay["page_bytes"])
        if pages is None:
            self._log_once("kv cache is not a QSA uint8 page; using the private pool")
            return False
        want = torch.device(device)
        if pages.device.type != want.type:
            self._log_once("kv cache device does not match the activation")
            return False
        table = np.zeros((max_seqs, max_blocks), dtype=np.int32)
        self.batch = QsaBatch(
            g,
            table,
            block_size,
            pages.device,
            pooled=self.pooled,
            pages=pages,
        )
        self._alloc_side()
        self._apply_gamma()
        self._cache_token = self._cache_id()
        return True

    def _paged_live(self, device: torch.device) -> bool:
        token = self._cache_id()
        if token is None:
            return False
        if self.batch is not None and token == self._cache_token:
            return True
        if _capturing():
            raise RuntimeError("B70_QSA allocates on the first forward; warmup before capture")
        return self._try_bind(device)

    def _metadata(self):
        try:
            from vllm.forward_context import get_forward_context, is_forward_context_available
        except Exception:
            return None
        if not is_forward_context_available():
            return None
        md = get_forward_context().attn_metadata
        if isinstance(md, list):
            md = md[0] if md else None
        if isinstance(md, dict):
            return md.get(getattr(self.attn, "layer_name", ""))
        return md

    def _copy_table(self, metadata, rows: int) -> None:
        bt = metadata.block_table
        if int(bt.shape[0]) < rows:
            raise RuntimeError("B70_QSA block table has fewer rows than the batch")
        cols = int(bt.shape[1])
        if cols > self.batch.max_blocks:
            raise RuntimeError("B70_QSA block table is wider than the runner")
        self.batch.block_table[:rows, :cols].copy_(bt[:rows, :cols])
        if cols < self.batch.max_blocks:
            self.batch.block_table[:rows, cols:].zero_()

    def _prepare_side(self, rows: int) -> None:
        b = self.batch
        phys0 = b.block_table[:rows, 0]
        torch.where(
            b.pos[:rows] >= 0,
            phys0,
            self._row_sink[:rows],
            out=self._index[:rows],
        )
        index = self._index[:rows].to(torch.long)
        b.tail[:rows].copy_(self.side_tail.index_select(0, index))
        b.dead[:rows].copy_(self.side_dead.index_select(0, index))
        b.block_pos[:rows].copy_(self.side_block_pos.index_select(0, index))
        b.length[:rows].copy_(self.side_length.index_select(0, index))
        reset_open_rows(
            b.tail[:rows],
            b.dead[:rows],
            b.block_pos[:rows],
            b.length[:rows],
            b.pos[:rows],
        )

    def _store_side(self, rows: int) -> None:
        index = self._index[:rows].to(torch.long)
        b = self.batch
        self.side_tail.index_copy_(0, index, b.tail[:rows])
        self.side_dead.index_copy_(0, index, b.dead[:rows])
        self.side_block_pos.index_copy_(0, index, b.block_pos[:rows])
        self.side_length.index_copy_(0, index, b.length[:rows])

    def _rope_rows(self, x: torch.Tensor, positions: torch.Tensor, token0: int, n: int) -> None:
        half = int(self.batch.cos.shape[1])
        if half == 0 or n == 0:
            return
        if positions.ndim == 2:
            for j in range(n):
                thw = positions[:, token0 + j].to(torch.int32)
                pair = torch.arange(half, device=x.device)
                p = thw[pair % 3].to(torch.long)
                c = self.batch.cos[p, pair]
                s = self.batch.sin[p, pair]
                x1 = x[j, :, :half].clone()
                x2 = x[j, :, half : 2 * half].clone()
                x[j, :, :half] = x1 * c - x2 * s
                x[j, :, half : 2 * half] = x2 * c + x1 * s
            return
        pos = positions[token0 : token0 + n].to(torch.long)
        c = self.batch.cos[pos]
        s = self.batch.sin[pos]
        x1 = x[..., :half].clone()
        x2 = x[..., half : 2 * half].clone()
        x[..., :half] = x1 * c.unsqueeze(1) - x2 * s.unsqueeze(1)
        x[..., half : 2 * half] = x2 * c.unsqueeze(1) + x1 * s.unsqueeze(1)

    def _note_mrope(self, positions: torch.Tensor, n: int, row_base: int, token0: int) -> None:
        if positions.ndim != 2 or n == 0:
            return
        if _capturing():
            raise RuntimeError("B70_QSA image mRoPE decode is eager")
        b = self.batch
        if b.mrope.numel() == 0:
            b.mrope = torch.zeros(
                b.batch, b.max_cells, 3, dtype=torch.int32, device=b.device
            )
        for j in range(n):
            p = int(positions[0, token0 + j].item())
            b.mrope[row_base + j, p, :].copy_(positions[:, token0 + j].to(torch.int32))

    def _load_rows(
        self, q, k, v, gate, iq, raw, positions, n: int, row0: int = 0, token0: int = 0
    ) -> None:
        g = self.batch.geom
        sl = slice(token0, token0 + n)
        dst = slice(row0, row0 + n)
        self.batch.q[dst].copy_(q[sl].reshape(n, g.n_head, g.head_dim).float())
        self.batch.k[dst].copy_(k[sl].reshape(n, g.n_kv, g.head_dim).float())
        self.batch.v[dst].copy_(v[sl].reshape(n, g.n_kv, g.head_dim).float())
        gate_i = gate[sl].reshape(n, g.n_head, g.head_dim).float()
        self.batch.q_full[dst, :, : g.head_dim].copy_(self.batch.q[dst])
        self.batch.q_full[dst, :, g.head_dim :].copy_(gate_i)
        iq_i = iq[sl].reshape(n, g.idx_n_head, g.idx_dim).float()
        var = iq_i.pow(2).mean(dim=-1, keepdim=True)
        iq_i = iq_i * torch.rsqrt(var + g.eps) * self._q_gamma
        self._rope_rows(iq_i, positions, token0, n)
        self.batch.q_idx[dst].copy_(iq_i)
        raw_i = raw[sl].reshape(n, int(self.indexer.index_kv_heads), g.idx_dim)[:, 0, :]
        self.batch.raw[dst].copy_(raw_i.float())

    def _copy_draft_ids(self, rows: int) -> None:
        if self._draft_ids is None:
            raise RuntimeError("B70_QSA draft attends caller-supplied ids; call set_draft_ids")
        ids = self._draft_ids
        n = min(int(ids.shape[0]), rows)
        cols = min(int(ids.shape[1]), int(self.batch.ids.shape[1]))
        self.batch.ids[:n, :cols].copy_(ids[:n, :cols].to(dtype=torch.int32, device=self.batch.device))

    def set_draft_ids(self, ids: torch.Tensor) -> None:
        """Stable int32 [rows, width] buffer. Update it with copy_ between replays."""
        self._draft_ids = ids

    def _decode(self, q, k, v, gate, iq, raw, positions, meta, hidden):
        rows = int(hidden.shape[0])
        if rows > self.batch.batch:
            raise RuntimeError(
                f"B70_QSA batch {rows} exceeds {self.batch.batch}; set B70_QSA_MAX_SEQS"
            )
        if positions.ndim == 2 and _capturing():
            raise RuntimeError("B70_QSA image mRoPE decode is eager")
        self._copy_table(meta, rows)
        pos_src = positions if positions.ndim == 1 else positions[0]
        self.batch.pos[:rows].copy_(pos_src[:rows].to(torch.int32))
        sm = meta.slot_mapping
        if int(sm.shape[0]) < rows:
            raise RuntimeError("B70_QSA slot_mapping is short")
        self.batch.pos[:rows].masked_fill_(sm[:rows] < 0, -1)
        if rows < self.batch.batch:
            self.batch.pos[rows:].fill_(-1)
        self._load_rows(q, k, v, gate, iq, raw, positions, rows)
        self._note_mrope(positions, rows, 0, 0)
        if not self.pooled:
            self._copy_draft_ids(rows)
        self._prepare_side(rows)
        self.batch.run_decode()
        self._store_side(rows)
        y = self.batch.gated[:rows].reshape(rows, -1).to(dtype=hidden.dtype)
        out, _ = self.attn.o_proj(y)
        return out

    def _prefill(self, q, k, v, gate, iq, raw, positions, meta, hidden):
        qsl = meta.query_start_loc.detach().to("cpu").tolist()
        nreq = int(meta.num_reqs)
        if nreq + 1 > len(qsl):
            raise RuntimeError("B70_QSA query_start_loc is short")
        if nreq > self.batch.batch:
            raise RuntimeError("B70_QSA prefill has more requests than the runner")
        self._copy_table(meta, nreq)
        pos_src = positions if positions.ndim == 1 else positions[0]
        pos_cpu = pos_src.detach().to("cpu").tolist()
        t_total = int(hidden.shape[0])
        g = self.batch.geom
        y = torch.empty(t_total, g.n_head * g.head_dim, dtype=hidden.dtype, device=hidden.device)
        for s in range(nreq):
            t0 = int(qsl[s])
            t1 = int(qsl[s + 1])
            for t in range(t0, t1):
                self.batch.pos.fill_(-1)
                self.batch.pos[s] = int(pos_cpu[t])
                self._load_rows(q, k, v, gate, iq, raw, positions, 1, row0=s, token0=t)
                self._note_mrope(positions, 1, s, t)
                if not self.pooled:
                    self._copy_draft_ids(nreq)
                self._prepare_side(nreq)
                self.batch.run_decode()
                self._store_side(nreq)
                y[t].copy_(self.batch.gated[s].reshape(-1).to(dtype=hidden.dtype))
        out, _ = self.attn.o_proj(y)
        return out

    def _forward_paged(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.batch is None:
            raise RuntimeError("B70_QSA paged forward has no batch")
        meta = self._metadata()
        if meta is None or not hasattr(meta, "block_table"):
            raise RuntimeError("B70_QSA paged forward needs attention metadata")
        self._apply_gamma()
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
        if int(getattr(meta, "max_query_len", 1)) == 1:
            return self._decode(q, k, v, gate, iq, raw, positions, meta, hidden)
        return self._prefill(q, k, v, gate, iq, raw, positions, meta, hidden)

    def export_request(self, phys) -> dict:
        """Park one request. `phys` is physical pages in logical order."""
        if self.batch is None or self.side_tail is None:
            raise RuntimeError("B70_QSA has no paged state")
        return export_request_state(
            self.batch.pages,
            self.side_tail,
            self.side_dead,
            self.side_block_pos,
            self.side_length,
            np.asarray(phys),
            self.batch.block_size,
        )

    def import_request(self, dest_phys, blob: dict) -> None:
        """Restore into `dest_phys` (logical order). Block 0's physical id is the side key."""
        if self.batch is None or self.side_tail is None:
            raise RuntimeError("B70_QSA has no paged state")
        import_request_state(
            self.batch.pages,
            self.side_tail,
            self.side_dead,
            self.side_block_pos,
            self.side_length,
            np.asarray(dest_phys),
            blob,
        )

    def export_state(self, seq: int) -> dict:
        if self.batch is None:
            raise RuntimeError("B70_QSA has no paged state")
        phys = self.batch.block_table[int(seq)].detach().to("cpu").numpy()
        return self.export_request(phys)

    def import_state(self, seq: int, blob: dict) -> None:
        if self.batch is None:
            raise RuntimeError("B70_QSA has no paged state")
        dest = self.batch.block_table[int(seq)].detach().to("cpu").numpy()
        self.import_request(dest, blob)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not _tp_is_one():
            raise RuntimeError("B70_QSA is TP=1 only")
        if positions is None:
            raise RuntimeError("B70_QSA needs positions")
        if self._paged_live(hidden.device):
            return self._forward_paged(hidden, positions)
        return self._forward_private(hidden, positions)

    def _forward_private(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
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
