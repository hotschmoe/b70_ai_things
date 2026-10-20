"""Paged QSA batch. One uint8 page per vLLM block, graph-capturable decode.

Tensors stay allocated after the first call. `pos < 0` is an inactive
slot: the kernels write a zero step and return before their barriers.
`length` is the durable token count. The step row of an inactive slot
is cleared, and `length` is not.

Export keys match flashnext_ref.qsa_cache.QsaPaged.export_seq.
"""

from __future__ import annotations

import numpy as np
import torch

from b70_qsa_xpu import ops
from b70_qsa_xpu.layout import page_layout
from b70_qsa_xpu.state import Geom


class QsaBatch:
    def __init__(
        self,
        geom: Geom,
        block_table: np.ndarray,
        block_size: int,
        device: torch.device,
        *,
        pooled: bool = True,
        pos_base: np.ndarray | None = None,
        max_cells: int | None = None,
    ) -> None:
        table = np.asarray(block_table, dtype=np.int32)
        if table.ndim != 2:
            raise ValueError("block_table [batch, max_blocks]")
        self.geom = geom
        self.device = torch.device(device)
        self.block_size = int(block_size)
        self.pooled = bool(pooled)
        self.batch = int(table.shape[0])
        self.max_blocks = int(table.shape[1])
        self.layout = page_layout(
            geom.n_kv,
            geom.head_dim,
            geom.idx_dim,
            geom.idx_block,
            self.block_size,
            pooled=self.pooled,
        )
        flat = table.reshape(-1)
        self.num_pages = int(flat.max()) + 1
        dev = self.device
        self.pages = torch.zeros(self.num_pages, self.layout["page_bytes"], dtype=torch.uint8, device=dev)
        self.block_table = torch.from_numpy(table.copy()).to(device=dev)
        if max_cells is None:
            max_cells = self.max_blocks * self.block_size
        self.max_cells = int(max_cells)
        r = geom.idx_block
        self.pos = torch.full((self.batch,), -1, dtype=torch.int32, device=dev)
        self.step = torch.zeros(self.batch, 4, dtype=torch.int32, device=dev)
        self.length = torch.zeros(self.batch, dtype=torch.int32, device=dev)
        self._length_scratch = torch.zeros(self.batch, dtype=torch.int32, device=dev)
        self.tail = torch.zeros(self.batch, r - 1, geom.idx_dim, dtype=torch.float32, device=dev)
        self.dead = torch.zeros(self.batch, geom.idx_dim, dtype=torch.float32, device=dev)
        self.block_pos = torch.zeros(self.batch, dtype=torch.int32, device=dev)
        if pos_base is None:
            base = np.zeros((self.batch,), dtype=np.int32)
        else:
            base = np.asarray(pos_base, dtype=np.int32).reshape(-1)
        self.pos_base = torch.from_numpy(base.copy()).to(device=dev)
        half = geom.n_rot // 2
        span = self.max_cells + int(base.max()) + 1
        self.cos = torch.zeros(span, half, dtype=torch.float32, device=dev)
        self.sin = torch.zeros(span, half, dtype=torch.float32, device=dev)
        self.k_norm = torch.ones(geom.idx_dim, dtype=torch.float32, device=dev)
        self.mrope = torch.empty(0, dtype=torch.int32, device=dev)
        self.bias = torch.empty(0, dtype=torch.float32, device=dev)
        self.seq_slot = torch.arange(self.batch, dtype=torch.int32, device=dev)
        cap = geom.width_cap()
        self.scores = torch.zeros(self.batch, self.max_cells, dtype=torch.float32, device=dev)
        self.ids = torch.zeros(self.batch, cap, dtype=torch.int32, device=dev)
        scratch = (self.batch, cap, geom.n_kv, geom.head_dim)
        self.k_f16 = torch.zeros(scratch, dtype=torch.float16, device=dev)
        self.v_f16 = torch.zeros(scratch, dtype=torch.float16, device=dev)
        self.attn = torch.zeros(self.batch, geom.n_head, geom.head_dim, dtype=torch.float32, device=dev)
        self.gated = torch.zeros(self.batch, geom.n_head, geom.head_dim, dtype=torch.float16, device=dev)
        self.q = torch.zeros(self.batch, geom.n_head, geom.head_dim, dtype=torch.float32, device=dev)
        self.k = torch.zeros(self.batch, geom.n_kv, geom.head_dim, dtype=torch.float32, device=dev)
        self.v = torch.zeros(self.batch, geom.n_kv, geom.head_dim, dtype=torch.float32, device=dev)
        self.q_idx = torch.zeros(self.batch, geom.idx_n_head, geom.idx_dim, dtype=torch.float32, device=dev)
        self.raw = torch.zeros(self.batch, geom.idx_dim, dtype=torch.float32, device=dev)
        self.q_full = torch.zeros(self.batch, geom.n_head, geom.head_dim * 2, dtype=torch.float32, device=dev)
        self._fill_tables()

    def _fill_tables(self) -> None:
        half = int(self.cos.shape[1])
        if half == 0:
            return
        span = int(self.cos.shape[0])
        inv = 1.0 / (self.geom.rope_base ** (np.arange(half, dtype=np.float64) / half))
        pos = np.arange(span, dtype=np.float64)[:, None]
        ang = pos * inv
        self.cos.copy_(torch.from_numpy(np.cos(ang).astype(np.float32)))
        self.sin.copy_(torch.from_numpy(np.sin(ang).astype(np.float32)))

    def set_mrope(self, table: np.ndarray | None) -> None:
        if table is None:
            self.mrope = torch.empty(0, dtype=torch.int32, device=self.device)
            return
        tab = np.asarray(table, dtype=np.int32)
        self.mrope = torch.from_numpy(np.ascontiguousarray(tab)).to(device=self.device)

    def run_decode(self) -> None:
        """Append the cell in `pos` for every active sequence and attend.

        Query tensors are the batch itself: seq_slot[b] = b. Inactive
        rows (pos < 0) do not write pages or scratch.
        """
        g = self.geom
        ops.write_step_batch(self.pos, self.step, g.idx_top_k, g.idx_block)
        ops.kv_append_batch(
            self.k,
            self.v,
            self.pages,
            self.block_table,
            self.step,
            self.block_size,
            g.idx_block,
            g.idx_dim,
            1 if self.pooled else 0,
        )
        if self.pooled:
            ops.indexer_append_batch(
                self.raw,
                self.k_norm,
                self.tail,
                self.dead,
                self.block_pos,
                self.pages,
                self.block_table,
                self.cos,
                self.sin,
                self.step,
                self.mrope,
                self.pos_base,
                g.eps,
                self.block_size,
            )
            ops.index_scores_batch(
                self.q_idx,
                self.pages,
                self.dead,
                self.block_table,
                self.seq_slot,
                self.scores,
                self.step,
                self.bias,
                self.block_size,
                g.idx_block,
            )
            ops.topk_batch(self.scores, self.ids, self.step)
        ops.kv_gather_batch(
            self.ids,
            self.pages,
            self.block_table,
            self.seq_slot,
            self.k_f16,
            self.v_f16,
            self.step,
            self.block_size,
            g.idx_block,
            g.idx_dim,
            1 if self.pooled else 0,
        )
        ops.attend_batch(self.q, self.k_f16, self.v_f16, self.attn, self.step)
        ops.gate_batch(self.attn, self.q_full, self.gated)
        # Durable length. Inactive rows keep the count they already had.
        # where() writes a scratch buffer so the false branch is not aliased.
        torch.where(self.pos >= 0, self.step[:, 1], self.length, out=self._length_scratch)
        self.length.copy_(self._length_scratch)

    def score_queries(
        self,
        q_idx: torch.Tensor,
        pos: torch.Tensor,
        seq_slot: torch.Tensor,
        scores: torch.Tensor,
        ids: torch.Tensor,
        step: torch.Tensor,
    ) -> None:
        """Score many queries against pages that are already appended.

        Each query carries its own pos (and therefore its own n_kv). The
        open block reads `dead`, so a chunk scored after the append matches
        token-at-a-time.
        """
        g = self.geom
        ops.write_step_batch(pos, step, g.idx_top_k, g.idx_block)
        ops.index_scores_batch(
            q_idx,
            self.pages,
            self.dead,
            self.block_table,
            seq_slot,
            scores,
            step,
            self.bias,
            self.block_size,
            g.idx_block,
        )
        ops.topk_batch(scores, ids, step)

    def attend_ids(
        self,
        q: torch.Tensor,
        ids: torch.Tensor,
        seq_slot: torch.Tensor,
        step: torch.Tensor,
        k_f16: torch.Tensor,
        v_f16: torch.Tensor,
        attn: torch.Tensor,
    ) -> None:
        """Gather `ids` and attend. `step` supplies the width. Draft pages
        pass has_pooled 0; the KV region layout is the same."""
        g = self.geom
        ops.kv_gather_batch(
            ids,
            self.pages,
            self.block_table,
            seq_slot,
            k_f16,
            v_f16,
            step,
            self.block_size,
            g.idx_block,
            g.idx_dim,
            1 if self.pooled else 0,
        )
        ops.attend_batch(q, k_f16, v_f16, attn, step)

    def union(
        self,
        q: torch.Tensor,
        ids: torch.Tensor,
        step: torch.Tensor,
        seq_slot: torch.Tensor,
        ulist: torch.Tensor,
        ucount: torch.Tensor,
        attn: torch.Tensor,
    ) -> None:
        ops.union_build(ids, step, ulist, ucount)
        ops.union_attend(
            q,
            self.pages,
            self.block_table,
            seq_slot,
            ulist,
            ucount,
            attn,
            step,
            self.block_size,
            self.geom.idx_block,
            self.geom.idx_dim,
            1 if self.pooled else 0,
        )

    def export_seq(self, seq: int) -> dict:
        """CPU copy of one sequence. Pages are in logical-block order."""
        n_kv = int(self.length[seq].item())
        n_used = (n_kv + self.block_size - 1) // self.block_size if n_kv else 0
        logical = np.arange(n_used, dtype=np.int32)
        table = self.block_table[seq].detach().to("cpu").numpy()
        pages = np.zeros((n_used, self.layout["page_bytes"]), dtype=np.uint8)
        src = self.pages.detach().to("cpu").numpy()
        for i, block in enumerate(logical.tolist()):
            pages[i] = src[int(table[block])]
        pos = int(self.pos[seq].item()) if n_kv else -1
        return {
            "pos": pos if pos >= 0 else (n_kv - 1 if n_kv else -1),
            "n_kv": n_kv,
            "pos_base": int(self.pos_base[seq].item()),
            "tail": self.tail[seq].detach().to("cpu").numpy().copy(),
            "dead": self.dead[seq].detach().to("cpu").numpy().copy(),
            "block_pos": int(self.block_pos[seq].item()),
            "have_dead": bool(n_kv > 0),
            "block_table": table.copy(),
            "pages": pages,
            "logical_blocks": logical,
        }

    def zero_seq(self, seq: int) -> None:
        table = self.block_table[seq].detach().to("cpu").numpy()
        for phys in table.tolist():
            self.pages[int(phys)].zero_()
        self.tail[seq].zero_()
        self.dead[seq].zero_()
        self.block_pos[seq] = 0
        self.length[seq] = 0
        self.pos[seq] = -1
        self.step[seq].zero_()

    def import_seq(self, seq: int, blob: dict) -> None:
        """Write pages through the live block table, by logical block."""
        if int(blob["pos_base"]) != int(self.pos_base[seq].item()):
            raise ValueError("pos_base mismatch")
        logical = np.asarray(blob["logical_blocks"], dtype=np.int32).reshape(-1)
        pages = np.asarray(blob["pages"], dtype=np.uint8)
        table = self.block_table[seq]
        self.zero_seq(seq)
        for i, block in enumerate(logical.tolist()):
            phys = int(table[int(block)].item())
            self.pages[phys].copy_(torch.from_numpy(np.ascontiguousarray(pages[i])))
        self.tail[seq].copy_(torch.from_numpy(np.ascontiguousarray(blob["tail"])))
        self.dead[seq].copy_(torch.from_numpy(np.ascontiguousarray(blob["dead"])))
        self.block_pos[seq] = int(blob["block_pos"])
        self.length[seq] = int(blob["n_kv"])
        self.pos[seq] = int(blob["pos"])
