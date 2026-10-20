"""Per-request QSA buffers. One layer, one sequence.

Layout matches flashnext_ref.qsa_layer.QsaState and Strata's
QsaIndexerBuffers plus the kv_q8 page: codes
[page, kv_head, page_size, head_dim] int8, scales
[page, kv_head, page_size, head_dim/64] f32 (the fp16 scale widened).

The first forward allocates. Replay of a captured decode must not be that
forward: every later token only writes these tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b70_qsa_xpu import ops


@dataclass
class Geom:
    n_head: int
    n_kv: int
    head_dim: int
    n_rot: int
    idx_n_head: int
    idx_dim: int
    idx_block: int
    idx_top_k: int
    rope_base: float = 1.0e7
    eps: float = 1.0e-6
    page_size: int = 4

    def width_cap(self) -> int:
        return int(self.idx_top_k + self.idx_block - 1)


class QsaBuffers:
    def __init__(self, max_cells: int, geom: Geom, device: torch.device, *, pos_base: int = 0) -> None:
        if max_cells < 1:
            raise ValueError("max_cells")
        if geom.head_dim % 64:
            raise ValueError("head_dim must be a multiple of 64")
        if geom.idx_block < 2:
            raise ValueError("idx_block")
        self.geom = geom
        self.device = torch.device(device)
        self.pos_base = int(pos_base)
        self.max_cells = int(max_cells)
        r = geom.idx_block
        self.max_blocks = max_cells // r + 1
        pages = (max_cells + geom.page_size - 1) // geom.page_size
        groups = geom.head_dim // 64
        half = geom.n_rot // 2
        span = max_cells + max(self.pos_base, 0) + 1
        dev = self.device
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)
        self.step = torch.zeros(4, dtype=torch.int32, device=dev)
        self.tail = torch.zeros(r - 1, geom.idx_dim, dtype=torch.float32, device=dev)
        self.dead = torch.zeros(geom.idx_dim, dtype=torch.float32, device=dev)
        self.pooled = torch.zeros(self.max_blocks, geom.idx_dim, dtype=torch.float32, device=dev)
        self.block_pos = torch.zeros(1, dtype=torch.int32, device=dev)
        code_shape = (pages, geom.n_kv, geom.page_size, geom.head_dim)
        self.k_codes = torch.zeros(code_shape, dtype=torch.int8, device=dev)
        self.v_codes = torch.zeros(code_shape, dtype=torch.int8, device=dev)
        scale_shape = (pages, geom.n_kv, geom.page_size, groups)
        self.k_scale = torch.zeros(scale_shape, dtype=torch.float32, device=dev)
        self.v_scale = torch.zeros(scale_shape, dtype=torch.float32, device=dev)
        self.page_table = torch.arange(pages, dtype=torch.int32, device=dev)
        self.cos = torch.zeros(span, half, dtype=torch.float32, device=dev)
        self.sin = torch.zeros(span, half, dtype=torch.float32, device=dev)
        self.k_norm = torch.ones(geom.idx_dim, dtype=torch.float32, device=dev)
        self.scores = torch.zeros(max_cells, dtype=torch.float32, device=dev)
        cap = geom.width_cap()
        self.ids = torch.zeros(cap, dtype=torch.int32, device=dev)
        scratch = (cap, geom.n_kv, geom.head_dim)
        self.k_f16 = torch.zeros(scratch, dtype=torch.float16, device=dev)
        self.v_f16 = torch.zeros(scratch, dtype=torch.float16, device=dev)
        self.attn = torch.zeros(geom.n_head, geom.head_dim, dtype=torch.float32, device=dev)
        self.gated = torch.zeros(geom.n_head, geom.head_dim, dtype=torch.float16, device=dev)
        self.mrope: torch.Tensor | None = None
        self._no_mrope = torch.empty(0, dtype=torch.int32, device=dev)
        self._no_bias = torch.empty(0, dtype=torch.float32, device=dev)

    def set_mrope(self, table: torch.Tensor | None) -> None:
        if table is None:
            self.mrope = None
            return
        if table.ndim != 2 or table.shape[1] != 3:
            raise ValueError("mrope is [cells, 3]")
        self.mrope = table.to(device=self.device, dtype=torch.int32).contiguous()

    def run(self, q, k, v, q_idx, raw, q_full) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append the cell at self.pos and attend. All tensors are preallocated."""
        g = self.geom
        mrope = self.mrope if self.mrope is not None else self._no_mrope
        ops.write_step(self.pos, self.step, g.idx_top_k, g.idx_block)
        ops.kv_append(
            k, v, self.k_codes, self.v_codes, self.k_scale, self.v_scale, self.page_table, self.step
        )
        ops.indexer_append(
            raw,
            self.k_norm,
            self.tail,
            self.dead,
            self.pooled,
            self.block_pos,
            self.cos,
            self.sin,
            self.step,
            mrope,
            self.pos_base,
            g.eps,
        )
        ops.index_scores(q_idx, self.pooled, self.dead, self.scores, self.step, self._no_bias, g.idx_block)
        ops.topk(self.scores, self.ids, self.step)
        ops.kv_gather(
            self.ids,
            self.k_codes,
            self.v_codes,
            self.k_scale,
            self.v_scale,
            self.page_table,
            self.k_f16,
            self.v_f16,
            self.step,
        )
        ops.attend(q, self.k_f16, self.v_f16, self.attn, self.step)
        ops.gate(self.attn, q_full, self.gated)
        return self.ids, self.attn, self.gated
