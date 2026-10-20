"""Paged QSA cache. One uint8 page per vLLM block.

Region order inside the page, matching the SYCL batch kernels:

    k codes   int8   n_kv * block_size * head_dim
    v codes   int8   same
    k scales  fp16   n_kv * block_size * groups
    v scales  fp16   same
    pooled    f32    n_states * idx_dim     (omitted when pooled=False)

groups = head_dim / 64. n_states = block_size / idx_block. A code row for
head h, slot s is (h * block_size + s) * head_dim bytes into its region.
A scale row is that index times groups, stored as fp16 bits.

Completed indexer rows live in the page. The spare key, the tail, and
block_pos stay beside the page: the next vLLM block may be unallocated,
so a completion writes pooled[b+1] = dead into the page only when that
next indexer state is inside the block already being written. Scoring
reads `dead` for the open block (qsa_select.hpp), not the spare alias.

Draft pages are the KV region only. The caller supplies selection ids.

mRoPE tables are indexed by cell, the same as QsaState._rot_thw. That
equals an absolute position only when pos_base is 0. The device kernel
indexes by absolute position; tests that set mRoPE keep pos_base at 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qsa import attend, indexer_scores
from .qsa_layer import (
    KV_Q8_GROUP,
    QsaGeom,
    _rms_double,
    apply_rope_table,
    dequant_q8,
    gate_fp16,
    quantize_q8,
    rope_tables,
    topk_radix,
)


@dataclass(frozen=True)
class PageLayout:
    block_size: int
    n_kv: int
    head_dim: int
    groups: int
    idx_dim: int
    idx_block: int
    n_states: int
    pooled: bool
    k_codes_off: int
    v_codes_off: int
    k_scale_off: int
    v_scale_off: int
    pooled_off: int
    kv_bytes: int
    pooled_bytes: int
    page_bytes: int

    def code_off(self, region: int, head: int, slot: int) -> int:
        return region + (head * self.block_size + slot) * self.head_dim

    def scale_off(self, region: int, head: int, slot: int) -> int:
        return region + (head * self.block_size + slot) * self.groups * 2

    def indexer_addr(self, block_b: int) -> tuple[int, int]:
        """Logical vLLM block and state row for indexer block `block_b`."""
        token0 = block_b * self.idx_block
        logical = token0 // self.block_size
        state = (token0 % self.block_size) // self.idx_block
        return logical, state


def make_layout(geom: QsaGeom, block_size: int, *, pooled: bool = True) -> PageLayout:
    """Byte layout shared by the numpy reference and the SYCL kernels."""
    if block_size < 1 or block_size % geom.idx_block:
        raise ValueError("block_size must be a positive multiple of idx_block")
    if geom.head_dim % KV_Q8_GROUP:
        raise ValueError("head_dim must be a multiple of 64")
    groups = geom.head_dim // KV_Q8_GROUP
    k_codes = geom.n_kv * block_size * geom.head_dim
    v_codes = k_codes
    k_scales = geom.n_kv * block_size * groups * 2
    v_scales = k_scales
    kv = k_codes + v_codes + k_scales + v_scales
    if kv != geom.kv_bytes_per_cell() * block_size:
        raise RuntimeError(f"kv region {kv} != per-cell {geom.kv_bytes_per_cell()} * {block_size}")
    if kv % 4:
        raise ValueError("kv region must be 4-byte aligned so pooled f32 starts aligned")
    n_states = block_size // geom.idx_block
    pooled_bytes = n_states * geom.idx_dim * 4 if pooled else 0
    return PageLayout(
        block_size=int(block_size),
        n_kv=int(geom.n_kv),
        head_dim=int(geom.head_dim),
        groups=int(groups),
        idx_dim=int(geom.idx_dim),
        idx_block=int(geom.idx_block),
        n_states=int(n_states),
        pooled=bool(pooled),
        k_codes_off=0,
        v_codes_off=k_codes,
        k_scale_off=k_codes + v_codes,
        v_scale_off=k_codes + v_codes + k_scales,
        pooled_off=kv,
        kv_bytes=int(kv),
        pooled_bytes=int(pooled_bytes),
        page_bytes=int(kv + pooled_bytes),
    )


def write_step_rows(pos: np.ndarray, top_k: int, block: int) -> np.ndarray:
    """pos [B] -> step [B, 4] = pos, n_kv, n_bid, width.

    pos < 0 is an inactive slot: the row is 0,0,0,0. That is the batch
    contract. The single-sequence kernel clamps a negative pos to 0.
    """
    src = np.asarray(pos, dtype=np.int32).reshape(-1)
    out = np.zeros((src.shape[0], 4), dtype=np.int32)
    cap = int(top_k + block - 1)
    for i, p in enumerate(src.tolist()):
        if p < 0:
            continue
        n_kv = p + 1
        out[i, 0] = p
        out[i, 1] = n_kv
        out[i, 2] = n_kv // int(block)
        out[i, 3] = n_kv if n_kv < cap else cap
    return out


def union_pack(ids: np.ndarray, widths: np.ndarray) -> tuple[np.ndarray, int]:
    """Sorted union of up to 8 id lists. Word is cell | (mask << 24).

    Bit q of mask is set when query q selected the cell. Cells must fit
    in 24 bits. Duplicate ids inside one query set the bit once.
    """
    table = np.asarray(ids, dtype=np.int32)
    if table.ndim != 2:
        raise ValueError("ids [queries, cap]")
    n_q = int(table.shape[0])
    if n_q < 1 or n_q > 8:
        raise ValueError("union tile is 1..8 queries")
    width = np.asarray(widths, dtype=np.int32).reshape(-1)
    if width.shape[0] != n_q:
        raise ValueError("widths")
    masks: dict[int, int] = {}
    for q in range(n_q):
        w = int(width[q])
        if w < 0 or w > table.shape[1]:
            raise ValueError("width")
        for cell in table[q, :w].tolist():
            cell = int(cell)
            if cell < 0 or cell >= (1 << 24):
                raise ValueError("cell does not fit in 24 bits")
            masks[cell] = masks.get(cell, 0) | (1 << q)
    ordered = sorted(masks)
    words = np.zeros((len(ordered),), dtype=np.uint32)
    for i, cell in enumerate(ordered):
        words[i] = np.uint32(cell) | (np.uint32(masks[cell]) << np.uint32(24))
    return words.view(np.int32), int(words.shape[0])


def union_cells(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split packed words into cells and uint8 masks, both ascending."""
    words = np.asarray(packed, dtype=np.int32).view(np.uint32)
    cells = (words & np.uint32(0x00FFFFFF)).astype(np.int32)
    masks = (words >> np.uint32(24)).astype(np.uint8)
    return cells, masks


def attend_masked(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    *,
    n_kv_heads: int | None = None,
) -> np.ndarray:
    """Attend the cells whose mask is set, in ascending order.

    An empty mask returns zeros. This is the numeric target of a union
    softmax that writes -inf on unselected cells: same ids, same order.
    """
    query = np.asarray(q, dtype=np.float32)
    key = np.asarray(k, dtype=np.float32)
    val = np.asarray(v, dtype=np.float32)
    live = np.asarray(mask, dtype=bool).reshape(-1)
    if live.shape[0] != key.shape[0]:
        raise ValueError("mask")
    if not np.any(live):
        return np.zeros((query.shape[0], query.shape[1]), dtype=np.float32)
    attn, _w = attend(query, key[live], val[live], n_kv_heads=n_kv_heads)
    return attn


class QsaPaged:
    """Batch of sequences over one page pool and one block table.

    block_table[b, logical] is the physical page. Pages do not overlap.
    pos < 0 marks a sequence that has not started. Appends are sequential:
    the next pos is the current n_kv. Construct with QsaPaged.create.
    """

    @staticmethod
    def create(
        geom: QsaGeom,
        block_table: np.ndarray,
        block_size: int,
        *,
        pooled: bool = True,
        pos_base: np.ndarray | None = None,
        mrope: np.ndarray | None = None,
    ) -> "QsaPaged":
        obj = object.__new__(QsaPaged)
        table = np.asarray(block_table, dtype=np.int32)
        if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
            raise ValueError("block_table [batch, max_blocks]")
        flat = table.reshape(-1)
        if np.any(flat < 0) or np.unique(flat).size != flat.size:
            raise ValueError("block_table pages must be unique and non-negative")
        obj.geom = geom
        obj.block_table = table.copy()
        obj.batch = int(table.shape[0])
        obj.max_blocks = int(table.shape[1])
        obj.layout = make_layout(geom, block_size, pooled=pooled)
        obj.num_pages = int(flat.max()) + 1
        obj.pages = np.zeros((obj.num_pages, obj.layout.page_bytes), dtype=np.uint8)
        r = geom.idx_block
        obj.tail = np.zeros((obj.batch, r - 1, geom.idx_dim), dtype=np.float32)
        obj.dead = np.zeros((obj.batch, geom.idx_dim), dtype=np.float32)
        obj.have_dead = np.zeros((obj.batch,), dtype=bool)
        obj.block_pos = np.zeros((obj.batch,), dtype=np.int32)
        obj.n_kv = np.zeros((obj.batch,), dtype=np.int32)
        obj.pos = np.full((obj.batch,), -1, dtype=np.int32)
        if pos_base is None:
            obj.pos_base = np.zeros((obj.batch,), dtype=np.int32)
        else:
            base = np.asarray(pos_base, dtype=np.int32).reshape(-1)
            if base.shape[0] != obj.batch:
                raise ValueError("pos_base")
            obj.pos_base = base.copy()
        span = obj.max_blocks * obj.layout.block_size + int(obj.pos_base.max()) + 1
        half = geom.n_rot // 2
        obj.cos_tab, obj.sin_tab = rope_tables(span, half, geom.rope_base)
        obj.mrope = None
        if mrope is not None:
            tab = np.asarray(mrope, dtype=np.int32)
            if tab.ndim != 3 or tab.shape[0] != obj.batch or tab.shape[2] != 3:
                raise ValueError("mrope [batch, cells, 3]")
            obj.mrope = tab
        return obj

    def _phys(self, seq: int, logical: int) -> int:
        if logical < 0 or logical >= self.max_blocks:
            raise IndexError(f"logical block {logical} outside 0..{self.max_blocks}")
        return int(self.block_table[seq, logical])

    def _thw(self, seq: int, cell: int) -> np.ndarray | None:
        if self.mrope is None:
            return None
        return self.mrope[seq, cell]

    def _write_bytes(self, phys: int, off: int, raw: np.ndarray) -> None:
        blob = np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
        self.pages[phys, off : off + blob.shape[0]] = blob

    def _read_bytes(self, phys: int, off: int, nbytes: int) -> np.ndarray:
        return np.array(self.pages[phys, off : off + nbytes], copy=True)

    def _read_pod(self, phys: int, off: int, dtype: np.dtype, count: int) -> np.ndarray:
        """Copy `count` elements. The page is uint8, so the view is built
        on an aligned destination rather than on the slice."""
        item = int(np.dtype(dtype).itemsize)
        raw = self.pages[phys, off : off + count * item]
        out = np.empty((count,), dtype=dtype)
        out.view(np.uint8)[:] = raw
        return out

    def write_kv_cell(self, seq: int, pos: int, k: np.ndarray, v: np.ndarray) -> None:
        """Quantize one cell into its page. K is already rotated."""
        g = self.geom
        lay = self.layout
        k = np.asarray(k, dtype=np.float32).reshape(g.n_kv, g.head_dim)
        v = np.asarray(v, dtype=np.float32).reshape(g.n_kv, g.head_dim)
        kc, ks = quantize_q8(k)
        vc, vs = quantize_q8(v)
        logical = pos // lay.block_size
        slot = pos % lay.block_size
        phys = self._phys(seq, logical)
        page = self.pages[phys]
        for head in range(g.n_kv):
            coff = lay.code_off(lay.k_codes_off, head, slot)
            page[coff : coff + g.head_dim] = kc[head].view(np.uint8)
            voff = lay.code_off(lay.v_codes_off, head, slot)
            page[voff : voff + g.head_dim] = vc[head].view(np.uint8)
            kbits = np.ascontiguousarray(ks[head].astype(np.float16)).view(np.uint16)
            vbits = np.ascontiguousarray(vs[head].astype(np.float16)).view(np.uint16)
            self._write_bytes(phys, lay.scale_off(lay.k_scale_off, head, slot), kbits)
            self._write_bytes(phys, lay.scale_off(lay.v_scale_off, head, slot), vbits)

    def read_kv_cell(self, seq: int, pos: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """codes int8 [n_kv, dim], scales f32 [n_kv, groups], for K then V."""
        g = self.geom
        lay = self.layout
        logical = pos // lay.block_size
        slot = pos % lay.block_size
        phys = self._phys(seq, logical)
        kc = np.zeros((g.n_kv, g.head_dim), dtype=np.int8)
        vc = np.zeros_like(kc)
        ks = np.zeros((g.n_kv, lay.groups), dtype=np.float32)
        vs = np.zeros_like(ks)
        for head in range(g.n_kv):
            coff = lay.code_off(lay.k_codes_off, head, slot)
            kc[head] = self.pages[phys, coff : coff + g.head_dim].view(np.int8)
            voff = lay.code_off(lay.v_codes_off, head, slot)
            vc[head] = self.pages[phys, voff : voff + g.head_dim].view(np.int8)
            kbits = self._read_pod(phys, lay.scale_off(lay.k_scale_off, head, slot), np.uint16, lay.groups)
            vbits = self._read_pod(phys, lay.scale_off(lay.v_scale_off, head, slot), np.uint16, lay.groups)
            ks[head] = kbits.view(np.float16).astype(np.float32)
            vs[head] = vbits.view(np.float16).astype(np.float32)
        return kc, ks, vc, vs

    def _write_pooled(self, seq: int, block_b: int, row: np.ndarray) -> None:
        lay = self.layout
        if not lay.pooled:
            raise RuntimeError("draft page has no pooled rows")
        logical, state = lay.indexer_addr(block_b)
        if state < 0 or state >= lay.n_states:
            raise RuntimeError("indexer state")
        phys = self._phys(seq, logical)
        vec = np.asarray(row, dtype=np.float32).reshape(lay.idx_dim)
        self._write_bytes(phys, lay.pooled_off + state * lay.idx_dim * 4, vec)

    def read_pooled(self, seq: int, block_b: int) -> np.ndarray:
        lay = self.layout
        logical, state = lay.indexer_addr(block_b)
        phys = self._phys(seq, logical)
        return self._read_pod(phys, lay.pooled_off + state * lay.idx_dim * 4, np.float32, lay.idx_dim)

    def append_kv(self, seq: int, k: np.ndarray, v: np.ndarray, pos: int) -> None:
        if int(pos) != int(self.n_kv[seq]):
            raise ValueError(f"seq {seq} expected pos {int(self.n_kv[seq])}, got {pos}")
        self.write_kv_cell(seq, int(pos), k, v)
        self.pos[seq] = np.int32(pos)
        self.n_kv[seq] = np.int32(pos + 1)

    def append_indexer(self, seq: int, raw: np.ndarray, k_norm: np.ndarray, pos: int) -> None:
        """Same sums as QsaState.append_indexer. Does not advance n_kv.

        Call it for the pos just stored by append_kv. The completing cell
        is not written into the tail. The spare row is written into the
        page only when it lands in the same vLLM block.
        """
        if not self.layout.pooled:
            raise RuntimeError("draft page has no indexer")
        g = self.geom
        lay = self.layout
        if int(pos) + 1 != int(self.n_kv[seq]):
            raise ValueError("indexer pos must be the cell just appended")
        raw64 = np.asarray(raw, dtype=np.float64).reshape(g.idx_dim)
        gamma = np.asarray(k_norm, dtype=np.float64).reshape(g.idx_dim)
        r = g.idx_block
        slot = int(pos) % r
        if slot < r - 1:
            self.tail[seq, slot] = raw64.astype(np.float32)
        if int(pos) == 0:
            y = _rms_double(raw64, gamma, g.eps)
            y = apply_rope_table(y, 0, self.cos_tab, self.sin_tab)
            self.dead[seq] = y
            self.have_dead[seq] = True
            # Token 0 allocates logical block 0, so the angle-0 spare may
            # sit in pooled row 0 until the first completion overwrites it.
            self._write_pooled(seq, 0, y)
        if slot == r - 1:
            acc = np.zeros((g.idx_dim,), dtype=np.float64)
            for j in range(r - 1):
                acc += self.tail[seq, j].astype(np.float64)
            acc += raw64
            mean = acc / np.float64(r)
            y = _rms_double(mean, gamma, g.eps)
            b = int(pos) // r
            cell0 = b * r
            y = apply_rope_table(
                y,
                int(self.pos_base[seq]) + cell0,
                self.cos_tab,
                self.sin_tab,
                thw=self._thw(seq, cell0),
            )
            self._write_pooled(seq, b, y)
            logical, _state = lay.indexer_addr(b)
            logical_s, _state_s = lay.indexer_addr(b + 1)
            if logical_s == logical:
                self._write_pooled(seq, b + 1, self.dead[seq])
            self.block_pos[seq] = np.int32(int(self.pos_base[seq]) + cell0)

    def append_token(
        self,
        seq: int,
        k: np.ndarray,
        v: np.ndarray,
        raw: np.ndarray | None,
        k_norm: np.ndarray | None,
        pos: int,
    ) -> None:
        self.append_kv(seq, k, v, pos)
        if self.layout.pooled:
            if raw is None or k_norm is None:
                raise ValueError("pooled append needs the raw indexer key")
            self.append_indexer(seq, raw, k_norm, pos)

    def scores(
        self,
        seq: int,
        q_idx: np.ndarray,
        n_kv: int,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        g = self.geom
        n_bid = int(n_kv) // g.idx_block
        if int(n_kv) > int(self.n_kv[seq]):
            raise ValueError("query sees past the appended prefix")
        if not self.have_dead[seq] and int(n_kv) > n_bid * g.idx_block:
            raise RuntimeError("tail score needs the dead key")
        view = np.zeros((n_bid + 1, g.idx_dim), dtype=np.float32)
        for b in range(n_bid):
            view[b] = self.read_pooled(seq, b)
        view[n_bid] = self.dead[seq]
        q = np.asarray(q_idx, dtype=np.float32).reshape(g.idx_n_head, g.idx_dim)
        return indexer_scores(view, q, int(n_kv), n_bid=n_bid, block=g.idx_block, bias=bias)

    def select(self, seq: int, q_idx: np.ndarray, n_kv: int) -> np.ndarray:
        width = self.geom.width_at(int(n_kv))
        return topk_radix(self.scores(seq, q_idx, n_kv), width)

    def gather(self, seq: int, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = self.geom
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        k = np.zeros((ids.shape[0], g.n_kv, g.head_dim), dtype=np.float32)
        v = np.zeros_like(k)
        for i, cell in enumerate(ids.tolist()):
            kc, ks, vc, vs = self.read_kv_cell(seq, int(cell))
            k[i] = dequant_q8(kc, ks)
            v[i] = dequant_q8(vc, vs)
        return k, v

    def attend_ids(self, seq: int, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
        k, v = self.gather(seq, ids)
        attn, _w = attend(q, k, v, n_kv_heads=self.geom.n_kv)
        return attn

    def decode_prepared(
        self,
        seq: int,
        k: np.ndarray,
        v: np.ndarray,
        raw: np.ndarray,
        k_norm: np.ndarray,
        q: np.ndarray,
        q_idx: np.ndarray,
        q_full: np.ndarray,
        pos: int,
    ) -> dict[str, np.ndarray]:
        self.append_token(seq, k, v, raw, k_norm, pos)
        ids = self.select(seq, q_idx, pos + 1)
        attn = self.attend_ids(seq, q, ids)
        gated = gate_fp16(attn, q_full)
        return {"ids": ids, "attn": attn, "gated": gated}

    def prefill_prepared(
        self,
        seq: int,
        tokens: list[dict[str, np.ndarray]],
        k_norm: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        """Append every token, then score each query at its own n_kv.

        Historical queries still match token-at-a-time because completed
        rows stay put and the open block reads `dead`.
        """
        for tok in tokens:
            self.append_token(seq, tok["k"], tok["v"], tok["raw"], k_norm, int(tok["pos"]))
        out = []
        for tok in tokens:
            n_kv = int(tok["pos"]) + 1
            ids = self.select(seq, tok["q_idx"], n_kv)
            attn = self.attend_ids(seq, tok["q"], ids)
            gated = gate_fp16(attn, tok["q_full"])
            out.append({"ids": ids, "attn": attn, "gated": gated, "q_idx": tok["q_idx"]})
        return out

    def used_logical(self, seq: int) -> np.ndarray:
        n_kv = int(self.n_kv[seq])
        n_used = (n_kv + self.layout.block_size - 1) // self.layout.block_size if n_kv else 0
        return np.arange(n_used, dtype=np.int32)

    def export_seq(self, seq: int) -> dict[str, np.ndarray | int | bool]:
        """Side state plus resident pages in logical-block order.

        `pages[i]` is the physical page of logical_blocks[i]. A later
        import writes those bytes through the destination block table,
        so a remap of physical ids still restores the sequence.
        """
        logical = self.used_logical(seq)
        pages = np.zeros((logical.shape[0], self.layout.page_bytes), dtype=np.uint8)
        for i, block in enumerate(logical.tolist()):
            pages[i] = self.pages[self._phys(seq, int(block))]
        return {
            "pos": int(self.pos[seq]),
            "n_kv": int(self.n_kv[seq]),
            "pos_base": int(self.pos_base[seq]),
            "tail": self.tail[seq].copy(),
            "dead": self.dead[seq].copy(),
            "block_pos": int(self.block_pos[seq]),
            "have_dead": bool(self.have_dead[seq]),
            "block_table": self.block_table[seq].copy(),
            "pages": pages,
            "logical_blocks": logical.copy(),
        }

    def zero_seq(self, seq: int) -> None:
        for logical in range(self.max_blocks):
            self.pages[self._phys(seq, logical)].fill(0)
        self.tail[seq].fill(0)
        self.dead[seq].fill(0)
        self.have_dead[seq] = False
        self.block_pos[seq] = np.int32(0)
        self.n_kv[seq] = np.int32(0)
        self.pos[seq] = np.int32(-1)

    def import_seq(self, seq: int, blob: dict) -> None:
        """Restore one sequence into the live block table.

        Pages are addressed by logical block, not by the exported physical
        ids. pos_base must already match; it is a property of the request.
        """
        if int(blob["pos_base"]) != int(self.pos_base[seq]):
            raise ValueError("pos_base mismatch")
        logical = np.asarray(blob["logical_blocks"], dtype=np.int32).reshape(-1)
        pages = np.asarray(blob["pages"], dtype=np.uint8)
        if pages.shape != (logical.shape[0], self.layout.page_bytes):
            raise ValueError("pages")
        self.zero_seq(seq)
        for i, block in enumerate(logical.tolist()):
            phys = self._phys(seq, int(block))
            self.pages[phys] = pages[i]
        self.tail[seq] = np.asarray(blob["tail"], dtype=np.float32).reshape(self.tail.shape[1:])
        self.dead[seq] = np.asarray(blob["dead"], dtype=np.float32).reshape(self.geom.idx_dim)
        self.block_pos[seq] = np.int32(blob["block_pos"])
        self.have_dead[seq] = bool(blob["have_dead"])
        self.n_kv[seq] = np.int32(blob["n_kv"])
        self.pos[seq] = np.int32(blob["pos"])
