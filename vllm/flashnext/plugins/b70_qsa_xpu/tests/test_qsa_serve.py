"""CPU checks for the paged serve hook. Does not load the SYCL library."""

from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import patch

from b70_qsa_xpu.batch import QsaBatch
from b70_qsa_xpu.layout import page_layout
from b70_qsa_xpu.model_hook import QsaRunner, mark_draft
from b70_qsa_xpu.serve import as_pages, export_request_state, import_request_state, reset_open_rows
from b70_qsa_xpu.state import Geom


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    print(f"PASS {name} {detail}", flush=True)


class _Cfg:
    rope_parameters = {"rope_theta": 1.0e7, "partial_rotary_factor": 0.25}
    indexer_compress_ratio = 4
    indexer_budget = 4
    rms_norm_eps = 1.0e-6


class _Rotary:
    rotary_dim = 8


class _Indexer:
    index_n_heads = 2
    index_head_dim = 32
    index_kv_heads = 1


class _Attn:
    def __init__(self) -> None:
        self.config = _Cfg()
        self.rotary_emb = _Rotary()
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 64
        self.indexer = _Indexer()
        self.layer_name = "layers.0.self_attn"
        self.kv_cache = torch.tensor([])


class _Cache:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size


class _Vllm:
    def __init__(self, block_size: int) -> None:
        self.cache_config = _Cache(block_size)
        self.scheduler_config = SimpleNamespace(max_num_seqs=2)
        self.model_config = SimpleNamespace(max_model_len=24)


def _geom() -> Geom:
    return Geom(
        n_head=4,
        n_kv=2,
        head_dim=64,
        n_rot=8,
        idx_n_head=2,
        idx_dim=32,
        idx_block=4,
        idx_top_k=4,
        rope_base=1.0e7,
        eps=1.0e-6,
        page_size=4,
    )


def test_as_pages() -> None:
    page = 32
    view = torch.arange(3 * page, dtype=torch.uint8).reshape(3, 1, 1, page)
    pages = as_pages(view, page)
    _check("view pages", pages is not None and tuple(pages.shape) == (3, page))
    pages[1, 0] = 9
    _check("view alias", int(view[1, 0, 0, 0].item()) == 9)
    flat = torch.arange(2 * page, dtype=torch.uint8)
    _check("flat pages", tuple(as_pages(flat, page).shape) == (2, page))
    split = torch.zeros(4, 1, 1, page // 2, dtype=torch.uint8)
    _check("split refused", as_pages(split, page) is None)
    _check("dtype refused", as_pages(torch.zeros(page, dtype=torch.float32), page) is None)
    _check("empty refused", as_pages(torch.zeros(0, dtype=torch.uint8), page) is None)


def test_reset() -> None:
    tail = torch.ones(3, 3, 4)
    dead = torch.ones(3, 4)
    block_pos = torch.tensor([7, 8, 9], dtype=torch.int32)
    length = torch.tensor([4, 5, 6], dtype=torch.int32)
    pos = torch.tensor([5, 0, -1], dtype=torch.int32)
    reset_open_rows(tail, dead, block_pos, length, pos)
    _check("keep open", float(tail[0].sum()) == 12.0 and int(length[0]) == 4)
    _check("clear origin", float(tail[1].sum()) == 0.0 and int(block_pos[1]) == 0)
    _check("clear inactive", float(dead[2].sum()) == 0.0 and int(length[2]) == 0)


def test_external_pages_and_park() -> None:
    g = _geom()
    lay = page_layout(g.n_kv, g.head_dim, g.idx_dim, g.idx_block, 4, pooled=True)
    page = lay["page_bytes"]
    raw = torch.zeros(6, page, dtype=torch.uint8)
    table = np.zeros((2, 3), dtype=np.int32)
    table[0] = (1, 2, 0)
    table[1] = (4, 5, 0)
    batch = QsaBatch(g, table, 4, torch.device("cpu"), pages=raw)
    _check("external alias", batch.pages.data_ptr() == raw.data_ptr())
    raw[1, 7] = 11
    _check("page byte", int(batch.pages[1, 7].item()) == 11)

    attn = _Attn()
    runner = QsaRunner(attn, attn.indexer, pooled=True)
    runner.batch = batch
    runner._alloc_side()
    runner.side_tail[1, 0, 0] = 1.25
    runner.side_dead[1, 1] = 2.5
    runner.side_block_pos[1] = 12
    runner.side_length[1] = 4
    runner.batch.pos[:] = -1
    runner.batch.pos[0] = 5
    runner._prepare_side(2)
    _check("gather tail", float(runner.batch.tail[0, 0, 0].item()) == 1.25)
    _check("inactive not live", float(runner.batch.tail[1].abs().sum().item()) == 0.0)
    runner.batch.tail[0, 0, 0] = 3.5
    runner.batch.length[0] = 4
    runner._store_side(2)
    _check("scatter tail", float(runner.side_tail[1, 0, 0].item()) == 3.5)
    _check("neighbor tail", float(runner.side_tail[4].abs().sum().item()) == 0.0)

    # Position 0 clears the gathered row before the kernel, then the store
    # keeps that cleared row only on the active key.
    runner.side_tail[1].fill_(9)
    runner.batch.pos[0] = 0
    runner._prepare_side(2)
    _check("origin cleared", float(runner.batch.tail[0].abs().sum().item()) == 0.0)
    runner.side_length[1] = 4
    runner.batch.length[0] = 4
    runner.batch.tail[0, 1, 2] = 6
    runner._store_side(2)

    blob = runner.export_state(0)
    _check("export n", blob["n_kv"] == 4 and blob["pos_base"] == 0)
    _check("export page", int(blob["pages"][0, 7]) == 11)
    _check("export tail", float(blob["tail"][1, 2]) == 6.0)
    runner.side_tail[1].zero_()
    runner.side_length[1] = 4
    raw[1].zero_()
    raw[2].zero_()
    dest = np.array([3, 5, 0], dtype=np.int32)
    runner.import_request(dest, blob)
    _check("remap byte", int(raw[3, 7].item()) == 11)
    _check("old page clear", int(raw[1, 7].item()) == 0)
    _check("side moved", float(runner.side_tail[3, 1, 2].item()) == 6.0 and int(runner.side_length[3].item()) == 4)
    # Import writes the new block-0 key. The vacated key stays until that
    # page is reused as some request's block 0 and position 0 resets it.
    _check("vacated key stays", int(runner.side_length[1].item()) == 4)
    _check("neighbor page", int(raw[4].sum().item()) == 0)

    # Round trip through the free functions leaves the neighbor sequence.
    raw[4, 0] = 8
    runner.side_length[4] = 2
    other = export_request_state(
        raw, runner.side_tail, runner.side_dead, runner.side_block_pos, runner.side_length,
        np.array([4, 5], dtype=np.int32), 4,
    )
    import_request_state(
        raw, runner.side_tail, runner.side_dead, runner.side_block_pos, runner.side_length,
        np.array([4, 5], dtype=np.int32), other,
    )
    _check("neighbor kept", int(raw[4, 0].item()) == 8 and int(runner.side_length[4].item()) == 2)


def test_spec_patch() -> None:
    attn = _Attn()
    # Model the actual projection-wrapper/registered-Attention split.
    # The old flat fake let a serve keep FLASH_ATTN pages unnoticed.
    cache = type("RegisteredAttention", (), {})()
    cache.layer_name = "layers.3.self_attn.attn"
    cache.kv_cache = torch.tensor([])
    attn.attn = cache
    runner = QsaRunner(attn, attn.indexer, pooled=True)
    cfg = _Vllm(8)
    spec = cache.get_kv_cache_spec(cfg)
    with patch("vllm.config.get_current_vllm_config_or_none", return_value=None):
        _check("worker config outside context", runner._limits() == (8, 2, 16))
        cfg.cache_config.block_size = 16
        _check("late aligned geometry", runner._limits() == (16, 2, 8))
        cfg.cache_config.block_size = 8
    _check("patched page", spec.page_size_bytes == page_layout(2, 64, 32, 4, 8, pooled=True)["page_bytes"])
    _check("patched backend", cache.get_attn_backend().get_name() == "B70_QSA")
    _check("backend attribute", cache.attn_backend is cache.get_attn_backend())
    _check("wrapper not registered", not hasattr(attn, "get_kv_cache_spec"))
    cache.kv_cache = torch.zeros(2, 1, 1, spec.page_size_bytes, dtype=torch.uint8)
    _check("cache from inner layer", runner._cache_id()[0] == cache.kv_cache.data_ptr())
    runner._limits = lambda: (8, 2, 3)
    runner._apply_gamma = lambda: None
    runner._prepare_batch(torch.device("cpu"))
    scratch_ptr = runner.batch.k_f16.data_ptr()
    _check("bind inner pages", runner._try_bind(torch.device("cpu")))
    _check("bound alias", runner.batch.pages.data_ptr() == cache.kv_cache.data_ptr())
    _check("reuse profiled scratch", runner.batch.k_f16.data_ptr() == scratch_ptr)
    expected_meta = object()
    context = SimpleNamespace(attn_metadata={cache.layer_name: expected_meta})
    with patch("vllm.forward_context.is_forward_context_available", return_value=True), \
         patch("vllm.forward_context.get_forward_context", return_value=context):
        _check("metadata from inner name", runner._metadata() is expected_meta)
    positions = torch.tensor([[0, 2], [1, 3], [2, 4]], dtype=torch.int64)
    runner._note_mrope(positions, 2, 0, 0)
    _check("mrope row zero", runner.batch.mrope[0, 0].tolist() == [0, 1, 2])
    _check("mrope row one", runner.batch.mrope[1, 2].tolist() == [2, 3, 4])
    layer = type("L", (), {})()
    layer.qsa_runner = runner
    mark_draft(layer)
    draft = cache.get_kv_cache_spec(_Vllm(8))
    _check("draft page", draft.page_size_bytes == page_layout(2, 64, 32, 4, 8, pooled=False)["page_bytes"])
    _check("draft flag", runner.pooled is False)


def main() -> None:
    test_as_pages()
    test_reset()
    test_external_pages_and_park()
    test_spec_patch()
    print("QSA_SERVE_OK", flush=True)


if __name__ == "__main__":
    main()
