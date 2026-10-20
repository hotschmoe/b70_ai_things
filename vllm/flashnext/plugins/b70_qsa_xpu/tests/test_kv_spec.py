"""CPU check: the QSA page is one vLLM block and is not kernel-split.

Run with the flashnext venv, from the plugin directory:
  PYTHONPATH=<plugin> python tests/test_kv_spec.py
"""

from __future__ import annotations

import torch

from b70_qsa_xpu import kv_spec
from b70_qsa_xpu.layout import page_layout
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheTensor,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.worker.utils import customize_attention_spec, select_common_block_size


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    print(f"PASS {name} {detail}", flush=True)


def main() -> None:
    kv_spec.register_specs()
    B70QsaBackend = kv_spec.B70QsaBackend
    make_qsa_spec = kv_spec.make_qsa_spec

    class _Fixed16(B70QsaBackend):
        @staticmethod
        def get_name() -> str:
            return "FIXED16"

        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return [16]

    flash = dict(n_kv=2, head_dim=256, idx_dim=128, idx_block=4, block_size=16)
    lay = page_layout(**flash, pooled=True)
    spec = make_qsa_spec(**flash, pooled=True)
    _check("pooled page", spec.page_size_bytes == lay["page_bytes"] == 18944, str(spec.page_size_bytes))
    _check("pooled class", type(spec).__name__ == "QsaInt8PooledSpec")
    _check("one state", spec.num_states == 1 and spec.num_heads == 1)
    _check("dtype", spec.dtype == torch.uint8)
    draft_lay = page_layout(**flash, pooled=False)
    draft = make_qsa_spec(**flash, pooled=False)
    _check(
        "draft page",
        draft.page_size_bytes == draft_lay["page_bytes"] == 1056 * 16,
        str(draft.page_size_bytes),
    )
    _check("draft class", type(draft).__name__ == "QsaInt8DraftSpec")
    for block in (4, 8, 16):
        small = make_qsa_spec(
            n_kv=2, head_dim=64, idx_dim=32, idx_block=4, block_size=block, pooled=True
        )
        want = page_layout(2, 64, 32, 4, block, pooled=True)["page_bytes"]
        _check(f"small block {block}", small.page_size_bytes == want, str(want))

    _check(
        "manager",
        KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager,
    )
    _check(
        "uniform base",
        KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is FullAttentionSpec,
    )
    merged = type(spec).merge([spec, spec])
    _check("merge page", merged.page_size_bytes == spec.page_size_bytes)

    tuned = customize_attention_spec(B70QsaBackend, spec)
    chosen = select_common_block_size(spec.block_size, [B70QsaBackend], [tuned])
    _check("no split 16", chosen == 16, str(chosen))
    wide = make_qsa_spec(**{**flash, "block_size": 64}, pooled=True)
    wide = customize_attention_spec(B70QsaBackend, wide)
    chosen64 = select_common_block_size(64, [B70QsaBackend], [wide])
    _check("no split 64", chosen64 == 64, str(chosen64))
    split = select_common_block_size(64, [_Fixed16], [wide])
    _check("fixed 16 would split", split == 16, str(split))
    _check(
        "backend name",
        B70QsaBackend.get_name() == "B70_QSA",
    )
    empty = B70QsaBackend.get_supported_kernel_block_sizes(None)
    _check("empty spec", len(empty) == 1 and isinstance(empty[0], MultipleOf) and empty[0].base == 4)

    num_blocks = 3
    num_layers = 2
    page = spec.page_size_bytes
    raw = torch.zeros(num_layers * num_blocks * page, dtype=torch.int8)
    tensor = KVCacheTensor(
        size=raw.numel(),
        layers=["layer.0", "layer.1"],
        layer_stride=num_blocks * page,
        block_stride=page,
    )
    views = create_kv_cache_views(
        raw,
        spec,
        num_blocks,
        KVCacheLayout.LBHNC,
        tensor,
        kernel_block_size=chosen,
    )
    _check("view count", len(views) == 2)
    _check("view shape", tuple(views[0].shape) == (num_blocks, 1, 1, page), str(tuple(views[0].shape)))
    _check("view dtype", views[0].dtype == torch.uint8)
    views[0][1, 0, 0, 0] = 7
    views[1][0, 0, 0, page - 1] = 9
    _check("block byte", int(raw[page].item()) == 7, str(int(raw[page].item())))
    layer1 = num_blocks * page
    _check("layer byte", int(raw[layer1 + page - 1].item()) == 9)

    from vllm.v1.attention.backend import AttentionCGSupport

    builder_cls = B70QsaBackend.get_builder_cls()
    support = builder_cls.get_cudagraph_support(None, spec)
    _check(
        "decode graph",
        support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE,
        str(support),
    )
    try:
        B70QsaBackend.get_impl_cls()
        _check("impl stays in the runner", False)
    except NotImplementedError:
        _check("impl stays in the runner", True)

    class _Meta:
        num_actual_tokens = 2
        max_query_len = 1
        num_reqs = 2
        max_seq_len = 4
        query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
        seq_lens = torch.tensor([3, 4], dtype=torch.int32)
        block_table_tensor = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
        slot_mapping = torch.tensor([0, 5], dtype=torch.int32)

    builder = builder_cls(spec, ["layers.0"], None, torch.device("cpu"))
    built = builder.build(0, _Meta())
    _check("built table", torch.equal(built.block_table, _Meta.block_table_tensor))
    new_bt = torch.tensor([[4, 5], [6, 7]], dtype=torch.int32)
    new_sm = torch.tensor([8, 9], dtype=torch.int32)
    built = builder.update_block_table(built, new_bt, new_sm)
    _check("updated table", torch.equal(built.block_table, new_bt))
    _check("updated slots", torch.equal(built.slot_mapping, new_sm))
    print("QSA_SPEC_OK", flush=True)


if __name__ == "__main__":
    main()
