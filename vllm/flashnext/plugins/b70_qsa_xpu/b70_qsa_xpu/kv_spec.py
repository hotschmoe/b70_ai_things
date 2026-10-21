"""vLLM KV-cache spec for one QSA page.

The page is the uint8 layout in layout.py: int8 group-64 K and V codes,
fp16 scales, then completed pooled indexer rows. A draft page omits the
pooled tail. One stored state is the whole vLLM block, so a kernel-block
split would cut the pooled tail off the codes. B70QsaBackend reports the
manager block size as its only kernel block size, and the selector keeps
the page whole.

Registration does not load the SYCL library.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from b70_qsa_xpu.layout import page_layout


def _classes():
    from vllm.v1.attention.backend import AttentionBackend, MultipleOf
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
    from vllm.v1.kv_cache_spec_registry import (
        KVCacheSpecRegistry,
        register_kv_cache_spec,
    )

    return {
        "AttentionBackend": AttentionBackend,
        "MultipleOf": MultipleOf,
        "FullAttentionManager": FullAttentionManager,
        "FullAttentionSpec": FullAttentionSpec,
        "KVQuantMode": KVQuantMode,
        "KVCacheSpecRegistry": KVCacheSpecRegistry,
        "register_kv_cache_spec": register_kv_cache_spec,
    }


_BUILT = False
QsaInt8PooledSpec = None
QsaInt8DraftSpec = None
B70QsaBackend = None


def _build() -> None:
    global _BUILT, QsaInt8PooledSpec, QsaInt8DraftSpec, B70QsaBackend
    if _BUILT:
        return
    c = _classes()
    FullAttentionSpec = c["FullAttentionSpec"]
    AttentionBackend = c["AttentionBackend"]
    MultipleOf = c["MultipleOf"]
    register_kv_cache_spec = c["register_kv_cache_spec"]
    FullAttentionManager = c["FullAttentionManager"]

    @register_kv_cache_spec(
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    @dataclass(frozen=True, kw_only=True)
    class QsaInt8PooledSpec(FullAttentionSpec):
        """int8 QSA page, pooled indexer rows included."""

    @register_kv_cache_spec(
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    @dataclass(frozen=True, kw_only=True)
    class QsaInt8DraftSpec(FullAttentionSpec):
        """int8 QSA page with no pooled rows. The caller supplies ids."""

    from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder

    class B70QsaMetadata:
        """Block table and query spans for the plugin runner.

        The runner attends. This object only carries the tensors the
        forward already has, so a decode step can be graph-captured.
        """

        def __init__(
            self,
            num_actual_tokens: int,
            max_query_len: int,
            query_start_loc: torch.Tensor,
            max_seq_len: int,
            seq_lens: torch.Tensor,
            block_table: torch.Tensor,
            slot_mapping: torch.Tensor,
            num_reqs: int,
        ) -> None:
            self.num_actual_tokens = num_actual_tokens
            self.max_query_len = max_query_len
            self.query_start_loc = query_start_loc
            self.max_seq_len = max_seq_len
            self.seq_lens = seq_lens
            self.block_table = block_table
            self.slot_mapping = slot_mapping
            self.num_reqs = num_reqs

    class B70QsaMetadataBuilder(AttentionMetadataBuilder):
        # Decode graphs are one token per request. Chunked prefill stays eager.
        _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        supports_update_block_table = True

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            del common_prefix_len, fast_build
            meta = common_attn_metadata
            return B70QsaMetadata(
                num_actual_tokens=int(meta.num_actual_tokens),
                max_query_len=int(meta.max_query_len),
                query_start_loc=meta.query_start_loc,
                max_seq_len=int(meta.max_seq_len),
                seq_lens=meta.seq_lens,
                block_table=meta.block_table_tensor,
                slot_mapping=meta.slot_mapping,
                num_reqs=int(meta.num_reqs),
            )

        def update_block_table(self, metadata, blk_table, slot_mapping):
            metadata.block_table = blk_table
            metadata.slot_mapping = slot_mapping
            return metadata

    class B70QsaBackend(AttentionBackend):
        # Filled by the runner before vLLM aligns hybrid attention/GDN
        # pages. Alignment probes a synthetic one-token dense spec.
        indexer_geometry = (128, 4)

        @staticmethod
        def get_name() -> str:
            return "B70_QSA"

        @staticmethod
        def get_impl_cls():
            raise NotImplementedError("B70_QSA attends in the plugin runner")

        @staticmethod
        def get_builder_cls():
            return B70QsaMetadataBuilder

        @classmethod
        def customize_spec(cls, spec):
            if isinstance(spec, (QsaInt8PooledSpec, QsaInt8DraftSpec)):
                return spec
            idx_dim, idx_block = cls.indexer_geometry
            one_token_bytes = page_layout(
                spec.num_kv_heads, spec.head_size, idx_dim, idx_block,
                idx_block, pooled=True,
            )["page_bytes"] // idx_block
            # Hybrid sizing calls this with block_size=1, which is not a
            # complete QSA page. Report its linear byte slope; real pages
            # come from the registered layer's get_kv_cache_spec.
            return replace(
                spec, dtype=torch.uint8, num_head_slots=1,
                state_content_bytes=one_token_bytes, tokens_per_state=1,
                page_size_padded=None,
            )

        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            if kv_cache_spec is None:
                return [MultipleOf(4)]
            return [int(kv_cache_spec.block_size)]

    # vLLM passes specs across process boundaries. These lazily-created
    # classes must resolve by their exported names, not _build.<locals>.
    for cls in (QsaInt8PooledSpec, QsaInt8DraftSpec, B70QsaBackend):
        cls.__qualname__ = cls.__name__
    globals()["QsaInt8PooledSpec"] = QsaInt8PooledSpec
    globals()["QsaInt8DraftSpec"] = QsaInt8DraftSpec
    globals()["B70QsaBackend"] = B70QsaBackend
    _BUILT = True


def register_specs() -> None:
    """Register both page specs. Importing vLLM is deferred to this call."""
    _build()


def make_qsa_spec(
    *,
    n_kv: int,
    head_dim: int,
    idx_dim: int,
    idx_block: int,
    block_size: int,
    pooled: bool = True,
):
    """One page per block. page_size_bytes matches layout.page_layout."""
    _build()
    lay = page_layout(
        n_kv,
        head_dim,
        idx_dim,
        idx_block,
        block_size,
        pooled=pooled,
    )
    kv_quant_mode = _classes()["KVQuantMode"].INT8_PER_TOKEN_HEAD
    cls = QsaInt8PooledSpec if pooled else QsaInt8DraftSpec
    return cls(
        block_size=int(block_size),
        num_kv_heads=1,
        num_head_slots=1,
        head_size=1,
        dtype=torch.uint8,
        state_content_bytes=int(lay["page_bytes"]),
        tokens_per_state=int(block_size),
        kv_quant_mode=kv_quant_mode,
    )
