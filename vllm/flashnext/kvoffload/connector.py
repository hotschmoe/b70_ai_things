"""KV connector: pinned-host offload of full-attention KV and GDN state.

Subclasses upstream OffloadingConnector. The parent copies every
prefix-cacheable group with XPU ``swap_blocks_batch`` into pinned host
memory and loads it back on a prefix hit. This subclass refuses to
start unless the offload set contains both a full-attention group and
a GDN group, which is the hybrid contract.

``supports_divergent_local_hybrid_hits`` stays false. Turning it on
without a scheduler backfill asks the connector at the deep
full-attention boundary, gets 0 when the GDN state is the piece that
was evicted, and then drops the hit (vLLM issue 52773). The full
eviction path does not need that flag: the common boundary is the
shorter of the two groups, and the parent loads both groups from host
memory past that boundary. A scheduler hook that loads only the missing
GDN boundary is a fork request to C7, not a flag we flip here.
"""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from kvoffload.park import missing_kinds

logger = init_logger(__name__)


def _iter_layer_specs(spec: object) -> list[object]:
    nested = getattr(spec, "kv_cache_specs", None)
    if isinstance(nested, dict):
        return list(nested.values())
    return [spec]


def _kind(spec: object) -> str:
    name = type(spec).__name__
    if name == "MambaSpec":
        mtype = getattr(spec, "mamba_type", None)
        label = getattr(mtype, "name", str(mtype))
        if "GDN" in label:
            return "gdn"
        return "mamba"
    if name == "FullAttentionSpec":
        return "full_attention"
    return name


def offload_kinds(kv_cache_config: KVCacheConfig) -> list[str]:
    """Kinds of the groups the parent connector will actually offload."""
    kinds: list[str] = []
    for group_id in kv_cache_config.prefix_cacheable_group_ids:
        group = kv_cache_config.kv_cache_groups[group_id]
        for spec in _iter_layer_specs(group.kv_cache_spec):
            kind = _kind(spec)
            if kind not in kinds:
                kinds.append(kind)
    return kinds


class HybridCPUOffloadConnector(OffloadingConnector):
    """OffloadingConnector that requires full-attention KV and GDN state."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        kinds = offload_kinds(kv_cache_config)
        absent = missing_kinds(set(kinds))
        if absent:
            raise RuntimeError(
                "HybridCPUOffloadConnector needs full_attention and gdn "
                "groups in the prefix-cacheable offload set. "
                f"present={kinds} missing={list(absent)}. "
                "Serve with --enable-prefix-caching and "
                "--mamba-cache-mode align."
            )
        super().__init__(vllm_config, role, kv_cache_config)
        logger.info(
            "G1 hybrid offload groups: %s (role=%s). "
            "Divergent full-attention hits are not backfilled; "
            "a fully evicted prefix reloads both groups from host RAM.",
            ",".join(kinds),
            role.name,
        )
