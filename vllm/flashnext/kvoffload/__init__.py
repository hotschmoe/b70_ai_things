"""Hybrid KV offload: full-attention blocks plus GDN state in host RAM.

The copy engine is upstream OffloadingConnector (pinned host, XPU
swap_blocks_batch). This package owns the hybrid contract, the byte
budget, and the Ornith serve/test that prove a parked conversation
resumes without recomputing the prefix.
"""

from kvoffload.index import ParkIndex
from kvoffload.layout import (
    FlashNextTextLayout,
    OrnithTextLayout,
    checkpoint_boundary,
    scaled_attention_block_size,
)
from kvoffload.park import Park, resume_tokens

__all__ = [
    "FlashNextTextLayout",
    "OrnithTextLayout",
    "Park",
    "ParkIndex",
    "checkpoint_boundary",
    "resume_tokens",
    "scaled_attention_block_size",
]
