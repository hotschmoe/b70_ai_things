"""Flash-Next geometry taken from the ISTA IQ3_XXS GGUF and Strata headers.

Sources:
- include/strata/kernels/ngram.hpp
- include/strata/kernels/qsa.hpp (qsa_real_shapes)
- include/strata/kernels/gr.hpp
- vllm/flashnext/reports/02_model_support.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ngram.cpp ple_artifact_consts(), transcribed. The compare script checks
# these against the GGUF header rather than trusting the transcription.
_MULT = (23703573157769, 20109073645365, 8052911324071)
_VOCAB = (
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
)
_OFFSET = (
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
)


@dataclass(frozen=True)
class PleConsts:
    mult: tuple[int, int, int]
    vocab: tuple[int, ...]
    offset: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.vocab) != 16 or len(self.offset) != 16:
            raise ValueError("PLE hash wants 16 heads")
        if len(self.mult) != 3:
            raise ValueError("PLE hash wants 3 multipliers")


def artifact_ple_consts() -> PleConsts:
    return PleConsts(mult=_MULT, vocab=_VOCAB, offset=_OFFSET)


@dataclass(frozen=True)
class FlashNextGeom:
    n_embd: int = 2560
    n_layer: int = 48
    vocab: int = 248320
    n_routed: int = 512
    top_k: int = 10
    expert_ffn: int = 640
    hc: int = 4
    hc_lr: int = 320
    rms_eps: float = 1e-6
    rope_dim: int = 64
    rope_base: float = 1e7
    # 11+11+10 = 32 = rope_dim/2. The trailing 0 is not a section.
    rope_sections: tuple[int, int, int, int] = (11, 11, 10, 0)
    n_q_heads: int = 24
    n_kv_heads: int = 2
    head_dim: int = 256
    full_every: int = 4
    idx_n_head: int = 4
    idx_dim: int = 128
    idx_block: int = 4
    idx_top_k: int = 2048
    ngram: int = 3
    heads_per_ngram: int = 8
    ple_head_dim: int = 160
    ple_conv_kernel: int = 4
    ple_eos: int = 248044
    ple_image_token: int = 248056
    token_null: int = -1
    # HF ple_layer_ids are 1-indexed. blk.1 -> [2]. GGUF ple.layers=[1]
    # is the 0-based block, not this id.
    ple_layer_ids_1based: tuple[int, ...] = (2,)
    bos: int = 248044
    eos: int = 248046
    ple_table_rows: int = 320001536
    ple_row_bytes_iq4nl: int = 90

    @property
    def hc_dim(self) -> int:
        return self.hc * self.n_embd

    @property
    def ple_n_heads(self) -> int:
        return (self.ngram - 1) * self.heads_per_ngram

    @property
    def ple_hist(self) -> int:
        return (self.ple_conv_kernel - 1) * self.ngram

    @property
    def qsa_width_extra(self) -> int:
        # min(n_kv, top_k + block - 1). The extra block-1 cells are the tail.
        return self.idx_top_k + self.idx_block - 1

    def is_full_layer(self, layer_idx0: int) -> bool:
        return (layer_idx0 + 1) % self.full_every == 0


FLASHNEXT = FlashNextGeom()

IQ4NL_CODES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)
