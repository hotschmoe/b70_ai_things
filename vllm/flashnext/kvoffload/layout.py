"""Byte layout for a Qwen3.5-MoE / Ornith hybrid cache.

Full-attention KV is stored per token. GDN state is one (conv, recurrent)
snapshot per linear layer at a block boundary, not per token. A restore
that has only one of those two pieces cannot resume: the recurrent state
is what makes the full-attention prefix valid.

XPU grows the attention block until one attention page is at least one
GDN page (platforms/interface.py). On this Ornith GPTQ fp16 config that
lands at 1088 tokens, matching the A3 serve log.
"""

from __future__ import annotations

from dataclasses import dataclass


DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "int8": 1,
    "uint8": 1,
}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def gdn_state_bytes(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel: int,
    conv_dtype: str,
    ssm_dtype: str,
) -> int:
    """Unpadded conv state plus recurrent state for one linear layer.

    Qwen GDN convolves q, k, and v. The conv width is 2 * key + value.
    """
    conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    conv_tokens = (conv_kernel - 1) * conv_dim
    conv_bytes = conv_tokens * DTYPE_BYTES[conv_dtype]
    ssm_bytes = num_v_heads * head_v_dim * head_k_dim * DTYPE_BYTES[ssm_dtype]
    return conv_bytes + ssm_bytes


def attn_bytes_per_token_per_layer(
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: str,
) -> int:
    return num_kv_heads * head_dim * 2 * DTYPE_BYTES[kv_dtype]


@dataclass(frozen=True)
class OrnithTextLayout:
    """qwen3_5_moe text tower from the Ornith GPTQ config.

    Defaults are the on-disk checkpoint
    models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2.
    """

    full_attention_layers: int = 10
    linear_attention_layers: int = 30
    num_kv_heads: int = 2
    head_dim: int = 256
    kv_dtype: str = "float16"
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel: int = 4
    conv_dtype: str = "float16"
    ssm_dtype: str = "float32"
    # XPU FLASH_ATTN / GDN kernel alignment used before the mamba-page scale.
    kernel_block_alignment: int = 64

    def attn_bytes_per_token_per_layer(self) -> int:
        return attn_bytes_per_token_per_layer(
            self.num_kv_heads, self.head_dim, self.kv_dtype
        )

    def attn_bytes_per_token(self) -> int:
        return self.full_attention_layers * self.attn_bytes_per_token_per_layer()

    def conv_dim(self) -> int:
        # q, k, and v are convolved. Qwen GDN uses 2 * key + value.
        return (
            self.head_k_dim * self.num_k_heads * 2
            + self.head_v_dim * self.num_v_heads
        )

    def gdn_state_bytes_per_layer(self) -> int:
        """Unpadded conv state plus recurrent state for one linear layer."""
        return gdn_state_bytes(
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel,
            self.conv_dtype,
            self.ssm_dtype,
        )

    def gdn_snapshot_bytes(self) -> int:
        """One conversation-boundary snapshot across every linear layer."""
        return self.linear_attention_layers * self.gdn_state_bytes_per_layer()

    def scaled_block_size(self) -> int:
        return scaled_attention_block_size(
            mamba_page_bytes=self.gdn_state_bytes_per_layer(),
            attn_bytes_per_token_per_layer=self.attn_bytes_per_token_per_layer(),
            kernel_block_alignment=self.kernel_block_alignment,
        )

    def padded_page_bytes(self, block_size: int) -> int:
        """Attention page at ``block_size``. Mamba pages are padded up to it."""
        return block_size * self.attn_bytes_per_token_per_layer()


def scaled_attention_block_size(
    mamba_page_bytes: int,
    attn_bytes_per_token_per_layer: int,
    kernel_block_alignment: int,
) -> int:
    """Smallest aligned block whose attention page covers one GDN page.

    Mirrors platforms/interface.py: the manager block is
    alignment * cdiv(mamba_page, alignment * attn_bytes_per_token).
    """
    if mamba_page_bytes <= 0 or attn_bytes_per_token_per_layer <= 0:
        raise ValueError("page sizes must be positive")
    if kernel_block_alignment <= 0:
        raise ValueError("alignment must be positive")
    stride = kernel_block_alignment * attn_bytes_per_token_per_layer
    return kernel_block_alignment * _cdiv(mamba_page_bytes, stride)


def checkpoint_boundary(num_tokens: int, block_size: int, drop_eagle: bool = False) -> int:
    """Reusable prefix length for align-mode Mamba without a partial tail.

    vLLM leaves the last prompt token uncached unless EAGLE already dropped
    a block (get_mamba_prefill_checkpoint_position). The boundary is then
    floored to a whole block, so an exact multiple still loses one block.
    """
    if num_tokens < 0 or block_size <= 0:
        raise ValueError("num_tokens and block_size must be non-negative / positive")
    if drop_eagle:
        proof = num_tokens
        boundary = (proof // block_size) * block_size - block_size
    else:
        proof = num_tokens - 1
        boundary = (proof // block_size) * block_size
    return max(boundary, 0)


def full_attn_bytes(layout: OrnithTextLayout, num_tokens: int) -> int:
    return num_tokens * layout.attn_bytes_per_token()


# Spec 5.4 rounds the int8 KV kernel (not built yet) to this many bytes
# per attention layer per token: 1024 B of int8 payload plus 16 B of scale.
INT8_SPEC_BYTES_PER_LAYER_TOKEN = 1040

# XPU PLE private tensor is _conv_state[slots, 9, 10240] bf16 (spec 2.7).
PLE_STATE_ROWS = 9
PLE_STATE_WIDTH = 10240


@dataclass(frozen=True)
class FlashNextTextLayout:
    """qwen4_exp text tower from flashnext_ref/qwen4exp_text_config.json.

    48 layers, full attention every 4th layer (12 full, 36 GDN). MTP is
    one extra full-attention KV stream. PLE is one short conv state.
    """

    full_attention_layers: int = 12
    linear_attention_layers: int = 36
    num_kv_heads: int = 2
    head_dim: int = 256
    kv_dtype: str = "bfloat16"
    num_k_heads: int = 16
    num_v_heads: int = 48
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel: int = 4
    conv_dtype: str = "bfloat16"
    ssm_dtype: str = "float32"
    mtp_layers: int = 1
    ple_rows: int = PLE_STATE_ROWS
    ple_width: int = PLE_STATE_WIDTH
    ple_dtype: str = "bfloat16"

    def attn_bytes_per_token_per_layer(self) -> int:
        return attn_bytes_per_token_per_layer(
            self.num_kv_heads, self.head_dim, self.kv_dtype
        )

    def attn_bytes_per_token(self) -> int:
        return self.full_attention_layers * self.attn_bytes_per_token_per_layer()

    def mtp_bytes_per_token(self) -> int:
        return self.mtp_layers * self.attn_bytes_per_token_per_layer()

    def gdn_state_bytes_per_layer(self) -> int:
        return gdn_state_bytes(
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel,
            self.conv_dtype,
            self.ssm_dtype,
        )

    def gdn_snapshot_bytes(self) -> int:
        return self.linear_attention_layers * self.gdn_state_bytes_per_layer()

    def ple_state_bytes(self) -> int:
        return self.ple_rows * self.ple_width * DTYPE_BYTES[self.ple_dtype]

    def conversation_bytes(self, num_tokens: int) -> int:
        """One parked conversation: full-attn KV, MTP KV, GDN end, PLE."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        per_token = self.attn_bytes_per_token() + self.mtp_bytes_per_token()
        return (
            num_tokens * per_token
            + self.gdn_snapshot_bytes()
            + self.ple_state_bytes()
        )

    def int8_conversation_bytes(self, num_tokens: int) -> int:
        """Spec 5.4 int8 column. GDN and PLE stay in their native dtypes."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        per_token = (
            self.full_attention_layers + self.mtp_layers
        ) * INT8_SPEC_BYTES_PER_LAYER_TOKEN
        return num_tokens * per_token + self.gdn_snapshot_bytes() + self.ple_state_bytes()

    def parked_with_one_checkpoint_each(
        self, conversations: int, num_tokens: int, *, int8: bool = False
    ) -> int:
        """4 parked conversations plus one aligned GDN checkpoint each."""
        if conversations < 0:
            raise ValueError("conversations must be non-negative")
        one = (
            self.int8_conversation_bytes(num_tokens)
            if int8
            else self.conversation_bytes(num_tokens)
        )
        return conversations * (one + self.gdn_snapshot_bytes())


def upstream_pool_bytes(
    layout: OrnithTextLayout,
    num_tokens: int,
    block_size: int,
    conversations: int,
) -> int:
    """Upper bound on pinned bytes if every block's GDN page is stored.

    Full-attention KV is per token. Each cached block may also hold a
    padded GDN page per linear layer. The Strata-shaped minimum is smaller
    (one GDN snapshot per conversation); the pool has to cover the
    upstream connector, which can store a page per block.
    """
    if conversations < 0:
        raise ValueError("conversations must be non-negative")
    boundary = checkpoint_boundary(num_tokens, block_size)
    blocks = boundary // block_size
    per_conv = full_attn_bytes(layout, boundary) + blocks * (
        layout.linear_attention_layers * layout.padded_page_bytes(block_size)
    )
    return conversations * per_conv
