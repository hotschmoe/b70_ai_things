# Paged GQA decode. Installed from sitecustomize when B70_D3_ATTN=1.
#
# FlashAttentionImpl.forward stays on FA for prefill, sinks, sliding
# window, and any head shape other than Ornith 16/2/256 fp16. Decode
# (including MTP T<=4, num_prefill_tokens==0) calls b70_gdn.paged_decode.
# B70_D3_ATTN_SPLITS is the fixed split count baked into the captured grid.

from __future__ import annotations

import os
import sys
import traceback

_LOGGED = False
_FALLBACK = set()


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _once(key: str, msg: str) -> None:
    if key in _FALLBACK:
        return
    _FALLBACK.add(key)
    _log(msg)


def install() -> None:
    global _LOGGED
    sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_gdn_xpu")
    import b70_gdn_xpu

    b70_gdn_xpu.register()
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    if getattr(FlashAttentionImpl, "_b70_d3_attn", False):
        return
    orig = FlashAttentionImpl.forward

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        try:
            if _want(self, query, kv_cache, attn_metadata, output_scale, output_block_scale):
                return _decode(
                    self, query, kv_cache, attn_metadata, output
                )
        except Exception:
            _once("exc", "b70_d3 attn fallback:\n" + traceback.format_exc())
        return orig(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    FlashAttentionImpl.forward = forward
    FlashAttentionImpl._b70_d3_attn = True
    if not _LOGGED:
        _LOGGED = True
        _log(
            "b70_d3 paged decode installed splits="
            + os.environ.get("B70_D3_ATTN_SPLITS", "1")
        )


def _want(self, query, kv_cache, attn_metadata, output_scale, output_block_scale) -> bool:
    if output_scale is not None or output_block_scale is not None:
        return False
    if attn_metadata is None or kv_cache is None:
        return False
    if getattr(self, "sinks", None) is not None:
        _once("sinks", "b70_d3 attn skip: sinks")
        return False
    window = getattr(self, "sliding_window", None)
    if window not in (None, (-1, -1)):
        _once("window", "b70_d3 attn skip: sliding window")
        return False
    if self.num_heads != 16 or self.num_kv_heads != 2 or self.head_size != 256:
        _once("geom", "b70_d3 attn skip: geometry")
        return False
    import torch

    if query.dtype != torch.float16 or kv_cache.dtype != torch.float16:
        _once("dtype", "b70_d3 attn skip: dtype")
        return False
    npre = int(getattr(attn_metadata, "num_prefill_tokens", 0) or 0)
    ndec = int(getattr(attn_metadata, "num_decode_tokens", 0) or 0)
    nact = int(getattr(attn_metadata, "num_actual_tokens", 0) or 0)
    if npre != 0 or ndec != nact or nact <= 0:
        _once("path", f"b70_d3 attn skip path pre={npre} dec={ndec} act={nact}")
        return False
    if int(getattr(attn_metadata, "max_query_len", 1) or 1) > 4:
        _once("qlen", "b70_d3 attn skip: max_query_len")
        return False
    causal = getattr(attn_metadata, "causal", True)
    if causal is not True:
        _once("causal", "b70_d3 attn skip: causal")
        return False
    return True


def _decode(self, query, kv_cache, attn_metadata, output):
    import torch

    n = int(attn_metadata.num_actual_tokens)
    q = query[:n]
    if q.dim() != 3:
        raise RuntimeError(f"query dim {q.dim()}")
    out = output[:n]
    if out.dim() == 2:
        out = out.view(n, self.num_heads, self.head_size)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
    splits = int(os.environ.get("B70_D3_ATTN_SPLITS", "1"))
    partial_o = partial_m = partial_l = None
    if splits > 1:
        partial_o = torch.empty(
            (splits, n, self.num_heads, self.head_size),
            dtype=torch.float32,
            device=q.device,
        )
        partial_m = torch.empty((splits, n, self.num_heads), dtype=torch.float32, device=q.device)
        partial_l = torch.empty_like(partial_m)
    torch.ops.b70_gdn.paged_decode(
        out,
        q,
        key_cache,
        value_cache,
        attn_metadata.block_table,
        attn_metadata.seq_lens,
        attn_metadata.query_start_loc,
        float(self.scale),
        splits,
        partial_o,
        partial_m,
        partial_l,
    )
    return output
