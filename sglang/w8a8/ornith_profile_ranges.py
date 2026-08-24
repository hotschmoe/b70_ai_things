"""Default-off semantic profiler ranges for the Sglang Ornith runtime.

This module is mounted only by the Ornith research serve.  When
``B70_ORNITH_PROFILE_RANGES=1`` it wraps model boundaries with
``torch.profiler.record_function`` annotations.  The annotations let the
offline trace analyzer attribute XPU kernels by the CPU enqueue site without
adding a synchronization to the live path.
"""

from __future__ import annotations

import functools
import os

import torch


_INSTALLED = False


def _wrap_method(cls, method_name: str, label: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_b70_profile_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    wrapped._b70_profile_wrapped = True
    setattr(cls, method_name, wrapped)


def install() -> None:
    global _INSTALLED
    if _INSTALLED or os.environ.get("B70_ORNITH_PROFILE_RANGES") != "1":
        return

    import sglang.srt.models.qwen3_5 as model
    import sglang.srt.models.qwen3_5_mtp as mtp_model
    import sglang.srt.models.qwen2_moe as moe_model
    from sglang.srt.layers.moe.topk import TopK

    # Whole target/draft and decoder-layer boundaries.
    _wrap_method(
        model.Qwen3_5MoeForConditionalGeneration,
        "forward",
        "b70::target.forward",
    )
    _wrap_method(
        model.Qwen3_5MoeForCausalLM,
        "forward",
        "b70::target.transformer",
    )
    _wrap_method(
        mtp_model.Qwen3_5ForCausalLMMTP,
        "forward",
        "b70::mtp.forward",
    )
    _wrap_method(
        model.Qwen3_5LinearDecoderLayer,
        "forward",
        "b70::layer.gdn_total",
    )
    _wrap_method(
        model.Qwen3_5AttentionDecoderLayer,
        "forward",
        "b70::layer.full_attention_total",
    )

    # Gated DeltaNet: projection, recurrent core, normalization/gate, and output
    # projection are nested below the total range.  Int8DequantLinear supplies
    # the projection-specific ranges when a projection is quantized.
    _wrap_method(model.Qwen3_5GatedDeltaNet, "forward", "b70::gdn.total")
    _wrap_method(
        model.Qwen3_5GatedDeltaNet,
        "_forward_input_proj",
        "b70::gdn.input_projection",
    )
    _wrap_method(model.RadixLinearAttention, "forward", "b70::gdn.core")
    _wrap_method(model.RMSNormGated, "forward", "b70::gdn.output_norm_gate")

    # Full attention, including qkv projection, attention kernel, gate, and
    # output projection.  The sub-projections receive method-level ranges from
    # Int8DequantLinear when applicable.
    _wrap_method(
        model.Qwen3_5AttentionDecoderLayer,
        "self_attention",
        "b70::full_attention.total",
    )
    _wrap_method(model.RadixAttention, "forward", "b70::full_attention.core")

    # MoE semantic boundaries.  Int8MoEMethod.apply provides the narrower
    # routed-expert W8A8 range inside router_experts.
    _wrap_method(model.Qwen2MoeSparseMoeBlock, "forward", "b70::moe.total")
    _wrap_method(
        model.Qwen2MoeSparseMoeBlock,
        "_forward_shared_experts",
        "b70::moe.shared_expert",
    )
    _wrap_method(
        model.Qwen2MoeSparseMoeBlock,
        "_forward_router_experts",
        "b70::moe.router_experts",
    )
    # This name is imported directly into qwen3_5.py.  Wrapping the module
    # binding captures the post-expert TP reduction without touching global
    # distributed behavior elsewhere in Sglang.
    original_all_reduce = moe_model.tensor_model_parallel_all_reduce
    if not getattr(original_all_reduce, "_b70_profile_wrapped", False):

        @functools.wraps(original_all_reduce)
        def profiled_all_reduce(*args, **kwargs):
            with torch.profiler.record_function("b70::moe.tp_all_reduce"):
                return original_all_reduce(*args, **kwargs)

        profiled_all_reduce._b70_profile_wrapped = True
        moe_model.tensor_model_parallel_all_reduce = profiled_all_reduce

    _wrap_method(TopK, "forward_native", "b70::moe.topk")
    _wrap_method(TopK, "forward_cuda", "b70::moe.topk")

    _INSTALLED = True
    print("[ornith-profile] installed semantic record_function ranges", flush=True)
