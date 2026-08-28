"""Default-off vLLM 0.28 Qwen3.8 draft-only INT4 runtime overlay."""

from __future__ import annotations

import inspect
import os
import textwrap


def install_draft_lmhead_int4() -> None:
    import torch

    import vllm.v1.spec_decode.llm_base_proposer as proposer_module
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    gate = "B70_DRAFT_LMHEAD_INT4"
    original_share = proposer_module.SpecDecodeBaseProposer._maybe_share_lm_head
    source = textwrap.dedent(inspect.getsource(original_share))
    old = '''\
    else:
        # MTP model
        share_lm_head = True
        logger.info(
            "Detected MTP model. "
            "Sharing target model lm_head weights with the draft model."
        )
'''
    new = '''\
    else:
        # MTP model
        if __import__("os").environ.get("B70_DRAFT_LMHEAD_INT4") == "1":
            share_lm_head = False
            logger.info(
                "B70 draft LM-head INT4 is active. "
                "Keeping the target and draft lm_head weights isolated."
            )
        else:
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )
'''
    if source.count(old) != 1:
        raise RuntimeError(
            "vLLM 0.28 MTP LM-head sharing anchor changed; refusing overlay"
        )
    namespace = dict(proposer_module.__dict__)
    patched = source.replace(old, new, 1)
    exec(
        compile(patched, inspect.getsourcefile(original_share), "exec"),
        namespace,
    )
    proposer_module.SpecDecodeBaseProposer._maybe_share_lm_head = namespace[
        "_maybe_share_lm_head"
    ]

    class DraftLMHeadInt4Method:
        def __init__(self, packed_base, scales, zero, group_size, output_size):
            self.packed_base = packed_base
            self.qweight = packed_base.t()
            self.scales = scales
            self.zero = zero
            self.group_size = group_size
            self.output_size = output_size
            if self.qweight.stride(0) != 1:
                raise RuntimeError(
                    f"draft LM-head INT4 qweight is not NT: {self.qweight.stride()}"
                )

        def process_weights_after_loading(self, layer):
            del layer

        def apply(self, layer, x, bias=None):
            del layer
            if bias is not None:
                raise RuntimeError("draft LM-head INT4 does not support bias")
            original_shape = x.shape
            original_dtype = x.dtype
            if original_dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    f"unsupported draft LM-head activation dtype: {original_dtype}"
                )
            flat = x.reshape(-1, original_shape[-1]).to(torch.float16).contiguous()
            output = torch.ops._xpu_C.int4_gemm_w4a16(
                flat,
                self.qweight,
                None,
                self.scales,
                self.zero,
                self.group_size,
                None,
            )
            return output.to(original_dtype).reshape(
                *original_shape[:-1], self.output_size
            )

    @torch.no_grad()
    def build_draft_lmhead_int4(model) -> None:
        if os.environ.get(gate) != "1":
            return
        if getattr(model, "b70_draft_lmhead_int4", False):
            raise RuntimeError("draft LM-head INT4 was installed twice")
        if type(model) is not Qwen3_5MTP:
            raise RuntimeError(
                "draft LM-head INT4 is scoped to Qwen3_5MTP, got "
                f"{type(model).__module__}.{type(model).__name__}"
            )
        head = getattr(model, "lm_head", None)
        if not isinstance(head, ParallelLMHead):
            raise RuntimeError(
                f"draft LM-head INT4 requires ParallelLMHead, got {type(head).__name__}"
            )
        if not isinstance(head.quant_method, UnquantizedEmbeddingMethod):
            raise RuntimeError(
                "draft LM-head INT4 requires an unquantized checkpoint head"
            )
        if getattr(head, "bias", None) is not None:
            raise RuntimeError("draft LM-head INT4 does not support bias")
        weight = head.weight.detach()
        if weight.dtype != torch.float16 or weight.ndim != 2:
            raise RuntimeError(
                "draft LM-head INT4 expected a 2D FP16 weight, got "
                f"{weight.dtype} {tuple(weight.shape)}"
            )
        output_size, input_size = weight.shape
        group_size = 128
        if input_size != 5120 or input_size % group_size:
            raise RuntimeError(
                f"unsupported draft LM-head input size: {input_size}"
            )
        head_dtype = getattr(model.logits_processor, "head_dtype", None)
        if head_dtype is not None and head_dtype != weight.dtype:
            raise RuntimeError(
                "draft LM-head INT4 requires logits head dtype to match FP16"
            )

        groups = input_size // group_size
        packed_base = torch.empty(
            (output_size, input_size // 8),
            dtype=torch.int32,
            device=weight.device,
        )
        scales = torch.empty(
            (groups, output_size), dtype=torch.float16, device=weight.device
        )
        shifts = torch.tensor(
            (0, 4, 8, 12, 16, 20, 24, 28),
            dtype=torch.int32,
            device=weight.device,
        )
        chunk_rows = int(
            os.environ.get("B70_DRAFT_LMHEAD_INT4_CHUNK_ROWS", "1024")
        )
        if chunk_rows <= 0:
            raise RuntimeError("B70_DRAFT_LMHEAD_INT4_CHUNK_ROWS must be positive")
        for row0 in range(0, output_size, chunk_rows):
            row1 = min(row0 + chunk_rows, output_size)
            values = weight[row0:row1].float()
            grouped = values.view(row1 - row0, groups, group_size)
            row_scales = grouped.abs().amax(dim=-1).clamp_min_(1.0e-10) / 7.0
            quant = torch.round(grouped / row_scales.unsqueeze(-1)).clamp_(
                -8, 7
            ).to(torch.int32)
            stored = quant + 8
            packed = (
                stored.view(row1 - row0, groups, group_size // 8, 8)
                << shifts
            ).sum(dim=-1, dtype=torch.int32)
            packed_base[row0:row1].copy_(
                packed.reshape(row1 - row0, input_size // 8)
            )
            scales[:, row0:row1].copy_(row_scales.t().to(torch.float16))
            del values, grouped, row_scales, quant, stored, packed

        zero = torch.tensor((8,), dtype=torch.int8, device=weight.device)
        head.quant_method = DraftLMHeadInt4Method(
            packed_base, scales, zero, group_size, output_size
        )
        old_parameter = head.weight
        head.weight = torch.nn.Parameter(
            torch.empty(0, dtype=weight.dtype, device=weight.device),
            requires_grad=False,
        )
        del old_parameter, weight
        model.b70_draft_lmhead_int4 = True
        torch.xpu.empty_cache()
        print(
            "[b70-vllm028-draft-lmhead-int4] ready "
            f"shape=({output_size},{input_size}) group={group_size} "
            f"packed_bytes={packed_base.numel() * packed_base.element_size()} "
            f"scale_bytes={scales.numel() * scales.element_size()} "
            "draft_fp16_released=1 target_untouched=1",
            flush=True,
        )

    original_load_weights = Qwen3_5MTP.load_weights

    def load_weights_with_draft_lmhead_int4(self, weights):
        result = original_load_weights(self, weights)
        build_draft_lmhead_int4(self)
        return result

    Qwen3_5MTP.load_weights = load_weights_with_draft_lmhead_int4
    print(
        "[b70-vllm028-draft-lmhead-int4] installed load-time draft-only overlay",
        flush=True,
    )


if os.environ.get("B70_DRAFT_LMHEAD_INT4") == "1":
    install_draft_lmhead_int4()
