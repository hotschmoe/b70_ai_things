"""Opt-in full embedding replication for Qwen3.5 target plus NEXTN draft.

The stock Qwen3.5 model hardcodes a TP-sharded VocabParallelEmbedding even
though SGLang's embedding layer supports a native unsharded layout.  For TP=2,
replicating the target table costs one additional shard per rank and removes
the target embedding all-reduce.  NEXTN then shares that exact module, which
also removes one embedding all-reduce from each draft step.

This patch is intentionally narrow:

* Qwen3_5ForCausalLM target models only (not is_nextn).
* TP=2, PP=1, BF16, no added-vocabulary padding.
* The LM head remains TP-sharded.
* The draft placeholder remains sharded during load, then is discarded when
  the already-loaded target embedding is shared.  This keeps initialization
  peak memory bounded and makes pool sizing conservative.

Install only when B70_XPU_REPLICATE_MTP_EMBED=1.
"""

from __future__ import annotations

import threading

import torch


_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_TARGET_EMBED_BY_PTR = {}


def _validate_full_embedding(module, config) -> None:
    expected_vocab = int(config.vocab_size)
    expected_hidden = int(config.hidden_size)
    expected_shape = (expected_vocab, expected_hidden)
    actual_shape = tuple(module.weight.shape)

    if module.tp_size != 1 or module.enable_tp:
        raise RuntimeError(
            "replicated Qwen3.5 embedding did not select the native TP=1 layout"
        )
    if module.num_embeddings != expected_vocab:
        raise RuntimeError(
            f"embedding vocab mismatch: {module.num_embeddings} != {expected_vocab}"
        )
    if module.org_vocab_size != expected_vocab:
        raise RuntimeError(
            f"embedding original vocab mismatch: {module.org_vocab_size} != {expected_vocab}"
        )
    if module.num_embeddings_padded != expected_vocab:
        raise RuntimeError(
            "this prototype requires a vocabulary with no added padding: "
            f"padded={module.num_embeddings_padded} vocab={expected_vocab}"
        )
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"replicated embedding shape mismatch: {actual_shape} != {expected_shape}"
        )
    if module.weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"replicated embedding requires BF16, got {module.weight.dtype}"
        )


def install() -> None:
    global _INSTALLED

    with _INSTALL_LOCK:
        if _INSTALLED:
            return

        from sglang.srt.distributed import get_pp_group, get_tp_group
        from sglang.srt.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )
        import sglang.srt.models.qwen3_5 as qwen35
        import sglang.srt.models.qwen3_5_mtp as qwen35_mtp

        original_model_init = qwen35.Qwen3_5ForCausalLM.__init__
        original_set_embed_and_head = qwen35_mtp.Qwen3_5ForCausalLMMTP.set_embed_and_head

        def replicated_target_init(
            self, config, quant_config=None, prefix="", is_nextn=False
        ):
            if is_nextn:
                return original_model_init(
                    self,
                    config,
                    quant_config=quant_config,
                    prefix=prefix,
                    is_nextn=is_nextn,
                )

            tp_group = get_tp_group()
            pp_group = get_pp_group()
            if tp_group.world_size != 2 or pp_group.world_size != 1:
                raise RuntimeError(
                    "B70 replicated MTP embedding requires TP=2 and PP=1, got "
                    f"TP={tp_group.world_size} PP={pp_group.world_size}"
                )

            created = []
            saved_embedding_class = qwen35.VocabParallelEmbedding

            def make_full_embedding(*args, **kwargs):
                kwargs["enable_tp"] = False
                module = VocabParallelEmbedding(*args, **kwargs)
                created.append(module)
                return module

            qwen35.VocabParallelEmbedding = make_full_embedding
            try:
                original_model_init(
                    self,
                    config,
                    quant_config=quant_config,
                    prefix=prefix,
                    is_nextn=is_nextn,
                )
            finally:
                qwen35.VocabParallelEmbedding = saved_embedding_class

            if len(created) != 1 or self.embed_tokens is not created[0]:
                raise RuntimeError(
                    "expected exactly one Qwen3.5 target input embedding constructor"
                )
            _validate_full_embedding(self.embed_tokens, config)
            ptr = self.embed_tokens.weight.data_ptr()
            _TARGET_EMBED_BY_PTR[ptr] = self.embed_tokens
            gib = self.embed_tokens.weight.numel() * self.embed_tokens.weight.element_size() / 2**30
            print(
                "[mtp-replicated-embed] target ENABLED: "
                f"rank={tp_group.rank_in_group} shape={tuple(self.embed_tokens.weight.shape)} "
                f"dtype={self.embed_tokens.weight.dtype} storage_gib={gib:.6f}",
                flush=True,
            )

        def share_replicated_embed(self, embed, head):
            if embed is None:
                raise RuntimeError("target embedding is missing on PP=1")
            target_module = _TARGET_EMBED_BY_PTR.get(embed.data_ptr())
            if target_module is None:
                raise RuntimeError(
                    "target embedding was not created by the replicated layout patch"
                )
            _validate_full_embedding(target_module, self.config)

            old_module = self.model.embed_tokens
            old_shape = tuple(old_module.weight.shape)
            if old_module.tp_size != 2 or old_module.enable_tp is not True:
                raise RuntimeError(
                    "expected a TP=2 sharded NEXTN placeholder before target sharing"
                )

            del old_module.weight
            self.model.embed_tokens = target_module
            if not self.config.tie_word_embeddings:
                del self.lm_head.weight
            self.lm_head.weight = head
            torch.xpu.empty_cache()
            torch.xpu.synchronize()

            tp_group = get_tp_group()
            shared = self.model.embed_tokens is target_module
            same_ptr = self.model.embed_tokens.weight.data_ptr() == embed.data_ptr()
            if not shared or not same_ptr or self.model.embed_tokens.tp_size != 1:
                raise RuntimeError("NEXTN failed to share the replicated target embedding")
            print(
                "[mtp-replicated-embed] draft SHARE OK: "
                f"rank={tp_group.rank_in_group} old_shape={old_shape} "
                f"full_shape={tuple(embed.shape)} same_ptr={same_ptr}",
                flush=True,
            )

        qwen35.Qwen3_5ForCausalLM.__init__ = replicated_target_init
        qwen35_mtp.Qwen3_5ForCausalLMMTP.set_embed_and_head = share_replicated_embed
        _INSTALLED = True
        print(
            "[mtp-replicated-embed] install OK: native full target table plus shared NEXTN module",
            flush=True,
        )
