# qwen35_text_hybrid.py -- imported lazily when the 'Qwen3_5ForCausalLM' arch is resolved (via the
# sitecustomize registration). On PYTHONPATH (mounted dir); re-imported in every spawned vLLM worker.
#
# This checkpoint is a language-model-only Qwen3.5 (gated-delta-net hybrid) quant. The real text class
# qwen3_5:Qwen3_5ForCausalLM is the SAME LM the VL wrapper uses, BUT three pieces of machinery live only on
# the VL wrapper (Qwen3_5ForConditionalGeneration), so loading the text class standalone fails in stages:
#   1) is_hybrid defaults False -> the GDN/mamba KV-cache setup is skipped -> mamba_block_size assert;
#   2) the GDN cache alignment calls model_cls.get_mamba_state_shape_from_config -> AttributeError;
#   3) the shared (VL) config declares M-RoPE -> vLLM's per-request _init_mrope_positions asserts the model
#      supports_mrope. (The text decoder uses standard RoPE; for text-only input M-RoPE positions are just
#      sequential broadcast to [3, N], i.e. identical to 1D RoPE.)
# Fix: a marker subclass that (1) sets is_hybrid=True, (2) grafts the VL wrapper's GDN-state classmethods
# (they compute purely from vllm_config), and (3) sets supports_mrope=True + a text-only
# get_mrope_input_positions. Pinned to vLLM 0.23 (image :v0230).
import torch

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _Base,
    Qwen3_5ForConditionalGeneration as _VL,
)


class Qwen3_5ForCausalLM(_Base):
    is_hybrid = True
    supports_mrope = True

    # GDN (gated-delta-net) state shape/dtype/copy -- borrowed from the VL wrapper (compute from config only).
    get_mamba_state_dtype_from_config = classmethod(_VL.get_mamba_state_dtype_from_config.__func__)
    get_mamba_state_shape_from_config = classmethod(_VL.get_mamba_state_shape_from_config.__func__)
    get_mamba_state_copy_func = classmethod(_VL.get_mamba_state_copy_func.__func__)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # Text-only serve: no image/video tokens -> T=H=W=sequential positions (== standard 1D RoPE).
        n = len(input_tokens)
        positions = torch.arange(n, dtype=torch.long).unsqueeze(0).expand(3, n).contiguous()
        return positions, 0

    def load_weights(self, weights):
        # THE fix for garbage output: this checkpoint was quantized as the VL model's `.language_model`,
        # so every key is `model.language_model.<...>` (+ `lm_head.weight`). The standalone text class's
        # params are `model.<...>`, so without a remap AutoWeightsLoader routes `language_model.<...>` into
        # self.model and finds nothing -> ALL weights skip ("not found in params_dict") -> random init ->
        # "!!!!" garbage. Strip the infix so the keys match.
        remapped = (
            (name.replace("model.language_model.", "model."), w) for name, w in weights
        )
        return super().load_weights(remapped)


# NOTE: the stock XPUwNa16 int4_gemm_w4a16 kernel was VERIFIED CORRECT in isolation (op output matches a
# reference dequant on both synthetic and real checkpoint layers, maxerr ~0.016). So the linear kernel is
# NOT the garbage-output cause -- do not force the dequant fallback (it also OOMs at ~4x weight memory).

# Best-effort: also run the Qwen3.5 ssm-dtype config hook for this arch. Not load-bearing.
try:
    from vllm.model_executor.models import config as _cfg

    _cfg.MODELS_CONFIG_MAP.setdefault(
        "Qwen3_5ForCausalLM", _cfg.Qwen3_5ForConditionalGenerationConfig
    )
except Exception:  # pragma: no cover - config internals may shift
    pass
