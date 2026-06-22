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


# W4A16 LINEAR KERNEL: the stock XPUwNa16 int4_gemm_w4a16 path produces garbage ("!!!!") on this
# compressed-tensors checkpoint (likely a weight/scale layout mismatch in its shared gptq-marlin transpose
# dance). Force our explicit dequant kernel (int4 -> bf16 at load, dense GEMM) ahead of it for XPU WNA16.
try:
    from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
    from vllm.platforms import PlatformEnum
    from xpu_wna16_dequant import XPUDequantWNA16LinearKernel

    _xpu_list = _POSSIBLE_KERNELS[PlatformEnum.XPU]
    if XPUDequantWNA16LinearKernel not in _xpu_list:
        _xpu_list.insert(0, XPUDequantWNA16LinearKernel)  # highest priority -> wins over XPUwNa16
    import sys
    print("[qwen35-text-shim] forced XPUDequantWNA16 ahead of XPUwNa16", file=sys.stderr)
except Exception as e:  # pragma: no cover
    import sys
    print(f"[qwen35-text-shim] dequant wiring skipped: {e}", file=sys.stderr)

# Best-effort: also run the Qwen3.5 ssm-dtype config hook for this arch. Not load-bearing.
try:
    from vllm.model_executor.models import config as _cfg

    _cfg.MODELS_CONFIG_MAP.setdefault(
        "Qwen3_5ForCausalLM", _cfg.Qwen3_5ForConditionalGenerationConfig
    )
except Exception:  # pragma: no cover - config internals may shift
    pass
