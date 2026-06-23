# Force ONLY the Qwen3.5 MTP drafter to instantiate unquantized/BF16, for the BF16 mtp.* graft.
# Without this, vLLM builds the Qwen3_5MultiTokenPredictor drafter through the target's compressed-tensors
# quant path and skips/garbles the grafted BF16 mtp.* linears -> 0% accept. The target body stays W8A8.
# (This is the corrected ASCII shim; the originally-grafted one had quote-stripping corruption.)
try:
    import vllm.model_executor.models.qwen3_5_mtp as mtp_mod
    _orig = mtp_mod.Qwen3_5MultiTokenPredictor.__init__
    def _patched_init(self, *, vllm_config, prefix=""):
        old_q = getattr(vllm_config, "quant_config", None)
        try:
            vllm_config.quant_config = None
            return _orig(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_config.quant_config = old_q
    mtp_mod.Qwen3_5MultiTokenPredictor.__init__ = _patched_init
    print("[mtp-bf16-shim] Qwen3_5MultiTokenPredictor forced unquantized for grafted BF16 mtp.*")
except Exception as e:
    print("[mtp-bf16-shim] patch failed:", repr(e))
