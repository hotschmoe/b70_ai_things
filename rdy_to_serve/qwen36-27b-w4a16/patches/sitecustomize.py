# sitecustomize.py -- register the TEXT-ONLY Qwen3_5ForCausalLM architecture.
#
# Why: this checkpoint is a language-model-only quant (architectures=["Qwen3_5ForCausalLM"], zero vision
# tensors). vLLM's registry only maps the VL "Qwen3_5ForConditionalGeneration", and its _normalize_arch
# suffix-maps our unregistered "...ForCausalLM" onto that VL class -> it builds a vision tower that has no
# weights and an odd-dim (4304) W4A16 MLP that asserts. Registering the EXACT text arch (which already
# exists as a real class, qwen3_5:Qwen3_5ForCausalLM) stops the normalization -> vLLM loads text-only,
# never builds the vision tower. The class is the same one the VL model uses as its .language_model, so
# the hybrid/GDN (gated-delta-net) cache path is preserved. Lazy string ref -> no heavy import here.
#
# Loaded via PYTHONPATH at interpreter startup (see ../serve.sh). Pinned to vLLM 0.23 (image :v0230).
try:
    from vllm.model_executor.models.registry import ModelRegistry

    # Register the EXACT text arch to our marker subclass (qwen35_text_hybrid, also on PYTHONPATH) which
    # adds is_hybrid=True. Lazy string ref -> the module is imported at arch-resolve time, not here.
    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM",
        "qwen35_text_hybrid:Qwen3_5ForCausalLM",
    )
    import sys
    print("[qwen35-text-shim] registered Qwen3_5ForCausalLM -> text-only hybrid class", file=sys.stderr)
except Exception as e:  # never block startup if the internals shift
    import sys
    print(f"[qwen35-text-shim] registration skipped: {e}", file=sys.stderr)
