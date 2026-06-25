# sitecustomize.py -- injected via PYTHONPATH so it runs at interpreter startup,
# INCLUDING inside the `bfcl` CLI subprocess. It registers ONE extra entry in BFCL's
# MODEL_CONFIG_MAPPING whose key == our vLLM served-model-id, so that:
#   * `bfcl --model <served-id>` resolves to a usable handler (no QWEN_API_KEY path), and
#   * the OpenAI `completions.create(model=<served-id>)` request matches what vLLM serves.
#
# We reuse BFCL's stock Qwen OSS handler (QwenFCHandler / QwenHandler). Those handlers:
#   * talk to an existing OpenAI-compatible endpoint via REMOTE_OPENAI_BASE_URL (set in run.sh),
#   * use the /v1/completions endpoint and build the Qwen3 prompt themselves, embedding tools as
#     <tools>...</tools> and parsing <tool_call>{...}</tool_call> XML out of the raw completion --
#     i.e. BFCL "prompt-FC" mode. This does NOT rely on vLLM's native --tool-call-parser; it works
#     against any plain completions endpoint, which is exactly what we have.
#
# The served-id to register is taken from env BFCL_REGISTER_MODEL (set by run.sh from $EVAL_SERVED).
# If unset, this module is a no-op (so `bfcl --help`, `bfcl models`, etc. still work standalone).
import os


def _register():
    served = os.getenv("BFCL_REGISTER_MODEL")
    if not served:
        return
    try:
        from bfcl_eval.constants import model_config as mc
        from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
        from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
    except Exception:  # pragma: no cover - import guard
        # bfcl_eval is only present in THIS harness's .venv. When PYTHONPATH leaks this module into
        # another interpreter (e.g. the system python3 that runs evallib.py), bfcl_eval is absent --
        # that is expected and harmless, so we silently no-op rather than spam stderr.
        return

    # FC (native tool-call XML, prompt-injected tools) is the default; set BFCL_REGISTER_FC=0
    # to fall back to the plain prompt handler (model is asked to emit [func(arg=...)] text).
    use_fc = os.getenv("BFCL_REGISTER_FC", "1") != "0"
    handler = QwenFCHandler if use_fc else QwenHandler

    entry = mc.ModelConfig(
        model_name=served,           # <-- sent as the OpenAI `model` field; MUST equal served id
        display_name=f"b70 {served}",
        url="local-vllm",
        org="b70",
        license="apache-2.0",
        model_handler=handler,
        input_price=None,            # open-source -> cost computed from latency, not token price
        output_price=None,
        is_fc_model=use_fc,
        underscore_to_dot=False,     # Qwen3 keeps dotted function names (matches stock Qwen3 FC entry)
    )
    mc.MODEL_CONFIG_MAPPING[served] = entry

    # Some code paths import the assembled map indirectly; keep both views consistent.
    try:
        mc.local_inference_model_map[served] = entry
    except Exception:
        pass


_register()
