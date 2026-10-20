# Runtime overlay imported because xpu-run.sh puts this directory on
# PYTHONPATH (site.py imports sitecustomize from sys.path).
# Add small XPU workarounds here. Keep this file import-safe: a failure
# here breaks every Python process in the container.
#
# B70_GGUF=1 loads the fork quant config, the b70_gguf model loader, and
# the b70_gguf_xpu ops. Unset, this file does not import vllm.

import os

# The loader check runs /usr/bin/python3.12, which has no torch. Skip
# there. site.py swallows sitecustomize errors, so a failed register in
# the real interpreter would also be silent; re-raise only when torch
# imported and registration itself failed.

# D1: eager-step timer. The finder does not import torch or vLLM.
if os.environ.get("B70_STEP_TIMER") == "1":
    try:
        from d1_step_timer import install as _d1_timer_install

        _d1_timer_install()
    except Exception:
        import traceback

        traceback.print_exc()

# D1: XPUGraph replay reclaim. Off unless B70_XPU_CG_RECLAIM>0.
if os.environ.get("B70_XPU_CG_RECLAIM", "0") not in ("", "0"):
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None
    if torch is not None:
        try:
            from d1_graph import install_reclaim

            install_reclaim()
        except Exception:
            import traceback

            traceback.print_exc()
            raise
if os.environ.get("B70_GGUF") == "1":
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None
    if torch is not None:
        try:
            from vllm.models.qwen4_exp.gguf import register

            register()
            import b70_gguf_xpu

            b70_gguf_xpu.register()
        except Exception:
            import traceback

            traceback.print_exc()
            raise
