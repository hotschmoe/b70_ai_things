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
