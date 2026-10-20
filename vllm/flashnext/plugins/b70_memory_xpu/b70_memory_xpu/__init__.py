"""Small, opt-in GGUF memory fixes. Importing this module has no GPU effects."""
import os

_INSTALLED = False


def register():
    global _INSTALLED
    if _INSTALLED:
        return
    fix = os.environ.get("B70_MEMORY_FIX") == "1"
    report = os.environ.get("B70_MEMORY_REPORT")
    if not fix and not report:
        return
    _INSTALLED = True
    if fix:
        from .fixes import install
        install()
        import torch
        from vllm.platforms import current_platform
        if current_platform.is_xpu():
            # The installed vllm-xpu-kernels override reports a different
            # total/free pair from the native PyTorch allocator API.
            # Keep profiling and allocation on the same device-memory API.
            torch.accelerator.get_memory_info = torch.xpu.mem_get_info
            print("b70_memory: native PyTorch XPU memory info enabled", flush=True)
            from .profiling import install as install_profiling
            install_profiling()
    if report:
        from .observe import install
        install(report)
