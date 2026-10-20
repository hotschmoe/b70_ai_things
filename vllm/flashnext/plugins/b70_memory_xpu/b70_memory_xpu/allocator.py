"""Extend vLLM's scoped split limit to the XPU caching allocator.

The upstream worker limits splitting on CUDA/ROCm while profiling/loading,
but returns immediately on XPU. Large freed prefill buffers can then retain
an entire segment when a small persistent allocation splits it. This keeps
the original limit and restoration semantics via the device-neutral API.
"""
from contextlib import contextmanager


def install():
    import torch
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker._scoped_allocator_max_split
    get_settings = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    set_settings = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if get_settings is None or set_settings is None:
        print("b70_memory: allocator split API absent; preserving default", flush=True)
        return

    @contextmanager
    def scoped(self, max_split_size_mb):
        if not current_platform.is_xpu():
            with original(self, max_split_size_mb):
                yield
            return
        previous = get_settings()
        prefix = previous.strip().rstrip(",")
        settings = f"max_split_size_mb:{max_split_size_mb}"
        if prefix:
            settings = f"{prefix},{settings}"
        try:
            set_settings(settings)
            yield
        finally:
            set_settings(previous)

    Worker._scoped_allocator_max_split = scoped
    print("b70_memory: XPU scoped allocator split limit enabled", flush=True)
