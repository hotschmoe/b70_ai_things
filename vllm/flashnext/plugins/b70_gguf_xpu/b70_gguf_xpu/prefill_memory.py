"""Optional startup-profile allocator limit, restored before KV allocation."""
from contextlib import contextmanager
import os


@contextmanager
def profile_allocator():
    value = os.environ.get("B70_PREFILL_ALLOC_FRACTION")
    if value is None:
        yield
        return
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise ValueError("B70_PREFILL_ALLOC_FRACTION must be in (0,1]")
    import torch

    previous = torch.xpu.get_per_process_memory_fraction()
    torch.xpu.set_per_process_memory_fraction(fraction)
    print(f"X3_PROFILE_ALLOC fraction={fraction} previous={previous}", flush=True)
    try:
        yield
    finally:
        torch.xpu.set_per_process_memory_fraction(previous)
        print(f"X3_PROFILE_ALLOC restored={previous}", flush=True)
