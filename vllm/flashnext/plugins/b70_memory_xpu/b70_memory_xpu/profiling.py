"""Count retained allocator segments once when calculating XPU headroom."""
from contextlib import contextmanager


def install():
    import torch
    import vllm.v1.worker.gpu_worker as worker
    from vllm.platforms import current_platform

    original = worker.memory_profiling

    @contextmanager
    def profile(*args, **kwargs):
        with original(*args, **kwargs) as result:
            yield result
        if current_platform.is_xpu():
            after = result.after_profile
            old = result.transient_peak_headroom
            # Driver consumption includes retained Torch reservations. Use
            # the physical reservation peak, not allocated-byte headroom:
            # fragmented inactive blocks cannot necessarily satisfy a later
            # large allocation. Non-Torch consumption remains accounted for.
            stats = torch.xpu.memory_stats(after.device_)
            reserved_peak = stats.get("reserved_bytes.all.peak")
            if reserved_peak is None:
                return
            result.reserved_peak_bytes = reserved_peak
            result.original_transient_peak_headroom = old
            result.transient_peak_headroom = max(0, reserved_peak - after.torch_memory)
            result.non_kv_cache_memory = (
                result.total_consumed + result.transient_peak_headroom
            )
            print(
                f"b70_memory: transient headroom {old} -> "
                f"{result.transient_peak_headroom}; retained Torch "
                f"{after.torch_memory}, live {after.torch_allocated}",
                flush=True,
            )

    worker.memory_profiling = profile
