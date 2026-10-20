"""Keep gathered decode rows and captured forwards on the decode kernels."""


def is_long_prefill(rows):
    if rows <= 8:
        return False
    import torch

    if torch.xpu.is_initialized() and torch.xpu.is_current_stream_capturing():
        return False
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return True
    context = get_forward_context()
    if context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
        return False
    metadata = context.dp_metadata
    if metadata is not None:
        # This tensor is scheduler-owned CPU metadata. No device read or
        # synchronization is needed to distinguish DP=2 rows16 from8+8.
        return max(metadata.num_tokens_across_dp_cpu.tolist()) > 8
    return True
