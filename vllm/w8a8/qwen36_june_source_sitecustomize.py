"""Host-only completion fence for Steve's June vLLM source control.

The e190923b source already owns the accepted Quark INT8, GDN, PIECEWISE,
sampler, scheduler, and clone-safe collective paths. Do not duplicate those
paths here. This adapter only waits for the large profile-run clone before
oneCCL consumes it. The row threshold excludes every recorded decode graph.
"""

import inspect
import os
import sys

import torch

from vllm.distributed.device_communicators.xpu_communicator import (
    XpuCommunicator,
)


_MIN_ROWS = int(os.environ.get("B70_QWEN36_JUNE_PROFILE_FENCE_MIN_ROWS", "8192"))
_original_all_reduce = XpuCommunicator.all_reduce
_fence_calls = 0


def _profile_clone_complete_all_reduce(
    self: XpuCommunicator, input_: torch.Tensor
) -> torch.Tensor:
    global _fence_calls
    should_fence = (
        _MIN_ROWS > 0
        and input_.ndim > 0
        and input_.shape[0] >= _MIN_ROWS
        and os.environ.get("VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT", "0") == "1"
    )
    if should_fence:
        _fence_calls += 1
        torch.xpu.synchronize()
        print(
            "[qwen36-june-source] profile clone complete "
            f"call={_fence_calls} shape={tuple(input_.shape)} dtype={input_.dtype}",
            file=sys.stderr,
            flush=True,
        )
    return _original_all_reduce(self, input_)


XpuCommunicator.all_reduce = _profile_clone_complete_all_reduce

print(
    "[qwen36-june-source] source_stack=e190923b "
    f"communicator={inspect.getsourcefile(XpuCommunicator)} "
    f"profile_fence_min_rows={_MIN_ROWS}",
    file=sys.stderr,
    flush=True,
)
