"""Narrow runtime interventions for Steve's June vLLM source control.

The e190923b source already owns the accepted Quark INT8, GDN, PIECEWISE,
sampler, scheduler, and clone-safe collective paths. Do not duplicate those
paths here. The default adapter only waits for the large profile-run clone
before oneCCL consumes it. The row threshold excludes every recorded decode
graph. Experimental interventions are opt-in and print an explicit marker.
"""

import inspect
import os
import sys
from contextlib import contextmanager

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


if os.environ.get("B70_QWEN36_INT8_MOE_TRITON_INTERVENTION", "0") == "1":
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    _original_triton_supports_quant_scheme = TritonExperts._supports_quant_scheme

    @staticmethod
    def _triton_supports_xpu_int8(weight_key, activation_key):
        if (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        ):
            return True
        return _original_triton_supports_quant_scheme(weight_key, activation_key)

    TritonExperts._supports_quant_scheme = _triton_supports_xpu_int8
    print(
        "[qwen36-june-source] intervention=triton-int8-xpu-support-gate",
        file=sys.stderr,
        flush=True,
    )


if os.environ.get("B70_QWEN36_WORKER_ONLY_NUMA_BIND", "0") == "1":
    from vllm.utils import numa_utils

    _original_configure_subprocess = numa_utils.configure_subprocess

    @contextmanager
    def _worker_only_configure_subprocess(
        vllm_config,
        local_rank,
        dp_local_rank=None,
        process_kind="worker",
    ):
        # e190 binds EngineCore with GPU index 0 before that process spawns
        # both TP workers. Linux forbids worker 1 from expanding beyond the
        # parent's CPU mask, so disjoint per-rank CPU lists otherwise fail.
        if process_kind == "EngineCore":
            yield
            return
        with _original_configure_subprocess(
            vllm_config,
            local_rank,
            dp_local_rank,
            process_kind,
        ):
            yield

    numa_utils.configure_subprocess = _worker_only_configure_subprocess
    print(
        "[qwen36-june-source] NUMA binding leaves EngineCore unbound; "
        "workers use explicit CPU lists",
        file=sys.stderr,
        flush=True,
    )

print(
    "[qwen36-june-source] source_stack=e190923b "
    f"communicator={inspect.getsourcefile(XpuCommunicator)} "
    f"profile_fence_min_rows={_MIN_ROWS}",
    file=sys.stderr,
    flush=True,
)
