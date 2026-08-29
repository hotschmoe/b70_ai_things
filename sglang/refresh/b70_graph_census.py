"""Default-off collective annotations for bounded graph census profiles."""

from __future__ import annotations

import functools

import torch


_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from sglang.srt.distributed.parallel_state import GroupCoordinator

    original = GroupCoordinator.all_reduce
    if getattr(original, "_b70_graph_census", False):
        _INSTALLED = True
        return

    @functools.wraps(original)
    def annotated_all_reduce(self, input_):
        dtype = str(input_.dtype).removeprefix("torch.")
        shape = "x".join(str(dimension) for dimension in input_.shape)
        label = f"b70::collective all_reduce dtype={dtype} shape={shape}"
        with torch.profiler.record_function(label):
            return original(self, input_)

    annotated_all_reduce._b70_graph_census = True
    GroupCoordinator.all_reduce = annotated_all_reduce
    _INSTALLED = True
    print("[b70-graph-census] installed all-reduce annotations", flush=True)
