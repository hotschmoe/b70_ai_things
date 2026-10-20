"""Load libb70_kvpark.so. Copies do not stage through a CPU buffer."""

from __future__ import annotations

import os

import torch

DEFAULT_SO = "/mnt/vm_8tb/b70/build/vllm-flashnext/b70_kvpark/libb70_kvpark.so"
_LOADED = False


def load(path: str | None = None) -> None:
    global _LOADED
    if _LOADED:
        return
    lib = path or os.environ.get("B70_KVPARK_SO", DEFAULT_SO)
    torch.ops.load_library(lib)
    _LOADED = True
