"""Explicit loader for the isolated P1 selected-expert dequant primitive.

No vLLM plugin entry point or automatic serving registration is installed.
"""
from __future__ import annotations
import os
from pathlib import Path

_LOADED = None

def load(path: str | None = None) -> str:
    global _LOADED
    path = str(Path(path or os.environ.get("B70_P1_SO", "/mnt/vm_8tb/b70/build/vllm-flashnext/p1/_C.p1selected.so")).resolve())
    if _LOADED is not None:
        if path != _LOADED:
            raise RuntimeError("b70_p1 already loaded a different library")
        return _LOADED
    import torch
    torch.ops.load_library(path)
    _LOADED = path
    return path
