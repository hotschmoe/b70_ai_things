"""torch.ops.b70_ep wrappers. Importing this module does not touch the GPU."""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False
_FAKES = False


def _register_fakes() -> None:
    global _FAKES
    if _FAKES:
        return
    import torch
    from torch.library import register_fake

    @register_fake("b70_ep::moe_ep")
    def _moe_ep_fake(x, w, ids, layer):
        del w, ids, layer
        return torch.empty(x.shape, dtype=torch.float32, device=x.device)

    _FAKES = True


def load() -> None:
    global _LOADED
    if _LOADED:
        return
    import torch

    so = Path(__file__).resolve().parent / "_C.so"
    if not so.is_file():
        raise FileNotFoundError(f"missing {so}; run scripts/build.sh")
    torch.ops.load_library(str(so))
    _register_fakes()
    _LOADED = True
    gguf = os.environ.get("B70_GGUF_SO", "")
    print(f"b70_ep: loaded {so} B70_GGUF_SO={gguf or '(unset)'}", flush=True)


def init(max_t: int, max_k: int, max_topk: int, nlayers: int):
    load()
    import torch

    return torch.ops.b70_ep.ep_init(int(max_t), int(max_k), int(max_topk), int(nlayers))


def register0(layer, gate, gate_type, up, up_type, down, down_type, e, intermediate, hidden):
    load()
    import torch

    return torch.ops.b70_ep.ep_register0(
        int(layer), gate, int(gate_type), up, int(up_type), down, int(down_type),
        int(e), int(intermediate), int(hidden),
    )


def register1(layer, gate, up, down, e):
    load()
    import torch

    return torch.ops.b70_ep.ep_register1(int(layer), gate, up, down, int(e))


def register_map(layer, map0, map1):
    load()
    import torch

    return torch.ops.b70_ep.ep_register_map(int(layer), map0, map1)


def capture_begin(key: int):
    load()
    import torch

    return torch.ops.b70_ep.ep_capture_begin(int(key))


def capture_end(key: int):
    load()
    import torch

    return torch.ops.b70_ep.ep_capture_end(int(key))


def replay_prepare(key: int):
    load()
    import torch

    return torch.ops.b70_ep.ep_replay_prepare(int(key))


def replay_submit(key: int):
    load()
    import torch

    return torch.ops.b70_ep.ep_replay_submit(int(key))


def replay(key: int):
    load()
    import torch

    return torch.ops.b70_ep.ep_replay(int(key))


def read_errors():
    load()
    import torch

    return torch.ops.b70_ep.ep_read_errors()
