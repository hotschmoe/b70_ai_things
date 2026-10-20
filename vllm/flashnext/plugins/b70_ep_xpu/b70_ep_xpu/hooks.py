"""Pair card 1's SYCL command graph with vLLM's FULL decode graph.

The outermost CUDAGraphWrapper records both graphs in one forward.
Card 1 begins recording on the first moe_ep, after torch has entered
its capture (so a device sync at capture entry does not meet a spinning
wait). Replay submits card 1 before card 0. Nested wrappers call the
original method.
"""

from __future__ import annotations

_INSTALLED = False
_BUSY = 0
# Set only around the outermost replay. d1 calls _before_replay after
# instantiate and before the card-0 submit. One submit per forward.
_PENDING = {"key": None, "sent": False}


def _before_replay() -> None:
    key = _PENDING["key"]
    if key is None or _PENDING["sent"]:
        return
    _PENDING["sent"] = True
    from . import ops

    ops.replay_submit(int(key))


def _install_replay_callback() -> None:
    import torch

    classes = [torch.xpu.XPUGraph]
    cuda_graph = getattr(torch.cuda, "CUDAGraph", None)
    if isinstance(cuda_graph, type) and cuda_graph not in classes:
        classes.append(cuda_graph)
    for cls in classes:
        if getattr(cls, "_b70_ep_before_replay", None) is None:
            cls._b70_ep_before_replay = staticmethod(_before_replay)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.compilation.cuda_graph import CUDAGraphMode, CUDAGraphWrapper
    from vllm.forward_context import get_forward_context, is_forward_context_available

    _install_replay_callback()
    orig = CUDAGraphWrapper.__call__

    def wrapped(self, *args, **kwargs):
        global _BUSY
        if _BUSY:
            return orig(self, *args, **kwargs)
        if not is_forward_context_available():
            return orig(self, *args, **kwargs)
        ctx = get_forward_context()
        mode = ctx.cudagraph_runtime_mode
        if mode == CUDAGraphMode.NONE or mode != self.runtime_mode:
            return orig(self, *args, **kwargs)
        bd = ctx.batch_descriptor
        if bd is None:
            return orig(self, *args, **kwargs)
        nt = int(bd.num_tokens)
        entry = self.concrete_cudagraph_entries.get(bd)
        capturing = entry is None or entry.cudagraph is None
        from . import ops

        _BUSY += 1
        try:
            if capturing:
                ops.capture_begin(nt)
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    ops.capture_end(nt)
            graph = entry.cudagraph if entry is not None else None
            hooked = bool(graph is not None and getattr(type(graph), "_b70_ep_replay_hook", False))
            ops.replay_prepare(nt)
            if hooked:
                # d1 instantiate runs inside orig, then _before_replay submits
                # card 1, then the card-0 graph replays.
                _PENDING["key"] = nt
                _PENDING["sent"] = False
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    if not _PENDING["sent"]:
                        ops.replay_submit(nt)
                    _PENDING["key"] = None
                    _PENDING["sent"] = False
            ops.replay_submit(nt)
            return orig(self, *args, **kwargs)
        finally:
            _BUSY -= 1

    CUDAGraphWrapper.__call__ = wrapped
    _INSTALLED = True
    print("b70_ep: CUDAGraphWrapper hook installed", flush=True)
