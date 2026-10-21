"""Explicit experimental card-1 native-prefill hook. Default off."""
from __future__ import annotations
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

_TARGET = 'b70_ep_xpu.prefill_ep'
_INSTALLED = False
_BUCKET_LOADED = False


def _load_bucket():
    global _BUCKET_LOADED
    from b70_gguf_xpu.prefill import load as load_control
    from b70_p1_xpu import load as load_selected
    import torch
    load_control()
    load_selected()
    if not _BUCKET_LOADED:
        path = Path(os.environ.get('B70_P1_BUCKET_SO', '/mnt/vm_8tb/b70/build/vllm-flashnext/p1/_C.p1bucket.so'))
        torch.ops.load_library(str(path))
        _BUCKET_LOADED = True


def _patch(module):
    original = module._experts
    if getattr(original, '_b70_p1', False):
        return
    selected = os.environ.get('B70_PREFILL_SELECTED_DQ', '0') == '1'
    logged = False

    def experts(x, ids, w, gate, up, down, gt, ut, dt, h, f, mode, card):
        nonlocal logged
        if card != 1 or not mode.startswith('native'):
            return original(x, ids, w, gate, up, down, gt, ut, dt, h, f, mode, card)
        if gt != ut:
            raise RuntimeError('P1 native bucket requires equal gate/up type')
        import torch
        _load_bucket()
        tile = int(mode.split(':', 1)[1]) if ':' in mode else 32
        maxpad = module._C1_MAXPAD
        if not logged:
            print(f'P1_BUCKET_APPLIED card=1 selected={int(selected)} tile={tile} maxpad={maxpad}', flush=True)
            logged = True
        return torch.ops.b70_p1.prefill_moe_bucketed(x, ids, w, gate, up, down, gt, dt, h, f, tile, maxpad, selected)

    experts._b70_p1 = True
    module._experts = experts


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, 'create_module', None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    """Called at top level by p1_api.py in API and spawned EngineCore."""
    global _INSTALLED
    if _INSTALLED or os.environ.get('B70_P1_BUCKET_OVERLAY') != '1':
        return
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
    _INSTALLED = True
    print(f'P1_BUCKET_HOOK_INSTALLED pid={os.getpid()} selected={os.environ.get("B70_PREFILL_SELECTED_DQ", "0")}', flush=True)
