"""Decode-only QSA selector overlay; stdlib imports before worker setup.

Selection is authorized only by QsaRunner._decode and only for the exact
persistent QsaBatch buffers. Tiny prefill tails and direct run_decode calls
outside that runner context retain the native selector. Each live QsaBatch
owns one selector/workspace; no global pointer cache retains dead batches.
"""

import functools
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import threading

_OPS = 'b70_qsa_xpu.ops'
_RUNNER = 'b70_qsa_xpu.model_hook'
_TARGETS = (_OPS, _RUNNER)
_FACTORIES = {'torch': 'q1_torch.selector', 'fused': 'q1_fused_r2.selector'}
_INSTALLED = False
_CONTEXT = threading.local()
_STATE_ATTR = '_q1_decode_selector_state_r2'


def _buffers(owner):
    return owner.scores, owner.ids, owner.step


class _DecodeState:
    def __init__(self, mode, buffers, selector):
        self.mode = mode
        self.buffers = buffers
        self.shapes = tuple(tuple(value.shape) for value in buffers)
        self.selector = selector

    def matches(self, mode, buffers):
        return (self.mode == mode
                and all(a is b for a, b in zip(self.buffers, buffers))
                and self.shapes == tuple(tuple(value.shape) for value in buffers))


def _patch_ops(module, mode):
    original = module.topk_batch
    if getattr(original, '_q1_selector', False):
        return

    @functools.wraps(original)
    def select(scores, ids, step):
        owner = getattr(_CONTEXT, 'owner', None)
        passed = (scores, ids, step)
        if owner is None or not all(a is b for a, b in zip(passed, _buffers(owner))):
            return original(scores, ids, step)
        # QsaBatch constructor creates contiguous persistent tensors. Avoid
        # converting transient/foreign tensors into a new cached workspace.
        module._load()
        ready = tuple(module._ready(value) for value in passed)
        if not all(a is b for a, b in zip(passed, ready)):
            raise RuntimeError('Q1 decode selector requires contiguous owner buffers')
        state = getattr(owner, _STATE_ATTR, None)
        if state is None:
            factory = importlib.import_module(_FACTORIES[mode]).PackedTopk
            selector = factory(scores.shape[1], ids.shape[1], scores.device)
            state = _DecodeState(mode, passed, selector)
            setattr(owner, _STATE_ATTR, state)
            print(f'Q1_SELECTOR_APPLIED mode={mode} scope=decode '
                  f'batch={scores.shape[0]} columns={scores.shape[1]} '
                  f'cap={ids.shape[1]} owner={id(owner)}', flush=True)
        elif not state.matches(mode, passed):
            # Never replace/evict a workspace that captured graphs may own.
            # Rebinding requires a fresh QsaBatch and fresh graph warmup.
            raise RuntimeError('Q1 persistent decode buffers changed; create a fresh QsaBatch')
        return state.selector(*ready)

    select._q1_selector = True
    select._q1_selector_scope = 'QsaRunner._decode/persistent_QsaBatch'
    module.topk_batch = select


def _patch_runner(module):
    cls = module.QsaRunner
    if getattr(cls._decode, '_q1_selector_scope', False):
        return
    decode = cls._decode
    prefill = cls._prefill

    @functools.wraps(decode)
    def scoped_decode(self, *args, **kwargs):
        previous = getattr(_CONTEXT, 'owner', None)
        _CONTEXT.owner = self.batch
        try:
            return decode(self, *args, **kwargs)
        finally:
            _CONTEXT.owner = previous

    @functools.wraps(prefill)
    def native_prefill(self, *args, **kwargs):
        previous = getattr(_CONTEXT, 'owner', None)
        _CONTEXT.owner = None
        try:
            return prefill(self, *args, **kwargs)
        finally:
            _CONTEXT.owner = previous

    scoped_decode._q1_selector_scope = True
    native_prefill._q1_selector_scope = True
    cls._decode = scoped_decode
    cls._prefill = native_prefill


def _patch(module, mode):
    if module.__name__ == _OPS:
        _patch_ops(module, mode)
    elif module.__name__ == _RUNNER:
        _patch_runner(module)


class _Loader(importlib.abc.Loader):
    def __init__(self, original, mode):
        self.original, self.mode = original, mode

    def create_module(self, spec):
        create = getattr(self.original, 'create_module', None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original.exec_module(module)
        _patch(module, self.mode)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, mode):
        self.mode = mode

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader, self.mode)
        return spec


def install():
    global _INSTALLED
    mode = os.environ.get('B70_Q1_SELECTOR', 'native')
    if _INSTALLED or mode == 'native':
        return
    if mode not in _FACTORIES:
        raise ValueError('B70_Q1_SELECTOR must be native, torch or fused')
    # Existing instances dispatch _decode/_prefill through the class; patch
    # existing modules too. A finder handles fresh API/spawn imports.
    for name in _TARGETS:
        if name in sys.modules:
            _patch(sys.modules[name], mode)
    sys.meta_path.insert(0, _Finder(mode))
    _INSTALLED = True
    print(f'Q1_SELECTOR_HOOK_INSTALLED pid={os.getpid()} mode={mode} '
          'scope=decode', flush=True)
