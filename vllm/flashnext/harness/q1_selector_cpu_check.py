"""Regression of decode ownership; stdlib, no Torch or GPU."""

import argparse
import gc
import importlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import threading
import types
import weakref

S = Path(__file__).resolve().parent
API = S / 'q1_selector_api.py'
OUT = Path(os.environ.get('Q1_SELECTOR_CPU_OUT', '/mnt/vm_8tb/b70/build/vllm-flashnext/q1/selector-cpu-default'))


class Tensor:
    def __init__(self, rows, cols):
        self.shape = (rows, cols)
        self.device = 'fake-device'


class Batch:
    def __init__(self):
        self.scores = Tensor(4, 147456)
        self.ids = Tensor(4, 2051)
        self.step = Tensor(4, 4)


def factory_module():
    mod = types.ModuleType('q1_fused_r2.selector')
    mod.created = []

    class PackedTopk:
        def __init__(self, columns, cap, device):
            self.calls = 0
            mod.created.append(weakref.ref(self))

        def __call__(self, scores, ids, step):
            self.calls += 1
            return ids

    mod.PackedTopk = PackedTopk
    sys.modules[mod.__name__] = mod
    return mod


def runtime():
    ops = types.ModuleType('b70_qsa_xpu.ops')
    ops.native_calls = []
    ops._load = lambda: None
    ops._ready = lambda value: value

    def native(scores, ids, step):
        ops.native_calls.append((scores, ids, step))
        return ids

    ops.topk_batch = native
    runner = types.ModuleType('b70_qsa_xpu.model_hook')

    class QsaRunner:
        def __init__(self):
            self.batch = Batch()

        def _decode(self, nested=None, fail=False, barrier=None):
            if nested:
                nested()
            if barrier:
                barrier.wait()
            result = ops.topk_batch(self.batch.scores, self.batch.ids, self.batch.step)
            if fail:
                raise ValueError('decode witness')
            return result

        def _prefill(self, fail=False):
            # Both persistent-buffer fallback and a tiny transient tail must
            # stay native, even if called from inside _decode.
            ops.topk_batch(self.batch.scores, self.batch.ids, self.batch.step)
            ops.topk_batch(Tensor(1, 147456), Tensor(1, 2051), Tensor(1, 4))
            if fail:
                raise ValueError('prefill witness')

    runner.QsaRunner = QsaRunner
    return ops, runner


def check_scope():
    spec = importlib.util.spec_from_file_location('q1_review_hook', S / 'q1_selector_hook.py')
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    ops, mod = runtime()
    factory = factory_module()
    hook._patch_ops(ops, 'fused')
    hook._patch_runner(mod)
    r = mod.QsaRunner()
    # Persistent owner creates exactly one retained workspace across calls.
    for _ in range(4):
        r._decode()
    assert len(factory.created) == 1
    state = getattr(r.batch, hook._STATE_ATTR)
    assert state.selector.calls == 4
    # Same-sized, tiny and nested prefill never select or allocate a cache.
    r._prefill()
    r._decode(nested=lambda: r._prefill())
    assert len(factory.created) == 1 and len(ops.native_calls) == 4
    assert state.selector.calls == 5
    # A nested prefill exception restores the enclosing decode owner.
    def nested_failure():
        try:
            r._prefill(fail=True)
        except ValueError:
            pass
        assert hook._CONTEXT.owner is r.batch
    r._decode(nested=nested_failure)
    try:
        r._decode(fail=True)
    except ValueError:
        pass
    assert hook._CONTEXT.owner is None
    ops.topk_batch(r.batch.scores, r.batch.ids, r.batch.step)
    assert len(ops.native_calls) == 7
    # Fail closed without evicting the old workspace after buffer rebinding.
    r.batch.scores = Tensor(4, 147456)
    try:
        r._decode()
    except RuntimeError as exc:
        assert 'fresh QsaBatch' in str(exc)
    else:
        raise AssertionError('replaced buffers accepted')
    assert getattr(r.batch, hook._STATE_ATTR) is state
    # Thread-local scopes and separate owners do not share selector state.
    pair = [mod.QsaRunner(), mod.QsaRunner()]
    barrier = threading.Barrier(2)
    failures = []
    def threaded(owner):
        try:
            owner._decode(barrier=barrier)
        except Exception as exc:
            failures.append(exc)
    threads = [threading.Thread(target=threaded, args=(owner,)) for owner in pair]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures and len(factory.created) == 3
    assert getattr(pair[0].batch, hook._STATE_ATTR).selector is not getattr(pair[1].batch, hook._STATE_ATTR).selector
    # Dead owners are not retained by a global pointer/shape cache.
    owner_ref = weakref.ref(r.batch)
    selector_ref = weakref.ref(state.selector)
    del state, r
    gc.collect()
    assert owner_ref() is None and selector_ref() is None
    assert 'torch' not in sys.modules
    return ['persistent_workspace', 'tiny_prefill_native', 'nested_prefill_native',
            'exception_owner_restore', 'replaced_buffer_fail_closed',
            'thread_local_owners', 'owner_lifetime_no_global_cache']


def spawn_check(mode, output):
    # Real spawn reimports this file, then the real API shim as __mp_main__.
    # Pure fake package files exercise the fresh-module meta-path loader.
    os.environ['B70_Q1_SELECTOR'] = mode
    root = OUT / ('fake-runtime-' + mode)
    pkg = root / 'b70_qsa_xpu'
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / '__init__.py').write_text('')
    (pkg / 'ops.py').write_text('native_calls=[]\ndef _load(): pass\ndef _ready(x): return x\ndef topk_batch(a,b,c):\n native_calls.append(a)\n return b\n')
    (pkg / 'model_hook.py').write_text('from . import ops\nclass QsaRunner:\n def _decode(self): return ops.topk_batch(self.batch.scores,self.batch.ids,self.batch.step)\n def _prefill(self): return ops.topk_batch(self.batch.scores,self.batch.ids,self.batch.step)\n')
    sys.path[:0] = [str(S), str(root)]
    factory = factory_module()
    runpy.run_path(str(API), run_name='__mp_main__')
    hook = importlib.import_module('q1_selector_hook')
    ops = importlib.import_module('b70_qsa_xpu.ops')
    runner = importlib.import_module('b70_qsa_xpu.model_hook').QsaRunner()
    runner.batch = Batch()
    runner._decode()
    runner._prefill()
    if mode == 'native':
        assert not hook._INSTALLED and len(factory.created) == 0
        assert len(ops.native_calls) == 2
    else:
        assert hook._INSTALLED and len(factory.created) == 1
        assert len(ops.native_calls) == 1
    assert 'torch' not in sys.modules
    Path(output).write_text(json.dumps(dict(mode=mode, no_torch=True,
                                         selectors=len(factory.created), native_calls=len(ops.native_calls))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ['Q1_SELECTOR_CPU_OUT'] = str(OUT)
    checks = check_scope()
    ctx = multiprocessing.get_context('spawn')
    for mode in ('native', 'fused'):
        output = OUT / ('spawn-' + mode + '.json')
        child = ctx.Process(target=spawn_check, args=(mode, str(output)))
        child.start()
        child.join(10)
        assert child.exitcode == 0
        checks.append('spawn_' + mode + '_no_torch')
    result = dict(pass_cpu=True, checks=checks, scope='stdlib fake CPU only; no GPU/actual factory execution')
    (OUT / 'cpu-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
