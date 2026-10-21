#!/usr/bin/env python3
"""CPU-only regressions for real keyword calls and concurrent breadcrumbs."""
import importlib.util
import json
import os
from pathlib import Path
import threading
import types

D = Path(__file__).resolve().parent
OUT = Path('/mnt/vm_8tb/b70/build/vllm-flashnext/d1/liveness-cpu-check-4')
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('d1_under_test', D / 'd1_liveness.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._FD = os.open(OUT / 'phases.jsonl', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
m._ARMED = m._DETAIL = True
calls = []


class TensorTrap:
    shape = (21, 2560)

    def __repr__(self):
        raise AssertionError('arbitrary repr touched')

    def __str__(self):
        raise AssertionError('arbitrary str touched')

    def __int__(self):
        raise AssertionError('arbitrary int touched')

    def __bool__(self):
        raise AssertionError('arbitrary bool touched')


class GDN:
    prefix = 'layer.1.gdn'
    gdn_xpu_backend = 'sycl'

    def forward(self, *, hidden_states):
        calls.append(hidden_states)
        return hidden_states


x = TensorTrap()
m._wrap(GDN, 'forward', 'gdn', True)
assert GDN().forward(hidden_states=x) is x
assert calls == [x]
rows = [json.loads(line) for line in (OUT / 'phases.jsonl').read_text().splitlines()]
row = next(r for r in rows if r['event'] == 'enter' and r['phase'] == 'gdn')
assert row['shape'] == [21, 2560] and row['backend'] == 'sycl'
assert not m._ACTIVE


class BadShape:
    @property
    def shape(self):
        raise RuntimeError(x)


bad = BadShape()
assert GDN().forward(hidden_states=bad) is bad
assert calls == [x, bad]
assert not m._ACTIVE

# Keyword and unknown-value metadata must never coerce a tensor-like object.
assert m._meta('ep_prepare', (), dict(key=x))['key'] == {'object_type': 'TensorTrap'}
assert m._meta('fullgraph', (None,), dict(desc=types.SimpleNamespace(num_tokens=x)))['tokens'] == {'object_type': 'TensorTrap'}
assert m._meta('pf_experts', (), dict(x=x, mode='fused', card=1)) == dict(shape=[21,2560], mode='fused', card=1)
assert m._meta('ple_conv', (None,), dict(normalized=x))['shape'] == [21,2560]
assert m._meta('ple_stage', (), dict(ids=x))['shape'] == [21,2560]
m._emit('unknown_value', value=x, nested=[x], keyed={'value': x})

# Failures in a diagnostic sink cannot replace original results/exceptions.
original_emitter = m._emit
def failing_emitter(*args, **kwargs):
    raise RuntimeError(x)
m._emit = failing_emitter
assert GDN().forward(hidden_states=x) is x
sentinel = ValueError(x)
def explode():
    raise sentinel
owner = types.SimpleNamespace(explode=explode)
m._wrap(owner, 'explode', 'failure')
try:
    owner.explode()
except ValueError as caught:
    assert caught is sentinel
else:
    raise AssertionError('original exception lost')
assert not m._ACTIVE
m._emit = original_emitter
try:
    owner.explode()
except ValueError as caught:
    assert caught is sentinel
else:
    raise AssertionError('original exception lost')
assert not m._ACTIVE

# A enters, B enters, A returns while B is still active. Global LIFO would
# incorrectly remove B. Per-thread/token cleanup must leave exactly B.
a_enter = threading.Event()
b_enter = threading.Event()
a_release = threading.Event()
b_release = threading.Event()
thread_ids = {}
failures = []

def phase(name, entered, release):
    thread_ids[name] = threading.get_native_id()
    entered.set()
    assert release.wait(3), 'CPU handshake timed out'
    return name

owner = types.SimpleNamespace(phase=phase)
m._wrap(owner, 'phase', 'overlap')

def run(name, entered, release):
    try:
        assert owner.phase(name, entered, release) == name
    except BaseException as exc:
        failures.append(type(exc).__name__)

a = threading.Thread(target=run, args=('a', a_enter, a_release))
b = threading.Thread(target=run, args=('b', b_enter, b_release))
a.start(); assert a_enter.wait(3)
b.start(); assert b_enter.wait(3)
active = m._active_snapshot()
assert {r['tid'] for r in active} == {thread_ids['a'], thread_ids['b']}
with m._STATE_LOCK:
    m._THREAD_LAST[thread_ids['a']] -= 10_000_000_000
active = {row['tid']: row for row in m._active_snapshot()}
assert active[thread_ids['a']]['gap_ns'] >= 8_000_000_000
assert active[thread_ids['b']]['gap_ns'] < 8_000_000_000
a_release.set(); a.join(3)
assert not a.is_alive()
active = m._active_snapshot()
assert len(active) == 1 and active[0]['tid'] == thread_ids['b']
assert active[0]['frames'][-1]['tid'] == thread_ids['b']
b_release.set(); b.join(3)
assert not b.is_alive() and not failures and not m._ACTIVE and not m._THREAD_LAST

rows = [json.loads(line) for line in (OUT / 'phases.jsonl').read_text().splitlines()]
assert any(r.get('metadata_error', {}).get('error_type') == 'RuntimeError' for r in rows)
assert any(r.get('exception', {}).get('args') == [{'object_type':'TensorTrap'}] for r in rows)
result = dict(result='PASS', scope='CPU mocks only; no Torch/XPU import or real worker integration', checks=[
    'actual keyword-only GDN hidden_states call', 'all signature metadata accepts kwargs',
    'no arbitrary repr/str/int/bool coercion', 'metadata getter failure preserves result',
    'logging failure preserves result and original exception identity',
    'exception cleanup', 'overlapping threads finish out of global LIFO order',
    'per-native-TID/token cleanup and active snapshots'])
(OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
os.close(m._FD)
print(json.dumps(result))
