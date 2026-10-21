"""Opt-in host breadcrumbs for a diagnostic serve; no XPU calls on this path.

Use --worker-extension-cls d1_liveness.LivenessExtension and put this
directory first in PYTHONPATH. B70_D1_LIVENESS_DIR selects the output.
Create OUT/arm containing 'coarse' or 'detail' after normal timed work,
with no requests in flight. Wait for OUT/armed-PID.json before probing.
All inference methods remain untouched until this explicit arm operation.
The first armed execute_model starts a 120-second observation period while
Python progresses. The owning callback must enforce its own wall timeout.
This instrumented period cannot qualify latency. Removing arm disarms it.
"""

from __future__ import annotations

import faulthandler
import functools
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

_ROOT = None
_FD = None
_STACK_FILE = None
_ARMED = False
_DETAIL = False
_DEADLINE = 0.0
_EXPIRED = False
_STEP = 0
_SEQ = 0
_ACTIVE = {}
_THREAD_LAST = {}
_STATE_LOCK = threading.Lock()
_TOKEN = 0
_INSTALLED = False
_HOOKS_INSTALLED = False
_STOP = threading.Event()


def _type_name(value):
    return type(value).__name__


def _safe_value(value, depth=0):
    """Only serialize exact builtins; never coerce an unknown object.

    In particular, tensor __str__/__repr__/__int__/__bool__ may touch XPU.
    Shape entries and metadata values that are not Python scalars are
    represented by their type, not their value.
    """
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if depth < 5 and type(value) in (list, tuple):
        return [_safe_value(item, depth + 1) for item in value[:128]]
    if depth < 5 and type(value) is dict:
        return {key: _safe_value(item, depth + 1)
                for key, item in list(value.items())[:128] if type(key) is str}
    return {"object_type": _type_name(value)}


def _error_info(exc):
    out = {"error_type": _type_name(exc)}
    try:
        out["args"] = _safe_value(BaseException.args.__get__(exc))
    except BaseException:
        pass
    return out


def _emit(event, **fields):
    global _SEQ
    # Diagnostics must never prevent the original operation, even if JSON,
    # metadata, time sampling or file writes fail. No arbitrary default=str.
    try:
        if _FD is None:
            return
        with _STATE_LOCK:
            _SEQ += 1
            seq, step = _SEQ, _STEP
        row = dict(event=event, seq=seq, step=step, pid=os.getpid(),
                   tid=threading.get_native_id(), mono_ns=time.monotonic_ns(),
                   wall_ns=time.time_ns(), **fields)
        payload = json.dumps(_safe_value(row), ensure_ascii=True) + "\n"
        os.write(_FD, payload.encode("ascii"))
    except BaseException:
        pass


def _try_emit(event, **fields):
    # Keep the inference boundary fail-open even if the emitter itself is
    # replaced by a failing test/future sink.
    try:
        _emit(event, **fields)
    except BaseException:
        pass


def _push_frame(label, meta):
    global _TOKEN
    tid = threading.get_native_id()
    now = time.monotonic_ns()
    with _STATE_LOCK:
        _TOKEN += 1
        token = _TOKEN
        frame = dict(phase=label, token=token, tid=tid, step=_STEP,
                     started_ns=now, **meta)
        _ACTIVE.setdefault(tid, []).append(frame)
        _THREAD_LAST[tid] = now
    return tid, token


def _remove_frame(tid, token):
    # Remove this exact frame; another thread's return or nested phase
    # cannot pop it. No state lock is held while the original op runs.
    with _STATE_LOCK:
        frames = _ACTIVE.get(tid, [])
        frames[:] = [frame for frame in frames if frame["token"] != token]
        if frames:
            _THREAD_LAST[tid] = time.monotonic_ns()
        else:
            _ACTIVE.pop(tid, None)
            _THREAD_LAST.pop(tid, None)


def _active_snapshot():
    now = time.monotonic_ns()
    with _STATE_LOCK:
        return [dict(tid=tid, gap_ns=now - _THREAD_LAST.get(tid, now),
                     frames=[dict(frame) for frame in frames])
                for tid, frames in _ACTIVE.items() if frames]


def _refresh_arm():
    global _ARMED, _DETAIL, _DEADLINE, _EXPIRED
    if _EXPIRED:
        return
    try:
        mode = (_ROOT / "arm").read_text(encoding="ascii").strip()
    except OSError:
        mode = ""
    want = mode in ("coarse", "detail")
    if _ARMED and time.monotonic() >= _DEADLINE:
        want = False
        _EXPIRED = True
    if want != _ARMED:
        _ARMED = want
        _try_emit("arm" if want else "disarm", mode=mode)
        if want:
            _DEADLINE = time.monotonic() + 120
            # CPython's C watchdog prints even if a native call holds the GIL.
            # No signals are sent to an engine with an unknown signal handler.
            faulthandler.dump_traceback_later(10, repeat=True, file=_STACK_FILE)
        else:
            faulthandler.cancel_dump_traceback_later()
    _DETAIL = mode == "detail"


def _shape(x):
    shape = getattr(x, "shape", None)
    # torch.Size is a tuple subclass; only dimensions are copied, never
    # values from the input tensor.
    if shape is None:
        return None
    if type(shape) not in (tuple, list) and not (
            type(shape).__name__ == "Size" and type(shape).__module__ == "torch"):
        return {"object_type": _type_name(shape)}
    return [_safe_value(dim) for dim in shape]


def _arg(args, kwargs, name, position):
    return kwargs[name] if name in kwargs else (args[position] if len(args) > position else None)


def _meta(label, args, kwargs):
    out = {}
    if label in ("fullgraph", "graph_replay", "instantiate", "ep_prepare", "ep_submit"):
        hooks = sys.modules.get("b70_ep_xpu.hooks")
        if hooks is not None:
            out["ep_pending"] = dict(getattr(hooks, "_PENDING", {}))
            out["ep_busy"] = getattr(hooks, "_BUSY", None)
    if label == "execute":
        so = _arg(args, kwargs, "scheduler_output", 1)
        counts = getattr(so, "num_scheduled_tokens", {})
        out.update(total=getattr(so, "total_num_scheduled_tokens", None),
                   scheduled=list(counts.values()),
                   new_reqs=len(getattr(so, "scheduled_new_reqs", [])))
    elif label in ("ep_prepare", "ep_submit", "ep_capture_begin", "ep_capture_end"):
        out["key"] = _arg(args, kwargs, "key", 0)
    elif label in ("graph_replay", "instantiate"):
        out["graph"] = id(_arg(args, kwargs, "self", 0))
    elif label == "fullgraph":
        out["tokens"] = getattr(_arg(args, kwargs, "desc", 1), "num_tokens", None)
    elif label == "pf_apply":
        out.update(layer=getattr(_arg(args, kwargs, "method", 0), "_layer_idx", None),
                   shape=_shape(_arg(args, kwargs, "x", 2)))
    elif label == "pf_experts":
        out.update(shape=_shape(_arg(args, kwargs, "x", 0)),
                   mode=_arg(args, kwargs, "mode", 11), card=_arg(args, kwargs, "card", 12))
    elif label == "gdn":
        instance = _arg(args, kwargs, "self", 0)
        out.update(prefix=getattr(instance, "prefix", None),
                   shape=_shape(_arg(args, kwargs, "hidden_states", 1)),
                   backend=getattr(instance, "gdn_xpu_backend", None))
    elif label == "ple_conv":
        out["shape"] = _shape(_arg(args, kwargs, "normalized", 1))
    elif label == "ple_stage":
        out["shape"] = _shape(_arg(args, kwargs, "ids", 2))
    return _safe_value(out)


def _wrap(owner, name, label, detailed=False):
    orig = getattr(owner, name)
    if getattr(orig, "_d1_liveness", False):
        return

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        global _STEP
        observe = False
        try:
            if label == "execute":
                _refresh_arm()
                with _STATE_LOCK:
                    _STEP += 1
            observe = _ARMED and not (detailed and not _DETAIL)
        except BaseException:
            pass
        if not observe:
            return orig(*args, **kwargs)
        try:
            meta = _meta(label, args, kwargs)
        except BaseException as exc:
            meta = dict(metadata_error=_error_info(exc))
        frame = None
        started = None
        try:
            frame = _push_frame(label, meta)
            started = time.monotonic_ns()
        except BaseException:
            pass
        try:
            _try_emit("enter", phase=label, token=frame[1] if frame else None, **meta)
        except BaseException:
            pass
        try:
            result = orig(*args, **kwargs)
        except BaseException as exc:
            try:
                _try_emit("exception", phase=label, token=frame[1] if frame else None,
                          exception=_error_info(exc), **meta)
            except BaseException:
                pass
            raise
        else:
            try:
                _try_emit("return", phase=label, token=frame[1] if frame else None,
                          elapsed_ns=time.monotonic_ns() - started if started is not None else None,
                          **meta)
            except BaseException:
                pass
            return result
        finally:
            if frame is not None:
                try:
                    _remove_frame(*frame)
                except BaseException:
                    pass

    wrapped._d1_liveness = True
    setattr(owner, name, wrapped)
    _try_emit("installed", phase=label, owner=getattr(owner, "__name__", _type_name(owner)), name=name)


def _host_errors():
    """Existing C++ ops read malloc_host error words and return CPU data only.

    This deliberately does not read device counters or wait for any queue.
    Zero errors do not establish queue progress or successful graph submission.
    """
    import torch

    out = {}
    for name in ("ep_read_errors", "ep_prefill_errors"):
        try:
            result = getattr(torch.ops.b70_ep, name)()
            values = result.tolist() if hasattr(result, "tolist") else list(result)
            out[name] = dict(words=len(values), nonzero=[(i, v) for i, v in enumerate(values) if v])
        except BaseException as exc:
            out[name] = _error_info(exc)
    return out


def _watch():
    # This Python watcher can be starved by the GIL. The C faulthandler
    # watchdog above is the independent fallback, and external /proc sampling
    # does not depend on either. No Torch call occurs unless an armed gap >8s.
    global _ARMED, _EXPIRED
    dumped = {}
    while not _STOP.wait(1):
        if not _HOOKS_INSTALLED:
            try:
                mode = (_ROOT / "arm").read_text(encoding="ascii").strip()
            except OSError:
                mode = ""
            if mode not in ("coarse", "detail"):
                continue
            try:
                _install_hooks()
                _refresh_arm()
                (_ROOT / f"armed-{os.getpid()}.json").write_text(
                    json.dumps(dict(pid=os.getpid(), mode=mode, wall_ns=time.time_ns())),
                    encoding="ascii")
            except Exception as exc:
                _try_emit("install_failed", error=_error_info(exc))
                (_ROOT / f"install-failed-{os.getpid()}.txt").write_text(json.dumps(_error_info(exc)), encoding="ascii")
                return
        if _ARMED and time.monotonic() >= _DEADLINE:
            _ARMED = False
            _EXPIRED = True
            faulthandler.cancel_dump_traceback_later()
            _emit("observation_expired")
        if _HOOKS_INSTALLED and _ARMED and not (_ROOT / "arm").exists():
            _refresh_arm()
        if not _ARMED:
            continue
        active = _active_snapshot()
        long_active = [state for state in active if state["gap_ns"] >= 8_000_000_000
                   and dumped.get(state["tid"]) != state["frames"][-1]["token"]]
        if long_active:
            for state in long_active:
                dumped[state["tid"]] = state["frames"][-1]["token"]
            try:
                _try_emit("long_active_phase", active=active, long_active=long_active,
                          interpretation="duration only; coarse 32K prefill can normally exceed8s",
                          errors=_host_errors())
                faulthandler.dump_traceback(file=_STACK_FILE, all_threads=True)
            except BaseException:
                pass


def _install_hooks():
    global _HOOKS_INSTALLED
    specs = [
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "execute_model", "execute", False),
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "sample_tokens", "sample", False),
        ("vllm.v1.worker.gpu.cudagraph_utils", "CudaGraphManager", "run_fullgraph", "fullgraph", False),
        ("torch.xpu", "XPUGraph", "instantiate", "instantiate", False),
        ("torch.xpu", "XPUGraph", "replay", "graph_replay", False),
        ("b70_ep_xpu.ops", None, "replay_prepare", "ep_prepare", False),
        ("b70_ep_xpu.ops", None, "replay_submit", "ep_submit", False),
        ("b70_ep_xpu.ops", None, "capture_begin", "ep_capture_begin", False),
        ("b70_ep_xpu.ops", None, "capture_end", "ep_capture_end", False),
        ("b70_ep_xpu.prefill_ep", None, "apply", "pf_apply", True),
        ("b70_ep_xpu.prefill_ep", None, "_experts", "pf_experts", True),
        ("vllm.models.qwen4_exp.gguf.mtp_ple", "PLEHistory", "conv", "ple_conv", True),
        ("vllm.models.qwen4_exp.gguf.ple_device", None, "device_stage", "ple_stage", True),
        ("vllm.models.qwen4_exp.xpu.model", "Qwen4ExpPLE", "forward", "ple", True),
        # CustomOp caches a bound forward_xpu during construction. Wrap
        # its outer forward so already-loaded instances are covered.
        ("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn", "QwenGatedDeltaNetAttention", "forward", "gdn", True),
        ("vllm.v1.attention.backends.flash_attn", "FlashAttentionImpl", "forward", "fa", True),
    ]
    for module, cls, name, label, detailed in specs:
        owner = importlib.import_module(module)
        if cls:
            owner = getattr(owner, cls)
        _wrap(owner, name, label, detailed)
    _HOOKS_INSTALLED = True


class LivenessExtension:
    """Supported vLLM worker extension; no Worker method names are overridden."""

    def init_worker_extension(self):
        del self
        global _ROOT, _FD, _STACK_FILE, _INSTALLED
        if _INSTALLED:
            return
        _ROOT = Path(os.environ["B70_D1_LIVENESS_DIR"])
        _ROOT.mkdir(parents=True, exist_ok=True)
        _FD = os.open(_ROOT / f"phases-{os.getpid()}.jsonl",
                      os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        _STACK_FILE = open(_ROOT / f"stacks-{os.getpid()}.txt", "a", encoding="ascii")
        _INSTALLED = True
        _emit("monitor_ready", max_observation_seconds=120, hooks_installed=False,
              no_device_events=True, no_device_sync=True, arm=str(_ROOT / "arm"))
        threading.Thread(target=_watch, daemon=True, name="d1-host-watch").start()

    def destroy_worker_extension(self):
        del self
        _STOP.set()
        faulthandler.cancel_dump_traceback_later()
        _emit("worker_destroy")
