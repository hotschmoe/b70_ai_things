# Count torch.distributed collectives per model-runner step.
# Installed from sitecustomize when B70_COLL_LOG is set to a log path.
# One device synchronize brackets each call so the time includes the
# collective. The wrapper does not synchronize while the XPU stream is
# capturing. ASCII only.

import importlib.machinery
import os
import sys
import time

_LOG = os.environ["B70_COLL_LOG"]
_PID = os.getpid()
_BUF: list[tuple[str, int, float, int]] = []
_DETAIL_LEFT = 1
_NAMES = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "broadcast",
    "barrier",
    "all_to_all",
    "all_to_all_single",
)


def _log(msg: str) -> None:
    line = f"pid={_PID} {msg}\n"
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        sys.stderr.write(line)


def _capturing() -> bool:
    try:
        import torch
    except Exception:
        return False
    fn = getattr(getattr(torch, "xpu", None), "is_current_stream_capturing", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _nbytes(args) -> int:
    try:
        import torch
    except Exception:
        return 0
    total = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            total += int(arg.numel()) * int(arg.element_size())
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.Tensor):
                    total += int(item.numel()) * int(item.element_size())
    return total


def _sync() -> None:
    import torch

    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        torch.xpu.synchronize()


def _wrap(name: str, fn):
    def wrapped(*args, **kwargs):
        if _capturing():
            _BUF.append((name, _nbytes(args), -1.0, 1))
            return fn(*args, **kwargs)
        try:
            _sync()
        except Exception:
            pass
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        try:
            _sync()
        except Exception:
            pass
        ms = (time.perf_counter() - t0) * 1000.0
        _BUF.append((name, _nbytes(args), ms, 0))
        return out

    wrapped._c7_wrapped = True
    return wrapped


def _flush(tag: str, n_tok: int, wall_ms: float) -> None:
    global _DETAIL_LEFT
    if not _BUF:
        _log(f"step tag={tag} tokens={n_tok} wall_ms={wall_ms:.3f} coll=0")
        return
    counts: dict[str, int] = {}
    times: dict[str, float] = {}
    for name, _nbytes_i, ms, _cap in _BUF:
        counts[name] = counts.get(name, 0) + 1
        if ms >= 0:
            times[name] = times.get(name, 0.0) + ms
    parts = []
    for name in sorted(counts):
        parts.append(f"{name}={counts[name]}/{times.get(name, 0.0):.3f}ms")
    total_ms = sum(times.values())
    _log(
        f"step tag={tag} tokens={n_tok} wall_ms={wall_ms:.3f} "
        f"coll={len(_BUF)} coll_ms={total_ms:.3f} " + " ".join(parts)
    )
    if tag == "execute" and _DETAIL_LEFT > 0 and n_tok == 1:
        for name, nbytes, ms, cap in _BUF:
            _log(f"call {name} bytes={nbytes} ms={ms:.3f} capture={cap}")
        _DETAIL_LEFT -= 1
    _BUF.clear()


def _patch_dist(module) -> None:
    for name in _NAMES:
        fn = getattr(module, name, None)
        if fn is None or getattr(fn, "_c7_wrapped", False):
            continue
        setattr(module, name, _wrap(name, fn))
    _log("patched torch.distributed collectives")


def _patch_runner(module, label: str) -> None:
    cls = getattr(module, "GPUModelRunner", None)
    if cls is None:
        _log(f"runner_miss {label}")
        return
    current = cls.execute_model
    if getattr(current, "_c7_wrapped", False):
        return
    orig = current

    def execute_model(self, *args, **kwargs):
        dummy = bool(kwargs.get("dummy_run", False))
        n_tok = -1
        if args:
            n_tok = int(getattr(args[0], "total_num_scheduled_tokens", -1) or -1)
        if _BUF:
            _flush("before", n_tok, 0.0)
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        wall = (time.perf_counter() - t0) * 1000.0
        tag = "dummy" if dummy else "execute"
        _flush(tag, n_tok, wall)
        return out

    execute_model._c7_wrapped = True
    cls.execute_model = execute_model
    _log(f"patched {label}.GPUModelRunner.execute_model")


class _Loader:
    def __init__(self, inner, fullname: str):
        self._inner = inner
        self._fullname = fullname

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _apply(self._fullname, module)


class _Finder:
    _targets = {
        "torch.distributed",
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    }
    _done: set[str] = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._targets or fullname in self._done:
            return None
        self._done.add(fullname)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            _log(f"finder_miss {fullname}")
            return None
        spec.loader = _Loader(spec.loader, fullname)
        return spec


def _apply(fullname: str, module) -> None:
    if fullname == "torch.distributed":
        _patch_dist(module)
    else:
        _patch_runner(module, fullname)


def install() -> None:
    sys.meta_path.insert(0, _Finder())
    for name in _Finder._targets:
        module = sys.modules.get(name)
        if module is not None:
            _Finder._done.add(name)
            _apply(name, module)
    _log(f"coll_install log={_LOG}")
