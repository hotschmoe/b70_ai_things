# Eager-step timer for the D1 decode-overhead pass.
# Installed from sitecustomize when B70_STEP_TIMER=1.
# Import-safe until the vLLM modules load. ASCII logs only.
#
# Logs, per real (non-dummy) step:
#   engine_step wall, execute_model wall, sample_tokens wall, token count.
# On B70_DISPATCH_STEP (default 30) records leaf torch-dispatch XPU
# events and writes a ranked device-time table.
# On B70_CPROFILE_STEP (default 36) writes a cProfile tottime table.
# On B70_PROFILE_STEP (default 48) writes a torch.profiler CPU+XPU table.

import importlib.machinery
import os
import sys
import time

_LOG = os.environ.get(
    "B70_STEP_LOG",
    "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d1-step-timer.log",
)
_DISPATCH_STEP = int(os.environ.get("B70_DISPATCH_STEP", "30"))
_CPROFILE_STEP = int(os.environ.get("B70_CPROFILE_STEP", "36"))
_PROFILE_STEP = int(os.environ.get("B70_PROFILE_STEP", "48"))
_PID = os.getpid()

_state = {
    "n": 0,
    "timer": None,
    "cprof": None,
    "prof": None,
    "decode_walls": [],
}


def _log(msg: str) -> None:
    line = f"pid={_PID} {msg}\n"
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        sys.stderr.write(line)


def _summarize_timer(timer) -> None:
    try:
        import torch

        torch.xpu.synchronize()
    except Exception as exc:
        _log(f"dispatch_sync_fail {exc!r}")
        return
    buckets: dict[str, list[float]] = {}
    total = 0.0
    nleaf = 0
    for name, start, end in timer.leaves:
        try:
            ms = float(start.elapsed_time(end))
        except Exception:
            continue
        if ms < 0:
            ms = 0.0
        nleaf += 1
        total += ms
        buckets.setdefault(name, []).append(ms)
    ranked = sorted(buckets.items(), key=lambda kv: -sum(kv[1]))
    _log(
        f"dispatch_summary leaves={nleaf} device_ms={total:.3f} "
        f"ops={len(ranked)}"
    )
    for name, samples in ranked[:40]:
        sm = sum(samples)
        _log(
            f"dispatch_op ms={sm:.3f} n={len(samples)} "
            f"mean={sm / len(samples):.4f} {name}"
        )


def _dump_cprofile(prof) -> None:
    import io
    import pstats

    for sort_key in ("tottime", "cumtime"):
        buf = io.StringIO()
        stats = pstats.Stats(prof, stream=buf)
        stats.sort_stats(sort_key)
        stats.print_stats(30)
        _log(f"cprofile_begin sort={sort_key}")
        for line in buf.getvalue().splitlines():
            _log("cprofile " + line)
        _log(f"cprofile_end sort={sort_key}")


def _dump_torch_prof(prof) -> None:
    try:
        events = prof.key_averages()
    except Exception as exc:
        _log(f"profiler_averages_fail {exc!r}")
        return
    if len(events) == 0:
        _log("profiler_empty")
        return
    sample = events[0]
    fields = [a for a in dir(sample) if "time" in a or a in ("count", "key")]
    _log("profiler_fields " + ",".join(fields))
    for sort_key in (
        "self_xpu_time_total",
        "self_device_time_total",
        "self_cpu_time_total",
        "cpu_time_total",
    ):
        try:
            table = events.table(sort_by=sort_key, row_limit=25)
        except Exception as exc:
            _log(f"profiler_table_skip sort={sort_key} {exc!r}")
            continue
        _log(f"profiler_table_begin sort={sort_key}")
        for line in table.splitlines():
            _log("profiler " + line)
        _log(f"profiler_table_end sort={sort_key}")
    try:
        stacked = prof.key_averages(group_by_stack_n=6)
        table = stacked.table(sort_by="self_cpu_time_total", row_limit=15)
        _log("profiler_stack_begin")
        for line in table.splitlines():
            _log("profiler " + line)
        _log("profiler_stack_end")
    except Exception as exc:
        _log(f"profiler_stack_fail {exc!r}")


def _start_profile():
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    xpu = getattr(ProfilerActivity, "XPU", None)
    if xpu is not None:
        activities.append(xpu)
    _log(f"profiler_start activities={activities}")
    prof = profile(activities=activities, record_shapes=False, with_stack=False)
    prof.__enter__()
    return prof


def _patch_runner(module) -> None:
    cls = module.GPUModelRunner
    if getattr(cls.execute_model, "_d1_wrapped", False):
        return
    orig_exec = cls.execute_model
    orig_sample = cls.sample_tokens

    def execute_model(self, scheduler_output, *args, **kwargs):
        dummy = bool(kwargs.get("dummy_run", False))
        if dummy:
            return orig_exec(self, scheduler_output, *args, **kwargs)
        _state["n"] += 1
        n = _state["n"]
        n_tok = getattr(scheduler_output, "total_num_scheduled_tokens", -1)
        if _state["timer"] is not None and n != _DISPATCH_STEP:
            try:
                _state["timer"].__exit__(None, None, None)
            except Exception as exc:
                _log(f"dispatch_exit_early {exc!r}")
            _summarize_timer(_state["timer"])
            _state["timer"] = None
        if n == _DISPATCH_STEP:
            try:
                from torch.utils._python_dispatch import TorchDispatchMode

                class _LeafTimer(TorchDispatchMode):
                    def __init__(self):
                        super().__init__()
                        self.leaves = []

                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        import torch

                        kwargs = {} if kwargs is None else kwargs
                        n0 = len(self.leaves)
                        start = torch.xpu.Event(enable_timing=True)
                        end = torch.xpu.Event(enable_timing=True)
                        start.record()
                        try:
                            return func(*args, **kwargs)
                        finally:
                            end.record()
                            if len(self.leaves) == n0:
                                self.leaves.append((str(func), start, end))

                timer = _LeafTimer()
                timer.__enter__()
                _state["timer"] = timer
                _log(f"dispatch_armed step={n}")
            except Exception as exc:
                _log(f"dispatch_arm_fail {exc!r}")
        if n == _CPROFILE_STEP:
            import cProfile

            prof = cProfile.Profile()
            prof.enable()
            _state["cprof"] = prof
            _log(f"cprofile_armed step={n}")
        if n == _PROFILE_STEP and _PROFILE_STEP > 0:
            try:
                _state["prof"] = _start_profile()
            except Exception as exc:
                _state["prof"] = None
                _log(f"profiler_arm_fail {exc!r}")
        t0 = time.perf_counter()
        try:
            return orig_exec(self, scheduler_output, *args, **kwargs)
        finally:
            wall = (time.perf_counter() - t0) * 1000.0
            _log(f"execute step={n} tokens={n_tok} wall_ms={wall:.2f}")

    def sample_tokens(self, *args, **kwargs):
        n = _state["n"]
        t0 = time.perf_counter()
        try:
            return orig_sample(self, *args, **kwargs)
        finally:
            wall = (time.perf_counter() - t0) * 1000.0
            _log(f"sample step={n} wall_ms={wall:.2f}")
            timer = _state["timer"]
            if timer is not None and n == _DISPATCH_STEP:
                try:
                    timer.__exit__(None, None, None)
                except Exception as exc:
                    _log(f"dispatch_exit_fail {exc!r}")
                _summarize_timer(timer)
                _state["timer"] = None
            prof = _state["cprof"]
            if prof is not None and n == _CPROFILE_STEP:
                prof.disable()
                _state["cprof"] = None
                try:
                    _dump_cprofile(prof)
                except Exception as exc:
                    _log(f"cprofile_dump_fail {exc!r}")
            tprof = _state["prof"]
            if tprof is not None and n == _PROFILE_STEP:
                _state["prof"] = None
                try:
                    tprof.__exit__(None, None, None)
                    _dump_torch_prof(tprof)
                except Exception as exc:
                    _log(f"profiler_dump_fail {exc!r}")

    execute_model._d1_wrapped = True
    sample_tokens._d1_wrapped = True
    cls.execute_model = execute_model
    cls.sample_tokens = sample_tokens
    _log("patched GPUModelRunner.execute_model and sample_tokens")


def _patch_engine(module) -> None:
    cls = module.EngineCore
    if getattr(cls.step, "_d1_wrapped", False):
        return
    orig = cls.step

    def step(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        wall = (time.perf_counter() - t0) * 1000.0
        executed = False
        try:
            executed = bool(out[1])
        except Exception:
            executed = False
        if executed and wall >= 1.0:
            _state["decode_walls"].append(wall)
            _log(f"engine_step wall_ms={wall:.2f} n_exec={_state['n']}")
        return out

    step._d1_wrapped = True
    cls.step = step
    _log("patched EngineCore.step")


class _Loader:
    def __init__(self, inner, fullname):
        self.inner = inner
        self.fullname = fullname

    def create_module(self, spec):
        if hasattr(self.inner, "create_module"):
            return self.inner.create_module(spec)
        return None

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            if self.fullname == "vllm.v1.worker.gpu.model_runner":
                _patch_runner(module)
            elif self.fullname == "vllm.v1.engine.core":
                _patch_engine(module)
        except Exception as exc:
            _log(f"patch_fail {self.fullname} {exc!r}")


class _Finder:
    _targets = {
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.engine.core",
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


def install() -> None:
    sys.meta_path.insert(0, _Finder())
    _log(
        f"timer_install dispatch={_DISPATCH_STEP} "
        f"cprofile={_CPROFILE_STEP} profiler={_PROFILE_STEP}"
    )
