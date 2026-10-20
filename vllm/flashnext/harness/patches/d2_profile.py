# In-graph device-time breakdown for one Ornith decode step.
# Installed from sitecustomize when B70_D2_PROFILE=1.
#
# Graph replay hides per-kernel times. This records eager op counts, then
# re-captures each region (GDN proj, GDN core, attention, MoE, shared
# expert, norms, lm_head) as its own XPU graph and times the replay.
# That is the same device-time method as D1's 11.4 ms full-graph sample.
# Eager per-leaf events are not used: D1's dispatch timer reported 539 ms
# because host gaps sit inside the event pair.

from __future__ import annotations

import os
import sys
import traceback

_LOG = os.environ.get(
    "B70_D2_LOG",
    "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d2-profile.log",
)
_WANTED = {1, 2, 3, 4}
_DONE_FLAG = os.environ.get(
    "B70_D2_DONE",
    "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d2-profile.done",
)

_state = {
    "installed": False,
    "hooked": False,
    "benching": False,
    "seen": set(),
    "layers": None,
    "regions": {},
    "counts": None,
}


def _log(msg: str) -> None:
    line = msg + "\n"
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        sys.stderr.write(line)


def _classify(name: str) -> str:
    n = name.lower()
    if "moe_forward" in n or "fused_moe" in n or "xpu_fused_moe" in n:
        return "moe"
    if "gdn_attention" in n or "qwen_gdn" in n:
        return "gdn_core"
    if "unified_attention" in n or "flash_attn" in n or "varlen" in n:
        return "full_attn"
    if "kv_cache" in n or "reshape_and_cache" in n:
        return "kv_update"
    if "rms_norm" in n or "gemma_rms" in n or "layernorm" in n:
        return "norm"
    if "embedding" in n:
        return "embed"
    if "aten.linear" in n or "addmm" in n or n.endswith("mm.default"):
        return "linear"
    if "topk" in n or "argmax" in n or "sampler" in n or "softmax" in n:
        return "router_sampler"
    if any(
        s in n
        for s in (
            "aten.add",
            "aten.mul",
            "aten.silu",
            "sigmoid",
            "aten.view",
            "reshape",
            "aten.empty",
            "aten.slice",
            "aten.cat",
            "aten.split",
            "aten.clone",
            "aten.copy",
            "aten.to.",
        )
    ):
        return "glue"
    return "other"


def _find_layers(root):
    import torch

    best = None
    for _name, mod in root.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 40:
            first = mod[0]
            if hasattr(first, "mlp") and (
                hasattr(first, "linear_attn") or hasattr(first, "self_attn")
            ):
                best = mod
                break
    return best


def _clone(obj):
    import torch

    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(_clone(x) for x in obj)
    if isinstance(obj, list):
        return [_clone(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    return obj


def _param_bytes(mod) -> tuple[int, str]:
    total = 0
    dtypes = []
    for p in mod.parameters(recurse=True):
        total += p.numel() * p.element_size()
        if p.dtype not in dtypes:
            dtypes.append(p.dtype)
    return total, ",".join(str(d) for d in dtypes)


def _active_bytes(name: str, stored: int) -> int:
    # Routed experts store all 256; a token reads top-8.
    if "experts" in name and "shared" not in name:
        return stored * 8 // 256
    return stored


def _replay(fn, nreplay: int) -> tuple[float, str]:
    import torch

    fn()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(nreplay):
        graph.replay()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / float(nreplay)
    del graph
    torch.xpu.empty_cache()
    return ms, "graph"


def _call(mod, args, kwargs):
    if kwargs:
        return mod(*args, **kwargs)
    return mod(*args)


def _bench_saved(t: int) -> None:
    import torch

    _state["benching"] = True
    try:
        regions = _state["regions"].get(t, {})
        _log(f"d2_bench_begin T={t} regions={sorted(regions)}")
        for name, payload in regions.items():
            mod = payload["mod"]
            args = payload["args"]
            kwargs = payload["kwargs"]
            stored, dtypes = _param_bytes(mod)
            active = _active_bytes(name, stored)
            try:
                ms, how = _replay(lambda: _call(mod, args, kwargs), 6)
            except Exception as exc:
                _log(f"d2_region T={t} name={name} FAIL {exc!r}")
                continue
            gbps = 0.0
            if ms > 0 and active > 0:
                gbps = (active / (ms * 1e-3)) / 1e9
            _log(
                f"d2_region T={t} name={name} ms={ms:.4f} via={how} "
                f"stored={stored} active={active} dtype={dtypes} "
                f"GB/s={gbps:.1f} roof%={100.0 * gbps / 608.0:.1f}"
            )
        torch.xpu.synchronize()
    finally:
        _state["benching"] = False


def _install_hooks(runner) -> None:
    if _state["hooked"]:
        return
    layers = _find_layers(runner.model)
    _state["layers"] = layers
    if layers is None:
        _log(f"d2_no_layers model={type(runner.model).__name__}")
        _state["hooked"] = True
        return
    _log(
        f"d2_layers n={len(layers)} layer0={type(layers[0]).__name__} "
        f"has_linear={hasattr(layers[0], 'linear_attn')} "
        f"has_full={hasattr(layers[0], 'self_attn')}"
    )
    # Parameter census once.
    gdn = layers[0]
    full = layers[3] if len(layers) > 3 else None
    for label, mod in (
        ("layer0", gdn),
        ("layer0.mlp", getattr(gdn, "mlp", None)),
        ("layer0.linear_attn", getattr(gdn, "linear_attn", None)),
        ("layer3", full),
        ("layer3.self_attn", getattr(full, "self_attn", None) if full else None),
        ("lm_head", getattr(runner.model, "lm_head", None)),
    ):
        if mod is None:
            continue
        stored, dtypes = _param_bytes(mod)
        _log(f"d2_params {label} bytes={stored} dtype={dtypes} type={type(mod).__name__}")

    targets = []

    def add(name, mod):
        if mod is None:
            return
        targets.append((name, mod))

    la = getattr(gdn, "linear_attn", None)
    if la is not None:
        add("gdn_in_qkvz", getattr(la, "in_proj_qkvz", None))
        add("gdn_in_ba", getattr(la, "in_proj_ba", None))
        add("gdn_out", getattr(la, "out_proj", None))
        # Whole GDN module minus the three projections is the recurrent core.
        add("gdn_core_module", la)
    mlp = getattr(gdn, "mlp", None)
    if mlp is not None:
        add("moe_experts", getattr(mlp, "experts", None))
        add("shared_expert", getattr(mlp, "shared_expert", None))
        add("router", getattr(mlp, "gate", None))
    add("norm_in", getattr(gdn, "input_layernorm", None))
    add("norm_post", getattr(gdn, "post_attention_layernorm", None))
    if full is not None:
        attn = getattr(full, "self_attn", None)
        if attn is not None:
            add("attn_qkv", getattr(attn, "qkv_proj", None))
            add("attn_o", getattr(attn, "o_proj", None))
            add("attn_core", getattr(attn, "attn", None))
    # lm_head may live on the causal LM, not the outer VL wrapper.
    for obj in (runner.model, getattr(runner.model, "language_model", None)):
        if obj is None:
            continue
        if hasattr(obj, "compute_logits"):
            add("compute_logits", obj.compute_logits)
        if hasattr(obj, "lm_head"):
            add("lm_head", obj.lm_head)

    def make_hook(name, mod):
        def hook(*args, **kwargs):
            if _state["benching"]:
                return
            # pre-hook vs function wrapper: both go through _save.
            return None

        return hook

    # nn.Module hooks. compute_logits is a bound method; wrap it separately.
    for name, mod in targets:
        if name == "compute_logits":
            continue
        if not hasattr(mod, "register_forward_pre_hook"):
            continue

        def _pre(mod, args, kwargs=None, name=name):
            if _state["benching"]:
                return
            t = _state.get("cur_t")
            if t not in _WANTED:
                return
            bucket = _state["regions"].setdefault(t, {})
            if name in bucket:
                return
            bucket[name] = {
                "mod": mod,
                "args": _clone(args),
                "kwargs": _clone(kwargs or {}),
            }

        try:
            mod.register_forward_pre_hook(_pre, with_kwargs=True)
        except TypeError:
            def _pre_nokw(mod, args, name=name):
                if _state["benching"]:
                    return
                t = _state.get("cur_t")
                if t not in _WANTED:
                    return
                bucket = _state["regions"].setdefault(t, {})
                if name in bucket:
                    return
                bucket[name] = {"mod": mod, "args": _clone(args), "kwargs": {}}

            mod.register_forward_pre_hook(_pre_nokw)
        _log(f"d2_hook {name} type={type(mod).__name__}")
    _state["hooked"] = True
    _state["targets_fn"] = targets


def _dump_counts(t: int, counts: dict) -> None:
    buckets: dict[str, list] = {}
    for name, n in counts.items():
        buckets.setdefault(_classify(name), []).append((n, name))
    _log(f"d2_counts_begin T={t} ops={len(counts)} calls={sum(counts.values())}")
    for cls, items in sorted(buckets.items(), key=lambda kv: -sum(n for n, _ in kv[1])):
        calls = sum(n for n, _ in items)
        _log(f"d2_class T={t} class={cls} calls={calls} kinds={len(items)}")
        items.sort(reverse=True)
        for n, name in items[:8]:
            _log(f"d2_op T={t} class={cls} n={n} {name}")
    _log(f"d2_counts_end T={t}")


def _maybe_finish() -> None:
    if not _WANTED.issubset(_state["seen"]):
        missing = sorted(_WANTED - _state["seen"])
        _log(f"d2_waiting missing={missing}")
        return
    _log("d2_profile_done")
    try:
        with open(_DONE_FLAG, "w", encoding="utf-8") as fh:
            fh.write("done\n")
    except OSError as exc:
        _log(f"d2_done_flag_fail {exc!r}")


def install() -> None:
    if _state["installed"]:
        return
    _state["installed"] = True

    # Import hook: GPUModelRunner may not be imported yet.
    import importlib

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            return None

    def _patch() -> None:
        try:
            from vllm.v1.worker import gpu_model_runner as gmr
        except Exception:
            return
        cls = gmr.GPUModelRunner
        if getattr(cls.execute_model, "_d2_wrapped", False):
            return
        orig = cls.execute_model

        def execute_model(self, scheduler_output, *args, **kwargs):
            if kwargs.get("dummy_run", False) or _state["benching"]:
                return orig(self, scheduler_output, *args, **kwargs)
            n_tok = getattr(scheduler_output, "total_num_scheduled_tokens", -1)
            _state["cur_t"] = n_tok
            try:
                _install_hooks(self)
            except Exception:
                _log("d2_hook_fail\n" + traceback.format_exc())
            want = n_tok in _WANTED and n_tok not in _state["seen"]
            counts = None
            if want:
                from torch.utils._python_dispatch import TorchDispatchMode

                class _Count(TorchDispatchMode):
                    def __init__(self):
                        super().__init__()
                        self.counts = {}

                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        key = str(func)
                        self.counts[key] = self.counts.get(key, 0) + 1
                        return func(*args, **(kwargs or {}))

                counts = _Count()
                counts.__enter__()
                _log(f"d2_step_armed T={n_tok}")
            try:
                out = orig(self, scheduler_output, *args, **kwargs)
            finally:
                if counts is not None:
                    counts.__exit__(None, None, None)
            if want and counts is not None:
                try:
                    _dump_counts(n_tok, counts.counts)
                    _bench_saved(n_tok)
                    _state["seen"].add(n_tok)
                    _maybe_finish()
                except Exception:
                    _log("d2_bench_fail\n" + traceback.format_exc())
            return out

        execute_model._d2_wrapped = True  # type: ignore[attr-defined]
        cls.execute_model = execute_model
        _log("d2_profile_installed")

    # Patch now if vLLM is already imported, and on every later import.
    try:
        _patch()
    except Exception:
        _log("d2_patch_now_fail\n" + traceback.format_exc())

    real_import = importlib.import_module

    def _import(name, package=None):
        mod = real_import(name, package)
        if name == "vllm.v1.worker.gpu_model_runner" or name.startswith(
            "vllm.v1.worker"
        ):
            try:
                _patch()
            except Exception:
                _log("d2_patch_import_fail\n" + traceback.format_exc())
        return mod

    # sitecustomize cannot wrap importlib for the worker that is already
    # inside vLLM. The engine worker imports sitecustomize first, then
    # vLLM, so a meta hook on the module is the reliable path.
    class _Loader:
        pass

    if "vllm.v1.worker.gpu_model_runner" in sys.modules:
        _patch()
    else:
        import importlib.abc

        class _ExecHook(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname != "vllm.v1.worker.gpu_model_runner":
                    return None
                sys.meta_path.remove(self)
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec is None or spec.loader is None:
                    return spec
                orig_exec = spec.loader.exec_module

                def exec_module(module, _orig=orig_exec):
                    _orig(module)
                    try:
                        _patch()
                    except Exception:
                        _log("d2_exec_patch_fail\n" + traceback.format_exc())

                spec.loader.exec_module = exec_module  # type: ignore[method-assign]
                return spec

        sys.meta_path.insert(0, _ExecHook())
        _log("d2_finder_installed")
