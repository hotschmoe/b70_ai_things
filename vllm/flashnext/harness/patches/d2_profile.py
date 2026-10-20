# In-graph device-time breakdown for one Ornith decode step.
# Installed from sitecustomize when B70_D2_PROFILE=1.
#
# Graph replay hides per-kernel times. This records eager op counts, then
# re-captures each region (GDN proj, GDN core, attention, MoE, shared
# expert, norms, lm_head) as its own XPU graph and times the replay.
# That is the same device-time method as D1's 11.4 ms full-graph sample.
# Replay runs under the forward context saved during the real step.
# Replaying after execute_model returns fails for MoE, GDN, and attention
# because those ops call get_forward_context().
# Op event brackets are logged separately. They include host gaps inside
# one op, so a linear bracket is the calibration against its graph time.

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


def _event_once(fn) -> tuple[float, str]:
    """Device bracket around one eager call. Host time before the launch
    sits inside the pair, so this is a fallback, not the graph number."""
    import torch

    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), "event"


def _time_logits(t: int) -> None:
    hs = _state.get("logits_hs")
    orig = _state.get("logits_orig")
    model = _state.get("logits_model")
    _state["logits_hs"] = None
    _state["logits_orig"] = None
    _state["logits_model"] = None
    if hs is None or orig is None:
        _log(f"d2_region T={t} name=compute_logits SKIP no_call")
        return
    head = getattr(model, "lm_head", None) if model is not None else None
    if head is not None:
        stored, dtypes = _param_bytes(head)
    else:
        stored, dtypes = 0, ""
    _state["benching"] = True
    try:
        try:
            ms, how = _replay(lambda: orig(hs), 4)
        except Exception as exc:
            _log(f"d2_region T={t} name=compute_logits graph_fail {exc!r}")
            try:
                ms, how = _event_once(lambda: orig(hs))
            except Exception as exc2:
                _log(f"d2_region T={t} name=compute_logits FAIL {exc2!r}")
                return
        gbps = (stored / (ms * 1e-3)) / 1e9 if ms > 0 and stored else 0.0
        _log(
            f"d2_region T={t} name=compute_logits ms={ms:.4f} via={how} "
            f"stored={stored} active={stored} dtype={dtypes} "
            f"GB/s={gbps:.1f} roof%={100.0 * gbps / 608.0:.1f}"
        )
    finally:
        _state["benching"] = False


def _call(mod, args, kwargs):
    if kwargs:
        return mod(*args, **kwargs)
    return mod(*args)


def _bench_saved(t: int) -> None:
    import contextlib

    import torch

    from vllm.forward_context import override_forward_context

    _state["benching"] = True
    try:
        regions = _state["regions"].get(t, {})
        ctx = _state.get("ctx")
        _log(
            f"d2_bench_begin T={t} regions={sorted(regions)} "
            f"ctx={type(ctx).__name__ if ctx is not None else None}"
        )
        if ctx is not None:
            try:
                ctx.moe_layer_index = 0
            except Exception:
                pass
            cm = override_forward_context(ctx)
        else:
            cm = contextlib.nullcontext()
        with cm:
            for name, payload in regions.items():
                mod = payload["mod"]
                args = payload["args"]
                kwargs = payload["kwargs"]
                stored, dtypes = _param_bytes(mod)
                active = _active_bytes(name, stored)

                def _run(mod=mod, args=args, kwargs=kwargs, ctx=ctx):
                    if ctx is not None:
                        try:
                            ctx.moe_layer_index = 0
                        except Exception:
                            pass
                    return _call(mod, args, kwargs)

                how = "graph"
                try:
                    ms, how = _replay(_run, 4)
                except Exception as exc:
                    _log(f"d2_region T={t} name={name} graph_fail {exc!r}")
                    try:
                        if ctx is not None:
                            ctx.moe_layer_index = 0
                        ms, how = _event_once(_run)
                    except Exception as exc2:
                        _log(f"d2_region T={t} name={name} FAIL {exc2!r}")
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
            if _state.get("ctx") is None:
                try:
                    from vllm.forward_context import get_forward_context

                    _state["ctx"] = get_forward_context()
                    _log("d2_ctx_saved")
                except Exception as exc:
                    _log(f"d2_ctx_missing {exc!r}")
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

    # V1 runner and the V2 runner selected by XPUWorker ("Using V2 Model Runner").
    _MODULES = (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    )

    def _wrap(cls) -> None:
        if getattr(cls.execute_model, "_d2_wrapped", False):
            return
        orig = cls.execute_model

        def execute_model(self, scheduler_output, *args, **kwargs):
            dummy = kwargs.get("dummy_run", False)
            if not dummy and len(args) >= 2 and args[1] is True:
                dummy = True
            if dummy or _state["benching"]:
                return orig(self, scheduler_output, *args, **kwargs)
            n_tok = getattr(scheduler_output, "total_num_scheduled_tokens", -1)
            _state["cur_t"] = n_tok
            _state["steps"] = _state.get("steps", 0) + 1
            if _state["steps"] <= 12:
                _log(f"d2_step n={_state['steps']} T={n_tok}")
            try:
                _install_hooks(self)
            except Exception:
                _log("d2_hook_fail\n" + traceback.format_exc())
            want = n_tok in _WANTED and n_tok not in _state["seen"]
            counts = None
            logits_restore = None
            if want:
                # Drop the previous step's context so the pre-hook saves this T.
                _state["ctx"] = None
                _state["logits_hs"] = None
                from torch.utils._python_dispatch import TorchDispatchMode

                _event_ops = (
                    "moe_forward_shared",
                    "gdn_attention_core_xpu",
                    "unified_attention_with_output",
                    "unified_kv_cache_update",
                )

                class _Count(TorchDispatchMode):
                    def __init__(self):
                        super().__init__()
                        self.counts = {}
                        self.events = {}
                        self.linear_ms = None

                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        import torch

                        key = str(func)
                        self.counts[key] = self.counts.get(key, 0) + 1
                        kw = kwargs or {}
                        want_event = any(s in key for s in _event_ops)
                        want_linear = self.linear_ms is None and "aten.linear" in key
                        if not want_event and not want_linear:
                            return func(*args, **kw)
                        torch.xpu.synchronize()
                        start = torch.xpu.Event(enable_timing=True)
                        end = torch.xpu.Event(enable_timing=True)
                        start.record()
                        out_op = func(*args, **kw)
                        end.record()
                        end.synchronize()
                        ms = start.elapsed_time(end)
                        if want_linear and self.linear_ms is None:
                            self.linear_ms = ms
                        if want_event:
                            self.events.setdefault(key, []).append(ms)
                        return out_op

                counts = _Count()
                counts.__enter__()
                model = getattr(self, "model", None)
                orig_cl = getattr(model, "compute_logits", None) if model is not None else None
                if orig_cl is not None:
                    def wrapped_cl(hidden_states, *a, _orig=orig_cl, **k):
                        result = _orig(hidden_states, *a, **k)
                        if (
                            hidden_states is not None
                            and _state.get("logits_hs") is None
                        ):
                            _state["logits_hs"] = hidden_states.detach().clone()
                            _state["logits_orig"] = _orig
                            _state["logits_model"] = model
                        return result

                    model.compute_logits = wrapped_cl
                    logits_restore = (model, orig_cl)
                _log(f"d2_step_armed T={n_tok}")
            try:
                out = orig(self, scheduler_output, *args, **kwargs)
            finally:
                if counts is not None:
                    counts.__exit__(None, None, None)
                if logits_restore is not None:
                    logits_restore[0].compute_logits = logits_restore[1]
            if want and counts is not None:
                try:
                    _dump_counts(n_tok, counts.counts)
                    if counts.linear_ms is not None:
                        _log(
                            f"d2_op_event T={n_tok} op=aten.linear.first "
                            f"ms={counts.linear_ms:.4f}"
                        )
                    for key, arr in counts.events.items():
                        total = float(sum(arr))
                        _log(
                            f"d2_op_event T={n_tok} op={key} n={len(arr)} "
                            f"sum_ms={total:.3f} mean_ms={total / len(arr):.4f} "
                            f"min_ms={min(arr):.4f} max_ms={max(arr):.4f}"
                        )
                    _time_logits(n_tok)
                    _bench_saved(n_tok)
                    _state["seen"].add(n_tok)
                    _maybe_finish()
                except Exception:
                    _log("d2_bench_fail\n" + traceback.format_exc())
            return out

        execute_model._d2_wrapped = True  # type: ignore[attr-defined]
        cls.execute_model = execute_model
        _log(f"d2_profile_installed cls={cls.__module__}.{cls.__name__}")

    def _patch() -> None:
        for name in _MODULES:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            cls = getattr(mod, "GPUModelRunner", None)
            if cls is not None:
                _wrap(cls)

    try:
        _patch()
    except Exception:
        _log("d2_patch_now_fail\n" + traceback.format_exc())

    import importlib.abc
    import importlib.machinery

    class _ExecHook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _MODULES:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                try:
                    cls = getattr(module, "GPUModelRunner", None)
                    if cls is not None:
                        _wrap(cls)
                except Exception:
                    _log("d2_exec_patch_fail\n" + traceback.format_exc())

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _ExecHook())
    _log("d2_finder_installed")
