"""XPU-event brackets for one real prefill forward.

Installed from the plugin register hook when B70_LOWBIT_FWD_SPLIT=1.
Not Kineto. Events are recorded on the forward and resolved with one
synchronize after execute_model returns. Decode steps (few tokens) are
not instrumented and are not synchronized here.

The log is B70_LOWBIT_FWD_LOG. Each large step appends one block:
token count, dtype, attention and GDN config, then milliseconds for
linear_attn, self_attn, mlp, router, norms, and the lowbit unpack/bmm
spans inside the MoE.
"""

from __future__ import annotations

import os
import traceback

_LOG = os.environ.get(
    "B70_LOWBIT_FWD_LOG",
    "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/e2-fwd-split.log",
)
_MIN_TOKENS = int(os.environ.get("B70_LOWBIT_FWD_MIN_TOKENS", "512"))
_INSTALLED = False
_STEP = 0


def _log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    try:
        with open(_LOG, "a", encoding="ascii") as fh:
            fh.write(line)
    except OSError:
        pass


def _find_layers(root):
    import torch

    best = None
    for _name, mod in root.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
            first = mod[0]
            if hasattr(first, "mlp") and (
                hasattr(first, "linear_attn") or hasattr(first, "self_attn")
            ):
                best = mod
                break
    return best


def _config_lines(runner) -> list[str]:
    lines = []
    vc = getattr(runner, "vllm_config", None)
    if vc is None:
        return ["config missing"]
    model = getattr(vc, "model_config", None)
    sched = getattr(vc, "scheduler_config", None)
    if model is not None:
        lines.append(f"dtype {model.dtype}")
        lines.append(f"max_model_len {model.max_model_len}")
    if sched is not None:
        lines.append(f"max_num_batched_tokens {sched.max_num_batched_tokens}")
        lines.append(f"max_num_seqs {sched.max_num_seqs}")
    attn = getattr(vc, "attention_config", None)
    if attn is not None:
        backend = getattr(attn, "backend", None)
        lines.append(f"attention_backend {backend}")
    extra = getattr(vc, "additional_config", None)
    lines.append(f"additional_config {extra}")
    lines.append(f"prefill_flag {os.environ.get('B70_LOWBIT_PREFILL', '')}")
    lines.append(f"gemv_rows {os.environ.get('B70_LOWBIT_GEMV_ROWS', '')}")
    return lines


def _bucket_for(layer_attr: str) -> str:
    if layer_attr == "linear_attn":
        return "linear_attn"
    if layer_attr == "self_attn":
        return "self_attn"
    if layer_attr == "mlp":
        return "mlp"
    if layer_attr in ("input_layernorm", "post_attention_layernorm"):
        return "norm"
    if layer_attr == "gate":
        return "router"
    if layer_attr == "experts":
        return "experts"
    if layer_attr in ("shared_expert", "shared_expert_gate"):
        return "shared_expert"
    return layer_attr


def _attach(layers):
    import torch

    spans = []
    handles = []
    seen = {"dtype": None, "shape": None}

    def _pre(mod, args, _spans=spans):
        if seen["dtype"] is None and args and torch.is_tensor(args[0]):
            seen["dtype"] = str(args[0].dtype)
            seen["shape"] = tuple(args[0].shape)
        ev = torch.xpu.Event(enable_timing=True)
        ev.record()
        mod._b70_t0 = ev

    def _post(mod, args, output, bucket, _spans=spans):
        del args, output
        ev = torch.xpu.Event(enable_timing=True)
        ev.record()
        start = getattr(mod, "_b70_t0", None)
        if start is not None:
            _spans.append((bucket, start, ev))

    for layer in layers:
        for attr in (
            "input_layernorm",
            "post_attention_layernorm",
            "linear_attn",
            "self_attn",
            "mlp",
        ):
            mod = getattr(layer, attr, None)
            if mod is None:
                continue
            bucket = _bucket_for(attr)
            handles.append(mod.register_forward_pre_hook(_pre))
            handles.append(
                mod.register_forward_hook(
                    lambda m, a, o, b=bucket: _post(m, a, o, b)
                )
            )
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for attr in ("gate", "experts", "shared_expert", "shared_expert_gate"):
            mod = getattr(mlp, attr, None)
            if mod is None:
                continue
            bucket = _bucket_for(attr)
            handles.append(mod.register_forward_pre_hook(_pre))
            handles.append(
                mod.register_forward_hook(
                    lambda m, a, o, b=bucket: _post(m, a, o, b)
                )
            )
    return spans, handles, seen


def _sum_spans(spans) -> dict:
    acc = {}
    counts = {}
    for name, start, end in spans:
        acc[name] = acc.get(name, 0.0) + float(start.elapsed_time(end))
        counts[name] = counts.get(name, 0) + 1
    return acc, counts


class _Count:
    def __init__(self) -> None:
        self.item = 0
        self.tolist = 0
        self.sync = 0


def _patch_syncs(count: _Count):
    import torch

    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_sync = torch.xpu.synchronize

    def item(self):
        count.item += 1
        return orig_item(self)

    def tolist(self, *args, **kwargs):
        count.tolist += 1
        return orig_tolist(self, *args, **kwargs)

    def synchronize(*args, **kwargs):
        count.sync += 1
        return orig_sync(*args, **kwargs)

    torch.Tensor.item = item
    torch.Tensor.tolist = tolist
    torch.xpu.synchronize = synchronize
    return orig_item, orig_tolist, orig_sync


def _restore_syncs(orig) -> None:
    import torch

    torch.Tensor.item, torch.Tensor.tolist, torch.xpu.synchronize = orig


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    orig = GPUModelRunner.execute_model

    def wrapped(self, scheduler_output, *args, **kwargs):
        global _STEP
        ntok = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        if ntok < _MIN_TOKENS:
            return orig(self, scheduler_output, *args, **kwargs)
        _STEP += 1
        step = _STEP
        import torch

        from b70_lowbit_xpu.moe import drain_timer_ms

        if step == 1:
            _log("E2 fwd_split")
            for line in _config_lines(self):
                _log(line)
        layers = _find_layers(self.model)
        spans: list = []
        handles = []
        seen = {"dtype": None, "shape": None}
        if layers is not None:
            spans, handles, seen = _attach(layers)
        else:
            _log(f"step {step} tokens {ntok} NO_LAYERS")
        count = _Count()
        patched = _patch_syncs(count)
        wall0 = torch.xpu.Event(enable_timing=True)
        wall1 = torch.xpu.Event(enable_timing=True)
        try:
            wall0.record()
            out = orig(self, scheduler_output, *args, **kwargs)
            wall1.record()
            torch.xpu.synchronize()
        finally:
            _restore_syncs(patched)
            for handle in handles:
                handle.remove()
        acc, counts = _sum_spans(spans)
        lowbit = drain_timer_ms()
        _log(
            f"step {step} tokens {ntok} "
            f"hidden_dtype {seen['dtype']} hidden_shape {seen['shape']}"
        )
        _log(f"step {step} forward_ms {wall0.elapsed_time(wall1):.2f}")
        for name in (
            "linear_attn",
            "self_attn",
            "mlp",
            "router",
            "experts",
            "shared_expert",
            "norm",
        ):
            if name in acc:
                _log(
                    f"step {step} {name} ms {acc[name]:.2f} n {counts[name]}"
                )
        for name in sorted(lowbit):
            _log(f"step {step} lowbit_{name} ms {lowbit[name]:.2f}")
        _log(
            f"step {step} host_sync item {count.item} "
            f"tolist {count.tolist} synchronize {count.sync}"
        )
        other = sorted(k for k in acc if k not in (
            "linear_attn",
            "self_attn",
            "mlp",
            "router",
            "experts",
            "shared_expert",
            "norm",
        ))
        for name in other:
            _log(f"step {step} {name} ms {acc[name]:.2f} n {counts[name]}")
        return out

    GPUModelRunner.execute_model = wrapped
    _INSTALLED = True
    _log("fwd_split installed")


def _guard() -> None:
    """Import-time install is done by the plugin. This catches a late import."""
    del _guard


# Keep traceback available for a manual debug call.
_ = traceback
