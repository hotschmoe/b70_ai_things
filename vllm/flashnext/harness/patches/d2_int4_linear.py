# Decode-only int4 GEMV for the fp16 trunk and lm_head.
# Installed from sitecustomize when B70_D2_INT4_LINEAR=1.
#
# Ornith GPTQ stores routed experts as int4 and the trunk (GDN/attention
# projections, shared expert, router, lm_head) as fp16. b70_lowbit gemv
# reads an RTN int4 plane. M=1..8 and M*K bf16 must fit SLM
# (8*2560 bf16 values). Larger tokens, including prefill and the
# capture-size-8 graph for K=4096, keep the fp16 weight path.
# Planes are packed once at the end of load_model, before graph capture.
# fp16 weights stay resident so the fallback path is the original GEMM.

from __future__ import annotations

import os
import sys
import traceback

_LOG = os.environ.get(
    "B70_D2_INT4_LOG",
    "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d2-int4.log",
)
_SLM_BYTES = 8 * 2560 * 2
_MIN_N = 32
_installed = False
_wrapped = False
_GEMV = None


def _log(msg: str) -> None:
    line = msg + "\n"
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass
    sys.stderr.write(line)
    sys.stderr.flush()


def _plugin_path() -> str:
    return os.environ.get(
        "B70_LOWBIT_PYTHONPATH",
        "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_lowbit_xpu",
    )


def _ops():
    path = _plugin_path()
    if path not in sys.path:
        sys.path.insert(0, path)
    from b70_lowbit_xpu import ops
    from b70_lowbit_xpu.format import quantize_and_pack

    return ops, quantize_and_pack


def _pack_weight(weight, quantize_and_pack):
    import torch

    planes2 = []
    planes_hi = []
    scales = []
    rows = 4096
    n = weight.shape[0]
    for start in range(0, n, rows):
        chunk = weight[start : start + rows]
        plane2, plane_hi, scale = quantize_and_pack(chunk.detach().float(), 4)
        planes2.append(plane2)
        planes_hi.append(plane_hi)
        scales.append(scale)
        del chunk
    plane2 = torch.cat(planes2, dim=0).contiguous()
    scale = torch.cat(scales, dim=0).contiguous()
    if any(t is None for t in planes_hi):
        plane_hi = torch.empty(0, dtype=torch.uint8, device=plane2.device)
    else:
        plane_hi = torch.cat(planes_hi, dim=0).contiguous()
    return plane2, plane_hi, scale


def _eligible(name: str, module) -> bool:
    import torch

    if "embed_tokens" in name or ".mtp." in name or name.startswith("mtp."):
        return False
    weight = getattr(module, "weight", None)
    if weight is None or not hasattr(weight, "ndim"):
        return False
    if weight.ndim != 2 or weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return False
    n, k = weight.shape
    if n < _MIN_N or k < 128 or k % 128 != 0:
        return False
    return True


def _pack_model(model) -> None:
    import torch

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    ops, quantize_and_pack = _ops()
    del ops
    packed = 0
    total = 0
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)):
            continue
        if getattr(module, "_b70_int4", False):
            continue
        if not _eligible(name, module):
            continue
        weight = module.weight.data
        try:
            plane2, plane_hi, scale = _pack_weight(weight, quantize_and_pack)
        except Exception:
            _log(f"d2_int4_pack_fail {name}\n" + traceback.format_exc())
            continue
        if plane2.device != weight.device:
            plane2 = plane2.to(weight.device)
            scale = scale.to(weight.device)
            plane_hi = plane_hi.to(weight.device)
        module.register_buffer("b70_plane2", plane2, persistent=False)
        module.register_buffer("b70_plane_hi", plane_hi, persistent=False)
        module.register_buffer("b70_scale", scale, persistent=False)
        module._b70_int4 = True
        nbytes = plane2.numel() + plane_hi.numel() + scale.numel() * scale.element_size()
        total += nbytes
        packed += 1
        _log(
            f"d2_int4_pack {name} N={weight.shape[0]} K={weight.shape[1]} "
            f"dtype={weight.dtype} plane_bytes={nbytes}"
        )
        torch.xpu.synchronize()
    _log(f"d2_int4_packed count={packed} plane_bytes={total}")


def _gemv():
    global _GEMV
    if _GEMV is None:
        ops, _pack = _ops()
        _GEMV = ops.gemv
    return _GEMV


def _try_gemv(layer, x, bias):
    import torch

    if not getattr(layer, "_b70_int4", False):
        return None
    if x.shape[-1] != layer.weight.shape[1]:
        return None
    flat = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
    m, k = flat.shape
    if not (1 <= m <= 8 and m * k * 2 <= _SLM_BYTES):
        return None
    xb = flat if flat.dtype is torch.bfloat16 else flat.to(torch.bfloat16)
    if not xb.is_contiguous():
        xb = xb.contiguous()
    hi = layer.b70_plane_hi
    y = _gemv()(
        xb,
        layer.b70_plane2,
        None if hi.numel() == 0 else hi,
        layer.b70_scale,
        4,
    )
    if y.dtype != x.dtype:
        y = y.to(dtype=x.dtype)
    if bias is not None:
        y = y + bias
    if x.ndim != 2:
        y = y.view(*x.shape[:-1], y.shape[-1])
    return y


def _wrap_apply() -> None:
    global _wrapped
    if _wrapped:
        return
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    def _wrap(cls):
        orig = cls.apply
        if getattr(orig, "_d2_int4_wrapped", False):
            return

        def apply(self, layer, x, bias=None):
            y = _try_gemv(layer, x, bias)
            if y is None:
                return orig(self, layer, x, bias)
            return y

        apply._d2_int4_wrapped = True  # type: ignore[attr-defined]
        cls.apply = apply

    _wrap(UnquantizedLinearMethod)
    _wrap(UnquantizedEmbeddingMethod)
    _wrapped = True
    _log("d2_int4_apply_wrapped")


def apply(runner) -> None:
    model = getattr(runner, "model", None)
    if model is None:
        _log("d2_int4_no_model")
        return
    _wrap_apply()
    _pack_model(model)
    _gemv()


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    _MODULES = (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    )

    def _wrap(cls) -> None:
        current = cls.load_model
        if getattr(current, "_d2_int4_wrapped", False):
            return
        orig = current

        def load_model(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            try:
                apply(self)
            except Exception:
                _log("d2_int4_fail\n" + traceback.format_exc())
            return out

        load_model._d2_int4_wrapped = True  # type: ignore[attr-defined]
        cls.load_model = load_model
        _log(f"d2_int4_installed cls={cls.__module__}.{cls.__name__}")

    def _patch() -> None:
        for name in _MODULES:
            mod = sys.modules.get(name)
            cls = getattr(mod, "GPUModelRunner", None) if mod is not None else None
            if cls is not None:
                _wrap(cls)

    try:
        _patch()
    except Exception:
        _log("d2_int4_patch_now_fail\n" + traceback.format_exc())

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
                    _log("d2_int4_patch_exec_fail\n" + traceback.format_exc())

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _ExecHook())
