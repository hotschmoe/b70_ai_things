# Replace the loaded Ornith GPTQ MTP head with the Shisa code-tuned head.
# Installed from sitecustomize when B70_D2_SHISA_MTP=1.
#
# The GPTQ checkpoint keeps the native 785-tensor BF16 MTP
# (quantization_config.dynamic excludes "*:mtp" via "-:.*mtp.*"). vLLM
# then forces quant_config=None for those layers, so the drafter is
# unquantized. The Shisa head is the 19-tensor fused BF16 file in
# bf16-mtp-shisa/model-mtp.safetensors. Trunk norms match across the two
# checkpoints; the MTP tensors do not. This overlay copies the 19 tensors
# onto the drafter after load_model and before graph capture.
#
# GemmaRMSNorm stores the residual (runtime does x * (1 + w)). Checkpoint
# values are copied as stored. lm_head and embed_tokens stay the target
# model's tied weights.

from __future__ import annotations

import os
import sys
import traceback

_LOG = os.environ.get(
    "B70_D2_SHISA_LOG",
    "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d2-shisa.log",
)
_PATH = os.environ.get(
    "B70_D2_SHISA_PATH",
    "/mnt/vm_8tb/github/b70_ai_things/models/files/"
    "ornith-1.5-35b-a3b/bf16-mtp-shisa/model-mtp.safetensors",
)
_LAYER = "mtp.layers.0"
_installed = False


def _log(msg: str) -> None:
    line = msg + "\n"
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass
    sys.stderr.write(line)
    sys.stderr.flush()


def _load_tensors():
    from safetensors import safe_open

    out = {}
    with safe_open(_PATH, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            out[key] = handle.get_tensor(key)
    return out


def _with_fused(raw):
    import torch

    src = dict(raw)
    q = src[f"{_LAYER}.self_attn.q_proj.weight"]
    k = src[f"{_LAYER}.self_attn.k_proj.weight"]
    v = src[f"{_LAYER}.self_attn.v_proj.weight"]
    src[f"{_LAYER}.self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)
    gate = src[f"{_LAYER}.mlp.shared_expert.gate_proj.weight"]
    up = src[f"{_LAYER}.mlp.shared_expert.up_proj.weight"]
    src[f"{_LAYER}.mlp.shared_expert.gate_up_proj.weight"] = torch.cat(
        [gate, up], dim=0
    )
    src["__w13__"] = src[f"{_LAYER}.mlp.experts.gate_up_proj"]
    src["__w2__"] = src[f"{_LAYER}.mlp.experts.down_proj"]
    return src


def _ckpt_key(name: str) -> str:
    if ".model." in name:
        suffix = name.split(".model.")[-1]
        if suffix.startswith("model."):
            suffix = suffix[len("model.") :]
        return "mtp." + suffix
    if name.startswith("model."):
        return "mtp." + name[len("model.") :]
    return "mtp." + name


def _candidate(name: str, src):
    if "embed_tokens" in name or "lm_head" in name:
        return None, "skip-embed-or-lmhead"
    key = _ckpt_key(name)
    if key in src:
        return src[key], key
    experts = "experts" in name and "shared_expert" not in name
    if experts and (name.endswith("w13_weight") or name.endswith("gate_up_proj")):
        return src["__w13__"], "experts.gate_up_proj"
    if experts and (name.endswith("w2_weight") or name.endswith("down_proj")):
        return src["__w2__"], "experts.down_proj"
    return None, "no-key"


def _align(src, shape):
    import torch

    if tuple(src.shape) == tuple(shape):
        return src, "exact"
    if src.ndim == 2 and src.T.shape == torch.Size(shape):
        return src.T.contiguous(), "transpose"
    if src.ndim == 3 and src.transpose(1, 2).shape == torch.Size(shape):
        return src.transpose(1, 2).contiguous(), "transpose12"
    # Routed experts plus one fused shared slot.
    if (
        src.ndim == 3
        and src.shape[0] + 1 == shape[0]
        and tuple(src.shape[1:]) == tuple(shape[1:])
    ):
        return src, "routed-prefix"
    return None, f"shape src={tuple(src.shape)} dst={tuple(shape)}"


def _cosine(a, b) -> float:
    import torch

    n = min(a.numel(), b.numel(), 8192)
    if n == 0:
        return float("nan")
    aa = a.detach().flatten()[:n].float().cpu()
    bb = b.detach().flatten()[:n].float().cpu()
    denom = aa.norm() * bb.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(aa, bb) / denom)


def apply(runner) -> None:
    import torch

    drafter = getattr(runner, "drafter", None)
    model = getattr(drafter, "model", None)
    if model is None:
        _log("d2_shisa_no_drafter")
        return
    _log(f"d2_shisa_load path={_PATH}")
    raw = _load_tensors()
    _log("d2_shisa_keys " + ",".join(sorted(raw)))
    src = _with_fused(raw)
    used = set()
    copied = 0
    skipped = 0
    for name, param in model.named_parameters():
        _log(
            f"d2_shisa_param {name} shape={tuple(param.shape)} "
            f"dtype={param.dtype}"
        )
        tensor, how = _candidate(name, src)
        if tensor is None:
            skipped += 1
            _log(f"d2_shisa_skip {name} {how}")
            continue
        aligned, align = _align(tensor, tuple(param.shape))
        if aligned is None:
            skipped += 1
            _log(f"d2_shisa_skip {name} from={how} {align}")
            continue
        if align == "routed-prefix":
            # Caller asked not to invent a fused-shared layout. Leave it.
            skipped += 1
            _log(f"d2_shisa_skip {name} from={how} routed-prefix-unfilled")
            continue
        before = _cosine(param.data, aligned.to(dtype=param.dtype))
        with torch.no_grad():
            param.data.copy_(aligned.to(device=param.device, dtype=param.dtype))
        after_n = min(param.numel(), 4096)
        fingerprint = float(
            param.data.detach().flatten()[:after_n].float().abs().sum().cpu()
        )
        copied += 1
        used.add(how)
        _log(
            f"d2_shisa_copy {name} from={how} align={align} "
            f"cos_before={before:.4f} fp_abs_sum={fingerprint:.6g}"
        )
    missing = sorted(k for k in raw if k not in used and not k.startswith("__"))
    # Fused views consume the pieces; mark those pieces used when the
    # fused parameter was copied.
    if f"{_LAYER}.self_attn.qkv_proj.weight" in used:
        for piece in ("q_proj", "k_proj", "v_proj"):
            missing = [k for k in missing if not k.endswith(f"self_attn.{piece}.weight")]
    if f"{_LAYER}.mlp.shared_expert.gate_up_proj.weight" in used:
        missing = [
            k
            for k in missing
            if not k.endswith("shared_expert.gate_proj.weight")
            and not k.endswith("shared_expert.up_proj.weight")
        ]
    if "experts.gate_up_proj" in used:
        missing = [k for k in missing if not k.endswith("experts.gate_up_proj")]
    if "experts.down_proj" in used:
        missing = [k for k in missing if not k.endswith("experts.down_proj")]
    _log(
        f"d2_shisa_applied copied={copied} skipped={skipped} "
        f"unused={missing}"
    )


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    def _patch() -> None:
        from vllm.v1.worker import gpu_model_runner as gmr

        cls = gmr.GPUModelRunner
        current = cls.load_model
        if getattr(current, "_d2_shisa_wrapped", False):
            return
        orig = current

        def load_model(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            try:
                apply(self)
            except Exception:
                _log("d2_shisa_fail\n" + traceback.format_exc())
            return out

        load_model._d2_shisa_wrapped = True  # type: ignore[attr-defined]
        cls.load_model = load_model
        _log("d2_shisa_installed")

    try:
        _patch()
    except Exception:
        _log("d2_shisa_patch_now_fail\n" + traceback.format_exc())

    import importlib
    import importlib.abc
    import importlib.machinery

    if "vllm.v1.worker.gpu_model_runner" in sys.modules:
        try:
            _patch()
        except Exception:
            _log("d2_shisa_patch_imported_fail\n" + traceback.format_exc())
        return

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
                    _log("d2_shisa_patch_exec_fail\n" + traceback.format_exc())

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _ExecHook())
