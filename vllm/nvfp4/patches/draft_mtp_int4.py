# draft_mtp_int4.py -- cookbook steal: INT4 the MTP drafter at load_weights.
# Gated B70_DRAFT_MTP_INT4=1. Target body stays NVFP4 fused. Never hook forward
# (Qwen3_5MTP.forward is @support_torch_compile).
from __future__ import annotations

import os
import sys

import torch


def quantize_to_int4(weight: torch.Tensor, group_size: int = 128):
    """RTN INT4 g128 sym in int4_gemm_w4a16 layout (cookbook patch)."""
    device = weight.device
    n_out, k_in = weight.shape
    if k_in % group_size != 0:
        raise ValueError(f"K={k_in} not divisible by group_size={group_size}")
    num_groups = k_in // group_size
    chunk = 4096
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    parts = []
    scale_parts = []
    for i in range(0, n_out, chunk):
        wc = weight[i : i + chunk].float()
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(
            wc.shape[0], k_in // 8
        )
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight = torch.cat(parts, dim=0).t()
    scales = torch.cat(scale_parts, dim=0).t().contiguous()
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def _ensure_cast_op():
    """Opaque bf16-in/bf16-out wrapper. L65: fake(input.dtype)+real(fp16) made
    GRAPH dummy_run DCE the .to() and die Half!=BF16. Hide the fp16 kernel."""
    if getattr(_ensure_cast_op, "_ready", False):
        return
    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        return

    @torch.library.custom_op("b70::int4_gemm_w4a16_cast", mutates_args=())
    def int4_gemm_w4a16_cast(
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if flat.dtype != torch.float16:
            flat = flat.to(torch.float16)
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            flat, qweight, None, scales, qzeros, group_size, None
        )
        return out.to(x.dtype).reshape(*x.shape[:-1], qweight.shape[1])

    @int4_gemm_w4a16_cast.register_fake
    def _(x, qweight, scales, qzeros, group_size):
        return torch.empty(
            (*x.shape[:-1], qweight.shape[1]),
            dtype=x.dtype,
            device=x.device,
        )

    _ensure_cast_op._ready = True


class _Int4LinearMethod:
    def __init__(self, qweight, scales, qzeros, group_size):
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.group_size = group_size

    def create_weights(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, *args, **kwargs):
        pass

    def apply(self, layer, x, bias):
        _ensure_cast_op()
        if hasattr(torch.ops, "b70") and hasattr(torch.ops.b70, "int4_gemm_w4a16_cast"):
            out = torch.ops.b70.int4_gemm_w4a16_cast(
                x, self.qweight, self.scales, self.qzeros, int(self.group_size)
            )
        else:
            flat = x.reshape(-1, x.shape[-1])
            if flat.dtype != torch.float16:
                flat = flat.to(torch.float16)
            out = torch.ops._xpu_C.int4_gemm_w4a16(
                flat,
                self.qweight,
                None,
                self.scales,
                self.qzeros,
                self.group_size,
                None,
            )
            out = out.to(x.dtype).reshape(*x.shape[:-1], self.qweight.shape[1])
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out


def _collect_dense_linears(root, prefix=""):
    found = []
    fc = getattr(root, "fc", None)
    if fc is not None and hasattr(fc, "weight"):
        found.append((prefix + "fc", fc))
    layers = getattr(root, "layers", None)
    if layers is None:
        return found
    for li, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is not None:
            for an in ("qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(attn, an, None)
                if lin is not None and hasattr(lin, "weight"):
                    found.append((f"{prefix}layers.{li}.attn.{an}", lin))
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for mn in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, mn, None)
            if lin is not None and hasattr(lin, "weight"):
                found.append((f"{prefix}layers.{li}.mlp.{mn}", lin))
        shared = getattr(mlp, "shared_expert", None)
        if shared is not None:
            for mn in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
                lin = getattr(shared, mn, None)
                if lin is not None and hasattr(lin, "weight"):
                    found.append(
                        (f"{prefix}layers.{li}.mlp.shared.{mn}", lin)
                    )
    return found


@torch.no_grad()
def build_draft_mtp_int4(model):
    if os.environ.get("B70_DRAFT_MTP_INT4", "0") != "1":
        return
    if getattr(model, "_b70_mtp_int4_built", False):
        return
    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        print(
            "[nvfp4-shim] draft MTP INT4: no int4_gemm_w4a16; skip",
            file=sys.stderr,
            flush=True,
        )
        return
    root = getattr(model, "model", model)
    linears = _collect_dense_linears(root)
    if not linears:
        print(
            "[nvfp4-shim] draft MTP INT4: no dense linears; skip",
            file=sys.stderr,
            flush=True,
        )
        return
    print(
        f"[nvfp4-shim] draft MTP INT4: packing {len(linears)} dense linears",
        file=sys.stderr,
        flush=True,
    )
    nbytes_in = 0
    nbytes_out = 0
    for name, lin in linears:
        w = getattr(lin, "weight", None)
        if w is None or w.ndim != 2 or w.numel() == 0:
            continue
        if w.shape[-1] % 128 != 0:
            print(
                f"[nvfp4-shim] draft MTP INT4: skip {name} K={w.shape[-1]}",
                file=sys.stderr,
                flush=True,
            )
            continue
        orig_shape = tuple(w.shape)
        qweight, scales, qzeros, gs = quantize_to_int4(w.detach())
        lin.quant_method = _Int4LinearMethod(qweight, scales, qzeros, gs)
        nbytes_in += w.numel() * w.element_size()
        nbytes_out += qweight.numel() * qweight.element_size()
        nbytes_out += scales.numel() * scales.element_size()
        lin.weight.set_(torch.empty(0, dtype=w.dtype, device=w.device))
        print(
            f"[nvfp4-shim] draft MTP INT4: {name} {orig_shape}",
            file=sys.stderr,
            flush=True,
        )
    model._b70_mtp_int4_built = True
    print(
        f"[nvfp4-shim] draft MTP INT4: {nbytes_in/1e6:.0f} MB -> "
        f"{nbytes_out/1e6:.0f} MB",
        file=sys.stderr,
        flush=True,
    )


def install():
    if os.environ.get("B70_DRAFT_MTP_INT4", "0") != "1":
        return
    _ensure_cast_op()
    hooked = 0
    for mod_name, cls_name in (
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP"),
        ("vllm.model_executor.models.qwen3_5_moe", "Qwen3_5MoeMTP"),
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is None or not hasattr(cls, "load_weights"):
                continue
            orig = cls.load_weights

            def _wrapped(self, weights, _orig=orig):
                result = _orig(self, weights)
                build_draft_mtp_int4(self)
                return result

            cls.load_weights = _wrapped
            hooked += 1
            print(
                f"[nvfp4-shim] draft MTP INT4: hooked {mod_name}.{cls_name}"
                ".load_weights",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(
                f"[nvfp4-shim] draft MTP INT4: hook {cls_name} skipped:",
                repr(e),
                file=sys.stderr,
                flush=True,
            )
    if hooked == 0:
        print(
            "[nvfp4-shim] draft MTP INT4: no MTP class hooked",
            file=sys.stderr,
            flush=True,
        )
