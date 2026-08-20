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


class _Int4MoeMethod:
    """Slot apply over packed w13/w2. Static T/K loops; tensor-indexed ids."""

    def __init__(self, orig, w13_q, w13_s, w13_z, w2_q, w2_s, w2_z, group_size):
        self.orig = orig
        self.w13_q = w13_q
        self.w13_s = w13_s
        self.w13_z = w13_z
        self.w2_q = w2_q
        self.w2_s = w2_s
        self.w2_z = w2_z
        self.group_size = int(group_size)

    def create_weights(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, *args, **kwargs):
        pass

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        _ensure_cast_op()
        t_n = int(x.shape[0])
        k_n = int(topk_ids.shape[-1])
        acc = torch.zeros_like(x)
        ids = topk_ids.to(torch.int64)
        wts = topk_weights.to(dtype=x.dtype)
        for t in range(t_n):
            xt = x[t : t + 1]
            acc_t = torch.zeros_like(xt)
            for k in range(k_n):
                e = ids[t, k].reshape(1)
                qw = self.w13_q.index_select(0, e).squeeze(0).t()
                sw = self.w13_s.index_select(0, e).squeeze(0)
                zw = self.w13_z
                y13 = torch.ops.b70.int4_gemm_w4a16_cast(
                    xt, qw, sw, zw, self.group_size
                )
                gate, up = y13.chunk(2, dim=-1)
                h = torch.nn.functional.silu(gate) * up
                qw2 = self.w2_q.index_select(0, e).squeeze(0).t()
                sw2 = self.w2_s.index_select(0, e).squeeze(0)
                y2 = torch.ops.b70.int4_gemm_w4a16_cast(
                    h, qw2, sw2, zw, self.group_size
                )
                acc_t = acc_t + y2 * wts[t, k]
            acc[t : t + 1] = acc_t
        # SharedExperts.forward is owned by MoERunner (order enum). Do not
        # call it from here -- dummy_run double-call asserts.
        return acc

    def forward(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


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


def _collect_fused_moe(root, prefix=""):
    found = []
    layers = getattr(root, "layers", None)
    if layers is None:
        return found
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        exp = getattr(mlp, "experts", None)
        if exp is None:
            continue
        for cand, tag in (
            (exp, "experts"),
            (getattr(exp, "routed_experts", None), "experts.routed_experts"),
            (getattr(exp, "experts", None), "experts.inner"),
        ):
            if cand is None:
                continue
            if hasattr(cand, "w13_weight"):
                found.append((f"{prefix}layers.{li}.mlp.{tag}", cand))
    return found


@torch.no_grad()
def _pack_fused_moe(name, moe):
    w13 = moe.w13_weight
    w2 = moe.w2_weight
    if w13 is None or w2 is None or w13.ndim != 3 or w2.ndim != 3:
        print(
            f"[nvfp4-shim] draft MTP INT4: skip moe {name} shape",
            file=sys.stderr,
            flush=True,
        )
        return 0, 0
    e_n = int(w13.shape[0])
    # XPU UnquantizedFusedMoEMethod transposes last two dims: w13 [E,H,2I]
    # w2 [E,I,H]. Linear INT4 wants [N,K] with K=hidden for w13 and K=I for w2.
    xpu_layout = w13.dim() == 3 and w2.dim() == 3 and w13.shape[1] == w2.shape[2]
    print(
        f"[nvfp4-shim] draft MTP INT4: {name} E={e_n} "
        f"w13={tuple(w13.shape)} w2={tuple(w2.shape)} xpu_layout={xpu_layout}",
        file=sys.stderr,
        flush=True,
    )
    w13_qs, w13_ss, w2_qs, w2_ss = [], [], [], []
    qz = None
    gs = 128
    nbytes_in = w13.numel() * w13.element_size() + w2.numel() * w2.element_size()
    for e in range(e_n):
        w13_e = w13[e].detach()
        w2_e = w2[e].detach()
        if xpu_layout:
            w13_e = w13_e.t().contiguous()
            w2_e = w2_e.t().contiguous()
        if w13_e.shape[-1] % 128 != 0 or w2_e.shape[-1] % 128 != 0:
            print(
                f"[nvfp4-shim] draft MTP INT4: skip moe {name} e={e} "
                f"K {tuple(w13_e.shape)},{tuple(w2_e.shape)}",
                file=sys.stderr,
                flush=True,
            )
            return 0, 0
        q, s, z, gs = quantize_to_int4(w13_e)
        # Store [N, K/8] so apply can pass .t() NT view (stride0==1).
        w13_qs.append(q.t().contiguous())
        w13_ss.append(s)
        qz = z
        q2, s2, z2, gs = quantize_to_int4(w2_e)
        w2_qs.append(q2.t().contiguous())
        w2_ss.append(s2)
    w13_q = torch.stack(w13_qs, 0).contiguous()
    w13_s = torch.stack(w13_ss, 0).contiguous()
    w2_q = torch.stack(w2_qs, 0).contiguous()
    w2_s = torch.stack(w2_ss, 0).contiguous()
    packed = _Int4MoeMethod(
        getattr(moe, "quant_method", None),
        w13_q, w13_s, qz, w2_q, w2_s, qz, gs,
    )
    # RoutedExperts.quant_method is an nn.Module child; do not assign a
    # plain object. Patch apply/forward in place.
    qm = getattr(moe, "quant_method", None)
    if qm is None:
        print(
            f"[nvfp4-shim] draft MTP INT4: skip moe {name} no quant_method",
            file=sys.stderr,
            flush=True,
        )
        return 0, 0
    object.__setattr__(moe, "_b70_int4_moe", packed)
    qm.apply = packed.apply
    if hasattr(qm, "forward"):
        qm.forward = packed.forward
    nbytes_out = (
        w13_q.numel() * w13_q.element_size()
        + w13_s.numel() * w13_s.element_size()
        + w2_q.numel() * w2_q.element_size()
        + w2_s.numel() * w2_s.element_size()
    )
    moe.w13_weight.set_(
        torch.empty(0, dtype=w13.dtype, device=w13.device)
    )
    moe.w2_weight.set_(torch.empty(0, dtype=w2.dtype, device=w2.device))
    print(
        f"[nvfp4-shim] draft MTP INT4: {name} packed NT "
        f"{nbytes_in/1e6:.0f}->{nbytes_out/1e6:.0f} MB",
        file=sys.stderr,
        flush=True,
    )
    return nbytes_in, nbytes_out


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
    moes_pre = _collect_fused_moe(root)
    if not linears and not moes_pre:
        print(
            "[nvfp4-shim] draft MTP INT4: no dense linears or moe; skip",
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
    moes = _collect_fused_moe(root)
    print(
        f"[nvfp4-shim] draft MTP INT4: packing {len(moes)} fused-moe",
        file=sys.stderr,
        flush=True,
    )
    for name, moe in moes:
        inn, outn = _pack_fused_moe(name, moe)
        nbytes_in += inn
        nbytes_out += outn
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
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MoeMTP"),
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
