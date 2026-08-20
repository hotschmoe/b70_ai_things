# fused_moe_apply.py -- capture-safe NVFP4 MoE apply (sitecustomize block 7).
#
# Decode / graph capture cannot call torch.unique().tolist() (host sync inside
# an XPU command graph). Two paths share the same gemm:
#   slots   - static Python loops over T and top_k. Tensor-indexed copies
#             into fixed workspaces so replay can change expert ids.
#             Used when T <= SLOT_MAX and expert_map is empty (TP, no EP).
#   grouped - old unique()+index_add path for large eager prefill.
# The dispatcher lives in an opaque custom op so inductor does not inline
# unique() into the compiled graph.
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as _F

_SLOT_MAX = int(os.environ.get("B70_NVFP4_MOE_SLOT_MAX", "16"))
_FORCE = os.environ.get("B70_NVFP4_MOE_PATH", "").strip().lower()
# L63: llm-scaler #505/#507 -- grow-only sticky scratch. Default OFF
# (L63 measured 33.1 vs hold 34.9; extra copy). Opt-in B70_NVFP4_MOE_STICKY=1.
_STICKY = os.environ.get("B70_NVFP4_MOE_STICKY", "0") == "1"
# L64: specialize T==1 (decode GEMV-shaped). Default OFF (L64 33.3).
# O4 LOOP 12: ESIMD M=1 1D block_load GEMV in vllm/nvfp4/proto_moe_m1/.
# Isolated ~2.3x vs oneDNN M=1. LOOP 13: WG/SLM occupancy NO-GO.
# O4c: sidecar torch op. Default OFF. Do not swap live _xpu_C.
_M1 = os.environ.get("B70_NVFP4_MOE_M1", "0") == "1"
_M1K = os.environ.get("B70_NVFP4_MOE_M1_KERNEL", "0") == "1"
_M1_SO = os.environ.get("B70_NVFP4_M1_SO", "").strip()
_HAS_M1K = False
if _M1K and _M1_SO:
    try:
        torch.ops.load_library(_M1_SO)
        _HAS_M1K = hasattr(torch.ops, "b70_nvfp4_m1")
    except Exception as e:
        print(
            "[nvfp4-shim] m1_gemv load_library failed:",
            repr(e),
            file=sys.stderr,
            flush=True,
        )
_GS = 16
_RING = int(os.environ.get("B70_NVFP4_MOE_RING", "4"))
_M1K_LOGGED = False


class _GrowScratch:
    """Grow-only buffers + 4-deep output ring (llm-scaler #505/#507)."""

    def __init__(self, ring=_RING):
        self._buf = {}
        self._ring = []
        self._ri = 0
        self._ring_n = max(1, ring)

    def view(self, key, shape, dtype, device):
        n = 1
        for s in shape:
            n *= int(s)
        cur = self._buf.get(key)
        if (
            cur is None
            or cur.device != device
            or cur.dtype != dtype
            or cur.numel() < n
        ):
            cur = torch.empty(n, dtype=dtype, device=device)
            self._buf[key] = cur
        return cur.narrow(0, 0, n).view(shape)

    def stash_out(self, src):
        """Copy src into the next ring slot; return that slot (stable addr)."""
        if not _STICKY:
            return src
        need = src.numel()
        slot = None
        if self._ri < len(self._ring):
            slot = self._ring[self._ri]
        if (
            slot is None
            or slot.device != src.device
            or slot.dtype != src.dtype
            or slot.numel() < need
        ):
            slot = torch.empty(need, dtype=src.dtype, device=src.device)
            if self._ri < len(self._ring):
                self._ring[self._ri] = slot
            else:
                self._ring.append(slot)
        dst = slot.narrow(0, 0, need).view(src.shape)
        dst.copy_(src)
        self._ri = (self._ri + 1) % self._ring_n
        return dst


_SCRATCH = _GrowScratch()


def _sticky_copy(key, src):
    if not _STICKY:
        return src
    dst = _SCRATCH.view(key, src.shape, src.dtype, src.device)
    dst.copy_(src)
    return dst


def _silu_and_mul(x):
    d = x.shape[-1] // 2
    return _F.silu(x[..., :d]) * x[..., d:]


def _gemm(x, w_row, scale):
    # w_row [N, K/2] uint8 packed; oneDNN wants NT view [K/2, N].
    # O4c sidecar: M=1 and K%256==0 (Ornith H=2048 I=512). Small unit
    # shapes (H=64) stay on oneDNN so test_fused_moe_apply stays PASS.
    if (
        _HAS_M1K
        and x.shape[0] == 1
        and int(x.shape[-1]) % 256 == 0
        and w_row.is_contiguous()
        and scale.is_contiguous()
    ):
        global _M1K_LOGGED
        if not _M1K_LOGGED:
            print(
                "[nvfp4-shim] m1_gemv dispatch "
                f"N={int(w_row.shape[0])} K={int(x.shape[-1])}",
                file=sys.stderr,
                flush=True,
            )
            _M1K_LOGGED = True
        return torch.ops.b70_nvfp4_m1.gemv(x, w_row, scale)
    return torch.ops._xpu_C.nvfp4_gemm_w4a16(
        x, w_row.transpose(0, 1), None, scale, _GS
    )


def build_fused_moe_scales(mod):
    """Fold f8 block * per-expert global -> [E, K/16, N] bf16 NT. Once."""
    g1 = mod.quant_config.g1_alphas.reshape(-1).to(torch.float32)
    g2 = mod.quant_config.g2_alphas.reshape(-1).to(torch.float32)
    s1 = mod.w1_scale_val
    s2 = mod.w2_scale_val
    if s1 is None or s2 is None:
        raise RuntimeError("fused MoE scales already consumed")
    # s1 [E, 2I, H/16] -> [E, H/16, 2I]; s2 [E, H, I/16] -> [E, I/16, H]
    mod._s13_nt = (
        (s1.to(torch.float32) * g1.view(-1, 1, 1))
        .permute(0, 2, 1)
        .contiguous()
        .to(torch.bfloat16)
    )
    mod._s2_nt = (
        (s2.to(torch.float32) * g2.view(-1, 1, 1))
        .permute(0, 2, 1)
        .contiguous()
        .to(torch.bfloat16)
    )
    mod.w1_scale_val = None
    mod.w2_scale_val = None
    mod._fused_ready = True


def apply_slots(output, xb, w1, w2, s13, s2, topk_w, topk_ids, apply_on_input):
    """Static T x top_k M=1 gemms. No unique / nonzero / .item / .tolist.

    XPU `w1[scalar_tensor]` host-syncs (LOOP 61 unit: event.wait on a command
    graph). Gather every slot with one index_select, then index the gathered
    buffer with Python ints (capture-constant offsets).
    """
    t_count = int(xb.shape[0])
    k_count = int(topk_ids.shape[-1])
    n_slot = t_count * k_count
    out = output.reshape(t_count, -1)
    out.zero_()
    ids_flat = topk_ids.reshape(n_slot)
    wts_flat = _sticky_copy(
        "wts", topk_w.to(torch.bfloat16).reshape(n_slot, 1)
    )
    w1_g = _sticky_copy("w1g", w1.index_select(0, ids_flat))
    w2_g = _sticky_copy("w2g", w2.index_select(0, ids_flat))
    s13_g = _sticky_copy("s13g", s13.index_select(0, ids_flat))
    s2_g = _sticky_copy("s2g", s2.index_select(0, ids_flat))
    # L64: T==1 decode skips repeat_interleave (xb is already [1, H]).
    use_m1 = _M1 and t_count == 1
    if use_m1:
        x_rep = None
    else:
        x_rep = _sticky_copy("xrep", xb.repeat_interleave(k_count, dim=0))
    for i in range(n_slot):
        wr = wts_flat[i : i + 1]
        if use_m1:
            x_in = xb * wr if apply_on_input else xb
        else:
            x_in = x_rep[i : i + 1] * wr if apply_on_input else x_rep[i : i + 1]
        gu = _gemm(x_in, w1_g[i], s13_g[i])
        h = _silu_and_mul(gu).to(torch.bfloat16)
        dn = _gemm(h, w2_g[i], s2_g[i])
        if not apply_on_input:
            dn = dn * wr
        t = i // k_count
        out[t : t + 1].add_(dn.to(out.dtype))


def apply_grouped(
    output, xb, w1, w2, s13, s2, topk_w, topk_ids, expert_map, apply_on_input
):
    """Eager prefill: one gemm per distinct expert (host unique OK)."""
    ids = topk_ids
    output.zero_()
    out_flat = output.reshape(-1, output.shape[-1])
    for g in torch.unique(ids).tolist():
        local = g
        if expert_map is not None and expert_map.numel() > 0:
            local = int(expert_map[g].item())
            if local < 0:
                continue
        mask = ids == g
        tok_idx, slot_idx = mask.nonzero(as_tuple=True)
        if tok_idx.numel() == 0:
            continue
        w_route = topk_w[tok_idx, slot_idx].to(torch.bfloat16).unsqueeze(1)
        x_e = _sticky_copy("xe", xb.index_select(0, tok_idx))
        if apply_on_input:
            x_e = x_e * w_route
        gu = _gemm(x_e, w1[local], s13[local])
        h = _silu_and_mul(gu).to(torch.bfloat16)
        dn = _gemm(h, w2[local], s2[local])
        if not apply_on_input:
            dn = dn * w_route
        out_flat.index_add_(0, tok_idx, dn.to(out_flat.dtype))


def dispatch(
    output, xb, w1, w2, s13, s2, topk_w, topk_ids, expert_map, apply_on_input
):
    t_count = xb.shape[0]
    use_slots = expert_map is None or expert_map.numel() == 0
    use_slots = use_slots and (t_count <= _SLOT_MAX)
    if _FORCE == "slots":
        use_slots = True
    elif _FORCE == "grouped":
        use_slots = False
    if use_slots:
        apply_slots(
            output, xb, w1, w2, s13, s2, topk_w, topk_ids, apply_on_input
        )
    else:
        apply_grouped(
            output,
            xb,
            w1,
            w2,
            s13,
            s2,
            topk_w,
            topk_ids,
            expert_map,
            apply_on_input,
        )


def _register_custom_op():
    custom_op = getattr(torch.library, "custom_op", None)
    if custom_op is None:
        return False

    @custom_op("b70_nvfp4::moe_fused_apply", mutates_args=("output",))
    def moe_fused_apply(
        output: torch.Tensor,
        xb: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        s13: torch.Tensor,
        s2: torch.Tensor,
        topk_w: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor,
        apply_on_input: bool,
    ) -> None:
        em = expert_map if expert_map.numel() > 0 else None
        dispatch(
            output, xb, w1, w2, s13, s2, topk_w, topk_ids, em, apply_on_input
        )

    @moe_fused_apply.register_fake
    def _(
        output,
        xb,
        w1,
        w2,
        s13,
        s2,
        topk_w,
        topk_ids,
        expert_map,
        apply_on_input,
    ):
        return None

    return True


_HAS_CUSTOM_OP = _register_custom_op()


def fused_apply(
    mod,
    output,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation,
    global_num_experts,
    expert_map,
    a1q_scale,
    a2_scale,
    workspace13,
    workspace2,
    expert_tokens_meta,
    apply_router_weight_on_input,
):
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
    if not getattr(mod, "_fused_ready", False):
        build_fused_moe_scales(mod)
    xb = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.bfloat16)
    apply_on_input = bool(apply_router_weight_on_input)
    if expert_map is None:
        em = torch.empty(0, dtype=torch.long, device=xb.device)
    else:
        em = expert_map
    if _HAS_CUSTOM_OP:
        torch.ops.b70_nvfp4.moe_fused_apply(
            output,
            xb,
            w1,
            w2,
            mod._s13_nt,
            mod._s2_nt,
            topk_weights,
            topk_ids,
            em,
            apply_on_input,
        )
    else:
        dispatch(
            output,
            xb,
            w1,
            w2,
            mod._s13_nt,
            mod._s2_nt,
            topk_weights,
            topk_ids,
            em if em.numel() > 0 else None,
            apply_on_input,
        )


def install(experts_cls):
    orig_init = experts_cls.__init__

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            build_fused_moe_scales(self)
        except Exception as e:
            print(
                "[nvfp4-shim] (7) scale stack at init deferred:",
                repr(e),
                file=sys.stderr,
                flush=True,
            )

    experts_cls.__init__ = _init
    experts_cls.apply = fused_apply
    print(
        "[nvfp4-shim] (7) FUSED per-expert NVFP4 MoE apply installed "
        f"(slot-static T<={_SLOT_MAX}, grouped unique above; "
        f"sticky={int(_STICKY)} m1={int(_M1)} m1k={int(_HAS_M1K)} "
        f"ring={_RING} custom_op={_HAS_CUSTOM_OP})",
        file=sys.stderr,
        flush=True,
    )
