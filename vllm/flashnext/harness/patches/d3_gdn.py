# Fused GDN decode. Installed from sitecustomize when B70_D3_GDN=1.
#
# Replaces QwenGatedDeltaNetAttention.forward_xpu for Ornith non-spec
# decode (and, if B70_D3_GDN_SPEC=1, the MTP spec path). Projections stay
# on oneDNN. The plugin op is conv + gated delta + RMSNorm-gated-silu.
# Prefill and any other shape call the stock SYCL op.

from __future__ import annotations

import os
import sys
import traceback

_LOGGED = False
_FUSED_SEEN = set()
_FALLBACK = set()


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _once(key: str, msg: str) -> None:
    if key in _FALLBACK:
        return
    _FALLBACK.add(key)
    _log(msg)


def install() -> None:
    global _LOGGED
    sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_gdn_xpu")
    import b70_gdn_xpu

    b70_gdn_xpu.register()
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    if getattr(QwenGatedDeltaNetAttention, "_b70_d3_gdn", False):
        return
    orig = QwenGatedDeltaNetAttention.forward_xpu

    def forward_xpu(self, hidden_states):
        try:
            return _forward(self, hidden_states, orig)
        except Exception as exc:
            key = "exc:" + type(exc).__name__ + ":" + str(exc).split("\n", 1)[0][:160]
            _once(key, "b70_d3 gdn fallback:\n" + traceback.format_exc())
            return orig(self, hidden_states)

    QwenGatedDeltaNetAttention.forward_xpu = forward_xpu
    QwenGatedDeltaNetAttention._b70_d3_gdn = True
    if not _LOGGED:
        _LOGGED = True
        spec = os.environ.get("B70_D3_GDN_SPEC", "0")
        _log(f"b70_d3 gdn fused decode installed spec={spec}")


def _meta(self):
    from vllm.forward_context import get_forward_context

    raw = get_forward_context().attn_metadata
    if not isinstance(raw, dict):
        return None
    return raw.get(self.prefix)


def _norm_w(self):
    import torch

    weight = self.norm.weight
    if weight.dtype == torch.float32:
        return weight
    cached = getattr(self, "_b70_norm_f", None)
    if cached is None or cached.device != weight.device:
        cached = weight.detach().to(dtype=torch.float32)
        self._b70_norm_f = cached
    return cached


def _dt(self, dtype):
    import torch

    if self.dt_bias.dtype == dtype:
        return self.dt_bias
    cached = getattr(self, "_b70_dt_h", None)
    if cached is None or cached.dtype != dtype or cached.device != self.dt_bias.device:
        cached = self.dt_bias.detach().to(dtype=dtype)
        self._b70_dt_h = cached
    return cached


def _has_ones(self, ref):
    # Pure decode sets has_initial_state to None. Stock treats that as
    # "every row has state". A stable all-true buffer keeps the pointer
    # fixed across CUDA-graph capture sizes.
    import torch

    cached = getattr(self, "_b70_has_init", None)
    if (
        cached is None
        or cached.device != ref.device
        or cached.numel() < 64
    ):
        cached = torch.ones(64, dtype=torch.bool, device=ref.device)
        self._b70_has_init = cached
    return cached


def _warm(self, hidden_states):
    # Casts must exist before capture. The GDN warmup prefill is the
    # last eager call, so build them on every path once the cache exists.
    _norm_w(self)
    _dt(self, hidden_states.dtype)
    cache = getattr(self, "kv_cache", None)
    if cache is not None and len(cache) >= 1:
        _has_ones(self, cache[0])


def _forward(self, hidden_states, orig):
    import torch

    if self.tp_size != 1 or self.gqa_interleaved_layout:
        _once("shape", "b70_d3 gdn skip: tp or interleaved")
        return orig(self, hidden_states)
    if (
        self.num_k_heads != 16
        or self.num_v_heads != 32
        or self.head_k_dim != 128
        or self.head_v_dim != 128
    ):
        _once("geom", "b70_d3 gdn skip: geometry")
        return orig(self, hidden_states)
    cache = getattr(self, "kv_cache", None)
    if cache is None or len(cache) < 2:
        return orig(self, hidden_states)
    _warm(self, hidden_states)
    md = _meta(self)
    if md is None:
        return orig(self, hidden_states)
    npre = int(getattr(md, "num_prefills", 0) or 0)
    ndec = int(getattr(md, "num_decodes", 0) or 0)
    nspec = int(getattr(md, "num_spec_decodes", 0) or 0)
    spec_on = os.environ.get("B70_D3_GDN_SPEC", "0") == "1"
    use_spec = spec_on and nspec > 0 and npre == 0 and ndec == 0
    use_dec = nspec == 0 and npre == 0 and ndec > 0
    if not use_spec and not use_dec:
        _once("path", f"b70_d3 gdn skip path pre={npre} dec={ndec} spec={nspec}")
        return orig(self, hidden_states)

    qkvz, _ = self.in_proj_qkvz(hidden_states)
    ba, _ = self.in_proj_ba(hidden_states)
    qkvz = qkvz.contiguous()
    ba = ba.contiguous()
    ntok = qkvz.shape[0]
    core = torch.zeros(
        (ntok, 32, 128), dtype=hidden_states.dtype, device=hidden_states.device
    )
    out = torch.empty_like(core)
    conv_w = self.conv1d.weight.view(8192, 4)
    if use_spec:
        token_indx = md.spec_token_indx
        slot = md.spec_state_indices_tensor
        num_accepted = md.num_accepted_tokens
        cu = md.spec_query_start_loc
        if token_indx is None or slot is None or num_accepted is None or cu is None:
            _once("specmeta", "b70_d3 gdn skip: spec metadata missing")
            return orig(self, hidden_states)
        has_initial = torch.ones(slot.shape[0], dtype=torch.bool, device=slot.device)
    else:
        cu = md.non_spec_query_start_loc
        slot = md.non_spec_state_indices_tensor
        has_initial = md.has_initial_state
        token_indx = None
        num_accepted = None
        # Pure decode leaves has_initial_state as None. That means load
        # every row, which is what the SYCL kernel does with a null mask.
        if cu is None or slot is None:
            _once("decmeta", "b70_d3 gdn skip: decode metadata missing")
            return orig(self, hidden_states)
        if has_initial is None:
            has_initial = _has_ones(self, slot)
    conv_state = cache[0]
    ssm_state = cache[1]
    if (
        not conv_state.is_contiguous()
        or not ssm_state.is_contiguous()
        or conv_state.dtype != torch.float16
        or ssm_state.dtype != torch.float32
        or conv_state.dim() < 3
        or conv_state.size(2) != 8192
        or conv_state.size(1) < 3
        or slot.dtype != torch.int32
        or cu.dtype != torch.int32
        or (not use_spec and (slot.dim() != 1 or cu.shape[0] != slot.shape[0] + 1))
        or conv_w.shape[0] != 8192
        or conv_w.shape[1] != 4
    ):
        _once(
            "layout",
            "b70_d3 gdn skip layout "
            f"conv={tuple(conv_state.shape)}:{conv_state.dtype} "
            f"ssm={tuple(ssm_state.shape)}:{ssm_state.dtype} "
            f"cu={None if cu is None else tuple(cu.shape)} "
            f"slot={None if slot is None else tuple(slot.shape)} "
            f"w={tuple(conv_w.shape)}",
        )
        return orig(self, hidden_states)
    dt = _dt(self, hidden_states.dtype)
    nw = _norm_w(self)
    if (
        qkvz.dtype != torch.float16
        or qkvz.shape[-1] != 12288
        or ba.shape[-1] != 64
        or self.A_log.dtype != torch.float32
        or self.A_log.numel() != 32
        or not self.A_log.is_contiguous()
        or dt.dtype != torch.float16
        or dt.numel() != 32
        or not dt.is_contiguous()
        or nw.dtype != torch.float32
        or nw.numel() != 128
        or not nw.is_contiguous()
        or not conv_w.is_contiguous()
    ):
        _once(
            "weights",
            "b70_d3 gdn skip weights "
            f"qkvz={tuple(qkvz.shape)}:{qkvz.dtype} "
            f"ba={tuple(ba.shape)}:{ba.dtype} "
            f"A={tuple(self.A_log.shape)}:{self.A_log.dtype} "
            f"dt={tuple(dt.shape)}:{dt.dtype} "
            f"nw={tuple(nw.shape)}:{nw.dtype}",
        )
        return orig(self, hidden_states)
    cap = False
    try:
        cap = bool(torch.xpu.is_current_stream_capturing())
    except Exception:
        cap = False
    seen = (int(cap), int(ntok), tuple(slot.shape), int(use_spec))
    if seen not in _FUSED_SEEN:
        _FUSED_SEEN.add(seen)
        _log(
            "b70_d3 gdn fused call "
            f"ntok={ntok} capturing={int(cap)} spec={int(use_spec)} "
            f"conv={tuple(conv_state.shape)} ssm={tuple(ssm_state.shape)} "
            f"cu={tuple(cu.shape)} slot={tuple(slot.shape)}"
        )
    torch.ops.b70_gdn.decode_fused(
        core,
        out,
        qkvz,
        ba,
        conv_state,
        ssm_state,
        conv_w,
        self.A_log,
        dt,
        nw,
        cu,
        slot,
        has_initial,
        token_indx,
        num_accepted,
        float(self.layer_norm_epsilon),
    )
    projected, _ = self.out_proj(out.reshape(ntok, -1))
    return projected
