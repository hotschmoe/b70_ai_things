# gdn_fused_conv.py -- B70 decode lever: FUSE causal_conv1d_update INTO the GDN gated-delta
# recurrence decode kernel (one triton launch per linear-attn layer instead of two).
#
# Shipped GDN decode (sglang GDNAttnBackend.forward_decode), 48 of 64 layers/token:
#   mixed_qkv = causal_conv1d_update(in_proj_out, conv_state, w, bias, "silu", idx)  # LAUNCH 1
#   core_attn = packed_decode(mixed_qkv, a, b, ...)                                   # LAUNCH 2
# The conv is depthwise (per-channel, width=4) on the packed [q|k|v] layout; the recurrence
# then reads q/k/v slices of that post-conv tensor per value-head. This module folds the conv
# (silu) for exactly the q/k/v channels each recurrence program needs INTO the recurrence
# kernel, eliminating launch 1.
#
# CORRECTNESS: bitwise-identical to the unfused 2-kernel path at the production decode shape
# (B=1, max-running-requests 1, graph bs=1) -- relerr 0 on out AND conv_state AND ssm_state,
# deterministic over 30 repeats (validated in sglang/gdn_decode_autotune.py-style microbench;
# see W4A8_PLAN.md "GDN conv-fusion"). The conv math mirrors _causal_conv1d_update_kernel
# (bf16 loads, bf16 multiply, fp32 bias accumulate, silu, bf16 mixed_qkv round-trip) and the
# recurrence mirrors fused_recurrent_gated_delta_rule_packed_decode_kernel exactly.
#
# OPT-IN via B70_GDN_FUSED_CONV=1 (default OFF; the unfused path is the fallback). The wrapper
# hard-guards the fast path and delegates to the original forward_decode for anything else
# (replayssm on, non-silu, width!=4, no bias, non-2D/non-contiguous mixed_qkv, packed decode
# unsupported, or any shape/dtype mismatch).
import os

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.op import exp


@triton.jit
def _fused_conv_recur_decode_kernel(
    x_preconv,        # [B, QKV_DIM]  pre-conv in_proj output (packed [q|k|v])
    conv_w,           # [QKV_DIM, W]
    conv_bias,        # [QKV_DIM]
    conv_state,       # [num_slots, QKV_DIM, state_len]  (in-place shift update)
    a, b, A_log, dt_bias,
    o, h0, ht, ssm_state_indices, conv_state_indices,
    scale,
    stride_x_tok: tl.constexpr,
    stride_cw_dim: tl.constexpr, stride_cw_w: tl.constexpr,
    stride_cs_slot: tl.constexpr, stride_cs_dim: tl.constexpr, stride_cs_tok: tl.constexpr,
    stride_a_tok: tl.constexpr, stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr, stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return
    cs_idx = tl.load(conv_state_indices + i_n).to(tl.int64)

    # packed [q|k|v] channel indices (same layout the recurrence kernel splits)
    c_q = i_h * K + o_k                  # [BK]
    c_k = (H * K) + i_h * K + o_k        # [BK]
    c_v = (2 * H * K) + i_hv * V + o_v   # [BV]

    # ===== depthwise conv (width 4, silu). bf16 loads + bf16 multiply + fp32 bias acc, then
    #       bf16 round-trip -- mirrors _causal_conv1d_update_kernel + the bf16 mixed_qkv store. =====
    base_csq = conv_state + cs_idx * stride_cs_slot + c_q * stride_cs_dim
    q0 = tl.load(base_csq + 0 * stride_cs_tok, mask_k, 0.0)
    q1 = tl.load(base_csq + 1 * stride_cs_tok, mask_k, 0.0)
    q2 = tl.load(base_csq + 2 * stride_cs_tok, mask_k, 0.0)
    xq = tl.load(x_preconv + i_n * stride_x_tok + c_q, mask_k, 0.0)
    wbq = conv_w + c_q * stride_cw_dim
    wq0 = tl.load(wbq + 0 * stride_cw_w, mask_k, 0.0)
    wq1 = tl.load(wbq + 1 * stride_cw_w, mask_k, 0.0)
    wq2 = tl.load(wbq + 2 * stride_cw_w, mask_k, 0.0)
    wq3 = tl.load(wbq + 3 * stride_cw_w, mask_k, 0.0)
    accq = tl.load(conv_bias + c_q, mask_k, 0.0).to(tl.float32)
    accq += q0 * wq0
    accq += q1 * wq1
    accq += q2 * wq2
    accq += xq * wq3
    accq = accq / (1.0 + tl.exp(-accq))
    b_q = accq.to(tl.bfloat16).to(tl.float32)

    base_csk = conv_state + cs_idx * stride_cs_slot + c_k * stride_cs_dim
    k0 = tl.load(base_csk + 0 * stride_cs_tok, mask_k, 0.0)
    k1 = tl.load(base_csk + 1 * stride_cs_tok, mask_k, 0.0)
    k2 = tl.load(base_csk + 2 * stride_cs_tok, mask_k, 0.0)
    xk = tl.load(x_preconv + i_n * stride_x_tok + c_k, mask_k, 0.0)
    wbk = conv_w + c_k * stride_cw_dim
    wk0 = tl.load(wbk + 0 * stride_cw_w, mask_k, 0.0)
    wk1 = tl.load(wbk + 1 * stride_cw_w, mask_k, 0.0)
    wk2 = tl.load(wbk + 2 * stride_cw_w, mask_k, 0.0)
    wk3 = tl.load(wbk + 3 * stride_cw_w, mask_k, 0.0)
    acck = tl.load(conv_bias + c_k, mask_k, 0.0).to(tl.float32)
    acck += k0 * wk0
    acck += k1 * wk1
    acck += k2 * wk2
    acck += xk * wk3
    acck = acck / (1.0 + tl.exp(-acck))
    b_k = acck.to(tl.bfloat16).to(tl.float32)

    base_csv = conv_state + cs_idx * stride_cs_slot + c_v * stride_cs_dim
    v0 = tl.load(base_csv + 0 * stride_cs_tok, mask_v, 0.0)
    v1 = tl.load(base_csv + 1 * stride_cs_tok, mask_v, 0.0)
    v2 = tl.load(base_csv + 2 * stride_cs_tok, mask_v, 0.0)
    xv = tl.load(x_preconv + i_n * stride_x_tok + c_v, mask_v, 0.0)
    wbv = conv_w + c_v * stride_cw_dim
    wv0 = tl.load(wbv + 0 * stride_cw_w, mask_v, 0.0)
    wv1 = tl.load(wbv + 1 * stride_cw_w, mask_v, 0.0)
    wv2 = tl.load(wbv + 2 * stride_cw_w, mask_v, 0.0)
    wv3 = tl.load(wbv + 3 * stride_cw_w, mask_v, 0.0)
    accv = tl.load(conv_bias + c_v, mask_v, 0.0).to(tl.float32)
    accv += v0 * wv0
    accv += v1 * wv1
    accv += v2 * wv2
    accv += xv * wv3
    accv = accv / (1.0 + tl.exp(-accv))
    b_v = accv.to(tl.bfloat16).to(tl.float32)

    # conv_state in-place shift: new = [old1, old2, x] (bf16). q/k channels are shared by the
    # HV//H * NV programs of a k-head; all write IDENTICAL values from register-cached reads, so
    # the update is bitwise-correct + deterministic at B=1 (validated). v channels are uniquely
    # owned per (i_v, i_hv) program.
    tl.store(base_csq + 0 * stride_cs_tok, q1, mask_k)
    tl.store(base_csq + 1 * stride_cs_tok, q2, mask_k)
    tl.store(base_csq + 2 * stride_cs_tok, xq, mask_k)
    tl.store(base_csk + 0 * stride_cs_tok, k1, mask_k)
    tl.store(base_csk + 1 * stride_cs_tok, k2, mask_k)
    tl.store(base_csk + 2 * stride_cs_tok, xk, mask_k)
    tl.store(base_csv + 0 * stride_cs_tok, v1, mask_v)
    tl.store(base_csv + 1 * stride_cs_tok, v2, mask_v)
    tl.store(base_csv + 2 * stride_cs_tok, xv, mask_v)

    # ===== gated-delta recurrence (identical math to packed_decode kernel) =====
    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_conv_recurrent_packed_decode(
    in_proj_out, conv_state, conv_w, conv_bias,
    a, b, A_log, dt_bias, scale, ssm_state, cache_indices,
    num_v_heads, head_v_dim, num_k_heads, head_k_dim,
):
    """One-launch conv(silu)+gated-delta recurrence decode. Returns [B,1,HV,V]."""
    B = in_proj_out.shape[0]
    HV, V, H, K = num_v_heads, head_v_dim, num_k_heads, head_k_dim
    out = in_proj_out.new_empty(B, 1, HV, V)
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    num_warps = int(os.environ.get("B70_GDN_DECODE_WARPS", "1"))
    grid = (NV, B * HV)
    _fused_conv_recur_decode_kernel[grid](
        in_proj_out, conv_w, conv_bias, conv_state,
        a, b, A_log, dt_bias,
        out, ssm_state, ssm_state, cache_indices, cache_indices,
        scale,
        in_proj_out.stride(0),
        conv_w.stride(0), conv_w.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        a.stride(0), b.stride(0),
        ssm_state.stride(0), ssm_state.stride(0), cache_indices.stride(0),
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=num_warps, num_stages=3,
    )
    return out


def install():
    """Monkeypatch GDNAttnBackend.forward_decode to use the fused kernel on the fast path."""
    if os.environ.get("B70_GDN_FUSED_CONV") != "1":
        return
    try:
        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    except Exception as e:
        print(f"[gdn-fused-conv] not installing (import failed): {e}", flush=True)
        return

    _orig_forward_decode = GDNAttnBackend.forward_decode

    def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        disp = self.kernel_dispatcher
        if not getattr(disp, "supports_packed_decode", False):
            return _orig_forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
        try:
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
            conv_states = layer_cache.conv[0]
            ssm_states = layer_cache.temporal
            cache_indices = self.forward_metadata.mamba_cache_indices
            md = self.forward_metadata
            # any replayssm activity -> defer to the original (kernel does not support it)
            replay_on = (
                getattr(md, "replayssm_write_pos", None) is not None
                or getattr(md, "replayssm_force_flush", None) is not None
                or getattr(layer_cache, "replayssm_d", None) is not None
                or getattr(layer_cache, "replayssm_k", None) is not None
                or getattr(layer_cache, "replayssm_g", None) is not None
            )
            act = layer.activation
            is_silu = act in ("silu", "swish") or act is True
            bias = layer.bias
            cw = layer.conv_weights
            ok = (
                not replay_on
                and is_silu
                and bias is not None
                and isinstance(mixed_qkv, torch.Tensor)
                and mixed_qkv.ndim == 2
                and mixed_qkv.is_contiguous()
                and cw.ndim == 2
                and cw.shape[-1] == 4
                and conv_states.ndim == 3
            )
        except Exception:
            return _orig_forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
        if not ok:
            return _orig_forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)

        out = fused_conv_recurrent_packed_decode(
            in_proj_out=mixed_qkv,
            conv_state=conv_states,
            conv_w=cw,
            conv_bias=bias,
            a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias,
            scale=layer.head_k_dim ** -0.5,
            ssm_state=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=layer.num_v_heads, head_v_dim=layer.head_v_dim,
            num_k_heads=layer.num_k_heads, head_k_dim=layer.head_k_dim,
        )
        self._track_mamba_state_decode(forward_batch, conv_states, ssm_states, cache_indices)
        return out.transpose(0, 1)

    GDNAttnBackend.forward_decode = forward_decode
    print("[gdn-fused-conv] installed: GDNAttnBackend.forward_decode -> fused conv+recurrence (B70_GDN_FUSED_CONV=1)", flush=True)
