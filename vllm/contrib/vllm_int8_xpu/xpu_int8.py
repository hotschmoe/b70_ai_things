import os
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import convert_to_channelwise
from vllm.platforms import current_platform
from .ScaledMMLinearKernel import Int8ScaledMMLinearKernel, Int8ScaledMMLinearLayerConfig

# B1 (docs/kernel/23): route the int8 W8A8 linear through the FUSED op
# int8_gemm_w8a8_fusedq -- one op that quantizes the f16/bf16 activation inline
# (parallel SYCL kernel) and runs the oneDNN s8s8 matmul, so the per-token quant
# is NOT a separate captured graph node (kills the ~101us capture-persistent
# hotspot without the inductor-fusion loss that regressed the opaque-op swap).
# Default ON when the op is present; set B70_FUSEDQ=0 to A/B the old two-step
# (standalone dynamic_per_token_int8_quant + int8_gemm_w8a8) path.
_FUSEDQ = os.environ.get("B70_FUSEDQ", "1") == "1"

# --- FakeTensor/meta registration so torch.compile / XPU graph capture
# (VLLM_XPU_ENABLE_XPU_GRAPH=1) can trace through our custom SYCL ops. Without
# these, dynamo's fake-tensor tracing raises UnsupportedOperatorException and
# graph capture aborts. Registered lazily (after _xpu_C loads) and idempotently.
_FAKES_REGISTERED = False


def _register_int8_fakes():
    global _FAKES_REGISTERED
    if _FAKES_REGISTERED:
        return
    register_fake = getattr(torch.library, "register_fake", None) \
        or getattr(torch.library, "impl_abstract", None)
    if register_fake is None:
        return
    # Force-load the _xpu_C library so the op SCHEMAS exist before we register fakes
    # (register_fake requires the op to be defined). Don't set the flag if it's not loaded yet
    # -> we'll retry on the next call.
    try:
        import vllm._xpu_ops  # noqa: F401  (triggers torch.ops._xpu_C library load)
    except Exception:
        pass
    if not hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):
        return

    # dynamic_per_token_int8_quant(input, use_sym_quant, bits)
    #   -> (q int8 [..., K], scale input-dtype [..., 1], zero_point int32 [..., 1])
    def _fake_quant(input, use_sym_quant, bits):
        q = torch.empty_like(input, dtype=torch.int8)
        sc_shape = list(input.shape[:-1]) + [1]
        scale = input.new_empty(sc_shape)                       # same dtype as input
        zp = torch.empty(sc_shape, dtype=torch.int32, device=input.device)
        return q, scale, zp

    # int8_gemm_w8a8(A[M,K] i8, A_scale, A_zp?, B[K,N] i8, B_scale, azp_adj?, bias?, out_dtype?)
    #   -> [M, N] out_dtype (defaults to A_scale dtype if out_dtype is None)
    def _fake_int8_gemm(A, A_scale, A_zp, B, B_scale, azp_adj, bias, out_dtype):
        dt = out_dtype if out_dtype is not None else A_scale.dtype
        return A.new_empty((A.shape[0], B.shape[1]), dtype=dt)

    # int8_gemm_w8a8_fusedq(A[..,K] f16/bf16, B[K,N] i8, B_scale, bias?, out_dtype?)
    #   -> [.., N] out_dtype (defaults to A.dtype). A is the UNQUANTIZED activation;
    #   the quant happens inside the op, so the fake only needs the output shape.
    def _fake_int8_gemm_fusedq(A, B, B_scale, bias, out_dtype):
        dt = out_dtype if out_dtype is not None else A.dtype
        return A.new_empty(tuple(A.shape[:-1]) + (B.shape[1],), dtype=dt)

    # int8_gemm_w8a16(A[..,K] f16/bf16, B[K,N] i8 NT, B_scale[N], bias?) -> [.., N]
    #   quant-free decode route (skips the act-quant); out dtype follows A (f16 here).
    def _fake_int8_gemm_w8a16(A, B, B_scale, bias):
        return A.new_empty(tuple(A.shape[:-1]) + (B.shape[1],), dtype=A.dtype)

    # int4_gemm_w4a8(A_ i8 [M,K], A_scale, A_zp, B int32 [K/8,N], B_scale, B_zp,
    #                group_size, g_idx?, bias?) -> [M, N] float16
    #   N = B.shape[1] (mat2 is packed [K/8, N]); M = A_.shape[0]. Out dtype is
    #   HARD-CODED float16 in the C++ kernel (torch::kHalf), NOT derived from input.
    #   The W4A8 activation quant uses the pure-PyTorch dynamic_per_token_int8_quant_ref
    #   (no custom op -> dynamo traces it directly -> no fake needed there).
    def _fake_int4_gemm_w4a8(A_, A_scale, A_zp, B, B_scale, B_zp,
                             group_size, g_idx, bias):
        return A_.new_empty((A_.shape[0], B.shape[1]), dtype=torch.float16)

    _int4_present = hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")

    import sys
    _fakes = [("_xpu_C::dynamic_per_token_int8_quant", _fake_quant),
              ("_xpu_C::int8_gemm_w8a8", _fake_int8_gemm)]
    if hasattr(torch.ops._xpu_C, "int8_gemm_w8a8_fusedq"):
        _fakes.append(("_xpu_C::int8_gemm_w8a8_fusedq", _fake_int8_gemm_fusedq))
    if hasattr(torch.ops._xpu_C, "int8_gemm_w8a16"):
        _fakes.append(("_xpu_C::int8_gemm_w8a16", _fake_int8_gemm_w8a16))
    if _int4_present:
        _fakes.append(("_xpu_C::int4_gemm_w4a8", _fake_int4_gemm_w4a8))
    for name, fn in _fakes:
        try:
            register_fake(name, fn)
            print(f"[xpu_int8] registered fake for {name}", file=sys.stderr, flush=True)
        except (RuntimeError, ValueError) as e:
            # already registered (e.g. native abstract impl present) -> fine
            print(f"[xpu_int8] register_fake({name}) skipped: {e}", file=sys.stderr, flush=True)
    _FAKES_REGISTERED = True


class XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """INT8 W8A8 dynamic-symmetric scaled-MM via oneDNN s8s8s32 on XPU (Battlemage)."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_xpu():
            return False, "XPUInt8ScaledMM is only supported on XPU."
        if not hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):
            return False, "int8_gemm_w8a8 op not present in the installed vllm-xpu-kernels wheel."
        # ops are loaded by now (hasattr forced the .so load) -> safe to register fakes
        _register_int8_fakes()
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig):
        if c.is_static_input_scheme:
            return False, "XPU int8 kernel supports dynamic activation quantization only."
        if not c.input_symmetric:
            return False, "XPU int8 kernel supports symmetric activations only."
        return True, None

    def process_weights_after_loading(self, layer):
        w_q_name, w_s_name, _, _, _ = self.layer_param_names
        weight = getattr(layer, w_q_name)                      # [N,K] s8
        # --- W8A16 small-M DECODE route (additive, env-gated by B70_W8A16_M_MAX): keep an
        # NT weight view [K,N] stride0==1 + an [N] f16 per-channel scale, so decode/MTP-verify
        # (small M) can go through the quant-free int8_gemm_w8a16 op and skip the per-token
        # activation quant. Matches the sglang shim.
        # [!] MEMORY COST (2026-07-21): the NT backing is a SECOND full copy of the s8 weight
        # (the s8s8 path needs [K,N] contiguous = the opposite physical layout) -> ~2x int8
        # weight residency (27B TP=2: model load 26.21 vs 17.5 GiB/card, KV drops to 0.55 GiB
        # -> a 253952-ctx serve cannot init). So only materialize it when the route is enabled;
        # B70_W8A16_M_MAX=0 (default) now costs nothing. Real fix = teach int8_gemm_w8a16 to
        # consume the s8s8 [K,N] layout (kernel TODO). ---
        if int(os.environ.get("B70_W8A16_M_MAX", "0")) > 0:
            _w_NK = weight.contiguous()                        # [N,K] s8 backing (keep alive)
            layer._w8a16_B_backing = _w_NK
            layer._w8a16_B_nt = _w_NK.t()                      # [K,N] view, stride0==1 (NT)
            weight_scale_nt = getattr(layer, w_s_name)
            if len(layer.logical_widths) > 1 and not self.config.is_channelwise:
                weight_scale_nt = convert_to_channelwise(weight_scale_nt, layer.logical_widths)
            layer._w8a16_wscale = weight_scale_nt.reshape(-1).to(torch.float16).contiguous()  # [N]
        replace_parameter(layer, w_q_name,
                          Parameter(weight.t().contiguous().data, requires_grad=False))
        weight_scale = getattr(layer, w_s_name)
        is_fused_module = len(layer.logical_widths) > 1
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(layer, w_s_name,
                          Parameter(weight_scale.reshape(1, -1).contiguous().data, requires_grad=False))

    def apply_weights(self, layer, x, bias=None):
        from vllm._xpu_ops import xpu_ops as ops
        w_q, w_s, _i_s, _i_zp, _azp_adj = self._get_layer_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        # W8A16 small-M DECODE route (B70_W8A16_M_MAX, default 0 = OFF = old behavior):
        # the int8 GEMM is already at the BW roofline; the avoidable cost at small M
        # (decode + MTP verify, M<=~64) is the per-token int8 ACTIVATION QUANT. Route
        # those through the quant-free int8_gemm_w8a16 op (f16 act x s8 wt, per-channel
        # scale, one launch) -> ~1.47x on the linear-GEMM slice, matches FP8, more accurate.
        # Measured: research/w8a8/decode_gemv/decode_roofline.md. Rollback: B70_W8A16_M_MAX=0.
        M = x_2d.shape[0]
        _wmax = int(os.environ.get("B70_W8A16_M_MAX", "0"))
        if (0 < M <= _wmax and hasattr(layer, "_w8a16_B_nt")
                and hasattr(torch.ops._xpu_C, "int8_gemm_w8a16")):
            xf = x_2d.to(torch.float16)
            b16 = bias.to(torch.float16) if bias is not None else None
            out = torch.ops._xpu_C.int8_gemm_w8a16(
                xf, layer._w8a16_B_nt, layer._w8a16_wscale, b16)
            return out.reshape(x.shape[:-1] + (out.size(-1),)).to(x.dtype)
        # FUSED path (B1): one op quantizes x inline + runs the s8s8 matmul, so
        # the per-token quant is not a separate captured graph node.
        if _FUSEDQ and hasattr(torch.ops._xpu_C, "int8_gemm_w8a8_fusedq"):
            out = torch.ops._xpu_C.int8_gemm_w8a8_fusedq(x_2d, w_q, w_s, bias, x.dtype)
            return out.reshape(x.shape[:-1] + (out.size(-1),))
        # Baseline two-step path (A/B with B70_FUSEDQ=0, or if the op is absent).
        if hasattr(torch.ops._xpu_C, "dynamic_per_token_int8_quant"):
            x_q, x_s, _x_zp = torch.ops._xpu_C.dynamic_per_token_int8_quant(x_2d, True, 8)
        else:
            x_q, x_s, _x_zp = ops.dynamic_per_token_int8_quant_ref(x_2d, True, 8)
        out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, bias, x.dtype)
        return out.reshape(x.shape[:-1] + (out.size(-1),))


# Register fakes at import time too (in addition to the is_supported() hook), so they are present
# in whichever process imports this module (the engine-core worker runs graph capture).
_register_int8_fakes()
