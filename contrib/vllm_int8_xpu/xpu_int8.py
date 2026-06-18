import torch
from torch.nn import Parameter
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import convert_to_channelwise
from vllm.platforms import current_platform
from .ScaledMMLinearKernel import Int8ScaledMMLinearKernel, Int8ScaledMMLinearLayerConfig

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

    import sys
    for name, fn in (("_xpu_C::dynamic_per_token_int8_quant", _fake_quant),
                     ("_xpu_C::int8_gemm_w8a8", _fake_int8_gemm)):
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
        weight = getattr(layer, w_q_name)
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
        if hasattr(torch.ops._xpu_C, "dynamic_per_token_int8_quant"):
            x_q, x_s, _x_zp = torch.ops._xpu_C.dynamic_per_token_int8_quant(x_2d, True, 8)
        else:
            x_q, x_s, _x_zp = ops.dynamic_per_token_int8_quant_ref(x_2d, True, 8)
        out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, bias, x.dtype)
        return out.reshape(x.shape[:-1] + (out.size(-1),))


# Register fakes at import time too (in addition to the is_supported() hook), so they are present
# in whichever process imports this module (the engine-core worker runs graph capture).
_register_int8_fakes()
