# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# b70 OVERLAY of vllm/model_executor/kernels/linear/mixed_precision/xpu.py for
# vllm-xpu-env:int8g-v0251 (vLLM 0.25.1). The mount REPLACES the whole file, so this
# file is the verbatim upstream 0.25.1 content (extracted from the image 2026-07-21;
# upstream now SHIPS both XPUwNa16LinearKernel and XPUW4A8IntLinearKernel -- our old
# v0.23 patch classes were upstreamed) PLUS three b70 deltas, each marked "b70:":
#   (1) VLLM_W4A8_PREPACKED: skip the on-load _pack_int4_weight when the checkpoint
#       already stores int32 [out, in/8] packed weights (avoids the ~28 GiB unpacked
#       int8 GPU transient that hangs/OOMs a 32 GB B70 on the 27B).
#   (2) B70_W4A8_HYBRID=N (default 0=off): route M <= N through the quant-free
#       int4_gemm_w4a16 op (fp16 act, no per-token act-quant) -- the sglang-proven
#       hybrid (sglang/patches/w4a8_shim.py: decode M==1 1.83x vs woqgemm). Shares
#       weight storage with the w4a8 path via the same weight_packed.t() NT view --
#       NO second weight copy (never .contiguous() that view).
#   (3) a lazy register_fake for int4_gemm_w4a16 (fp16 output) so the hybrid path is
#       traceable under PIECEWISE capture. NOTE (verified in-image 2026-07-21):
#       upstream vllm/_xpu_ops.py ALREADY registers fakes for BOTH int4 ops when the
#       loaded .so exposes them (ours does), so this registration normally logs
#       "skipped: already has a fake impl" -- it is a defensive fallback only.
# If a future image drifts this upstream file, re-extract and re-apply the deltas:
#   docker run --rm --entrypoint cat vllm-xpu-env:int8g-v0251 \
#     /opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py


import os

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_XPUWNA16_SUPPORTED_QUANT_TYPES = (scalar_types.uint4, scalar_types.uint4b8)

logger = init_logger(__name__)

# b70 (2): hybrid small-M route threshold. 0 = off (byte-identical upstream behavior).
# 1 = decode-only (M==1); the MTP spec=3 verify batch is M=4, so set 4 to also cover
# the verify step. Value read once per process (env comes from the serve DOCKER_ENV).
_W4A8_HYBRID_M_MAX = int(os.environ.get("B70_W4A8_HYBRID", "0") or "0")

# b70 (3): lazy fake registration for int4_gemm_w4a16 (needed only by the hybrid
# route under torch.compile / XPU graph capture). Mirrors the image's baked
# scaled_mm/xpu_int8.py _register_int8_fakes pattern (lazy + idempotent).
_W4A16_FAKE_DONE = False


def _register_w4a16_fake() -> None:
    global _W4A16_FAKE_DONE
    if _W4A16_FAKE_DONE:
        return
    register_fake = getattr(torch.library, "register_fake", None) or getattr(
        torch.library, "impl_abstract", None
    )
    if register_fake is None:
        return
    try:
        import vllm._xpu_ops  # noqa: F401  (triggers torch.ops._xpu_C library load)
    except Exception:
        pass
    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        return  # op not in the mounted .so yet -> retry on the next call

    # int4_gemm_w4a16(A[M,K] f16, B[K/8,N] i32 NT, bias?, B_scale[K/g,N], B_zp[1] i8,
    #                 group_size, g_idx?) -> [M, N] float16 (out dtype HARD-CODED in C++)
    def _fake_int4_gemm_w4a16(A, B, bias, B_scale, B_zp, group_size, g_idx):
        return A.new_empty((A.shape[0], B.shape[1]), dtype=torch.float16)

    try:
        register_fake("_xpu_C::int4_gemm_w4a16", _fake_int4_gemm_w4a16)
        logger.info("registered fake for _xpu_C::int4_gemm_w4a16 (b70 hybrid route)")
    except (RuntimeError, ValueError) as e:
        # already registered (e.g. by another overlay) -> fine
        logger.info("register_fake(int4_gemm_w4a16) skipped: %s", e)
    _W4A16_FAKE_DONE = True


class XPUwNa16LinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUwNa16 only supported on XPU"

        if c.act_type != torch.bfloat16 and c.act_type != torch.float16:
            return False, "XPUwNa16 only supports BF16/FP16 activations"

        if c.weight_type not in _XPUWNA16_SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "XPUwNa16, supported types are: "
                f"{_XPUWNA16_SUPPORTED_QUANT_TYPES}",
            )
        if c.group_size != -1 and c.group_size % 32 != 0:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "XPUwNa16, supported group sizes are multiples of 32",
            )

        if c.partition_weight_shape[0] % 32 != 0:
            return (
                False,
                f"Input size ({c.partition_weight_shape[0]}) not supported by "
                "XPUwNa16, supported sizes are multiples of 32",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        need_transpose = False
        qweight_shape = getattr(layer, self.w_q_name).shape
        scale_shape = getattr(layer, self.w_s_name).shape
        # gptq marlin and compressed tensors wna16 expect different default
        # layouts for weight and scale, so we check the shapes to determine
        # if we need to transpose
        if qweight_shape[0] != scale_shape[0]:
            need_transpose = True

        if need_transpose:
            getattr(layer, self.w_q_name).data = (
                getattr(layer, self.w_q_name).data.t().contiguous()
            )
            getattr(layer, self.w_s_name).data = getattr(layer, self.w_s_name).data
        else:
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data.t().contiguous()
            )

        if self.config.zero_points:
            # (FIXME): maybe zero points should also be transposed.
            getattr(layer, self.w_zp_name).data = (
                getattr(layer, self.w_zp_name).data.t().contiguous()
            )
        else:
            weight_zero_point = torch.Tensor([8]).to(torch.int8).to("xpu")
            setattr(
                layer, self.w_zp_name, Parameter(weight_zero_point, requires_grad=False)
            )
        if self.config.has_g_idx:
            setattr(
                layer,
                self.w_gidx_name,
                Parameter(
                    getattr(layer, self.w_gidx_name).data.t().contiguous(),
                    requires_grad=False,
                ),
            )
        else:
            setattr(layer, self.w_gidx_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            w_q.t(),
            bias if bias is not None else None,
            w_s,
            w_zp,
            self.config.group_size,
            w_gidx,
        )
        return out


class XPUW4A8IntLinearKernel(MPLinearKernel):
    """XPU kernel for W4A8 integer quantization using oneDNN int4_gemm_w4a8.

    Weights are symmetric group-quantized int4 packed as uint4.
    Activations are dynamically quantized per-token to symmetric int8.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUW4A8Int only supported on XPU"
        if c.act_type not in (torch.bfloat16, torch.float16):
            return False, "XPUW4A8Int requires BF16/FP16 activations"
        if c.weight_type != scalar_types.int4:
            return (
                False,
                f"XPUW4A8Int requires int4 weights, got {c.weight_type}",
            )
        if c.zero_points:
            return False, "XPUW4A8Int only supports symmetric weight quantization"
        if c.group_size != -1 and c.group_size % 32 != 0:
            return (
                False,
                f"Group size ({c.group_size}) not supported by XPUW4A8Int, "
                "must be a multiple of 32",
            )
        in_size, out_size = c.partition_weight_shape
        if in_size % 8 != 0 or out_size % 8 != 0:
            return (
                False,
                f"in/out sizes ({in_size}, {out_size}) must be multiples of 8",
            )

        if c.act_type != torch.float16:
            logger.warning_once(
                "XPUW4A8IntLinearKernel is running with model dtype %s, "
                "but int4_gemm_w4a8 produces float16 output. Recommend "
                "setting --dtype float16 for best performance.",
                c.act_type,
            )

        return True, None

    def _pack_int4_weight(self, w: torch.Tensor) -> torch.Tensor:
        # w is [N, K] int8 with values in [-8, 7]
        w_u4 = w.to(torch.int32) + 8  # shift to [0, 15]
        w_u4 = w_u4.reshape(w.shape[0], w.shape[1] // 8, 8)  # [N, K/8, 8]
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=w.device)
        packed = ((w_u4 & 0xF) << shifts[None, None, :]).sum(dim=2).to(torch.int32)
        return packed

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale.data = layer.weight_scale.data.t().contiguous()

        device = layer.weight_packed.device
        # TODO: support asymmetric quantization
        weight_zero_point = torch.tensor([8], dtype=torch.int8, device=device)
        layer.weight_zero_point = Parameter(weight_zero_point, requires_grad=False)

        # weight_packed is [out, in] int8, signed int4 values in [-8, 7]
        w = layer.weight_packed.data  # [out, in]

        # TODO: implement asym case
        # b70 (1): prepacked offline -> w is ALREADY int32 [out, in/8]; skip the
        # on-load pack (avoids the ~28 GiB unpacked-int8 GPU transient that
        # hangs/OOMs the 32 GB B70 on the 27B). Requires the matching int32
        # allocation in compressed_tensors_w4a8_int.py (mounted alongside).
        if os.environ.get("VLLM_W4A8_PREPACKED"):
            assert w.dtype == torch.int32 and w.shape[1] * 8 == self.config.partition_weight_shape[0], (
                f"VLLM_W4A8_PREPACKED set but weight_packed is {w.dtype} {tuple(w.shape)}; "
                "expected int32 [out, in/8] -- is the checkpoint actually prepacked "
                "(quantization_config.is_prepacked_w4a8: true)?"
            )
            packed = w
        else:
            packed = self._pack_int4_weight(w)  # [out, in/8] packed uint4

        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(packed, requires_grad=False),
        )

        # Free the original unpacked int8 weight (still registered as "weight")
        # to avoid double-storing both int8 [N, K] and int32 [N, K/8] in memory.
        layer.register_parameter("weight", None)

        # b70 (2)+(3): hybrid small-M w4a16 route setup. The w4a16 op consumes the
        # SAME stored tensors as the w4a8 op -- weight_packed.t() ([in/8, out] int32
        # NT view; never .contiguous() it, that would materialize a 2nd full copy,
        # the exact trap the W8A8 W8A16-routing hit) and the transposed
        # weight_scale [in/g, out] -- so the route costs ZERO extra weight memory.
        if _W4A8_HYBRID_M_MAX > 0:
            wq = getattr(layer, self.w_q_name)
            assert wq.is_contiguous(), (
                "hybrid w4a16 route expects a contiguous [out, in/8] weight_packed "
                "so that .t() is the NT view the op wants"
            )
            _register_w4a16_fake()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])  # [M, K]

        # b70 (2): HYBRID small-M route (default OFF). At small M (decode/GEMV) the
        # int4 GEMM is bandwidth-bound and int8 activations do not help; the
        # avoidable cost is the per-token act-quant. Route M <= _W4A8_HYBRID_M_MAX
        # through the quant-free fp16-act int4_gemm_w4a16 (sglang-measured 1.83x at
        # M==1 vs the woqgemm baseline, w4a8_shim.py). Both ops emit fp16; cast back.
        if 0 < reshaped_x.shape[0] <= _W4A8_HYBRID_M_MAX:
            xf = (
                reshaped_x
                if reshaped_x.dtype == torch.float16
                else reshaped_x.to(torch.float16)
            )
            b16 = (
                bias
                if bias is None or bias.dtype == torch.float16
                else bias.to(torch.float16)
            )
            out = torch.ops._xpu_C.int4_gemm_w4a16(
                xf,
                layer.weight_packed.t(),  # [in/8, out] int32 NT view (shared storage)
                b16,
                layer.weight_scale,  # [in/g, out] (transposed at load)
                layer.weight_zero_point,  # [1] int8 -> symmetric
                self.config.group_size,
                None,  # g_idx not currently supported
            )
            return out.to(x.dtype)

        from vllm._xpu_ops import xpu_ops as ops

        # TODO: static and asymmetric quantization case
        # Common code for CompressedTensorsW4A8Int does not read act symmetry data
        quant_x, x_scale, x_zero = ops.dynamic_per_token_int8_quant_ref(
            reshaped_x, True, 8
        )

        out = torch.ops._xpu_C.int4_gemm_w4a8(
            quant_x,
            x_scale,
            x_zero,
            layer.weight_packed.t(),
            layer.weight_scale,
            layer.weight_zero_point,
            self.config.group_size,
            None,  # g_idx not currently supported
            bias,
        )

        return out.to(x.dtype)
