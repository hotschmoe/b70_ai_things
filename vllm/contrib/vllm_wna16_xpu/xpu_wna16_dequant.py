# SPDX-License-Identifier: Apache-2.0
# Last-resort XPU fallback for compressed-tensors W4A16 (WNA16) layers that the aligned
# XPUwNa16 kernel cannot take -- e.g. input size K not a multiple of 32, or group_size not
# dividing K (as in some Qwen3-VL vision-tower MLPs: K=4304, 4304%128=80). It dequantizes the
# packed int4 weights -> bf16/fp16 ONCE at load, then runs a plain dense GEMM. Correctness-first
# (slower than the packed kernel); the point is to make the model LOAD + run on XPU so we can keep
# every model in compressed-tensors format. For a text-only serve the vision tower is never executed,
# so this path is free there. Registered as the LAST XPU candidate (the fast XPUwNa16 still wins for
# the normal %32-aligned layers). See ../README.md + ../../../docs and the b70 ORGANIZATION contract.
import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)
_SUPPORTED = (scalar_types.uint4, scalar_types.uint4b8)


class XPUDequantWNA16LinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUDequantWNA16 only supported on XPU"
        if c.weight_type not in _SUPPORTED:
            return False, f"Quant type ({c.weight_type}) not supported by XPUDequantWNA16"
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "XPUDequantWNA16 only supports BF16/FP16 activations"
        if c.has_g_idx:
            return False, "XPUDequantWNA16 does not support act-reordering (g_idx)"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        # bias=8 for symmetric uint4b8 (GPTQ-style); explicit zeros for asymmetric uint4.
        symmetric = c.weight_type == scalar_types.uint4b8

        # --- qweight [N, K//8] int32 -> unpack to [N, K] ints in [0,15] ---
        wq = getattr(layer, self.w_q_name)
        permute_param_layout_(wq, input_dim=1, output_dim=0, packed_dim=1)
        w = wq.data
        N, K8 = w.shape
        K = K8 * 8
        shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
        w_int = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(N, K).to(torch.float32)  # [N, K]

        # --- scales [N, n_groups] -> expand along K via group index ---
        ws = getattr(layer, self.w_s_name)
        permute_param_layout_(ws, input_dim=1, output_dim=0)
        scale = ws.data.to(torch.float32)  # [N, n_groups]
        n_groups = scale.shape[1]
        gs = c.group_size if (c.group_size is not None and c.group_size > 0) else K
        gidx = torch.clamp(torch.arange(K, device=w.device) // gs, max=n_groups - 1)
        scale_exp = scale.index_select(1, gidx)  # [N, K]

        # --- zero point ---
        if symmetric:
            zero = 8.0
        else:
            zp = getattr(layer, self.w_zp_name or "", None)
            if zp is None:
                zero = 8.0
            else:
                # zp packed [N//8, n_groups] int32 along N -> unpack to [N, n_groups]
                zpp = zp.data
                Np8 = zpp.shape[0]
                zp_un = ((zpp.unsqueeze(1) >> shifts) & 0xF).permute(0, 2, 1).reshape(Np8 * 8, n_groups)
                zero = zp_un.to(torch.float32).index_select(1, gidx)  # [N, K]

        w_deq = ((w_int - zero) * scale_exp).to(c.act_type).contiguous()  # [N, K] dense
        replace_parameter(layer, self.w_q_name, torch.nn.Parameter(w_deq, requires_grad=False))
        logger.debug("XPUDequantWNA16: dequantized %s layer to dense [%d, %d] %s", "sym" if symmetric else "asym", N, K, c.act_type)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        w = getattr(layer, self.w_q_name)  # [N, K] dense
        return F.linear(x, w.to(x.dtype), bias)
