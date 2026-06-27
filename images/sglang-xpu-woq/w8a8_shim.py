# w8a8_shim.py -- wire torch._int_mm (oneDNN INT8 XMX, ~1.8x bf16 on B70) into sglang's
# compressed-tensors W8A8-int8 scheme on XPU. sgl_kernel's int8_scaled_mm + per_token_quant_int8
# are CUDA-only; we replace apply_weights with a torch path: dynamic per-token symmetric int8 act
# quant -> torch._int_mm(x_q, weight_t) [int32] -> dequant by (x_scale * weight_scale).
# Weight [N,K] int8 is pre-transposed to weight_t [K,N] once at load (and the original freed to fit VRAM).
# Gated opt-in via B70_XPU_W8A8=1 (installed from woq_shim). Validated: torch._int_mm 1.91x (M=1) / 1.81x (M=512).
import os


def install():
    import torch
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    # CompressedTensorsConfig._check_scheme_supported does DeviceCapability(*torch.cuda.get_device_capability())
    # which asserts "Torch not compiled with CUDA" on XPU. Return a high capability so the int8 scheme passes.
    torch.cuda.get_device_capability = lambda *a, **k: (9, 0)

    _orig_pw = CompressedTensorsW8A8Int8.process_weights_after_loading

    def _pw(self, layer):
        _orig_pw(self, layer)  # keeps azp_adj etc.
        w = layer.weight.data  # [N, K] int8
        # pre-transpose to [K, N] contiguous int8 for torch._int_mm(x[M,K], wt[K,N]) -> [M,N]
        layer.weight_t = w.t().contiguous()
        ws = layer.weight_scale.data.reshape(1, -1).to(torch.float32)  # [1, N] per-channel
        layer.wscale_row = ws
        # free the original [N,K] weight (we only need weight_t) to fit one card
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False
        )
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()

    def _apply(self, layer, x, bias=None):
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        # dynamic per-token symmetric int8 quant
        amax = x2.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
        x_scale = amax * (1.0 / 127.0)  # [M, 1] float
        x_q = torch.round(x2 / x_scale).clamp_(-127, 127).to(torch.int8)
        acc = torch._int_mm(x_q, layer.weight_t)  # [M, N] int32
        out = acc.to(torch.float32) * x_scale.to(torch.float32) * layer.wscale_row
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*orig[:-1], -1)

    CompressedTensorsW8A8Int8.process_weights_after_loading = _pw
    CompressedTensorsW8A8Int8.apply_weights = _apply
    print("[w8a8-shim] installed: CompressedTensorsW8A8Int8 -> torch._int_mm (XPU INT8 XMX)", flush=True)
