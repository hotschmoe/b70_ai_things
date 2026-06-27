# w8a8_shim.py -- wire torch._int_mm (oneDNN INT8 XMX, ~1.8x bf16 on B70) into sglang's
# compressed-tensors W8A8-int8 scheme on XPU. sgl_kernel's int8_scaled_mm + per_token_quant_int8
# are CUDA-only; we replace apply_weights with a torch path: dynamic per-token symmetric int8 act
# quant -> torch._int_mm(x_q, weight_t) [int32] -> dequant by (x_scale * weight_scale).
# Weight [N,K] int8 is pre-transposed to weight_t [K,N] once at load (and the original freed to fit VRAM).
# Gated opt-in via B70_XPU_W8A8=1 (installed from woq_shim). Validated: torch._int_mm 1.91x (M=1) / 1.81x (M=512).
import os

_DBG = {"on": os.environ.get("B70_W8A8_DEBUG") == "1", "n": 0}  # per-layer NaN/range trace (first forward)


def install():
    import torch
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    # The W8A8 scheme's CompressedTensorsConfig._check_scheme_supported does
    # DeviceCapability(*torch.cuda.get_device_capability()), which throws "Torch not compiled with CUDA" on XPU.
    # The OLD fix faked torch.cuda.get_device_capability -> (9,0) GLOBALLY -- but that makes sglang believe it is
    # an sm90 CUDA GPU and take CUDA-only code paths (e.g. custom all-reduce setup). Scope the patch to ONLY the
    # scheme-support check (emulate capability 90 there) so no global sm90 side-effects leak into the TP path.
    try:
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        def _xpu_check_scheme_supported(self, min_capability, error=True):
            supported = 90 >= int(min_capability)
            if error and not supported:
                raise RuntimeError(
                    f"[w8a8-shim] scheme min_capability {min_capability} > emulated XPU cap 90"
                )
            return supported

        CompressedTensorsConfig._check_scheme_supported = _xpu_check_scheme_supported
        print("[w8a8-shim] scoped _check_scheme_supported (no global sm90 fake)", flush=True)
    except Exception as e:
        print(f"[w8a8-shim] scoped scheme patch failed, falling back to global cap fake: {e}", flush=True)
        torch.cuda.get_device_capability = lambda *a, **k: (9, 0)

    _orig_pw = CompressedTensorsW8A8Int8.process_weights_after_loading

    def _pw(self, layer):
        _orig_pw(self, layer)  # CHANNEL strategy ALREADY transposes layer.weight [N,K] -> [K,N] (a view)
        # _orig_pw left layer.weight as [K, N]; use it directly for torch._int_mm(x[M,K], wt[K,N]) -> [M,N].
        # (Do NOT transpose again -- the earlier w.t() double-transposed -> [N,K] and crashed the GEMM.)
        w = layer.weight.data  # [K, N] int8 (transposed view of the original [N,K])
        layer.weight_t = w.contiguous()
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
        if _DBG["on"] and _DBG["n"] < 80:
            n = _DBG["n"]; _DBG["n"] += 1
            in_nan = bool(torch.isnan(x2).any() or torch.isinf(x2).any())
            o_nan = bool(torch.isnan(out).any() or torch.isinf(out).any())
            print(f"[w8a8-dbg] call={n:>3} M={x2.shape[0]} K={layer.weight_t.shape[0]} "
                  f"N={layer.weight_t.shape[1]} in_absmax={x2.abs().max().item():.4g} in_bad={in_nan} "
                  f"out_absmax={out.abs().max().item():.4g} out_bad={o_nan} "
                  f"wscale_absmax={layer.wscale_row.abs().max().item():.4g}", flush=True)
        return out.reshape(*orig[:-1], -1)

    CompressedTensorsW8A8Int8.process_weights_after_loading = _pw
    CompressedTensorsW8A8Int8.apply_weights = _apply
    print("[w8a8-shim] installed: CompressedTensorsW8A8Int8 -> torch._int_mm (XPU INT8 XMX)", flush=True)
