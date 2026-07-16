# w8a8_shim.py -- wire INT8 W8A8 into sglang's compressed-tensors W8A8-int8 scheme on XPU (B70).
#
# TWO paths, both gated by B70_XPU_W8A8=1 (installed from woq_shim):
#  (A) FUSED hybrid (NEW, B70_XPU_W8A8_FUSED=1, the fast path): our built oneDNN ops --
#        decode  (M==1): int8_gemm_w8a16(x_f16, B_nt, wscale[N])           -- 1 fused launch, ~1.9x bf16
#        prefill (M>1) : dynamic_per_token_int8_quant -> int8_gemm_w8a8     -- s8xs8 XMX, ~2.0x bf16
#      Mirrors the W4A8 hybrid (woq_shim _XpuW4A8WoqKernel). Validated card-0: w8a8/w8a8_fused_probe.py
#      (decode 1.86-1.91x, prefill 1.95-2.07x bf16; matches/beats fp8 bar; int8-accurate).
#  (B) LEGACY _int_mm chain (default if FUSED unset): per-token int8 quant -> torch._int_mm -> dequant
#      (3 launches/layer, decode launch-bound ~0.8x bf16 eager).
# Needs the built _xpu_C.so (B70_XPU_C_SO) for path A; falls back to (B) if the ops don't load.
import os

_DBG = {"on": os.environ.get("B70_W8A8_DEBUG") == "1", "n": 0}


def _load_int8_gemm_op():
    """Make torch.ops._xpu_C.int8_gemm_w8a16 / int8_gemm_w8a8 callable (built oneDNN int8 GEMMs).
    ctypes-dlopen the built _xpu_C*.so (B70_XPU_C_SO) RTLD_GLOBAL so its oneAPI deps resolve."""
    import ctypes
    import torch
    have = lambda: hasattr(torch.ops._xpu_C, "int8_gemm_w8a16") and hasattr(
        torch.ops._xpu_C, "int8_gemm_w8a8"
    )
    if have():
        return True
    so = os.environ.get("B70_XPU_C_SO")
    if so and os.path.exists(so):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            print(f"[w8a8-fused] dlopen'd {so}", flush=True)
        except OSError as e:
            print(f"[w8a8-fused] ctypes.CDLL({so}) failed: {e}", flush=True)
    elif so:
        print(f"[w8a8-fused] B70_XPU_C_SO={so} does not exist", flush=True)
    else:
        print("[w8a8-fused] B70_XPU_C_SO unset", flush=True)
    return have()


def install():
    import torch
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    # The W8A8 scheme's CompressedTensorsConfig._check_scheme_supported does
    # DeviceCapability(*torch.cuda.get_device_capability()) -> throws on XPU. Scope the patch to ONLY
    # the scheme-support check (emulate cap 90) so no global sm90 side-effects leak into the TP path.
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

    fused = os.environ.get("B70_XPU_W8A8_FUSED") == "1"
    if fused and not _load_int8_gemm_op():
        print("[w8a8-fused] int8_gemm ops NOT available -> FALLING BACK to _int_mm chain", flush=True)
        fused = False

    # optional fused act-quant for prefill (reuse the W4A8 Triton single-launch kernel; eager fallback)
    _aq = None
    if fused and os.environ.get("B70_W8A8_FUSED_AQ", "op") != "eager":
        # prefer the built oneDNN dynamic_per_token_int8_quant op (single launch)
        if hasattr(torch.ops._xpu_C, "dynamic_per_token_int8_quant"):
            def _aq(xf):
                q, s, z = torch.ops._xpu_C.dynamic_per_token_int8_quant(xf, True, 8)
                return q, s, z
            print("[w8a8-fused] prefill act-quant: oneDNN dynamic_per_token_int8_quant (1 launch)", flush=True)

    _orig_pw = CompressedTensorsW8A8Int8.process_weights_after_loading

    # ---------------- FUSED hybrid path ----------------
    def _pw_fused(self, layer):
        _orig_pw(self, layer)  # CHANNEL strategy transposes weight [N,K] -> [K,N] (a view)
        w = layer.weight.data  # [K,N] view of original [N,K] s8
        # NT format for the oneDNN ops: B [K,N] with stride[0]==1. Materialize an [N,K] contiguous
        # backing buffer and view its transpose (pure relayout, no value change).
        weight_NK = w.t().contiguous()           # [N,K] s8 contiguous backing
        layer._w8a8_B_contig = weight_NK         # keep storage alive (B_nt is a view of it)
        layer.B_nt = weight_NK.t()               # [K,N] VIEW, stride[0]==1 (NT)
        assert layer.B_nt.stride()[0] == 1, (
            f"[w8a8-fused] B_nt NOT NT (stride0={layer.B_nt.stride()[0]})"
        )
        layer.wscale_n = layer.weight_scale.data.reshape(-1).to(torch.float16)  # [N] per-channel f16
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False
        )
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()

    def _apply_fused(self, layer, x, bias=None):
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        M = x2.shape[0]
        b = bias.to(torch.float16) if bias is not None else None
        xf = x2.to(torch.float16).contiguous()        # ops are fp16
        if M == 1:
            out = torch.ops._xpu_C.int8_gemm_w8a16(xf, layer.B_nt, layer.wscale_n, b)  # decode
        else:
            if _aq is not None:
                xq, xs, xz = _aq(xf)
            else:
                amax = xf.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
                xs = (amax / 127.0).to(torch.float16)
                xq = (xf / xs).round().clamp_(-127, 127).to(torch.int8).contiguous()
            out = torch.ops._xpu_C.int8_gemm_w8a8(
                xq, xs.contiguous(), None, layer.B_nt, layer.wscale_n, None, b, torch.float16
            )                                          # prefill: per-token sym int8 act
        if _DBG["on"] and _DBG["n"] < 80:
            n = _DBG["n"]; _DBG["n"] += 1
            o_nan = bool(torch.isnan(out).any() or torch.isinf(out).any())
            print(f"[w8a8-fused-dbg] call={n:>3} M={M} K={layer.B_nt.shape[0]} N={layer.B_nt.shape[1]} "
                  f"out_absmax={out.abs().max().item():.4g} out_bad={o_nan}", flush=True)
        return out.to(x.dtype).reshape(*orig[:-1], -1)

    # ---------------- LEGACY _int_mm chain ----------------
    def _pw_legacy(self, layer):
        _orig_pw(self, layer)
        w = layer.weight.data  # [K, N] int8
        layer.weight_t = w.contiguous()
        layer.wscale_row = layer.weight_scale.data.reshape(1, -1).to(torch.float32)  # [1, N]
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False
        )
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()

    def _apply_legacy(self, layer, x, bias=None):
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        amax = x2.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
        x_scale = amax * (1.0 / 127.0)
        x_q = torch.round(x2 / x_scale).clamp_(-127, 127).to(torch.int8)
        acc = torch._int_mm(x_q, layer.weight_t)
        out = acc.to(torch.float32) * x_scale.to(torch.float32) * layer.wscale_row
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*orig[:-1], -1)

    if fused:
        CompressedTensorsW8A8Int8.process_weights_after_loading = _pw_fused
        CompressedTensorsW8A8Int8.apply_weights = _apply_fused
        print("[w8a8-shim] installed: FUSED hybrid (decode=int8_gemm_w8a16, prefill=int8_gemm_w8a8)", flush=True)
    else:
        CompressedTensorsW8A8Int8.process_weights_after_loading = _pw_legacy
        CompressedTensorsW8A8Int8.apply_weights = _apply_legacy
        print("[w8a8-shim] installed: LEGACY torch._int_mm chain (XPU INT8 XMX)", flush=True)
