# nvfp4_shim.py -- stand up the ModelOpt NVFP4 (MIXED_PRECISION) checkpoint on sglang-XPU (B70).
#
# The target is nvidia/Qwen3.6-27B-NVFP4: a ModelOpt MIXED_PRECISION checkpoint whose
#   * MLP gate/up/down   -> W4A16_NVFP4 (4-bit E2M1 weight, per-16-K fp8 block scale, bf16 acts)
#   * self_attn + GDN in_proj -> FP8 (per-tensor E4M3 weight + activation scale)
#   * norms/conv1d/embed/lm_head/vision/mtp -> BF16 (unquantized)
#   * KV cache -> FP8 (we run bf16 KV instead; the serve script strips kv_cache_scheme)
#
# sglang 0.5.6 HAS the ModelOpt loader natively (ModelOptMixedPrecisionConfig in
# srt/layers/quantization/modelopt_quant.py), and the multimodal Qwen3_5ForConditionalGeneration
# arch (GDN hybrid + vision + mtp) is in its EntryClass -- so unlike the W4A8 text-only ckpt this
# needs NO model-registry / config hacks. But the ModelOpt NVFP4 path is CUDA-only in two ways this
# shim fixes for XPU:
#
#   (1) ROUTING GAP. The MLP layers carry quant_algo "W4A16_NVFP4"; ModelOptMixedPrecisionConfig.
#       get_quant_method only matches "NVFP4" and "FP8" exactly, so every W4A16_NVFP4 linear falls
#       through to UnquantizedLinearMethod -> it would try to load a uint8 4-bit packed weight as a
#       plain bf16 Linear (garbage / shape crash). We normalize "W4A16_NVFP4" -> "NVFP4" in
#       _resolve_quant_algo so those layers route to ModelOptFp4LinearMethod.
#
#   (2) CUDA-ONLY KERNELS. ModelOptFp4LinearMethod.process_weights_after_loading requires Marlin
#       (group_size 16 CUDA kernel) or SM100+ (raises "require SM100+" on XPU) and .cuda()s scales;
#       .apply uses cutlass/flashinfer fp4_gemm. ModelOptFp8LinearMethod.apply uses cutlass fp8
#       (cutlass_fp8_supported() is False on XPU). We REPLACE both classes' process_weights/apply
#       with XPU paths:
#         * NVFP4 (W4A16) -> our oneDNN op torch.ops._xpu_C.nvfp4_gemm_w4a16 (weights stay 4-bit
#           f4_e2m1 resident, decompressed in the JIT gemm, bf16 acts, [K/16,N] bf16 folded scale).
#           This is the exact analogue of the vLLM fused path (vllm/nvfp4 _XPUW4A16NvFp4Kernel);
#           the op .so is built against sglang's torch 2.12 ABI by sglang/nvfp4/build_nvfp4_kernel_sglang.sh.
#         * FP8 attention -> dequant-at-load to bf16 + plain F.linear. Conservative + XPU-safe (no
#           cutlass / torch._scaled_mm dependency). The attention is a small share of compute; this
#           keeps the bring-up robust. Native XPU fp8 compute is an OPEN optimization (see NVFP4_PORT.md).
#
# The op is loaded exactly like w4a8_shim/w8a8_shim: ctypes.CDLL(B70_XPU_C_SO, RTLD_GLOBAL) after
# `import torch`, with the oneAPI compiler libs prepended to LD_LIBRARY_PATH by the serve script.
#
# Gated ENTIRELY on B70_XPU_NVFP4=1 (bottom self-install). Unset -> this module is a no-op, so a bare
# `import nvfp4_shim` (via the mounted .pth) is byte-identical on every non-NVFP4 serve.
import os

_STATE = {"installed": False}


def _load_op():
    """Make torch.ops._xpu_C.nvfp4_gemm_w4a16 callable. Prefer the packaged extension; fall back to a
    ctypes dlopen of the built _xpu_C*.so (B70_XPU_C_SO) with RTLD_GLOBAL so its oneAPI libs resolve."""
    import torch

    if hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"):
        return True
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C on import)
    except Exception as e:
        print(f"[nvfp4-shim] vllm_xpu_kernels._xpu_C import failed ({e}); trying B70_XPU_C_SO", flush=True)
        so = os.environ.get("B70_XPU_C_SO")
        if so and os.path.exists(so):
            import ctypes

            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                print(f"[nvfp4-shim] dlopen'd {so}", flush=True)
            except Exception as e2:
                print(f"[nvfp4-shim] ctypes.CDLL({so}) failed: {e2}", flush=True)
        elif so:
            print(f"[nvfp4-shim] B70_XPU_C_SO={so} does not exist", flush=True)
        else:
            print("[nvfp4-shim] B70_XPU_C_SO unset", flush=True)
    return hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16")


def _register_fake():
    """FakeTensor meta for nvfp4_gemm_w4a16 so torch.compile / XPUGraph capture can trace through it
    (same move as w4a8_shim + the vLLM sitecustomize). out: [M, N] A.dtype; N = B.shape[1]."""
    import torch

    reg = getattr(torch.library, "register_fake", None) or getattr(torch.library, "impl_abstract", None)
    if reg is None or not hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"):
        return

    def _fake(A, B, bias, B_scale, group_size):
        return A.new_empty((A.shape[0], B.shape[1]), dtype=A.dtype)

    try:
        reg("_xpu_C::nvfp4_gemm_w4a16", _fake)
        print("[nvfp4-shim] registered fake for _xpu_C::nvfp4_gemm_w4a16", flush=True)
    except (RuntimeError, ValueError) as e:
        print(f"[nvfp4-shim] register_fake nvfp4_gemm_w4a16 skipped: {e}", flush=True)


def install():
    if _STATE["installed"]:
        return
    import torch

    try:
        from sglang.srt.utils import is_xpu

        if not is_xpu():
            print("[nvfp4-shim] not XPU; skip", flush=True)
            return
    except Exception:
        return

    if not _load_op():
        raise RuntimeError(
            "[nvfp4-shim] torch.ops._xpu_C.nvfp4_gemm_w4a16 NOT FOUND. Build the op against sglang's "
            "torch 2.12 ABI (sglang/nvfp4/build_nvfp4_kernel_sglang.sh) and point B70_XPU_C_SO at the "
            "built _xpu_C*.so. The vLLM v0230/torch-2.11 .so will NOT load into the torch-2.12 image."
        )
    _register_fake()

    import torch.nn.functional as F
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp4LinearMethod,
        ModelOptFp8LinearMethod,
        ModelOptMixedPrecisionConfig,
    )

    _DBG = {"on": os.environ.get("B70_NVFP4_DEBUG") == "1", "n": 0}

    # ---- (0) drop the SM/capability gate so the loader accepts the scheme on XPU -----------------
    # sglang's quant loader compares get_min_capability() against the device capability; on XPU the
    # capability probe throws / is meaningless. Emulate an always-satisfied capability (the real
    # kernel gate is bypassed by our process_weights/apply overrides below).
    try:
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        ModelOptFp4Config.get_min_capability = classmethod(lambda cls: 0)
        ModelOptMixedPrecisionConfig.get_min_capability = classmethod(lambda cls: 0)
        print("[nvfp4-shim] (0) get_min_capability -> 0 (XPU capability gate spoofed)", flush=True)
    except Exception as e:
        print(f"[nvfp4-shim] (0) min_capability spoof failed: {e}", flush=True)

    # ---- (1) route W4A16_NVFP4 -> NVFP4 so the MLP layers reach ModelOptFp4LinearMethod ----------
    # The MLP linears carry quant_algo "W4A16_NVFP4" (weight-only 4-bit, bf16 acts); the stock
    # get_quant_method only matches "NVFP4"/"FP8" exactly -> W4A16_NVFP4 -> UnquantizedLinearMethod
    # (wrong). Normalize it to "NVFP4"; our overridden ModelOptFp4LinearMethod is W4A16 (bf16 acts),
    # which is precisely what W4A16_NVFP4 wants.
    _orig_resolve = ModelOptMixedPrecisionConfig._resolve_quant_algo

    def _resolve_norm(self, prefix):
        algo = _orig_resolve(self, prefix)
        if algo is not None and algo.upper().endswith("NVFP4"):
            return "NVFP4"
        return algo

    ModelOptMixedPrecisionConfig._resolve_quant_algo = _resolve_norm
    print("[nvfp4-shim] (1) _resolve_quant_algo normalizes *NVFP4 -> NVFP4 (W4A16_NVFP4 MLP routing)", flush=True)

    # ---- (2) NVFP4 W4A16 linear -> our oneDNN nvfp4_gemm_w4a16 op --------------------------------
    # Signed E2M1 grid is symmetric (no zero point). At load: fold the fp8 block scale x the fp32
    # global scale (weight_scale_2.max(), matching sglang's own reference at modelopt_quant.py:1431)
    # into ONE [K/16, N] bf16 tensor in the op's NT layout; the 4-bit weight stays [N, K/2] uint8
    # resident and is passed as a free .t() NT view each forward.
    def _nvfp4_process_weights(self, layer):
        dev = layer.weight.device
        gs2 = layer.weight_scale_2.max().to(torch.float32)                      # scalar global
        wscale = (layer.weight_scale.data.to(torch.float32) * gs2).to(torch.bfloat16)  # [N, K/16]
        layer.nvfp4_wscale_nt = wscale.t().contiguous()                         # [K/16, N] bf16
        layer.nvfp4_group_size = int(self.quant_config.group_size)             # 16
        layer.nvfp4_N = int(layer.weight.shape[0])
        # Free the now-folded scales (keep `weight`: the NT view aliases its storage).
        for nm in ("weight_scale", "weight_scale_2", "input_scale"):
            if hasattr(layer, nm) and getattr(layer, nm) is not None:
                setattr(layer, nm, torch.nn.Parameter(
                    torch.empty(0, dtype=torch.float32, device=dev), requires_grad=False))
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        wt = layer.weight.data.t()
        assert wt.stride()[0] == 1, f"[nvfp4-shim] weight NT view not stride0==1 (got {wt.stride()})"
        print(f"[nvfp4-shim] NVFP4 layer ready N={layer.nvfp4_N} K2={layer.weight.shape[1]} "
              f"g={layer.nvfp4_group_size}", flush=True)

    def _nvfp4_apply(self, layer, x, bias=None):
        orig = x.shape
        x2 = x.reshape(-1, orig[-1]).to(torch.bfloat16).contiguous()           # [M, K] bf16
        out = torch.ops._xpu_C.nvfp4_gemm_w4a16(
            x2, layer.weight.data.t(), bias, layer.nvfp4_wscale_nt, layer.nvfp4_group_size)
        if _DBG["on"] and _DBG["n"] < 60:
            n = _DBG["n"]; _DBG["n"] += 1
            print(f"[nvfp4-dbg] call={n:>3} M={x2.shape[0]} K={orig[-1]} N={layer.nvfp4_N} "
                  f"out_absmax={out.abs().max().item():.4g} "
                  f"out_bad={bool(torch.isnan(out).any() or torch.isinf(out).any())}", flush=True)
        return out.to(x.dtype).reshape(*orig[:-1], layer.nvfp4_N)

    ModelOptFp4LinearMethod.process_weights_after_loading = _nvfp4_process_weights
    ModelOptFp4LinearMethod.apply = _nvfp4_apply
    print("[nvfp4-shim] (2) ModelOptFp4LinearMethod -> XPU nvfp4_gemm_w4a16 (4-bit resident, bf16 acts)", flush=True)

    # ---- (3) FP8 attention linear -> dequant-at-load to bf16 (XPU-safe, no cutlass) --------------
    # Per-tensor E4M3 weight with a per-logical-width static weight_scale. Dequant once at load
    # (real = fp8 * weight_scale, per partition) to a bf16 [N, K] weight; forward is plain F.linear
    # with bf16 activations (input_scale unused). Robust on XPU; native fp8 compute is an OPEN win.
    def _fp8_process_weights(self, layer):
        w = layer.weight.data                                                  # fp8_e4m3 [N, K]
        ws = layer.weight_scale.data.reshape(-1).to(torch.float32)             # [P] per logical width
        lw = list(layer.logical_widths)
        if len(lw) == 1 or ws.numel() == 1:
            deq = (w.to(torch.float32) * ws.reshape(-1)[0]).to(torch.bfloat16)
        else:
            parts = torch.split(w, lw, dim=0)
            deq = torch.cat([p.to(torch.float32) * ws[i] for i, p in enumerate(parts)], dim=0).to(torch.bfloat16)
        layer.weight = torch.nn.Parameter(deq.contiguous(), requires_grad=False)  # [N, K] bf16
        for nm in ("weight_scale", "input_scale"):
            if hasattr(layer, nm) and getattr(layer, nm) is not None:
                setattr(layer, nm, torch.nn.Parameter(
                    torch.empty(0, dtype=torch.float32, device=w.device), requires_grad=False))
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        print(f"[nvfp4-shim] FP8->bf16 layer ready N={deq.shape[0]} K={deq.shape[1]}", flush=True)

    def _fp8_apply(self, layer, x, bias=None):
        return F.linear(x.to(torch.bfloat16), layer.weight, bias).to(x.dtype)

    ModelOptFp8LinearMethod.process_weights_after_loading = _fp8_process_weights
    ModelOptFp8LinearMethod.apply = _fp8_apply
    print("[nvfp4-shim] (3) ModelOptFp8LinearMethod -> XPU dequant-at-load bf16 F.linear", flush=True)

    _STATE["installed"] = True
    print("[nvfp4-shim] installed: ModelOpt NVFP4 MIXED_PRECISION on sglang-XPU "
          "(W4A16_NVFP4 -> nvfp4_gemm_w4a16, FP8 -> bf16 dequant)", flush=True)


# Auto-install at import (mirrors w4a8_shim); gated so a bare import is a no-op unless enabled.
if os.environ.get("B70_XPU_NVFP4") == "1":
    try:
        install()
    except Exception as _e:
        print(f"[nvfp4-shim] auto-install FAILED: {_e}", flush=True)
