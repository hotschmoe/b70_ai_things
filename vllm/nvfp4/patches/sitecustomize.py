# sitecustomize.py -- NVFP4-on-XPU shim (vllm/nvfp4 experiment, 2026-07-04).
#
# vLLM v0.24.0 ships a complete ModelOpt NVFP4 load path (ModelOptNvFp4LinearMethod)
# but _POSSIBLE_NVFP4_KERNELS in vllm/model_executor/kernels/linear/__init__.py has
# NO PlatformEnum.XPU entry, so any NVFP4 checkpoint dies at engine init with
# "Failed to find a kernel that can implement the NVFP4 linear layer".
# The in-tree EmulationNvFp4LinearKernel is device-generic (LUT nibble unpack +
# float8_e4m3fn view + pure-torch math; its triton fast paths are gated behind
# current_platform.is_cuda_alike() and never run on XPU), so the minimal unlock
# is a registry entry -- plus a faster dequant-at-load variant defined here.
#
# Modes (env NVFP4_XPU_MODE):
#   emul    - stock EmulationNvFp4LinearKernel: weight dequant EVERY forward and
#             activation fake-quant to nvfp4 (true W4A4 emulation; slow; the
#             numerics reference).
#   dequant - (default) one-time NVFP4 -> BF16 weight dequant at load; forward is
#             a plain F.linear (W4A16-style: full-precision activations; fast;
#             weights cost bf16 bytes in VRAM, ~15GB for 8B).
import os
import sys

try:
    import torch
    from vllm.model_executor.kernels import linear as _linmod
    from vllm.model_executor.kernels.linear.nvfp4.emulation import (
        EmulationNvFp4LinearKernel,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.platforms.interface import PlatformEnum

    _MODE = os.environ.get("NVFP4_XPU_MODE", "dequant")

    class XPUDequantAtLoadNvFp4LinearKernel(EmulationNvFp4LinearKernel):
        """One-time NVFP4 -> BF16 weight dequant at load; plain F.linear after.

        At process_weights time the ModelOpt linear method has already renamed
        scales: layer.weight [N, K/2] uint8 (2 nibbles/byte), layer.weight_scale
        [N, K/16] float8_e4m3fn, layer.weight_global_scale scalar fp32.
        dequant = e2m1_lut[nibble] * e4m3(block_scale) * weight_global_scale.
        """

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
                dequantize_to_dtype,
                kE2M1ToFloat_handle,
            )

            kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(layer.weight.device)
            w = dequantize_to_dtype(
                layer.weight.data.view(torch.uint8),
                layer.weight_scale.data,
                layer.weight_global_scale.data.to(torch.float32),
                dtype=torch.bfloat16,
                block_size=16,
                swizzle=False,
            )
            replace_parameter(layer, "weight", w)
            # Free the block scales (bf16 weight no longer needs them).
            replace_parameter(
                layer,
                "weight_scale",
                torch.zeros(1, dtype=torch.float32, device=w.device),
            )

        def apply_weights(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    # E2M1 magnitude * 2 -> exact int8 codes (sign applied separately).
    _E2M1_INT8 = [0, 1, 2, 3, 4, 6, 8, 12]

    class XPUInt8XmxNvFp4LinearKernel(EmulationNvFp4LinearKernel):
        """W4A16-via-INT8-XMX: repack NVFP4 -> s8 weight + per-16-K-group bf16
        scale at load, then oneDNN int8_gemm_w8a16 (INT8 XMX) each forward.

        Needs the K-group-fixed nvfp4_kernel/_xpu_C.abi3.so mounted (the stock op
        applies square {g,g} groups and is numerically wrong for NVFP4). Weights
        stay int8 in VRAM (~half the bf16 dequant footprint).
        """

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            import vllm_xpu_kernels._xpu_C  # noqa: F401 (registers the op)

            dev = layer.weight.device
            packed = layer.weight.data  # [N, K/2] uint8
            N = packed.shape[0]
            lut = torch.tensor(_E2M1_INT8, dtype=torch.int8, device=dev)
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            both = torch.stack([lo, hi], dim=-1).reshape(N, -1)  # [N,K] low-first
            sign = torch.where((both & 0x8) != 0, -1, 1).to(torch.int8)
            w_int8 = sign * lut[(both & 0x7).long()]             # [N,K] exact s8
            wt = w_int8.t().contiguous()                          # [K,N] s8

            # scale [N, K/16] f8e4m3 -> [K/16, N] bf16, folded * ws2 / 2
            ws2 = layer.weight_global_scale.data.to(torch.float32)
            g = (layer.weight_scale.data.to(torch.float32) * ws2 / 2.0)  # [N,K/16]
            g = g.t().contiguous().to(torch.bfloat16)             # [K/16, N]

            replace_parameter(layer, "weight", wt)
            replace_parameter(layer, "weight_scale", g)

        def apply_weights(self, layer, x, bias=None):
            out_shape = (*x.shape[:-1], layer.weight.shape[1])
            x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
            y = torch.ops._xpu_C.int8_gemm_w8a16(
                x2, layer.weight, layer.weight_scale, bias
            )
            return y.reshape(out_shape)

    if _MODE == "emul":
        _kern = [EmulationNvFp4LinearKernel]
    elif _MODE == "dequant":
        _kern = [XPUDequantAtLoadNvFp4LinearKernel]
    elif _MODE == "int8xmx":
        _kern = [XPUInt8XmxNvFp4LinearKernel]
    else:
        raise ValueError(
            f"NVFP4_XPU_MODE={_MODE!r} not in (emul, dequant, int8xmx)"
        )

    _linmod._POSSIBLE_NVFP4_KERNELS[PlatformEnum.XPU] = _kern
    print(
        f"[nvfp4-shim] registered {_kern[0].__name__} for PlatformEnum.XPU "
        f"(NVFP4_XPU_MODE={_MODE})",
        file=sys.stderr,
        flush=True,
    )

    # ---- (2) tolerate shard_id on KV-cache scale loads ------------------------
    # qwen2.py load_weights routes k_proj.k_scale/v_proj.v_scale through the
    # stacked-params (qkv fusion) branch, which calls
    # weight_loader(param, weight, shard_id) -- but KVCacheScaleParameter's
    # loader is (param, weight) only -> TypeError on any qwen2-family ModelOpt
    # checkpoint that carries FP8 KV scales. Scales are all 1.0 in this ckpt
    # (and unused with kv_cache_dtype=auto), so dropping shard_id is lossless.
    from vllm.model_executor.layers.quantization.kv_cache import (
        KVCacheScaleParameter,
    )

    _orig_kv_loader = KVCacheScaleParameter.weight_loader

    @staticmethod
    def _kv_loader_tolerant(param, loaded_weight, *_shard_id):
        return _orig_kv_loader(param, loaded_weight)

    KVCacheScaleParameter.weight_loader = _kv_loader_tolerant
    print(
        "[nvfp4-shim] KVCacheScaleParameter.weight_loader now tolerates shard_id",
        file=sys.stderr,
        flush=True,
    )
except Exception as e:  # never break unrelated python processes in the container
    print(f"[nvfp4-shim] FAILED: {e!r}", file=sys.stderr, flush=True)
