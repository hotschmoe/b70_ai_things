# quark_moe_int8.py -- DRAFT / UNTESTED (2026-06-29). Route A: W8A8 int8 MoE on sglang-XPU (B70).
#
# WHAT THIS IS
#   The sglang LOADER for the Qwen3.6-35B-A3B Quark W8A8 INT8 checkpoint (256 experts, top-8,
#   shared expert + GDN hybrid attention). NOT a kernel build: sglang already ships the pure-Triton
#   `fused_moe_kernel` `use_int8_w8a8` path in-tree (see SGLANG_MOE_PLAN.md sec 1). This file only
#   provides the missing LOADER + dispatch so that int8 expert weights flow into that kernel, plus a
#   dense int8->bf16 dequant fallback for the non-MoE linears (linear_attn.*, mlp.shared_expert.*).
#
# WHY A PATCH IS NEEDED
#   sglang's stock Quark config (sglang/srt/layers/quantization/quark/quark.py) only dispatches FP8 and
#   MXFP4 schemes:
#     - get_moe_scheme()         quark.py:477-502  -> RuntimeError("Unsupported FusedMoe scheme") for int8
#     - _get_scheme_from_config() quark.py:434-460 -> NotImplementedError for int8 linears
#   So an int8 Quark checkpoint hard-fails on load. This module's install() monkeypatches
#   QuarkConfig.get_quant_method to intercept the int8 layers and route them to our methods, falling
#   back to the stock dispatch for everything else (fp8/mxfp4/excluded). This mirrors how the vLLM side
#   patched its Quark config (rdy_to_serve/vllm/qwen36-35b-a3b-w8a8/patches/quark.py).
#
# THE TWO CONFIG-DETECTION PATHS (both produce the SAME runtime layout)
#   (1) QUARK format (what we have): config.json quantization_config.quant_method == "quark";
#       global_quant_config.weight = {int8, per_channel, ch_axis=0, static, symmetric},
#       input_tensors      = {int8, per_channel, ch_axis=1, dynamic,  symmetric}.
#       -> handled here by patching QuarkConfig (install()).
#   (2) COMPRESSED-TENSORS W8A8 (we can also produce this with llmcompressor): quant_method ==
#       "compressed-tensors", weights CHANNEL static + activations TOKEN dynamic.
#       -> sglang's NATIVE W8A8Int8MoEMethod (sglang/srt/layers/quantization/w8a8_int8.py:238-387)
#          ALREADY handles the MoE for that format via the same Triton runner; only the dense int8
#          linears need the XPU fallback (already shipped as sglang/patches/w8a8_shim.py, which patches
#          CompressedTensorsW8A8Int8). So for path (2) you typically DON'T need this file's MoE method;
#          you need w8a8_shim for the linears + the native MoE method for the experts. We keep
#          Int8MoEMethod here anyway as the single source of truth that BOTH paths can import.
#
# REFERENCES MIRRORED (file:line)
#   - sglang native int8 MoE method:  sglang/srt/layers/quantization/w8a8_int8.py:238-387
#       (create_weights :252, process_weights :316, create_moe_runner :329,
#        get_triton_quant_info :335 -> use_int8_w8a8=True/per_channel_quant=True, apply :347)
#   - sglang Triton MoE runner:       sglang/srt/layers/moe/moe_runner/triton.py:53-71,129-166
#       (TritonMoeQuantInfo carries use_int8_w8a8; _fused_moe_kernel_sequence consumes it)
#   - sglang Triton int8 kernel:      sglang/srt/layers/moe/moe_runner/triton_utils/
#       fused_moe_triton_kernels.py:324 fused_moe_kernel, :374 use_int8_w8a8 constexpr,
#       :560-606 int8 tl.dot->int32 + per-channel dequant (a_scale*b_scale);
#       :778-785 dynamic per-token activation int8 quant (per_token_quant_int8) inside the kernel.
#   - vLLM live int8 MoE (port ref):  /mnt/vm_8tb/b70/build/vllm/vllm/.../quark/quark_moe.py:518-815
#       (QuarkW8A8Int8MoEMethod: per-channel scale [E,N]->[E,N,1], a_scale=None dynamic)
#   - vLLM dense int8->bf16 dequant:  rdy_to_serve/vllm/qwen36-35b-a3b-w8a8/patches/quark.py:109-178
#       (QuarkW8A8Int8DequantXPU) + int8 detect :448-474,:501-530
#   - hook pattern (how to plug in):  sglang/patches/awq.py:442 AWQMoEMethod,
#       :138-202 get_quant_method dispatch by isinstance(layer, FusedMoE/LinearBase)
#   - dense fused oneDNN reuse:       sglang/patches/w8a8_shim.py (int8_gemm_w8a16 decode /
#       int8_gemm_w8a8 prefill); B70_XPU_W8A8_FUSED=1.
#
# STATUS: DRAFT. Gated on the Route-A probe (research/w8a8/sglang_moe_int8_probe.py) proving the int8
#   fused_moe Triton kernel codegens + runs on triton-xpu (B70). Do not serve before the probe is green.

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# int8 W8A8 detection helpers (mirror vLLM quark.py:448-474 _is_static_tensor_w8a8 and
# :501-530 _is_dynamic_per_token_w8a8). The 35B Quark ckpt is the DYNAMIC per-token case.
# --------------------------------------------------------------------------------------------------
def _is_int8_w8a8(weight_cfg: Optional[Dict[str, Any]], input_cfg: Optional[Dict[str, Any]]) -> bool:
    """True for int8 weight + int8 activation, per-(tensor|channel) weight, symmetric weight.

    Covers both static-per-tensor-act and dynamic-per-token-act. The Triton kernel runs symmetric
    int8 regardless; activation scale source (static vs in-kernel dynamic) is decided in
    get_triton_quant_info via a1/a2_scale = None for dynamic.
    """
    if weight_cfg is None or input_cfg is None:
        return False
    is_int8 = weight_cfg.get("dtype") == "int8" and input_cfg.get("dtype") == "int8"
    weight_ok = weight_cfg.get("qscheme") in ("per_tensor", "per_channel")
    weight_sym = weight_cfg.get("symmetric") is True
    return bool(is_int8 and weight_ok and weight_sym)


def _input_is_dynamic(input_cfg: Optional[Dict[str, Any]]) -> bool:
    return bool(input_cfg and input_cfg.get("is_dynamic") is True)


def _weight_is_per_channel(weight_cfg: Optional[Dict[str, Any]]) -> bool:
    return bool(weight_cfg and weight_cfg.get("qscheme") == "per_channel")


# --------------------------------------------------------------------------------------------------
# A. Int8MoEMethod -- the routed-expert loader. Mirrors sglang native W8A8Int8MoEMethod
#    (w8a8_int8.py:238-387) AND vLLM QuarkW8A8Int8MoEMethod (quark_moe.py:518-815).
#    It is a FusedMoEMethodBase, hooked in exactly like AWQMoEMethod (awq.py:442).
# --------------------------------------------------------------------------------------------------
def _make_int8_moe_method_cls():
    """Build the class lazily so importing this module on the host (no sglang) does not fail."""
    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
    from sglang.srt.utils import set_weight_attrs

    class Int8MoEMethod(FusedMoEMethodBase):
        """W8A8 INT8 routed-expert method for sglang (Triton fused_moe use_int8_w8a8 path).

        Weight  : per-channel, static, symmetric int8  -> w13/w2 _weight [int8], _weight_scale [E,N,1]
        Activation: per-token, dynamic, symmetric int8 -> quantized INSIDE the kernel (a_scale = None)
        """

        def __init__(self, quant_config, weight_cfg=None, input_cfg=None):
            self.quant_config = quant_config
            self.weight_cfg = weight_cfg or {}
            self.input_cfg = input_cfg or {}
            # dynamic per-token activation -> no static input scale (mirror quark_moe.py:768-777)
            self.static_input_scales = not _input_is_dynamic(self.input_cfg)
            self.per_channel = _weight_is_per_channel(self.weight_cfg)

        # create_weights signature matches FusedMoE (mirror w8a8_int8.py:252, NPU moe scheme :55).
        def create_weights(
            self,
            layer: torch.nn.Module,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ):
            from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

            # WEIGHTS -- [E, 2I, H] and [E, H, I] int8 (mirror quark_moe.py:548-570, w8a8_int8.py:266-288)
            w13_weight = torch.nn.Parameter(
                torch.empty(num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=torch.int8),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            w2_weight = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, intermediate_size_per_partition, dtype=torch.int8),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)

            # WEIGHT SCALES -- per-channel [E, N, 1] (mirror w8a8_int8.py:290-308; quark_moe.py:726-738
            # reshapes Quark's 2D [E,N] -> 3D [E,N,1]. sglang's CHANNEL weight_loader fills [E,N,1].)
            if self.per_channel:
                w13_weight_scale = torch.nn.Parameter(
                    torch.ones(num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32),
                    requires_grad=False,
                )
                w2_weight_scale = torch.nn.Parameter(
                    torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
                    requires_grad=False,
                )
                extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
            else:
                # per-tensor weight (not our ckpt; kept for completeness, mirror quark_moe.py:593-609)
                w13_weight_scale = torch.nn.Parameter(
                    torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
                )
                w2_weight_scale = torch.nn.Parameter(
                    torch.ones(num_experts, dtype=torch.float32), requires_grad=False
                )
                extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

            # INPUT SCALES -- dynamic per-token => None (kernel quantizes activations). Mirror
            # w8a8_int8.py:310-314 / quark_moe.py:626-628.
            if self.static_input_scales:
                w13_input_scale = torch.nn.Parameter(
                    torch.ones(num_experts, dtype=torch.float32), requires_grad=False
                )
                w2_input_scale = torch.nn.Parameter(
                    torch.ones(num_experts, dtype=torch.float32), requires_grad=False
                )
                layer.register_parameter("w13_input_scale", w13_input_scale)
                layer.register_parameter("w2_input_scale", w2_input_scale)
                set_weight_attrs(w13_input_scale, extra_weight_attrs)
                set_weight_attrs(w2_input_scale, extra_weight_attrs)
            else:
                layer.w13_input_scale = None
                layer.w2_input_scale = None

            # Quark ckpts also carry zero-points (symmetric -> all zero). The HF/sglang FusedMoE
            # weight_loader does NOT know these names; they are dropped at load (the loader only maps
            # weight + weight_scale + optional input_scale). If the loader complains about unexpected
            # *zero_point tensors, add them to the model's exclude/skip list (they are unused: the
            # kernel is symmetric). See SGLANG_MOE_PLAN.md "open risks".

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # Mirror w8a8_int8.py:316-327. Per-channel scales already [E,N,1] from the loader; if a
            # 2D [E,N] slipped through (some loaders), promote to [E,N,1] (mirror quark_moe.py:726-738).
            if self.per_channel:
                for attr in ("w13_weight_scale", "w2_weight_scale"):
                    p = getattr(layer, attr, None)
                    if p is not None and p.dim() == 2:
                        setattr(
                            layer,
                            attr,
                            torch.nn.Parameter(p.data.unsqueeze(-1).contiguous(), requires_grad=False),
                        )
            layer.w13_weight = torch.nn.Parameter(layer.w13_weight.data, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(layer.w2_weight.data, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.data, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(layer.w2_weight_scale.data, requires_grad=False)

        # create_moe_runner / apply: mirror w8a8_int8.py:329-387 exactly (Triton backend).
        def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig):
            self.moe_runner_config = moe_runner_config
            self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

        def get_triton_quant_info(self, layer: torch.nn.Module) -> "TritonMoeQuantInfo":
            return TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_int8_w8a8=True,            # <-- the whole point (kernel constexpr, kernels:374)
                per_channel_quant=self.per_channel,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=getattr(layer, "w13_input_scale", None),   # None -> dynamic per-token in-kernel
                a2_scale=getattr(layer, "w2_input_scale", None),
            )

        def apply(self, layer: torch.nn.Module, dispatch_output):
            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    return Int8MoEMethod


# --------------------------------------------------------------------------------------------------
# B. Int8DequantLinear -- dense int8 linears (linear_attn.*, mlp.shared_expert.*). XPU has no int8
#    scaled-mm in sgl_kernel (w8a8_int8.py:46 imports int8_scaled_mm only under _is_cuda), so we
#    dequant per-channel symmetric int8 -> bf16 ONCE at load and run a plain GEMM (== W8A16, slightly
#    MORE accurate than the ckpt's W8A8, correctness-first). Mirrors vLLM QuarkW8A8Int8DequantXPU
#    (rdy_to_serve/vllm/qwen36-35b-a3b-w8a8/patches/quark.py:109-178).
#
#    OPT-IN FAST PATH: set B70_XPU_W8A8_FUSED=1 (+ the built _xpu_C.so via B70_XPU_C_SO) to instead
#    route these linears to the fused oneDNN ops (int8_gemm_w8a16 decode / int8_gemm_w8a8 prefill)
#    exactly as sglang/patches/w8a8_shim.py does for the dense 27B model. Dequant stays the DEFAULT
#    because on this MoE the linears are a minority and int8 linear was NOT a serve win (vLLM finding,
#    rdy_to_serve/.../patches/quark.py:54-55). Keeping dequant default = fewer moving parts for the
#    first green serve; flip to fused once Route-A MoE is proven.
# --------------------------------------------------------------------------------------------------
def _make_int8_dequant_linear_cls():
    from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
    from sglang.srt.layers.quantization.base_config import LinearMethodBase

    class Int8DequantLinear(LinearMethodBase):
        def __init__(self, quant_config=None, prefix=""):
            self.quant_config = quant_config
            self.prefix = prefix
            self.out_dtype = torch.get_default_dtype()

        def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: List[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ):
            # Mirror w8a8_int8.py:174-203 (int8 weight + per-channel scale params) and
            # vLLM quark.py:130-157.
            self.out_dtype = params_dtype
            weight_loader = extra_weight_attrs.get("weight_loader")
            if os.environ.get("B70_MOE_DEBUG") == "1" and weight_loader is not None:
                _wl, _pfx = weight_loader, self.prefix

                def weight_loader(param, loaded_weight, *a, **k):  # noqa: F811
                    try:
                        return _wl(param, loaded_weight, *a, **k)
                    except Exception as e:
                        print(f"[MOEDBG] {_pfx} param={tuple(param.shape)} loaded={tuple(loaded_weight.shape)} "
                              f"args={a} opp={list(output_partition_sizes)} ispp={input_size_per_partition} "
                              f"-> {type(e).__name__}: {e}", flush=True)
                        raise
            layer.logical_widths = output_partition_sizes
            weight = ModelWeightParameter(
                data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)
            # The Quark checkpoint stores EVERY weight_scale as 1-D [N] (per-output-channel). sglang's
            # generic linear weight_loader does expert/param.copy_(loaded) WITHOUT the (N,)->(N,1) reshape
            # that ChannelQuantScaleParameter would otherwise do (the GDN merged projections even strip
            # that loader). A 2-D (N,1) param then shape-asserts against the 1-D [N] checkpoint scale (or
            # broadcasts to [N,N]). So register the scale 1-D for ALL dense int8 linears (self_attn,
            # shared_expert, and the GDN in_proj_*); process_weights reshapes to (N,1) for the dequant.
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(sum(output_partition_sizes), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # Dequant int8 [N,K] * per-channel scale[N,1] -> bf16, replace the int8 param. Compute
            # outside inference_mode so the dequant tensor carries a version counter (mirror
            # vLLM quark.py:159-173 -- avoids torch.compile functionalization tripping on it).
            w = layer.weight.data  # int8 [N, K]
            with torch.inference_mode(False), torch.no_grad():
                ws = layer.weight_scale.data.to(torch.float32).reshape(-1, 1)
                w_deq = (w.to(torch.float32) * ws).to(self.out_dtype).contiguous().clone()
            layer.weight = torch.nn.Parameter(w_deq, requires_grad=False)
            if hasattr(layer, "weight_scale"):
                try:
                    delattr(layer, "weight_scale")
                except Exception:
                    layer.weight_scale = None
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()

        def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    return Int8DequantLinear


# --------------------------------------------------------------------------------------------------
# install() -- monkeypatch sglang's QuarkConfig so the int8 Quark checkpoint dispatches our methods.
#   Intercepts BEFORE the stock get_moe_scheme/_get_scheme_from_config raise on int8. Falls back to
#   the original dispatch for fp8/mxfp4/excluded layers. Call this from the serve entrypoint (or a
#   sitecustomize/launch shim) BEFORE the model is built, same as w8a8_shim.install().
# --------------------------------------------------------------------------------------------------
def install():
    # XPU-safe dynamic per-token int8 activation quant FIRST. sglang's stock per_token_quant_int8 uses
    # tl.extra.cuda.libdevice.round, which does NOT link on triton-xpu -> the int8 fused-MoE kernel dies
    # at launch (ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED). Proven by sglang_moe_int8_probe.py. Patch it
    # before any int8 MoE forward. See sglang/patches/int8_actquant_xpu.py.
    try:
        import int8_actquant_xpu
        int8_actquant_xpu.install()
    except Exception as e:
        print(f"[quark_moe_int8] WARN: int8_actquant_xpu install failed (int8 MoE will fail on XPU): {e}", flush=True)

    # Quark stores MoE expert weight_scale as 1-D [N]; sglang's FusedMoE per-channel scale loader
    # (_load_per_channel_weight_scale) expects [N,1] (compressed-tensors style) and does
    # expert_data.copy_(loaded_weight) -> a 1-D [N] broadcasts to [N,N] and fails. Unsqueeze 1-D scales
    # to [N,1] so both the w2 (direct copy) and w1/w3 (_load_w13 narrow on dim 0) paths match.
    try:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE as _FME
        if not getattr(_FME, "_b70_pcs_patched", False):
            _orig_pcs = _FME._load_per_channel_weight_scale

            def _pcs(self, *args, **kwargs):
                if "loaded_weight" in kwargs:
                    lw = kwargs["loaded_weight"]
                    if hasattr(lw, "dim") and lw.dim() == 1:
                        kwargs["loaded_weight"] = lw.unsqueeze(-1)
                elif len(args) >= 4 and hasattr(args[3], "dim") and args[3].dim() == 1:
                    args = list(args); args[3] = args[3].unsqueeze(-1); args = tuple(args)
                return _orig_pcs(self, *args, **kwargs)

            _FME._load_per_channel_weight_scale = _pcs
            _FME._b70_pcs_patched = True
            print("[quark_moe_int8] patched FusedMoE._load_per_channel_weight_scale (1-D->[N,1] scale)", flush=True)
    except Exception as e:
        print(f"[quark_moe_int8] WARN: FusedMoE per-channel scale patch failed: {e}", flush=True)

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig
    from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    Int8MoEMethod = _make_int8_moe_method_cls()
    Int8DequantLinear = _make_int8_dequant_linear_cls()

    _orig_get_quant_method = QuarkConfig.get_quant_method

    def _patched_get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # Excluded layers (visual.*, shared_expert_gate, lm_head per config.json) -> unquantized.
        if should_ignore_layer(prefix, ignore=self.exclude_layers, fused_mapping=self.packed_modules_mapping):
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return _orig_get_quant_method(self, layer, prefix)

        # Resolve the matched quant spec for this layer (mirror quark.py:388-432 _find_matched_config).
        try:
            cfg = self._find_matched_config(prefix, layer)
            weight_cfg = cfg.get("weight")
            input_cfg = cfg.get("input_tensors")
        except Exception:
            weight_cfg = input_cfg = None

        if _is_int8_w8a8(weight_cfg, input_cfg):
            if isinstance(layer, FusedMoE):
                self._quantized_layers.add(prefix)
                logger.info("quark_moe_int8: int8 MoE method -> %s", prefix)
                return Int8MoEMethod(self, weight_cfg=weight_cfg, input_cfg=input_cfg)
            if isinstance(layer, LinearBase):
                self._quantized_layers.add(prefix)
                return Int8DequantLinear(self, prefix=prefix)

        # Not int8 (fp8/mxfp4) or non-quantizable -> stock dispatch.
        return _orig_get_quant_method(self, layer, prefix)

    QuarkConfig.get_quant_method = _patched_get_quant_method
    logger.info("quark_moe_int8: patched QuarkConfig.get_quant_method (int8 MoE + dense dequant)")
    print("[quark_moe_int8] installed: int8 Quark MoE loader + dense int8->bf16 dequant linear", flush=True)


if os.environ.get("B70_QUARK_MOE_INT8_AUTOINSTALL") == "1":
    try:
        install()
    except Exception as e:  # pragma: no cover - only meaningful inside the sglang image
        print(f"[quark_moe_int8] auto-install deferred ({e}); call install() after sglang import", flush=True)
