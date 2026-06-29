# w4a8_shim.py -- wire vLLM's torch.ops._xpu_C.int4_gemm_w4a8 (oneDNN int4-weight x
# int8-activation FUSED GEMM; verified 3.75x bf16 decode / 1.86x prefill on B70, relerr 2e-4)
# into sglang's compressed-tensors path on XPU as a DENSE W4A8 linear scheme.
#
# WHY A NEW SCHEME (not a monkeypatch of an existing one):
#   sglang has NO dense compressed-tensors W4A8-int8 scheme -- only an NPU MoE variant
#   (NPUCompressedTensorsW4A8Int8DynamicMoE). The dense linear picker
#   CompressedTensorsConfig._get_scheme_from_parts() never calls _is_dynamic_token_w4a8;
#   for a W4A8 int-quantized checkpoint it falls through every branch and RAISES
#   NotImplementedError("No compressed-tensors compatible scheme was found.").
#   So we (1) define a CompressedTensorsLinearScheme that wraps int4_gemm_w4a8,
#   (2) patch _get_scheme_from_parts to route W4A8 groups to it (reusing the already-present
#   detector self._is_dynamic_token_w4a8), and (3) spoof _check_scheme_supported (exactly
#   like w8a8_shim) so the scheme is accepted on XPU (the original calls
#   torch.cuda.get_device_capability() which throws "Torch not compiled with CUDA").
#
# CHECKPOINT LAYOUT (Qwen3.6-27B-W4A8-sqgptq-prepacked; compressed-tensors "int-quantized"):
#   per-Linear `weight`        I32  [N, K/8]   (8 signed int4 per int32; nibble = value+8; symmetric, zp=8)
#   per-Linear `weight_scale`  BF16 [N, K/g]   (group_size=128 along K)
#   weights: 4-bit group sym static; input_activations: 8-bit token dynamic sym; targets ["Linear"].
#   The op wants qweight [K/8, N] and wscale [K/g, N] -> we pass weight.t() / weight_scale.t().
#   Layout + .t() convention verified by sglang/int4_gemm_w4a8_probe.py.
#
# OP (verified signature; output dtype HARD-CODED float16 -> serve with --dtype float16):
#   int4_gemm_w4a8(actInt8[M,K], actScale[M,1] fp16, actZero[M,1] i32, qweight[K/8,N] i32,
#                  wscale[K/g,N], wzp=tensor([8]) i8, group_size:int, g_idx=None, bias=None) -> [M,N] fp16
#   Activation quant is OUTSIDE the op: dynamic per-token symmetric int8 (ported below from
#   vllm _xpu_ops.dynamic_per_token_int8_quant_ref). The op consumes `bias` internally
#   (we do NOT add it again), matching vLLM XPUW4A8IntLinearKernel.apply_weights.
#
# Gated opt-in via B70_XPU_W4A8=1. To wire into the baked chain, add a block to woq_shim.py
# mirroring the W8A8 one:  if os.environ.get("B70_XPU_W4A8")=="1": import w4a8_shim; w4a8_shim.install()
# (This module also self-installs at import when the env gate is set, so a bare `import w4a8_shim` works.)
import os

_STATE = {"installed": False}


def _load_op():
    """Make torch.ops._xpu_C.int4_gemm_w4a8 callable. Prefer the packaged extension (its
    ABI must match sglang's torch); fall back to a ctypes dlopen of the built _xpu_C*.so
    (path via B70_XPU_C_SO) with RTLD_GLOBAL so its sibling oneAPI libs resolve."""
    import torch

    if hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):
        return True
    # (a) packaged extension -- registers torch.ops._xpu_C on import
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except Exception as e:
        print(f"[w4a8-shim] vllm_xpu_kernels._xpu_C import failed ({e}); trying B70_XPU_C_SO", flush=True)
        so = os.environ.get("B70_XPU_C_SO")
        if so and os.path.exists(so):
            import ctypes

            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                print(f"[w4a8-shim] dlopen'd {so}", flush=True)
            except Exception as e2:
                print(f"[w4a8-shim] ctypes.CDLL({so}) failed: {e2}", flush=True)
        elif so:
            print(f"[w4a8-shim] B70_XPU_C_SO={so} does not exist", flush=True)
    return hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")


def _register_fake():
    """Best-effort fake/meta for int4_gemm_w4a8 so torch.compile / inductor-graph-partition
    can trace THROUGH the custom op (out: [M, N] float16; M=A.shape[0], N=B.shape[1]).
    Harmless if a native abstract impl is already registered."""
    import torch

    reg = getattr(torch.library, "register_fake", None) or getattr(
        torch.library, "impl_abstract", None
    )
    if reg is None or not hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):
        return

    def _fake(A_, A_scale, A_zp, B, B_scale, B_zp, group_size, g_idx=None, bias=None):
        return A_.new_empty((A_.shape[0], B.shape[1]), dtype=torch.float16)

    def _fake16(A_, B, bias, B_scale, B_zp, group_size, g_idx=None):
        return A_.new_empty((A_.shape[0], B.shape[1]), dtype=torch.float16)

    try:
        reg("_xpu_C::int4_gemm_w4a8", _fake)
        print("[w4a8-shim] registered fake for _xpu_C::int4_gemm_w4a8", flush=True)
    except (RuntimeError, ValueError) as e:
        print(f"[w4a8-shim] register_fake w4a8 skipped: {e}", flush=True)
    if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        try:
            reg("_xpu_C::int4_gemm_w4a16", _fake16)
            print("[w4a8-shim] registered fake for _xpu_C::int4_gemm_w4a16", flush=True)
        except (RuntimeError, ValueError) as e:
            print(f"[w4a8-shim] register_fake w4a16 skipped: {e}", flush=True)


def install():
    if _STATE["installed"]:
        return
    import torch

    try:
        from sglang.srt.utils import is_xpu

        if not is_xpu():
            print("[w4a8-shim] not XPU; skip", flush=True)
            return
    except Exception:
        return

    if not _load_op():
        raise RuntimeError(
            "[w4a8-shim] torch.ops._xpu_C.int4_gemm_w4a8 NOT FOUND. Build the op against "
            "sglang's torch and either install vllm_xpu_kernels or point B70_XPU_C_SO at the "
            "built _xpu_C*.so. See sglang/W4A8_PLAN.md (the v0230 .so is torch-2.11 ABI and "
            "will NOT load into the torch-2.12 sglang image)."
        )
    _register_fake()

    # --- force sglang's Qwen3_5TextConfig for model_type 'qwen3_5_text' ---
    # transformers ALSO knows 'qwen3_5_text' natively, so AutoConfig builds the TRANSFORMERS config,
    # which is NOT a Qwen3NextConfig subclass and lacks the derived `mamba2_cache_params` property.
    # sglang gates GDN on isinstance(text_config, Qwen3NextConfig|...) and needs mamba2_cache_params to
    # size the GDN state cache. Registering sglang's class in _CONFIG_REGISTRY makes get_config reload
    # via it (it derives layers_block_type + mamba2_cache_params), so the dense text serve is GDN-correct.
    try:
        from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY
        from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig as _Q35TC

        if _CONFIG_REGISTRY.get("qwen3_5_text") is not _Q35TC:
            _CONFIG_REGISTRY["qwen3_5_text"] = _Q35TC
            print(
                "[w4a8-shim] registered sglang Qwen3_5TextConfig for model_type qwen3_5_text",
                flush=True,
            )
    except Exception as e:
        print(f"[w4a8-shim] _CONFIG_REGISTRY qwen3_5_text register failed: {e}", flush=True)

    # --- register the text-only dense Qwen3_5ForCausalLM as a servable architecture ---
    # sglang's qwen3_5 EntryClass only exposes the MULTIMODAL Qwen3_5ForConditionalGeneration; the
    # text-only Qwen3_5ForCausalLM class exists but is NOT in the registry, so a text-only ckpt
    # (architectures=['Qwen3_5ForCausalLM']) fails with "no SGLang implementation". This W4A8 ckpt has
    # NO vision (re:.*visual.* excluded) + no preprocessor_config.json, so we serve it text-only via a
    # flattened config (text_config lifted to top level). Qwen3_5ForCausalLM.load_weights already maps
    # model.language_model.*->model.* and skips visual, so this is the correct loader. Inject the class.
    try:
        from sglang.srt.models.registry import ModelRegistry
        from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

        if "Qwen3_5ForCausalLM" not in ModelRegistry.models:
            ModelRegistry.models["Qwen3_5ForCausalLM"] = Qwen3_5ForCausalLM
            print(
                "[w4a8-shim] registered Qwen3_5ForCausalLM (text-only) in ModelRegistry",
                flush=True,
            )

        # Qwen3_5ForCausalLM.get_model_config_for_expert_location() blindly reads config.num_experts,
        # which a DENSE config lacks -> AttributeError at expert-location init. Dense Qwen3.6-27B has no
        # MoE experts; return None (the caller, _init_common, treats None as "no experts" and skips it).
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        def _eploc_dense_safe(cls, config):
            ne = getattr(config, "num_experts", None)
            if ne is None:
                return None
            return ModelConfigForExpertLocation(
                num_layers=config.num_hidden_layers,
                num_logical_experts=ne,
                num_groups=None,
            )

        Qwen3_5ForCausalLM.get_model_config_for_expert_location = classmethod(
            _eploc_dense_safe
        )

        # Qwen3_5ForCausalLM is FLAT (self.embed_tokens / self.layers / self.norm directly), but its
        # load_weights maps ckpt names model.language_model.* -> model.* (a leftover 'model.' prefix that
        # only matches the MULTIMODAL class where the LM is nested under self.model). Standalone, that
        # prefix matches NOTHING -> 1107 "not found in params_dict" -> weights+scales stay uninitialized
        # (scale=garbage -> 0*inf=NaN in the GEMM). Fix: pre-strip 'model.language_model.' to flat names
        # so the original loader's stacked-fusion + params_dict lookup hit the flat module tree.
        _orig_lw = Qwen3_5ForCausalLM.load_weights
        _PFX = "model.language_model."

        def _flat_load_weights(self, weights):
            def _gen():
                for name, w in weights:
                    if name.startswith(_PFX):
                        name = name[len(_PFX):]
                    yield name, w
            return _orig_lw(self, _gen())

        Qwen3_5ForCausalLM.load_weights = _flat_load_weights
        print(
            "[w4a8-shim] wrapped Qwen3_5ForCausalLM.load_weights (strip model.language_model. -> flat)",
            flush=True,
        )
    except Exception as e:
        print(f"[w4a8-shim] Qwen3_5ForCausalLM registry inject failed: {e}", flush=True)

    # model_runner.hybrid_gdn_config only recognizes the MULTIMODAL Qwen3_5Config / Qwen3_5MoeConfig as
    # a GDN hybrid model. Our flattened text config parses as Qwen3_5TextConfig, so sglang skips building
    # the HybridLinearAttnBackend (+ mamba/GDN state cache) and the 48 GDN linear-attn layers crash at
    # forward: radix_linear_attention falls to get_attn_backend().forward(layer, fb, mixed_qkv, a, b) but
    # gets a PLAIN full-attn backend (TypeError: forward() missing q,k,v). Patch the property to also
    # accept Qwen3_5TextConfig (it carries the same GDN dims the backend/cache need).
    try:
        import sglang.srt.model_executor.model_runner as _mr
        from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig as _Q35T

        _orig_hgc = _mr.ModelRunner.hybrid_gdn_config.fget
        _hgc_dbg = {"done": False}

        def _hgc(self):
            r = _orig_hgc(self)
            if r is not None:
                return r
            hf = self.model_config.hf_config
            cfg = hf.get_text_config() if hasattr(hf, "get_text_config") else hf
            ok = isinstance(cfg, _Q35T) or type(cfg).__name__ == "Qwen3_5TextConfig"
            if not ok and type(hf).__name__ == "Qwen3_5TextConfig":
                cfg, ok = hf, True
            if not _hgc_dbg["done"]:
                _hgc_dbg["done"] = True
                print(
                    f"[w4a8-shim][hgc] hf={type(hf).__name__} text={type(cfg).__name__} "
                    f"matched_GDN={ok}",
                    flush=True,
                )
            return cfg if ok else None

        _mr.ModelRunner.hybrid_gdn_config = property(_hgc)
        print(
            "[w4a8-shim] patched ModelRunner.hybrid_gdn_config to accept Qwen3_5TextConfig (GDN hybrid)",
            flush=True,
        )
    except Exception as e:
        print(f"[w4a8-shim] hybrid_gdn_config patch failed: {e}", flush=True)

    from sglang.srt.layers.parameter import (
        GroupQuantScaleParameter,
        ModelWeightParameter,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsLinearScheme,
    )

    _compile = os.environ.get("B70_W4A8_COMPILE") == "1"
    _DBG = {"on": os.environ.get("B70_W4A8_DEBUG") == "1", "n": 0}

    # --- activation quant: dynamic per-token SYMMETRIC int8 ---
    # Exactly mirrors the VERIFIED probe (sglang/int4_gemm_w4a8_probe.py act_q / w4a8_builtso_test.py):
    #   scale = amax/127 (clamp 1e-5); zero_point = 0 (sym); q = round(x/scale).clamp(-127,127) int8.
    # Returns (int8 [M,K], scale x.dtype [M,1] -> fp16 under --dtype float16, zero i32 [M,1]); all contiguous
    # (the op wants A_scale fp16 and contiguous A). Pure torch -> torch.compile-fuses the ~6 elementwise launches.
    def _quant_per_token_int8(x2):
        amax = x2.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
        scale = (amax / 127.0).to(x2.dtype)
        q = torch.round(x2 / scale).clamp_(-127, 127).to(torch.int8)
        zero = torch.zeros_like(amax, dtype=torch.int32)
        return q.contiguous(), scale.contiguous(), zero.contiguous()

    # torch.compile fuses the ~7 elementwise launches (min/max/abs/round/clamp/cast) into a
    # single kernel -> cuts the per-decode-layer launch overhead. Opt-in (B70_W4A8_COMPILE=1)
    # because dynamic M can trigger recompiles; under the bs=1 XPUGraph decode the shape is fixed.
    _quant = torch.compile(_quant_per_token_int8) if _compile else _quant_per_token_int8
    print(f"[w4a8-shim] act-quant path: {'torch.compile-fused' if _compile else 'eager'}", flush=True)

    class CompressedTensorsW4A8Int8XPU(CompressedTensorsLinearScheme):
        """Dense W4A8: int4 group-quantized weight x int8 per-token-dynamic activation,
        fused by oneDNN torch.ops._xpu_C.int4_gemm_w4a8."""

        def __init__(self, strategy, group_size, is_static_input_scheme, input_symmetric):
            self.strategy = strategy
            self.group_size = 128 if group_size is None else int(group_size)
            self.is_static_input_scheme = is_static_input_scheme
            self.input_symmetric = input_symmetric

        @classmethod
        def get_min_capability(cls) -> int:
            return 1  # XPU oneDNN path; the capability gate is spoofed below anyway

        def create_weights(
            self,
            layer,
            output_partition_sizes,
            input_size_per_partition,
            params_dtype,
            weight_loader,
            input_size=None,
            output_size=None,
            **kwargs,
        ):
            out_features = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            gs = self.group_size if self.group_size != -1 else input_size_per_partition
            assert input_size_per_partition % 8 == 0, (
                f"[w4a8-shim] in_per_partition {input_size_per_partition} not %8"
            )
            assert input_size_per_partition % gs == 0, (
                f"[w4a8-shim] in_per_partition {input_size_per_partition} not %group {gs}"
            )

            # PREPACKED int4 weight: on-disk key `weight` is int32 [N, K/8] (loaded directly,
            # no on-load pack -> avoids the ~28 GiB unpacked-int8 GPU transient). Plain
            # ModelWeightParameter (mirrors vLLM's VLLM_W4A8_PREPACKED path): output_dim=0
            # is the unpacked N (fused/column shards are clean); input_dim=1 is K/8.
            weight = ModelWeightParameter(
                data=torch.empty(
                    out_features, input_size_per_partition // 8, dtype=torch.int32
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            # group scale `weight_scale` [N, K/g], stored in params_dtype so it matches the
            # activation dtype the op expects (on-disk bf16 is cast on load if --dtype float16).
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    out_features, input_size_per_partition // gs, dtype=params_dtype
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

        def process_weights_after_loading(self, layer) -> None:
            w = layer.weight.data            # int32 [N, K/8]  (the real packed int4 weight)
            s = layer.weight_scale.data      # params_dtype [N, K/g]
            dev = w.device
            # BOTH ops want B in NT format: qweight [K/8, N] as a VIEW with stride[0]==1 -> w.t()
            # (NO .contiguous() -- contiguous would give stride[0]==N and the oneDNN kernel rejects it;
            #  the VERIFIED probe passes the bare w.t() view). The view shares storage with w, so the
            #  weight stays alive; we drop the redundant Parameter wrapper but keep the storage.
            layer.qweight_t = w.t()                  # [K/8, N] int32 VIEW (stride[0]==1)
            # B_scale -> fp16 (the ops run fp16 internally; matches the fp16 activation we feed). Small
            # tensor, so the cast is cheap and removes any bf16/fp16 ambiguity when --dtype bfloat16.
            layer.wscale_t = s.t().to(torch.float16).contiguous()  # [K/g, N] fp16
            layer.wzp = torch.tensor([8], dtype=torch.int8, device=dev)  # 1-D -> symmetric int4 zp
            # free the redundant scale Parameter (we use wscale_t). Do NOT free `weight`: qweight_t
            # is a view of its storage; replacing the Parameter would orphan nothing but keeps it tidy.
            layer.weight_scale = torch.nn.Parameter(
                torch.empty(0, dtype=s.dtype, device=dev), requires_grad=False
            )
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
            assert layer.qweight_t.stride()[0] == 1, (
                f"[w4a8-shim] qweight_t NOT NT-format (stride0={layer.qweight_t.stride()[0]})"
            )
            if _DBG["on"]:
                ws_bad = bool(
                    torch.isnan(layer.wscale_t).any() or torch.isinf(layer.wscale_t).any()
                )
                zin = torch.zeros(
                    1, layer.qweight_t.shape[0] * 8, device=dev, dtype=torch.float16
                )
                zout = torch.ops._xpu_C.int4_gemm_w4a16(
                    zin, layer.qweight_t, None, layer.wscale_t, layer.wzp,
                    self.group_size, None,
                )
                print(
                    f"[w4a8-load-dbg] N={layer.qweight_t.shape[1]} K8={layer.qweight_t.shape[0]} "
                    f"wscale_bad={ws_bad} wscale_absmax={layer.wscale_t.abs().max().item():.4g} "
                    f"zero_in_out_finite={bool(torch.isfinite(zout).all())}",
                    flush=True,
                )
            print(
                f"[w4a8-shim] W4A8 layer ready N={layer.qweight_t.shape[1]} "
                f"K8={layer.qweight_t.shape[0]} g={self.group_size} "
                f"qw_stride={tuple(layer.qweight_t.stride())}",
                flush=True,
            )

        def apply_weights(self, layer, x, bias=None):
            # HYBRID dispatch (verified faster than int4-woqgemm on BOTH halves, same stored weights):
            #   decode  M==1 -> int4_gemm_w4a16 (fp16 act, NO act-quant): 0.079ms, 1.83x faster than woqgemm
            #   prefill M>1  -> int4_gemm_w4a8  (per-token int8 act, compile-fused quant): 1.9x faster
            # int8 acts do NOT help the bandwidth-bound M==1 decode, so decode uses the fp16-act op.
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])               # [M, K]
            M = x2.shape[0]
            path = "w4a16" if M == 1 else "w4a8"
            # Both ops are fp16-only (emit fp16). The model may run bf16 (the GDN/attn weights are
            # unquantized bf16 -> serve --dtype bfloat16 so the triton conv1d/GDN kernels match); so we
            # convert the activation to fp16 at the op boundary and cast the result back to x.dtype.
            xf = x2.to(torch.float16).contiguous()
            if M == 1:
                out = torch.ops._xpu_C.int4_gemm_w4a16(
                    xf,
                    layer.qweight_t,                   # [K/8, N] int32 (NT view)
                    bias,                              # bias|None (3rd arg; op applies internally)
                    layer.wscale_t,                    # [K/g, N]
                    layer.wzp,                         # [1] int8 -> symmetric
                    self.group_size,
                    None,                              # g_idx (no desc_act)
                )                                      # [M, N] float16
            else:
                q, x_scale, x_zero = _quant(xf)        # int8 [M,K], scale fp16 [M,1], zero i32 [M,1]
                out = torch.ops._xpu_C.int4_gemm_w4a8(
                    q,
                    x_scale,
                    x_zero,
                    layer.qweight_t,                   # [K/8, N] int32 (NT view)
                    layer.wscale_t,                    # [K/g, N]
                    layer.wzp,                         # [1] int8 -> symmetric
                    self.group_size,
                    None,                              # g_idx (no desc_act)
                    bias,                              # op applies bias internally (last arg)
                )                                      # [M, N] float16
            out = out.to(x.dtype)
            if _DBG["on"] and _DBG["n"] < 80:
                n = _DBG["n"]; _DBG["n"] += 1
                in_bad = bool(torch.isnan(x2).any() or torch.isinf(x2).any())
                o_bad = bool(torch.isnan(out).any() or torch.isinf(out).any())
                print(
                    f"[w4a8-dbg] call={n:>3} path={path} M={M} K={layer.qweight_t.shape[0]*8} "
                    f"N={layer.qweight_t.shape[1]} in_absmax={x2.abs().max().item():.4g} "
                    f"in_bad={in_bad} out_absmax={out.abs().max().item():.4g} out_bad={o_bad}",
                    flush=True,
                )
            return out.reshape(*orig[:-1], -1)

    # --- spoof the capability check (same approach as w8a8_shim) ---
    # The original _check_scheme_supported does DeviceCapability(*torch.cuda.get_device_capability())
    # which throws on XPU. Emulate capability 90 for the scheme-support check ONLY (no global sm90 fake).
    def _xpu_check_scheme_supported(self, min_capability, error=True):
        supported = 90 >= int(min_capability)
        if error and not supported:
            raise RuntimeError(
                f"[w4a8-shim] scheme min_capability {min_capability} > emulated XPU cap 90"
            )
        return supported

    CompressedTensorsConfig._check_scheme_supported = _xpu_check_scheme_supported

    # --- route W4A8 dense-linear groups to our scheme ---
    # _get_scheme_from_parts never handles W4A8 dense linears (only the NPU MoE picker does),
    # so we wrap it: detect W4A8 via the existing self._is_dynamic_token_w4a8, else delegate.
    _orig_parts = CompressedTensorsConfig._get_scheme_from_parts

    def _patched_get_scheme_from_parts(self, weight_quant, input_quant):
        try:
            if (
                weight_quant is not None
                and input_quant is not None
                and self._is_dynamic_token_w4a8(weight_quant, input_quant)
            ):
                return CompressedTensorsW4A8Int8XPU(
                    strategy=weight_quant.strategy,
                    group_size=getattr(weight_quant, "group_size", 128),
                    is_static_input_scheme=not input_quant.dynamic,
                    input_symmetric=input_quant.symmetric,
                )
        except Exception as e:
            print(f"[w4a8-shim] W4A8 detect failed, delegating to original picker: {e}", flush=True)
        return _orig_parts(self, weight_quant, input_quant)

    CompressedTensorsConfig._get_scheme_from_parts = _patched_get_scheme_from_parts

    _STATE["installed"] = True
    print(
        "[w4a8-shim] installed: CompressedTensorsW4A8Int8XPU -> "
        "torch.ops._xpu_C.int4_gemm_w4a8 (XPU oneDNN int4w x int8a)",
        flush=True,
    )


# Auto-install at import (mirrors woq_shim) so a bare `import w4a8_shim` works; gated by env.
if os.environ.get("B70_XPU_W4A8") == "1":
    try:
        install()
    except Exception as _e:
        print(f"[w4a8-shim] auto-install FAILED: {_e}", flush=True)
