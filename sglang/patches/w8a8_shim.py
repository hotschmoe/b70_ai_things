# w8a8_shim.py -- wire INT8 W8A8 into sglang's compressed-tensors W8A8-int8 scheme on XPU (B70).
#
# TWO paths, both gated by B70_XPU_W8A8=1 (installed from woq_shim):
#  (A) FUSED hybrid (NEW, B70_XPU_W8A8_FUSED=1, the fast path): our built oneDNN ops --
#        small rows (M<=B70_W8A16_M_MAX): int8_gemm_w8a16(x_f16, B_nt, wscale[N])
#        larger rows: dynamic_per_token_int8_quant -> int8_gemm_w8a8
#      B70_W8A16_M_MAX is default-off in the sense that unset preserves the old M==1-only route.
#      The currently validated range is 1..11; larger configured values fail closed.
#      Mirrors the W4A8 hybrid (woq_shim _XpuW4A8WoqKernel). Validated card-0: w8a8/w8a8_fused_probe.py
#      (decode 1.86-1.91x, prefill 1.95-2.07x bf16; matches/beats fp8 bar; int8-accurate).
#  (B) LEGACY _int_mm chain (default if FUSED unset): per-token int8 quant -> torch._int_mm -> dequant
#      (3 launches/layer, decode launch-bound ~0.8x bf16 eager).
# Needs the built _xpu_C.so (B70_XPU_C_SO) for path A; falls back to (B) if the ops don't load.
import os

_DBG = {"on": os.environ.get("B70_W8A8_DEBUG") == "1", "n": 0}
_ROUTE_COUNTS = {"w8a16": 0, "w8a8": 0}
_ROUTE_LOGGED = set()


def _w8a16_m_max(environ=None):
    """Return the strict W8A16 row threshold and whether it was configured."""
    environ = os.environ if environ is None else environ
    if "B70_W8A16_M_MAX" not in environ:
        return 1, False
    raw = environ["B70_W8A16_M_MAX"]
    if not raw or not raw.isascii() or not raw.isdecimal():
        raise RuntimeError(
            "B70_W8A16_M_MAX must be an ASCII decimal integer in [1, 11]"
        )
    value = int(raw, 10)
    if not 1 <= value <= 11:
        raise RuntimeError("B70_W8A16_M_MAX must be in the validated range [1, 11]")
    return value, True


def _use_w8a16(m, m_max):
    return 1 <= m <= m_max


def _record_w8a8_route(route, m, k, n, m_max):
    """Count every route and log each shape at its threshold relation once."""
    _ROUTE_COUNTS[route] += 1
    if m == 1:
        relation = "m1"
    elif m == m_max:
        relation = "at_max"
    elif m < m_max:
        relation = "below_max"
    else:
        relation = "above_max"
    signature = (route, relation, k, n)
    if signature in _ROUTE_LOGGED:
        return
    _ROUTE_LOGGED.add(signature)
    print(
        f"[w8a8-route] route={route} M={m} K={k} N={n} "
        f"m_max={m_max} relation={relation} "
        f"route_calls={_ROUTE_COUNTS[route]}",
        flush=True,
    )


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
    w8a16_m_max, w8a16_m_max_configured = _w8a16_m_max()
    route_debug = os.environ.get("B70_W8A16_ROUTE_DEBUG") == "1"
    if route_debug:
        _ROUTE_COUNTS.update(w8a16=0, w8a8=0)
        _ROUTE_LOGGED.clear()
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

    # The TP-sharded BF16 lm_head is the largest post-push decode kernel: each
    # rank streams [124160,5120] for every target and draft logits projection.
    # This opt-in path replaces the target Parameter's BF16 storage with its
    # per-channel RTN INT8 storage, then aliases the already-loaded draft head
    # to that same storage before KV-pool sizing. The normal MTP sharing call
    # remains authoritative and is assertion-wrapped below. All row counts use
    # W8A16 so target and draft never diverge through activation quantization.
    # It is output-sensitive and remains default-off until serving and quality
    # gates pass.
    if os.environ.get("B70_W8A8_QUANT_LMHEAD") == "1":
        if not fused:
            raise RuntimeError("B70_W8A8_QUANT_LMHEAD=1 requires the fused W8A8 path")
        if os.environ.get("B70_XPU_REPLICATE_MTP_EMBED") != "1":
            raise RuntimeError(
                "B70_W8A8_QUANT_LMHEAD=1 requires replicated MTP embedding"
            )

        _expected_shape = (124160, 5120)
        _scale_name = "_b70_lmhead_weight_scale"
        _target_by_rank = {}

        def _reject_lmhead_reload(*_args, **_kwargs):
            raise RuntimeError(
                "runtime lm_head weight reload is unsupported with "
                "B70_W8A8_QUANT_LMHEAD=1; restart the server"
            )

        def _quant_lmhead_int8(weight):
            rows, cols = weight.shape
            chunk_rows = int(os.environ.get("B70_W8A8_LMHEAD_CHUNK_ROWS", "8192"))
            if chunk_rows <= 0:
                raise ValueError("B70_W8A8_LMHEAD_CHUNK_ROWS must be positive")
            quant = torch.empty((rows, cols), dtype=torch.int8, device=weight.device)
            scale = torch.empty(rows, dtype=torch.float16, device=weight.device)
            for row0 in range(0, rows, chunk_rows):
                row1 = min(row0 + chunk_rows, rows)
                values = weight[row0:row1].to(torch.float32)
                scales = values.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8) / 127.0
                quant[row0:row1] = torch.round(values / scales).clamp_(-127, 127).to(torch.int8)
                scale[row0:row1] = scales.reshape(-1).to(torch.float16)
            return quant, scale

        def _register_scale(lm_head, scale):
            if _scale_name in lm_head._buffers:
                del lm_head._buffers[_scale_name]
            lm_head.register_buffer(_scale_name, scale, persistent=True)

        def _validate_scope(model, lm_head, role, rank):
            if role not in ("target", "draft") or rank not in (0, 1):
                raise RuntimeError(f"unsupported lm_head role/rank: {role}/{rank}")
            expected_model = {
                "target": (
                    "sglang.srt.models.qwen3_5",
                    "Qwen3_5ForConditionalGeneration",
                ),
                "draft": (
                    "sglang.srt.models.qwen3_5_mtp",
                    "Qwen3_5ForCausalLMMTP",
                ),
            }[role]
            actual_model = (type(model).__module__, type(model).__name__)
            if actual_model != expected_model:
                raise RuntimeError(
                    "lm_head INT8 is scoped only to Qwen3.5 target/MTP, got "
                    f"{actual_model[0]}.{actual_model[1]} for {role}"
                )
            config = getattr(model, "config", None)
            if config is None:
                raise RuntimeError("lm_head INT8 requires a model config")
            if getattr(config, "tie_word_embeddings", None) is not False:
                raise RuntimeError("lm_head INT8 requires tie_word_embeddings=False")
            if type(lm_head).__name__ != "ParallelLMHead":
                raise RuntimeError(
                    f"lm_head INT8 requires ParallelLMHead, got {type(lm_head).__name__}"
                )
            if int(getattr(lm_head, "tp_size", -1)) != 2:
                raise RuntimeError("lm_head INT8 requires a TP=2-sharded head")
            if getattr(lm_head, "bias", None) is not None:
                raise RuntimeError("lm_head INT8 does not support lm_head bias")
            if type(getattr(lm_head, "quant_method", None)).__name__ != (
                "UnquantizedEmbeddingMethod"
            ):
                raise RuntimeError("lm_head INT8 requires an unquantized checkpoint head")
            if hasattr(lm_head, "set_lora") or hasattr(lm_head, "apply_lora"):
                raise RuntimeError("lm_head INT8 does not support a LoRA-wrapped head")

        def _attach_int8_lmhead(model, role, rank):
            lm_head = getattr(model, "lm_head", None)
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise RuntimeError("model has no lm_head.weight")
            if getattr(lm_head, "_b70_int8", None) is not None:
                raise RuntimeError(f"lm_head INT8 attached twice for {role} rank={rank}")
            _validate_scope(model, lm_head, role, rank)
            weight_param = lm_head.weight
            weight = weight_param.data
            if (
                weight is None
                or tuple(weight.shape) != _expected_shape
                or weight.dtype not in (
                torch.bfloat16,
                torch.float16,
                )
            ):
                raise RuntimeError(
                    f"unsupported lm_head weight shape/dtype: "
                    f"{getattr(weight, 'shape', None)} {getattr(weight, 'dtype', None)}"
                )
            original_n, original_k = tuple(weight.shape)

            if role == "target":
                if rank in _target_by_rank:
                    raise RuntimeError(f"duplicate target lm_head for rank={rank}")
                quant_nk, weight_scale = _quant_lmhead_int8(weight)
                torch.xpu.synchronize()
                weight_param.data = quant_nk
                if (
                    weight_param.dtype != torch.int8
                    or weight_param.data_ptr() != quant_nk.data_ptr()
                ):
                    raise RuntimeError(
                        f"target lm_head INT8 storage replacement failed for rank={rank}"
                    )
                _register_scale(lm_head, weight_scale)
                weight_nt = weight_param.data.t()
                if weight_nt.stride(0) != 1:
                    raise RuntimeError(
                        f"lm_head INT8 NT stride mismatch: {weight_nt.stride()}"
                    )
                bundle = {
                    "quant_nk": weight_param.data,
                    "weight_nt": weight_nt,
                    "weight_scale": getattr(lm_head, _scale_name),
                    "rank": rank,
                }
                _target_by_rank[rank] = {
                    "module": lm_head,
                    "weight": weight_param,
                    "bundle": bundle,
                }
                storage_mode = "replaced"
            else:
                target = _target_by_rank.get(rank)
                if target is None:
                    raise RuntimeError(
                        f"draft lm_head loaded before target bundle for rank={rank}"
                    )
                bundle = target["bundle"]
                torch.xpu.synchronize()
                # Preserve the draft Parameter object until SGLang's official
                # MTP sharing call, but release its BF16 storage now.
                weight_param.data = target["weight"].data
                if weight_param.data_ptr() != target["weight"].data_ptr():
                    raise RuntimeError(
                        f"draft lm_head INT8 storage alias failed for rank={rank}"
                    )
                _register_scale(lm_head, bundle["weight_scale"])
                storage_mode = "aliased"

            del weight
            lm_head._b70_int8 = bundle
            lm_head._b70_lmhead_role = role
            lm_head._b70_lmhead_rank = rank
            lm_head._b70_lmhead_required = True
            # Initial checkpoint loading is complete. A later loader using the
            # old BF16 contract would silently cast into INT8, so reject it.
            weight_param.weight_loader = _reject_lmhead_reload
            torch.xpu.empty_cache()
            print(
                f"[lmhead-int8] ready role={role} rank={rank} "
                f"N={original_n} K={original_k} "
                f"storage={storage_mode} w8a16_only=1 "
                f"int8_gib={(bundle['quant_nk'].numel() + 2 * bundle['weight_scale'].numel()) / 2**30:.3f} "
                "bf16_released=1",
                flush=True,
            )

        import sglang.srt.model_executor.model_runner as _model_runner
        import sglang.srt.models.qwen3_5_mtp as _qwen35_mtp

        _original_load_model = _model_runner.ModelRunner.load_model
        _original_set_embed_and_head = (
            _qwen35_mtp.Qwen3_5ForCausalLMMTP.set_embed_and_head
        )

        def _load_model_with_int8_lmhead(self):
            _original_load_model(self)
            server_args = self.server_args
            if int(self.tp_size) != 2 or int(self.pp_size) != 1:
                raise RuntimeError(
                    "lm_head INT8 requires TP=2 PP=1, got "
                    f"TP={self.tp_size} PP={self.pp_size}"
                )
            if getattr(server_args, "speculative_token_map", None) is not None:
                raise RuntimeError("lm_head INT8 does not support speculative token maps")
            if getattr(server_args, "enable_lora", False):
                raise RuntimeError("lm_head INT8 does not support LoRA")
            role = "draft" if self.is_draft_worker else "target"
            _attach_int8_lmhead(self.model, role, int(self.tp_rank))

        def _set_embed_and_head_int8(self, embed, head):
            rank = int(getattr(self.lm_head, "_b70_lmhead_rank", -1))
            target = _target_by_rank.get(rank)
            if target is None or head is not target["weight"]:
                raise RuntimeError(
                    f"MTP lm_head share received an unknown target head for rank={rank}"
                )
            _original_set_embed_and_head(self, embed, head)
            target_bundle = target["bundle"]
            if self.lm_head.weight is not target["weight"]:
                raise RuntimeError(f"MTP lm_head Parameter was not shared for rank={rank}")
            if self.lm_head.weight.data_ptr() != target_bundle["quant_nk"].data_ptr():
                raise RuntimeError(f"MTP lm_head storage was not shared for rank={rank}")
            _register_scale(self.lm_head, target_bundle["weight_scale"])
            self.lm_head._b70_int8 = target_bundle
            self.lm_head._b70_lmhead_role = "draft"
            self.lm_head._b70_lmhead_rank = rank
            self.lm_head._b70_lmhead_required = True
            self.lm_head.weight.weight_loader = _reject_lmhead_reload
            same_scale = (
                getattr(self.lm_head, _scale_name).data_ptr()
                == target_bundle["weight_scale"].data_ptr()
            )
            if not same_scale:
                raise RuntimeError(f"MTP lm_head scale was not shared for rank={rank}")
            torch.xpu.empty_cache()
            print(
                f"[lmhead-int8] SHARED role=draft rank={rank} "
                "same_weight=1 same_scale=1 w8a16_only=1",
                flush=True,
            )

        _model_runner.ModelRunner.load_model = _load_model_with_int8_lmhead
        _qwen35_mtp.Qwen3_5ForCausalLMMTP.set_embed_and_head = (
            _set_embed_and_head_int8
        )

        # These APIs would apply BF16 loader semantics to an INT8 Parameter.
        # Reject them explicitly instead of allowing silent state corruption.
        for _update_name in (
            "update_weights_from_disk",
            "update_weights_from_distributed",
            "update_weights_from_tensor",
            "update_weights_from_ipc",
        ):
            if hasattr(_model_runner.ModelRunner, _update_name):
                setattr(
                    _model_runner.ModelRunner,
                    _update_name,
                    _reject_lmhead_reload,
                )

        from sglang.srt.layers.logits_processor import LogitsProcessor as _LogitsProcessor

        _original_compute_lm_head = _LogitsProcessor._compute_lm_head
        _lmhead_routes = {}

        def _compute_lm_head_int8(self, hidden_states, lm_head, embedding_bias=None):
            quant = getattr(lm_head, "_b70_int8", None)
            if quant is None:
                if getattr(lm_head, "_b70_lmhead_required", False):
                    raise RuntimeError("required lm_head INT8 bundle is missing")
                return _original_compute_lm_head(
                    self, hidden_states, lm_head, embedding_bias
                )
            if embedding_bias is not None:
                raise RuntimeError("lm_head INT8 does not support embedding bias")
            if self.use_fp32_lm_head:
                raise RuntimeError("lm_head INT8 does not support fp32 lm_head")
            role = getattr(lm_head, "_b70_lmhead_role", None)
            rank = getattr(lm_head, "_b70_lmhead_rank", None)
            if role not in ("target", "draft") or rank not in (0, 1):
                raise RuntimeError(f"invalid lm_head INT8 route identity: {role}/{rank}")
            if (
                tuple(lm_head.weight.shape) != _expected_shape
                or lm_head.weight.dtype != torch.int8
                or lm_head.weight.data_ptr() != quant["quant_nk"].data_ptr()
                or tuple(quant["weight_scale"].shape) != (_expected_shape[0],)
            ):
                raise RuntimeError(f"invalid lm_head INT8 storage for {role} rank={rank}")
            original_shape = hidden_states.shape
            x = hidden_states.reshape(-1, original_shape[-1]).to(torch.float16).contiguous()
            output = torch.ops._xpu_C.int8_gemm_w8a16(
                x, quant["weight_nt"], quant["weight_scale"], None
            )
            route_key = (rank, role)
            calls = _lmhead_routes.get(route_key, 0) + 1
            _lmhead_routes[route_key] = calls
            if calls in (1, 100, 1000) or (
                calls % 5000 == 0
            ):
                print(
                    f"[lmhead-int8] ROUTES role={role} rank={rank} "
                    f"calls={calls} latest_rows={x.shape[0]} w8a16_only=1",
                    flush=True,
                )
            return output.to(hidden_states.dtype).reshape(*original_shape[:-1], -1)

        _LogitsProcessor._compute_lm_head = _compute_lm_head_int8
        print(
            "[lmhead-int8] ENABLED: shared per-channel RTN INT8, W8A16-only logits",
            flush=True,
        )

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
        if _use_w8a16(M, w8a16_m_max):
            if route_debug:
                _record_w8a8_route(
                    "w8a16",
                    M,
                    layer.B_nt.shape[0],
                    layer.B_nt.shape[1],
                    w8a16_m_max,
                )
            out = torch.ops._xpu_C.int8_gemm_w8a16(xf, layer.B_nt, layer.wscale_n, b)  # decode
        else:
            if route_debug:
                _record_w8a8_route(
                    "w8a8",
                    M,
                    layer.B_nt.shape[0],
                    layer.B_nt.shape[1],
                    w8a16_m_max,
                )
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
        threshold_source = "env" if w8a16_m_max_configured else "default"
        print(
            "[w8a8-shim] installed: FUSED hybrid "
            f"(M<={w8a16_m_max}=int8_gemm_w8a16, "
            f"M>{w8a16_m_max}=int8_gemm_w8a8, source={threshold_source})",
            flush=True,
        )
    else:
        CompressedTensorsW8A8Int8.process_weights_after_loading = _pw_legacy
        CompressedTensorsW8A8Int8.apply_weights = _apply_legacy
        print("[w8a8-shim] installed: LEGACY torch._int_mm chain (XPU INT8 XMX)", flush=True)
