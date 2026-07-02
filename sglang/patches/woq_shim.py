# woq_shim.py -- wire auto_round_kernel.woqgemm (the proven-fast XPU int4 GEMM, vLLM's 30 t/s path)
# into sglang's GPTQ/AutoRound path on XPU. sglang's GPTQLinearScheme delegates to self.kernel via
# _init_kernel(); we patch _init_kernel to return an XPU WOQ kernel (mirrors the NPU backend pattern).
# The AutoRound int4 ckpt (auto_round:auto_gptq) -> AutoRoundConfig -> GPTQLinearMethod -> GPTQLinearScheme
# -> (patched) WOQ kernel -> woqgemm. GDN/vision/lm_head stay bf16 (AutoRoundConfig.get_layer_config).
# Auto-imported at interpreter startup via a .pth file; safe no-op off-XPU or if deps are missing.
import os


def _load_int4_gemm_op():
    """Make torch.ops._xpu_C.int4_gemm_w4a16 / int4_gemm_w4a8 callable (the oneDNN int4w GEMMs).
    Prefer the packaged extension; else ctypes-dlopen the built _xpu_C*.so (B70_XPU_C_SO) with
    RTLD_GLOBAL so its sibling oneAPI libs resolve. Needs the oneAPI compiler lib on
    LD_LIBRARY_PATH (see sglang/W4A8_BUILD.md). Returns True iff both ops are registered."""
    import torch

    if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16") and hasattr(
        torch.ops._xpu_C, "int4_gemm_w4a8"
    ):
        return True
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C on import)
    except Exception:
        so = os.environ.get("B70_XPU_C_SO")
        if so and os.path.exists(so):
            import ctypes

            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                print(f"[w4a8-woq] dlopen'd {so}", flush=True)
            except Exception as e:
                print(f"[w4a8-woq] ctypes.CDLL({so}) failed: {e}", flush=True)
        elif so:
            print(f"[w4a8-woq] B70_XPU_C_SO={so} does not exist", flush=True)
        else:
            print("[w4a8-woq] B70_XPU_C_SO unset and vllm_xpu_kernels import failed", flush=True)
    return hasattr(torch.ops._xpu_C, "int4_gemm_w4a16") and hasattr(
        torch.ops._xpu_C, "int4_gemm_w4a8"
    )


def _install():
    try:
        from sglang.srt.utils import is_xpu
        if not is_xpu():
            return
    except Exception:
        return
    if os.environ.get("WOQ_SHIM_DISABLE") == "1":
        return
    try:
        import torch
        import auto_round_kernel  # noqa: F401  (ensures the XPU .so loads)
        from auto_round_kernel.qlinear import QuantLinearGPTQ
        from sglang.srt.layers.quantization.gptq.schemes.gptq_linear import (
            GPTQLinearScheme,
        )
    except Exception as e:
        print(f"[woq-shim] not installing (import failed): {e}", flush=True)
        return

    class _XpuWoqGptqKernel:
        """Replaces the CUDA gptq_gemm kernel with auto_round_kernel.woqgemm on XPU."""

        def __init__(self, quant_config):
            self.qc = quant_config

        def process_weights_after_loading(self, layer):
            qw = layer.qweight.data          # [in//8, out] int32 (gptq pack)
            qz = layer.qzeros.data           # [in//g, out//8] int32
            sc = layer.scales.data           # [in//g, out]
            dev = qw.device
            in_f = qw.shape[0] * 8
            out_f = qw.shape[1]
            bits = int(getattr(self.qc, "weight_bits", getattr(self.qc, "bits", 4)))
            gs = int(getattr(self.qc, "group_size", 128))
            # AutoRound int is symmetric (sym=True) for our ckpts; desc_act None -> trivial g_idx.
            sym = bool(getattr(self.qc, "sym", True))
            ql = QuantLinearGPTQ(bits, gs, sym, in_f, out_f, False,
                                 weight_dtype=torch.bfloat16)
            ql.qweight.data = qw
            ql.qzeros.data = qz
            ql.scales.data = sc.to(torch.float16)
            ql = ql.to(dev)
            ql.post_init()
            layer._woq = ql
            # free the now-redundant gptq buffers (the ARK blob holds the packed weight)
            empty = torch.nn.Parameter(torch.empty(0, device=dev), requires_grad=False)
            for n in ("qweight", "qzeros", "scales", "g_idx"):
                if hasattr(layer, n):
                    setattr(layer, n, empty)
            torch.xpu.empty_cache() if hasattr(torch, "xpu") else None
            print(f"[woq-shim] WOQ layer ready in={in_f} out={out_f} bits={bits} g={gs}", flush=True)

        def apply(self, layer, x, bias=None):
            out = layer._woq(x)
            if bias is not None:
                out = out + bias
            return out

    # --- W4A8/W4A16 HYBRID kernel for auto_round (auto_gptq-packed) int4 linears (OPT-IN) ---
    # Same int4 weights as the woqgemm path, but dispatched to the oneDNN int4_gemm ops:
    #   decode  M==1 -> int4_gemm_w4a16 (fp16 act)  -- numerically == woqgemm (relerr 1e-3)
    #   prefill M>1  -> int4_gemm_w4a8  (per-token sym int8 act, Triton-fused quant)  -- ~1.9x faster prefill
    # Conversion (auto_gptq qweight [K/8,N] -> op B [K/8,N] NT, B_zp=[8] sym, B_scale=scales) is
    # numerically GATED by sglang/w4a8_from_woq_probe.py (relerr<1e-2 vs woqgemm on MLP + GDN layers).
    # Act-quant: the prefill per-token int8 quant uses a SINGLE-LAUNCH Triton kernel
    # (w4a8_actquant_triton, 8.3x faster than the eager ~8-launch chain @M=2048 -> lower TTFT). torch.compile
    # of the quant HANGS serve startup (inductor async-worker deadlock); Triton compiles in-process, no hang.
    # Gated by B70_W4A8_TRITON_AQ (default on); falls back to the eager chain if Triton import/JIT fails or =0.
    class _XpuW4A8WoqKernel:
        def __init__(self, quant_config):
            self.qc = quant_config
            gs = int(getattr(quant_config, "group_size", 128))
            self.gs = gs  # -1 (per-tensor) handled per-layer in process_weights_after_loading

        def process_weights_after_loading(self, layer):
            qw = layer.qweight.data          # [K/8, N] int32 (auto_gptq pack, contiguous stride0==N)
            sc = layer.scales.data           # [K/g, N]
            dev = qw.device
            in_f = qw.shape[0] * 8
            out_f = qw.shape[1]
            gs = self.gs if self.gs != -1 else in_f
            # auto_gptq qweight is [K/8,N] contiguous (stride[0]==N); the op needs [K/8,N] with
            # stride[0]==1 (NT). PURE relayout (no value change): keep the [N,K/8] contiguous
            # backing buffer alive and view its transpose -> [K/8,N] stride[0]==1.
            B_contig = qw.t().contiguous()           # [N, K/8] contiguous backing storage
            layer.qweight_t = B_contig.t()           # [K/8, N] VIEW, stride[0]==1
            layer._w4a8_B_contig = B_contig          # keep storage alive (qweight_t is a view of it)
            layer.wscale_t = sc.to(torch.float16).contiguous()      # [K/g, N] fp16
            layer.wzp = torch.tensor([8], dtype=torch.int8, device=dev)  # 1-D -> symmetric int4 zp
            layer._w4a8_gs = gs
            assert layer.qweight_t.stride()[0] == 1, (
                f"[w4a8-woq] qweight_t NOT NT (stride0={layer.qweight_t.stride()[0]})"
            )
            # free the now-redundant gptq buffers (B_contig holds the packed weight)
            empty = torch.nn.Parameter(torch.empty(0, device=dev), requires_grad=False)
            for n in ("qweight", "qzeros", "scales", "g_idx"):
                if hasattr(layer, n):
                    setattr(layer, n, empty)
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
            print(
                f"[w4a8-woq] layer ready in={in_f} out={out_f} g={gs} "
                f"qw_stride={tuple(layer.qweight_t.stride())}",
                flush=True,
            )

        def apply(self, layer, x, bias=None):
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])                 # [M, K]
            M = x2.shape[0]
            gs = layer._w4a8_gs
            b = bias.to(torch.float16) if bias is not None else None
            xf = x2.to(torch.float16).contiguous()       # ops are fp16-only (emit fp16)
            if M == 1:
                out = torch.ops._xpu_C.int4_gemm_w4a16(
                    xf, layer.qweight_t, b, layer.wscale_t, layer.wzp, gs, None
                )                                        # decode: fp16 act, no act-quant
            else:
                if _w4a8_aq is not None:
                    xq, xs, xz = _w4a8_aq(xf)            # Triton single-launch per-token sym int8
                else:
                    amax = xf.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
                    xs = (amax / 127.0).to(torch.float16)
                    xq = (xf / xs).round().clamp_(-127, 127).to(torch.int8).contiguous()
                    xz = torch.zeros_like(amax, dtype=torch.int32).contiguous()
                out = torch.ops._xpu_C.int4_gemm_w4a8(
                    xq, xs.contiguous(), xz, layer.qweight_t, layer.wscale_t,
                    layer.wzp, gs, None, b,
                )                                        # prefill: per-token sym int8 act
            return out.to(x.dtype).reshape(*orig[:-1], -1)

    _w4a8_woq = os.environ.get("B70_XPU_W4A8_WOQ") == "1"
    if _w4a8_woq:
        if _load_int4_gemm_op():
            print("[w4a8-woq] int4_gemm ops loaded; routing GPTQ int4 linears -> W4A8/W4A16 hybrid", flush=True)
        else:
            print("[w4a8-woq] int4_gemm op NOT available -> FALLING BACK to woqgemm", flush=True)
            _w4a8_woq = False

    # Prefill act-quant kernel: prefer the single-launch Triton path (lower TTFT); eager fallback.
    _w4a8_aq = None
    if _w4a8_woq and os.environ.get("B70_W4A8_TRITON_AQ", "1") != "0":
        try:
            import w4a8_actquant_triton as _aqt
            if _aqt.available():
                _w4a8_aq = _aqt.per_token_int8
                print("[w4a8-woq] prefill act-quant: TRITON single-launch per-token int8", flush=True)
            else:
                print(f"[w4a8-woq] triton act-quant unavailable ({getattr(_aqt, '_TRITON_ERR', '?')}) -> EAGER", flush=True)
        except Exception as _e:
            print(f"[w4a8-woq] triton act-quant import failed ({_e}) -> EAGER", flush=True)
    if _w4a8_woq and _w4a8_aq is None:
        print("[w4a8-woq] prefill act-quant: EAGER (~8-launch chain)", flush=True)

    def _patched_init_kernel(self, quant_config):
        if _w4a8_woq:
            return _XpuW4A8WoqKernel(quant_config)
        return _XpuWoqGptqKernel(quant_config)

    GPTQLinearScheme._init_kernel = _patched_init_kernel

    # --- int4 lm_head (OPT-IN via B70_W4A8_QUANT_LMHEAD=1) -------------------------------------
    # The Lorbus int4 ckpt EXCLUDES lm_head from quant -> it stays BF16 [vocab, hidden] (2.54 GB),
    # read in FULL every decode step (~10% of per-token decode weight bandwidth; the lm_head GEMV is
    # ~4.3 ms bf16 vs the whole step ~40 ms @25 t/s). RTN-quantize it to int4 group-g sym ONCE at
    # load time and route the logits GEMV through int4_gemm_w4a16 (the same captured decode op as the
    # body) -> ~3.3-3.8x faster lm_head -> est +~8% decode. lm_head is OUTPUT-SENSITIVE so this MUST
    # gate on accuracy (HumanEval+). Naive RTN int4 weight relerr is ~10-13% (g32 best); keep bf16 if
    # it regresses. We KEEP the bf16 weight resident (cheap, ~0.6 GB extra for int4) for revertibility
    # and so get_embed_and_head()/PP/spec paths still see lm_head.weight; only the matmul is rerouted.
    if os.environ.get("B70_W4A8_QUANT_LMHEAD") == "1":
        if not _load_int4_gemm_op():
            print("[lmhead-int4] int4_gemm op NOT available -> lm_head stays bf16", flush=True)
        else:
            _lmh_g = int(os.environ.get("B70_W4A8_LMHEAD_GROUP", "32"))

            def _quant_lmhead_w(weight, g):
                # weight [N,K] float -> (qw_contig [N,K/8] int32, scales_fp16 [N,K/g]); sym zp=8, q in [-7,7].
                # Chunked over N to bound the fp32 scratch (the full fp32 view of a 248k-row lm_head is ~5 GB).
                N, K = weight.shape
                assert K % g == 0, f"lm_head K={K} not divisible by group {g}"
                dev = weight.device
                qw = torch.empty(N, K // 8, dtype=torch.int32, device=dev)
                sc = torch.empty(N, K // g, dtype=torch.float16, device=dev)
                step = 16384
                for r0 in range(0, N, step):
                    r1 = min(r0 + step, N)
                    Wg = weight[r0:r1].reshape(r1 - r0, K // g, g).to(torch.float32)
                    amax = Wg.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
                    scale = amax / 7.0
                    q = torch.round(Wg / scale).clamp_(-7, 7).to(torch.int32)  # [-7,7]
                    sc[r0:r1] = scale.squeeze(-1).to(torch.float16)
                    nib = (q + 8).reshape(r1 - r0, K // 8, 8)                  # nibble in [1,15]
                    acc = torch.zeros(r1 - r0, K // 8, dtype=torch.int32, device=dev)
                    for i in range(8):
                        acc |= (nib[:, :, i] & 0xF) << (4 * i)
                    qw[r0:r1] = acc
                return qw, sc

            def _attach_int4_lmhead(model):
                lm = getattr(model, "lm_head", None)
                if lm is None or not hasattr(lm, "weight") or getattr(lm, "_b70_int4", None) is not None:
                    return
                w = lm.weight.data
                if w is None or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
                    print(f"[lmhead-int4] skip: weight dtype/shape unsupported ({getattr(w,'dtype',None)})", flush=True)
                    return
                N, K = w.shape
                qw, sc = _quant_lmhead_w(w, _lmh_g)          # qw [N,K/8] contig, sc [N,K/g] fp16
                qweight_t = qw.t()                            # [K/8, N] NT view (stride0==1)
                assert qweight_t.stride()[0] == 1
                lm._b70_int4 = {
                    "qw": qw,                                 # keep contiguous base alive (qweight_t is its view)
                    "qweight_t": qweight_t,
                    "wscale_t": sc.t().contiguous(),          # [K/g, N] fp16
                    "wzp": torch.tensor([8], dtype=torch.int8, device=w.device),
                    "g": _lmh_g,
                }
                if hasattr(torch, "xpu"):
                    torch.xpu.empty_cache()
                print(f"[lmhead-int4] lm_head int4 ready N={N} K={K} g={_lmh_g} "
                      f"(int4 {qw.numel()*4/1e9:.2f}GB, bf16 kept {w.numel()*2/1e9:.2f}GB)", flush=True)

            try:
                import sglang.srt.model_executor.model_runner as _mr
                _orig_load_model = _mr.ModelRunner.load_model

                def _load_model_q(self):
                    _orig_load_model(self)
                    try:
                        _attach_int4_lmhead(self.model)
                    except Exception as _e:
                        print(f"[lmhead-int4] quant FAILED (lm_head stays bf16): {_e}", flush=True)

                _mr.ModelRunner.load_model = _load_model_q

                from sglang.srt.layers.logits_processor import LogitsProcessor as _LP
                _orig_compute_lm_head = _LP._compute_lm_head

                def _compute_lm_head_int4(self, hidden_states, lm_head, embedding_bias=None):
                    q = getattr(lm_head, "_b70_int4", None)
                    if q is not None and embedding_bias is None and not self.use_fp32_lm_head:
                        xf = hidden_states.to(torch.float16).contiguous()
                        out = torch.ops._xpu_C.int4_gemm_w4a16(
                            xf, q["qweight_t"], None, q["wscale_t"], q["wzp"], q["g"], None
                        )
                        return out.to(hidden_states.dtype)
                    return _orig_compute_lm_head(self, hidden_states, lm_head, embedding_bias)

                _LP._compute_lm_head = _compute_lm_head_int4
                print(f"[lmhead-int4] ENABLED: lm_head -> int4_gemm_w4a16 (g={_lmh_g}); "
                      "load-time RTN quant + LogitsProcessor reroute", flush=True)
            except Exception as _e:
                print(f"[lmhead-int4] install FAILED (lm_head stays bf16): {_e}", flush=True)

    # AutoRound's gptq/awq dispatch calls check_marlin_supported(..., device_capability=None) on XPU,
    # which does `None * int` -> TypeError before reaching the scheme. Guard it: no marlin on XPU.
    try:
        import sglang.srt.layers.quantization.marlin_utils as mu
        _orig_cms = mu.check_marlin_supported

        def _safe_cms(*a, **k):
            dc = k.get("device_capability", a[3] if len(a) > 3 else None)
            if dc is None:
                return False
            return _orig_cms(*a, **k)

        mu.check_marlin_supported = _safe_cms
    except Exception as e:
        print(f"[woq-shim] marlin guard failed: {e}", flush=True)

    # sglang's spec-decode (EAGLE/NEXTN) path hardcodes torch.cuda.{synchronize,Stream,Event,...} which
    # assert "Torch not compiled with CUDA". On XPU, redirect them to the torch.xpu equivalents so MTP runs.
    try:
        if hasattr(torch, "xpu"):
            torch.cuda.synchronize = lambda *a, **k: torch.xpu.synchronize()
            torch.cuda.current_stream = lambda *a, **k: torch.xpu.current_stream(*a, **k)
            torch.cuda.stream = torch.xpu.stream
            torch.cuda.Stream = torch.xpu.Stream
            torch.cuda.Event = torch.xpu.Event
            torch.cuda.empty_cache = torch.xpu.empty_cache
            print("[woq-shim] torch.cuda.{synchronize,Stream,Event,...} -> torch.xpu (for spec-decode)", flush=True)
    except Exception as e:
        print(f"[woq-shim] cuda->xpu redirect failed: {e}", flush=True)

    # --- XPU CUDAGRAPH (the eager-ceiling breaker; OPT-IN via B70_XPU_CUDAGRAPH=1) ---
    # torch.xpu supports graph capture (XPUGraph/graph); the GDN + triton attn backends have graph state.
    # We flip support_cuda_graph->True and redirect torch.cuda.{CUDAGraph,graph,graph_pool_handle}->torch.xpu.
    if os.environ.get("B70_XPU_CUDAGRAPH") == "1":
        try:
            torch.cuda.CUDAGraph = torch.xpu.XPUGraph
            torch.cuda.graph_pool_handle = torch.xpu.graph_pool_handle

            # sglang's full_cuda_graph_backend does `self._device_module.graph(cuda_graph=graph, ...)` where
            # device_module is torch.xpu -> torch.xpu.graph(cuda_graph=) but its sig is graph(xpu_graph, pool, stream).
            # Patch torch.xpu.graph (and torch.cuda.graph) to an adapter that maps cuda_graph->positional + drops
            # the cuda-only kwargs. Save the original to avoid recursion.
            _orig_xpu_graph = torch.xpu.graph

            def _xpu_graph_ctx(*a, cuda_graph=None, pool=None, stream=None, **kw):
                g = a[0] if a else cuda_graph
                return _orig_xpu_graph(g, pool=pool, stream=stream)

            torch.xpu.graph = _xpu_graph_ctx
            torch.cuda.graph = _xpu_graph_ctx
            import sglang.srt.platforms as _p
            _p.current_platform.__class__.support_cuda_graph = lambda self: True
            # NOTE: do NOT set is_out_of_tree=True -- that route needs 8 [Planned] platform methods the
            # sglang CORE hasn't migrated to ("future PR"). Instead we add "xpu" to the IN-TREE device list
            # (model_runner.py, mounted patch), which uses the hardcoded torch.cuda.* (redirected to xpu here).
            print("[woq-shim] XPU CUDAGRAPH ENABLED (support_cuda_graph->True; torch.cuda.graph->xpu; in-tree path)", flush=True)
            # device-gate patch + XPUAttentionBackend decode graph hooks (the actual capture enablement).
            try:
                import xpu_cudagraph
                xpu_cudagraph.install()
            except Exception as _e:
                print(f"[woq-shim] xpu_cudagraph.install FAILED: {_e}", flush=True)
        except Exception as e:
            print(f"[woq-shim] xpu cudagraph enable FAILED: {e}", flush=True)

    # --- MTP/NEXTN tree kernels (the stable-speedup lever; OPT-IN via B70_XPU_MTP=1) ---
    # sgl_kernel's build_tree_kernel_efficient + verify_tree_greedy are CUDA-only (unregistered on XPU).
    # Install pure-torch chain (topk=1) fallbacks so NEXTN spec-decode runs. Must patch eagle_utils BEFORE
    # eagle_worker_v2 imports build_tree_kernel_efficient (the shim runs at startup -> before server init).
    if os.environ.get("B70_XPU_MTP") == "1":
        try:
            import mtp_tree_xpu
            mtp_tree_xpu.install()
        except Exception as e:
            print(f"[woq-shim] MTP tree fallback install FAILED: {e}", flush=True)

    # --- W8A8 INT8 (torch._int_mm = oneDNN INT8 XMX, ~1.8x bf16; OPT-IN via B70_XPU_W8A8=1) ---
    if os.environ.get("B70_XPU_W8A8") == "1":
        try:
            import w8a8_shim
            w8a8_shim.install()
        except Exception as e:
            print(f"[woq-shim] W8A8 shim install FAILED: {e}", flush=True)

    # --- PUSH ALL-REDUCE (hand-rolled L0-IPC push collective, decode AR ~34-45us vs oneCCL ~85-88us;
    #     OPT-IN via B70_XPU_PUSH_AR=1 + PUSH_AR_SO=/path/to/libxpu_push_ar_graph.so). P2PACCESS-independent. ---
    if os.environ.get("B70_XPU_PUSH_AR") == "1":
        try:
            import push_ar_xpu
            push_ar_xpu.install()
        except Exception as e:
            print(f"[woq-shim] push-AR install FAILED: {e}", flush=True)

    # --- W4A8/W4A16 HYBRID (oneDNN int4w x {int8a prefill, fp16a decode}; OPT-IN via B70_XPU_W4A8=1) ---
    # Wraps compressed-tensors _get_scheme_from_parts to route dense W4A8 int-quantized linears to
    # CompressedTensorsW4A8Int8XPU -> torch.ops._xpu_C.int4_gemm_w4a{8,16}. Needs the built _xpu_C.abi3.so
    # (B70_XPU_C_SO) + the oneAPI compiler lib on LD_LIBRARY_PATH. See sglang/W4A8_BUILD.md.
    if os.environ.get("B70_XPU_W4A8") == "1":
        try:
            import w4a8_shim
            w4a8_shim.install()
        except Exception as e:
            print(f"[woq-shim] W4A8 shim install FAILED: {e}", flush=True)

    print("[woq-shim] installed: GPTQLinearScheme -> auto_round_kernel.woqgemm (XPU int4)", flush=True)


_install()
