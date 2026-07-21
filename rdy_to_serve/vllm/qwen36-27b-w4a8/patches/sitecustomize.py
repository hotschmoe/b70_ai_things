# W4A8 27B shelf shim (vLLM 0.25.1, single-card TP=1). Derived 2026-07-21 from
# rdy_to_serve/vllm/qwen36-27b-w8a8/patches/sitecustomize.py; block NUMBERS are kept from
# that file so cross-references (JOURNAL, docs/20260707_dd_mtp_piecewise_neo_abort.md) stay
# valid. Blocks kept: (1) BF16 MTP drafter, (4) XPU mamba ptr fix, (6) drafter-eager
# fallback, (7) XPUGraph reclaim, plus W4A8-only block (8). Blocks DROPPED vs w8a8:
#   (2) capture-safe all_gather -- TP=1: no TP collectives exist here.
#   (3)/(5) recapture/sync experiments -- superseded by (7) (docs 2026-07-08).
import os, sys

# ---- (1) BF16 MTP drafter -------------------------------------------------------------------------
# Force ONLY the Qwen3.5 MTP drafter to instantiate unquantized (the checkpoint's 15 mtp.* tensors are
# BF16 and quantization_config.ignore has re:.*mtp.*, but without this the drafter MODULE is still
# built through the compressed-tensors quant path -> 0% accept, the same failure the W8A8 shelf hit).
try:
    import vllm.model_executor.models.qwen3_5_mtp as mtp_mod
    _orig_init = mtp_mod.Qwen3_5MultiTokenPredictor.__init__
    def _patched_init(self, *, vllm_config, prefix=""):
        old_q = getattr(vllm_config, "quant_config", None)
        try:
            vllm_config.quant_config = None
            return _orig_init(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_config.quant_config = old_q
    mtp_mod.Qwen3_5MultiTokenPredictor.__init__ = _patched_init
    print("[w4a8-shim] (1) Qwen3_5MultiTokenPredictor forced unquantized for grafted BF16 mtp.*", file=sys.stderr, flush=True)
except Exception as e:
    print("[w4a8-shim] (1) MTP patch failed:", repr(e), file=sys.stderr, flush=True)

# ---- (4) XPU mamba align-mode pointer fix (enables --enable-prefix-caching) ------------------------
# --enable-prefix-caching on the hybrid GDN model auto-switches vLLM's mamba KV cache to "align" mode.
# Combined with MTP spec-decode that activates MambaSpecDecodeGPUContext, whose
# initialize_from_forward_context (vllm/v1/worker/mamba_utils.py) packs raw device pointers into SIGNED
# int64 tensors. On Intel XPU the Level-Zero USM device addresses are >= 2**63, so the python-int ->
# int64 conversion overflows ("Overflow when unpacking long long") at engine init. Storing the
# two's-complement signed value is BIT-IDENTICAL to what CUDA stores (the triton kernel reinterprets
# the bits back to a pointer). Same fix as the W8A8 shelf (validated 2026-07-03 on v0.24.0 and running
# on the 0.25.1 DD). Default-on no-op when PREFIXCACHE=0; toggle off with VLLM_XPU_MAMBA_PTR_FIX=0.
if os.environ.get("VLLM_XPU_MAMBA_PTR_FIX", "1") != "0":
    try:
        import inspect, textwrap
        import vllm.v1.worker.mamba_utils as _mu

        def _wrap_i64(p):
            # Reinterpret an unsigned device address as a two's-complement signed int64 so it fits
            # torch's int64 tensor assignment. No-op for CUDA-range (< 2**63) pointers.
            return p - (1 << 64) if p >= (1 << 63) else p

        _cls = _mu.MambaSpecDecodeGPUContext
        _src = textwrap.dedent(inspect.getsource(_cls.initialize_from_forward_context))
        assert "state.data_ptr()" in _src and "bt.data_ptr()" in _src, "mamba_utils source changed"
        _patched_src = (_src
                        .replace("state.data_ptr()", "_wrap_i64(state.data_ptr())")
                        .replace("bt.data_ptr()", "_wrap_i64(bt.data_ptr())"))
        _ns = dict(_mu.__dict__)
        _ns["_wrap_i64"] = _wrap_i64
        exec(_patched_src, _ns)
        _cls.initialize_from_forward_context = _ns["initialize_from_forward_context"]
        print("[w4a8-shim] (4) MambaSpecDecodeGPUContext pointer packing wrapped for XPU USM (>=2**63) "
              "-- prefix caching unblocked", file=sys.stderr, flush=True)
    except Exception as e:
        print("[w4a8-shim] (4) mamba ptr fix failed:", repr(e), file=sys.stderr, flush=True)

# ---- (6) DRAFTER-EAGER (default OFF): fallback leak fix if (7) ever misbehaves ---------------------
# Forces the MTP DRAFTER to CUDAGraphMode.NONE (eager, no graph replay -> no NEO command-list
# accumulation) while the TARGET decode stays PIECEWISE-captured. Validated 2026-07-07 on NVFP4 TP=2;
# superseded as the default by block (7) but kept as the fallback (B70_XPU_DRAFTER_EAGER=1 +
# CGRECLAIM=0). OFF unless B70_XPU_DRAFTER_EAGER=1 (default byte-identical).
if os.environ.get("B70_XPU_DRAFTER_EAGER", "0") == "1":
    try:
        import vllm.v1.spec_decode.llm_base_proposer as _lbp
        _CGM = _lbp.CUDAGraphMode
        _orig_init_keys = _lbp.SpecDecodeBaseProposer.initialize_cudagraph_keys
        def _drafter_eager_keys(self, cudagraph_mode):
            return _orig_init_keys(self, _CGM.NONE)   # force drafter -> eager, target unchanged
        _lbp.SpecDecodeBaseProposer.initialize_cudagraph_keys = _drafter_eager_keys
        print("[w4a8-shim] (6) DRAFTER-EAGER ENABLED: MTP drafter forced to CUDAGraphMode.NONE; "
              "target decode stays captured", file=sys.stderr, flush=True)
    except Exception as e:
        print("[w4a8-shim] (6) drafter-eager setup failed:", repr(e), file=sys.stderr, flush=True)

# ---- (7) graph-replay command-list RECLAIM -- re-instantiate the exec graph every N replays ---------
# The NEO linear_stream.h:84 replay-accumulation abort fix (root-caused 2026-07-08, transport-agnostic:
# it hit single-collective NVFP4 AND the collective-free replay path, so a single-card captured+MTP
# serve is NOT exempt): replaying a captured XPUGraph accumulates L0 immediate-command-list space per
# replay; a queue drain does NOT reclaim it, only graph RE-INSTANTIATION does (keep_graph=True +
# g.instantiate(), no re-trace, zero throughput cost). Gated B70_XPU_CG_RECLAIM=N (0/unset = OFF ->
# byte-identical); serve.sh defaults N=1000 for GRAPH=1.
_RECLAIM_N = int(os.environ.get("B70_XPU_CG_RECLAIM", "0"))
if _RECLAIM_N > 0:
    try:
        import torch as _t
        import torch.xpu.graphs as _tgraphs
        _Base = _t.xpu.XPUGraph
        _orig_replay = _Base.replay
        _rc_counts = {}
        # keep_graph is consumed in __init__ (NOT just __new__) -- vLLM constructs via torch.cuda.CUDAGraph()
        # (rebound to torch.xpu.XPUGraph) with NO args, so a __new__-only override leaves keep_graph=False and
        # instantiate() refuses. Force it in BOTH __new__ and __init__ via a subclass (validated on box).
        class _XPUGraphReclaim(_Base):
            def __new__(cls, *a, **k):
                return _Base.__new__(cls, keep_graph=True)
            def __init__(self, *a, **k):
                super().__init__(keep_graph=True)
            def replay(self):
                k = id(self)
                n = _rc_counts.get(k, 0) + 1
                _rc_counts[k] = n
                if n % _RECLAIM_N == 0:
                    try:
                        self.instantiate()  # re-finalize exec graph -> reset the accumulated command list
                    except Exception as _e:
                        print("[w4a8-shim] (7) instantiate() failed:", repr(_e), file=sys.stderr, flush=True)
                return _orig_replay(self)
        _t.xpu.XPUGraph = _XPUGraphReclaim      # the rebind (xpu_model_runner:53) picks this up
        _tgraphs.XPUGraph = _XPUGraphReclaim
        print(f"[w4a8-shim] (7) XPUGraph RECLAIM ON (subclass): keep_graph=True + re-instantiate every "
              f"{_RECLAIM_N} replays/graph (full-speed captured+MTP leak fix)", file=sys.stderr, flush=True)
    except Exception as e:
        print("[w4a8-shim] (7) reclaim patch failed:", repr(e), file=sys.stderr, flush=True)

# ---- (8) W4A8-only note: int4 op fakes for PIECEWISE capture ---------------------------------------
# No code needed here. Upstream 0.25.1 vllm/_xpu_ops.py registers fakes for BOTH int4_gemm_w4a8 and
# int4_gemm_w4a16 when the loaded _xpu_C .so exposes them (the mounted w8a8_kernel_v0240 .so does;
# verified in-image 2026-07-21 -- the image's baked scaled_mm/xpu_int8.py w4a8 fake then logs
# "skipped: already registered", which is fine). The mounted mixed_precision/xpu.py additionally
# carries a defensive lazy _register_w4a16_fake for the B70_W4A8_HYBRID route. This block exists so
# the next reader does not go hunting for a missing registration.
