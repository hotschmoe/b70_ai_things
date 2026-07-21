# Bug B plan-B shim (combined). Two patches:
#
# (1) BF16-MTP graft: force ONLY the Qwen3.5 MTP drafter to instantiate unquantized/BF16 (same as the recipe
#     patches/sitecustomize.py). Without this the drafter loads through the W8A8 quant path -> 0% accept.
#
# (2) CAPTURE-SAFE all_gather: the root cause of Bug B is that EJECTING a TP collective to eager breaks vLLM's
#     piecewise-cudagraph input-address contract (CUDAGraphWrapper does pure replay() with no input copy; the
#     ejected collective's OUT-OF-PLACE output does not land at the capture-time address -> next captured piece
#     reads stale data -> garbage). For NO-MTP we simply do not eject (all_reduce records fine inside the graph).
#     For MTP the spec-verify path calls vllm::all_gather, and oneCCL 2021.17's allgather scheduler algorithm has
#     NO SYCL-graph-recordable impl -> if left captured it CRASHES capture; if ejected it CORRUPTS (the boundary).
#     Fix: reimplement all_gather as an ALL-REDUCE of a padded buffer (dist.all_reduce DOES record under
#     CCL_ENABLE_SYCL_KERNELS=1). Then all_gather is recordable -> keep ALL collectives captured (EJECT=none) ->
#     no ejected boundary anywhere -> coherent AND fully captured (fast). Cost: world_size x bytes moved (TP=2 = 2x),
#     which MTP amortizes. Semantics are byte-identical to the base concat-style all_gather.
import os, sys

# ---- (1) BF16 MTP drafter -------------------------------------------------------------------------
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
    print("[csag-shim] (1) Qwen3_5MultiTokenPredictor forced unquantized for grafted BF16 mtp.*", file=sys.stderr, flush=True)
except Exception as e:
    print("[csag-shim] (1) MTP patch failed:", repr(e), file=sys.stderr, flush=True)

# ---- (2) capture-safe all_gather (all-reduce of a padded buffer) ----------------------------------
# Toggle off with CSAG_DISABLE=1 to A/B test.
if os.environ.get("CSAG_DISABLE", "0") != "1":
    try:
        import torch
        import torch.distributed as dist
        from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator

        def _all_gather_via_allreduce(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
            if self.world_size == 1:
                return input_
            if dim < 0:
                dim += input_.dim()
            input_ = input_.contiguous()
            input_size = tuple(input_.size())
            # buf[r] holds rank r's input; everyone else's slot is zero -> all_reduce(sum) fills all slots.
            buf = torch.zeros((self.world_size,) + input_size, dtype=input_.dtype, device=input_.device)
            buf[self.rank_in_group] = input_
            dist.all_reduce(buf, group=self.device_group)            # RECORDABLE (CCL sycl kernels)
            # concat-style along `dim`, matching base_device_communicator.all_gather exactly
            out = buf.movedim(0, dim).reshape(
                input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1:]
            )
            return out

        XpuCommunicator.all_gather = _all_gather_via_allreduce
        print("[csag-shim] (2) XpuCommunicator.all_gather -> capture-safe all-reduce-of-padded", file=sys.stderr, flush=True)
    except Exception as e:
        print("[csag-shim] (2) all_gather patch failed:", repr(e), file=sys.stderr, flush=True)

# ---- (3) Tier F (EXPERIMENTAL, default OFF): bound the PIECEWISE graph command-stream accumulation -----------
# The XPU graph REPLAY appends to a Level-Zero command list that is never reset, overflowing NEO after ~5000
# replays (the W8A8-27B MTP crash; cudagraph_mode=NONE avoids it but loses capture speed). This recaptures the
# WHOLE captured-graph set every N decode steps so the command buffer stays bounded -> keep PIECEWISE speed,
# no crash. OFF unless B70_XPU_CG_RECYCLE_STEPS>0 (so the shelf/production default is byte-identical). N counts
# calls on the first-seen PIECEWISE wrapper (~one per decode step). See docs/20260625_..._campaign.md sec 13.
_RECYCLE_N = int(os.environ.get("B70_XPU_CG_RECYCLE_STEPS", "0") or "0")
if _RECYCLE_N > 0:
    try:
        import torch
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        _orig_cgw_call = CUDAGraphWrapper.__call__
        _cg = {"n": 0, "root": None, "recaptures": 0}

        def _recycling_call(self, *args, **kwargs):
            try:
                mode = getattr(getattr(self, "runtime_mode", None), "name", None)
                if mode == "PIECEWISE":
                    if _cg["root"] is None:
                        _cg["root"] = id(self)
                    if id(self) == _cg["root"]:          # ~once per decode step
                        _cg["n"] += 1
                        if _cg["n"] >= _RECYCLE_N:
                            _cg["n"] = 0
                            if hasattr(torch, "xpu"):
                                torch.xpu.synchronize()
                            # coherent: clear ALL captured graphs (target + drafter + every piecewise layer)
                            cleared = False
                            if hasattr(CUDAGraphWrapper, "clear_all_graphs"):
                                CUDAGraphWrapper.clear_all_graphs(); cleared = True
                            elif hasattr(self, "clear_graphs"):
                                self.clear_graphs(); cleared = True
                            _cg["recaptures"] += 1
                            print(f"[cg-recycle] recapture #{_cg['recaptures']} after {_RECYCLE_N} steps "
                                  f"(clear_all={cleared})", file=sys.stderr, flush=True)
            except Exception as e:
                print("[cg-recycle] step error:", repr(e), file=sys.stderr, flush=True)
            return _orig_cgw_call(self, *args, **kwargs)

        CUDAGraphWrapper.__call__ = _recycling_call
        print(f"[cg-recycle] (3) Tier F ENABLED: recapture PIECEWISE graphs every {_RECYCLE_N} decode steps "
              f"(EXPERIMENTAL)", file=sys.stderr, flush=True)
    except Exception as e:
        print("[cg-recycle] (3) setup failed:", repr(e), file=sys.stderr, flush=True)

# ---- (5) FIX-SYNC (EXPERIMENTAL, default OFF): drain the queue every N decode steps, NO recapture ----
# Competing hypothesis to block (3): the NEO command-stream / L0 event-pool growth from graph REPLAY
# (at::xpu::XPUGraphImpl::replay submits the executable SYCL command_graph via submit_with_event onto the
# in-order queue and NEVER synchronizes -> per-replay graph-exec commands + dropped un-waited events pile up
# in the immediate command list, reclaimed only on a full queue synchronize). If a plain torch.xpu.synchronize
# every N decode steps RECLAIMS that growth, we do NOT need block (3)'s expensive clear+recapture -- we keep the
# captured graphs intact (no recapture stall) AND bound the command list. This isolates "sync reclaims" (this
# block) from "reset/recapture required" (block 3). N counts calls on the first-seen PIECEWISE wrapper
# (~one per decode step); N=1 = sync every step (the cadence normal non-spec decode already has). OFF unless
# B70_XPU_CG_SYNC_STEPS>0 so the shelf/production serve is byte-identical. Mutually exclusive with block (3).
_SYNC_N = int(os.environ.get("B70_XPU_CG_SYNC_STEPS", "0") or "0")
if _SYNC_N > 0:
    try:
        import torch
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        _orig_cgw_call_sync = CUDAGraphWrapper.__call__
        _sy = {"n": 0, "root": None, "syncs": 0}

        def _syncing_call(self, *args, **kwargs):
            out = _orig_cgw_call_sync(self, *args, **kwargs)
            try:
                mode = getattr(getattr(self, "runtime_mode", None), "name", None)
                if mode == "PIECEWISE":
                    if _sy["root"] is None:
                        _sy["root"] = id(self)
                    if id(self) == _sy["root"]:            # ~once per decode step
                        _sy["n"] += 1
                        if _sy["n"] >= _SYNC_N:
                            _sy["n"] = 0
                            if hasattr(torch, "xpu"):
                                torch.xpu.synchronize()
                            _sy["syncs"] += 1
            except Exception as e:
                print("[cg-sync] step error:", repr(e), file=sys.stderr, flush=True)
            return out

        CUDAGraphWrapper.__call__ = _syncing_call
        print(f"[cg-sync] (5) FIX-SYNC ENABLED: torch.xpu.synchronize every {_SYNC_N} decode steps "
              f"(no recapture) (EXPERIMENTAL)", file=sys.stderr, flush=True)
    except Exception as e:
        print("[cg-sync] (5) setup failed:", repr(e), file=sys.stderr, flush=True)

# ---- (6) DRAFTER-EAGER (EXPERIMENTAL, default OFF): the VALIDATED leak fix (keeps target capture) ---
# VALIDATED 2026-07-07 on NVFP4 TP=2 (survived 44k tokens vs baseline crash ~8-12k); see
# docs/20260707_dd_mtp_piecewise_neo_abort.md. Root cause: at::xpu::XPUGraphImpl::replay submits the
# captured SYCL graph via submit_with_event with no sync -> per-replay NEO command-list entries accumulate;
# the MTP drafter propose loop (SpecDecodeBaseProposer.propose, llm_base_proposer.py) fires spec x pieces
# replays/step with no host sync = the dominant leak. This forces the DRAFTER to CUDAGraphMode.NONE (eager,
# no graph replay -> no accumulation) while the TARGET decode stays PIECEWISE-captured (fast). Per-step sync
# does NOT reclaim (tested); recapture is racy (tested); drafter-eager WORKS. OFF unless
# B70_XPU_DRAFTER_EAGER=1 (default byte-identical). Cost: drafter runs eager (~loses some MTP speed) but
# keeps MTP accept + target capture -> beats full enforce-eager, and beats graph+MTP-off on high-accept code.
if os.environ.get("B70_XPU_DRAFTER_EAGER", "0") == "1":
    try:
        import vllm.v1.spec_decode.llm_base_proposer as _lbp
        _CGM = _lbp.CUDAGraphMode
        _orig_init_keys = _lbp.SpecDecodeBaseProposer.initialize_cudagraph_keys
        def _drafter_eager_keys(self, cudagraph_mode):
            return _orig_init_keys(self, _CGM.NONE)   # force drafter -> eager, target unchanged
        _lbp.SpecDecodeBaseProposer.initialize_cudagraph_keys = _drafter_eager_keys
        print("[drafter-eager] (6) ENABLED: MTP drafter forced to CUDAGraphMode.NONE (eager); "
              "target decode stays captured", file=sys.stderr, flush=True)
    except Exception as e:
        print("[drafter-eager] (6) setup failed:", repr(e), file=sys.stderr, flush=True)

# ---- (4) XPU mamba align-mode pointer fix (enables --enable-prefix-caching) ------------------------
# --enable-prefix-caching on the hybrid GDN model auto-switches vLLM's mamba KV cache to "align" mode.
# Combined with MTP spec-decode that activates MambaSpecDecodeGPUContext, whose
# initialize_from_forward_context (vllm/v1/worker/mamba_utils.py) packs raw device pointers into SIGNED
# int64 tensors:  self.state_base_addrs[idx] = state.data_ptr()  and  self.block_table_ptrs[i] = bt.data_ptr().
# On CUDA a device pointer sits below 2**63 and fits a signed int64; on Intel XPU the Level-Zero USM
# device addresses are >= 2**63, so torch's python-int -> int64 conversion (THPUtils_unpackLongLong)
# overflows -> "Overflow when unpacking long long" at engine init (the JOURNAL 2026-07-03 crash that
# forced PREFIXCACHE=0). The stored int64 is only ever reinterpreted back into a pointer inside the
# triton kernel via .to(tl.pointer_type(...)) (a bit-pattern reinterpret) with modular int64 arithmetic,
# so storing the two's-complement signed value (ptr - 2**64 when ptr >= 2**63) is BIT-IDENTICAL to what
# CUDA stores. We re-exec the method source with the two pointer assignments wrapped -- the same
# getsource -> substitute -> exec -> rebind trick as sglang/patches/mtp_tree_xpu.py._strip_is_cuda_guard.
# NOTE the sibling batch_memcpy path (collect_mamba_copy_meta ~L642) already uses uint64 buffers and does
# NOT overflow, so it is intentionally left untouched (wrapping it to a negative would corrupt uint64).
# Toggle off with VLLM_XPU_MAMBA_PTR_FIX=0 (default on; a no-op for CUDA-range < 2**63 pointers).
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
        # exec with a copy of the module globals (so get_conv_copy_spec/is_conv_state_dim_first/etc.
        # resolve) plus the injected _wrap_i64 helper, then rebind the method.
        _ns = dict(_mu.__dict__)
        _ns["_wrap_i64"] = _wrap_i64
        exec(_patched_src, _ns)
        _cls.initialize_from_forward_context = _ns["initialize_from_forward_context"]
        print("[csag-shim] (4) MambaSpecDecodeGPUContext pointer packing wrapped for XPU USM (>=2**63) "
              "-- prefix caching unblocked", file=sys.stderr, flush=True)
    except Exception as e:
        print("[csag-shim] (4) mamba ptr fix failed:", repr(e), file=sys.stderr, flush=True)

# ---- (7) graph-replay command-list RECLAIM -- re-instantiate the exec graph every N replays ---------
# Ported VERBATIM from vllm/nvfp4/patches/sitecustomize.py block (9) after the 2026-07-21 overnight DD
# crash: W8A8 TP=2 captured+MTP3 on v0.25.1 hit the NEO linear_stream.h:84 abort ~36 min in under 4-way
# concurrent load (results/logs/dd_w8a8_crash_20260721.log) -- the SAME transport-agnostic graph-replay
# accumulation root-caused 2026-07-08 (docs/20260707_dd_mtp_piecewise_neo_abort.md): replaying a captured
# XPUGraph that contains a cross-device collective accumulates L0 immediate-command-list space per replay;
# a queue drain does NOT reclaim it, only graph RE-INSTANTIATION does (keep_graph=True + g.instantiate(),
# no re-trace, zero throughput cost). Supersedes the experimental blocks (3)/(5)/(6) above as the default
# leak fix. Gated B70_XPU_CG_RECLAIM=N (0/unset = OFF -> byte-identical); serve.sh defaults N=1000 for GRAPH=1.
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
                        print("[csag-shim] (7) instantiate() failed:", repr(_e), file=sys.stderr, flush=True)
                return _orig_replay(self)
        _t.xpu.XPUGraph = _XPUGraphReclaim      # the rebind (xpu_model_runner:53) picks this up
        _tgraphs.XPUGraph = _XPUGraphReclaim
        print(f"[csag-shim] (7) XPUGraph RECLAIM ON (subclass): keep_graph=True + re-instantiate every "
              f"{_RECLAIM_N} replays/graph (full-speed captured+MTP leak fix)", file=sys.stderr, flush=True)
    except Exception as e:
        print("[csag-shim] (7) reclaim patch failed:", repr(e), file=sys.stderr, flush=True)
