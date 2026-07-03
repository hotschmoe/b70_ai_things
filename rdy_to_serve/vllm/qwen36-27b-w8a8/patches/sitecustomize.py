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
