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
        # COVERAGE (codex-reviewed 2026-06-24): this patches the all_gather path used by the dense-hybrid MTP serve
        # (vllm::all_gather -> _all_gather_out_place -> device_communicator.all_gather, incl. the seq-parallel custom
        # op). NOT patched: all_gatherv (raw dist.all_gather, used by MoE all2all) and gather() (all_gather_into_tensor).
        # If a future MoE/gather model captures those on XPU they can still hit the oneCCL allgather record crash --
        # extend this shim then. For Qwen3.6-27B dense hybrid they are not on the captured path (verified: capture OK).
        print("[csag-shim] (2) XpuCommunicator.all_gather -> capture-safe all-reduce-of-padded", file=sys.stderr, flush=True)
    except Exception as e:
        print("[csag-shim] (2) all_gather patch failed:", repr(e), file=sys.stderr, flush=True)
