# sitecustomize.py -- MERGED shim for Qwen3.6-27B W4A16 (compressed-tensors, TEXT-ONLY) + BF16 MTP graft,
# TP=2, cudagraph_mode=NONE. Loaded via PYTHONPATH at interpreter startup; re-run in EVERY spawned vLLM
# worker (VLLM_WORKER_MULTIPROC_METHOD=spawn). Pinned to vLLM 0.23 (image :v0230).
#
# Python imports only ONE sitecustomize.py per interpreter, so the w4a16 arch-registration shim and the
# w8a8-mtp drafter-unquant/csag shim are MERGED here. Three patches, in dependency order:
#   (a) ARCH REG    register the EXACT text-only Qwen3_5ForCausalLM arch (the w4a16 fix) so vLLM loads
#                   text-only and never builds the weightless vision tower. MUST run before the model is
#                   built -> on PYTHONPATH (runs at interpreter startup). [verbatim from ../qwen36-27b-w4a16]
#   (b) MTP UNQUANT force ONLY the Qwen3_5MultiTokenPredictor drafter to instantiate unquantized/BF16, else
#                   the grafted BF16 mtp.* linears load through the W4A16 quant path -> 0% accept.
#   (c) CSAG        capture-safe all_gather -- ONLY needed when the spec-verify all_gather is RECORDED into a
#                   graph (PIECEWISE/FULL). On cudagraph_mode=NONE there is NO capture, so the base oneCCL
#                   all_gather runs eagerly and is correct -> serve.sh sets CSAG_DISABLE=1 on NONE (skip).
import os, sys

# ---- (a) w4a16 text-only arch registration (verbatim from ../qwen36-27b-w4a16/patches/sitecustomize.py) ----
try:
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM",
        "qwen35_text_hybrid:Qwen3_5ForCausalLM",
    )
    print("[qwen35-text-shim] registered Qwen3_5ForCausalLM -> text-only hybrid class", file=sys.stderr)
except Exception as e:  # never block startup if the internals shift
    print(f"[qwen35-text-shim] registration skipped: {e}", file=sys.stderr)

# ---- (b) BF16 MTP drafter, force unquantized (verbatim from ../qwen36-27b-w8a8-sqgptq-mtp/patches) ----
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

# ---- (c) capture-safe all_gather -- UNNECESSARY on cudagraph_mode=NONE (serve.sh sets CSAG_DISABLE=1) ----
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
            out = buf.movedim(0, dim).reshape(
                input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1:]
            )
            return out

        XpuCommunicator.all_gather = _all_gather_via_allreduce
        print("[csag-shim] (2) XpuCommunicator.all_gather -> capture-safe all-reduce-of-padded", file=sys.stderr, flush=True)
    except Exception as e:
        print("[csag-shim] (2) all_gather patch failed:", repr(e), file=sys.stderr, flush=True)
else:
    print("[csag-shim] (2) csag DISABLED (CSAG_DISABLE=1) -- base oneCCL all_gather (correct on cudagraph_mode=NONE)", file=sys.stderr, flush=True)
