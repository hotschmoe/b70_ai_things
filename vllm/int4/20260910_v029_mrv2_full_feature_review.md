# Official vLLM 0.29 XPU MRV2 full-feature source review

CONFIG -> Exact existing phase-only official image
`sha256:29f480e90dc6b88e7898392500b89a22a49e13209f7b6e3e3c7b47088748a0e1`.
Question: GPTQ INT4 + MTP3 + calibrated FP8 KV + prefix + FULL_DECODE_ONLY,
using MRV2 to avoid the separately confirmed MRV1 accepted-count permutation.
CPU/source inspection only; no new image, port, serving, or GPU execution.

COMMAND -> Export installed worker/config/attention/model Python sources from a
no-device, no-network container. Inspect actual runner selection, speculative
routing, Mamba accepted-count storage, XPU graph controls and calibration hooks.
Execute the actual isolated MRV2 unsupported-feature filter on CPU for ordinary
TP1/TP2 MTP configurations, without full EngineArgs or model construction.

RESULT -> XPU does NOT force MRV1. The earlier successful official phase TP1
screen explicitly logged `Using V2 Model Runner` at 04:07:36 in
`vllm029-tp1-gdnphase-card0/server.log`. Installed XPUWorker selects
XPUModelRunnerV2 when VllmConfig.use_v2_model_runner is true. XPUModelRunnerV2
inherits the new gpu/model_runner.py and wraps CUDA API names with XPU APIs.

The MRV2 default filter allows method mtp; dedicated MTPSpeculator delegates to
the autoregressive speculator and the shared draft model loader. GPTQ, ordinary
prefix caching, FP8 KV and FULL_DECODE_ONLY are not fallback conditions. The
isolated filter returns no unsupported features for the examined TP1 and TP2
configurations. This does not validate every full configuration combination.

VERDICT -> Official 0.29 is a credible updated MRV2 investigation target, not a
drop-in qualified full-feature replacement. MTP3 is source-supported on this
runner, but our measured official-image pass remains TP1/eager/MTP0/FP16 KV/
prefix-off only. Calibration-hook provenance and multi-GPU graph support are
concrete remaining limits.

## Accepted-count ownership differs from MRV1

`v1/worker/gpu/model_states/mamba_hybrid.py` allocates a persistent request-slot
accepted-count array. postprocess_state scatters sampled counts using the
current idx_mapping; prepare_attn gathers once using the new batch idx_mapping.
It does not apply MRV1's previous-batch prev_positions gather to an already
permuted accepted-count array. This structurally bypasses that specific double
permutation path, without proving the absence of other Mamba/MTP ordering bugs.
Reordering, prefix boundaries, mixed batches and accepted-count variation still
need matched runtime tests. An explicit VLLM_USE_V2_MODEL_RUNNER=0 would select
the old runner; do not carry that or other stale R276 tuning into a fresh arm.

## Graph and cache limits

- XPU graph support checks Torch >=2.11; this image has the newer held Torch.
  VLLM_XPU_ENABLE_XPU_GRAPH=1 is required. The previous official eager screen
  logged graphs disabled by that environment flag.
- Installed platforms/xpu.py explicitly warns that XPU graph support is
  experimental and currently supports single-GPU execution only. It does not
  establish TP2 graph safety merely because configuration accepts the flags.
  No official TP2 graph launch is proposed without a loaded-context proof and
  the project's lease/health/recovery discipline.
- MRV2 graph code handles separate decode routines, and GDN reports uniform
  batch graph support. These source paths are insufficient to qualify MTP3
  target/draft capture and replay on B70.
- XPU FlashAttention advertises FP8 KV and consumes separate K/V scales;
  MTP draft KV inherits the target dtype unless explicitly overridden.
  Full model target/draft scale coverage has not been executed on this image.
- Record resolved cache layout and capacity, not just requested block size64.
  The current R276 matched dtype arms resolve FP16 to832 and FP8 to1600; those
  are R276 observations, not official 0.29 predictions. Hybrid Mamba sizing can
  change the effective attention block size. The synthetic64-token-block
  oracle is not qualification of that resolved full-model layout.

## Existing calibration hook is deliberately incompatible as-is

The fresh be02 artifact retains source routing fingerprint
`6ee6b8db18759873246aca28e85ca6d2ba177eb08bfd3b9b0f0feea168cee9b3`.
Actual official-image _xpu_ops.py is
`47080ce161348db461ac6777f818742a067ef6db5cb64a67165864fbad21d3e1`.
The R276 kv_calibration_hook.py load path checks this fingerprint before
installing scales, so it will reject this image. That rejection is correct.

The hook's structural API remains recognizable: installed Attention has
process_weights_after_loading, scalar _q/_k/_v buffers, float/CPU mirrors and
query_quant. A deliberate source adapter could pin both routing identities and
validate the target/draft names, loaded model hashes, scale computation and
actual numeric write/read behavior. Do not simply disable the fingerprint or
copy old ABI binaries. Neither such a port nor an updated calibration artifact
was created in this task.

Other default-MRV2 exclusions include stock torch.compile, TP sequence
parallelism, unsupported speculative methods/parallel drafting, dual batch
overlap, custom logits processors, and KV-sharing fast prefill. Ordinary prefix
caching is distinct from KV-sharing fast prefill. Capture the final runner and
resolved configuration explicitly rather than relying on release defaults.

Raw evidence:
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/v029-full-feature-source/`
contains exported installed source hashes, hook fingerprints and the CPU filter
result. The earlier actual MRV2 log remains in the original TP1 arm unchanged.
