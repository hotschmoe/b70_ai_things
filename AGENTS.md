# AGENTS.md -- standing rules for the B70 project

Keep this file short. Detailed evidence lives in JOURNAL.md and the curated
documents listed below.

## Style and evidence

- ASCII only in repository files and terminal output.
- Record experiments as CONFIG -> COMMAND -> RESULT -> VERDICT.
- Append new JOURNAL.md entries at the bottom.
- Do not claim speed or stability without matched configuration, coherence,
  identity, health, and teardown evidence.
- Preserve user changes in a dirty worktree.

## Current scope

Backends:

- sglang is the primary serving and new-development backend.
- vLLM is a paused baseline plus the Steve transfer/control workspace.
- llama.cpp and ZML were removed from the live tree on 2026-08-26.
- Do not restore quarantined backend code or ABI-specific binaries into a new
  PyTorch/backend stack. Port the relevant source or finding deliberately.

Live model set:

- Qwen3.8-27B: BF16, RadixArk NVFP4, and compressed-tensors W8A8 GPTQ.
- Qwen3.6-27B: NVIDIA ModelOpt NVFP4 only.
- Ornith-1.5-35B-A3B: BF16+Shisa MTP, W8A8 RTN+Shisa MTP, NVFP4, and local
  GPTQ INT4 MixedCal-v2.

W8A8 INT8 remains the headline research target. Prefer compressed-tensors and
GPTQ for new W8A8 artifacts unless a measured comparison changes that choice.
W4A8 is secondary. Do not restart W4A4 work before W8A8 is robust.

## Clean-slate boundary

The 2026-08-26 cleanup moved old weights, repositories, caches, build trees,
Docker images, raw results, and historical docs to the gitignored directory:

  archive/to-delete-20260826/

Nothing there is part of the live source tree. It is a review buffer before
permanent deletion, not a dependency. The only external research trees kept
under /mnt/vm_8tb/b70 are steve-s2b and steve-repro.

When updating drivers, PyTorch, sglang, or vLLM:

1. Record host kernel, UMD, Level Zero, oneCCL, PyTorch, backend, and image
   identity before changing anything.
2. Change one layer at a time.
3. Rebuild every ABI-specific extension from tracked source.
4. Run per-card and compiled two-rank collective health before GPU serving.
5. Requalify model identity, deterministic coherence, concurrent serving,
   teardown, and post-health before shelf promotion.

## Host and GPU discipline

- Work locally as hotschmoe in
  /mnt/vm_8tb/github/b70_ai_things.
- Runtime root is /mnt/vm_8tb/b70. Model weights are in models/files/.
- Use bin/gpu-run for every real GPU touch.
- Use bin/gpu-run --card N for a one-card workload and pair it with the
  workload's device pin.
- Never bypass the lease for serving, benchmarking, profiling, compilation
  that touches XPU, or quantization.
- Kernel 7.1.0-070100 is the fixed host baseline. Do not downgrade it to
  imitate an older result.

## Multi-GPU safety

Kernel 7.1 plus Compute Runtime 26.22 cured the former GuC/BCS hardware wedge.
A separate vLLM multiprocess/oneCCL queue-handoff failure still exists.

- Do not run arbitrary CCL_TOPO_P2P_ACCESS=1 vLLM TP>1 serves.
- The old scoped Qwen3.6 direct-P2P control is historical evidence, not a
  production setting.
- Repeated TP>1 worker-init or graph-capture crashes can poison later
  collectives even with P2P disabled.
- Run bin/xpu-health and bin/xpu-collective-health around risky TP>1 work.
- Use bin/xe-reset after a failed/crashed TP>1 attempt. Reboot only if the
  non-reboot ladder fails.
- No card is display-held. Do not diagnose current reset failures using the
  retired display theory.

For dense 27B TP=2 experiments, measure the real profile shape and collective
count. The prior 8192-row clone-completion fence was specific to one Qwen3.6
control. Require matched per-rank entry/return evidence and post-health.

## Current transfer findings

These findings survived cleanup and should guide the refreshed stack:

- Exact June Qwen3.6 PIECEWISE control: 41 graph pieces, 41 fence resets,
  41 host event synchronizations, and 82 command-list submissions per token.
- No-MTP FULL decode with Triton attention collapses that to one fence, two
  waits, and two submissions and measured 61.5536 tok/s.
- Triton W8A8 MoE under FULL measured 64.9843 tok/s; native grouped MoE was
  slower on the local stack.
- Source-default c10d collectives replicated at a 66.3438 tok/s mean versus
  64.9944 for the custom route.
- June122 native scratch-targeted quant output improved the matched June9
  control by 3.79 percent but did not explain Steve's remaining gap.
- Steve's accepted path remains PIECEWISE with default FlashAttention.
  FlashAttention FULL capture fails because SYCL work-group scratch is not
  available to the graph extension.
- Push-AR preinit fixes IPC import but the loaded native push graph still
  stalls at first submission. Do not run a full-model push arm until the
  loaded-context oracle captures and replays.
- The pinned exact vLLM image already contains Steve-generation UMD 26.14.
  Host UMD changes are not required to reproduce that process boundary.

Primary evidence:

- docs/P2P_GPU.md
- docs/20260825_xe_nonreboot_recovery_and_pcie_topology.md
- docs/20260825_steve_stack_component_ledger.md
- docs/20260826_qwen36_graph_runtime_profile.md

## Repository layout

- sglang/: retained W8A8 and NVFP4 work.
- vllm/: retained W8A8/Steve controls, NVFP4, shared patches, and contributed
  custom-op source.
- kernels/: shared custom-kernel source of truth.
- rdy_to_serve/<backend>/<model-quant>/: verified shelf only.
- models/: manifest, fetch tooling, and gitignored weights.
- bin/: shared lifecycle, health, lease, and recovery tools.
- docs/: curated current evidence.
- evals/: retained model-quality evaluation tooling.
- archive/: ignored cleanup quarantine.

The retained pre-refresh shelf contains only Qwen3.6 NVFP4 and Qwen3.8 NVFP4
configuration controls. Their old native runtime libraries are quarantined, so
rebuild and requalify them before serving. Do not add a shelf entry without a
measured concurrent coherence and speed qualification.

## Model identity

Before trusting an eval or benchmark:

1. Query /v1/models on the live server.
2. Cross-check the served ID against evals/configs/models.yaml.
3. Encode model, method, and scheme in served IDs and output directories.
4. Never use a bare ambiguous ID such as qwen3-27b-w8a8.

## Git workflow

- Commit from the single clone at /mnt/vm_8tb/github/b70_ai_things.
- Commit and push coherent checkpoints often.
- Do not rewrite old experiment evidence. New experiments belong under the
  relevant backend root.
- Changes to bin/ or rdy_to_serve/_common/ require the applicable live-shelf
  smoke checks before commit.
