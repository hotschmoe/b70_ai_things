# RESEARCH_TODO.md -- clean-stack work order

Updated 2026-08-26. Work top to bottom. Do not mix environment refresh,
performance changes, and model changes in one comparison.

## P0: freeze the clean baseline

- Record kernel, firmware, host UMD, Level Zero loader, oneCCL, PyTorch,
  sglang, vLLM, and retained Docker image identities.
- Run bin/xpu-health on both cards.
- Run bin/xpu-collective-health.
- Confirm archive/to-delete-20260826 is not referenced by live scripts.
- Confirm models/manifest.yaml has exactly the eight retained artifacts.
- Leave Steve's steve-s2b and steve-repro runtime trees untouched.

Exit gate: identities recorded, both health layers green, no live server, and
the worktree cleanup diff reviewed.

## P1: refresh one software layer at a time

Order:

1. Host user-mode driver and Level Zero, only if an update is intended.
2. oneAPI/oneCCL runtime.
3. PyTorch XPU.
4. sglang.
5. vLLM paused baseline.
6. Custom extensions rebuilt from kernels/ and retained source.

For every layer:

- record before and after versions;
- rebuild ABI-bound extensions;
- run CPU import/static tests;
- run one-card XPU smoke through bin/gpu-run;
- run compiled TP=2 collective health;
- record result and verdict before the next layer.

Do not reuse quarantined wheels, shared libraries, compiler caches, or old
container layers as proof that the refreshed stack works.

## P2: re-establish serving baselines

Test in this order:

1. Qwen3.6-27B NVIDIA NVFP4, one card.
2. Qwen3.6-27B NVIDIA NVFP4, TP=2.
3. Qwen3.8-27B RadixArk NVFP4.
4. Qwen3.8-27B BF16 correctness reference.
5. Qwen3.8-27B W8A8 GPTQ on sglang.
6. Ornith-1.5-35B-A3B W8A8 RTN+Shisa MTP on sglang.
7. Ornith NVFP4 and BF16 references as needed.

Each candidate needs:

- exact /v1/models identity;
- deterministic semantic probes;
- concurrent prefill plus decode coherence;
- model and KV memory ledger;
- c1 and c4 performance;
- graceful teardown;
- per-card and compiled collective post-health.

Only then update rdy_to_serve/.

## P3: dense 27B graph and collective census

Use Qwen3.8-27B or Qwen3.6 NVFP4 as the dense control.

- Count real graph pieces, collectives, fences, waits, and submissions.
- Measure profile input shapes on both ranks.
- Derive any queue-completion fence from those shapes; do not copy the old
  8192-row threshold.
- Compare PIECEWISE with a no-MTP FULL-decode arm.
- Separate host-boundary timing from synchronized model-forward timing.
- Require matched per-rank entry and return evidence.
- Run health before and after every crash-prone TP=2 arm.

Exit gate: reproducible graph census and a clear verdict on FULL capture.

## P4: W8A8 first

Qwen3.8 dense:

- restore a clean compressed-tensors W8A8 loader;
- verify true INT8 weight and activation routes;
- rebuild B70 XMX kernels against the refreshed ABI;
- measure prefill, decode, and concurrent serving;
- keep BF16 and NVFP4 matched controls.

Ornith MoE:

- restore fused routed-expert W8A8;
- verify retained Shisa MTP identity;
- separate dense linears that dequantize from true W8A8 operators;
- profile launch count and collective cost;
- qualify coherence before performance tuning.

W4A8 follows only after both W8A8 paths are stable. W4A4 remains deferred.

## P5: Steve transfer after the refresh

- Re-run the exact PIECEWISE provenance control from the preserved Steve trees.
- Re-run synchronized component timing.
- Re-test source-default c10d versus custom collectives.
- Re-test no-MTP FULL with Triton attention.
- Keep default FlashAttention PIECEWISE as the exact attention control.
- Do not retry Flash FULL without a concrete runtime/kernel change.
- Do not run loaded push-AR beyond the bounded oracle until graph replay works.

Target: identify the remaining in-graph cost without changing host kernel 7.1
or conflating source, native binary, graph policy, and collective route.

## P6: documentation and purge

- Keep AGENTS.md, FINDINGS.md, and this file short.
- Move superseded evidence to quarantine instead of adding another banner.
- After the refreshed baselines are green, review the quarantine manifest.
- Permanently delete only after confirming Steve trees, retained model weights,
  current docs, and rebuild source are outside the quarantine.
