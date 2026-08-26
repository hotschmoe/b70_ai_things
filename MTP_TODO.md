# MTP_TODO.md -- current retained-model plan

Updated 2026-08-26.

## Rules

- MTP work follows backend/PyTorch refresh and base-model coherence.
- Never mix an MTP change with a kernel, graph-policy, or collective-route
  change in the same A/B.
- Verify the exact MTP tensors and served model identity before performance
  claims.
- Test deterministic output, c1 latency/decode, c4 aggregate throughput,
  prefix-cache behavior, long context, teardown, and post-health.
- Record acceptance length, draft count, and target verification cost.

## Qwen3.8-27B

1. Requalify target-only W8A8 and NVFP4.
2. Verify the retained native MTP payload against BF16.
3. Compare MTP off/on under the same PIECEWISE graph policy.
4. Test no-MTP FULL decode to establish the boundary-collapse control.
5. Add MTP to FULL only after target-only FULL is coherent and replay-stable.

## Ornith-1.5-35B-A3B

1. Verify the Shisa trained MTP sidecar hash and tensor mapping.
2. Requalify target-only W8A8 on refreshed sglang.
3. Measure MTP1 acceptance and launch/collective overhead.
4. Keep native untrained and Shisa trained MTP results separate.
5. Do not promote until concurrent coherence and long-context behavior pass.

## Steve control

Steve's exact result used PIECEWISE and default FlashAttention. Retain that as
provenance. FULL+Triton is a measured intervention, not an exact reproduction.
Push-AR remains bounded to the loaded-context oracle until graph replay works.
