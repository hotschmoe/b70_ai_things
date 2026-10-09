# Optional matched-health source preparation review

CONFIG -> CPU-only review on 2026-10-09. The audit-v1 JSON, audit Markdown and
plan-v1 describe the preserved 09:14..09:16 UTC identity-failure incident. Their
words "Current" and "INVALID" are incident-snapshot status, not a live status
claim. Incident JSON and plan bytes are preserved. The audit Markdown preserves
its prior prose and appends a timestamp-scoped later status.

The parent subsequently reported source recovery, all four restored full hashes,
CPU-idle and existing-health controls with unchanged buffered/direct sentinels,
and passing one-card and two-card C1 controls with post-run full hashes. Those
later results are separate evidence. This source review inspected the restored
four-hash receipt and both C1 qualification receipts, confirming their recorded
hash, lifecycle, teardown and post-health passes, with full model numerical /
state and concurrent qualification explicitly outstanding. It did not execute
those controls. They did not reproduce or establish the cause of
the earlier byte change. The optional controls below have not been built or run.

COMMAND -> Decode all nine preparation files as ASCII; parse JSON; compile Python
syntax in memory; verify the plan's source, builder and admission-contract SHA256
bindings; read source scope and qualification statements. Re-run the preserved-
page CPU admission test. No GPU, image, driver, model or cache operations.

RESULT -> ASCII, JSON, Python syntax and bound hashes pass. The preserved-page
test passes eight rejection controls. No speculative cause assertion or build /
GPU qualification claim was found. Pure SYCL matmul is a distinct workload from
Torch GEMM / XCCL. The optional derived health image is unbuilt; its compatibility
remains unproven.

Files ready for a coherent source-only commit, including this review:

- build_matched_sycl_health.py
- health-dma-source-audit-v1.json
- health-source-recurrence-audit.md
- health-source-preparation-review-v1.md
- health_page_control_contract.py
- matched-health-control-plan-v1.json
- matched_sycl_health_control.cpp
- test_health_page_control_cpu.py
- matched_health_runtime/Dockerfile
- matched_health_runtime/official-driver-packages-v1.json

Unresolved pinned identities and qualifications:

- No verified model loader 1.28.2 PPA package bundle is prepared. The first
  candidate retains health loader 1.32.0, so it cannot claim full model-runtime
  identity matching.
- Official IGC 2.36.3+21719 package bytes have not been downloaded or compared to
  model-image libraries from its 2.36.3 package build.
- Official NEO / IGC / GMM package checksums are declared; no downloaded-bundle
  verification or derived-image package / library census has been performed.
- Existing Torch, SYCL and oneCCL compatibility with the older runtime remains
  unproven. Non-GPU package / library equality must be verified before GPU use.
- Pure-SYCL source has not been compiled or qualified. Its builder pins the exact
  39992d70 image; a future binary still requires a separate build receipt.

VERDICT -> Ready as optional source preparation only. Preserve the frozen
incident evidence. This review grants no GPU launch or stack replacement and
does not identify or claim to fix the original cause.
