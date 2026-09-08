# R187 KV campaign, 2026-09-08

## Milestone 1: implementation and preserved baseline

CONFIG -> User authorized the maintenance campaign, testing, and conditional
daily-driver/systemd promotion. Original R187/MTP3 TP2 service was idle.
Preservation and raw evidence: /mnt/vm_8tb/b70/results/kv_campaign_20260908/.

COMMAND -> Save sanitized runtime configuration, original launcher/unit,
host/container package inventory, kernel and model identity. Stop the
container through its existing launcher cleanup. Implement a leased campaign
server with explicit native offload flags, isolated result/cache directories,
queued bounded probes, failure recovery, and pre/post health. Register model
identities. Add opt-in per-process calibration/scale loading source.

RESULT -> Original systemd service is inactive after clean teardown. Both
per-card and compiled two-rank pre-health checks passed for the new baseline.
Baseline startup and quality probes are in progress. Native ABI unchanged:
image f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152,
PyTorch 2.13.0+xpu, oneCCL 2022.0.0, container UMD 26.27.39122.11-0,
Level Zero loader 1.32.0, vLLM ac7509e2b with preserved R187 overlays,
host kernel 7.1.0-070100-generic. Campaign cgroup budget is 64 GiB for
both baseline and offload arms. No native extension was changed.

VERDICT -> Implementation checkpoint, not a performance/coherence result.
No shelf or service default changed. Interactive root authentication remains
necessary for systemd administrative updates; experiments use the authorized
Docker lifecycle and GPU lease. Retain the original installed boot config
until qualification finishes.

## Calibration and probe implementation checks

CONFIG -> Same preserved image; fresh model-specific per-tensor E4M3 scales
are planned. No quantized cache enabled yet.

COMMAND -> Implement a 256-sample synthetic code/math/chat/tool-oriented
corpus with reasoning enabled on a quarter of samples, additional long
prompts and continuations, cross-rank amax merging, and native DMA oracle.
Run Python compilation checks and four scale-merging unit tests.

RESULT -> Checks passed. The merger rejects missing ranks, mismatched layer
coverage and nonfinite observations, uses conservative cross-rank maxima,
and floors zero scales. The baseline passed 24/24 C1 and 24/24 C4 short
checks with exact repeats, plus 32K retrieval. The initial long-generation
regex incorrectly flagged a code-comment hyphen ruler; manual inspection
identified this false positive. The detector now excludes formatting rulers.
Original responses and the original failed automatic result remain intact.

VERDICT -> Pure-Python implementation checks passed; scale collection/loading
and offload are still unqualified. This is not a coding benchmark or a claim
that the current model is free of repetition. Larger-context probes continue.
