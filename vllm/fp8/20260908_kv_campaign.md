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

## Milestone 2: baseline and native DMA evidence

CONFIG -> A0 preserved R187/MTP3, FP16 KV, no host offload, 64 GiB cgroup.
Available KV memory 10.13 GiB per rank; reported shared capacity 292968 tokens.

COMMAND -> Run C1/C4 repeats, long generation, 32K retrieval, sequential
150K histories A/B/C/A, simultaneous 150K requests with 1024 output tokens,
four concurrent four-turn tool sessions, and native DMA round trips.

RESULT -> Short checks 48/48 exact; tool-history checks 32/32; 32K and four
150045-token retrievals correct. Evicted-history A/B/C/A took 373.418 s
total, with TTFTs 93.047/93.150/93.277/93.280 s and zero cached tokens.
Two 150071-token requests each produced 1024 tokens correctly in 219.141 s
total; TTFTs 202.633 and 93.312 s. No preemption: full-input reservation
serialized admission. Native swap_blocks_batch copied 4 MiB GPU -> pinned
RAM -> GPU byte-exactly on both cards after clearing the GPU source.
Cgroup peak 11574472704 bytes, no OOM/max events.

A queued stop-job rename caused a FileNotFoundError in the harness after
the completed probes, before a proposed growth test. The model shut down
cleanly. An unnecessary reset command then attempted a nested lease because
Python subprocesses do not inherit the lease FDs by default; that waiting
command was terminated before any reset. Per-card AND compiled two-rank
post-health passed. A0 exit.rc=1 records the harness error, not GPU failure.
The runner now re-reads one pending job at a time, tolerates withdrawn jobs,
and explicitly propagates its existing lease to xe-reset. Recovery uses
--no-probe followed by health on the actual campaign image.

VERDICT -> Baseline controls and DMA primitive passed the stated checks;
no active preemption recovery result yet. The growth case remains pending.
A1 adds only a 32 GiB native CPU tier, larger than the approximately 20.3 GiB
GPU pool, with the same 64 GiB cgroup. No service promotion.
