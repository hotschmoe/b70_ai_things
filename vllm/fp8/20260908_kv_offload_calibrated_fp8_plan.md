# Later: host KV offload and calibrated FP8 KV on the R187 endpoint

Date: 2026-09-08 UTC. Status: research and proposed tests only.
The user explicitly deferred execution while the endpoint is in use.
No inference, GPU health workload, restart, installation, calibration, or
serving configuration change was performed for this plan.

## Read-only findings

CONFIG -> Live container hotschmoe-dd, Qwen3.8-27B official FP8 weights,
W8A16 runtime, R187/MTP3, TP2, FP16 KV, prefix caching, 200000 max context,
4 sequences, 32768 batched tokens, GPU utilization limit 0.96.

COMMAND -> Read repository launchers and historical evidence; docker inspect;
GET http://127.0.0.1:18124/v1/models; free -b; read installed Python source
with docker exec shell tools (no torch/vLLM imports); review upstream docs.

RESULT -> Live model ID is hotschmoe-dd. Image ID is
sha256:f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152.
Installed version metadata is 0.27.2rc1.dev77+gac7509e2b; local overlays
mean the version string alone is not a complete source identity.
Source package pin is 8319e0964df12a1f0bc920301efc662ac49a949e.
September 7 startup logs report Flash Attention, 832-token attention blocks,
4.26 percent Mamba page padding, and 292968 shared cache tokens (1.46x 200K).
Container memory and memory+swap limits are both 32 GiB; no additional
container swap allowance. Snapshot cgroup usage was about 9.87 GiB.
Host RAM is about 121.42 GiB, with about 101.12 GiB available at inspection.
These memory readings are snapshots, not peak budgets.

VERDICT -> Both ideas have concrete source support worth qualifying, but
neither is validated for this exact hybrid/MTP3/graph combination. Preserve
the current endpoint until a maintenance window is explicitly started.

## A. Native CPU KV offload first

The installed OffloadingConnector subclasses SupportsHMA. Its CPU worker
contains XPU transfer paths, and its scheduler handles Mamba state alignment
and preempted requests. The local configuration adapter explicitly requires
compatible hash/group block sizes and points hybrid models at prefix caching.
This is stronger evidence than generic hardware support, but still source
inspection rather than an exercised round trip.

Use the native OffloadingConnector as the first candidate. The upstream
[offloading guide](https://docs.vllm.ai/en/v0.28.0/features/kv_offloading_usage/)
describes a prefix-cache extension: completed blocks are saved to host RAM
as produced, then loaded on demand. It is not an unlimited active GPU cache.
CPU capacity is aggregate across workers; default chunking follows GPU
blocks. A useful CPU-only tier should exceed aggregate GPU KV bytes, since
a smaller mirror can add transfers without retaining evicted history.

Candidate configuration fragment for a FUTURE experimental launcher:

```text
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":17179869184,"blocks_per_chunk":1}}'
```

The 16 GiB value is a bounded functional screen, not the selected production
size. Compute actual aggregate KV allocation first, including all ranks,
attention, recurrent state, draft cache, padding, and allocation overhead.
Then test a CPU tier larger than that allocation, provisionally 32/48 GiB
if host headroom permits. Set the experimental cgroup limit from measured
peak non-offload usage + CPU tier + explicit headroom (at least 8 GiB), and
reserve at least 24 GiB for host/other services. Match that cgroup limit in
the no-offload control so memory policy is not an A/B confound.

The installed --kv-offloading-size / --kv-offloading-backend native shortcut
can select SimpleCPUOffloadConnector through VLLM_USE_SIMPLE_KV_OFFLOAD.
Prefer the explicit connector above to remove that ambiguity. Do not also
set the shortcut. Do not copy a generic 64-token block size into this hybrid
model. Do not substitute weight --cpu-offload-gb, OS swap, or historical V0
--swap-space for V1 KV reuse. No LMCache dependency is needed for the first
arm; consider it only if a specific native limitation is demonstrated.

The decisive experiment is active-request preemption and recovery, separate
from a completed-prefix reuse benchmark. Prove which completed prompt AND
generated blocks survive, whether Mamba state resumes at the same boundary,
how many tail tokens are recomputed, and whether transfer cost beats avoided
prefill. A reduction in preemption count is not required if resume cost falls;
a higher cache-hit counter alone is not proof of success.

## B. Fresh calibrated E4M3 KV for these weights

Retained source references:

- vllm/nvfp4/kv_calibrate.py: diverse calibration-request driver.
- vllm/nvfp4/kv_make_scales.py: per-layer K/V amax to dequant scales.
- vllm/nvfp4/patches/sitecustomize.py, block 10: post-RoPE/post-norm recorder and
  post-load device/host scale injection.
- vllm/nvfp4/kv_gate.py and vllm/nvfp4/tp2_longctx_qualify.sh: quality/long-context gates.
- docs/20260905_qwen38_fp8_vs_int4_quality_review.md: quality synthesis.

Read-only historical evidence in archive/to-delete-20260826:
JOURNAL.before-trim.md, July 8 Thrust 2 (lines 8910 onward), and
docs/20260722_mtp_fp8_prefixcache_kickoff.md. Nothing is restored from archive.
The July 8 follow-up found scale consumption on XPU and long-context capacity
gains, but did NOT reproduce earlier repetition with scale=1 on a clean TP1
no-MTP configuration. Calibration was near-neutral there. Earlier claims
that uncalibrated KV alone caused repetition were explicitly revised.
The July prefix-cache report also records MTP/FP8 cache-hit failures.
These are Qwen3.6 NVFP4 findings, not Qwen3.8 official-FP8 qualification.

The upstream [quantized KV guide](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
recommends dataset calibration and distinguishes tensor/head scaling.
Its generic examples and CUDA backend claims do not establish the loaded
XPU implementation's behavior. Installed Flash Attention declares fp8/e4m3
and passes layer scales; exercise the actual native writer AND reader later.

Proposed implementation and calibration procedure:

1. Create a small tracked Qwen3.8-specific recorder/loader under vllm/fp8/.
   Port the relevant source only; do not load the entire NVFP4 shim or reuse
   Qwen3.6 scale values. Keep official weight files read-only and unchanged.
2. Record from a clean eager FP16-KV reference with the exact current weights,
   target/MTP modules and dtypes. The old recorder only observes FP8 layers,
   swallows errors, and shares an output path; port it to observe reference
   tensors, fail on missing/nonfinite data, and use separate rank files.
   Verify fused paths do not bypass the observation point. Disable prefix
   reuse during collection so samples are actually observed.
3. Use a fixed, hashed corpus of 256-512 coding/chat/tool/JSON/reasoning
   samples, plus a small long-context stratum at 32K/96K/150K/near-200K and
   several 4K-8K generated continuations. Keep quality evaluation prompts
   held out. Log actual token lengths and target/draft layer coverage.
4. Start with static per-layer, per-tensor K/V dequant scales:
   scale=max(amax/448 * headroom, 1e-6). Use 1.0 and 1.1 headroom as small
   calibration-validation candidates; freeze the choice before final eval.
   Record clipping and quantization error on held-out activations. Headroom
   is a measured tradeoff, not free precision. Merge TP maxima conservatively
   for shared layer scales or explicitly certify rank-specific scale mapping.
5. Enumerate all cache-writing target and draft layers. Inject finite positive
   scales after weight postprocessing but before graph capture, including
   required host mirrors. Assert counts and hashes on BOTH ranks. Check if
   this backend needs Q scales; calibrate them if Q is actually quantized.
   Prove write/read scale consumption in a bounded eager round-trip oracle.
6. Test --kv-cache-dtype fp8_e4m3 with the frozen artifact. Preserve recurrent
   GDN/SSM and convolution state precision. Record draft-cache dtype explicitly;
   keeping draft KV FP16 is a labeled mixed-cache arm, not an all-FP8 claim.
   Do not treat --calculate-kv-scales startup behavior as dataset calibration:
   older hybrid code disabled it, and installed behavior needs verification.
7. Remove recording hooks before performance tests. Recreate graphs with the
   frozen scales and remeasure cache group sizes, padding, MTP acceptance,
   and prefix hit boundaries. FP8 halves K/V element storage, not necessarily
   total cache memory or the hybrid model's effective token capacity.

## Controlled test order for the maintenance window

Before starting, drain clients and stop the service through its existing
leased lifecycle. Save the exact launcher/config/image/source/library hashes
and host kernel/UMD/Level Zero/oneCCL/PyTorch identity. Use bin/gpu-run for all
GPU work, with per-card and compiled two-rank collective health before/after.
Keep the kernel and current qualified collective route fixed; do not add new
P2P settings. Failed TP2 attempts require bin/xe-reset and recovery health
before another launch; reboot only after the non-reboot ladder fails.

Reconcile evals/configs/models.yaml before scoring: it currently lacks the
live alias/physical model mapping. Preserve the user-requested public alias
hotschmoe-dd; bind each experiment's descriptive directory/manifest to exact
weights, scheme, scales and runtime, and verify /v1/models each lifecycle.
Use isolated result/cache paths and a test port during the drained window.

| Arm | KV | CPU tier | Purpose |
| --- | --- | --- | --- |
| A0 | current FP16 | off | Matched baseline |
| A1 | current FP16 | native | Isolate save/load and recompute benefit |
| B0 | calibrated E4M3 | off | Isolate capacity, quality and graph effects |
| AB | calibrated E4M3 | native | Interaction; only after A1 and B0 pass |

Run small eager correctness/transfer oracles before full graph/MTP3 tests.
If isolation needs MTP0 or another attention backend, establish a matching
control and label it separately; do not credit its results to the daily driver.
The September 7 MTP/prefix repetition investigation remains relevant. Offload
extends cache reuse, so it cannot be presumed to fix that correctness issue.
If baseline quality fails, resolve it in a separate matched change first.
Prefix-off and MTP0 are diagnostic controls, not silent changes to this matrix.

For each qualified arm, use the same fixed arrival trace and tokenized prompts:

- Low pressure C1/C2/C4 with short prompts; cold and warm repeated prefixes.
- Long contexts at 32K/96K/150K/near-200K, leaving room within 200K for output.
- Pressure ladder around 0.8x/1.05x/1.3x measured aggregate GPU token capacity,
  counting generated and speculative reservations. Two independent 150K
  prompts plus output are a natural near-pressure case for the current 293K
  pool. Three/four requests can raise pressure without changing MAX_NUM_SEQS.
- Both unique prefixes and shared agent/tool histories; force eviction using
  competing requests, then return to the original history. Include active
  preemption, cancellations, partial blocks, resume, and repeated tool turns.
- Run fixed-demand comparisons across all arms, then capacity-normalized
  overload comparisons. A larger FP8 pool must not silently weaken its stress.

Capture request timing, TTFT p50/p95/p99, inter-token stalls, end-to-end
latency, total makespan, completed useful output tokens/sec, errors/timeouts,
preemptions, recomputed tokens, CPU load/store bytes and hits/misses, queue
depth, CPU/XPU cache occupancy, cgroup peak/events, host memory pressure,
MTP acceptance, actual block geometry and transfer bandwidth. Add scheduler
instrumentation if installed metrics cannot distinguish restored/recomputed
tokens. Retain generated outputs for grading; health metrics cannot grade
coherence. Do not claim tail-percentile reliability from a handful of samples.

Quality gates: deterministic repeated greedy canaries within each arm,
evict/reload output checks against uninterrupted execution, long generation
repetition checks, needles at varied depths, structured tool/JSON correctness,
and the same held-out coding suite used for the baseline. FP8 need not match
FP16 token-for-token, but within-arm repeatability and task quality must pass.
Byte-copy offload should preserve cached values; validate this directly as
well as end-to-end. Repeat the main pressure trace at least three times and
include a second clean lifecycle, matched teardown and post-health.

Proposed promotion criteria, to freeze before measurement: all coherence and
state-recovery gates pass; no new deterministic coding-suite failures;
at least 20 percent lower overload makespan or p95 request latency, with
supporting avoided-recompute evidence for offload; no more than 5 percent
low-pressure throughput regression beyond measured run variance. Report both
latency and throughput even when only one clears the benefit threshold.
Abort a bounded stress case on its predeclared timeout, repeated no-progress
preemption cycle, memory pressure breach, output corruption, or device error.
Never run an unbounded thrashing soak.

Rollback: stop the experimental launcher, complete teardown/post-health,
restore the exact saved service configuration and original image/scales-off
settings, then verify alias, deterministic coherence and normal serving.
No shelf or boot-default changes until concurrent qualification is complete.
Record each future experiment as CONFIG -> COMMAND -> RESULT -> VERDICT
under vllm/fp8/ and append its summary to JOURNAL.md.
