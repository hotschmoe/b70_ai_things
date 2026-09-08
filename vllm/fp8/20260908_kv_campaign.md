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

## Installed FP8 scale oracle

CONFIG -> Same image on each card independently while the campaign owns
both leases; no model configuration change. FP16 queries, E4M3 K/V storage,
12 query heads / 2 KV heads / head dimension 256 per rank, 31 cached tokens.

COMMAND -> Write K/V with non-unit scales through reshape_and_cache_flash;
read with the installed XPU flash_attn_varlen_func; compare against attention
on the actually dequantized cache. Double V read scale and perturb K read
scale to prove the kernel consumes both.

RESULT -> Both cards passed. K/V write relative L2 errors were 0.026386 and
0.026646, consistent with quantized storage. Native attention relative L2
error against the dequantized reference was 0.000241. Doubling V descale
doubled attention output; changing K descale changed attention weights.
The experiment is in a1/03-fp8-scale-oracle.log.

VERDICT -> This exact image mechanically supports scaled E4M3 KV writes and
reads with FP16 queries. This is not model-quality evidence. Fresh Qwen3.8
calibration and full-model qualification are still required.

## Stock offload and hybrid lookup diagnosis (in progress)

CONFIG -> A1 retains the A0 image, MTP3, graphs, cache allocation and 64 GiB
cgroup; adds 32 GiB through OffloadingConnector. The installed default is
prompt-only storage. Raw evidence is under the campaign root a1/.

COMMAND -> Repeat short C1/C4, tool-history, 32K retrieval, 150K A/B/C/A,
two concurrent 150K prompts with 1024 output tokens, and longer generation.
Run the installed scheduler lookup on synthetic sparse Mamba checkpoints.

RESULT -> Short checks 48/48 and tool-history checks 32/32 passed. All four
150K retrievals were correct but reported zero external hits and zero cached
tokens: 372.320 s total versus A0 373.418 s. Concurrent 150K requests took
218.827 s versus A0 219.141 s; neither arm preempted on that trace. Natural
132K growth also did not preempt. Two 2048-token outputs repeated exactly
within A1; A0 versus A1 differed in a late code comment, so cross-lifecycle
long-output byte identity is not claimed.

The installed scheduler's unannotated EAGLE fallback marks all four cache
groups as draft groups, including three target Mamba groups. Its extra
checkpoint requirement rejects isolated Mamba checkpoints. With identical
synthetic saved keys, the installed lookup returns zero tokens with all
groups marked EAGLE and 129792 tokens with only attention marked EAGLE.
The oracle is a1/08a-lookup-oracle.log. This is scheduler evidence, not yet
full-model evidence for the opt-in repair.

VERDICT -> Stock offload has not improved evicted-history reuse. A separate
forced-length stress trace has triggered preemptions and native loads; its
completion and matched no-offload comparison are pending. The experimental
repair changes only the unannotated hybrid fallback and requires prompt-only
storage. Explicit draft annotations and the attention safety margin remain.
No service promotion has been made.

## A1 completed pressure and teardown

CONFIG -> Same A1 lifecycle. Two independent 132071-token prompts with
ignore_eos=true and 8192 output tokens each; bounded 900-second request
deadline. This is a forced-length diagnostic, not a useful-output score.

COMMAND -> a1/11-forced-growth132k; then STOP and the runner's per-card and
compiled two-rank post-health. Repair CPU tests: a1/12-fix-unit.log.

RESULT -> Both requests completed at the length limit: 16384 tokens in
394.110 s; six preemptions; 2.416 GB loaded in two transfers, 2.284 s total
recorded transfer time. Maximum streaming gaps were 34.677 and 36.273 s.
Peak cgroup usage 11888189440 bytes, no OOM/max events. Clean shutdown,
per-card health and compiled collective health passed; a1/exit.rc=0.
All four guarded-repair CPU tests passed.

VERDICT -> Active preemption can load some native cached blocks, unlike
the zero-hit A/B/C/A history trace. Matched no-offload forced growth remains
required before claiming a benefit. A1fix cache seeding failed on unreadable
container-created files before health or serving; a1fix2 uses a fresh cache
and a frozen, hashed experiment-source snapshot.

## Held-out coding evaluator preparation

CONFIG -> Repository evals/orchestrator/tier1_code.py, EvalPlus 0.3.1,
HumanEval tasks 0-31, greedy seed42, thinking off, 2048 output-token cap.
Independent CPU Python 3.11.15 environment under the raw campaign root.
Sandbox image c0522083adc7557aab54abff248a4a4a9f7c32b4d9d66ddf9af05200ae5a3339.

COMMAND -> Freeze heldout-humaneval32.json; run canonical solutions through
the existing Docker grader before model-generated evaluation.

RESULT -> Dataset SHA256
45cdd6b24ffd395d415147ee95616cbebf70e909dd92a97ee767f0ccef48a414.
Canonical solutions passed all 32 base and extra-test cases. Dependency
freeze and grader-control logs are retained in the campaign root.

VERDICT -> Grader control passed. This is not a model coding-quality result;
matched model generations remain pending.

## Hybrid repair: real history reload, not yet promotion timing

CONFIG -> A1fix2, original image, auto KV, MTP3, native CPU32, guarded
Mamba fallback repair and read-only lookup tracing. Early sitecustomize
imports were used in this diagnostic. Profiling allocated 9.97 GiB/rank
and 288281 tokens versus A1's 10.13 GiB/rank and 292968 tokens.

COMMAND -> Short C1/C4; two concurrent 32K needles; identical 150K A/B/C/A.

RESULT -> Short 48/48 checks repeated exactly. Both 32K needles passed;
their C2 timings are not comparable with the earlier C1 probe. All four
150K answers matched exactly. TTFT: 135.476/135.683/135.820/8.857 s.
The final A request restored 148928 tokens and recomputed only the 1117-token
tail. Trace shows the three Mamba groups have sparse ready checkpoints
38/77/116/155/178, while attention has ready chunks 0-179. Clearing only
the Mamba EAGLE flags permits the real lookup. Native load metrics sum
10.527 GB and 13.488 s across two rank transfers; this sum is not wall time.

VERDICT -> Full-model retrieval validates the sparse-checkpoint diagnosis.
The 8.857 s restored TTFT is promising, but total makespan 416.849 s is
worse than the stock trace due to slower cold prefill. No performance
promotion claim. A follow-up loader defers hooks until normal vLLM module
imports; whether early imports caused the difference remains unproven.
Long interleaved tool-history and later matched lifecycle checks are pending.

Related upstream correctness report, a different stack/draft method:
https://github.com/vllm-project/vllm/issues/53505
It motivates interleaved hybrid-history checks, but does not establish that
this image has the reported corruption. The issue body is retained under
source/issue53505.json in raw campaign evidence.

## Repair lifecycle close and calibration protocol

CONFIG -> A1fix2 tool history: four interleaved sessions, four tool/result
turns each, 4000 background records, 44314-44662 prompt tokens.

COMMAND -> a1fix2/05-agent-long; STOP; per-card and compiled post-health.

RESULT -> All 32 checks passed. Peak cgroup usage 13203070976 bytes, no
OOM/max events. Teardown and both post-health types passed; exit.rc=0.

VERDICT -> Stated functional gates passed for this diagnostic. Cold-prefill
regression and broader matched pressure/quality qualification remain open.

CONFIG -> Calibration uses auto/FP16 KV, eager execution, prefix caching
explicitly off, MTP3 and the same publisher weights/native image. A COLLECT
sentinel arms recording only after readiness, excluding startup profiling.
The eight-sample functional pass recorded all 17 layers on both ranks with
finite values. The main corpus has 256 diverse synthetic short requests,
32K/96K/150K/180K contexts and two long continuations. Model file hashes
cover 81 files, 30890049597 bytes, under preservation/model-files-sha256.json.

COMMAND -> kv_campaign_calibrate.py --samples 256 --long --record-root ...
with per-layer eager Attention.forward activation collection.

RESULT -> Functional eight-sample pass succeeded; full collection in progress.
Installed fa_utils.py explicitly disables quantized query input on XPU.

VERDICT -> Before held-out evaluation, fix the artifact rule to per-layer
max amax across TP ranks times 1.10 / 448, floored at 1e-6. Q observations
are retained, but the XPU query path remains FP16. Frozen artifacts must
include observed counts and source hashes. This is synthetic text calibration;
it does not establish image/audio or arbitrary out-of-distribution quality.

## Frozen calibrated KV artifact

CONFIG -> Calibration protocol above, source snapshots in calibration/source/.

COMMAND -> Merge kv-record-479.json and kv-record-530.json with
kv_campaign_calibrate.py --headroom 1.1 --model-config
models/files/qwen3.8-27b/fp8-official/config.json --provenance
<raw-root>/calibration-provenance.json after clean shutdown/post-health.

RESULT -> Main corpus: 262/262 requests, 472317 input tokens and 20540
generated tokens. Plus eight functional requests. Both continuations reached
4096 tokens; contexts through 180K completed. Both ranks captured 17 layers,
7260-21780 forwards per layer and 500653-515173 token rows. These are actual
periodic snapshot counts, not a claim that every final decode step flushed.
Peak cgroup usage 23043252224 bytes, no OOM/max events. Clean teardown,
per-card and compiled collective post-health passed; calibration/exit.rc=0.

Artifact: kv_scales/qwen38_official_fp8_kv_e4m3_cal20260908_h110.json.
SHA256: e552af52da60cf3a35f5a981ee2ff89c11e144a020693cbfab1ca98b3da69bfc.
K scales span 0.030519-0.056550; V scales 0.020449-0.346205. The artifact
includes exact model-file, corpus, response, native-image and source evidence.
The loader rejects a mismatched model config fingerprint/schema and records
the actual query-quantization flag and attention implementation on each rank.

VERDICT -> Fresh model-specific static scales are frozen before held-out
evaluation. No FP8 model-quality or performance conclusion yet. First test
calibrated E4M3 eagerly, then qualify the production graph configuration if
the mechanical and coherence gates pass. Original service defaults remain.

## Calibrated E4M3 eager model gate

CONFIG -> B0eager: frozen e552af52da60 scales, E4M3 KV, FP16 queries,
unchanged FP32 SSM cache, MTP3 TP2, eager, prefix cache on, CPU tier off.

COMMAND -> Audit both rank load records; C1/C4 deterministic checks;
32K retrieval; four short interleaved tool sessions; STOP and post-health.

RESULT -> Both ranks loaded all 17 layers with the expected artifact hash,
FlashAttentionImpl and query_quantized=false. All 48 deterministic checks
passed with exact repeats; 32K retrieval and 32 tool-history checks passed.
Attention block size is 1600, Mamba page padding 0.25%. The eager profiler
allocated 9.31 GiB/rank and reported 511428 tokens, versus 271936 tokens for
the FP16 eager collector at the same profiled budget. The collector had
prefix caching off and recording enabled, so it is not a timing baseline.
Peak cgroup 11371433984 bytes; no OOM/max events. Clean teardown, per-card
and compiled collective post-health passed; b0eager/exit.rc=0.

VERDICT -> Mechanical scaled-KV loading and stated eager coherence gates
passed. Production graph capture, held-out coding, default thinking,
cancellation, longer contexts and matched pressure tests remain. The reuse
probe now accepts an explicit sequence so FP8's larger pool can be forced
to evict, rather than accidentally testing only resident GPU prefixes.
