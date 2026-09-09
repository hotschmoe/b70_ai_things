# Next serving campaign: 800K resident context and correct recovery

Date: 2026-09-09. Status: maintenance authorized and campaign started.

The user subsequently explicitly authorized bringing down vLLM and starting
this campaign. Normal owning-lifecycle shutdown and post-health completed.
Execution evidence is appended in
[the campaign log](../vllm/cache800k/20260909_campaign.md).
The phase descriptions below remain the test protocol; the user's explicit
authorization satisfies the maintenance-window prerequisite.

## Objectives and priority

1. Preserve terminal-agent quality while providing four independent 200,000
   token total contexts simultaneously, with useful concurrent progress.
2. Qualify FP8 E4M3 KV, freshly calibrated if supported, on the exact selected
   weights and runtime. Keep FP16 queries initially and preserve recurrent
   state precision. Do not silently quantize recurrent state.
3. Diagnose and reduce the concurrency-sensitive exclamation-mark corruption.
   Compare SGLang and vLLM on the same failing workload and model.
4. If correct, add RAM retention/reload for evicted conversation state.
   This is desirable but optional; do not block an otherwise qualified
   800K GPU-cache result on RAM offload.
5. Evaluate a smaller-cache MoE only if terminal-agent quality is competitive.
   Faster decode or a smaller cache alone does not justify replacing Qwen.

200K is prompt plus generated/reasoning tokens per conversation, not a 200K
prompt followed by unlimited generation. Use 192K input plus an 8K reserve
for capacity pressure, with separate meaningful tool/retrieval tests. If the
desired contract changes to 200K input plus output, increase both per-request
limits and the aggregate target explicitly. Four configured request slots
and a 200K maximum length are not proof of an 800K resident cache.

## Starting evidence and uncertainties

- Latest qualified reference: Qwen3.8-27B AutoRound INT4 W4A16 g128, R276,
  TP2, MTP3, FP16 KV, GPU prefix caching, no CPU tier, 200K/c4 and 32K
  batch budget. The measured pool was 451,562 shared logical tokens; this
  count is not multiplied by TP ranks. Re-record actual service identity
  at maintenance time instead of assuming this remains the deployed pin.
- Current bangs are not an FP8-KV-only defect. The MTP3 shared soak recorded
  4/132 raw failures, recovered by the client; oversized histories recorded
  1/9. These workload-specific rates remain visible. Neither is a long-run
  reliability guarantee. No assumption that every client has a guard.
- Calibration and B0 FP8 model controls from September 8 accidentally
  imported /workspace/vllm rather than the production installed package.
  Their failure observations belong to that shadowed package. The old
  calibration is research-only, not a qualified artifact for the INT4 model.
- Installed scaled FP8 read/write oracles passed independently on both
  cards. Full-model corruption occurred with prefix caching in eager and
  graph execution on the shadowed package. Prefix-off passed a bounded
  control. This does not isolate TP2, hardware, calibration, or one kernel.
- Correct-package RAM experiments demonstrated real reloads and avoided
  some recomputation, but failed tool-history correctness. Existing patches
  are candidates to inspect, not approved serving fixes.
- Mamba boundary CPU regressions confirmed a bug; the backport did not
  clear GPU correctness. Generic prefill/decode overlap did not reproduce
  the current bangs. Concurrent growing cached tool histories are the
  stronger reproducer. NaNs are not confirmed on the current stack.

Evidence:

- [Prefix campaign](../vllm/int4/20260908_prefix_campaign.md)
- [KV/offload campaign and package correction](../vllm/fp8/20260908_kv_campaign.md)
- [Replica campaign](../vllm/int4/20260908_neural_replica_campaign.md)
- [SGLang NVFP4 control](20260828_qwen38_nvfp4_sglang_tp2.md)
- [Journal](../JOURNAL.md), including W01's MTP-off/radix-off 50K control

## Capacity accounting

Raw target attention KV bytes/token =
2 * full-attention layers * KV heads * head dimension * bytes/element.
The factor 2 represents K and V. These are aggregate bytes across TP ranks,
assuming sharded KV heads. Account separately for any replication.

| Model/config inspected | Full-attention layers | KV heads | Head dim | FP16 bytes/token | 800K FP16 GiB | 800K FP8 GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B AutoRound config | 16 | 4 | 256 | 65,536 | 48.828 | 24.414 |
| Ornith-1.5-35B-A3B local config | 10 | 2 | 256 | 20,480 | 15.259 | 7.629 |

These figures exclude MTP attention, recurrent checkpoints, hybrid padding,
allocator fragmentation, weights, graph buffers and workspace. MoE itself
does not imply a smaller KV cache; Ornith's particular attention layout does.
All experts still contribute to weight residency unless deliberately offloaded.

Doubling the current 451,562-token pool suggests 903,124 FP8 tokens, but this
is a feasibility estimate, not an allocator or serving measurement. Require
the actual block layout and per-rank memory ledger. Reserve output capacity
and demonstrate four active contexts without capacity-driven recomputation.
RAM retention does not count toward the 800K resident GPU objective.

## Phase 0: CPU-only preparation while production remains in use

Use isolated source copies/environments and small CPU/RAM limits. No GPU
device mounts, production package edits, serving-image tag replacement,
driver changes, native XPU compilation, or production request traffic.
Avoid full-weight scans/downloads or large builds that contend with serving.

Prepare and run bounded tests for:

1. Import identity: reject empty PYTHONPATH entries; record package paths,
   revisions, relevant source hashes and installed overlay selection.
2. Calibration schema: rank/layer coverage, max-across-ranks merging,
   finite/nonzero scales, zero floors, source/model fingerprints and rejected
   mismatches. Extend test_kv_campaign_calibrate.py where needed.
3. Actual cache-manager logic with synthetic small pools: block boundaries,
   partial hits, alignment, speculative rollback, recurrent checkpoint
   selection, eviction, refcounts, cancellation and reuse after free.
4. Offload metadata: request/group identity, complete compatible hybrid state,
   missing-state fallback, serialization, mocked transfer completion and
   cancellation before/after reload. Do not equate mocked DMA with real DMA.
5. Workload harness: raw SSE preservation, bang detection, retry accounting,
   timeouts, first bad token localization and admission/eviction reporting.
6. CPU numerical references for FP8 scale arithmetic and dequantization.
   These do not qualify XPU FP8 attention or collectives.
7. SGLang feature audit at an exact pin: model and quant loader, Intel XPU
   attention/GDN route, FP8 scales, MTP, prefix cache, graph capture and
   hybrid RAM offload. Mark absent combinations unsupported, not passed.

Existing starting tests include vllm/int4/test_mamba_eagle_drop.py and
vllm/fp8/test_kv_campaign_{calibrate,preflight,probe,agent_probe}.py,
test_kv_offload_group_fix.py and test_merged_offload_image.py. Inspect imports
and resource use before running any test. Do not run the full GPU model on
CPU merely to approximate GPU timing/state behavior.

## Phase 1: explicit maintenance window and reference replay

Before downtime, freeze the test matrix, artifacts, output directories,
rollback recipe, client impact and available time with the user. Prepare
everything reviewable first. No automatic stop is authorized by this file.

At the agreed window:

- Record kernel (fixed 7.1.0-070100), UMD, Level Zero, oneCCL, PyTorch,
  backend/image identity, model tensor/config hashes and effective settings.
- Record production identity using /v1/models, cross-check models.yaml,
  preserve launch/service configuration and drain requests before stopping
  through the actual owning lifecycle. Never start a second owner.
- Use bin/gpu-run for every GPU touch, including native oracles and health.
  Single-card runs use --card N plus the workload's matching device pin.
- Complete teardown, bin/xpu-health and bin/xpu-collective-health. Keep the
  qualified scoped collective settings; do not generalize direct P2P.
- Reproduce the reference's identity, quality, warm reuse, known trigger and
  performance with identical prompts and metrics. Retain raw failures.

Change one factor at a time. Rebuild ABI-specific extensions from tracked
source when required. No quarantined binaries. After failed/crashed TP>1
attempts use bin/xe-reset through its lease and the non-reboot recovery
ladder; do not stack failed launches or downgrade the kernel.

## Phase 2: current-model FP8 and corruption isolation

First use the current INT4 target in vLLM to minimize changes. Establish an
MTP0 FP16 control, then calibrate on the verified package and actual target
weights with eager FP16 KV and prefix reuse disabled. Include short code,
tools, diverse long contexts through 200K and long continuations. Keep held-
out qualification prompts separate from calibration. Recalibrate/qualify
when weights or materially relevant execution paths change.

Start with per-layer static E4M3 scales, recording K/V observations across
both ranks and a frozen headroom rule. The previous max-amax * 1.10 / 448
rule is a starting protocol, not proof of adequate calibration. Record
clipping/error on held-out data before changing it. Keep queries FP16 and
recurrent state at the control precision. Later calibrate the MTP path when
it is introduced; target-only coverage does not qualify draft attention.

| Step | KV | MTP | Prefix | Execution | Purpose |
| --- | --- | --- | --- | --- | --- |
| F0 | FP16 | 0 | off | eager | Reference |
| F1 | calibrated FP8 | 0 | off | eager | Storage/scale/model correctness |
| F2 | calibrated FP8 | 0 | on | eager | Cache reuse and boundaries |
| F3 | calibrated FP8 | 3 | on | eager | Speculative state interaction |
| F4 | calibrated FP8 | 3 | on | production graphs | Candidate |

At a failure, run the matching FP16 arm before attribution. Use TP1 versus
TP2 only where the same model/shape fits, with single-card leases. A smaller
surrogate tests mechanics but cannot isolate full-model TP2 causality.
Run installed cache write/read oracles with non-unit scales on each card,
multi-block/interleaved storage and speculative query lengths. Query the
actual cache block size; test before/at/after handoffs rather than assuming
the historical 1600-token boundary is unchanged.

For bangs, capture actual scheduler row/request order, accepted speculative
tokens, cache hit lengths and recurrent state indices on both ranks near
the first bad step. Compare shared/private salts, submission permutations,
serialized requests and growing histories. Client overlap alone is not
proof of backend co-batching. Instrumentation can hide timing failures;
retain uninstrumented repeats and do not infer NaNs from punctuation.

## Phase 3: matched SGLang comparison

Use the same target tensor bytes, prompt tokenization/template, precision,
context, concurrency and output budgets wherever supported. Record any
quant loader transformation or parser difference. Do not port a vLLM
runtime binary or assume backend flags have identical semantics.

Start SGLang FP16 KV with MTP/prefix/graphs off; qualify those features in
steps, then repeat F1-F4 where supported. A same-model FP16 comparison helps
separate backend behavior from FP8. Retain old SGLang failures as historical
evidence; its short C1 NVFP4 and MTP-off W8A8 results are not matched to R276.

Rank candidates by terminal-task success and raw corruption, then resident
capacity, latency under contention, and throughput. A slower correct backend
may be preferable, but record the actual tradeoff rather than infer one.

## Phase 4: 800K and conversational qualification

- Four distinct/salted near-192K inputs with 8K output reserve each; prove
  measured residency, admission and useful overlapping progress. Audit
  allocator capacity and preemption/recompute counters. Follow with semantic
  retrieval and code/tool tasks; forced-length output only measures pressure.
- Resident warm repeats, at least five distinct 200K-class histories to
  exceed the target pool, revisit evicted histories, then warm repeats again.
- Four growing tool histories, both shared and private prefixes, including
  the retained 88K and 132K triggers; repeated eviction and boundary crossing.
- Cancellation during prefill/decode, reuse after cancellation, fresh cold
  requests after churn, near-199K retrieval and poststress coding.
- Freeze a shared-history soak before execution: at least 128 logical
  checks per fresh lifecycle and two fresh lifecycles. Report raw attempts,
  bangs, timeouts, recovered checks and unrecovered errors by workload.
  Do not stop a successful soak early or dilute failures with easy requests.
- Report cold/warm TTFT, inter-token gaps, per-client and aggregate useful
  throughput, p50/p95, token budgets and completed outputs. Keep strict
  first-100 timing separate from full-response wall throughput.

Correctness target is zero observed corruption. Prior user acceptance of a
below-5-percent observed rate was a specific compromise, not a claim that
the bug is fixed or blanket approval for future promotion. Any residual
failures require an explicit review of the concrete candidate and recovery.
Finite clean tests do not establish an arbitrary long-run reliability rate.

## Phase 5: optional RAM recovery

Only add RAM offload to an otherwise qualified cache configuration. First
test GPU-only versus the same candidate with one bounded CPU tier (32 GiB
initially, adjusted to measured host headroom). Record storage precision,
complete hybrid state, actual loaded bytes and external hit tokens.

Force genuine eviction and show correct subsequent continuation after CPU
reload; distinguish that from a resident GPU hit or full recomputation.
Test cancellation/transfer completion, mixed attention/recurrent groups,
speculative rollback, active growth and post-reload tool/coding quality.
Include the old four-88K trigger. Require a repeat lifecycle and matched
CPU-tier-off control. A faster wrong answer is failure. If unsupported or
incorrect, retain the GPU-only winner and document the unresolved boundary.

## Phase 6: model/quantization quality alternatives

Keep Qwen3.8-27B INT4 as the reference. Its publisher's 73.0 Terminal-Bench
2.1 score is not a measured score for our exact local quant/runtime. Run a
matched Terminal-Bench version, agent harness, template/thinking policy,
token/time budgets and task set on shortlisted serving candidates. Capture
timeouts and infrastructure failures separately without silently dropping
them. Use HumanEval+ only as an additional coding regression screen.

Shortlist as of September 9 (refresh before acquisition):

- Ornith-1.5-35B-A3B: local INT4/NVFP4/W8A8 artifacts and much smaller target
  KV geometry. Publisher TB2.1 Terminus-2 67.8, below Qwen's 73.0. Useful
  capacity control; not a presumed quality-preserving replacement.
- Nex-N2.5-mini: newly identified 35B MoE candidate outside the confirmed
  neural.download catalog. Publisher TB2.1 73.4 uses NexAU, so it is not a
  controlled win over Qwen/Terminus. Inspect config, license, template, XPU
  support and available quant provenance first. No local B70 qualification.
- Laguna-S-2.1 INT4: neural.download's deployment is four B70s and publisher
  TB2.1 is 70.2. Not a leading two-card capacity solution.
- Qwen3.8 Flash-Next FP8: recent four-B70 recipe lacks qualified long-context
  and multi-user profiles. Defer unless a viable two-card artifact emerges.
- Refresh neural.download's exact Qwen INT4 source/quant ledger. Runtime
  speed updates are not new target weights or proven TB quality improvements.

Sources (publisher scores use different harnesses and are screening evidence):

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B
- https://huggingface.co/nex-agi/Nex-N2.5-mini
- https://huggingface.co/poolside/Laguna-S-2.1
- https://neural.download/models/laguna-s.html
- https://neural.download/models/qwen38-27b-autoround-int4-b70.html
- https://neural.download/models/qwen38-flash-next-fp8-tp4-mtp0-w13n64-b70-34tps-20260908.html

## Evidence, stopping rules and restoration

Each arm records CONFIG -> COMMAND -> RESULT -> VERDICT with immutable
model/package/image identity, actual effective settings, raw responses,
metrics, per-rank profile shapes/collective counts when the path changes,
health and teardown. Append journal evidence; never overwrite failed runs.
Encode backend/model/method/KV scheme in served IDs and result directories.

Stop a candidate on corruption, identity mismatch, hardware/collective
failure or exceeded resource bounds. Preserve the first failure and perform
bounded diagnostics only after appropriate health/recovery. Do not continue
long speed runs on corrupted output. Unsupported feature cells stay explicit.

Before the window ends, stop the test through its owning lease, complete
post-health and restore the preserved reference unless the user approves a
concrete qualified replacement. Validate authenticated identity, concurrent
coherence and real warm reuse on the restored endpoint. A candidate needs
fresh-start parity, teardown/post-health and concurrent qualification before
shelf promotion. Changes to bin/ or rdy_to_serve/_common/ require their
applicable live-shelf smoke checks before commit; none are part of this doc.

Deliverables: exact capacity ledger; frozen fresh scale artifact; backend
comparison; raw bang/recovery report; optional RAM reload qualification;
matched model-quality results; and a deploy/rollback decision with unresolved
limits stated. An incomplete matrix is not a completed campaign.
