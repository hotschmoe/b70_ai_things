# Bang/corruption campaign: resume handoff

Written after the user paused testing and authorized the 2026-09-10 day trial.
Read this file first in the next session, then AGENTS.md and the linked evidence.
This is a backlog and operational handoff, not authorization to interrupt service.

## Live service boundary

- hotschmoe-dd.service is serving the user-authorized 200K day trial. Primary
  served name MUST remain hotschmoe-dd and first in --served-model-name.
- Public authenticated port 18080 -> loopback backend 18124. Existing API key
  file is unchanged; do not print credentials or replace client configuration.
- Image: sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1.
  This is Steve R276 plus the GDN phase fix and MRV1 accepted-count ordering fix.
- TP2, P2P0, MTP3, FULL_DECODE_ONLY graphs, prefix cache, FP8 E4M3 KV,
  context 200000, c4, prefill batch 32768, GPU memory utilization 0.96.
- Fresh scale SHA256:
  be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062.
  Calibration covered prompts through 96K in a 100K configuration. The user
  explicitly chose 200K for the trial; long-context qualification is NOT done.
- Launcher: [day_trial/serve.py](../vllm/int4/day_trial/serve.py).
  Installed override: /etc/systemd/system/hotschmoe-dd.service.d/20260910-day-trial.conf.
- Current pointer: /mnt/vm_8tb/b70/run/hotschmoe-dd-mrv1fix-day-trial-current.
  Initial live result: /mnt/vm_8tb/b70/results/qwen38_mrv1fix_day_trial/20260910T072531Z.
- At 07:32 UTC: systemd active/running; strict per-card and compiled pair
  pre-health passed; all 34 scale receipts matched; tiny24 passed; public
  /v1/models advertised hotschmoe-dd first with max_model_len 200000;
  unauthenticated access returned 401; authenticated SSE arithmetic returned 42.
- No full qualification file or shelf promotion was created. The separate
  vllm/int4/service_candidate/ full-qualification launcher remains unused.

Do not stop the endpoint, reset cards, start competing GPU jobs, or update its
stack merely because a new session or overnight investigation begins. Both
GPUs are leased by the live service. CPU-only source/log work can proceed
without touching its inputs; new GPU downtime needs user direction.

At approved downtime, stop hotschmoe-dd.service normally and wait for verified
container teardown, strict post-health, and released leases. The root-owned
service override required user sudo assistance in this session. Keep the
working day-trial recipe available for restoration. Do not blindly restart
the old serve_fp8_trial.py recipe: it pins the old image/scales/validator.

## Findings to retain, not rediscover

1. GDN initialization: a fresh singleton prefill was misclassified as decode.
   Stock R276 bangs at TP1 on both cards; the phase-only fix removes that
   bounded defect. Official vLLM 0.29 still needed the phase guard.
2. A separate cached-MTP corruption survives the phase fix. FP16 prefix-on
   fails on both cards, prefix-off passes; MTP0 with cache on passes. The MRV1
   repair completes the prior D2H before row movement and removes a redundant
   permutation of already-current accepted counts. Its two parts belong together.
3. FP8 actual-reuse crossover: phase-only fails the same request on both cards;
   repaired image passes 32/32 on both cards with identical normalized choices.
   Every session has positive 1600-token hits. TP2 is not required for this bang.
4. The first FP8 tool fixture used 180 records and had zero relevant cache hits.
   FP16 resolves block 832, FP8 block 1600. MTP drops a final candidate block.
   Use 360 records (4274-token initial tool prompt) and assert actual hits;
   cache-enabled flags alone do not establish coverage.
5. The new Pi diagnostic garble is unresolved. Same reconstructed agent2 input
   was valid in four prior configurations, but the fixed/fresh-FP8 TP2 run
   emitted a new malformed tool suffix twice. Do not blame history alone.
6. SGLang's current-main FP8 attention also quantized Q and softmax P to FP8.
   The read-policy patch keeps FP8 cache storage but FP16 compute operands:
   upstream failed all 9 numeric checks; patched passed all 9. Both post-health
   checks passed. This is a numeric fixture, not full-model feature parity.

Detailed findings: [investigation](20260910_pi_bang_investigation.md),
[runtime/source ledger](../vllm/int4/20260910_runtime_freshness.md),
[cache review](../vllm/int4/20260910_cache_correctness_review.md), and
[Pi input audit](../vllm/int4/20260910_pi_reconstructed_input_audit.md).

## Exact pause state

All paths below are relative to this raw root unless absolute:

    /mnt/vm_8tb/b70/results/bang_isolation_20260910

mrv1-tp2-fp8-clean100k/run contains:

| Stage | State at pause |
|---|---|
| tiny24 | Passed, no repeat drift |
| earlier clean Pi history, 3 repeats | Passed inference and strict quality |
| deterministic rounds 1/2/3, seed42 | 96/96 strict checks; choices exact across rounds after generated-ID normalization |
| sampled round1, T0.7 seed42 | 32/32 strict checks |
| sampled round2, T0.7 seed43 | Drained after stop request: summary says 32/32, no bangs; strict review NOT run |
| sampled round3, T0.7 seed44 | NOT started |
| final startup-trace review job | NOT run in this arm |
| teardown | Server and parent lifecycle-rc 0; strict per-card/compiled pair post-health passed |
| whole-arm WORKLOADS_PASSED | ABSENT; qualification incomplete |

The conservative accepted count remains 128 strict concurrent checks, plus
32 completed but not strictly reviewed. Do not call it 160 qualified checks.
See pause-partial-snapshot.json and deterministic-choice-comparison.json beside
that run; the former was taken before the in-flight round finished draining.

The earlier mrv1-tp2-fp8-qualification/run is a SEPARATE FAILED quality arm:
all 9 Pi requests had no bang/error, but agent2 target repeats had malformed
second tool arguments. It stopped before its clean/concurrent stages. Its
startup trace independently passed review: each rank had 138 all-reduces on
FP16[32768,5120] and 3 all-gathers on FP16[32768,2560], with paired host returns.
These are not independent device-completion or graph-replay counts.

## Remaining vLLM work, in priority order

- [ ] Review the drained sampled-round2 output and the clean arm's startup
  trace using CPU-only analysis, writing NEW retrospective reports. Do not
  rewrite arm-results.json, .done files, or manufacture WORKLOADS_PASSED.
- [ ] Complete the missing sampled seed44 coverage and obtain a complete
  matched qualification on the final selected configuration. The existing
  run directory cannot be relaunched: runners intentionally refuse reuse.
  Prepare a new plan/output and retain why it supersedes the interrupted arm.
  Any decision to aggregate evidence across arms must be explicit and reviewed;
  existing launchers currently require a complete predecessor marker.
- [ ] Isolate the new agent2 malformed-output diagnostic with the prepared
  matched TP1 pair below. Same fresh scales, graph/MTP/cache settings, warm
  request and two target repeats; only image repair differs, with card as an
  initial confounder. Require observed block1600 and positive cache hits.
  If outcomes split, cross over cards. If TP1 does not reproduce, decide on a
  bounded matched TP2 comparison. Change scales, repair, or TP one at a time.
- [ ] Run the long-context FP8 numeric geometry and padded Mamba-copy oracles
  below. They test different mechanisms; neither is a full-model correctness test.
- [ ] Run the separate 200K qualification after the predecessor gate is truly
  satisfied: four independent ~185K histories, exact secret retrieval/reuse,
  long guides, cancellation/drain and recovery. Runtime token counts must match
  the exact-image CPU tokenizer. Keep scale calibration coverage distinct from
  successful extrapolation tests; do not relabel <=96K calibration as 200K.
- [ ] Before full promotion, review concurrent coherence, model/native/image
  identity, actual cache reuse, graph/MTP evidence, teardown and strict health.
  Any speed claim needs a matched uninstrumented configuration and workload.
  The 30-second internal shutdown grace improved observed cleanup; retain
  actual process/health evidence rather than assuming all shutdowns are solved.

Prepared launch commands, ONLY after approved downtime and prerequisite review:

    python3 vllm/cache800k/run_arm.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/pi-agent2-freshscale-paired/phase/plan.json
    python3 vllm/cache800k/run_arm.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/pi-agent2-freshscale-paired/phase-mrv1fix/plan.json

These two arms can run concurrently: phase card0/18151, repaired card1/18152.
Their server helper acquires the selected GPU lease itself; do not double-wrap
run_arm in a second lease. Each uses a fresh output/cache and strict lifecycle.

    bin/gpu-run --card 0 python3 vllm/int4/fp8_geometry_oracle/lifecycle.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/vllm-fp8-geometry-plans/q1/plan.json --run
    bin/gpu-run --card 0 python3 vllm/int4/fp8_geometry_oracle/lifecycle.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/vllm-fp8-geometry-plans/q4/plan.json --run

Run q1 and q4 sequentially on card0: length23240, block1600, permuted hybrid
layout, exact write and existing read-error thresholds. Prior passing vLLM
numeric tests used length67/block64 and do not replace these geometry tests.

    bin/gpu-run --card 1 python3 vllm/int4/mamba_copy_oracle/lifecycle.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/mamba-copy-lifecycle-plans/card1/plan.json --run

A card0 plan also exists. Twelve actual-copy cases cover padded/interleaved
state, accepted-count biases and untouched bytes. CPU metadata checks passed;
GPU copies have NOT run. It does not test native GDN arithmetic/publication.

    python3 vllm/int4/long_context_qualification/launch.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/mrv1-tp2-fp8-long200k --check-only
    python3 vllm/int4/long_context_qualification/launch.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/mrv1-tp2-fp8-long200k

The 200K prerequisite currently points to the interrupted clean100K run and
correctly BLOCKS launch. If a new predecessor is qualified, freeze a reviewed
replacement plan/manifest/prerequisite, preserving the previous versions.
The day-trial authorization is not a substitute for this full-qualification gate.

## Remaining SGLang and backend work

- [ ] Run the prepared current-main full-model FP8/cache step on image
  sha256:14ee7d0112b4e321ea7618a2b6c6f351b7428c2efe43deb3c2e51d128c7dd57d.
  Its strict pair preflight and 9-case numeric gate already passed. The new
  model step is TP1/card1, Triton, prefix extra_buffer, fresh be02 loader,
  eager/MTP0, text-only skip-server-warmup. Require target16-layer loader
  receipt, 26 text checks, 32 tool checks and measured positive cache reuse.

    python3 sglang/refresh/20260910_fp8_read_policy/feature_gate.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang14ee-fp8-cache-plan/plan.json --run

  This helper delegates to the existing leased lifecycle. It can run on card1
  alongside the sequential vLLM geometry oracles on card0. Do not also schedule
  the card1 Mamba oracle at the same time.
- [ ] Qualify graph and MTP independently, then combine with FP8/cache.
  Prepared FP16/prefix-off controls (not full parity) are documented in
  [FP16_FEATURE_PLANS.md](../sglang/refresh/20260910_fp8_read_policy/FP16_FEATURE_PLANS.md):
  graph-only18231; MTP1 eager18232; combined MTP1 graph18233; MTP3 graph18234.
  These are staged prerequisites, not evidence that all features work together.
- [ ] Resolve SGLang XPU non-greedy NEXTN verification: current source takes
  argmax even for temperature0.7. CPU-tested unseeded target-sampling patch
  and FP32 seeded-Gumbel prototype are NOT activated. Seed mapping must be
  reviewed/shared with ordinary sampling before claiming seeded parity.
  Sources: sglang/cache800k/nextn_sampling/ and seeded_gumbel/.
- [ ] Before SGLang TP2 serving, qualify the previously problematic large
  compiled collectives (including [2048,5120] FP16/subgroup and loaded-context
  behavior). Tiny 4x5120 health is insufficient evidence for that old failure.
  Do not launch arbitrary TP2 graph serves from a TP1 or numeric pass.
- [ ] Optional modern-vLLM branch: 0.29 uses MRV2 and avoids this MRV1 double
  permutation, but still needs the GDN phase fix. Its XPU graph source only
  advertises single-GPU support; the R276 scale hook rejects its different
  fingerprint. Deliberate source adaptation and full feature qualification
  remain required; an update alone is not a demonstrated replacement.
  See [MRV2 review](../vllm/int4/20260910_v029_mrv2_full_feature_review.md).
- [ ] At the next stack refresh, recheck Steve/neural.download, Sergio and
  official backend source/image identities. Existing findings are pinned;
  newer publications must not silently replace an arm's stack. Record host
  kernel/UMD/Level Zero/oneCCL/PyTorch/backend/image before changing one layer,
  rebuild ABI extensions from tracked source, and rerun required health.

## Pi/client follow-up and unfinished source

- [ ] Obtain exact failed HTTP request bodies, actual Pi system/tool/sampling
  transformations and LAN-installed guard/gateway source/version. The supplied
  bundle is not wire capture. Shared source archive:
  /mnt/storage/StrongSync/junk/agent-world-diagnostics/pi-corruption-20260910T022712Z.tar.gz.
  Original SSH attempt to hotschmoe@192.168.10.56 was denied publickey; do not
  assume access exists or send messages to other people without authorization.
- [ ] Keep input/output failures distinct. Reconstruction excludes the first
  bang/aborted response, but earlier completed turns contained 1509 consecutive
  zeros, two garbled responses, and a malformed 1582-line shell command.
  All 27 retained tool calls had matching results; no orphan calls were introduced.
  The new b:,.: the/python: suffix is absent from input and present in raw SSE.
  Prior configurations handled the same reconstructed request with valid JSON;
  history alone is not a proven cause of the new output failure.
- [ ] Investigate the separate cancellation/admission race: all 36 recorded
  bang recoveries immediately hit the harness's per-agent429 guard. Gateway
  source/exact wire data are missing. A detector or kernel patch does not fix
  that admission race by itself.
- [ ] Finish review/documentation of the UNCOMMITTED, UNINSTALLED guard draft:
  vllm/int4/guard_candidate/{extreme-repetition-optin.patch,extreme.test.ts,test_candidate.py}.
  Nineteen actual-source CPU tests passed before the pause. It preserves the
  existing default bang behavior and adds an opt-in threshold of1024 repeated
  non-whitespace characters. False-positive/chunk/escape/context-cleaning review
  remains; it cannot classify arbitrary short gibberish. No external repo or
  LAN installation occurred. Source pin: /mnt/vm_8tb/github/bang-guard,
  d341fe6c9a3b53061e9936b0cb78b6f08e6f6438. Do not lose or silently promote the draft.

## Resume discipline and evidence locations

Every real GPU touch uses bin/gpu-run, selected-card pin for TP1, both leases
for TP2. P2P0 throughout; no arbitrary CCL_TOPO_P2P_ACCESS=1 TP>1 serves.
Use the reviewed strict probe vllm/int4/diagnostics/xpu_health_strict.sh;
the historical shared bin/xpu-health sentinel can falsely accept a failure.
After a crashed TP2 attempt, verify owned cleanup and follow the reset ladder
before further serving; do not reset both cards under a single-card lease.
Kernel7.1 remains fixed. Do not restore quarantined native binaries.

Plans are frozen preparations, not a promise that current hashes still match.
Check prerequisites, source/image/config hashes, device allocation and fresh
output paths before launch. If source changes, create a reviewed replacement;
do not edit an old plan in place, weaken gates, or overwrite failed evidence.
Record each experiment as CONFIG -> COMMAND -> RESULT -> VERDICT and append
JOURNAL.md entries. Keep startup host instrumentation distinct from device
completion and uninstrumented speed claims. No external tool actions from
replayed model messages are executed.

Key source checkpoints are discoverable in git log; live deployment checkpoint
43bbc81 records the authenticated200K trial. The curated investigation links
all primary source/identity/numeric reports. Raw plans, responses, SSE, source
snapshots, scale receipts and health are under the raw root above. Extracted
original bundle evidence is under /mnt/vm_8tb/b70/results/bang_investigation_20260910.

The worktree contains user changes predating the investigation. JOURNAL.md mixes
those changes with appended campaign entries and has intentionally not been
staged. Preserve all unrelated dirty files; never git add all. Commit/push only
coherent owned checkpoints from this clone. The uncommitted guard draft is the
one known unfinished campaign source artifact; its explicit TODO is above.
