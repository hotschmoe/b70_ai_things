# Qwen3.8 AutoRound INT4 neural.download replica campaign

User authorization: overnight replication, quality comparison against current
FP8 weights, conditional RAM offload with FP16 KV, and final hotschmoe-dd
restoration. Commit/push milestones; provide a local systemctl activation
command. No FP8 KV work or host kernel/driver changes.

## Frozen acquisition

CONFIG -> Recipe https://neural.download/models/qwen38-27b-autoround-int4-b70.html
retrieved 2026-09-08. Steve Seguin's source repository
https://github.com/steveseguin/b70-optimization-lab at
54aefaf067b422d507f51f1e362efeecc58668ba, cloned separately under
/mnt/vm_8tb/b70/steve-repro/qwen38-int4-neural-20260908.

COMMAND -> Fetch exact model revision bce40cacab0a4535b92fb3d57615c2bea9adf3d1
from devan-carlin/Qwen3.8-27B-int4-AutoRound into models/files/qwen3.8-27b/;
pull published R276 image from GHCR by digest:
521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad.

RESULT -> Acquisition in progress. Raw root:
/mnt/vm_8tb/b70/results/int4_replica_20260908.

VERDICT -> The live recipe supersedes the older AutoRound route discussed
in docs/20260905_qwen38_fp8_vs_int4_quality_review.md. Do not transfer either
its older rejection or the author's newer speed claim into a local verdict.

## Sequence and decision gates

1. Finish FP16 KV RAM-offload qualification on official FP8 weights. Keep
   the preserved serving configuration available as the fallback. Concurrent
   artifact downloads/builds make overlapping timings provisional; repeat
   promotion comparisons on a quiet host.
2. Verify all publisher model hashes/byte counts and reproduce the exact
   GPTQ relabel with the pinned script. The tensors stay AutoRound INT4;
   GPTQ here selects the fixed-K oneDNN kernel, not a new quantization.
3. Record native library hashes, torch/vLLM/UMD/Level Zero/oneCCL and image
   identities before GPU execution. Keep every native extension within the
   published image's matching stack. Preserve source patches/build scripts;
   no quarantine binaries or host ABI replacement.
4. Run per-card and compiled TP2 collective health, then the scoped exact
   R276 recipe. Its direct-P2P setting is confined to this documented replay
   with pre/post health and reset on failure, not a general serving default.
5. Reproduce the author's 12-prompt, six-class, 512-token strict suite on
   two fresh MTP0 and MTP4 lifecycles. Bind MTP verification to the same
   image's MTP0 output. Report the author's 99-interval token1-to-token100
   decode metric separately from total wall throughput and natural output.
6. Qualify C1/C4, long generations, tool histories, cancellation and actual
   long contexts on the intended daily-driver shape. The exact strict
   1024-context/cache-off recipe is not a 200K/cache-on qualification.
7. Compare all 164 sandboxed HumanEval+ problems with official FP8 weights,
   same harness, seed, thinking-off and 2048 output cap. Review new failures
   individually. More than two additional failures (1.22 percentage points)
   or any new critical arithmetic/tool/state corruption rejects promotion;
   also require default-thinking task checks. This is bounded local evidence,
   not proof of parity on every agent workload.
8. Only after the INT4 recipe passes, port the guarded RAM offload repair
   deliberately and rerun eviction/reload/coherence/performance/lifecycle
   gates. More GPU KV capacity requires more distinct histories to force
   eviction; equal input size alone does not create matched memory pressure.
9. Restore hotschmoe-dd on the best fully qualified recipe: INT4 if it
   meets speed and quality gates, otherwise original FP8 weights with FP16
   KV. Offload is independently conditional. Prepare the system service
   update and exact user command; do not leave the endpoint down at handoff.

Published reference, not a local measurement: R276 MTP4 strict pair
112.90/113.00 tok/s; R256 earlier pair about 112.3. The recipe credits fixed-K
W4A16 GEMM (R220/R221), FP16 row chunks (R224), grouped GDN (R228/R276),
size-independent Inductor reductions, graph capture and draft-only INT4
lm_head. Source details and all immutable inputs are in its pinned
publication-manifest.json. No local speed or quality result yet.

## Acquisition complete

CONFIG -> Frozen R276 GHCR digest and bce40cac AutoRound checkpoint.
COMMAND -> Download the model; verify every listed file with both O_DIRECT
and ordinary cached reads; run the author's R212 hard-link/config relabel
and verify its separate manifest. Fetch eight release assets and four
revision-pinned source archives; verify published hashes and byte counts.
RESULT -> Original and relabelled model identity checks passed. R212 has
99 dynamic exclusions and unchanged tensor payloads. All 12 release/source
artifacts downloaded (153604055 bytes). R276 installed _xpu_C is
271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f;
_xpu_ops.py is 6ee6b8db18759873246aca28e85ca6d2ba177eb08bfd3b9b0f0feea168cee9b3;
layernorm.py is 50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8.
All three match the recipe. The three core PyTorch native library hashes
match the current FP8 image. Runtime: torch 2.13.0+xpu, vLLM
0.27.2rc1.dev77+gac7509e2b.xpu, kernels 1e90ffa67, oneCCL 2022.0,
image UMD 26.27.39122.11-0 and Level Zero loader 1.32.0. The first inventory
attempt used an absent distribution name; the complete corrected inventory
is runtime-r276-v2.txt. No runtime was modified by inventory collection.
VERDICT -> Exact published artifacts are available locally. No local GPU
speed/quality claim yet. Acquisition identities are tracked in
neural_replica_lock.json. prepare_replica.py extracts the published
standalone invocation without executing shell text; strict config has no
model/runtime parameter overrides beyond local paths, port and served ID.
run_strict_replica.py owns the four fresh leased lifecycles and stops on
health, workload, canary or token-parity failure.

Lowest-priority follow-up: the user permits one bounded FP8 KV retest after
all FP16 offload, INT4 and serving-restoration work. The earlier package-path
mismatch makes a corrected retest useful, but it must not delay those tasks.

The full 164-task held-out data is frozen before model scoring at SHA-256
b52c70fb955ba9c1139174f56d58b0c2804558acf7513553d4edb08ab5df818f.
The coding client now verifies this hash and the exact sandbox image
c0522083adc7557aab54abff248a4a4a9f7c32b4d9d66ddf9af05200ae5a3339
before generation, and records the actual client/harness source hashes.
compare_code.py requires matched sampling/data/grader identity and reports
new failures and recovered tasks individually. Its score gate alone never
marks a serving configuration qualified.

Before INT4 GPU measurement, interpret the user's approximate 100 tok/s
target as at least 95 tok/s for the mean of the two strict class-balanced
decode medians (five percent tolerance). Report both runs and the natural
completion/wall metrics rather than selecting the better run. This speed
target does not waive token parity, coding quality, intended-context,
concurrency, teardown or post-health gates. A slower but healthy INT4 recipe
can be documented as a replica result without replacing the daily driver.

A single bounded torch profile is prepared for the end of a future INT4
qualification lifecycle, after timing and quality requests. It records real
prefill/collective tensor shapes and completed entry/return events on both
ranks; captured graph internals may remain opaque. Compare paired rank
counts and require clean teardown/post-health. Do not import the former
8192-row fence assumption, infer invisible graph collective counts, or use
profiled request timing as a speed result. Enable the profiler only once in
that lifecycle; the known repeated profiler start/stop path is avoided.

## Prepared daily startup (not activated)

CONFIG -> Pinned R276 image/source and the separately generated 200K/c4
prefix-on daily configuration, retaining FP16 KV. No CPU connector yet.
COMMAND -> serve_qwen38_autoround_r276_daily.py reuses the leased lifecycle
runner and existing API-key frontdoor at 18080/18124, alias hotschmoe-dd.
RESULT -> Python syntax checked. Startup requires an explicit successful
r276_daily_qualification.json, exact image/source pins and configuration
SHA; that qualification file has not been written. Stop requests use the
runner's STOP marker and wait for teardown health while the parent retains
the GPU lease. The existing systemd recipe remains unchanged.
VERDICT -> Prepared only. Live startup/stop and API smoke remain mandatory
before selecting this recipe. The strict four-lifecycle coordinator is
queued after the merged offload candidate's healthy exit; a health failure
prevents automatic continuation.

## Plain daily extension before cache experiments

CONFIG -> Add daily-prefix-off as a separate profile, retaining the author's
prefix-cache-off setting while increasing context to 200000, sequences to
four, batch tokens to 32768 and utilization to 0.96. Add daily tool/reasoning
parsers and xhigh thinking default. The previously prepared prefix-on profile
is preserved as a later experimental arm, not the first plain qualification.
COMMAND -> run_daily_replica.py requires strict PASSED plus the frozen >=95
mean speed gate. Reuse A0b's exact C1/C4, guides, 8000-record tool histories,
164 HumanEval+, cancellation, xhigh thinking and long recall prompts. One
150K history trace is sufficient for plain-cache-off correctness; the three
trace latency repetitions remain part of the separate offload comparison.
Add 199K recall and increase forced-growth diagnostic concurrency to four,
since two 132K prompts may fit in INT4's larger GPU pool. Capture actual
capacity/preemption metrics; do not call that diagnostic matched throughput.
Profile once after all timing and quality work, then teardown and post-health.
RESULT -> Config and coordinator prepared and syntax checked; no INT4 GPU
measurement yet. Coding score comparison and every newly failed task still
require review, regardless of the aggregate score. The coordinator stops
on a critical workload or coding-score failure, retaining all raw evidence.
VERDICT -> This ordering follows the request to validate the plain recipe
before applying cache/offload changes and the prior local MTP/prefix-cache
concern. The production launcher pins whichever daily profile is actually
qualified; no qualification manifest or systemd cutover has been created.


### 2026-09-08 - Clean FP16-cache reference and INT4 preflight correction

CONFIG -> A0clean: original FP8 weights/R187 native image, FP16 KV, MTP3,
200K/c4 shape, 64 GiB cgroup, prefix caching disabled; coding before long
stress. This is a clean quality control, not an FP8 KV experiment.
COMMAND -> C1/C4, all164 HumanEval+, three2048-token guides, clean teardown.
RESULT -> Every job and post-health passed, exit.rc=0. Coding157/164base,
152/164plus, no raw symbol loops; generation529.6s and grading53.0s.
Guides repeated byte-exactly in75.597s total. This control changes both
cache mode and preceding workload history, so it does not isolate the
baseline corruption's root cause. It supplies the coherent coding reference
for INT4: retain the frozen <=2 net additional failure limit and review
all newly failed tasks, with unconditional rejection of raw corruption.

CONFIG -> First INT4 strict preflight, exact published R276 image.
COMMAND -> Per-card health and compiled two-rank P2P0/P2P1 controls.
RESULT -> Original attempt strict/mtp0-a passed per-card/P2P0, then the
P2P1 shell guard refused to run because I_KNOW_P2P_WEDGES=1 was missing.
No model serving or P2P1 collective ran in that attempt. The generic error
path unnecessarily performed a non-reboot reset; recovery and post-health
passed. Preserve exit.rc=1 as an infrastructure failure, not hardware or
model evidence. The runner now supplies the opt-in only to the explicitly
requested scoped P2P1 preflight, after P2P0, and distinguishes probe rc2
from hardware failure. Two CPU lifecycle regressions pass. No bin/ changes.
The fresh strict-retry1/mtp0-a has now passed all three real preflights,
including P2P1, and is starting the model. Published serving settings and
12GiB/16GiB cgroup/swap limits are unchanged.
VERDICT -> Coherent reference accepted for bounded coding comparison.
INT4 model speed, parity, quality and serving qualification remain pending.

Preflight classification is deliberately conservative: rc=2 alone does not
prove that a GPU attempt never ran. Only the known opt-in refusal text with
no collective-probe start marker skips recovery. An inconclusive worker or
compiler failure after device setup still triggers the reset ladder. Three
CPU lifecycle regressions cover refusal, actual failure and inconclusive
device execution. This changes health handling only, not model/image flags.


### 2026-09-08 - R276 INT4 strict replica qualifies at 100.21 tok/s

CONFIG -> Published R276 image521eb277/source54aefaf0, AutoRound INT4
bce40cac tensors with R212 GPTQ routing, TP2, FP16 KV, prefix caching off,
1024 context/seq1/batch1024/util0.95, published 12GiB/16GiB cgroup/swap.
COMMAND -> run_strict_replica.py --out
/mnt/vm_8tb/b70/results/int4_replica_20260908/strict-retry1; two fresh MTP0
and two fresh MTP4 lifecycles, scoped per-card/P2P0/P2P1 preflight and
teardown/per-card/P2P0 post-health for each. Author 12-prompt/six-class
suite, temperature0/seed42, natural EOS policy with 512-token cap.
RESULT -> All four workload/canary/cache-identity and lifecycle gates pass.
MTP0 pair, each MTP4 against MTP0, and MTP4 pair all match 12/12 complete
token arrays. No cache hits. Class-balanced first99-interval rates:
MTP0 47.08406/47.09156; MTP4 100.53594/99.89262 tok/s, pair mean100.21428.
MTP4 full-output decode medians96.73944/96.45728 and wall-throughput
medians93.99733/94.15117 tok/s; median TTFT154.00/141.62ms. These metrics
are different aggregations, not interchangeable. All exit.rc=0; PASSED
written only after the final health and token comparison.
VERDICT -> Exact short-context replica meets frozen >=95tok/s target.
Not yet a daily-driver qualification. The separately pinned 200K/c4 plain
prefix-off profile has started, with clean A0clean coding reference,
FP16 KV and no CPU offload. FP8 KV remains deferred. Prepared authenticated
frontdoor smoke is syntax-checked only; no public endpoint promotion yet.
