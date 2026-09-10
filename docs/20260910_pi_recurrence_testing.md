# Patched Pi recurrence: authorized downtime tests

User authorized endpoint shutdown and testing on September 10, after arranging
the client pause. Preserve hotschmoe-dd as the first served name. This supersedes
the earlier read-only boundary for this testing window; it does not authorize
claiming qualification from HTTP success or installing unreviewed client guards.

## Shutdown

CONFIG -> Patched day trial 20260910T072531Z, image7b107d0e, TP2/P2P0,
MTP3/FULL/prefix/freshbe02 FP8, context200000.

COMMAND -> python3 vllm/int4/day_trial/serve.py stop. This is the existing owned
STOP-marker path; no sudo permission was needed to stop the user-owned runtime.
The parent retained both GPU leases through teardown and strict post-health.

RESULT -> Stop helper and parent exit0; systemd inactive/dead, MainPID0; model
container absent. Strict health passed both cards. Compiled two-rank collective
health passed ten iterations at shape4x5120 with P2P0. Kernel remains
7.1.0-070100-generic. Captured shutdown evidence is under:

    /mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/shutdown

VERDICT -> Endpoint is down for the authorized investigation. No reset or host
stack change was required. The live trial's quality failures remain preserved.

## Initial MTP comparison

CONFIG -> Both arms use image
sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1,
fresh scale SHA256
be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062,
TP1/P2P0, FP8 E4M3 KV, prefix on, FULL_DECODE_ONLY, context100000,
concurrency4 and prefill batch32768. Card0/port18151 uses MTP3;
card1/port18152 uses MTP0. Both preserve the primary hotschmoe-dd and register
their detailed research aliases in evals/configs/models.yaml.

COMMAND -> The frozen *.plan.json server argv launch kv_campaign_server.py
directly, which acquires the selected bin/gpu-run lease and pins that card.
This overlaps model startup with CPU client preparation. Frozen *.jobs.json
jobs are atomically queued into the owned server's jobs directory and execute
only after READY. The coordinator retains responsibility for STOP and final
health; empty-job run_arm plans must not be launched.

Raw root:

    /mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910

Plans: tp1-card0-mtp3-ctx100k.plan.json and
tp1-card1-mtp0-ctx100k.plan.json. Corresponding .launch.json, .jobs.json,
client-freeze.json and runtime-comparison-validation.json preserve exact
commands/source/payload/configuration identities. Model source/config inputs
were not changed while either process ran.

RESULT -> Startup resolved block1600 in both arms. Initial jobs are eleven
requests: reconstructed earliest agent1 warm request and two target repeats;
then exact integer-array tasks with caps64/256/1024/2048, each cold and reused.
The corpus is frozen at SHA256
7d4d112860c2db722db2423d981172a5c3e61f79dcbbd6410b2b392926579260.
Client collects all content outcomes and exits nonzero if quality gates fail.
Actual token usage/cache hits and SSE are retained. HTTP success is not a gate.

CPU client checks passed: exact-answer rejection, missing reuse, incomplete
output, and continued collection after an initial failure with failing final
status. Reconstructed Pi output still needs manual semantic review even when
structural checks pass. Its original system/tools/sampling/wire filtering are
unknown; these are explicitly controlled reconstructions, not exact replay.
The clean length tasks use thinking off, temperature0 and seed42. Requested
caps are not actual generated lengths; inspect usage and answer completeness.

VERDICT -> Initial inference results pending. Device differs with MTP, so
cross over cards if results split before attributing causality. These short
TP1 tests do not qualify the former200K TP2 service.

## Additional evidence and planned controls

The user's post-export report describes success declining with requested
output length. Its newest sessions are absent from the verified archive;
retain it as a hypothesis, not measured backend correctness evidence.
The citizen-authored observatory was downloaded without executing scripts.
One cross-check contradicts its claimed agent10 birth-only bang: the frozen
session has three completed assistant tool turns before the first bang at
line16/id17b41973. Avoid deriving causal triggers solely from citizen prose.
See raw observatory-crosscheck.json for the exact scope.

Cancellation followed by drained cached continuation is being prepared as a
separate controlled job. Actual-source CPU review found no additional proven
Python bug: prior accepted-count fixes and explicit padded graph-tail setup
remain present. Immediate block-free behavior relies on execution ordering;
that is a lifecycle test target, not a demonstrated defect. Native GDN
arithmetic with padded state strides and partial acceptance remains a separate
numerical-oracle target from the already prepared Mamba-copy checks.

No SGLang feature-parity conclusion follows from these tests. Retain the
[earlier campaign](20260910_bang_campaign_resume.md) and
[recurrence evidence](20260910_pi_bang_recurrence.md) for backend alternatives
and remaining controls. Append measured outcomes below without rewriting
failed or incomplete raw experiments.

## First direct-server failure, initial budget-limited corpus

CONFIG -> Initial pair above, direct loopback API client (Pi and agent-world
gateway bypassed), clean array tasks, thinking off, temperature0/seed42.

COMMAND -> Collect all eleven initial requests; inspect raw SSE and content,
finish reasons and actual usage rather than treating failed gates uniformly.

RESULT -> Both arms' three reconstructed Pi requests have no bangs or malformed
tool JSON. Manual review finds broadly coherent introductions, with ordinary
unsupported details in one MTP3 response; no exact replay or full semantic
qualification is claimed. Actual target prompt7855 differs from original9544;
target hits4800 with MTP3 versus6400 with MTP0. Same-arm target repeats differ
in wording despite temperature0, so deterministic identity is not established.

The clean request for integers1..256 produced a more useful failure: MTP3
stopped after769 output tokens, with the correct sequence through174 followed
by the malformed tail "174, 1,1ic ". Cold and reused responses match exactly.
MTP0 continued correctly past174 to a partial227, then exhausted its1024-token
budget. This is directly generated malformed output without Pi involvement.

The longer1..512 case also exposes an inadequate initial test budget: the
2048-token cap cannot contain the observed output formatting (MTP3 reaches a
partial432). Preserve these length failures as budget-limited checks, not
proof of degeneration. A new frozen corpus with identical prompts and ample
2048/4096 output caps is required before interpreting the complete comparison.
Do not edit the initial corpus, responses or failure statuses in place.

VERDICT -> A direct MTP-enabled garble has been reproduced, but device and
speculative behavior are still confounded. Repeat with adequate output budget,
then cross over cards if the split persists. Pi/gateway updates may address
their own errors, but cannot by themselves explain this direct-server failure.

## Ample budgets and moving-boundary reproduction

CONFIG -> Same running pair, same array256/512 prompts; new corpus-v2-length
increases only output caps to2048/4096. SHA256
05c08fae544111f8c64bb7dce77591c2f29709621be66028065633a503a59992.
Then corpus-boundary-v1, SHA256
16417345be341caae08f6efddbac3268c68f6c0e61ca212ff96130dc9191756d,
adds exactly0/128/256/512 neutral input tokens to array256, cold and reused.

COMMAND -> Queue separate immutable jobs02 and03 under each leased server.
Use the exact image's CPU tokenizer to verify prompts and re-encode the clean
output prefix before its first divergence/bang. Retain all raw SSE and errors.

RESULT -> With ample budgets, MTP3 array256 still stops at769 tokens on both
requests, now with whitespace after "174, 1". MTP0 completes both correctly
in1173 tokens. MTP3 cold array512 completes correctly in2453 tokens, but its
reused request bangs after the exact correct prefix through partial495.
MTP0 completes both array512 requests correctly. Thus crossing a boundary
does not fail universally; acceptance/state history can matter.

| Prompt tokens | Tokens before next block | MTP3 cold/reused | MTP0 cold/reused | First bang absolute position |
| ---: | ---: | --- | --- | ---: |
| 4035 | 765 | Both fail | Both exact pass | 4800 |
| 4163 | 637 | Both fail | Both exact pass | 4801 |
| 4291 | 509 | Both fail | Both exact pass | 4801 |
| 4547 | 253 | Both fail | Both exact pass | 4801 |

All eight boundary MTP3 requests bang; all eight MTP0 requests pass. MTP0
cold requests report zero cache hits, reused requests3200. Padding changes
output formatting, but the first bang remains at4800/4801. Positions are
derived from exact-tokenizer clean prefixes, not SSE chunk counts. Bang runs
can merge under retokenization; total decoded length is not claimed to equal
the unavailable runtime token count on aborted responses. Raw SSE matches
the parser's partial result without duplicated chunks.

The initial MTP3 array256 first wrong token also lands exactly at4800.
Original Pi examples are consistent with a nearby boundary, but removed
thinking/tool delimiters prevent an exact original generation offset claim.
See token-boundary-audit/REPORT.md and BOUNDARY_REVIEW.md in the raw root.

VERDICT -> Strong, repeatable localization to MTP-enabled cache-boundary
handling on this configuration. It is not Pi-specific or inherently TP2.
The failing arm remains card0 versus passing card1 until crossover completes.
Both initial model arms were stopped normally, exited0, and passed strict
selected-card post-health. Next: MTP3/prefixON on card1 and MTP3/prefixOFF on
card0, with focused identical array256 requests and independent zero-hit gate.

## Numeric copy probe: first attempt blocked before kernel execution

CONFIG -> Existing twelve-case actual PREcopy oracle, repaired7b, leasedcard0,
after the first MTP3 arm's verified teardown; card1 controls continued.

COMMAND -> bin/gpu-run --card 0 python3 vllm/int4/mamba_copy_oracle/lifecycle.py
with mamba-copy-lifecycle-plans/card0/plan.json and --run.

RESULT -> Strict pre/post health both pass and owned container cleanup is
verified, but actual initializer raises an unsigned-pointer-to-int64 overflow
before any copy kernel. Original raw result is preserved under
bang_isolation_20260910/mamba-copy-actual-card0. The standalone oracle omitted
the serving allocator setting PYTORCH_ALLOC_CONF=expandable_segments:True.
A new adapter matches that setting and records pointer ranges before the
unchanged initializer. It does not cast pointers or patch production source.

VERDICT -> No copy-kernel numerical verdict from this attempt. The allocator
mismatch is a testable infrastructure explanation, not yet verified. The old
oracle only covers PREcopy, so a separate actual POSTcopy oracle is being
prepared for positive-bias self-copy and backward boundary publication.


## Crossover and native state contract localization

CONFIG -> Same repaired 7b image, TP1, MTP3, FP8 KV, FULL graph. Compare
prefixOFF on card0 against prefixON on card1; keep the earlier independent
MTP0 results. Numerical probes use the exact installed native library,
SHA256 271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f.

COMMAND -> Run the focused array256 cold/repeated pair on both controls;
then the eight-request padding boundary matrix on card1. Run actual PREcopy
with the production allocator, actual POSTcopy, and native publication/read
probes through their owned, leased lifecycles.

RESULT -> PrefixOFF card0 passes both focused requests with zero cache hits.
PrefixON card1 initially passes both focused requests, then fails four of
its eight boundary requests (padding0 and256, cold/reused); padding128 and512
pass. All model arms stop normally and pass strict selected-card post-health.
The defect is therefore not exclusive to card0 or TP2, and its occurrence
depends on state history as well as boundary crossing.

With the serving allocator, all twelve actual PREcopy cases pass byte-exactly.
All twelve actual POSTcopy cases also pass, including positive-bias self-copy
and accepted-count reset. These kernels correctly implement their copy rules.

The actual native GDN producer implements different rules: it publishes a
three-row convolution history in each speculative column and reads column
accepted-1. The worker expects a six-row rolling history in column0 and copies
starting at row accepted-1. For accepted counts2-4, a boundary copy can therefore
combine correct temporal state with stale convolution history. All four native
prefix checkpoints match their independent numerical reference exactly; their
trailing three rows remain untouched. Subsequent native calls match the
per-prefix reference and diverge strongly from the worker rolling reference.
All numerical probe teardown and strict health checks pass. The original
rolling-contract oracle fails as expected and remains preserved unchanged.

VERDICT -> Confirmed native/worker state representation mismatch, with a
concrete corruption mechanism matching direct model failures at cache block
boundaries. This is a third issue beyond the earlier phase initialization and
accepted-count reorder patches. Updating Pi alone cannot repair this backend
defect. It does not establish the cause of every historical client error.
See [the numerical evidence](../vllm/int4/native_gdn_oracle/RESULT.md)
for complete measured evidence. Upstream native commit
5802a414d47855b01b63121bce3655795ef8dfa8 repairs this representation; an exact-stack
source rebuild and a narrowly gated Python copy adapter are under test.
Neither candidate is qualified for serving yet.


## First matched convolution adapter model comparison

CONFIG -> Candidate image09fc6b75041566c9248d76b985f2a30a4d32e5a847389f637d2a6043436f3bce
changes only the two approved Python copy-selection/metadata files from7b.
All2106 native files and282 packages are unchanged. Same candidate image on
both cards, TP1/MTP3/FP8/FULL/prefixON/100K; explicit adapter0 on card0 and
adapter1 on card1. Stable served name hotschmoe-dd remains first, with the
registered detailed research alias second. Corpus and client are hash-frozen.

COMMAND -> Strict both-card health and compiled pair P2P0, ten iterations at
4x5120, pass before serving. Run focused array256 cold/reused followed by the
eight-request boundary matrix on each card. Query live model identity in each
job. Compare exact generated arrays and actual cache reuse, not HTTP status.

RESULT -> AdapterOFF passes6/10 and corrupts padding0 and256 cold/reused;
adapterON passes10/10 exactly. Both focused requests pass even on OFF, showing
why short isolated successes are insufficient. OFF then stops normally with
lifecycle0 and strict selected-card post-health0. ON remains running for
longer output, concurrent, cancellation/recovery and reconstructed Pi tests.
The actual composed native/PRE/POST numerical oracle is running separately.

VERDICT -> Initial causal model evidence supports the contract correction.
This is not final stability, TP2, 200K, or speed qualification. Card assignment
is explicit; the earlier prefixON crossover already reproduced the defect on
card1. The rebuilt upstream native correction remains a separate candidate.
Raw plans/results are under bang_recurrence_testing_20260910, including
conv-adapter-first-comparison.json and tp1-card*-mtp3-conv-adapter*-ctx100k/.


## Adapter numerical composition and extended controls

CONFIG -> Same09fc adapterON image, unchanged FP8 scales and model settings.
Card0 runs the actual native -> POSTcopy -> PREcopy -> native composition;
card1 retains the running model for long output, cancellation and concurrency.

COMMAND -> Execute the frozen composed-boundary-plan-card0 through its selected
lease lifecycle. Run four ample-budget array cases, five cancellation/recovery
steps, sixteen four-client requests, and three existing Pi reconstructions.
The first concurrency launcher fails argument parsing before requests because
--alias is unsupported; preserve rc2 and use a separate identity wrapper for
the unchanged client in the replacement job03b.

RESULT -> All ten composed cases pass. Synchronized and unsynchronized results
are exact, as are the no-move native controls. Explicit metadata selects the
new copy function, conv widths[0,0], inner sizes[30720,786432], and hybrid page
strides3276800. Second-output max absolute reference error is4.76837e-7;
maximum checked state error is4.53508e-6. Oracle, strict pre/post health and
owned cleanup all pass. This validates composition, not full-model graph
execution or concurrent serving.

The model passes all four ample-budget requests and all five cancellation,
idle-drain, restart and follow-up steps. The three reconstructed Pi requests
pass the existing checks; these are not exact captured wire replays or full
semantic qualification.

Concurrency does not qualify: only8/16 pass the combined content/cache gate.
Two session0 thinkingOFF responses stop early despite max_tokens4096, with
2058 and2055 output tokens, at complete433 plus comma and partial433 respectively.
Raw SSE joins match the parser and finish_reason is stop. Reuse has cached0
on7/8 requests, a separate gate failure that masks one of the two content
failures. Cache pressure reached75-80%; misses must not automatically be
called kernel corruption. Independent content review and a same-payload
serial control are pending. No bangs were observed in these two truncations.

VERDICT -> The adapter fixes the reproduced boundary cases and passes composed
numerics, but the extended model qualification remains incomplete/failed.
Do not promote it. Continue a same-card ON crossover and the source-rebuilt
native correction, with serial/concurrent comparisons for the new truncation.


CONFIG -> Exact failing concurrent session0 payload, cap4096, now alone on
unchanged adapterON card1. Fresh serial cache namespace; content and cache
checks are independent.
COMMAND -> Run frozen card1-serial-session0.job.json after all prior jobs.
RESULT -> Both serial requests also stop early: cold2055 output tokens and
reused2058, ending around433 as before. Cold cached0 and reused cached1600;
both cache gates pass, both content gates fail. Final concurrent audit is
14/16 content passes,9/16 cache passes,8/16 combined passes. Corrected card1
then stops normally with lifecycle0; strict post-health passes.
VERDICT -> Concurrency and cache misses are not necessary for this remaining
early-stop failure. Compare the same payload under MTP0 and rebuilt native
before attributing it to another kernel bug or ordinary model behavior.


CONFIG -> Same physicalcard0 crossover, image09fc, adapterON. Config differs
from prior card0 OFF only by B70_XPU_GDN_PREFIX_CONV_COPY=1; paths/port/name
are fresh, while source, corpus, scales and serving settings match.
COMMAND -> Execute the same ten focused/boundary cases, then normal STOP and
strict selected-card post-health.
RESULT -> All10 responses pass exact content and required cache checks, versus
6/10 under OFF on the same card. Lifecycle0 and post-health0; output is
same-card tp1-card0-mtp3-conv-adapter1-crossover-ctx100k.
VERDICT -> Same-card evidence confirms the adapter corrects the reproduced
boundary cases. Remaining serial early stops are a separate unresolved
qualification finding; MTP0 and source-rebuilt native comparisons are next.


CONFIG -> Original repaired7b, same physicalcard1, MTP0/FP8/FULL/prefixON100K,
exact serial session0 prompt and4096 output cap. Adapter is absent.
COMMAND -> Run tp1-card1-mtp0-serial-session0-ctx100k's two frozen requests,
then normal STOP and strict selected-card post-health.
RESULT -> Both requests stop prematurely at2055 output tokens, with the same
partial433 prefix. Cold cached0 and reused cached3200; content0/2 and cache2/2.
Lifecycle0 and strict post-health0. No bangs observed in this counterexample.
VERDICT -> Speculative decoding is also not necessary for this remaining
premature-stop behavior. The native GDN contract correction remains supported
by distinct numerical and same-card boundary evidence. An explicit65536-context
FP16 KV diagnostic is prepared because100K FP16 capacity is not established;
if it passes, a matched65536 FP8 comparison is required before attributing the
difference to KV precision.


## Source-rebuilt native correction: numerical qualification

CONFIG -> Native-only image d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067
contains new_xpu_C SHA6717e3ec7fe3e7cdc26cd66b84aabdd076f211e5bed62b8711b7377fca8668d9.
All other2105 native files, packages and2227 vLLM Python files match7b.
Upstream5802 was ported to pinned1e90 plus original Steve patches and rebuilt
with exact Intel2026.1.1. Source/recipe/ABI receipts are tracked under
vllm/int4/native_gdn_oracle/native5802_build/. The Python adapter stays absent.

COMMAND -> Run strict both-card and compiled pair P2P0 preflight. Deliberately
repin only image/native identity in the original rolling-history oracle;
AST equality preserves every numerical and state-layout expectation. Run
128steps/all12gates on card0 TP1 geometry, then card1 local TP2 geometry.

RESULT -> Candidate pair health passes. Both numerical runs complete128steps
and all12gates at each step; strict pre/post health and owned cleanup pass.
The original deployed native failed this same rolling contract at step0.
Card1 uses localK8/V24, hybrid1638400-byte and packed1634304-byte pages; this
is not a two-rank serve or collective performance result.

The first full-model launcher accidentally acquired an already-owned lease.
It was stopped by terminating only its verified flock waiter before any GPU
work or output directory creation; rc124 and evidence are retained. V2 checks
the inherited selected FD and passes --leased once. Its CPU test exercises
the actual server lease branch. V2 is now serving the prepared native-only
boundary10/serial2/ample4 jobs. No model qualification is claimed yet.

VERDICT -> The source repair passes the unchanged failing native contract and
math oracle in both local geometries. Full-model/graph, actual TP2 serving,
concurrency, long-context and teardown qualification remain separate.

CONFIG -> FP16KV/MTP0/FULL at65536 context, exact same early-stop serial pair.
COMMAND -> Execute the two frozen requests, normal STOP, strict post-health.
RESULT -> Both stop at2058 tokens after433 plus comma, with cached0/3328;
content0/2, cache2/2. Lifecycle0 and post-health0.
VERDICT -> FP8 is not necessary for this same premature-stop pattern. The
context-cap change is explicit; this is not a matched precision performance
comparison. A same65536 eager FP16 control is now starting to test graph
necessity. No model/quantization cause has been established.
