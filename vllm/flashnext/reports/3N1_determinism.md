# N1 - deterministic GDN b|a projection

Owner: Codex / gpt-6.1-sol.

STATUS: primitive p2 PASS; whole-model n1-original-r1 UNQUALIFIED.
Short C1 repeats pass, short C2 repeat and cross-batch exactness fail.
Later arithmetic stalled; EngineCore required force-kill, posthealth passed.
O5 128-token repeats and canonical suites were not collected. Current library is the unique-namespace
r2 SHA c90919594b0e2b355073e2b398ac76755a3ace5eb0094d014a72a05a411246d0.
 Scope: GDN projection only. No DD changes,
commits, pushes, submissions, or GPU work as of the preparation stage.

CONFIG -> COMMAND -> RESULT -> VERDICT

## Evidence and cause

CONFIG: O5 recipe c, eager probe p0g, same-process solo/dup/trip/other/solo2.
COMMAND: inspect reports/2O5_mtp.md, campaign JOURNAL last 40 stages,
o5/run-p0g/{probe.log,hashes-p0g.json}, run-p0h/window.log, and actual fork.
RESULT: O5 p0g reports first prefill mismatch at
language_model.model.layers.2.linear_attn.in_proj_ba, M=20. Its input hash
matches; in_proj_qkvz on that input matches. The first later decode
row-0 mismatch is linear_attn.norm at position 27. The proposed p0h
B70_ONEDNN_DET run aborted before model load (MemAvailable 23768912 KB,
minimum 60000000); no evidence establishes whether that switch works.
Fork gguf/linear.py::_dense routes BF16 M<=8 to O4 bf_gemv with KS=8
when N<256, else KS=2. M>8 goes to F.linear. B70_LINEAR_CAT concatenates
dense b then a. The same arithmetic is required over prefill rows.
VERDICT: replace only the GDN b|a path with fixed per-row reductions;
leave B70_ONEDNN_DET as an independent unverified diagnostic.

## Prepared implementation

CONFIG: new default-off B70_GDN_BA_DETERMINISTIC=1. Model integration
in fork vllm/models/qwen4_exp/xpu/model.py immediately after GDN creation.
Plugin lives in plugins/b70_n1_xpu; library stays outside the repository
at /mnt/vm_8tb/b70/build/vllm-flashnext/n1/_C.so (B70_N1_SO override).
COMMAND: B70_AGENT=N1 MAX_JOBS=4 bin/cpu-run bash
vllm/flashnext/plugins/b70_n1_xpu/build.sh; Python py_compile; bash -n.
RESULT: build exit 0 in 30 seconds, 117376-byte library SHA256
`dd893593b5a0ffc1a54bc1ad3dbd6e38032ae3aadd1f864f1540c680651576fa`.
Syntax checks pass. C++ derives from O4 launch_bf with the same lane
FMA order, subgroup reduction, ordered partial sum, BF16 cast, and KS
selection. Token tiles of eight are independent grid work-groups, so
prefill uses one launch, avoiding a Python chunk loop. The module hook
makes a private shallow copy of its quant method, preserving loader
class checks and methods, and replaces apply only on this projection.
It joins b|a once after device loading; unsupported activation dtype or
device delegates to the original method. Native dense weights also work.
Packed/non-b|a GGUF shards fail explicitly under the opt-in switch.
VERDICT: CPU-prepared candidate only; exact numerical equivalence,
capture, performance, model coherence, and determinism remain unmeasured.
Default-off branch is frozen while root baseline runs. No shared plugin
sources or existing recipes modified.

## Queued primitive gate

CONFIG: physical card1 pinned as logical xpu:0, <=120-second Python alarm,
150-second outer body, 8 GB host RAM limit, native health before/after.
Actual GGUF BF16 beta|alpha weights at layers 0/2/14; seeds 5/17.
COMMAND: TAG=p1 bash vllm/flashnext/harness/n1_primitive_window.sh
(run only after explicit parent slot authorization).
RESULT: pending. Checks M=1/2/3/4/8/9/20/40/60/128/257/12288, row parity
against O4 bf_gemv, exact repeats, shifted/reversed token tiles, and
changed-input captured replay at M=1/3/8/20/40/257. Every row is compared
at M<=60; six distributed rows at larger M. It reports kernel device
latency and FP32-reference error, plus oneDNN M20 repeat diagnostics.
VERDICT: request this bounded slot after D1. No timing or stability claim
until the complete gate and health evidence exist.

## Serve parity plan

1. After primitive PASS, first run eager same-load O5 p0g-style probe with
   B70_GDN_BA_DETERMINISTIC=1, identical remaining recipe, and hash the
   full GDN b|a request output. Check solo vs solo2 and solo/dup/trip
   prefill, then compare the first subsequent GDN divergence.
2. With FULL_DECODE_ONLY, run O5's four prompts x128 greedy tokens C1/C2
   twice in the same load (including hash-map and lighthouse near-ties).
   Demand exact token IDs, arithmetic 323/667, identity, exit, health.
3. Repeat in a fresh load with identical flags, then matched MTP off/on
   k=2/k=3 or dynamic k to separate remaining batch variance from MTP.
4. Check 8K/32K prefill/needle and C1/C2 latency with the same library,
   aliases and config. This kernel is initially a correctness path;
   its row-GEMV arithmetic can cost more than oneDNN at long prefill.

Parent schedules model loads and owns health/requalification and any
promotion. No exact-repeat or speed qualification is claimed yet.

## Baseline limitation and stronger isolation

CONFIG: root merged restart-r1 promotion gate (independent of N1, flag off).
COMMAND: root reports gate.json exact-repeat outcomes.
RESULT: C1 repeats all four prompts x2 pass over 96 IDs. C2 differs from
C1 persistently on prompts0/1 and inconsistently on prompt3. N1 has not
run and cannot be credited with resolving these effects.
VERDICT: separate same-load determinism from cross-batch invariance.
New n1_repeat.py reports C1-vs-C2 explicitly and gates repeated C1/C2.
New n1_invar_probe.py scopes O5's eager trace to BA/QKVZ/norm/router,
hashes full-request and full-batch inputs (O5 hin was only row0), and
saves first layer2 prefill BA x/w as ba-layer2-input.pt for a true
identical-input primitive reproduction. It preserves the hreq/row0
comparison and emits the first mismatch with both input-equality flags.
No GPU execution yet. The old first-row hin evidence alone does not
establish equality of the entire M20 input tensor.

## Prepared full-model window

CONFIG: reuse root-maintained f1_merged_serve.sh, O4 recipe + F1 prefill
unchanged, stable hotschmoe-dd first, maxlen131072. N1 wrapper adds only
its flag/library/plugin path and HTTP callback. Parent may add a D1
overlay only after independent D1/N1 gates and matched control.
COMMAND: TAG=n1-control N1_FLAG=0 bash harness/n1_serve_window.sh;
then TAG=n1-candidate N1_FLAG=1 N1_REFERENCE=<control>/n1-repeat.json
bash harness/n1_serve_window.sh (only after parent authorization).
RESULT: prepared/syntax-checked only. Output is f1/merged-$TAG as in the
shared lifecycle. n1_repeat_http.py uses all four O5 prompts x128 tokens
in C1/C2 x2, reports same-load C1 and C2 separately, strict cross-batch
comparisons, and optional reference parity. It saves complete token IDs.
The wrapper snapshots N1 library/model/plugin hashes and callback checks
those identities before teardown. Root lifecycle owns pre/post health,
identity, arithmetic, promotion gate, F1 prefill checks, bench, teardown.
VERDICT: ready for scheduled independent control/candidate after primitive
PASS. No fork/model edits since root baseline began.

## Primitive p1 - first dispatch rejected, health passes

CONFIG: root-authorized card1 window, original N1 build, O4 loaded for
exact arithmetic comparison in the same process.
COMMAND: TAG=p1 bash harness/n1_primitive_window.sh; raw n1/run-p1/.
RESULT: pre/post native health HEALTHY; body exit1 after34s, before first
numerical result. First torch.ops.b70_n1.bf_gemm call throws Level Zero
error30 UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX. No process crash,
reset or teardown hang; card1 released for Q1.
VERDICT: candidate unqualified, do not enable for a model load. A likely
cause is copied anonymous SYCL kernel class bf_gemv_k<KS>: O4 and N1
register that same name with different captured-argument layouts. Rename
the new class n1_bf_gemm_k before isolated rebuild/retry. This is a source
hypothesis until the retry; no numerical claim from p1.

## Primitive retry build r2

CONFIG: all N1 kernels/helpers/entry now b70_n1_impl namespace, class
n1_bf_gemm_k. Original p1 library preserved as n1/_C.p1.so.
COMMAND: cpu-run/MAX_JOBS4 build; compare ASCII kernel metadata strings
against loaded O4 binary.
RESULT: build26s exit0; current SHA256
`c90919594b0e2b355073e2b398ac76755a3ace5eb0094d014a72a05a411246d0`.
All18 N1 kernel metadata strings are unique vsO4. Original p1/O4 share
exact getKernelParamDesc/getDeviceKernelInfo template symbols.
VERDICT: real collision confirmed in original binaries, causal role still
requires GPU retry. Await root authorization after Q1 correctness window.

## Primitive p2 - independent kernel gate passes

CONFIG: same card1 pinned primitive, r2 unique-namespace library; O4
loaded simultaneously for exact comparison. Actual BF16 concatenated
GDN beta|alpha shape N=96,K=2560 at layers0/2/14, seeds5/17.
COMMAND: TAG=p2 bash harness/n1_primitive_window.sh;
raw /mnt/vm_8tb/b70/build/vllm-flashnext/n1/run-p2/.
RESULT: pre/post native health HEALTHY; containerexit0, normal teardown;
GPU body/runtimecontainer34s (includes environment/startup), measured
primitive script1.675s, posthealth16s. N1_PRIMITIVE_OK, windowexit0.
All72 shape cases pass: M1/2/3/4/8/9/20/40/60/128/257/12288, exact O4
per-row parity (all rows M<=60, six samples above), full-output repeat,
shifted token tiles, reversed rows. All48 changed-input graph replays
match eager output at M1/3/8/20/40/257. The original oneDNN M20 output
matches only83/100 identical-input repeats, reproducing nondeterminism
with no fullmodel/state involvement. New fixed-path repeat/graph tests
all pass. M20 maxabs vs FP32 is0.00388932 for both N1 and oneDNN.
Layer2/seed5 hot-loop event timing (20calls, not E2E): N1/oneDNN M20
25.02/73.01us; M257 55.11/67.82us; M12288 3848.21/140.85us.
The small-shape event loop includes host submission gaps, so do not use
it as a production speed claim. At M12288 the extra~3.71ms/projection
would add~0.133s across36GDN layers if serialized (inference only).
VERDICT: primitive correctness/capture gate PASS. Unique symbols resolve
the p1 firstdispatch failure. OneDNN nondeterminism is independently
reproduced. Fullmodel repeat/coherence/latency and MTP parity remain
unqualified; N1 remains off in D1's forthcoming matched control/plan16
pair. Model integration and N1 hook source frozen for that pair.

REPRODUCTION: harness/patches/n1_gdn_ba_deterministic.patch contains only
the seven-line default-off model hook, no inherited O4/X6 dirty changes.
Apply from fork root to preserved handoff source, add plugin directory
to PYTHONPATH, build via cpu-run/MAX4, then enable the flag only in the
coordinated candidate. No agent commits or shared plugin edits.

## Candidate scheduling review during D1 pair freeze

CONFIG: D1 matched control/plan16, N1 off in both. No source/build/GPU
changes from N1 during the pair. Parent grants later N1-enabled slot.
COMMAND: CPU-review n1_serve_window.sh and d1_combined_callback.sh.
RESULT: N1 wrapper unconditionally selects its repeat-only callback.
The existing D1 combined callback already runs N1 O5-repeat and both
canonical512-token suites, so direct reuse of f1_merged_serve.sh avoids
any source edit or duplicate lifecycle. Candidate should match D1 plan16
runtime flags and F1 probe sizes, adding N1 plugin path and
B70_GDN_BA_DETERMINISTIC=1 B70_N1_SO=<current r2 library>. Preserve
F1_API_WRAPPER=harness/d1_api.py and
F1_POST_PROBE_SCRIPT=harness/d1_combined_callback.sh, maxlen131072,
batched12288, decode64, prefill1024, F1_LENGTHS=8192,
F1_COHERENCE_LEN=8192, F1_REPEAT=2, F1_DECODE=0, promotion gate on.
VERDICT: root gets the exact prepared environment; wait for pair outcome
and authorization. Compare N1-repeat JSON offline to N1-off plan16.
Optional N1_REFERENCE marks aggregate failure if token IDs change from
the nondeterministic control, even if the candidate itself repeats, so
omit it for a candidate-only determinism gate and report that comparison
separately. Reordering a BF16 reduction can legitimately change output
bits versus the old nondeterministic prefill path.

## Read-only D1 control repeat audit

CONFIG: D1 fullmodel control-r1, N1 off, original FA plan. No N1 GPU work
or source edits during its active window. Evidence:
f1/merged-d1-control-r1/{gate.json,n1-repeat.json}.
COMMAND: parse completed repeat comparisons without changing artifacts.
RESULT: O5 four-prompt128-token same-load C1 repeat streams are
[false,false,true,true]; C2 are[false,false,true,false]. C1 repeated
Paris differs49 and hash-map differs2. C2 comparisons against first C1
show Paris/hash-map differences54/2 on round1 and49/43 on round2;
lighthouse differs65 on round2. Promotion short-prompt repeat also
fails, including persistent FibonacciC2 mismatch72 and a ParisC2
round1 mismatch2 that disappears in round2.
VERDICT: strong actual same-load nondeterminism baseline for the later
N1 candidate. Paris5tok alone uses <=8 dispatch while paired with the
15tok Fibonacci prompt produces M20: that also changes HC/MoE prefill
paths, so N1 alone cannot be assumed to fix every cross-batch mismatch.
The later plan16 control is the intended matched reference for N1.

## Whole-model n1-original-r1 - unqualified, stalled arithmetic

CONFIG: original c7b FlashAttention ONLY, no D1 attention overlay; O4
recipe + F1 prefill, N1 flag1, r2 library. Both cards, single-process EP
sidecar. maxlen131072, batched12288, maxseq4, FULL_DECODE_ONLY1/2/3/4/8,
no prefix cache, Mamba align, no speculation. Stable hotschmoe-dd first;
registered alias qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4. Parent reused
D1-r2 native both-card HEALTHY as prehealth (SKIP_HEALTH_BEFORE=1).
COMMAND: root-authorized f1_merged_serve.sh TAG=n1-original-r1;
F1_API_WRAPPER=d1_api.py; F1_POST_PROBE_SCRIPT=d1_combined_callback.sh;
F1_PYPATH_PRE=plugins/b70_n1_xpu only; O4_SERVE_EXTRA contains
B70_D1_FA_PLAN=0 B70_GDN_BA_DETERMINISTIC=1 B70_N1_SO=<r2>. Full command
configuration saved n1-env.json beside raw artifacts at
/mnt/vm_8tb/b70/build/vllm-flashnext/f1/merged-n1-original-r1/.
RESULT: started16:52:33UTC, READY16:55:34, identity and323/667 PASS.
8K/32K needle gates PASS, identity-after unchanged. Promotion96-token
same-load C1 repeat:[true,true,true,true]; C2:[false,true,true,true].
Cross-batch gate FAIL: C2Paris differs from C1 at2 on first round but
matches on second; Fibonacci differs72 both; hash collision prompt
92both. Strict batch determinism remains unqualified.
The combined O5 repeat callback reached identity only. Its first
17*19 request, started approximately16:56:47, never returned before
abort16:58:35 (>108s, normally subsecond). At16:57:02 engine metrics
show Running1 and0generation; no later server logs or engine exception.
At witness time EngineCore hostPID43458 was alive/Rl,181%CPU, API in
ep_poll. Saved stall-witness-time/processes/client-processes/engine-status/
engine-stat/serve-tail. No Python stack or GPU trace was obtained, so
neither an operation nor N1 itself is identified as the stall cause.
Parent authorized stopping body rather than waiting600s. TERM to only
the callback and waiting HTTP client returned CALLBACK_RC143, skipped
canonical suites. O5 128-token same-load C1/C2 repeats are UNAVAILABLE,
not pass/fail. Native posthealth is HEALTHY(cards0,1), rc0 in34s;
no reset, both cards released. Source/N1SO/model/originalFA/recipe/
harness hashes and shared native SO checks all unchanged.
TEARDOWN: SERVE_RC0 only describes API/container return. EngineCore did
NOT exit normally: processmanager SIGTERM16:58:36, force-kill of remaining
EngineCore pid107 at16:58:41. Overall windowrc1, gate3, callback143,
bench0, F1probe0. This is a failed liveness/teardown qualification.
VERDICT: keep N1 default-off and unqualified for serving. Diagnose
serving liveness before another full-model experiment. This one run
shows a stalled N1-enabled recipe, not a proven causal kernel defect.

Matched original-control diagnostic results (no speed promotion):

| item | D1 original control | N1 original candidate |
| --- | --- | --- |
| C1 ITL p50,64tokens |15.168827ms|15.165022ms|
| C2 ITL p50 per stream |19.582852/19.604523ms|19.564345/19.566278ms|
| Short C1 same-load repeat |4/4|4/4|
| Short C2 same-load repeat |2/4|3/4|
| Cross-batch exact |FAIL|FAIL|
| O5 long-prompt repeat C1/C2 |2/4 and1/4|unavailable after stall|
| Native posthealth |HEALTHY|HEALTHY|

Control/candidate effective recipe diff, saved n1-control-config-diff.json,
is ONLY B70_GDN_BA_DETERMINISTIC0->1 and added B70_N1_SO path. N1 plugin
path is the only Python-path prefix addition; original FA remains c7b.
Diagnostic F1 prefill8192: candidate8186/8187tokens, TTFT4.43268/3.57780s,
1846.74/2288.28tok/s; long arithmetic/coherence PASS. These do not
establish speed or stability because strict repeat/teardown failed.

## Read-only lineage and cached-weight lifetime

CONFIG: exact N1 whole-model recipe above, no B70_D3_GDN and no
B70_GDN_SPEC_DECODE flags. No additional GPU work or library loads.
COMMAND: inspect fork _xpu_ops.py:181/265, platforms/xpu.py:11-13,
c7b overlay symlinks, ELF DT_NEEDED, D3 source and X5 ragged generator.
RESULT: Flash-Next uses _gdn_attention_core_xpu_impl ->
torch.ops._xpu_C.gdn_attention. xpu platform imports installed
vllm_xpu_kernels._xpu_C.abi3.so via c7b symlink; its ELF requires
libgdn_attn_kernels_xe_2.so with RUNPATH $ORIGIN. Both resolve under
venv/lib/python3.12/site-packages/vllm_xpu_kernels. Runtime process maps
were not preserved, so this is configured import/ELF resolution evidence.
D3 custom plugin activates only with B70_D3_GDN=1 (sitecustomize.py).
Its shared gdn_decode.cpp:97 reduce_128 still returns parts[0] without
a read-complete barrier. X5 x5_gdn_ragged.py:14-21 adds that barrier to
its private R3 binary. Neither private R3 nor shared D3 is enabled here;
the proxy fix is therefore NOT active, and its specific helper is also
NOT the native implementation used by this serving recipe.
N1 _n1_ba_weight is a strong layer-owned Tensor reference, created once
with beta|alpha cat/contiguous after device loading, then never replaced
or freed by the plugin. Existing O4 _apply_dense_cat (gguf/linear.py:224)
uses the same lazy strong layer-owned cached concatenation. Both N1 and
O4 obtain per-call output Tensor allocation from their native op; the
new op uses O4's current-stream submission pattern. N1's cached weight
outlives graph captures/reinstantiations by the same ownership pattern.
VERDICT: no demonstrated N1 cached-weight lifetime defect. The R3 race
cannot be attributed to current native GDN from shared-source similarity.
Further localization needs liveness/queue/operation evidence, not a blind
barrier port or a new full-model run.

## Partial repeat-client reporting repair

CONFIG: client-only n1_repeat_http.py change after parent source-freeze
release and confirmation D1 stress excludes this client. No server,
model, native kernel, GPU run, build, or new tests file.
COMMAND: compile syntax; offline AST-extracted summarize_rounds against
recorded D1 plan16-r2 partial JSON, N1 identity-only aborted JSON, complete
D1 original-control JSON, and missing/malformed round collections.
RESULT: preserve original gate.check HTTP500 error exactly. Recorded
D1-r2 has complete C1 rounds and first C2 round, missing second C2;
summary marks C1 complete, C2 incomplete/failed, unavailable streams as
null, and overallFAIL without KeyError. Empty rounds/reference cannot
pass via an empty zip. References require all four rounds, four streams
per round,128 token IDs per stream. Complete control self-reference
matches while its original repeat failure remains failed. Invalid
reference JSON is recorded without replacing the primary gate error.
Syntax/diff checks PASS; offline result saved as
n1/client-partial-r2-offline.json, original evidence untouched.
VERDICT: partial failures now remain reviewable instead of being masked
by a later missing-round KeyError. No mathematical behavior changed.
The first offline assertion initially assumed both D1-r2 C1/C2 rounds
were absent; inspection showed both C1 rounds complete, so the validation
expectation was corrected to match that recorded fixture.
