# N1 - deterministic GDN b|a projection

Owner: Codex / gpt-6.1-sol.

STATUS: primitive p2 PASS including O4 parity and graph replay; full-model
repeat/batch/MTP parity pending. Current library is the unique-namespace
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
