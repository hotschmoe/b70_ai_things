# 3P1 -- recovered card-1 prefill budget and prepared candidates

Owner: Sol 6.1 (gpt-6.1-sol), 2026-10-21. CPU-only preparation under
root's matched decode source freeze; no build, GPU access, shared source
edit, service lifecycle, commit or submission. Existing dirty WIP retained.

## Recovered evidence

CONFIG: F1 q2 merged O4+F1 sidecar, card1 E374, real request T12283,
H2560/F640/topk10, native tile32/maxpad32768. StageProf synchronizes
between each operation, so this is attribution, not clean E2E timing.

COMMAND: parse `f1/merged-q2/serve.log` BUCKET_PROF rows for T12283/E374;
read existing `restart-20261021/f1-q2-recovered-summary.json`; inspect
`plugins/b70_gguf_xpu/csrc/{bucket_moe,dequant_wide}.cpp`,
`plugins/b70_ep_xpu/b70_ep_xpu/prefill_ep.py`, F1/restart reports, and
GGUF metadata (mmap only). Build paths below share the prefix
`/mnt/vm_8tb/b70/build/vllm-flashnext/`.

RESULT: all 48 real layers recovered; synthetic T12288 excluded.

| Stage | median ms/layer | mean ms/layer |
| --- | ---: | ---: |
| gather packed experts | 4.415 | 4.534 |
| dequant | 13.350 | 12.026 |
| pad | 2.350 | 2.359 |
| bmm1 | 9.430 | 9.922 |
| swiglu | 1.615 | 1.616 |
| bmm2 | 5.405 | 6.426 |
| combine | 9.800 | 9.675 |

Mean total stage sum47.271ms. Per-layer values are in `p1/q2-real-rows.json`.
Chunks11..20, median16, mean15.625. Padded rows89120..150992,
median127416. Combine correlates with chunk count0.980 (Pearson), versus
0.444 with padded rows; dequant correlates0.748 with chunks. Correlation
is descriptive, not causation. Raw layer rows and GGUF type/shape metadata
are `p1/q2-real-rows.json` and `p1/expert-metadata.json`.

Gate/up types16/17/18/21/22 are mixed across layers; down is20 or42.
The profile's chronological 48-layer order was joined to GGUF metadata.
For example type21/down20 layers have mean dequant14.512ms versus
8.360ms for type17/down20 layers; routes and chunk counts differ too.
Do not attribute all layer variance to the quant decoder.

VERDICT: no q2 rerun. The native op's critical card1 work remains exposed
in card0 wait. Rank (1) remove packed gather, (2) reduce repeated combine
traversal, (3) reshape/fuse dequant+bmm only after the first two are measured.

## Source analysis and prepared default-off probes

CONFIG: scratch files only under `p1/`; original hashes are recorded in
`p1/source-manifest.json`. Each candidate flag defaults off, and a
namespace-unique `b70_p1` registration permits loading alongside the F1
control. All new SYCL kernel names live in unique C++ namespaces to avoid
N1's observed loaded-library metadata collision.

COMMAND: derive scratch C13 selected-source addressing from the hashed
control, prepare isolated registration and primitive, run Python syntax
checks and `bash -n` on build/window.
No compiler invocation or GPU import was performed by P1.

RESULT: prepared `p1/p1_dequant.cpp`, `p1/p1_selected_register.cpp`, header,
`p1/build.sh`, `p1/probe.py` and `p1/window.sh`. Root narrowed the next
slot to selected-dequant only. The build compiles the selected registration
and C13 source, excluding the scratch bucket/mask follow-on below.

1. `B70_P1_SELECTED_DQ=1`: directly address original packed expert `sel[e]`
   inside the existing C13 kernel. This removes three `index_select`
   copies per chunk (`bucket_moe.cpp:268`) and their intermediate packs.
   Destination gate/up interleave, element decoder, BF16 conversion,
   bmm inputs/shapes, bucket plan, and arithmetic stay identical.
   Generic fallback also maps source rows. Selection indices come from
   the existing host-validated bucket plan. A direct raw-op probe uses
   five real gate/up quant types plus both real down types, reordered
   selection (32 noncontiguous experts), and destination modes0/1/2; exact bit comparisons include
   untouched NaN sentinel halves. Address selection adds integer work,
   so the observed4.534ms gather mean is an opportunity, not a speed claim.

2. `B70_P1_COMBINE_MASK=1`: compute one uint32 bitmask per token/chunk
   from the existing inverse indices. Combine skips empty tokens and
   reads inverse entries only for selected slots, instead of testing all
   ten inverse entries for every hidden column (`bucket_moe.cpp:167`).
   It retains the existing chunk order, ascending-k order, FP32 multiply
   plus add expression, and output += operation. No new atomics or global
   accumulation reorder. Existing histogram/scatter atomics are unchanged.
   Probe compares four sequential chunk updates bitwise at slot densities
   0/.03/.10/.50/1 and real T12283/H2560/topk10. The additional mask launch
   can lose on dense chunks; only measured results decide adoption.

The optional full synthetic layer uses real layer8/card1 traffic-snake
packed weights and real H/F/T/tile/maxpad, with explicitly synthetic skewed
routing. Four arms (control, selected-DQ, mask, both) must match exactly.
It does not reproduce the real q2 route distribution or prove serve speed.
No actual request routing dump exists in the recovered artifacts.

VERDICT: candidates are prepared and unbuilt. The next build/window runs
only the selected-DQ primitive: all quant types and modes exact, plus
32-expert selected-DQ versus gather+dequant timings at real shapes.
The bucket/mask follow-on is scratch only and excluded from this test.
An isolated build and component gate are required before any serving hook.
Do not edit the shared source during root/D1's frozen-source matched pair.

## Memory floor correction (inference)

For all374 active experts, dequant emits
`3 * 374 * 640 * 2560 * 2 = 3,676,569,600` BF16 bytes/layer, before
bmm reads those weights again. Packed input adds roughly0.49..0.87GB
according to layer types. Thus the compulsory weight traffic of the
current dequant+bmm design is about7.84..8.22GB/layer, excluding activation,
padding, GEMM output, and gather traffic. At a hypothetical sustained
280GB/s this alone is about28..29ms, and even608GB/s gives about13..14ms.
These are arithmetic bandwidth bounds, not measured utilization; active
expert counts were not logged and may be below374. They show why the old
10..15ms total floor is too optimistic at the earlier empirical bandwidth.
After removing copies, larger gains probably require eliminating a BF16
materialization/read pass, overlapping independent work, or changing layout.
That is a larger kernel task than pointer selection and must retain exact
reference checks. Fusing dequant into DPAS must be compared end to end;
F1 already showed the current scratch-free fused path loses at real rows.

## Requested coordinated test and setup cost

CONFIG: physical card1 only, logical xpu0; root must grant a fresh slot.
No full Flash-Next model load, no PLE pin, no TP, no server lifecycle.

COMMAND (prepared, NOT executed):

```
B70_AGENT=P1 MAX_JOBS=4 bin/cpu-run bash /mnt/vm_8tb/b70/build/vllm-flashnext/p1/build.sh
TAG=component-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/p1/window.sh
```

RESULT: expected build1..3min CPU, max compiler parallel links4, no GPU.
Expected selected-only card1 body1..2min, script alarm180s and outer timeout210s;
plus native per-card pre/post health (roughly2..4min total), 12GiB host
container cap, less than4GiB expected device allocation. The window takes
cpu-run --quiet outside xpu-run's card lease and records SO/source hashes,
exact dequant components, primitive timings, teardown status, post-health.
Its selected-only `p1/probe.py` has no bmm, combine or full synthetic
layer calls. Broad follow-on preparation is explicitly named
`prepare_bucket_followup.py` / `probe_bucket_followup.py` and is unbuilt.
Full-model qualification is deliberately a later decision because it costs
90..120s and26.8GiB PLE loading, plus model/setup RAM.

VERDICT: await root authorization before build or card1 use. No speed,
stability, exact serving-repeat, or promotion claim. Root's prior Fable
and Opus consultations timed out without a responding model ID; P1
attributes no consultation advice to them.


## 2026-10-21 16:45 -- isolated selected-DQ build complete

CONFIG: root authorized CPU build only while D1 plan16-r2 model loads.
Scratch selected registration only; no bucket/mask translation unit,
no shared campaign source/harness/plugin edits, no GPU access.
`MAX_JOBS=4`, compiler parallel links4, on-disk P1 TMPDIR.

COMMAND: `mkdir -p /mnt/vm_8tb/b70/build/vllm-flashnext/p1/tmp`, then
`B70_AGENT=P1 MAX_JOBS=4 TMPDIR=/mnt/vm_8tb/b70/build/vllm-flashnext/p1/tmp
bin/cpu-run bash /mnt/vm_8tb/b70/build/vllm-flashnext/p1/build.sh`.
Inspect static kernel metadata via `strings -a`, templated
getKernelParamDesc/getDeviceKernelInfo names, plus `nm -D`.

RESULT: build38s exit0, P1_SELECTED_BUILD_OK. Binary
`p1/_C.p1selected.so`,965480bytes, SHA256
`0ebb5837a9467e5186566243684becc27cd5047bb0064e079763788b6dfcdbbd`.
Source SHA256 p1_dequant.cpp
`1e705b2a8f2a1734ac57c3d2d2d14d30ce271a41d4845550f57c20bce0dc87e9`;
selected registration
`92827be5caa09c3715509267c32eb2b80ba6492ab7865a16aa1705c6c2c11579`.
All105 candidate templated metadata strings carry b70_p1_dequant;
zero overlaps with F1's129 metadata strings. The initial broad string
check falsely included the ordinary shared SYCL getDeviceKernelInfo
runtime function; kernel-template-only filtering passes. This is static
namespace evidence, not a runtime dispatch result. Shared control sources
still match the original P1 source manifest. Raw build log and
`kernel-namespace-check.json` are in p1/.

VERDICT: selected-DQ primitive ready, not GPU-tested. Proposed command
`TAG=component-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/p1/window.sh`
remains pending explicit root authorization after D1/N1 full windows.
Physicalcard1 only, expected1..2min body plus native pre/post health,
180s alarm/210s outer timeout, no model/PLE/bmm/serve. No speed or
stability claim. No additional CPU build is needed before the probe.


## 2026-10-21 17:04 -- selected-DQ primitive passes; card1 released

CONFIG: root-authorized physicalcard1/logicalxpu0; real Flash-Next GGUF
packs with the existing traffic-snake card1 placement E374. Selection is
32 noncontiguous local experts `[373,0,191,7,23,100..126]`. Exact F1 C13
reference library sha958e5d06 (full hash in window identity file); candidate
sha0ebb5837 as above. No model load, bmm, bucket op or serving hook.

COMMAND: `TAG=component-r1 bash
/mnt/vm_8tb/b70/build/vllm-flashnext/p1/window.sh` under cpu-run --quiet
outside card1 xpu-run lease; 180s alarm,210s outer timeout. Compare
integer views of complete output tensors, including untouched NaN halves.
Two warmups then median of five queue-event timings per control/candidate.

RESULT:20/20 actual-weight type/mode cases pass bit-exact; all five gate
quant types16/17/18/21/22 and both down types20/42. Modes0/1/2 covered
for each gate type, mode0 for each sampled down pack. GPU body44s exit0,
P1_COMPONENT_OK; nativepre/post HEALTHY (19s each), normal teardown,
post identity check passes, no reset. Card1 released immediately to root;
Q1's Strata references next own both cards. Raw `p1/run-component-r1/`
contains components.json, summary.json, logs, identity and health.

Gate shape per selected expert is rows640/cols2560; down is
rows2560/cols640. Every mode0 call writes104857600bytes (100MiB) of BF16.
Gate modes1/2 allocate209715200bytes, write104857600bytes to gate/up
interleaved halves and leave the other104857600bytes sentinel unchanged.
They are exact-layout gates, not separately timed. Input byte counts below
are the32selected-expert packed copy that the candidate removes; original
source pack has374/32 times those bytes. The control reads+writes this
copy (2x packed bytes) before performing the same dequant output writes.

| Layer | Kind | Type | Selected packed bytes | Gather+DQ us | Selected DQ us | Reduction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | gate | 17 | 15,155,200 | 1318.4 | 1011.2 | 23.3% |
| 0 | down | 20 | 29,491,200 | 701.0 | 551.4 | 21.3% |
| 1 | gate | 16 | 13,516,800 | 717.8 | 668.3 | 6.9% |
| 1 | down | 42 | 14,745,600 | 865.3 | 804.8 | 7.0% |
| 8 | gate | 21 | 22,528,000 | 601.5 | 503.6 | 16.3% |
| 8 | down | 42 | 14,745,600 | 606.1 | 532.1 | 12.2% |
| 12 | gate | 22 | 16,793,600 | 610.2 | 533.6 | 12.5% |
| 12 | down | 42 | 14,745,600 | 635.1 | 557.7 | 12.2% |
| 35 | gate | 18 | 20,070,400 | 595.2 | 510.5 | 14.2% |
| 35 | down | 20 | 29,491,200 | 651.4 | 515.3 | 20.9% |

These primitive medians improve6.9..23.3percent (median across10 sampled
rows13.4percent), with0.049..0.307ms saved per32-expert call. Packed copy
traffic removed per call is27.0..59.0MB (decimal read+write bytes).
Control always precedes candidate; no interleaved statistical validation
or full card1 expert context was run. Repeated Q2_0/down samples themselves
vary0.606..0.865ms, so do not treat isolated per-type rankings as stable.
At E374 with limited free memory, different chunk widths and allocator
conditions may change the result. No all-layer or serving speed claim.

VERDICT: selected dequant is exact at the component gate and removes a
measured copy cost; queue a small default-off serving hook after current
reference priorities. No further P1 GPU window is queued.

## Reproducible campaign checkpoint (path-only port)

CONFIG: root authorized adding the selected primitive and its finite
build/probe/window to campaign paths. No bucket/mask follow-on ported,
no automatic vLLM serving registration. Build artifacts stay outside git.

COMMAND: copy three measured native files byte-for-byte into
`plugins/b70_p1_xpu/csrc/`; copy measured primitive into
`harness/p1_primitive.py`; adjust build source paths and window invocation
only; Python syntax/bash-n/ASCII/hash equivalence checks.

RESULT: native files and primitive match measured scratch SHA256s;
`plugins/b70_p1_xpu/measured-component-r1.json` records them. Added
`harness/p1_build.sh`, `harness/p1_primitive_window.sh` and an explicit
Python loader `b70_p1_xpu.load()` (no entrypoint/auto-install). Build forces
disk TMPDIR=p1/tmp and MAX_JOBS4; uses current shared decode.hpp as before.
No additional GPU or build was performed for this path-only port.

Measured ABI: PyTorch2.14.0+xpu git08187d9e0fba026dc8217405802ab5381dc88d90,
XPUversion20260100; oneAPIcompiler2026.1,C++20,C++11ABI1,
fsycl-device-code-split=per_kernel,libsycl.so.9. Dynamic native dependencies
include torch_xpu/torch_cpu/torch,c10/c10_xpu and Intel math runtimes.
Host kernel7.1.0-070100-generic, host loader1.28.2 and IGC2 resolver gate
passed in the measured window. This binary is ABI-specific and must be
rebuilt if the stack changes; it is not a generic shelf artifact.

The concrete next hook is isolated to
`b70_gguf_xpu/csrc/bucket_moe.cpp`: behind default-off
`B70_PREFILL_SELECTED_DQ=1`, explicitly load P1 once in the prefill setup,
and call selected dequant on original gate/up/down packs using the
existing sel vector, expert_rows F/F/H, destination modes1/2/0. Preserve
host bucket plan, bmm shapes, streams, and combine arithmetic. The native
bucket can call the registered P1 op through a cached typed dispatcher;
its default path keeps existing F1 calls. Avoid adding a new hard DT_NEEDED
on the P1 SO. This is a proposed hook, not installed. Before adoption,
require matched warm8K/12K prefill, identity/coherence/exact primitive,
normal teardown/post-health and unchanged short decode under the same
recipe. Existing strict serving nondeterminism must remain explicit.

VERDICT: finite selected-only checkpoint is ready for root review/commit.
No additional setup is justified for unchanged code. The serving-hook
candidate remains a subsequent experiment, not a promotion.


## 2026-10-21 17:24 -- isolated serving overlay CPU-built, GPU gate pending

CONFIG: user explicitly continues the vLLM campaign; root authorizes an
isolated generated P1 overlay, same-binary flag0/1 arms, CPU-only build.
No shared fork or original plugin file edited. Q1's original-FA,
QSA-disabled long-fixture control has GPU priority. No DD restore.

COMMAND: read current bucket and prefill dispatch; add
`plugins/b70_p1_xpu/generate_bucket_overlay.py` and explicit
`b70_p1_xpu/serving.py`; build via
`B70_AGENT=P1 MAX_JOBS=4 P1_TMPDIR=/mnt/vm_8tb/b70/build/vllm-flashnext/p1/tmp
bin/cpu-run bash vllm/flashnext/harness/p1_bucket_build.sh`.
Run Python/bash syntax, ASCII, frozen source hash, native metadata and
CPU spawn-hook checks. No GPU access.

RESULT: generated source is based on frozen bucket SHA256
`cc0374a7a46108aaa37a4b1c86f170ab7d6e7663fe451aabcd0b208e7b800abd`;
generator requires this hash and exactly one occurrence of every edit
anchor, then emits only to disk build/p1. It changes the native namespace,
adds a boolean op argument and typed-dispatch helpers, conditionally
skips three gathers, and uses selected-DQ on original packs F/F/H,
modes1/2/0. Grouping, host bucket planning, scratch shape, bmm, swiglu,
combine and queue ordering are copied unchanged. No bucket source copy
is tracked. P1 op namespace differs from F1, and all duplicated SYCL
kernels are under b70_p1_bucket.

Generated source SHA256
`3cdff51b9fb8f4ad68477183fb542bcf1ccfc36c722387074acad7f11abc7b69`.
CPU build21s exit0; binary p1/_C.p1bucket.so207264bytes, SHA256
`36d11c6f7a5fbc4223bfa0ea219cd152bdc08cf3cc062232d9604b9dfaee57db`.
All24 templated metadata strings use b70_p1_bucket and share no names
with F1's129 or selected-DQ's105. The overlay invokes existing registered
F1/P1 dequant through cached typed dispatcher handles, avoiding a new
hard DT_NEEDED dependency on either extension. ABI/compiler unchanged.

The optional hook only intercepts card1 with mode native; every other
call goes to original _experts. `B70_P1_BUCKET_OVERLAY=1` explicitly
installs it; unset is default off. `B70_PREFILL_SELECTED_DQ=0/1` selects
gather+F1C13 versus original-pack+P1selectedC13 on the same bucket binary.
Top-level installation in harness/p1_api.py runs in both API and spawned
EngineCore before model construction. CPU spawn smoke passes both flag
values and card0 passthrough, without importing torch or touching GPU;
actual EngineCore hook proof still requires serve logs. Explicit load of
both libraries happens on first intercepted native call. Nothing is
installed as an automatic vLLM entrypoint.

Raw CPU evidence: build/p1/build-bucket-r1.log,
bucket-namespace-check.json, hook-cpu.log. Shared original bucket hash
remains unchanged. Selected primitive binary still sha0ebb5837.

VERDICT: CPU implementation ready. Do not serve until the original-vs-
overlay control gate below passes. No new speed or stability claim.

### Prepared bounded integration gate

COMMAND (not yet executed):
`TAG=bucket-r1 bash vllm/flashnext/harness/p1_bucket_gate_window.sh`.
Physicalcard1/logical0,12GiB host cap,180s body alarm,210s outer timeout,
expected1..2min body plus native pre/post health. No model or PLE load.

The gate uses actual card1 traffic-snake gate/up/down packs for layers
0/1/8/12/35, covering all five gate and both down types. T33/129 plus
layer8 T12283, real H2560/F640/E374/topk10/tile32/maxpad32768, synthetic
skewed routes explicitly labeled. Each of11 shape/type groups compares
original, original repeat, overlay flag0, and overlay flag1 bit-exact:
44 outputs. This checks the overlay qualifies as a control before using
it in a matched serving pair; original-repeat failure must remain explicit.
Record original/control/selected shapes, bytes, output error and wall
only as diagnostics. Preserve failure, then post-health; no reset by P1.

### Proposed matched full serving pair after gate

Use unchanged merged maxlen131072/chunk12288/maxseq4/noMTP, original FA
and QSA off exactly as Q1's preceding control; retain its other effective
flags. Explicit wrapper paths/flags for both arms:

```
F1_API_WRAPPER=$H/p1_api.py
F1_PYPATH_PRE=$REPO/vllm/flashnext/plugins/b70_p1_xpu
O4_SERVE_EXTRA="<same baseline flags> B70_P1_BUCKET_OVERLAY=1 B70_PREFILL_SELECTED_DQ=0"
```

Treatment changes only selected flag0->1. Both libraries keep the same
paths and hashes in both arms. Parent owns p1_prefill_probe.py and
p1_prefill_callback.sh, using identical frozen tokenized8K/12288/32K
fixtures, warmup+3 samples,16 output tokens with natural continuation.
Measure first nonempty content arrival, content arrival arrays and wall
separately; require equal prompt token counts and content. No max1 wall
fallback. Two long math coherence checks, mandatory existing short decode
bench, /v1/models first hotschmoe-dd + registered research alias, and
P1_BUCKET_HOOK_INSTALLED/P1_BUCKET_APPLIED selected flag proof are gates.
Disable redundant old long probe using F1_LENGTHS=',' F1_COHERENCE_LEN=0
F1_DECODE=0; root's callback supplies the matched prefill evidence.

Before the pair, snapshot/verify d1_source_identity.py (all campaign code,
dirty fork and effective baseline libraries), plus explicit SHA256s for
both P1 SOs and generated bucket cpp/json outside the repository.
Freeze source/SOs across both arms and verify after each. Normal API and
EngineCore teardown, native post-health, no engine-error trace and matching
identity remain mandatory. Existing strict serving nondeterminism is
reported separately; this is a diagnostic experiment, not promotion.

Expected cost per full arm:90..180s load (PLE26.8GiB host pin), roughly
60..120s clients/bench, health and teardown; reserve5..7min each,10..14min
for the pair after the bounded bucket gate. Only one Flash-Next process;
no DD restore between prepared windows. Root must authorize each GPU stage.

### Full serving wrapper ready (still not executed)

Root's callback/client are present. Added only P1-owned
`harness/p1_serve.sh`: source manifest includes all effective campaign
sources through d1_source_identity.py plus explicit original bucket,
C13/decode common source, generator, selected native source/header,
loader/hook, generated cpp/json and both P1 SO hashes. The external fork
HEAD and complete binary tracked diff are saved and compared before/after;
full source manifest also covers its untracked source files.

The wrapper exports the host selected flag and frozen fixture path for the
callback, and separately passes B70_P1_BUCKET_OVERLAY=1 and selected0/1
into the container. Both arms use p1_api.py and the same P1 SO paths.
Activation requires P1_BUCKET_APPLIED with the correct selected flag from
an actual card1 native call. Source identity, fixture hash, short identity/
arithmetic, root callback content/row count, normal serve exit, native
post-health and no engine-error are required. Once READY, client evidence
files must advance at least every120s; server periodic statistics do not
reset the monitor. A monitor-initiated stop is always an arm failure;
F1 owns teardown/post-health afterward. No reset or DD lifecycle is added.

Default maxlen is65536 to match Q1's preceding original-FA control;
root can override131072 identically in both arms. Chunk12288/maxseq4,
originalFA, QSAoff, N1off, oneDNNdetoff are explicit and equal. This
supersedes the earlier131072 default proposal for this pair only.

Prepared commands (root must grant GPU stages first):

```
TAG=bucket-r1 bash vllm/flashnext/harness/p1_bucket_gate_window.sh
B70_PREFILL_SELECTED_DQ=0 P1_PREFILL_FIXTURES=/mnt/vm_8tb/b70/build/vllm-flashnext/p1/matched-r1/fixtures.json TAG=p1-control-r1 bash vllm/flashnext/harness/p1_serve.sh
B70_PREFILL_SELECTED_DQ=1 P1_PREFILL_FIXTURES=/mnt/vm_8tb/b70/build/vllm-flashnext/p1/matched-r1/fixtures.json TAG=p1-selected-r1 bash vllm/flashnext/harness/p1_serve.sh
```

Bash syntax/ASCII checks pass. No GPU run, no full model or extra build
was executed for this wrapper. Source must remain frozen between arms.

Root reviewed/approved p1_serve.sh layout. Final pre-gate addition:
treatment requires a passed control marker (P1_CONTROL_TAG defaults to
p1-control-r1), byte-compares the complete source identity and explicit
P1 hash manifests, and checks callback fixtureSHA plus target/promptSHA/
prompt-token equality across arms. These compare source/identity only;
flag, env, tag and output locations are recorded separately. Maxlen65536
is fixed for the requested pair. No collective health is added for this
world-size1 sidecar; native per-card pre/post health remains per arm.
Freeze all harness/plugin files after the bucket gate and across both
serving arms; root owns starts. The treatment preflight and pair checks
fail closed if control is missing or sources/fixtures differ.


## 2026-10-21 17:32 -- bucket gate PASS; control source freeze starts

CONFIG: root-authorized physicalcard1/logical0; standalone integration,
E374/topk10/H2560/F640/tile32/maxpad32768, real card1 traffic-snake weights,
synthetic skewed routes. Types sampled at layers0/1/8/12/35, T33/129 plus
layer8 T12283. Same original F1 serving bucket binary as the matched arms.

COMMAND: `TAG=bucket-r1 bash harness/p1_bucket_gate_window.sh`.

RESULT:44/44 output comparisons bit-exact, maxabs0: original reference,
original repeat, overlay flag0 and overlay flag1 at11 shape/type groups.
Body51s exit0/P1_BUCKET_GATE_OK; native pre/post HEALTHY; all five
source/SO hash checks pass, normal teardown, no reset. Raw
build/p1/run-bucket-r1/{bucket.json,bucket.log,identity-*,health-*}.
The single-call wall numbers include launch/setup and are gate diagnostics,
not latency qualification. Standalone free-memory conditions differ from
an actual loaded card1.

VERDICT: original-versus-overlay0 exact gate qualifies this overlay as the
matched control. Root authorized immediate both-card control+selected pair.
All campaign harness/plugin sources frozen after the gate at17:32UTC;
reports/JOURNAL remain writable. Control starts flag0/TAGp1-control-r1,
maxlen65536, same frozen fixture path p1/matched-r1/fixtures.json.
Treatment is conditional on complete control success, native health and
identical source/SO/fixture identities. No DD restore between windows.


## 2026-10-21 17:39 -- matched control PASS; selected arm starts

CONFIG: both cards, maxlen65536/chunk12288/maxseq4, originalFA/QSAoff,
noMTP, same overlay binary with selected0. Sources frozen after bucket gate.

COMMAND: `B70_PREFILL_SELECTED_DQ=0
P1_PREFILL_FIXTURES=/mnt/vm_8tb/b70/build/vllm-flashnext/p1/matched-r1/fixtures.json
TAG=p1-control-r1 MAX_LEN=65536 bash harness/p1_serve.sh`.
Read-only GET/server_info and GET/metrics before and after callback.

RESULT: complete wrapper passes, including actual card1 selected0 activation,
short identity/arithmetic/coherence,12 streaming fixture requests (warm+3
samples each), both32K long math checks, full source/P1 SO/fork diff/fixture
checks, SERVE_RC0/ENGINE_ERROR_RC0, normal teardown and native post-health
both HEALTHY. P1_ARM_PASSED:0. First served name hotschmoe-dd, registered
research alias unchanged. Runtime metadata explicitly proves
speculative_config=null, enable_prefix_caching=false,maxlen65536,
max_batched12288,maxseq4,TP1. Readonly proof at17:36:11 before callback;
metrics-after callback captured17:38:18 (59387bytes).

Control first-nonempty-content TTFT medians (three measured samples;
request wall is separate,16output tokens):

| Target | Actual prompt tokens | TTFT ms | Input tok/s |
| ---: | ---: | ---: | ---: |
| 8192 | 8189 | 3341.403 | 2450.8 |
| 12288 | 12285 | 4876.350 | 2519.3 |
| 32768 | 32765 | 13949.494 | 2348.8 |

Short decode C1 ITLp50=15.2146ms; C2=19.5567/19.5149ms per stream.
Bench coherence reports same=true,Paris valid; this is not the full strict
promotion repeat suite. Frozen fixture SHA256
`d3811e41588b9f28ae94c0a5fbf0967876172921e966abbabb71680b72cb349a`.
Raw control under f1/merged-p1-control-r1 and p1/full-p1-control-r1.

VERDICT: control qualifies the matched comparison, not promotion. Immediately
start selected1/TAGp1-selected-r1 with identical fixture path, sources,
binaries,maxlen and all other flags. No source edits or builds between arms.
Do not claim end-to-end benefit from control alone.


## 2026-10-21 17:46 -- matched selected-DQ serving pair closes PASS

CONFIG: two fresh sequential world-size1/EP-sidecar servers on both B70s,
originalFA/QSAoff/noMTP, BF16 activation, identical ISTA GSQ-RCO IQ3_XXS
GGUF files and aliases, maxlen65536/chunk12288/maxseq4. Same generated
bucket and selected-DQ binaries in both arms; only selected0->1 changed.
All campaign harness/plugin source frozen across arms, native pre/post
health each arm, no heavy CPU outside the quiet measurement protocol.

COMMAND: authorized bucket gate first, then harness/p1_serve.sh with
B70_PREFILL_SELECTED_DQ=0/TAGp1-control-r1 and1/TAGp1-selected-r1,
same frozen fixture path. Each target uses one warmup+3measured requests,
16output-token cap, prefix cache off. TTFT is observed first nonempty
content event; request wall and all content arrival arrays are stored
separately. No max1 wall fallback was used in this comparison.

RESULT: both P1_ARM_PASSED markers present. Both short arithmetic gates,
Paris coherence,12 streamed requests per arm and both32K long math gates
pass. Long math uses32791 prompt tokens and returns323/667 exactly in
both arms. Both normal SERVE_RC0/ENGINE_ERROR_RC0; native pre/post both
HEALTHY; no watchdog stop, reset, DD restore, or source edit between arms.
Actual card1 selected flag activation is recorded in each EngineCore.
Full431-file source manifests and explicit P1 SO/source manifests are
byte-identical between arms, external fork HEAD/binary diff unchanged,
fixture SHA/tokenization identical. Hash checks after each arm all pass.

Runtime GET/server_info in each arm proves speculative_config=null,
prefix caching=false,maxlen65536,chunk12288,maxseq4,TP1. The complete
serialized configs differ only by generated instance_id and the Python
address in quant_config's object repr (same B70GGUFConfig class); all
other serialized fields match. Both pre-callback and post-callback metrics
were captured. This is stronger no-spec evidence than missing draft counters.

| Actual prompt tokens | Control first-content TTFT ms | Selected TTFT ms | Reduction | Control tok/s | Selected tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8189 | 3341.403 | 3198.826 | 4.267% | 2450.8 | 2560.0 |
| 12285 | 4876.350 | 4678.923 | 4.049% | 2519.3 | 2625.6 |
| 32765 | 13949.494 | 13509.475 | 3.154% | 2348.8 | 2425.3 |

Measured sample ranges do not overlap:
- 8189tokens: control3340.898..3346.080ms; selected3198.272..3207.971ms.
- 12285tokens: control4870.249..4884.626ms; selected4677.460..4693.876ms.
- 32765tokens: control13941.318..13957.275ms; selected13508.582..13520.149ms.

Short decode C1 ITLp50 control15.2146ms versus selected15.1589ms;
C2 control19.5567/19.5149ms versus selected19.5521/19.5627ms per stream.
The hook only touches eager card1 native prefill; do not attribute the
small short-decode differences to selected-DQ. No canonical512-token suite
or speculative decode was measured here.

Strict fixture text repeatability is NOT qualified. Across warm+3 requests,
control has2/2/2 unique completion texts for8K/12K/32K, while selected has
1/2/3. Per-position completion hashes differ across arms. Both arms have
within-arm variation; existing whole-serving nondeterminism remains open.
This result does not establish that selection caused or repaired it.
The raw component and bucket gates were bit-exact, and sampled long math
and ordinary short coherence pass, but those do not prove strict full-model
exact repeat or broad long-context fidelity.

Raw matched summary is build/p1/matched-r1/summary.json, runtime config
diff alongside it. Detailed stream content_arrival_ms/itl_samples_ms,
completion text, counts, wall and usage are retained in each arm's
f1/merged-<tag>/p1-prefill.json. Source/fork/fixture/activation artifacts are
under p1/full-<tag>/. Complete fixed fixture SHA256 remains
`d3811e41588b9f28ae94c0a5fbf0967876172921e966abbabb71680b72cb349a`.

VERDICT: this matched process pair observes3.15..4.27percent lower first-
content TTFT from eliminating packed copies, with exact component/bucket
checks and clean sampled lifecycle. Keep the selected-only candidate opt-in
for research; no shelf/DD promotion. The experiment is one fixed-order
control-then-treatment pair, three warmed samples per target, not an
interleaved stability study. No QSA/200K/MTP/canonical-suite claim. The
12K saving197.4ms is consistent in scale with the earlier material gather
budget, but q2 profiling used different prompts/instrumentation and does
not give a direct per-stage attribution for these runs. No further GPU
work is justified to repeat unchanged passing gates. Both GPU leases
released to root; DD remains down per user direction.
