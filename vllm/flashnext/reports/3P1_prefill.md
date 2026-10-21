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
