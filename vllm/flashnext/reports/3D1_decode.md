# D1 -- no-spec decode after the 2026-10-21 outage

Owner: Codex D1 (Sol 6.1 requested by user). Parent coordinates both cards,
DD lifecycle, and shared source ownership. No commits or external submission.
Scratch: /mnt/vm_8tb/b70/build/vllm-flashnext/d1.

## Recovered Ld trace: short-context native FA owns the growth

CONFIG: CPU-only recovery of O4's completed Ld trace. Ld used O4 f14,
max_len 131072, no MTP, dense FA, real lmx reasoning chat input, 512 output
tokens, native GGUF o4g and EP o4d. It did not use Lc route or Q8F flags
(o4/ex-Ld.txt). The trace inflates absolute latency and is diagnostic.

COMMAND: B70_AGENT=D1 bin/cpu-run python3
vllm/flashnext/harness/d1_trace_growth.py
/mnt/vm_8tb/b70/build/vllm-flashnext/o4/run-Ld/steps.pkl --output
/mnt/vm_8tb/b70/build/vllm-flashnext/d1/Ld-growth.json.
The preexisting trusted local pickle derives from the raw 1000 MiB trace
o4/run-Ld/trace/python.7.json. Select modal-count segments (1938 kernels
on card0, 576 on card1), compare complete graph steps [8:72] to last64.

RESULT:

| Card0 traced metric | Early64 | Late64 | Change |
|---|---:|---:|---:|
| graph segment span, ms | 20.837 | 21.693 | +0.856 |
| XeFMHAFw family kernel sum, ms | 0.912 | 1.741 | +0.829 |
| EP wait kernel sum, ms | 0.412 | 0.612 | +0.200 |

506 complete card0 graphs and 511 card1 graphs were retained. Other card0
kernel families change by <=1.8 us each. Card1 grows mostly in EP waits
(+0.619 ms); expert families also change with routed experts. Kernel sums
and graph spans differ, so do not add the deltas as critical-path savings.
Raw Ld window.log: RUN_RC:0, gates 323/667 pass, C2 same text, teardown
SIGUSR1, HEALTH_AFTER_RC:0 and xpu-health HEALTHY. Ld skipped its own
pre-health after Lc; it is pre-outage evidence and does not qualify the
current host or a new serving recipe.

VERDICT: target short-context full-attention decode. The native Xe2 source
at vllm-xpu-kernels/csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp:385
sets kMinBlocksForSplit=32 for tile64 and collapses all KV splits below
that threshold. The actual 1600-token page is divisible by64, therefore
contexts below2048 still use one split even with B70_FA_KV_SPLITS=8.
This explains why the previous fixed-split experiment did not remove the
growth. The two KV heads expose too little parallelism in this range.
The expected benefit remains an inference until a matched GPU probe.

## Prepared bounded discriminator

CONFIG: physical card1 only; random BF16 Q/K/V, real shape24/2/256 and
page1600, graph metadata maxK131072, live KV128/512/900/1024/1536/2047/
2048/4096. Native FA vs upstream Triton normal/grouped split-K, splits4/8/
16. Independent FP32 reference, numerical error and bitwise FA comparison.
Three graph timing batches of32 replays per case. No model load or build.

COMMAND: D1_TAG=fa-probe-1 bash
vllm/flashnext/harness/d1_fa_window.sh. Uses cpu-run --quiet outside
xpu-run's card1 lease, separate leased pre/post health, Python alarm180s.
Expected <=4min including health. Root must authorize GPU scheduling.

RESULT: CPU py_compile and bash -n pass. GPU not run yet. Added native
compact-plan variants at splits4/8/16: d1_fa_plan.py computes live-length
descriptors on device, bypassing the legacy collapse through the existing
plan-driven native kernel path. Empty static-grid rows receive count0;
the actual per-sequence split count caps to the available KV tiles.

VERDICT: compare existing kernels before paying for a private native FA
rebuild. No shared FA, fork, plugin binary, or serving source changed.
Prepared isolated Python overlay via d1_prepare_fa_overlay.py at
d1/fa-overlay. B70_D1_FA_PLAN enables only pure one-token BF16 24/2/256,
page1600, causal full attention; prefill and all other shapes retain the
original interface. This is unmeasured and not enabled for any server.

## Card1 discriminator measured: select native compact plan16

CONFIG: the bounded discriminator above, physicalcard1/logical0. All
candidates captured in one process. Warm3 replays then median of three
device-event batches of32 graph replays, with live sequence length changed
between cases. This measures one attention call including the plan-fill
kernel; it is not full-model ITL. Reference is independent FP32 QK,
softmax, and AV on the same BF16 input values (atol0.003,rtol0.03).

COMMAND: D1_TAG=fa-probe-1 bash harness/d1_fa_window.sh, authorized by root
after root baseline WINDOW_DONE and post-health HEALTHY. Raw:
/mnt/vm_8tb/b70/build/vllm-flashnext/d1/fa-probe-1/{probe.log,status.txt,
health-before.txt,health-after.txt}. Body67s, no native build.

RESULT:

| KV tokens | Native FA graph us | Compact plan16 graph us |
|---:|---:|---:|
| 128 | 106.30 | 48.02 |
| 512 | 94.34 | 28.75 |
| 900 | 163.88 | 34.00 |
| 1024 | 174.18 | 34.66 |
| 1536 | 252.11 | 48.09 |
| 2047 | 138.84 | 48.49 |
| 2048 | 139.11 | 48.30 |
| 4096 | 228.34 | 69.97 |

All24 native compact-plan cases (splits4/8/16 x8 lengths) pass the
FP32 reference check. Plan16 maxabs0.003160 at128, <=0.001046 at512+;
the split path is not bitwise native FA. Native FA itself also differs
from FP32 (maxabs0.001833 at128). Upstream Triton grouped passes but is
slower (e.g. grouped16 at1024 roughly156us, at2048 271us). Upstream
Triton normal fails all3 split variants at128 (maxabs0.007892), so the
whole sweep exits1 with bad3. This is a rejected candidate, not a GPU
failure: native pre/post HEALTHY, normal container exit, no crash/reset.

VERDICT: select B70_D1_FA_PLAN=16 for a matched full-model experiment.
The native 2047/2048 discontinuity demonstrates the legacy split
threshold. Cold128 absolute timings are not a steady-state claim.
The component evidence supports the plan; model coherence, exact-repeat,
canonical512 ITL and teardown/post-health remain required.

## Baseline canonical collection and integrated callback

CONFIG: root live merged baseline, port18147, stable hotschmoe-dd first,
alias qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4, original O4+F1 flags,
maxlen131072, no D1 candidate. Parent requested client-only canonical
512-token collection before teardown, explicitly dirty diagnostic.

COMMAND: CPUquiet stock localmaxxing/lmx_vllm.py --measure-local,
both suites. Raw d1/baseline-lmx-r1/measure.log and command.txt.

RESULT: collector fails before any warmup/request because standard serve
does not expose /server_info?config_format=json (HTTP404). No speed data,
partial valid packet, retry or external request. Parent server was already
entering teardown, so no attempt was made to preserve it.

VERDICT: future control and plan arm use harness/d1_api.py (existing
x5_server_info read-only config route + stock CLI) and integrated
d1_lmx_callback.sh before teardown. Callback owns CPUquiet, records dirty
fork patch/untracked metadata, collects canonical warmup1+3 iterations
with authoritative usage and isolation counters, and labels local
validation ranking_eligible=false/diagnostic_only=true. Parent added
F1_API_WRAPPER,F1_PYPATH_PRE,F1_POST_PROBE_SCRIPT to its merged harness.
Candidate flags: F1_PYPATH_PRE=d1/fa-overlay, B70_D1_FA_PLAN=16. Same
N1 determinism flag and other settings must be used in both arms.
