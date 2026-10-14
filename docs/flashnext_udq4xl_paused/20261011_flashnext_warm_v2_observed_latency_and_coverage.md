# Warm V2 observed latency and missing compute coverage

CONFIG -> SDK43 mode0, short54-token prompt, GEN4, graph warmup then selected
request2 in each fresh OFF/ON process. Exact model and normal thread pins;
registry173 unchanged. This is an instrumented diagnostic, not natural serving.

COMMAND -> Actual producer35802 closed in1522s. Independent public45500
and observed analysis92649 are terminal. Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010

RESULT -> Producer report66d7c2a5ae4b17786d85ab06bb634af1c8f67f2e574cb3d015e26e007cb4d6e6.
Public source43-warm-pti-short-v2-root-public-binding-v1.json
SHA094ffb0d99ceeaec3ea1abb70b304ff4fd43face32bf66e741208af44c4637be
passes its declared narrow scope, including output equivalence, graph reuse,
actual IPC pause/resume, finalizer, normal retirement and full4/health closure.
It explicitly leaves model-complete device coverage and physical idle false.
Analysis source43-warm-pti-short-v2-observed-analysis-v1.json
SHAa040d748721c13bf06a9740476f2498a338ed1c7ad23fd0ed4734d39b3282d27.

| Selected request2 | OFF | ON |
| --- | ---: | ---: |
| Native TTFT ms | 4364.685889 | 4373.040876 |
| Completion ms | 4668.596867 | 4676.983315 |
| Token-gap P50 ms | 106.677347 | 106.676356 |
| Token-gap P95 ms | 122.641441 | 122.690364 |

ON completion is8.386448ms longer in this single pair; no repeatable overhead
or speed-gain claim. Native emission markers do not measure client delivery.
Host replay wait includes almost the entire request; nested host intervals
cannot be added or used to identify GPU compute, transfers or idle time.

The closed clock has89 normalized operations, all graph_command0. Chrome
SHA56cbb021b5426351bd956baf08c19b3d609cf4cd359b4c8de86f41d4304b8125
contains57 ImmediateAppendCommandLists submissions but only24 big_fill_zero,
39 fills and26 barriers as GPU operations. Model compute kernels are absent.
No utilization, physical idle, bandwidth or critical-path kernel conclusion
can be derived from that device union.

VERDICT -> Actual request/lifecycle closure and scoped host measurements,
not complete device profiling, clean serving speed or original model fidelity.
Source-backed next test retains ordinary closed-list descriptors while paused,
then checks two actual SYCL graph child kernels after named resume in a tiny
fixture. That opt-in executes timestamp instrumentation during paused capture;
no zero-added-query or zero-overhead claim. Full-model retry follows actual
fixture coverage and lifetime proof. Full campaign remains active.
