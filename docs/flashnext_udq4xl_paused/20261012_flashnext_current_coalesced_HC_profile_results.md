# Current coalesced HC configuration device profile, 2026-10-12

CONFIG -> Fresh55 originalUDQ4 pair24/24, P2048/pref512/ctx4096,
GEN8 warmup and selected request. HCtile2/coalesce1/copy0/mirror1,
noMTP/grouped0/ESIMD0; H36/Chrome CPU lanes OFF. Bounded unitrace.

COMMAND -> Producer86336 PASS445s, public reader81683 PASS, actual
raw inspector10926 PASS. Warmup/selected outputs equal. Source/all8/
full4/percard/compiledcollective/kernel/normalownedretirement passed.

RESULT -> Joined same-device raw Clock/Chrome instances, selected request:

| Device descriptor | Selected operations | HC Q8 union ms | Expert dequant union ms | Named/vendor GEMM union ms | Explicit DMA union ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 214295 | 2140.042 | 1480.112 | 435.137 | 1615.560 |
| 1 | 197455 | 2128.494 | 1615.990 | 480.894 | 1325.347 |

HC Q8 has384 observed operations per card, primarily the exact source55
HcQ8TokenReuseKernel<2> family. Dequant has57670/61678 operations, GEMM
58054/62062, explicit DMA9150/5829. Named classifications are observation
labels, not numerical proof from names alone. Family unions overlap and
are nonadditive; card sums are not a request critical-path duration.

The trace covers a selected prompt-plus-eight-output request. No calibrated
prefill/decode time split or host/device causal alignment is claimed.
Uncovered timeline hulls and operation gaps are observed gaps, not physical
idle. Direct mapped RAM traffic, physical bus bandwidth, full operation
coverage and clean serving speed remain unqualified. Instrumented native
PP time24937ms is not the clean241tok/s performance measurement.

VERDICT -> HC Q8 projections remain the largest named compute family,
followed by dequant; transfer is substantial. The next focused operator
comparison is current2/2 token reuse against down2/up4 at actualT512.
PriorT8 tile4 down regressed while up improved, so no uniform tile4
promotion or broad tile sweep is inferred. A whole-model gain still
requires fresh matched output and lifecycle evidence after operator work.

F17 producer report SHA5fe1c93e33ca630cc26382e651e9a5e319e145339a93fa8a4496a1feb0eeddac.
Public SHAf1d8caaff96ee821efda370af1cdc577cb887a08c04041b22859c5876d1fd00b.
Observed SHAc57c78dbb0468a4316f33f615a158ac8d87fbdd8d3a58e52621c79391bffda66.
