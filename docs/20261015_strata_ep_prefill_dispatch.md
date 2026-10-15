# Strata EP prefill dispatch: historical evidence and the next oracle

Date: 2026-10-15.

S34 distinguishes a short historical activation-gather kernel from a much larger
following primary trace gap associated with peer host submissions. It changes no
runtime code. S35 is prepared source for an aliasing and numerical oracle; it has
not been built or tested.

CONFIG -> Reuse the complete S10 PUSH prefill trace from the retained S9 binary
`932d2f347219edd668f60048204b6578a823df03c213a45edf043861b823184e`.
The request had 4,051 input tokens and 32 output tokens, with a 4,044-token main
prefill chunk. Historical ownership was 155 primary and 357 peer experts per
layer, versus the current 149 and 363. Source, batching and the later EP
publication fix also differ. These are historical measurements, not S33 timings.

COMMAND -> CPU-only reuse of the saved raw trace, client window and timebase.
`extract_s10.py` parsed the 161,712,114-byte raw file once: valid JSON, stable
size/mtime, 1,033,778 events, SHA256
`ccbd8cd652704039f17ce2970b34d018951a6c56212b87d71179feb7cb7169e3`.
`analyze_s10.py` then used the retained compressed extraction. No GPU, build,
new test or runtime edit was needed.

RESULT -> All 48 main-chunk primary gathers have ordered full-chunk finish
markers and a first primary expert-dequantization operation. Layer numbers are
attributed by source order; kernels do not expose an explicit layer argument.
All 48 layers are retained, including the first.

| Historical measurement | Total ms | Median ms | p95 ms | Range ms |
| --- | ---: | ---: | ---: | --- |
| Primary activation gather | 7.863 | 0.149 | 0.278 | 0.096 to 0.304 |
| Gather completion to first primary dequantization | 2,388.975 | 52.987 | 62.579 | 24.455 to 64.687 |
| Host gather-append return to first primary dequantization append | 2,395.300 | 53.146 | 62.724 | 24.673 to 64.710 |

No positive-duration primary GPU-operation event covers those device gaps;
zero-duration barrier markers can occur. This establishes uncovered primary
trace intervals, not physical GPU idle. H2D flow IDs uniquely link all 85,593
kernel append calls in the request to their target device. Every kernel append
inside the 48 host dispatch spans targets the peer: 58,716 calls, with a median
1,307 per layer, on the same host thread. Peer device identity classifies host
calls; peer GPU timestamps are never joined to primary intervals.

The recorded host L0 API union inside those disjoint host spans is 361.786 ms,
including 343.737 ms of kernel appends, compared with 2,395.300 ms elapsed.
Most elapsed time is outside those API scopes. The trace does not separate
SYCL/oneMKL/profiler work from scheduling there. API time is not a removable CPU
budget and must not be added to device gaps or peer computation. Unitrace roughly
doubled historical 4K wall time, so these values do not predict uninstrumented
latency or a speedup ceiling.

The current source requires an attribution correction. The actual SYCL build has
no `STRATA_PREFILL_MMQ`; `mmq::built()` is false and `gather_native` /
`gather_native_group` are unreachable stubs. `STRATA_PREFILL_GROUP_GATHER` therefore
changes no active serving path. The primary `gather` timer marker opens before
activation `gather_rows16`, remains open while the host submits peer work, and
ends at primary dequantization. It is not an isolated weight-copy measurement.
The peer IPC loop submits activation gather, fused IQ-to-FP16 dequantization,
gate/up GEMM, SiLU and down GEMM for each selected expert. This source ordering
matches the historical trace motif; its current unprofiled cost remains unknown.

S7 already tested per-layer `std::async` peer submission with explicit device
context, owned peer-order lifetime, CPU 1 affinity and a join before combine.
The same v2 binary was slower with async enabled: 4K prefill 1,046.8 to 1,002.3
tok/s and 32K 1,261.2 to 1,242.7 tok/s. Corrected three-repeat median TTFT was
3.910 to 4.082 seconds and 26.014 to 26.401 seconds. Repeating that mode without
a distinct mechanism is rejected. The result does not isolate task creation,
runtime contention or another cause.

Independent review provenance is recorded from assistant-event metadata:
`claude-opus-5-5` and `claude-fable-5-1` actually responded. Both initial source
explorations ended at their turn caps with exit 1. Their resumed verdicts ended
successfully, and the additional Opus alias review also succeeded. These are
usable source recommendations, not executed validation or approval of the exact
prepared S35 source.

Opus identified primary-only grouped FP16 XMX as an existing seam, but it changes
GEMM reduction order, needs peer buffer/capacity work and lacks the required
mode-1 parity coverage. Its retained primary result was 11 to 15 percent slower.
It is deferred. The review's approximate extra-slot memory estimate is not
accepted: the actual peer dequant buffers total 9,830,400 bytes per slot, so 15
additional slots would require 147,456,000 bytes.

The narrower candidate gathers all peer activation rows once into the existing
FP32 output region reinterpreted as FP16, then processes experts in reverse
output-offset order. Earlier unconsumed inputs end at byte `2*N*o`; current
output begins at `4*N*o`. The current expert can self-overlap, so its GU read must
finish before its same-queue down write. Dequantization, GEMM shapes, output maps
and later reduction order remain unchanged. Root checked the allocation at
`prefill.cpp:1893`, producer queue creation at `:1899`, and readers at
`:4382` through `:4387`: reduced mode reads `sum_reg`; the remote `rows_reg`
reader is under `!reduce_rows`. Debug checks read primary scratch, not this
peer alias. These source checks do not prove oneMKL ordering, mapped-memory cache
behavior or bitwise output.

Fable found no ready HC fusion and proposed a separate conditional K/V dense
fusion: two 512-row projections share the same quantized activation input, so a
two-segment entry could remove one submission while preserving each row's work.
No such entry currently exists. Root's actual GGUF header audit found matching
K/V type and shape in only 7 of 12 attention layers: 11, 19, 23, 27, 31, 35 and
39. Layers 3, 7, 15, 43 and 47 need fallback. The audit read metadata only; it does
not establish numerical equivalence or a latency win. This remains separate from
S35 and requires kernel-path guards and a real-weight numerical oracle.

VERDICT -> Historical evidence supports investigating repeated peer dispatch,
not a native-MMQ gather microbenchmark or an unchanged S7 async retry. The next
prepared discriminator is S35's one-card forward/per-expert reference versus
single-gather/reverse-alias candidate using exact production functions and real
mixed-format GGUF weights. It checks all FP32 output bits, gathered bytes,
finiteness, guards and unused-capacity poison at mean row counts 80 and 240.
Its synthetic routing and mixed layers are numerical coverage, not a production
single-layer performance model. Host enqueue, queue-completion wall and device
marker spans are recorded separately; cold/warm order prevents a performance
conclusion. There is no binary or GPU result yet. Any later production candidate
still needs conservative IPC/reduction/PUSH/residency guards, old-path fallback,
a bounded serving canary and coordinated health closure.

Primary evidence: [S34 report](/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/ep-prefill-dispatch-s34/s34-report.json),
[per-layer historical gaps](/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/ep-prefill-dispatch-s34/historical-gaps.json),
[review closure](/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/ep-prefill-dispatch-s34/review-closure.json),
[actual K/V header audit](/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/ep-prefill-dispatch-s34/kv-projection-types.json),
and [S35 source-only readiness](/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/peer-gather-oracle-s35/readiness.md).
S7 and S10 context, including the latency erratum, remains in
[the optimization record](20261014_flashnext_strata_optimization.md).
