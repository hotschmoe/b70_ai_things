# Strata peer activation gather: standalone alias oracle

Date: 2026-10-15. S35. Numerical and storage checks passed; real IPC serving
and performance remain unqualified.

CONFIG -> Frozen standalone binary SHA256
`4f2b42fb5eef48c0b2cfb624a0952fd397a5b01bef56bfb81aeea74f617bbcb1`.
Physical card 1, PCI `0000:44:00.0`, explicit in-order queue. Use the exact
production activation gather, native IQ-to-FP16 dequantization, FP16
gate/up GEMM, SiLU and down GEMM. The fixture selects 363 real expert blobs
across all ten GU/down format pairs in the GGUF, with synthetic routing.
This mixed-layer fixture tests numerical coverage, not one production
layer's performance. Weights occupy 650,060,800 bytes.

The reference gathers inputs separately for each expert and processes
experts in forward offset order. The candidate gathers inputs once into
the existing FP32 output allocation, reinterpreted as FP16, then processes
experts in reverse offset order. Both retain each expert's weights, GEMM
shapes, arithmetic and output offsets. No extra production VRAM is proposed.

For an expert starting at row offset `o`, earlier unconsumed inputs end at
byte `2*N*o`, while its output starts at `4*N*o`. Higher-offset outputs lie
beyond its input. Its own input may overlap its output; the in-order queue
must finish the GU read before the down GEMM writes that output. Production
queue, reader and lifetime source checks are recorded in the
[S34 audit](20261015_strata_ep_prefill_dispatch.md). This oracle uses ordinary
device USM, so it cannot substitute for actual IPC validation.

COMMAND -> One offline CPU-only AOT build under `bin/cpu-run`, no GPU device
mount, four link jobs; build completed in 135 seconds. Exact reviewed source
and all eight linked archives are recorded. Each case ran under a quiet CPU
lease outside a card-1 GPU lease, with explicit device pin and PCI assertion.
Mean 80 was inspected before starting mean 240. Per-card pre/post health
passed for each case; both containers were removed and wrappers exited 0.

RESULT -> All output bits and storage checks passed.

| Case | Actual T | Routed rows | Largest expert | FP32 values compared |
| --- | ---: | ---: | ---: | ---: |
| Mean 80 | 4096 | 29,040 | 641 | 74,342,400 |
| Mean 240 | 12288 | 87,120 | 1921 | 223,027,200 |

Each fixture puts the largest expert first and a one-row expert second,
then unequal counts. Every expert count fits its actual T. Both use the
fixed 1,258,291,200-byte output allocation corresponding to C=12288, K=10,
N=2560. Output use is 297,369,600 and 892,108,800 bytes respectively.

Checks cover byte-exact gathered inputs, finiteness of every used output,
bitwise equality of all 297,369,600 FP32 values, distinct output poisons,
the entire unused output capacity, and prefix/suffix guards on nine
buffers. Gather validation/readback occurs separately from the candidate
compute arm; there is no added host wait between its gather and GEMMs.

Host enqueue, subsequent drain, queue-completion wall and device-marker
spans are retained separately. The reference runs first and the candidate
second, so cold/warm initialization and order bias prevent a speedup claim.
No isolated timing is used to select a serving winner. Optional barrier
profiling failure emits JSON null while preserving numerical checks.

VERDICT -> The aliasing and arithmetic mechanism passes this standalone
gate. Next, test a default-OFF production path with conservative IPC,
reduction/PUSH, residency, offset/capacity and queue guards; fall back to
the original loop when they fail. Real imported-memory visibility,
concurrent primary readers, reduction, request state and end-to-end latency
still require a bounded serving comparison. No daily promotion, new cache
qualification or current throughput claim follows.

Actual responding model IDs `claude-opus-5-5` and `claude-fable-5-1` are
recorded from assistant events. Both completed the prepared-source review
before the build; root addressed their optional profiling-query failure
concern. A separate prepared prefill-marker patch was reviewed while CPU
leases were busy. Both reviewers caught a layer-attribution placement issue,
fixed in v2 before applying or compiling it. The next planned native build
can combine that diagnostic with the serving candidate. A new K/V fusion
kernel is deferred: its main C6 signature pool totals only 0.282 ms of
recorded service per cycle, and only 7/12 layers share K/V formats. That
pool is not an additive wall budget or a proven saving.

Loop cost: one build, two finite one-card cases, no model-server reload,
new profiler trace, daily restore or full-cache rerun. CPU/GPU windows were
coordinated with the separate backend agent through the shared log.

Exact tested source:
[peer_gather_oracle.cpp](../strata/flash-next/lowlevel_probe/peer_gather_oracle.cpp),
SHA256 `6d7c82043a9d66f4b46bafbb9b226f10e90c8fae971e29e9f54317365db96d06`.
Build/run recipes, link map, archive hashes, review events and raw closures:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/peer-gather-oracle-s35/`.
Start with `s35-report.json`, `build-closure.json`, and
`quiet-mean{80,240}/closure.json`. `ROOT_RUN.sh` is the original reviewed
recipe; actual runs used `leased-case.sh` to include card-1 health under
the same lease. Original preparation/review snapshots remain preserved.
