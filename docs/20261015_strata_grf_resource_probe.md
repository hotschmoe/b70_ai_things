# S39: existing HC register-allocation option (2026-10-15)

CONFIG -> Keep the current default register allocation. The existing
`STRATA_GR_LARGE_GRF=1` option passes numerical and graph checks but is
slower in every instrumented case. No serving arm or new engine build is
justified by this probe. S30's qualified152055a6/headpairON/depth2 remains
recoverable; neither absolute speed goal is newly met.

The standalone adapter calls unchanged production `fused_gr_read_multi`,
linked from current `libstrata_kernels.a(fused_gr.dp.cpp.o)` and
`libstrata_core.a`. No replacement math, setter or kernel is introduced.
One frozen executable, SHA256
`7365c788fd82a31a7485571f7767331a8f22f99a0dcc5fb47c51d2d860478ec7`,
runs in fresh GRF0 and GRF1 processes on physical card1/PCI44:00.0.
Controls stay MORE_GROUPS0, DOWN_SLICED1, UP_COLS16, UP_ILP0, V30,
NORM_SPLIT1; Q8/QFUSE environment keys are absent. Shape N2560/HC4/LR320,
synthetic seed7 fixture from the existing benchmark, BF16 projection weights.

COMMAND -> Actual Opus5.5/Fable5.1 exact-source reviews and resumed v2
reviews precede one29s offline CPU-only adapter build. Same current archives,
BMG-G31 AOT and correctly-rounded divide/sqrt options as production. No DRI
mount or production edit during compilation. Link map and actual symbol
confirm the production member rather than a test replacement.

Two finite quiet-CPU/card1 windows take39s and47s, including pre/post health.
Each runs the same12 cases: T1/2/6 x apply0/1 x inject0/1. All12 buffers have
allocation guards, active FP32 finite checks, inactive/tail/untaken-output
checks and immutable input checks. Export all8 allocated rows of R, lo, rs,
inject, mixed and xn,747776 bytes per snapshot. At T2/6 compare graph replay
with direct calls on reset original and changed inputs. T1 is a direct
control; no T1 graph claim. Three raw rounds/case use32 direct reads each.
Complete host enqueue/drain and optional device barrier spans stay separate.

Existing unitrace collects execution names and kernel properties during
those same numerical processes. There is no extra profiling process or
server. Every recorded probe timing is profiler-sensitive; quiet CPU does
not make these clean end-to-end measurements. OFF then ON is fixed order,
with another session's card0 work between the windows.

RESULT -> Both numerical processes pass. All48 full snapshot pairs,
35,893,248 bytes, match exactly, including inactive data and poison. Active
finite checks, guards, graph/direct equality and changed-input observability
pass. The trace records1260 expected DOWN and1260 expected UP executions
per mode. Mode0 executes default names; mode1 executes the `_grf256` names.
Thus the option selects the intended property-bearing source branches.

| Reported property or instrumented observation | Default | GRF option |
| --- | ---: | ---: |
| Register File Size Per Thread, DOWN and UP | 128 | 256 |
| Private / Spill Memory Per Thread, DOWN and UP | 0 / 0 | 0 / 0 |
| UP SLM Per Work Group, bytes | 12288 | 12288 |
| DOWN SLM Per Work Group, bytes | 0 | 0 |
| DOWN average service over1260 calls, us | 12.197 | 13.749 |
| UP average service over1260 calls, us | 19.505 | 25.467 |

The resource fields are actual unitrace kernel-property reports, not a
dynamic occupancy measurement or proof of the cause. Both kernels are AOT
SIMD32. Across all12 cases, median instrumented whole-read host duration
increases4.62-17.62%, three rounds each. Barrier-to-barrier intervals can
include host submission starvation. Kernel sums overlap other scopes and
are not an additive serving budget. No production regression is inferred.

Both unique containers are removed normally; both lease wrappers and CPU
wrappers exit0. Card1 pre/post health passes in each window, no reset. No
card0 computation, two-card collective or model-server test belongs to this
stage. All leases are released; no selected serving configuration changes.

VERDICT -> The existing resource knob is ruled out as the next serving
candidate under these tested controls. Save two model-server starts and a
full-context repeat. Do not retry the same knob without a new mechanism.
The MTP Q4 projection sharing omission is independently confirmed and its
small fix reviewed, but remains source-only preparation. Its kernel-selected
Q8 dense service pool is small; no new quantization sweep is queued.

Review corrections before the only build: export xn too, require active R
changes for apply1, and record summary iterations. Existing iteration-key
matching already prevented unequal run lengths. Actual bundled DPCT source
and existing graph parity resolve ownership, device-count and profiling
queue concerns; runtime graph replay then passes. The expected hash file
already existed, and production AOT options already matched. Model reports
were checked against evidence rather than adopted verbatim.

Raw root: `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/grf-oracle-s39-prep/`.
Key files: `s39-report.json`, `build-closure.json`, actual model review
folders, `v2/`, `run-profiled/window{0,1}/closure.json`, `comparison.json`
and `profiled-timing-comparison.json`. The reusable adapter is
`strata/flash-next/lowlevel_probe/grf_oracle.cpp`.
Frozen raw manifest:193 files/77,861,388 bytes, SHA256
`ee593e93ad068e0ec03d8406de3c7bce55e6d63f09ef91d0db871f1458f60fd5`.
