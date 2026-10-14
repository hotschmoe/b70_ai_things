# Source55 native HC prefill coalescing candidate

CONFIG -> Direct actual source54 SDK
`/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T032543Z-3_j1w16w`,
receipt `7684031d565fb800e7a3e71f9fc59717ed43c02d20d49daee03844cae6b790af`.
Default OFF: absent/0 `STRATA_HC_PREFILL_COALESCE` retains old eight-row calls.
Value1 selects new explicit prefill read/write APIs with rows1..512 and tile2
Q8 projections. Invalid values refuse the native prefill path. Decode, verifier,
old eight-row descriptors, primary quantization, and precision views are retained.
No new workspace, synchronization, query, allocation, or captured input exists.

COMMAND -> Apply `candidate-default-off-hc-prefill-coalescing-v1.patch` to
that exact source. Build the new `native55-hc-prefill-coalescing-engine-build-proposal-v1.json`
with the unchanged production O3/c++20 precise/correctly-rounded/device-link
configuration. Before a model run, compile the standalone
`kernels/hc_prefill_coalescing_fixture_v1.cpp` against those new composition and
projection sources plus the same original queue support used by the source54
operator fixture. Root executes every GPU operation under the appropriate lease.
The author has performed only six-file exact source apply/undo; no build or GPU
operation. The full74 source roster/eight targets stay unchanged.

RESULT -> Source-only candidate. `/tmp/b70-hc55-source-apply-undo-v1.json`
records exact before/after source SHA and reverse restoration. GPU operator,
fresh55 engine, model coherence, and performance are unexecuted. Current source54
job54311 and every source54 deliverable remain immutable.

VERDICT -> This can reduce the number of HC submissions; it has no measured
latency or full-model fidelity authority. At T512, the original native read
submits six kernels per eight-row slice: norm, down, silu, up, F32 injection,
and mix. This is 384 submissions per layer half versus six coalesced submissions.
The separate residual write changes from64 to1. These are source-derived counts,
not measured launches or a claimed runtime gain. Downstream attention/experts,
PLE, precision conversion, and model-state dependencies remain untouched.

## Actual ownership and stride table

The direct54 `sycl/src/prefill/prefill.cpp` Alloc::take at317-346 gives each
carve a distinct advancing byte interval, or a separate owned allocation.
`Prefill::carve` at1199-1205 allocates the complete chunk before use. No new
alias is introduced; a contiguous token slice is the only accepted layout.

| Array | Token stride, F32 elements | T512 bytes | Role and alias contract |
| --- | ---: | ---: | --- |
| m.R | 10240 | 20971520 | owned residual; standalone write updates this same array |
| m.xn | 10240 | 20971520 | distinct normalized workspace, allocated because native HC forces gr_unfused |
| m.lo | 320 | 655360 | distinct down output, then pointwise silu in place |
| m.gated | 10240 | 20971520 | distinct up output |
| m.grs | 4 | 8192 | per-token/per-channel RMS scale |
| m.mixed | 2560 | 5242880 | distinct mixed output |
| m.inj | 4 | 8192 | distinct F32 injection output |
| m.bo | 2560 | 5242880 | downstream block result used by standalone residual write |

Weight norm/down/up/inject views are independently owned exact original
F32/Q8_0/Q8_0/F32 source storage, unchanged by native_hc_bind. The native branch
at2863-2884 assigns each pointer at token offset tb*stride, already covering the
whole chunk. New rows512 uses the same arrays and capacities. T>512 remains
sliced into explicit chunks<=512; it does not weaken the old API. Standalone
write at4373 uses the new prefill-only dispatcher. Native HC forces gr_unfused
at310, so the compatibility fused-write branches at4266-4276 are not selected.
No pointer-array, padded xn16, or half-storage reinterpretation is introduced.

## Arithmetic and fixture

One shared composition implementation retains each kernel lambda exactly;
old and new public entry points select descriptor/projection limits only.
Pending write retains sigmoid and FMA; norm retains lane d,d+32 accumulation,
XOR16/8/4/2/1, rsqrt, and the same multiply order. Tile2 Q8 down/up retain
per-output lane FMA column order and the same XOR tree. F32 injection uses the
original launch<false> arithmetic at up to512 rows. Silu and mix are unchanged.
The conversions to F16/BF16 at the downstream operator boundary remain outside
the composed read in their original locations. No GEMM or requantization route
is introduced. Old mode1 numerical boundaries stay unchanged.

The standalone fixture uses dense deterministic original Q8_0 rows and F32
injection, compares old scalar eight-row slices against the whole composed chain
at T1/7/8/9/512, both pending-applyFalse/True. It compares xn, post-silu lo, gate,
rs, mixed, inject, standalone written residual, plus pending residual when used.
T512/applyTrue records and replays the candidate graph after changing original
residual inputs, and compares all fields again. Old descriptor>8, new>512, and
short-workspace refusals are checked. It consumes no model data.

Owned buffers and heap graphs retire only after the original queue drains;
failed drain retains storage and exits86, failed free exits87. Host backing
vectors precede the owner in scope and outlive that drain. Temporary readback
vectors use completed copies or fatal retention on copy failure. This fixture
has not been compiled or run yet. Natural model output equality remains required
after operator parity; no tolerance changes or native reference substitution.

## Actual root operator follow-up

COMMAND -> NoDri standalone compile67564 PASS43s. Same binary first
card1 exposure/affinity1 attempt failed with an early UMD host segfault
(rc139); posthealth passed, failure preserved. Controlled card0 exposure/
affinity0 run66943 PASS42s.
RESULT -> Ten complete-chain cases and changed-input graph replay are
bitwise equal, including T512 pending residual application and write.
Owned exit0/removal, posthealth and fresh kernel log passed on card0.
ELF SHA2206ed42a32abcb66072d4c824c508ba7cb5814003328f2f67e9e17fe7ba5a17.
Report SHAfe3fbc98595b5919cf67033ca1998e5be9d8f0d3f1e7ce31dd6e6e8325c04a7f,
F17/hc-prefill-coalescing-operator-card0-control-v2/report.json.
VERDICT -> Standalone arithmetic accepted on card0. Card1 enumeration
cause is suspected, not proven. Fresh full engine/model gain still pending;
source-derived submission reductions are not timing measurements.
