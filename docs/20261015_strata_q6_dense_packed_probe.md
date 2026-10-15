# S41: real Q6 dense packed-path gate (2026-10-15)

CONFIG -> The existing packed Q6 path fails the declared bitwise gate on
its first C6 comparison. Keep the serving winner152055a6/headpairON/depth2.
This is a deterministic-output mismatch, not evidence that either kernel
computes the wrong quantized dot product. There is no production change.

Current main dense weights contain71 Q6 tensors;52 meet the existing pack
limits. Extra packed planes would cost842,956,800 bytes (803.90625MiB).
That leaves974.094MiB of the earlier1778MiB primary-card headroom before
allocator/workspace margins. Packing before expert-cache sizing could change
the149/363 split. Peer-card headroom cannot pay primary-card allocations.
The existing environment option only registers certain head/MTP owners;
the main Q5 head does not activate it for main NativeDense Q6 projections.
No new loader hook, memory accounting or lifetime change was implemented.

S31 source/header attribution matches52 eligible C6 calls and3.569979ms
of instrumented service per cycle across four shapes. This ranks a lead;
it is neither a saving estimate nor an additive request-latency budget.

COMMAND -> Actual Opus5.5/Fable5.1 review the exact standalone adapter,
followed by narrow input-export reviews. One103s offline CPU-only AOT build
links the actual current production native MMVQ, quantizer and pack/unpack
archive member. No GPU device is exposed during compilation. Frozen binary
SHA256 `a09b4c321dd3b099fbcc696e4e007e40dc24834767e62524075ae965e2c96f74`;
compiled adapter SHA256
`0a76898275463f6b3ee3640e25869f24aa02f02cbc71fb8d6297711e16630ff3`.
The tracked `strata/flash-next/lowlevel_probe/q6_dense_oracle.cpp` is that
exact compiled source. Original v1 and all reviews remain in raw evidence.

One36s quiet-CPU/card1 window includes pinned pre/post health and teardown.
Physical PCI44:00.0 is explicitly checked. It uses the same DPCT context
for an in-order profiling queue and the existing packer's default queue,
with drains at ownership transitions. Original weights remain intact.
All four real tensor headers are checked before payload access. Original
columns6/1/2/3/4/8 run first; packed C6 is the first comparison and stops
immediately on mismatch. Whole outputs and actual FP32/native-Q8_1 inputs
are saved before comparison. Existing unitrace records the same numerical
process; no extra profiling process or model-server start is needed.

RESULT -> First fixture `blk.3.attn_output.weight`,6144->2560,
12,902,400 actual Q6 bytes. Six original outputs and one packed C6 output
are saved. C6 has15,360 active FP32 values;11,754 differ bitwise. Both paths
have finite active values, intact allocation guards and equal untouched
poison in the two unused rows. Quantized input is unchanged through the
failed comparison; original weights match before and after packing.

| Descriptive difference, packed versus original | Value |
| --- | ---: |
| Maximum absolute difference | 9.5367431640625e-7 |
| RMS absolute difference | 1.90544109383582e-7 |
| Relative L2 difference to original | 1.29859907527767e-7 |
| Median / p95 nonzero ULP difference | 2 / 24 |
| Maximum ULP difference | 9216 |
| Maximum absolute original output | 6.442998886108398 |

The maximum ULP occurs near zero; ULP and elementwise relative error alone
do not establish arithmetic accuracy. The original is not an independent
mathematical reference. Source shows different four/16-element integer-dot
grouping, floating-point multiplication grouping and reduction trees.

The trace proves one original C6 wideA2 and one packed C6 execution. Both
report128 registers, private/spill0; packed reports4608 bytes SLM per group
versus0. Original C8 reports5824 spill bytes and C4 reports256; main C6
does not spill. One cold profiled C6 service observation is363750 versus
70833ns. Original-first/packed-second order, instrumentation and n=1 make
this unsuitable for a speed claim, occupancy diagnosis or serving forecast.

No packed C1/2/3/4/8, later three payloads or graph replay executed. The
original C6 graph was captured before packing but was never replayed.
Full-success-only final input checks did not execute. Do not report24
passing pairs, four covered shapes or changed-input graph correctness.

Oracle/wrapper exit5 is the deliberate strict-mismatch result. Card1 health
passes before and after, the owned container is removed and leases close
normally. No hardware crash, reset or two-card collective is involved.

VERDICT -> Preserve the failed strict gate. Both actual models recommend
a bounded independent CPU reference before abandoning this path or defining
another numerical gate. Fable's suggested small ULP band is not accepted
as a criterion; any future error bound must follow the arithmetic rather
than fit these observed differences. There is no serving promotion,
clean latency result, full-context repeat or newly met absolute goal.

The CPU follow-up reads only19 selected Q6 rows (95,760 payload bytes),
using the saved native Q8_1 input. An independent scalar storage-format
decoder and FP64 `math.fsum` evaluate114 outputs. Three exact rational
sums independently match the rounded FP64 results. Rows are the declared
union of0/1/2559 and the top-eight absolute/ULP-difference rows, across
all six columns. This is an extreme-selected subset, not a random sample
or a whole-matrix qualification. CPU execution takes0.141s under cpu-run.

Against the quantized-operand reference, original maximum/RMS error is
5.78616e-7/1.77038e-7; packed is6.57094e-7/1.95475e-7. Original is closer
on59 outputs, packed on31, with24 ties. The near-zero maximum-ULP output
has a sum-of-absolute-products condition ratio of about634256. This supports
the rounding-order explanation on the selected outputs. It establishes
neither a whole-matrix arithmetic bound, original-FP32 quantization quality
nor acceptable downstream token changes. No new numerical policy is
approved or applied by this descriptive follow-up.

Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/q6-dense-packed-s41-prep/`.
Key files: `s41-report.json`, `build-closure.json`,
`numerical-difference.json`, `run-profiled/closure.json`, saved snapshots,
`unitrace.1.txt`, actual model review events and
`cpu-reference/{reference.py,reference.json,manifest.json}`. The arithmetic
follow-up reuses saved data; it requires no GPU rerun.

Frozen raw manifest `q6-dense-packed-s41-frozen-manifest.json` covers112
files/66,411,961 bytes, SHA256
`a0eb232ddedf801519f9dfda94045a4463b4c3c64d1f3e3e4ae4876fc26e993b`.
