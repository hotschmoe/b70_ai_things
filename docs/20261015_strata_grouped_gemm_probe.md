# S42: direct oneMKL grouped peer GEMM (2026-10-15)

CONFIG -> No serving change. The grouped GU call fails the declared bitwise
gate, with small differences consistent with a changed FP32 reduction tree.
Arithmetic accuracy remains unqualified. Keep152055a6/headpairON/depth2.

This uses the installed direct grouped half/half/float/float oneMKL API,
variable expert row counts and one group per expert. It differs from the
previously slower custom primary grouped-XMX implementation. Both paths use
the actual production expert dequantizer, gather, GEMM wrapper and SiLU;
the candidate adds no custom arithmetic kernel. The scalar route processes
experts in reverse through aliased input/output storage; grouped GU finishes
all reads before grouped down can overwrite that storage.

One compact wave covers16 real experts and all10 real quant format pairs.
Its synthetic route has1748 rows, largest641/smallest1, within4096 capacity.
Each expert has private GU/down dequant slots. GU/H activations use compact
shared wave buffers. Guarded shared-USM GU/down descriptors are distinct,
immutable and in the same context. Returned group events feed queue barriers
before consumers; timed paths have no per-expert host completion wait.

COMMAND -> Actual Opus5.5/Fable5.1 exact v2 reviews return READY, followed
by root checks of queue binding, byte-pointer layouts and export helpers.
One139s CPU-only offline build, no DRI/network or runtime edit. Actual
production archive members/symbols and the grouped oneMKL symbol are verified.
Binary SHA256
`64586e43c036d281eeabce0f6caf8642785436d7f04ac6c9e85524ada25d5f29`;
exact compiled/retained source SHA256
`f69603f275067b6675745c6a1bd53c1203a091327c16c892415a6d5fa1d4e425`.

One44s physical-card1/PCI44:00.0 window includes quiet CPU, pinned pre/post
health, export hashing and teardown. No unitrace or server. The single
process requests conditional warmed timing after all strict gates, avoiding
a second setup if it passes. Candidate GU must pass before SiLU, H before
down; final timed outputs would also be checked. Timing was never reached.

RESULT -> Full scalar GU/H/down completes. Grouped GU executes, then stops
at element0, one ULP different. Before comparison, all43 buffer guards,
metadata guards/bytes, unused tails and complete weights/source/IDs
immutability checks pass. Both GU arrays are finite. All compact operands,
locators, scalar stages and candidate GU are saved and hashed.

| Descriptive GU difference against scalar | Value |
| --- | ---: |
| Compared FP32 values | 2,237,440 |
| Values differing bitwise | 902,568 |
| Maximum absolute difference | 7.450580596923828e-8 |
| RMS absolute difference | 4.35628826568391e-9 |
| Relative L2 difference to scalar | 1.41750923323495e-7 |
| Median / p95 nonzero ULP difference | 2 / 23 |
| Maximum absolute scalar value | 0.1512281447649002 |

Scalar is not an independent mathematical reference. Magnitude and ULP
statistics support a rounding hypothesis; they do not exclude every subtle
operand or indexing error. Candidate SiLU/down, latency, graphs, production
IPC and serving behavior were not measured. The requested timing flag does
not mean timed samples exist. Oracle/wrappers exit5 for NUMERICAL_MISMATCH;
both health checks and export hashing exit0, container removal passes.
There is no crash, reset or multi-card collective.

Source arithmetic estimates340,794,192 explicit device-buffer bytes plus
the saved real blobs, guards and shared metadata. Root also verified the
GEMM wrapper's explicit32MiB workspace and2MiB scratch, beyond vendor
internal workspace. These are allocation accounting, not measured peak VRAM.

VERDICT -> Strict gate remains FAILED/unpromoted. Both actual models agree
to audit exact CPU reconstruction of the dequantized FP16 operands before
another grouped probe. Saved quantized blobs alone do not prove that a CPU
dequantizer rounds to the same FP16 bytes as this GPU path. No new epsilon,
operator-validity claim or speed result follows from these small differences.
Q6 decode collection takes priority; S45 is source-only operand feasibility.

Dev-loop correction: bitwise equality is appropriate for unchanged arithmetic
and for replay against the same algorithm. Different reduction trees need a
predeclared independent accuracy contract alongside strict memory/layout
checks. Do not repeatedly spend GPU windows stopping at an expected one-ULP
difference. Prior strict failures remain unchanged evidence. Useful timing
can share a numerically gated process; no extra model-server load is needed.

Raw root: `cachefix_20261014/peer-grouped-gemm-s42-prep/` under the shared
results root. See `s42-report.json`, `build-closure.json`,
`numerical-difference.json`, actual review events and
`run-clean/{closure.json,exports/export-sha256.json}`. The retained adapter
is `strata/flash-next/lowlevel_probe/grouped_peer_oracle.cpp`.

Frozen raw manifest `peer-grouped-gemm-s42-frozen-manifest.json`:88 files,
186,527,017 bytes, SHA256
`a93c367b7940baabcf11ccc40983565e6603e60727607914f5a9cdddf3712343`.
