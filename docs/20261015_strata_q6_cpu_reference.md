# S43: independent full first-matrix Q6 reference (2026-10-15)

CONFIG -> Both saved Q6 C6 outputs satisfy a source-derived arithmetic
bound against an independent CPU reference. This is a bounded mathematical
diagnosis, not production qualification. S41's bitwise gate remains FAILED.
Serving winner152055a6/headpairON/depth2 is unchanged.

The target is the real dot product of stored Q6 weights and the actual
saved native Q8_1 operands. It does not compare with original FP32 input or
qualify quantization/model quality. Check all2560 rows x6 columns of the
existing6144->2560 fixture,15,360 outputs per path. No GPU rerun, new kernel,
SYCL build or serving process is required.

COMMAND -> Actual Opus5.5/Fable5.1 independently review the source-derived
contract and exact CPU checker. Both accept the CPU diagnosis and explicitly
reject production sufficiency. Opus requests dimension, row-coverage and
rational-spot pin checks; root adds these seven assertion/counting lines in
v2, preserving v1 and reviews. Arithmetic, operands and bounds do not change.
No redundant review round is run for those requested guard additions.

Exact executed checker SHA256
`7831f155e9bf03a1326b8885920c0d9caf9877994c540c5749354dabb124783d`;
pins SHA256
`e3ad41b292071606e98c809d98f0391893e521efa13117d15e17968d779cef29`.
One cpu-run lease, single-thread BLAS environment, no GPU libraries. Decode
128 rows at a time; match the full12,902,400-byte payload's original FNV
identity and all19 earlier saved row SHA256 hashes. Save a new whole-payload
SHA256, full reference/error/bound arrays and result JSON. Source/snapshot/
build hashes and actual S41 kernel names are checked before calculation.

The independent decoder follows storage layout, not either GPU lane tree.
FP64 matrix products compute the reference and sum of absolute products.
Decoded term products fit conservatively within44 significand bits and are
exact in FP64. All19 selected rows match the frozen scalar decoder; three
exact rational sums check reference error independently. Actual half scales
are finite; CPU round-to-nearest, original precise TU flags and AOT options
are checked. Unsafe/fast-math options are absent from the retained commands.
This establishes source/build assumptions, not a new device ISA inspection.

Let `A=sum(abs(w_i*x_i))`, `u=2^-24` and `gamma_L=L*u/(1-L*u)`.
Source operation depth gives `L=15` for current wideA2 and `L=18` for packed
at K6144. Counts include scaling, local sums, per-lane accumulation and
cross-warp/XOR reductions conservatively. Integer dot/scale products are
exact; finite-half operands lie on dyadic grids that rule out FP32
underflow/FTZ and overflow in these pipelines. FMA reduces the counted depth.

The FP64 reference allowance is `gamma_6143(2^-53)*A`. A positive FP64 norm
dot supplies an outward upper bound on A. Compare each GPU path separately
against its own `(gamma_L32+gamma_ref64)*A_upper`, with outward rounding of
coefficient, norm, bound and difference. A=0 requires exact numerical zero.
No empirical ULP cutoff or tolerance fitted to S41 differences is used.

RESULT -> CPU execution exits0 in1.9545s; process peak RSS54,452KiB. Explicit
array budget24MiB excludes BLAS workspace; measured RSS includes process
overhead. All15,360 outputs in each path meet their own bound.

| Full first C6 matrix | Original wideA2 | Packed |
| --- | ---: | ---: |
| Bound violations | 0 | 0 |
| Maximum absolute reference error | 6.42022e-7 | 7.52776e-7 |
| RMS reference error | 1.26268e-7 | 1.43039e-7 |
| Maximum error / allowed bound | 0.009982 | 0.009134 |
| Predeclared one-row output-shift violations | 15,360 | 15,360 |

The negative control changes only saved output row identity, against the
same reference and bounds. It demonstrates gross-error sensitivity, not
power against every small indexing bug. Conservative bounds are much larger
than observed errors. CPU rechecks finite active values and complete poison
tails; GPU guard/readonly evidence is inherited from S41. No new GPU health
or guard measurement is claimed. FP32 round-to-nearest is a precise-build
assumption here, not a newly read device rounding-state register.

VERDICT -> Continue the packed-path investigation with a separately reviewed
collection adapter and full host reference for the remaining columns,
three shapes and changed-input direct/graph cases. Cross-algorithm bit
differences can be recorded without aborting collection; same-algorithm
direct/graph equality, memory, finite and immutable-input gates stay strict.
Collection alone must finish UNVALIDATED, then host analysis decides its
numerical result. Promotion still requires relevant functional, concurrency,
residency, context and health evidence. No speed claim or absolute goal is
newly achieved by this CPU result.

Raw root: `cachefix_20261014/q6-dense-reference-s43-prep/` under the shared
results root. See `s43-report.json`, `bound-proof.md`, `bound-proposal.json`,
actual contract/checker review events, `root-run-decision.json`,
`v2/{check_first_c6.py,checker-pins.json}` and `run-first-c6/gate.json`.
The tracked copy is `strata/flash-next/lowlevel_probe/q6_cpu_reference/`;
its pins deliberately require the retained S41 evidence and source identity.

Frozen raw manifest `q6-dense-reference-s43-frozen-manifest.json`:38 files,
1,057,325 bytes, SHA256
`d40bd02eeb270a7eeeb87d00a3023fe83eb03cd13494ef1f861349076d69cfa9`.
