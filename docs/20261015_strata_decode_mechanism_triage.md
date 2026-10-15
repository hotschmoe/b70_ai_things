# Decode mechanism triage, 2026-10-15

S69 holds a broad HC launch-fusion campaign. S72 identifies an existing,
distinct HC grouping mechanism worth one bounded current-code graph probe.
Neither audit measures latency or changes the production source.

## S69: HC launch boundaries

CONFIG -> N2560/HC4/LR320, T1/2/6, current HC common predicate. Native audit
author: `/root/strata_ep_s27`; root read the full findings and checked the
actual down/reduce/launch source. Historical S31's447 HC-family calls and
7.1990958ms service per cycle are context, not recoverable wall time.

COMMAND -> Read fused_gr, verifier/MTP consumers, residual postops and small
S31 operation records. No model payload, build, GPU or model consultation.

RESULT -> The ordinary read has norm,80-slice down, ascending reduction and
160-block UP stages. UP already combines SiLU, dot/butterfly, pending residual
write and gated mean. Norm needs the complete stream; reduction needs all80
partial producers; next norm needs all UP outputs. A local barrier cannot
replace these global dependencies. In-place residual ownership also matters.

Exact reduce-in-UP is possible in principle after the down kernel, but each
of160 UP groups needs all320 low-rank rows. It repeats320*80*T additions and
partial reads160 times: T6 gives24576000 versus153600 main-row additions/reads.
These are source counts, not physical traffic or a performance forecast.
Different-order V3 is a different algorithm, not mathematically invalid;
it needs a justified independent accuracy contract before testing.

An existing final-residual folding flag covers a small separate pool. S31
contains6 native post calls/.0115386ms service per cycle, not all independently
identified final writes. Algebra alone does not prove exact native/fused
FMA/signed-zero behavior. This does not justify a new tiny flag-only probe.

VERDICT -> HOLD bulk fusion and final-residual-only GPU work. Do not repeat
GRF, LO_ONCE, V3 or global QFUSE campaigns under another name.

Frozen evidence: `hc-launch-fusion-next-s69-prep/`, manifest SHA256
`eee30bc9b83fe00aea7ce174956e2b0525fb9d5aff69ea3ef53c512af5333e12`.
It binds findings, source pins and retained operation records.

## S72: existing MORE_GROUPS path

CONFIG -> Same current HC, fixed LARGE_GRF0/sliced1/normsplit1/V30,
HC-Q8/QFUSE absent. Native audit author: `/root/strata_ep_s27`; root read the
full findings and checked current MORE and ordinary launch templates.

COMMAND -> Inspect actual S15 treatment matrix/results, S24/S39 controls and
existing S39 ELF metadata. No new compilation, GPU use or source mutation.

RESULT -> S15 changed UP_COLS8/4, ILP and max-token unroll, not MORE_GROUPS.
UP8 alone was slower, median ratio1.1393877 across T1..6; that is a useful
caution but does not test the combined grouping path. S39 fixed MORE0 while
changing GRF. No B70 MORE treatment was found in the inspected evidence.
This bounded history finding does not claim nobody ever ran it.

MORE keeps four launches and changes three existing kernels:480 down groups
instead of80, eight partial loads followed by the same ascending additions,
and320 UP groups with8 columns instead of160 with16. Source preserves each
row's dot/XOR order and unique output ownership. Both current branches have
LO_ONCE. More groups duplicate activation staging and local low-rank work;
physical traffic, occupancy and latency remain unmeasured. A770 source comments
are not B70 measurements. MORE is a bundled mechanism, not isolated down tuning.

Six MORE GRF0/1 kernel images already appear in the old S39 executable, but
that executable asserts MORE0 and predates current HC common. It cannot be
reused as the current test. S60's common MMVQ/IQ/core device wrapper also lacks
HC; a new adapter needs a supported additional prelink of the current frozen
HC fat object. No production kernel-TU rebuild should be necessary; actual
link/runtime evidence must establish that rather than assuming it.

VERDICT -> Prepare one current-HC strict graph comparison, fresh MORE0/1
processes with GRF0 fixed. Require full initial/changed direct/graph output,
guards/immutability and actual capture bindings before interpreting timings.
No server toggle, broad sweep, promotion or speed claim from this source audit.
The separately numbered S73 preparation must pass exact-source review first.

Frozen evidence: `hc-more-groups-next-s72-prep/`, manifest SHA256
`898eb5d5bc6e21b85fad9217c5f7ec9378ea1c85b93dee524143f9bc9f6272ff`.
It binds the prior matrix/result subset, exact source and six AOT identities.

Both evidence directories are under
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
Current fused_gr source SHA256:
`d62bcb252eecdbcaae81095907c2241e39c4798b7854bb499fb5643fecb24e99`.
These audits preserve the historical170-operation map and existing qualified
RAM recovery. Absolute serving speed goals remain unmet.

## S70: independent projections sharing canonical Q8

CONFIG -> Main48-layer model, historical S31 C6 operation records, current
verify/MTP/shared/native dispatch. Native audit author: `/root/ple_upstream`;
root read the full findings and checked the GDN and shared-expert call sites.

COMMAND -> Join the existing GGUF tensor directory to source/format/shape
counts; read metadata only, seeking past large arrays. No tensor payload,
build, GPU, new test or model consultation.

RESULT -> Shared gate/up have48 same-input pairs:17 same-format (9 IQ3_S,
6 IQ4_XS,2 Q4_K) and31 mixed-format. GDN QKV/attn_gate have36 pairs:9 same
and27 mixed. Canonical Q8 is already shared; no duplicate quantization is
removed by pairing. The current native pair hook accepts only Q8_0, so it
does not fuse these actual main tensors, even with its caller flag enabled.
Forty IQ3_S640-row calls do not imply20 eligible same-type pairs: only18
belong to9 such pairs; the other partners use different formats.

A group-uniform union of the two original row grids could select separate
weight/output pointers and retain each format's own row arithmetic, FP order,
Q8 input and output stride. This needs no concatenated allocation, new queue
or requantization. File adjacency does not imply device-pointer adjacency.
Shared GU640+640 would use320 groups instead of two160-group launches.
GDN10240+6144 would use4096 groups instead of2560+1536. Compiler resources
and occupancy remain unmeasured. Shared GU retains its existing outer stream
fork/join. GDN pairing moves the independent gate earlier but delays conv
until the whole combined kernel completes; benefit is not guaranteed.

Historical enclosing service pools are96 shared-GU calls/1.2901684ms and
72 GDN QKV/gate calls/5.1845746ms per cycle. These joins do not label each
trace event by layer or establish same-type-subset service/savable wall time.
The two projections use different weights; there is no shared weight decode.
MTP embedding/hidden projections have different normalized inputs and mutable
Q8 scratch, so this same-input mechanism does not transfer to that pair.

VERDICT -> Concrete source sketch only; park implementation until the HC
screen. A future oracle needs separate public anchors for both outputs,
changed inputs, exact branch-specific arithmetic/strides and graph/memory
checks. Do not revive Q4 ownership, asynchronous queues, grouped-XMX, global
QFUSE or rejected shared-stream changes. No speed forecast follows.

Frozen evidence: `dense-independent-projections-s70-prep/`, manifest SHA256
`9bc50b979d5c457ed847911e60920e4af519327f3f50d3df243622a3f05259f5`.
Exact layer lists, format counts, source/metadata pins and reasoning are in
the two bound files, findings.md and pins.json.

## S71: canonical-Q8 INT8 matrix algebra

CONFIG -> IQ3_S C6, the two S65 shapes and current Split21 dense path.
Native audit author: `/root/mtp_q4_share`; root read all findings and checked
Split21's load/apply source. No build, GPU, weight payload or model review.

COMMAND -> Read actual quantization/dispatch, local matrix headers in the
pinned offline compiler image and the existing INT8 benchmark source.

RESULT -> IQ3_S's signed grid codes fit INT8. Each odd local scale applies
to32 values, exactly aligning with canonical Q8_1 blocks. An INT8 K32 dot
can therefore consume unchanged canonical activation bytes. Keep the odd
scale in INT32 after the dot: multiplying it into the signed code first can
reach465 and exceed INT8. Canonical integer sums including that scale are
bounded by1889760, below INT32 overflow and exact FP32 integer conversion.
Accumulate separately for each scaled K32 block, rather than across scales.

A proposed8-token/16-output/K32 tile pads C6 with two zero rows:25 percent
of issued output cells are padding. It needs neither FP16 activations nor a
persistent unpacked weight plane. Generic installed headers expose the
INT8/INT32 matrix shape, but do not prove B70 compilation or runtime support.
The existing benchmark skips C6 stores and uses another scale/coordinate
scheme; it is not a ready numerical oracle.

Native accumulation uses32 lane partial sums and an XOR tree. Sequential
matrix-dot epilogues would change FP32 order despite exact integer dots.
Preserving the tree needs an explicit storage/ownership design; changing it
needs an independent predeclared accuracy contract. Neither source algebra
nor INT32 equality establishes full output correctness or speed. Q6's per16
scales are excluded; this is not the rejected packed-Q6 or FP16-XMX path.

VERDICT -> Source feasibility GO, implementation/GPU held pending HC work.
The cheapest prerequisite is a tiny compile-only matrix capability canary;
it establishes compilation only. Root separately authorizes S74 source
preparation, with no runtime or performance claim. A later runtime integer
dot check would need an independent scalar INT64 reference before any full
quantized projection work. No serving/roofline forecast follows.

Frozen source findings and header records: `int8-xmx-dense-next-s71-prep/`.
External manifest `int8-xmx-dense-next-s71-manifest.json`,4 files/60268 bytes,
SHA256 `47584226a77b04280712d0fa0131193be1beb6f7ccb58c1e26e02f36aa42a811`.
