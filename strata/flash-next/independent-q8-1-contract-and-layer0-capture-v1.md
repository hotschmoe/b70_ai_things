# Independent Q8_1 activation contract and layer0 capture draft

CONFIG -> NEW CPU-only packet reference. Actual frozen18 pzbw4jgp and combined20
4uag6j5x packer sources are pinned independently in the plan. No Strata arithmetic
is imported by the encoder, no GPU, original model payload, whole-source scan or
frozen source edits. Existing source/native math evidence is not expanded by
these CPU tests. Full-model original fidelity remains unqualified.

COMMAND -> python3 strata/flash-next/test_independent_q8_1_activation_cpu_v1.py.
The test compiles a small CPU-only official GGML encoder probe in image39992d70,
with networknone/no devices, and links the frozen official CPU base archive.

RESULT ->28 synthetic profile/shape cases, tokens1/2/4/8 and widths640/2560/6144,
match an independent scalar struct/F32 oracle byte exactly.21 eligible cases (including the division-boundary witness)
match compiled official GGML integer-code-sum packets under the OFFICIAL variant,
not under the different native variant. Sixteen rival/rejection controls cover
half-away versus half-even, rawsum versus code-sum, divide versus reciprocal,
QFUSE unclamped semantics, changed packet, wrong extent/shape, nonfinite input
and nonzero-scale underflow. A concrete division witness uses amax31.6876964569,
x3.8673961163: native code16, official code15. No GPU packet was observed.

## Exact declared packet contracts

Input is raw little-endian F32, contiguous row-major, complete32-element blocks.
Output per block is36 bytes: LE FP16 d, LE FP16 s, then32 signed int8 codes.
The bytes, row/column count, source/provenance and contract label must accompany
any capture. An accidental F64->F32 conversion is not an accepted input identity.

NATIVE: amax over original32 F32 values; d32=F32(amax/127), finite clamp to65504;
codes=round-away(F32(x/d32)), clamp[-127,127], zero block codes0; d stored FP16 RNE.
s is the ORIGINAL raw F32 XOR-tree sum at offsets16/8/4/2/1, clamped+-65504 and
stored FP16 RNE. Codes divide by d32, not its FP16-rounded stored value. SIMD
lane/block order and no FMA contraction into the first sum are part of identity.

OFFICIAL: compiled ggml quantize_row_q8_1_ref uses F32 reciprocal1/d32 and F32
multiply x*reciprocal before round-away. s=F32(integer sum of codes*d32), then
FP16 storage, without native finite clamps. Native and official differ in both
code arithmetic and s semantics, even on finite ordinary data. They must never
share an unlabeled expectation. Packet consumers can use s in affine/minimum
corrections; reconstructed d*code alone does not describe every vec-dot contract.

GDN_QFUSE: consumed verify_kernels gdn_q8_1_store has rawsum/F32 division but
omits native finite clamp on d/s. Ordinary-range blocks match NATIVE; a saturated
sum becomes FP16 infinity while NATIVE stays finite. It is a separate contract,
implemented explicitly by label and EXCLUDED from current QFUSE0 serving probes.
Native HC itself never supplies QFUSE images in this source.

Nonfinite inputs, NaN reduction or nonzero amax with underflowed d32 are rejected
by the finite CPU reference. Source debug-marker/cast behavior is not qualified
by that rejection. Raw F32 sums can overflow even with finite inputs. The test
reports native finite clamp behavior separately; no claim of all possible F32
patterns or actual hardware conversion is made. Official half overflow remains
visible, never reclassified as native finite behavior.

## Actual consumed routes and precision seams

- Ordinary decode native_mmvq.dp.cpp:190-227 uses NATIVE; HC-down/up Q8 projections
  in hc_native_projection.cpp consume F32 directly without Q8_1 activations.
  Verifier native HC returnsfalse for QFUSE, then explicitly packs mixed input
  before ordinary projections. Its injection matrix remains original F32.
- Expert input / fused hidden iq_kernels.dp.cpp:2279-2370 uses identical NATIVE
  q8_1_store. Fused SwiGLU uses native::exp and does not write raw hiddenF32. Shared
  native path shared_expert.dp.cpp:320-352 uses native_mmvq input and fused hidden
  NATIVE packing. CPU std::exp cannot provide a byte-exact hardware hidden value.
- QFUSE gdn output verify_kernels.dp.cpp:298-328 is unclamped; launch plan requires
  STRATA_QFUSE=0. A future changed flag must use its own preregistration/GPU gate.
- Prefill nativeHC outputs original F32 mixed then creates F16/BF16 views at
  prefill.cpp:2824-2825. Ordinary native_proj at1939 and2914 consumes F16 views;
  Gemm::native gemm.dp.cpp:899-953 narrows original dequantized weights to F16 and
  runs F16 GEMM on SYCL (MMQ arm here is HIP-only). This preserves original packed
  owners while introducing BOTH weight and activation storage rounding, distinct
  from decode Q8_1. Alpha/beta/router use BF16(+optional low-part) projections.
  Their flags/views/weight arithmetic must be captured; they are not magically
  qualified by native decode packet identity. No alternate pack or flag is added.

Plan contains both generation hashes and a per-file equality check:9 of11
audited files are byte-identical; verifier/prefill wrapper files differ and
their actual packer callsites were read in each generation. This is not a claim
that the whole engines or state lifecycle match. The audit uses this evidence rather than
assuming patch20 scope implies unchanged math. Exact source line names above
refer to combined20; source-generation discrepancies remain explicit.

## Preregistered packet and quantization comparisons

Supplied F32 source -> independently encoded actual NATIVE packet must match
EVERY byte, including d/s, code order, row boundaries and strides. Negative
corruption of code, scale, sum or row/shape must reject before accepting a packet.
For fused hidden, raw original hidden is UNOBSERVED. A diagnostic GPU observer
may reconstruct the same contract(off) expression/native-exp from captured
actual gate/up into a DERIVED buffer; encode that F32 buffer independently and
compare actual HQ bytes, while separately checking the derived SiLU against
FP64 math. This retains honest derived coverage and does not switch kernel mode.

Activation reconstruction NMSE /max-normalized error are quantization cost,
reported separately and not an implementation-error acceptance threshold. Cost
scope excludes affine stored-sum corrections. Conditional projection/reference
math must independently consume the verified actual packets and original weights
including every affine correction before its existing NMSE1e-6/maxnorm1e-4 gate.
Complete original own-routing/state FP64 results must remain separate from
conditional native-input operator expectations; no native input/state injection
may be labeled an independent end-to-end baseline. Prefill adds separate F16
weight/activation and BF16 seam costs. Full native-storage propagation remains
unfinished; no full original-model accumulated-error gate is inferred here.

## Minimal default-off layer0 hook plan (not integrated)

layer0_capture_layout_draft_v1.hpp is a pure CPU layout draft, not a model patch.
One final T1 verifier row for each prefix1/2/4/8;33 named fields include HC input /
intermediates, ordinary raw input and packet, incoming/outgoing GDN state+conv,
GDN projections/gated output+packet, route IDs/logits/weights, shared/expert gate /
up/HQ and explicitly DERIVED hidden, and final layer0 residual. Four7023048-byte frames total28092192 bytes, below64MiB. Original sources/weights are bound, not dumped into the snapshot.

A future NEW patch against exact20 must default STRATA_LAYER0_Q8_DIAG off, make
no diagnostic allocations/copies/graph variants off, and reserve one explicit
snapshot owned by the stage containing global layer0. Bind copies to actual cs
queue/context/device; keep queue/residency guards. Copy each input/packet before
its scratch overwrite, GDN incoming state before recurrence and outgoing state
before reuse. Capture each resident/mirror expert entry by actual ent_tok/dst/
expert mapping; never treat scratch position as expert ID. Insert derived hidden
observer only in its own diagnostic graph with the actual hardware intrinsic.
Raw fused hidden remains UNOBSERVED in metadata and checker.

Use a distinct bounded T1 diagnostic graph for selected four armed raw GEN
prefixes, same source/native math/flags as the ordinary graph. Existing ordinary
prefill produces prefixL-1 state; the first prediction consumes positionL-1.
Captures are local operator views of that final row and whole layer0 outgoing
state, not observations of every earlier prefill token/operator. First-layer CPU
baseline computes allL inputs with its OWN state/routes; missing independently
propagated prefill/native-storage expectations remain unqualified. A later
prefill snapshot extension must explicitly bind F16/BF16 views and weight casts.

Readback/file output only after successful existing window completion; no new
barrier inside model math or timing claim. Trace request/GEN token IDs/absolute
positions, stage ranges, source/runtime/binary/patch identities, phase/shape/
provenance and skips. Cancelled/incomplete requests cannot publish a pass. Graph /
snapshot destruction must occur in original owning context after all replay
users retire, with chronological owner/frees and pre/post health. Require same
new binary diag-off/on output/raw-logit/state equality before interpreting it.
This layout neither implements those hooks nor establishes GPU lifecycle.

VERDICT -> Independent packet CPU contract and minimal capture source plan are
ready; original/native activation costs are distinct, and QFUSE/prefill seams
are explicit. Actual packet hardware comparison, new observer build/off-on /
lifecycle gates and original own-state full math remain parent-owned future work.
