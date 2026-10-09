# Original-source PLE fidelity increment

CONFIG -> Selected UD-Q4_K_XL, Strata pinned source plus patches0001..0004.
Patch0006 is independent of stage-mirror patch0005. Whole-native mode remains
STRATA_SYCL_NATIVE_HC=1, default-off and guarded by the existing strict policy.
The compatibility pack retains its disabled-path valueBF16 and convF16 images.

COMMAND -> Audit actual model inventory, packer, NativeDense, PleRun readiness,
generate wiring, scalar ple_block, prefill and verifier projection branches,
and native_ple_postops convolution. Prepare0006 against an independent temporary
source overlay. No external checkout, live backend, model weight or GPU changes.

RESULT -> Source audit found three PLE fidelity changes in the existing backend:

- Original ple_value isQ8_0[2560,2560],6963200 bytes. The compatibility image is
  BF16 and scalar/batched value kernels read it.
- Original ple_conv1d isF32[4,10240],163840 bytes. The loader narrows its image
  toF16; native postops still reads those narrowed values. The actual payload
  census found101 coefficients changed by that narrowing.
- Original ple_key isQ8_0[2560,10240],27852800 bytes. Existing native projection
  preserves weight bytes but scalar/verifier quantizes embedding activations
  toQ8_1; batched prefill converts activations toF16. It therefore does not satisfy
  the original-F32 activation contract.

Patch0006 adds typed original PLE views to WeightRef and PleWeights. NativeDense
owns original key/valueQ8 and convF32 device images on the PLE-owning stage, checks
source dtype, GGUF dimensions, byte bounds, duplicate ownership and completeness,
and prints source shard/offset/shape/byte/FNV64 receipts. PLE ownership must satisfy
both the explicit load bounds and the active static stage range. It reads only original
GGUF payload bytes; no new artifact quantization is used. Existing native source
and compatibility image ownership is preserved.

Generate requires all three exact source views before enabling source_exact.
PleRun readiness accepts those views without requiring BF16 value or F16 conv
pointers. The strict path bypasses the compatibility conv-type prerequisite.
The actual ple_exact_sources_ok header helper requires exact source capacities;
scalar dispatch also checks all required pointers, unsupported preprojected
inputs, and the explicit in-order queue before projection submission.

Both scalar projections reuse hc_q8_0_project_f32: exact FP16 scale times signed
int8 code, original F32 embedding, F32 FMA and fixed subgroup reduction. There
is no BF16/F16 embedding conversion or activation quantization on this path.
The source key/value dimensions meet the primitive's existing launch bounds.
If primitives are not compiled or a descriptor is rejected, execution throws;
it never switches to a compatibility projection.

Exact mode forces the already-existing per-token PLE routes in prompt and
verifier, disabling their lossy preprojected optimization branches. Ordinary
decode, prompt, and verifier thus share the same typed scalar dispatch. Existing
normalized-history append timing, row-fastest layout, previous-token state and
residual-write ordering are unchanged. This is a correctness-first dispatch;
no new batched implementation or performance claim is made.

Native postops reads originalF32 conv coefficients when source_exact is set.
It preserves GGML k-fastest flat indices k+4*channel, dilated taps9/6/3/0,
existing term-then-add ordering, SiLU and hidden+gated+activation order. It checks
and validates the originalF32 weight span instead of the F16 span. Disabled
mode retains its existing F16 kernel branch. Batch postops also gains the typed
F32 coefficient option, although exact model dispatch currently uses scalar
routes. Existing F32 norms, gate arithmetic and native exp remain unchanged;
their numerical accuracy requires composed GPU qualification.

VERDICT -> Patch applicability is verified after0001..0004. CPU checks exercise
its actual source-view header helper under ASan/UBSan, plus independent
FP64 scalar/batch convolution-history oracles at row counts1/2/4/8/9/16/17 and an
F16 coefficient negative control. These prove host contracts and reference
construction only. Source compilation and GPU numerical behavior remain
parent-controlled and unproven until receipts establish them.

Before model screening, require source-identity receipts and leased one-card
key/value projection parity against exact-source CPUFP64 references, conv taps
and normalized-history parity, all PLE intermediates, token/chunk transitions,
repeat/reset/history isolation, every prompt/decode/verifier route, and exact
source input bytes. Use the preregistered projection gates NMSE<=1e-6 and
normalized maximum error<=1e-4 for source projections; preregister composed
PLE/conv gates before execution. Include deliberate BF16 value, F16 conv and
Q8_1/F16 activation controls that change synthetic outputs. Post-health and
teardown are required; full model coherence, speed and two-device qualification
remain later gates. This patch alone does not qualify native serving.

CPU validation follow-up: the current0006 snapshot passes1 actual descriptor
acceptance,11 rejections under ASan/UBSan, and57 width10240 scalar/batch-history
oracle rows. Synthetic originalF32 conv coefficients differ from theirF16 control
for39179 values with a nonzero output difference. These counts describe synthetic
fixture evidence, not the actual model's101 changed conv coefficients. Patch0006
also applies cleanly after0001..0005. No SYCL compilation or GPU execution by the
fixture agent.
