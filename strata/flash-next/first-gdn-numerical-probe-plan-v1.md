# First-layer numerical prefix probes and matched capture contract

CONFIG -> Source-only preparation, prefix lengths1/2/4/8, explicit GEN token IDs,
empty owned state, layer0 original_fp64 CPU reference versus the native full48
layer forward. No actual reference/model payload execution, GPU, source scan or
new engine capture patch. Minimal raw GEN prefixes are numerical probes; they
are not natural API completions, serving quality or latency measurements.

COMMAND -> python3 strata/flash-next/test_first_gdn_numeric_probe_cpu_v1.py.
Future CPU-only plan preparation accepts an eight-integer JSON GEN roster:

```sh
python3 strata/flash-next/prepare_first_gdn_numeric_probe_v1.py \
  --gen-ids EXACT_GEN_IDS.json \
  --reference-plan strata/flash-next/original-first-gdn-layer-reference-plan-v1.json \
  --output NEW_NUMERICAL_PREFIX_PLAN.json
```

RESULT -> Synthetic prefix metadata covers1/2/4/8 and rejects18 identity/token/
position/state/layout/phase/prerequisite mutations. Preparation always sets
actual_execution_allowed=false. Metadata admission returns operator_math_pass
false; claimed contract flags cannot substitute for independently pinned raw
packet/operation evidence. This driver prepares/admit-checks contracts only,
and does not launch a backend or execute the CPU layer. The layer composition
module supplies the future own-state CPU evaluator after parent admission.

## Matched execution and snapshot boundaries

Each arm must bind original model lock/inventory, tokenizer metadata/chat
template, exact accepted GEN IDs (no independent rerender), positions0..L-1,
current pre/post whole-shard publisher hash receipts and original sentinel,
actual runtime image and package/library identities, engine source/patch/build/
binary receipt, stage range and precision/kernel flags. A future capture build
must pass its own ordinary source-upload/math/lifecycle gates and observer off/on
equality before diagnostic attribution. This source plan does not qualify that
new build or authorize launches.

Native raw GEN computes ALL48 layers. CPU computes ONLY layer0 for each of its
own embedding inputs, own router/expert choices and owned GDN/conv state. Compare
its final layer0 output to native layer0 after FFN HC write at positionL-1.
The final input token may run as a T1 verifier window after the priorL-1 prefill
tokens: record that seam and exact native batch/chunk/shape schedule. The CPU
reference must advance through allL input tokens from empty state. Never feed
native intermediate activations/state/routes into the original end-to-end
layer0 baseline to hide divergence. Later-layer/head agreement remains absent.

Native recurrent state physical[128,48,128] maps explicitly to CPU logical
[48,128,128]. Native convolution state physical[10240,3] maps to logical
[3,10240]. Capture labels must specify before/after layer0 and before/after each
token. Prefix-only existing residual observation does not prove those internal
states; a new bounded capture route remains unimplemented in this preparation.

## Required bounded numerical observations

Observe exact original tensor source identities plus raw F32 residuals after
both HC writes, HC normalized/down/SiLU/gate/injection/mixed, qkv/z/alpha/beta,
convolution raw/SiLU, normalized q/k/v, decay/beta, recurrent output and updated
state, gated/normed mixer output, router full512 logits and selected IDs/weights,
shared projections/gate and every selected expert's gate/up/actual HQ packet/
down, combined block. Native fused hiddenF32 may be UNOBSERVED: record its actual
packet and declare any GPU reconstruction DERIVED, never read unwritten scratch
or claim raw observation. Capture production input/output Q8_1 packets at each
quantized projection consumer. Keep last-layer0 residual plus per-token state /
operator snapshots under64MiB raw capture; choose a separate explicit subset if
complete counts exceed this bound. Missing stages remain unqualified.

The CPU layer evaluator keeps decoded source tiles<=64MiB and one-layer decoded
workspaces, gathers only its own selected experts and retains at most eight
owned states. Future49-token useful-prompt comparison needs a separately bounded
streaming state/output/cache plan; do not retain49 full state dumps or decode the
whole expert bank through this eight-token preparation.

## Preregistered comparisons and prerequisite gates

1. Identity, layout, token/position, source bytes, independent quantizer packets
   and conditional native-contract router IDs are exact checks. Packet reference
   must independently audit actual native Q8_1 round-away/clamp, FP16scale and
   sum semantics. Official GGML quantize_row_q8_1_ref's sum*scale is not presumed
   equal to native original-F32 warp sum. Weight decoding must retain the
   official C /Python /independent decoder dual checks.
2. Operator implementation error uses independently decoded ORIGINAL weights
   and the independently verified ACTUAL native input/storage packets. For HC/
   PLE projection gates retain NMSE<=1e-6 and maximum error normalized by
   max(1e-6,max_abs_reference)<=1e-4. Proposed first-layer linear/GDN/FFN
   conditional operators use the same preregistered limits, finite outputs and
   unchanged sources. Cancellation-sensitive dot comparisons additionally
   report error versus sum(abs(terms)); no relative-to-small-result replacement
   silently loosens the normalized maximum gate. Exact router comparisons bind
   the same conditional native input; ambiguous original-versus-native route
   changes are explicitly declared quantization outcomes, not hidden.
3. Activation/storage cost compares the independent native-packet expectation
   with the independent ORIGINAL F32-input /FP64 math expectation. Report NMSE,
   normalized maximum, per-stage vector differences and route/margin changes
   separately. Those cost reports do not replace strict implementation gates.
   A complete independently propagated native storage lane remains TODO; no
   one-layer or full-model accumulated original-math acceptance threshold is
   qualified or inferred from synthetic tests. Record whole layer0 FP64-versus-
   native residual/state differences as reported-only until that lane exists.

No acceptance from compilation, capture metadata, equality between two native
runs, a reconstructed buffer or readable outputs alone. Negative controls must
reject altered tensor code/scale/index, misplaced state, wrong temporal order,
wrong head map, overwritten selected packet, missing observation and deliberate
numeric perturbation before any successful GPU qualification is reported.

VERDICT -> Concrete source-only prefix and capture prerequisite plan is ready.
Native Q8_1 packet expectation, exact operation/storage lane and capture source
are required next; they are not implemented or GPU-qualified by this plan.
