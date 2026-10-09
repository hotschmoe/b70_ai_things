# Conditional original FFN checker v1

CONFIG -> New separate original-weight checker for 0023's complete33 layout.
Source/synthetic work only during parent GPU controls. No original model payload,
GPU, runtime, SDK build, commit or JOURNAL action. Frozen FP64/GDN/packet/reference
files remain unchanged. No actual new33 capture is available in this task.

COMMAND ->

```
python3 strata/flash-next/test_verify_layer0_ffn_original_v1.py
python3 -m py_compile strata/flash-next/verify_layer0_ffn_original_v1.py
```

RESULT -> Nineteen CPU controls PASS. Source/byte/nonce/ID/rank/tier/live-range and
DERIVED provenance controls plus conditional projection/aggregate/router/HC
perturbations are tested. Synthetic Q4_K/Q5_1 packed rows confirm exact selected
code-sum affine algebra. Actual original-weight FFN checks are UNOBSERVED.

VERDICT -> Source-ready conditional checker. Thresholds remain NMSE<=1e-6 and
normalized maximum<=1e-4, denominator floor1e-6; no widening from failure.
Future actual execution requires preserved four-prefix33-field captures, actual
merged producer logs and genuine current full4 original GGUF identity, after
parent controls terminate and source reads are authorized:

```
python3 strata/flash-next/verify_layer0_ffn_original_v1.py \
  --requests <actual-new33-candidate-on/requests.json> \
  --model-identity <genuine-current-full4-identity.json> \
  --output <new-conditional-FFN-original-receipt.json>
```

It refuses existing output and binds captures BEFORE original payload reads.
The exact fresh1/2/4/8 roster must share PID/source binding, have distinct request
ordinals and exclusive field files. Metadata SHA/content, actual SFD/L0 markers,
token/last-row/position/reuse boundaries, GPU nonce/key, all33 names/encodings/
extents/provenances, every field SHA and F32 finiteness are checked. Expert IDs
must be unique in0..511; entry rows are [ID,token0,router_rank]. Addresses and
tiers1VRAM/2own-host-mirror must match live3072000B source ranges in the owning
snapshot context at the same merged-log frame. The existing frozen range parser
is reexecuted and compared to saved binding, not trusted from a boolean. It
proves typed liveness, not original bytes by itself.

Each prefix has27 conditional original-weight checks:

- Shared gate/up original Q8_0 projections from verified FFN input Q8_1:2.
- Selected ten expert gate/up original Q4_K projections in actual router rank:20.
- Original F32 router weights rounded BF16-RNE, supplied actual F32 activation:1.
- FP64 softmax/selected normalization from supplied actual logits and IDs:1.
- Original Q5_1 expert and Q8_0 shared down from supplied actual HQ packets,
  actual routing weights, original shared-gate F32-to-BF16 seam and sigmoid,
  compared as weighted combined FFN output:1.
- Original FFN HC read/mixed and HC injection/write with supplied block and
  post-attention residual:2.

Derived shared/expert hidden versus supplied actual GU also uses the unchanged
packet-v3 numerical equation gate. HQ bytes must exactly match native Q8_1
encoding of those DERIVED fields. This does NOT prove the original fused raw
hidden. The two derived fields remain DERIVED_gpu_native_exp_from_actual_gate_up,
raw_fused_hidden_observed=false. All five packet families have SHA/size/source
binding; the FFN packet is independently encoded from actual supplied mixed F32.
This checker needs only that ordinary input family plus both supplied HQ families;
other packet operators remain covered by the separately frozen packet checker.

## Actual source findings and limits

Pinned consumed source is engine161219Z-n_v8o9lp/source (full source hashes in
layer0-conditional-ffn-original-source-plan-v1.json).

- sycl/src/kernels/cuda/shared_expert.dp.cpp:320-350 and :379-382 packetize
  shared GU/hidden and apply BF16-weight/F32-input scalar shared gate. The T1
  verifier calls shared_expert_multi from sycl/src/core/verify.cpp:1440-1454,
  with actual producer hooks. :1459-1464 supplies expert input packet. Fused
  hidden uses device native::exp; CPU FP64 math is not its bitwise emulation.
- sycl/src/kernels/cuda/iq_kernels.dp.cpp:447-470 and :559-563/:643-670
  apply selected Q4_K/Q5_1 affine minima to quantized activation code sums.
  Thus exact real decoded weights dot d_x*codes_x has the same algebra.
  Stored Q8_1 s is unused for these selected consumers; other affine formats
  are rejected. Native per-thread DP4A grouping, F32 multiplication and warp
  reduction are not reproduced by FP64 projection and are covered only by the
  fixed numerical error gate, never a bitwise claim.
- tools/iq_pack.py:74-79/:234-247 converts original F32 router/shared-gate
  weights to BF16-RNE; sycl/src/kernels/cuda/native_bf16.dp.cpp:59-105 uses
  F32 activation and ordered FMA/pair-stride/XOR reductions. Only the explicit
  BF16 weight seam is reproduced; native FMA/reducer order is not.
- sycl/src/kernels/cuda/native_router.dp.cpp:84-143 normalizes native-exp
  softmax, chooses top10 with lower-ID ties, and renormalizes selected weight
  using floor6.103515625e-5. The checker checks selected probabilities from
  supplied actual logits/IDs. Mathematical top10 IDs are reported as diagnostic
  only: native-exp/F32 ties can differ and native selector is not emulated.
- sycl/src/core/verify.cpp:1559-1585 chooses scalar/multi/gated combine routes
  and copies only final ffn_block_output. sycl/src/kernels/cuda/native_moe.dp.cpp
  distinguishes ordered scalar summation/FMA vector combination and separately
  rounded shared scaling. The checker uses FP64 mathematical combination.
- layer0_numerical_observer_v2.hpp:70-88 maps actual producer entry destinations,
  token, IDs, residency pointer and tier, then captures GU/HQ in router-rank order.
  :103-121 checks nonce/coverage and labels derived fields. Metadata does not
  claim raw fused hidden or model math.

There is NO individual expert down output, ungated/shared output, or scalar
shared-gate observation in the33-field contract. Their original mathematical
projections are recomputed from supplied HQ, but only the total FFN aggregate
can be compared. Aggregate cancellation can mask individual down errors: no
per-expert down qualification is emitted. The shared scalar-gate native-exp
value is not observed; its mathematical estimate is tested only through that
aggregate. HC injection is independently computed using exact original F32/Q8_0
HC source, not rounded BF16. HC reductions/exp are mathematical error gates.

All inputs/IDs/logits/weights/HQ/residuals are supplied, not owned prefix history.
Prefix2/4/8 earlier prefill state is not reconstructed. Upstream HC/GDN/full-layer
state, raw fused hidden, native intrinsic/reduction equivalence, per-expert down,
capture lifecycle and full-model math remain false. A future failed actual field
must be preserved/localized; no packet mismatch or numeric threshold is relaxed.
