# Native-storage own-prefix layer0 audit v1

CONFIG -> Frozen 21 actual consumed SYCL source, selected GGUF original tensor
metadata, frozen original FP64 reference retained. Pair numerical controls are
active; this increment performs no original model payload reads, GPU execution,
SDK compilation, source modifications, commit or JOURNAL update.

COMMAND ->

```
python3 strata/flash-next/test_native_storage_first_gdn_estimate_v1.py
python3 -m py_compile strata/flash-next/native_storage_first_gdn_estimate_v1.py
```

RESULT -> Nine CPU fixture controls PASS. A separate own-prefix implementation
starts from each prefix's embeddings and empty convolution/recurrent state. It
propagates its own history sequentially, emits token/position-bound native Q8_1
packet hashes at dense/shared/expert boundaries, applies F32 storage and original
F32-to-BF16-RNE weight seams, and never imports captured activations or state.
Actual original-weight own-prefix execution is UNOBSERVED. Actual prefix 2/4/8
earlier rows use prefill, not repeated verifier decode. tokens() rejects these
prefixes by default. Explicit allow_decode_surrogate=True is available only for
controlled CPU work and sets captured_prefix_history_comparison_admissible=false.

VERDICT -> Concrete storage-estimation increment, not an exact native arithmetic
oracle or qualified layer/model result. Its name and output lane are
native_storage_estimate_unqualified. It has no numerical pass threshold/API;
full-model, complete-layer and original own-state qualification remain false.
The earlier conditional supplied-state PASS28 remains a separate result.

## Exact consumer findings

The consumed source root is pinned in native-storage-own-prefix-source-plan-v1.json.
All paths below are relative to that root.

- sycl/src/core/layer.cpp:98-105 and :318-327 route native alpha/beta BF16
  projections with F32 activation, rather than activation BF16 rounding.
  :402-413 applies the same contract to router projection. Original selected
  alpha/beta/router/shared-gate tensors are F32; tools/iq_pack.py:63-80 and
  :234-247 convert their storage to BF16-RNE with --compat-bf16. Native HC
  original owners supersede the pack's HC conversion; do not apply BF16 to
  original HC norm, injection or Q8_0 down/up.
- sycl/src/kernels/cuda/native_bf16.dp.cpp:59-105 consumes BF16 weight pairs,
  F32 inputs and two ordered FMA operations. :521-532 chooses MMVF block size;
  each thread has its own pair stride, XOR reductions and partial-block sum.
  FP64 matrix multiplication plus an F32 output store is not this reducer.
- sycl/src/kernels/cuda/native_mmvq.dp.cpp:192-208 produces native Q8_1
  half-d, half-s and signed-code storage. The independently implemented frozen
  packet contract preserves the raw F32 XOR sum/division/rounding convention.
  :899-910 consumes original Q8_0 as d_w*d_x*integer-dot; s is unused.
- sycl/src/kernels/cuda/shared_expert.dp.cpp:320-350 and :537-547 packetize
  the input and fused SwiGLU hidden before down projection. The raw hidden in
  frozen 21 remains UNOBSERVED; estimated hidden is never relabelled captured.
- sycl/src/kernels/cuda/iq_kernels.dp.cpp:447-470 computes selected Q4_K
  affine minimum from integer code sums times d_x. :559-563 and :643-670 do
  the same for selected Q5_1 down. This Q5_1 expression deliberately departs
  from upstream CUDA's stored-s convention. For the selected layer0 formats,
  original real decoded weights dot d*codes has the same algebra, though not
  the same F32 integer-dot partition, multiplication order or reduction.
  Q5_0 and other format consumers can use stored s; they are rejected by the
  increment rather than assumed equivalent. Layer0 expert gate/up are Q4_K,
  down Q5_1; dense/shared Q8_0 consumer formats are explicitly admitted.
- sycl/src/kernels/hc_native_projection.cpp:9-45 uses original Q8_0/F32
  weights against F32 input, lane-strided accumulation and XOR sum. HC is not
  an ordinary Q8_1 consumer. HC composition uses sycl::exp, distinct from the
  native::exp expressions in GDN/expert kernels; neither is assumed bitwise
  equal to the CPU estimate.
- sycl/src/kernels/cuda/native_gdn_preprocess.dp.cpp:73-96 stores channel-major
  oldest-first convolution history; Q/K L2 uses sum plus float32 epsilon.
  Recurrent output requires query scale after readout. The increment owns
  logical [head,i,j] state, and can later compare captured physical [i,head,j]
  only through the explicit transpose, never by using that captured input.
- sycl/src/kernels/cuda/native_moe.dp.cpp:55-72 and :112-123 distinguish
  sequential scalar combines from explicit rank-ordered FMA vector combines.
  FP64 weighted summation followed by F32 storage does not emulate these.

- sycl/src/prefill/prefill.cpp:2827-2828 creates F16 mixed and BF16 mixed
  images even under original native HC. :2914-2917 uses F16 native_proj for
  Q8_0 qkv/z and BF16 bf16_proj for alpha/beta; :2923-2926 writes recurrence
  y_h and consumes it for output projection. native_proj at :1939-1943 passes
  16-bit X to Gemm::native, not the single-token Q8_1 consumer. The actual SYCL
  route in sycl/src/prefill/gemm.dp.cpp:939-953 dequantizes original weights
  to F16 scratch and invokes F16 GEMM. Its HIP-only MMQ branch cannot qualify
  actual SYCL prefill. Thus prefix2/4/8 state depends on distinct F16 weight and
  F16/BF16 activation seams and GEMM reductions; exact scope is unsupported.

## Increment and unresolved gates

native_storage_first_gdn_estimate_v1.py reuses frozen role/shape checks and
original decoders as dependencies, preserving their files. Only original
F32-to-BF16 small projection roles are rounded. Its selected packet-consuming
projections quantize each own F32 activation with the independent native packet
encoder. Original Q4_K/Q5_1 affine minimum semantics are retained via exact real
weight algebra. HC projections use direct F32 input with original weights.
It owns prefix histories from zero and stores estimated state and activations
as F32. CPU exponential, FP64 dot/recurrence, HC normalization, router selection
and combine reductions are explicitly approximations, not substituted native
arithmetic proofs. The inherited mathematical router uses deterministic lower-ID
ties and FP64 softmax; actual native top-k/reduction parity remains unsupported.
The approximate recurrence reads out from its FP64 updated state before its
F32 store; exact kernel intra-step rounding remains a specific unresolved gap.

Before declaring an own-prefix oracle: implement and independently test actual
Q8_0/Q4_K/Q5_1 partial-dot grouping plus F32 reductions; exact BF16 MMVF ordered
FMA and block selection; GDN recurrence/normalization stores and reduction;
HC lane reduction; native router tie/renormalization and ranked FMA combination.
Native-exp implementation cannot be claimed from NumPy exp. A bounded operator
or ULP/error contract must first be measured with matched source and hardware;
no whole-prefix tolerance is inferred or widened from conditional PASS28.
Actual checkpoints must include all earlier prefix inputs, each generated
packet, own incoming state, original tensor identity and stored output fields.
After a separately implemented actual-prefill lane exists, compare own prefix
1/2/4/8 without resetting each token or substituting any
observed intermediate. Producer23 derived-hidden metadata is separate from
raw-observed coverage. Exact full layer/model math remains false until the
remaining producer/consumer and independent own-state gates actually pass.
