# Owned prefill first-GDN storage model v1

CONFIG -> New independent source lane, preserving frozen original FP64 and
native_storage_first_gdn_estimate_v1.py. Earlier rows use actual SYCL prefill
operand/storage contracts; final row uses distinct verifier storage. Source and
synthetic tests only during parent controls. No actual GGUF payload, GPU/runtime,
SDK compilation, commit or JOURNAL operation.

COMMAND ->

```
python3 strata/flash-next/test_prefill_first_gdn_owned_storage_v1.py
python3 -m py_compile strata/flash-next/prefill_first_gdn_owned_storage_v1.py
```

RESULT -> Twelve CPU controls PASS. They distinguish prefill from decode
substitution, F16 weight AND input rounding, BF16 small-projection input versus
verifier F32, core-y versus gated-y16 semantics, state order/history/reset,
F32 state storage and strict overflow/unsupported-profile rejection. Fixture
Q8_0 values are encoded-real half scales times signed codes, not arbitrary F64
weights mislabeled quantized. Actual original-weight own-prefix runs UNOBSERVED.

VERDICT -> Implemented mathematical/storage model, not qualified native oracle.
No new threshold/pass claim is inferred. Returned original-own-state/full-layer/
full-model qualification flags remain false and tolerance_gate is None. Device
native-exp/log1p/rsqrt, FMA/XOR reduction and oneMKL GEMM accumulation/tiling are
explicitly unsupported. Actual BF16 GEMM device branch selection is UNOBSERVED.
This source increment closes the earlier decode-only substitution in its model;
it does not retroactively qualify frozen decode-estimate prefixes2/4/8.

## Audited actual source boundaries

Consumed source root and full fingerprints are in
prefill-first-gdn-owned-storage-source-plan-v1.json.

- sycl/src/prefill/prefill.cpp:2239-2266 uses original native embedding gather
  into F32 m.emb and :2820-2828 uses original F32/Q8_0 HC composition, followed
  by mixed F16 and BF16 operand images only at downstream boundaries.
  Selected original embedding is Q8_0: half scale times integer code is first
  represented F32. Embedding is broadcast to all four first-layer HC streams.
- :1939-1943 native_proj passes16-bit X to Gemm::native. :2914-2917 uses mixed_h
  for Q8_0 qkv/z and mixed_bf for BF16 alpha/beta. :2923-2926 passes recurrence
  y_h to the original Q8_0 output projection. These prefill rows do NOT consume
  decode Q8_1 activation packets.
- sycl/src/prefill/gemm.dp.cpp:939-950 dequantizes original native blocks to
  F16 scratch; :773-808 invokes half-input/half-weight GEMM with F32 compute and
  output. sycl/src/kernels/cuda/dequant_bf16.dp.cpp:33-37/:96-100 performs Q8_0
  F32 decoded product followed by half RNE. The model explicitly rounds through
  F32 then F16, retaining actual double-round order and rejecting overflow.
- tools/iq_pack.py retains original F32 alpha/beta through explicit BF16-RNE
  runtime weight conversion. kernels.dp.cpp:2045-2056 uses act16 for mixed_bf;
  the non-HIP act16 at :83 is BF16-RNE. Gemm::bf16:689-771 uses BF16 X/W with
  F32 output on its normal branch. :624-634 can select an alternate BF16-to-F16
  route from device properties/STRATA_BF16_TC; its actual runtime selection is
  not asserted here. BF16X2 low-part accumulation is not modeled and rejected.
- kernels.dp.cpp:434-447 first stores alpha+dt as F32, applies softplus using
  v>20 cutoff versus log1p(native_exp(v)), multiplies original F32 ssm_a, and
  sigmoid-gates beta. :450-465 walks channel-major oldest-first3-slot history,
  four F32 convolution taps and SiLU; :605-630 normalizes q/k with squared sum
  plus epsilon. Values/history retain F32 stores; CPU dots/transcendentals are
  mathematical estimates of those stages.
- :768-862 recurrence reads physical state[i,head,j], groups by head%key_heads,
  computes old-state k-dot then decay-scaled delta, updates each F32 state cell
  with FMA(g,old,k*delta), and reads out updated state scaled1/sqrt(S). The model
  separately rounds k*delta before its F32 update and reads its own updated
  F32 state. Actual partial-group/FMA reductions are not emulated.
- :1643-1661 default pipelined normalization leaves y as scaled CORE F32; the
  RMS/gamma/z-gated value is written ONLY y16 F16, which ssm_out consumes.
  Serial alternative :747-750 writes normalized y too. Merely setting
  STRATA_GDN_REC_HEADS=0 still enables the serial branch because presence is
  tested; the model rejects any presence, including empty string and zero.
  Alternative GDN head/key-head and BF16 modes remain fail-closed.
- sycl/src/kernels/hc_native_composition.cpp:31-56 consumes exact original HC
  norm/Q8_0/F32 projection weights, F32 normalized/intermediate storage, down
  division by4 before SiLU, and mean of normalized streams times sigmoid gate.
  HC uses sycl::exp, distinct from GDN native::exp; neither CPU estimate is a
  bitwise device intrinsic oracle. No BF16 or activation-Q8_1 is inserted into
  HC source projections.

## Independent state and limits

OwnedPrefillGdnStorage(provider).tokens(ids) accepts only token IDs, length1..8.
Each invocation starts independent zero convolution/recurrent state, reads own
embeddings, computes own first-layer HC attention input, walks all earlier rows
as prefill, then uses last-token verifier Q8_1 projections with original Q8_0
weights and F32 activation/BF16-weight small projections. It never imports
captured normalized qkv, logits, mixed inputs, states or histories. Prior state
is copied for diagnostics, not substituted from a capture. Final-row packet
hashes are tagged with token/position; earlier rows emit no Q8_1 packet events.
Logical recurrent state[head,i,j] maps to physical[i,head,j] by transpose(1,0,2);
logical convolution[tap,channel] maps to physical[channel,tap] by transpose.

Projection tiles are capped64MiB. Prefix length8 bounds trace/state copies;
values stored as F64 represent explicitly rounded F32/F16/BF16 stages. No actual
payload read occurs until provider rows are called in a separately authorized
future run. This is layer0 HC-attention/GDN only, not layer0 FFN or other layers.
Device GEMM/reduction/intrinsic errors and source-mode selection require separate
actual measurements. No own-prefix equality, model fidelity, coherence, cache,
concurrency, latency/stability or shelf verdict follows from these CPU controls.
