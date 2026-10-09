# Conditional original GDN/projection checker v1

CONFIG -> CPU-only source preparation against the frozen 0021 capture contract.
The supplied four-prefix capture remains the preserved, failed lifecycle run at
f08-20261009/layer0-numerical-onecard-v1. Its lifecycle failure is not repaired
or reclassified by these checks. Original GGUF payload reads were deferred while
the parent GPU rerun was active.

COMMAND ->

```
python3 strata/flash-next/test_verify_layer0_gdn_original_v1.py
python3 -m py_compile strata/flash-next/verify_layer0_gdn_original_v1.py
```

RESULT -> 18 CPU controls PASS. Four actual captured prefixes passed the separate
read-only metadata/frame/nonce/packet binding check. No original model payload
was read and no actual original-weight numerical result is claimed. The external
receipt is f08-20261009/layer0-conditional-gdn-source-cpu-v1/receipt.json.

VERDICT -> Source-ready conditional checker, not a full-model qualification.
The original-weight numerical command must run only after the parent's GPU
control terminates and its genuine new full-shard identity exists:

```
python3 strata/flash-next/verify_layer0_gdn_original_v1.py \
  --requests /absolute/actual/candidate21_on/requests.json \
  --model-identity /absolute/genuine/post-model-identity.json \
  --output /absolute/new/conditional-gdn-original-receipt.json
```

The checker refuses existing output. It requires the exact prefix 1/2/4/8 roster,
one capture binding/PID, four distinct request ordinals, exclusive field paths,
metadata SHA and content, actual SFD/L0 request correspondence, graph key 2/3,
and independently reconstructed request nonce. It pins the consumed 33-field
0021 layout, exact 26 observed names and original-buffer provenance, required
field SHA/bytes/encoding/finiteness, and native Q8_1 packets reconstructed from
actual supplied F32 activations. It does not accept a marker boolean alone.

Seven conditional checks per prefix use original FP64 math:

- Original symmetric Q8_0 attn_qkv, attn_gate and ssm_out projections from verified
  Q8_1 d*codes. Stored Q8_1 sum s is unused specifically for these Q8_0 weights.
- Convolution history shift using supplied history and projected qkv.
- Original F32 four-tap convolution, SiLU, and q/k squared-sum L2 normalization.
- GDN outgoing recurrent state from supplied normalized qkv, log-decay, beta and
  incoming state.
- GDN output from that core divided by sqrt(128), per-head RMS with original
  F32 gamma and float32(1e-6) epsilon, then supplied-z sigmoid gating.

Actual consumed source evidence: frozen 21 source
sycl/src/kernels/cuda/native_gdn_preprocess.dp.cpp:73-96 reads convolution as
history[channel*3 + tap] and weights[channel*4 + tap], oldest first, updates the
three history slots and produces SiLU. Original reference
original_first_gdn_layer_v1.py:133-143 reads original GGUF rows with ne0=4 and
normalizes q/k by sum-of-squares plus epsilon. Physical captured recurrent state
[128,48,128] maps through transpose(1,0,2) to logical [48,128,128]. The original
provider verifies the selected source artifact and exact original tensor shapes
and types; affine weight types are rejected for these projections.

Controls include input/gamma/projection-weight perturbations, wrong state
mapping, nonfinite state/gamma/actual fields, tap reversal, history shift,
sum-vs-mean normalization, changed epsilon, non-Q8_0 weight role, metadata digest,
PID/nonce, altered source encoding and packet corruption with updated SHA.
Synthetic projection-weight controls alter a mock projection result; they do not
modify original GGUF bytes or demonstrate an actual weight fault.

Thresholds remain NMSE <= 1e-6 and normalized maximum <= 1e-4 with denominator
floor 1e-6. Inputs and incoming convolution/recurrent states are supplied, not
reconstructed from an independent prefix. Alpha/beta projections, upstream HC,
MoE, other layers, full model math and graph/capture lifecycle remain outside
this check. Complete-layer and own-state qualification remain false even if
all seven actual checks pass. No natural-language, API, latency or stability
claim follows from this diagnostic.
