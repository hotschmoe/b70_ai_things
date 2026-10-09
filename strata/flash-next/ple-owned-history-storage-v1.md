# Conditional owned original PLE history/storage model v1

CONFIG -> New independent layer1 PLE source lane. Frozen scalar
ple_postprojection/ple_dilated_conv and original decoders remain unchanged.
Header/source and synthetic controls only during parent model controls. No
actual model payload, GPU/runtime, SDK, commit or JOURNAL work.

COMMAND ->

```
python3 strata/flash-next/test_ple_owned_history_storage_v1.py
python3 -m py_compile strata/flash-next/ple_owned_history_storage_v1.py
```

RESULT -> Fifteen CPU controls PASS: five frozen hash-oracle cases, actual
header constants/types/shapes, uint64 overflow/XOR/modulo rivals, EOS/null/window
semantics, synthetic encoded IQ4_NL head rows, owned normalized history,
dilation3/9rows, reset and source/token/input/history/order checkpoint negatives.
No actual original weights or table row was read. Future original payload runs
remain UNOBSERVED; no new numeric threshold or model qualification is inferred.

VERDICT -> Independent owned token lookup/history mathematical/storage model,
conditional on supplied layer1 residuals. Exact host hash semantics are modeled;
native projection/reduction/intrinsic/quant/lifecycle and all48-layer composition
remain unqualified. State is never supplied from a capture or selected table IDs.

## Authoritative original roles and lookup

Current selected GGUF header inventory, not stale pack labels, gives:

| Role | Original type | GGML shape |
| --- | --- | --- |
| per_layer_token_embd.weight | IQ4_NL | [160,320001536] |
| blk.1.ple_key.weight | Q8_0 | [2560,10240] |
| blk.1.ple_value.weight | Q8_0 | [2560,2560] |
| blk.1.ple_conv1d.weight | F32 | [4,10240] |
| blk.1.ple_norm_key.weight | F32 | [10240] |
| blk.1.ple_norm_query.weight | F32 | [10240] |
| blk.1.ple_norm_conv.weight | F32 | [10240] |

One IQ4_NL row is five32-value blocks,18B each,90B total. Sixteen head-slowest
rows flatten to2560 F32 values. Split nibbles are low-half[j] and high-half[j+16],
with the pinned signed nonlinear codebook and zero offset, not codebook minus8.
Original half scale times integer codebook values is represented F32. The
physical table has320001536 rows, while header vocab sum/lastheadend is320001446:
90 padded rows. Physical shape is not derived from vocab or stale documentation.

Metadata fixes ngram_size3,8heads/ngram,16heads total, EOS248044, layer[1], conv4,
image token248056 and irregular non-power-of-two per-head vocab/offset arrays.
Host src/kernels/ngram.cpp:47-105 multiplies current/newest/oldest context by
[23703573157769,20109073645365,8052911324071], wraps each uint64 product, XORs,
then modulo each head vocab and adds its own offset. Prev storage is oldest-first;
missing negative predecessors and predecessor EOS cut that position and all
older context to EOS. Token0 is real, and current EOS does not cut its own window.
The model owns last_two=[-1,-1] initially and updates only from committed tokens.
The original image ID participates as an integer hash token; full multimodal
embedding/residual generation is outside this conditional lane.

## Actual projection and postoperation seams

Strict source_exact PLE in sycl/src/kernels/cuda/ple.dp.cpp:419-426/:493-499 uses
original Q8_0 key/value projections against F32 table embedding with the HC
projection primitive. It does NOT apply ordinary activation Q8_1 or narrow those
original weights/inputs to BF16/F16. Constructor validates exact original types
and shapes before any row call. The selected F32 convolution remains F32,
despite legacy comments/pack conventions labeling it F16. Norm vectors are
original F32. source_exact disables the prefill BF16 batch path at
sycl/src/prefill/prefill.cpp:2173; :2787-2795 runs the same per-token PLE path,
advancing history after each token. Stale nonexact pack routes are not substituted.

native_ple_postops.dp.cpp:56-155 normalizes projected keys and supplied residual
queries per4stream/2560dim group, computes their materialized-product dot/sqrt2560,
then sigmoid(sign(score)*sqrt(max(abs(score),1e-6))). Value is broadcast by that
stream gate; gated values are normalized by original conv norm. History stores
nine normalized F32 rows. Four original F32 taps consume9/6/3tokens back and
current: logical history[0],[3],[6],current. Physical source is[channel,9]; the
logical model is[9,10240] and maps by transpose. Convolution is followed by SiLU,
and result is hidden+gated+activation. History then shifts exactly one row.

The frozen scalar ple_postprojection evaluates these equations with declared
F32 storage boundaries. Its convolved field is RAW mathematical convolution,
whereas native postops conv buffer stores SiLU activation. The model does not
relabel that raw field as an observed native buffer. Native materialized-product/
FMA/warp/RMS/dot and sequential addition/intrinsic rounding are not reproduced
bitwise by FP64 mathematical evaluation and F32 stage stores.

## Owned reconstruction, identity and scope

OwnedPleHistory(provider,source_identity_sha256).advance(token,residual) computes
its own16 row IDs from committed predecessor tokens, lazily obtains bounded
original rows, projects key/value, invokes frozen postprojection on its own
history, and commits normalized F32 history/last_two. It takes no supplied table
row IDs, projected values or captured states. The residual is explicitly supplied
at the layer1 PLE injection point, before its attention HC read; upstream model
residual generation is not qualified.

Source identity is an immutable verified-artifact receipt tag supplied by the
caller; it is not authenticated merely by matching a64hex string. A future real
provider must separately validate genuine original GGUF identity before payload
reads. Snapshots bind source tag/constants, token order and residual bytes, own
last_two/history. restore requires caller's expected committed prefix and
independently replays from zero before comparing the snapshot. Source identity,
token/input/history mutations are rejected. Actual conversation_state.cpp:170-202
saves PLE history and derives ple_prev from committed IDs; per-stage/slot PLE
state must still be separately measured. CPU roundtrip is not actual GPU byte or
cache-continuity proof.

All projection tiles are capped64MiB, token history max256 (default64). Original
payload access remains lazy; tests use explicitly synthetic encoded table rows
and analytic equivalent Q8_0 row weights, never real GGUF payloads. Header role
receipts, source fingerprints and frozen hash-vector prefix are in the plan.
No all-layer GDN/QSA/HC/FFN composition, full-model math, own whole-model state,
concurrency, natural-language, latency/stability or shelf claim follows from this
local conditional helper. No failure is resolved by changing math/tolerances.
