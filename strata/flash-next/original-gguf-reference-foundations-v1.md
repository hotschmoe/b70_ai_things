# Independent original-GGUF CPU reference foundations

CONFIG -> NEW files; exact original four-shard UD-Q4_K_XL inventory and model
lock; frozen0018 engine's official llama/ggml3cf03257 reference sources. CPU only,
no GPU devices, frozen-source edits, new weights, whole-source hash scan or actual
full-forward execution. All1224 selected backbone roles are pinned, including
F32/BF16/Q8_0/Q4_K/Q5_K/Q5_1/IQ4_NL. MTP sidecar is excluded.

COMMAND -> python3 strata/flash-next/test_original_gguf_reference_cpu.py and
python3 strata/flash-next/test_original_math_scalar_cpu.py. Decoder test builds
only a small CPU probe linked to the pinned official libggml-base.a; compilation
uses image39992d70 with networknone and no devices. The executable calls only
GGML CPU initialization, scalar dequantizers and cleanup. No Strata arithmetic
is imported. Python decoder test uses the independently pinned gguf-py path.

RESULT -> Seven formats /84 synthetic blocks match official Python decoder F32
output exactly. Five quantized formats /60 blocks also match the compiled
pinned official GGML C scalar decoder exactly. Fifteen rival/coverage controls
exercise Q5_1 high bits / half ordering / offset, Q4_K split scales, Q5_K group
high bits, Q8 signed codes, IQ4_NL nonlinear values, original exact real versus
F32-dequant storage, partial blocks, wrong role/type/shape/architecture, and
extra MTP Q4_K/Q5_0/Q6_K roles. Nine scalar properties pass modulo GDN pairing,
decay-before-update, oldest-first conv, PLE9-row/dilation3 history, per-head QSA
ReLU, selected-cell attention, division-versus-modulo pairing, separate F16
storage deviation and duplicate
selection rejection. Thirteen synthetic read-admission controls reject missing /
invalid identity rows, changed SHA/stat/lock/header/extents/types, caller-cap
override, decoded expansion despite packed-fit, bad indices and source change.
These are decoder/equation/reader foundations, not model results.

## Loader and identity contract

original_gguf_reference.py implements independent byte decoders and bounded
pread row gathers. Its manifest pins inventory and lock SHA, all1224 complete
name/type/shape roles, qwen4exp architecture/geometry and official decoder source
hashes. A loader instance requires an admitted four-row full publisher identity
receipt with exact lock/revision, expected/actual SHA and unchanged current
before/after dev/inode/size/mtime/ctime. It verifies the actual small GGUF headers
against pinned inventory hashes and validates encoded extents against shard size.
Tensor rows are bounded to64MiB packed AND decoded per call; caller overrides
above the manifest cap are rejected; source stat is checked
before/after each read. Head / dense outputs can be evaluated row tiles; an expert
is gathered via flattened rows [expert*out+row]. PLE gathers only required IQ4_NL
rows. The loader neither converts nor caches the entire bank/table.

This binding is not a new full-file hash, freshness or media-integrity guarantee.
The caller must supply the parent's current identity/sentinel admission and bind
this manifest SHA in a frozen experiment plan. A source change preserving stat
can evade per-read stat checks; existing full-hash / page guards remain mandatory.
Synthetic tests did NOT instantiate a loader against actual source headers or
read actual tensor payloads; those runtime admission checks remain to execute.
No decoded actual expert/projection or full model has been qualified by them.

## Independent expectation lanes

- original_fp64 decodes exact represented real values and accumulates equations
  in FP64. F32 source values are promoted exactly; encoded FP16 scales, offsets,
  split scales and signed/nonlinear codes are represented without intermediate
  F32 dequant rounding.
- ggml_f32 explicitly rounds official scalar dequant multiply/subtract storage.
  This differs from original_fp64 when a small offset is lost; the test requires
  that difference to be observable.
- original_math_scalar.py declared_storage rounds only NAMED F32/F16 boundaries.
  It is NOT exact GPU FMA/reduction/native-exp emulation. It does not yet encode
  the production Q8_1 packets. Do not call these storage comparisons a strict
  native-kernel implementation-error gate. QSA's storage-versus-original metric
  is declared quantization cost, not measured GPU error.

The production native Q8_1 sum field is not assumed identical to official GGML
quantize_row_q8_1_ref's integer-code sum times scale. Add an independently audited
production-format encoder and compare actual packets before claiming native
projection implementation error, retaining the separate original-F32 lane.
Existing HC/PLE NMSE1e-6/max-normalized1e-4 limits are unchanged. No full-forward
error budget is inferred from these synthetic comparisons.

## Concrete next layer replay inputs

First GDN (layer0) requires original embedding, first HC input/read, original
attn_qkv/gate/alpha/beta/dt/a/conv/norm/out weights, incoming zero-or-labelled
recurrent/conv states, and actual native Q8_1 packets. Scalar gdn_step uses logical
state[head,i,j] with modulo mapping, then explicitly maps physical [i,head,j].
Its isolated recurrence is ready; full preprocessing/gates/output projection and
actual layer replay driver remain unfinished.

Layer1 PLE requires original n-gram metadata and selected IQ4_NL rows, original
Q8_0 key/value and F32 conv/norm tensors, four HC-stream input residuals and
incoming normalized history. Independent ple_postprojection evaluates group
RMS, signed-square-root gate, broadcast, grouped norm, dilated convolution,
SiLU residual and history advance after independent key/value projection.
Actual retained history has9 rows, taps0/3/6/current in logical [9,10240], from
row-fastest physical [channel,9]. The prior audit's57-row claim was a wording
error:57 counts cumulative fixture token updates, not retained state size.
See independent-math-reference-route-v1.md's appended correction/source anchors.

First QSA (layer3) requires independently projected/normalized/rotated Q/K/V,
raw/pooled/dead indexer keys, bias/position/tail metadata, causal selection and
FP16 cache contents. qsa_indexer_scores and qsa_selected_attention supply scalar
per-head-ReLU and selected-only attention equations. Full indexer pooling/RoPE,
causal top-k/page mapping and actual layer driver remain unfinished. Supplied
selection IDs cannot prove the indexer. Conditional replay with GPU inputs/state
qualifies only that local operator; decisive end-to-end48-layer forward must
compute its OWN incoming activations, routing, recurrent/QSA/PLE state and logits.

VERDICT -> Bounded CPU decoder/identity foundation and partial scalar equation
building blocks are ready for review. They deliberately report
full_model_math_qualified=false. Actual source reads, production activation
formats, full first-layer composition, independent48-layer forwarding and GPU
comparison remain required; no self-consistency or readable-answer result can
replace them.

QSA head-map anchors for the scalar reference: frozen
sycl/src/kernels/qsa_parity.cpp:165 defaults kv_divide=true; :333 divides by
(n_head/n_head_kv), and :983 selects modulo only as a rival. The consumed
qsa_decode_attn.dp.cpp:161 groups query heads at kvh*G and :531 uses h/G.
Official ggml-cpu/ops.cpp:8662,8725 likewise uses iq2/(neq2/nek2). The actual24/2
labelled CPU property requires12 heads per KV head and rejects modulo mapping.
