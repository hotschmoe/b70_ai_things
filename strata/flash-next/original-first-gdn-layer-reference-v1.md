# Independent first GDN layer composition reference

CONFIG -> CPU-only NEW modules, original FP64 equations, actual0018 source and
original four-shard source-role binding. Frozen reference foundations remain
unchanged. No actual tensor payload, heavy forward, GPU or frozen-source edit.

COMMAND -> python3 strata/flash-next/test_original_vector_decoder_cpu_v2.py
and python3 strata/flash-next/test_original_first_gdn_layer_cpu_v1.py.

RESULT -> Vectorized independent decoder matches frozen scalar v1 in BOTH exact
original_fp64 and ggml_f32 lanes over112 synthetic blocks /seven actual formats,
and matches pinned official gguf-py in the F32-dequant lane. Frozen scalar v1 has
its own separate official compiled GGML C decoder checks. No speed claim or new
actual-payload result follows from this triangulation.

The asymmetric three-token layer test passes17 controls: decoded128-byte tiles,
tiled-versus-untiled equations, owned recurrent history versus per-token reset,
independent empty-state repeats, source expert offsets and ordered weighting,
selected softmax renormalization, lower-ID ties, HC per-stream mean norm and
injection /streams, GDN sum(q squared)+epsilon /output1/sqrt(S) /sigmoid gating,
and rejection of unsupported native lane, wrong layer, oversized tile or token,
plus nonidentity expert-weight scaling.

## Implemented source equations and ownership

original_first_gdn_layer_v1.py binds actual roles through the immutable
OriginalGguf admission contract. OriginalTensorRows uses the NEW vector decoder,
with exact shape/type/source lock already bound by the old reader. Every source
read retains packed AND decoded64MiB limits, original source stat before/after,
and strict index/extent checks. Row tiles cannot exceed the caller's bounded
cap; source expert rows are expert*out_rows+output_row. No compatibility weights
are read. The actual source provider requires actual2560/4/320,128-state,
16-key/48-value heads,512-expert/top10/640-FFN geometry. Synthetic toy geometry
has explicit actual_source=false and cannot be presented as model evidence.

The reference gathers exact source token embedding rows and starts four equal
residual streams. HC attention read applies per-stream RMS with original folded
F32 norm, original Q8_0 down, SiLU(down/4), original Q8_0 up gate, original F32
injection and gated stream mean. It does not add1 to the already folded norms.

GDN performs original qkv/z/alpha/beta projections, beta sigmoid,
A*softplus(alpha+dt), oldest-first convolution then SiLU, squared L2 q/k norms,
modulo key-head pairing, decay-before-rank1 update, and output dot /sqrt(state).
Its logical recurrence is state[value_head,i,j]; a future GPU comparison must
explicitly map physical state[i,value_head,j]. Per-head RMS with shared original
ssm_norm followed by SIGMOID(z), not SiLU(z), feeds the original output projection.
The first HC write materializes the original residual addition before FFN read.

FFN read uses original second-half HC sources. The reference computes512 own
router logits, mathematical full softmax, top10 with lower-ID ties, and selected
renormalization. It decodes ONLY its independently chosen experts. Each expert
computes original gate/up, SiLU(gate)*up and down; selected outputs accumulate in
router rank order. Shared expert uses its own gate/up/down and sigmoid scalar
gate. Their sum feeds the independent FFN HC write. Sequential token calls carry
owned recurrent/conv state forward, never feeding GPU-produced state/routing.
Only layer0 is computed: it produces layer1 inputs, not whole-model logits.

The reference is reviewed against independent embedded qwen4exp graph anchors:
267 HC read,323 write,353 embedding/residual initialization,476 sigmoid GDN
output gate,861 GDN preprocess/state,988 routed/shared experts. Delta-net-base
anchors45,319 specify inverse sqrt(state) output scaling. Source hashes are
bound in original-first-gdn-layer-reference-plan-v1.json.

## Explicit unimplemented contracts and remaining gates

Only original_fp64 arithmetic is supported. Requests for a native/rounded lane
raise rather than silently use FP64. Ordinary and expert/shared Q8_1 activation
packets, native F32 FMA/reduction ordering and native-exp accuracy, prefill
F16/BF16 precision seams, production router rounding/ties remain TODO. The new
vector decoder's ggml_f32 WEIGHT decoder lane does not implement those activation
or operation contracts. Do not classify GPU-versus-FP64 differences as native
implementation error until independent actual packet/storage references exist.

PLE/layer1, QSA/layer3, remaining47 layers, head/full logits, actual-payload
reference execution and comparison are unimplemented here. Loader actual-header
admission is independently executable but has not been invoked by these tests.
Future actual CPU runs still require parent's fresh whole-source identity and
known-page guards; old receipt/stat agreement cannot detect unchanged-stat page
recurrence. This source plan authorizes no actual payload execution while parent
GPU controls are active. It supplies no numerical tolerance for an unrun model.

VERDICT -> Bounded synthetic first-layer mathematical composition and vectorized
decoder foundations are ready. Source binding plus independent owned routing /
state are concrete, but tests explicitly retain native_storage_qualified=false
and full_model_math_qualified=false. Frozen source and old foundations unchanged.

L2 semantic anchor models/models.h:14-17 implements
rms_norm(x,eps/n)/sqrt(n), mathematically x/sqrt(sum(x squared)+eps).
The property label sum_plus_eps_not_rms refers to this additive-epsilon original
GGML contract, not a native max-floor interpretation. Expert weights scale is
absent in actual metadata; llama-hparams.h:124 defaults its parameter to0 as a
sentinel, and llama-graph.cpp:2149 skips scaling for0 or1 (effective1). Actual
source providers reject unsupported nonidentity scale metadata and changed RMS
epsilon instead of silently using the first-layer contract.
