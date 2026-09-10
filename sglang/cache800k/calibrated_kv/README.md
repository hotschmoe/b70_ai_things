# Calibrated FP8 KV parity design for SGLang Qwen3.8

CONFIG -> CPU-only source review of SGLang main 2f7393f0 and the existing R276
calibration artifact int4-mtp3-kv-scales.json. No active image, model, native
library, or GPU state changed.
COMMAND -> Review vLLM's kv_calibration_hook.py and the source SGLang GPTQ,
RadixAttention, Qwen3_5 target/MTP, KV pool, Triton attention, intel_xpu attention,
and generic scale loader. Validate scale_plan.py against the real artifact and
run eight rejection/mapping regression tests.
RESULT -> Six real-artifact plans pass: target and draft for TP1 rank0, TP2 rank0,
and TP2 rank1. Each target plan has16 full-attention layers; each draft plan has
one. All eight CPU tests pass. The mapper imports no torch, creates no tensor,
and applies nothing to a model.
VERDICT -> Calibration can be mapped exactly as data, but calibrated FP8 serving
parity is not implemented or GPU-qualified. Native intel_xpu prefill is a
separate blocker that a scale-loader port alone cannot fix.

## Exact scale identity and rank mapping

Artifact SHA256:
d53fb6565c485658b345d0b48aaf3ea4f5e0a48748e09a24bc6a29234c026a82

The17 entries are target full-attention layers3,7,...63 and
mtp.layers.0.self_attn.attn. GDN conv/SSM layers do not use this KV calibration.
Each q/k/v scalar equals max(amax on calibration ranks0 and1)*1.1/448. Every
serving TP rank receives the same merged scalar; no head slicing or rank-local
substitution is appropriate. TP1 receives those same scalars because they cover
both calibration shards. This does not establish matching activation ranges
under another backend or TP topology.

Expected named_modules paths at the reviewed source layout:

- Conditional Qwen target: model.layers.N.attn, N=3,7,...63.
- Qwen3_5ForCausalLMMTP draft: model.model.layers.0.attn.

The caller must enumerate actual RadixAttention instances and supply the exact
path-to-layer_id inventory. Different wrappers/names must fail until explicitly
ported. The mapper rejects incomplete/extra layers, missing rank observations,
nonfinite/nonpositive scales, wrong calibration provenance/config bytes,
unsupported topologies, quantized queries, and incompatible backends/formats.
There is no fallback to1.0. Target-only runs still validate the entire original
17-layer artifact, then select its16 target entries without discarding MTP
provenance.

The source model config fingerprint is validated, but the mapper does not hash
23GB of checkpoint shards. Before integration, verify all source model files
against artifact.provenance.model_files, or require an explicitly reviewed,
content-addressed transformation manifest connecting SGLang's artifact to them.
A changed config intentionally fails; do not rewrite the stored calibration
identity merely to make a transformed model pass.

## Required loader integration

SGLang's generic load_kv_cache_scales requires a model method when
--quantization-param-path is supplied. Qwen3_5 target/MTP have none. GPTQ does not
create scale parameters on RadixAttention, so renaming checkpoint scale keys is
insufficient. Add a narrowly scoped loader adapter for this schema/model pair;
pass model root, target/draft role, actual TP rank/size and attention backend
explicitly. Keep the default generic path unchanged for other models.

After ordinary weights load, but before KV profiling/warmup or graph capture:

1. Validate the full artifact/config/weight provenance and actual layer inventory.
2. Obtain the pure-data plan using scale_plan.make_plan.
3. On each actual layer device create persistent scalar float32 tensors for
   k_scale and v_scale. Initialize matching k_scale_float/v_scale_float from the
   exact same float32-rounded values. Do not update scales after cache entries
   are populated or graphs captured. Assign all four fields coherently.
4. Confirm exact loaded coverage per process:16 target or1 draft, with unique
   role/rank/layer keys. Reject missing/duplicate entries and non-FP8 cache pools.
   Aggregate all-rank startup records before admitting requests.
5. Record artifact/config hashes, source scalar and effective float32 scalar,
   role/rank, module path, cache dtype, attention backend, and query dtype.

q_scale remains recorded but unapplied: the reviewed SGLang path uses FP16
queries, unlike an FP8-query path requiring its own scale. The vLLM hook writes
q_scale too; parity must compare whether query quantization was actually active,
not simply whether a field existed. The mapper rejects query_quantized=True.
The artifact's _xpu_ops.py fingerprint authenticates source calibration; SGLang
must not impersonate that fingerprint or reuse that ABI-specific binary.

## Write/read semantics and remaining backend gap

E4M3 here means torch.float8_e4m3fn with finite max448, not E4M3FNUZ, E5M2, or
an unspecified byte dtype. Store quantization is cast(K/k_scale) and
cast(V/v_scale); attention must multiply cached values by the corresponding
scale in both prefix-prefill and decode. The pool may store uint8 views of the
same FP8 bits. Overflows/finiteness must be measured; correct loading alone does
not prove the old corpus covers new activations.

Current SGLang KV pool set_kv_buffer divides K/V in place before conversion.
The reviewed Triton prefill path clones K/V before this call to protect fresh
inputs; Triton prefill/extend, verification and decode contain k_scale_float /
v_scale_float descaling plumbing. That is source evidence, not a numeric GPU
oracle. The model's hybrid GDN backend and speculative/graph routes still need
independent compatibility checks.

In contrast, intel_xpu.forward_extend explicitly leaves k_descale/v_descale=None
with the comment that FP8 KV is unsupported there, then passes those values to
flash_attn_with_kvcache while reading cached K/V. Its decode path has descales.
Installing calibrated scales only at the layer would therefore leave a
prefill/decode inconsistency. scale_plan.py rejects intel_xpu until this path
has a deliberate source/kernel repair and numeric oracle. Do not just uncomment
an old CUDA/FA3-shaped branch that also casts query types.

## Qualification before any parity claim

Run leased per-card and compiled pair health for a new backend stack, then
small cache-write/read oracles with distinct K/V scales and both ranks. Compare
FP16-cache attention against calibrated FP8 on fresh prefill, cached-prefix
extend (including one-token remainder), decode, MTP verify/draft and graph
replay. Check copied prefix state and retained KV bytes, clipping/nonfinite
counts, request isolation, long reconstructed Pi traces, and matched concurrent
coherence. Finish with owned teardown and post-health. No such GPU tests were
performed by this mapper task.

Raw CPU plans and tests:
/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-calibrated-kv/
