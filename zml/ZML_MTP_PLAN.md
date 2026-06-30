# zml NEXTN/MTP speculative decoding for qwen3.6-27b -- design + plan

Goal: add NEXTN Multi-Token-Prediction (MTP) speculative decoding to the zml qwen3_5 W8A8 serve to
close the decode gap (13.7 t/s -> ~22 t/s, sglang's 1.62x on the same B70). Greedy-only (matches
sglang XPU; exact-output). Scoped 2026-06-30; branch zml-w8a8-optimize. See JOURNAL for status.

## The MTP head (1 NEXTN layer; reuses existing zml modules)

Checkpoint `model-mtp.safetensors` (bf16, num_nextn_predict_layers=1): `mtp.fc.weight [5120,10240]`
(2H->H, no bias), `mtp.pre_fc_norm_embedding [5120]`, `mtp.pre_fc_norm_hidden [5120]`, `mtp.layers.0.*`
(a FULL-ATTENTION decoder layer == zml `TransformerLayer` full-attn variant byte-for-byte: q_proj
with query+gate, q_norm/k_norm on head_dim, o_proj, dense MLP, 2 RMSNorms), `mtp.norm [5120]`. Reuses
the main model's `embed_tokens` + `lm_head` (NOT in the MTP file). All 3 mtp norms are the `(1+w)`
RMSNorm form == zml `RmsNorm` (model.zig:982) bit-for-bit.

Head math (vLLM Qwen3NextMultiTokenPredictor.forward, qwen3_next_mtp.py:101-137):
```
e = pre_fc_norm_embedding(embed_tokens(token))      # RMSNorm
h = pre_fc_norm_hidden(prev_hidden)                  # RMSNorm; prev_hidden = main model's LAST hidden
                                                     #   for the accepted token, BEFORE text_model.norm
x = fc(concat(e, h, axis=.d))                        # [..,2H] -> [..,H], plain bf16 Linear, no bias
hidden, mtp_kv = TransformerLayer.forwardSelfAttn(x, token_index, mtp_kv)   # the 1 full-attn layer
draft = argmax(lm_head(mtp.norm(hidden)))            # reuse the Sampler (greedy)
```
In zml the required `prev_hidden` is exactly `hidden_buf` between the last layer and the sampler
(inference.zig:247-266) -- already host-visible, no model surgery.

Reuse map: pre_fc/hidden/norm = `RmsNorm`; `fc` = `QuantizedLinear`(weight_scale=null) plain bf16 dot;
`mtp.layers.0` = `TransformerLayer` full-attn (`forwardSelfAttn` model.zig:385-405); embed/lm_head =
shared from `Model`. MTP layer needs its OWN tiny `SelfAttnCache` (num_self_attn_layers=1).

## Spec-decode loop (K=1 greedy chain first cut; sglang XPU is also topk==1 chain)

Per iter, main committed through pos p, last hidden h_p, freshly-sampled t=x_{p+1}:
1. DRAFT (MTP head, s=1): d_1 = argmax(MtpHead(t, h_p, p+1, mtp_kv)).
2. VERIFY (main model, ONE forward over Kv=K+1 tokens [t,d_1] at pos p+1..p+2, recompiled main exes
   at s=Kv -- the SAME path prefill uses at s=seqlen): produces hidden + main argmax y_0 (true next
   after t), y_1 (after d_1).
3. ACCEPT (host): t is the always-correct bonus root; accept d_1 iff d_1==y_0. Emit t (and d_1 if
   accepted); carry next = y_n. K=1 -> up-to-2x ceiling.
4. COMMIT/ROLLBACK: token_index = p+1+n. Full-attn KV: leave it (wrong entries past p+1+n are scratch,
   overwritten next iter -- no explicit rollback). GDN state: MUST roll back (see below).

## The GDN wrinkle (the hard part) -- even K=1 needs it

qwen3.6 is 3/4 GDN. The verify forward over Kv tokens advances each GDN layer's `conv_state` (last 3
acts -- easy: re-slice the accepted prefix via buildUpdatedConvState model.zig:872) AND
`recurrent_state` (accumulated delta-rule matrix, recurrentGatedDeltaRule:824-870 returns only the
FINAL state). On rejection the recurrent_state is poisoned by the wrong drafted token and cannot be
cheaply inverted. The GDN-free variant (verify only the confirmed token) has NO speedup -> even K=1
needs recurrent-state rollback.
- FIX (b) REPLAY (recommended first): before the verify, the committed GDN state is in the cache; after
  the host computes n, run a GDN-ONLY forward over the accepted n+1 tokens from the saved state to get
  the correct committed state. Cost: one extra GDN pass over <=Kv tokens (<=2 for K=1). Gate it to run
  only when n<Kv-1.
- FIX (a) PER-STEP STATE EMISSION (faster, later): modify recurrentGatedDeltaRule to emit the recurrent
  state at each step, then dynamicSlice at index n. (= sglang's fused_mamba_state_scatter_with_mask.)

## New compiled graphs needed (beyond existing prefill+decode)

1. `mtp_draft` exe (s=1): embed + MtpHead.forward + lm_head argmax. Inputs already host-visible.
2. `verify` exes (s=Kv): the EXISTING full_attention_layer/linear_attention_layer/sampler exes
   recompiled at seqlen=Kv (compileFullAttentionLayer/compileLinearAttentionLayer already take seqlen);
   sampler must return all Kv argmax predictions for the host accept compare.
3. GDN replay (or per-step state emission) for the recurrent-state rollback.
4. session.zig `runDecodeMtp`: draft -> verify -> host argmax-compare -> commit/rollback -> emit n+1.

## Step-by-step plan (each independently testable)

- Step 0 -- Add `MtpHead` to model.zig + load `mtp.*` (bf16) alongside the main checkpoint; reuse
  embed_tokens/lm_head. TEST: weights load, shapes match. [foundation; CPU-build only]
- Step 1 -- MTP head correctness in isolation: draft `d_1` for "The capital of France is" should equal
  the main model's next token ("Paris"). Accept ~0.84 first token. [needs a GPU run for h_p]
- Step 2 -- Verify graph at s=Kv (full-attn KV first): verify y_0 == plain-decode next token, byte-identical.
- Step 3 -- Host accept/commit loop (K=1) + GDN replay-rollback. ORACLE: greedy MTP-on output stream
  BYTE-IDENTICAL to MTP-off greedy decode (speculative greedy is exact). This is the gate that matters.
- Step 4 -- Measure t/s single-card then TP=2 vs the 13.7 baseline. Target accept ~0.84 -> ~1.5-1.6x ->
  ~21-22 t/s. zml's single-process PJRT + in-graph collectives sidesteps vLLM's "distributed = no graph
  capture" wall, so TP=2 MTP is a genuine opening other backends can't reach. GO = byte-identical AND >=1.4x.
- Step 5 (opt) -- per-step GDN state emission (drop the replay pass); K=2 sweep; non-greedy chain rejection sampler.

## References

- vLLM MTP math: /mnt/vm_8tb/b70/build/vllm/vllm/model_executor/models/qwen3_next_mtp.py:101-137,67-75,266-296
- sglang XPU MTP: sglang/patches/mtp_tree_xpu.py (chain :18-97, verify :99-140, 4 XPU gate fixes :153-256,
  greedy-only :227-233 -- ignores temp/top_p; --max-running-requests 4 + --skip-server-warmup + eager)
- prior B70 MTP analysis: docs/kernel/12_mtp_specdecode_plan.md (accept ~0.84/0.57/0.46 @ draft 1/2/3;
  GDN per-spec-step CPU-sync tax vLLM #35387), docs/kernel/21_gdn_spec_capture_issue.md
- zml model: examples/llm/models/qwen3_5/model.zig (SelfAttn :519, TransformerLayer :385, RmsNorm :982,
  GatedDeltaNet :882 / recurrentGatedDeltaRule :824, KvCache :1029); inference.zig (composed exes :303,
  compile-at-seqlen :592-666, sampler :668); session.zig:82-160.
