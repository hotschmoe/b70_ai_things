# zml GDN (Gated-DeltaNet) decode optimization plan -- qwen3.6-27b W8A8, dual B70

Scope: the Gated-DeltaNet (GDN, linear-attention) layers of the zml qwen3.6-27b W8A8 serve.
Profiling this arc found decode is ~79% GDN (48 bf16 linear-attention layers) vs ~21% int8
full-attention (16 layers) -- ZML_INT8_PERF_HANDOFF.md:56-64, ZML_MTP_PLAN.md:12-14. This doc is a
ranked, concrete plan to attack that 79%. It drives a dedicated kernel session. Read-only research;
every claim carries a file:line.

Baseline to beat: 13.7 t/s decode, TP=2, byte-identical greedy "The capital of France is Paris."
(ZML_INT8_PERF_HANDOFF.md:37-39). MTP is DONE + byte-exact at 98.8% accept but gives NO speedup
(ZML_MTP_PLAN.md:8-20).

---

## 0. TL;DR ranked plan (by win / effort)

| # | Lever | Decode win (s=1) | Prefill/verify win | Effort | Risk | Kernel? |
|---|-------|------------------|--------------------|--------|------|---------|
| 1 | int8 the 3 big GDN projections via the EXISTING `dotAcc` full-W8A8 path + act-quant dedup | +5..12% | prefill ~1.6-2x on GDN proj; verify amortizes only weakly | LOW (wiring ~10 lines) + MED (offline requant) | MED (GDN was deliberately left bf16 -- coherence) | no |
| 2 | Chunked/fused delta-rule scan in pure StableHLO (replace the per-step `stablehlo.while`) | ~0..3% (single step at s=1) | prefill O(s)->O(s/C) BIG; unblocks the MTP-verify scan cost | MED (StableHLO rewrite, CPU-validatable) | MED (f32 reassociation -> rel-tol, not bit-exact) | no |
| 3 | Fused w8a16 decode kernel (bf16 act x int8 weight, epilogue dequant) for GDN proj + the 21% full-attn | +10..20% (removes act-quant prologue tax; full ~1.4x at M=1) | enables projection amortization at M=Kv (the other half of MTP) | HIGH (XLA FFI custom-call -> oneDNN, or custom plugin) | HIGH (plugin FFI surface unknown) | yes |
| 4 | Custom XeTLA/SYCL chunked-scan kernel | -- | max prefill/verify if #2 is compute-bound | HIGH | HIGH | yes |

The honest headline: at decode s=1 the GDN is LATENCY/GEMV-bound, not bandwidth-bound (section 1), so
NO single lever is a large single-stream decode win on its own. The large decode win is INDIRECT:
lever 2 (chunked scan) + lever 3 (fused M=Kv kernel) make the MTP verify amortize -> the already-done
98.8%-accept MTP delivers ~1.9x (~22-26 t/s). Levers 1 and 2 also give immediate, large PREFILL/TTFT
wins on their own with no new kernel. Start with 1 and 2 in parallel (both CPU-validatable, no plugin
dependency); scope 3 as the session's main kernel deliverable.

---

## 1. Why decode is GEMV/latency-bound (the roofline that governs the ranking)

The measured facts, not assumptions:

- MTP verify at s=Kv=2 costs ~2x a single s=1 decode (143.7 ms vs ~73 ms/token) -- it does NOT
  amortize (ZML_MTP_PLAN.md:11-16, ZML_INT8_PERF_HANDOFF.md:48-53). If decode were bandwidth-bound,
  s=2 would reuse the same weight reads and cost ~1x; instead it is ~linear in s.
- The B70 int8 GEMM at M=1/M=2 dispatches to oneDNN's GEMV path (never XeTLA/DPAS) and is LINEAR in M
  (~1.14 ms/layer at M=1, ~2.2 ms at M=2) -- b70-int8-xmx-roofline, ZML_INT8_PERF_HANDOFF.md:53,
  ZML_INT8_PERF_HANDOFF.md:168-171. bf16 GEMV at M=1 ~= 434 GB/s; int8 GEMV ~= 309 GB/s
  (ZML_W8A8.md:66-73). Because int8 reads half the bytes, raw-i8 is ~1.42x at M=1, but full-W8A8
  (dynamic per-token act-quant) caps at ~1.09x at M=1 -- the act-quant PROLOGUE is the bottleneck --
  and `weightOnly()` int8 is a DEAD END (0.5-0.8x, it materializes a bf16 weight) (ZML_W8A8.md:64-71,
  common_quant.zig:87-95).

Consequence for the GDN: at s=1 the recurrent scan runs exactly ONE step (nn.zig:1391-1398), so the
79% is dominated by the 3 big bf16 projection GEMVs + conv1d + the single f32 scan step, all
latency-bound. Doubling s (MTP verify) doubles BOTH the projection GEMV cost (linear in M) AND the
scan (Kv sequential steps). So MTP amortization needs BOTH the scan to stop being Kv sequential steps
(lever 2) AND the projections to amortize across Kv (needs the DPAS kernel of lever 3; and even DPAS
needs M>=~8 to leave the GEMV regime, so K=1 -> Kv=2 is intrinsically hard -- see section 5).

P0 (section 6) must split the GDN-s=1 79% into {projections, scan, conv1d} before committing kernel
effort. My estimate: projections ~55-70%, scan+conv ~30-45% at s=1.

---

## 2. Current GDN implementation anatomy (what we are changing)

Config (from the served checkpoint): 48 linear-attention layers of 64; hidden_size=5120;
linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128,
linear_conv_kernel_dim=4 (model.zig:35-39, model.zig:1011-1016). Derived:
key_dim = 16*128 = 2048; value_dim = 48*128 = 6144; conv_dim = 2*key_dim + value_dim = 10240.

The layer (`GatedDeltaNet`, model.zig:976-1205):

- Projections, all `zml.nn.Linear` BF16 (NOT quantized), built by `initProj` with `weight_scale=null`
  (model.zig:994-996, model.zig:1002-1006):
  - `in_proj_qkv` [dout=10240, d=5120] = 52.4M params -- the biggest (model.zig:1002, forward
    model.zig:1111). Column-parallel (`.dout=.model, .d=.replicated`).
  - `in_proj_z`  [dout=6144,  d=5120] = 31.5M params (model.zig:1003, model.zig:1148). Column-parallel.
  - `in_proj_b`  [dout=48,    d=5120] = 0.25M (model.zig:1004, model.zig:1150). Tiny.
  - `in_proj_a`  [dout=48,    d=5120] = 0.25M (model.zig:1005, model.zig:1151). Tiny.
  - `out_proj`   [dout=5120,  d=6144] = 31.5M params (model.zig:1006, model.zig:1195). ROW-parallel
    (`.dout=.replicated, .d=.model`).
  - `conv1d_weight` [out=10240, in=1, kernel=4] = 41K, a Tensor (not Linear) used by `conv1d`
    (model.zig:1007, model.zig:1121-1139). Tiny; leave bf16.
  - Per-layer projection params ~= 115.9M -> bf16 232 MB/layer -> 48 layers ~= 5.56B params ~= 11.1 GB
    bf16. This is ~34% of the 32.6 GiB served model (ZML_INT8_PERF_HANDOFF.md:192-193) -- the single
    biggest bandwidth consumer at decode.

- The recurrent scan: `recurrentGatedDeltaRule` (model.zig:1032-1078) upcasts q/k/v/g/beta to f32,
  L2-normalizes q,k (model.zig:1041-1042), scales q by 1/sqrt(head_k_dim) (model.zig:1040,1045), then
  calls `zml.nn.GatedDeltaNet.forward` (model.zig:1065-1072). That forward is a `stablehlo.while` over
  s positions (nn.zig:1320-1404); the loop body calls `step()` per position (nn.zig:1370-1388). `step()`
  is the delta-rule single update (nn.zig:1298-1311):
  `v_hat = S.dot(k); delta = (v - v_hat)*beta; S' = S*alpha + delta (x) k; y = S'.dot(q)`
  -- exactly FLA's `fused_recurrent` decode kernel (fused_recurrent.py:97-123). State S is
  [b, h=48, v=128, k=128] f32 = 3.1 MB/layer.
- conv1d (causal, feature_group_count=conv_dim) + silu (model.zig:1121-1145), then split into q/k/v
  (model.zig:1153-1161); beta = sigmoid(b), g = -A_log.exp()*softplus(a+dt_bias) (model.zig:1163-1165);
  q/k heads stuttered x3 to match 48 v-heads (qk_head_repetition=48/16=3, model.zig:1167-1174,
  model.zig:999-1000).
- Decode (s==1) and MTP-verify (s=Kv) CONTINUE from the cached conv+recurrent state
  (model.zig:1090-1102, model.zig:1114-1119, model.zig:1183). Prefill (s>1) starts fresh.

The checkpoint leaves GDN bf16 on purpose: the w8a8-sqgptq compressed-tensors `ignore` list is
`["lm_head", "re:.*linear_attn.*", "re:.*visual.*", "re:.*mtp.*"]` (ZML_W8A8.md:124-127). So there is
NO int8 GDN weight on disk today -- lever 1 needs a checkpoint step.

---

## 3. LEVER 2 -- Chunked/fused delta-rule scan (replace the per-step `stablehlo.while`)

### 3.1 What the naive scan costs and why chunking helps

`nn.zig` runs an s-length `stablehlo.while`, one `step()` per token (nn.zig:1391-1398). Each step is a
chain of tiny per-head (batched over h=48) f32 GEMVs/broadcasts on the 128x128 state. This is:
- s=1 decode: 1 iteration -- cheap-ish, but carries the `stablehlo.while` wrapper overhead even for 1
  step, plus ~5 tiny ops.
- s=Kv verify: Kv SEQUENTIAL iterations, each a dependency on the previous -- no parallelism, latency-
  bound. This is a prime suspect for why verify ~= Kv x decode (ZML_MTP_PLAN.md:15-18).
- s=prefill: O(s) sequential steps -- the dominant prefill cost.

The FLA "chunked_gated_delta_rule" (the sglang/vLLM prefill algorithm) reorganizes the recurrence into
CHUNKS of C tokens: INTRA-chunk is a parallel (matmul) block, INTER-chunk is a short recurrence over
s/C chunk-states. This turns O(s) sequential tiny steps into O(s/C) sequential chunk-steps whose inner
work is DPAS-friendly matmuls.

### 3.2 The chunked math (verified against the FLA source under /mnt/vm_8tb/b70/build/vllm)

Pipeline: `chunk_gated_delta_rule_fwd` (chunk.py:23-86). Notation per (batch, head), chunk size C
(FLA_CHUNK_SIZE, chunk.py:37); g is the log forget-gate, beta the delta gate, scale = 1/sqrt(K).

1. `g = chunk_local_cumsum(g)` -- cumulative sum of the log-gate WITHIN each chunk (chunk.py:37-39).
2. `A = chunk_scaled_dot_kkt_fwd(k, beta, g)` (chunk.py:41-48, chunk_scaled_dot_kkt.py:74-99):
   `A[i,j] = beta_i * (k_i . k_j) * exp(g_i - g_j)` for i>j within the chunk, else 0 (STRICTLY lower
   triangular; chunk_scaled_dot_kkt.py:94-95).
3. `A = solve_tril(A)` -- forms `(I - A_strict)^{-1}` by blocked 16x16 forward substitution
   (chunk.py:49-51, solve_tril.py:82-89: `b_A = -tril(A)`, iterate, `b_A += I`). A_strict is nilpotent
   (strictly lower), so `(I - A)^{-1} = I + A + A^2 + ... + A^{C-1}` exactly (finite Neumann series).
4. `w, u = recompute_w_u_fwd(k, v, beta, A, g)` -- the WY representation (chunk.py:52-60,
   wy_fast.py:91-116): `u = A @ (v * beta)` (the new "pseudo-values"), `w = A @ (k * beta * exp(g))`.
5. `h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state)` -- the INTER-chunk
   recurrence over chunk states h (chunk.py:61-71, chunk_delta_h.py:131-296). Per chunk t, with the
   entering state h_prev [V,K]:
   - `v_new = u - w @ h_prev^T` (chunk_delta_h.py:172-198)
   - decay: `h_prev *= exp(g_last)` then `v_new *= exp(g_last - g)` (chunk_delta_h.py:206-219)
   - `h_next = h_prev + k^T @ v_new` (chunk_delta_h.py:272-296)
   This is the ONLY sequential part -- O(s/C) chunk-steps, each a couple of [C x K/V] matmuls.
6. `o = chunk_fwd_o(q, k, v_new, h, g, scale)` (chunk.py:72-82, chunk_o.py:90-140):
   `o = (q * exp(g)) @ h_prev  +  tril(q @ k^T * exp(g_i - g_j), incl. diagonal) @ v_new`, all * scale
   (chunk_o.py:89-99, chunk_o.py:130-139). First term = contribution of the pre-chunk state; second =
   intra-chunk causal attention on the pseudo-values.

Everything is matmuls + cumsum + a small triangular inverse. No data-dependent control flow inside a
chunk (only the O(s/C) chunk loop remains).

### 3.3 Is it expressible in StableHLO / zml ops? YES, no custom kernel required

Every primitive exists in zml today:
- cumsum for g: `Tensor.cumulativeSum` (tensor.zig:1641-1667).
- tril / strict-tril masks (steps 2, 6): `Tensor.triangular(axes, num_diagonals)` (tensor.zig:4098-4107)
  -- works at any rank, so it masks batched-over-heads tensors.
- all the matmuls (steps 2,4,5,6): `Tensor.dot` / `Tensor.dotAcc` / `Tensor.dotGeneralAcc`
  (tensor.zig:1353,1418). Head/chunk axes ride as BATCHING axes (shared tags) -- `dotAcc` auto-derives
  batching from shared tags (tensor.zig:1418-1435), so `[b,h,C,C] . [b,h,C,V]` etc. are one op.
- exp/sigmoid/silu/broadcast/slice/concatenate -- already used throughout model.zig.
- The `(I - A)^{-1}` solve (step 3): `Tensor.triangularSolve` EXISTS (tensor.zig:467-472) BUT is
  restricted to `rank <= 2` (tensor.zig:469) -- it CANNOT run batched over heads/chunks. So DO NOT use
  it; instead implement the inverse as the finite Neumann series `I + A + A^2 + ... + A^{C-1}` (log2(C)
  matmuls via repeated `M = M + M@A` doubling), which is fully batched via `dotAcc`. For C=64 that's ~6
  batched [C x C] matmuls; for the MTP-verify chunk C=Kv=2-4 it is 1-2 tiny matmuls. This is the one
  non-obvious implementation detail.

So lever 2 is a PURE StableHLO rewrite of `zml.nn.GatedDeltaNet.forward` (nn.zig:1320-1404) -- it can
be written and NUMERICALLY VALIDATED ON CPU first (like the W8A8 M0-M3 arc, ZML_W8A8.md:16-45), with
zero plugin/FFI/kernel dependency. That is what makes its win/effort attractive.

### 3.4 Expected win

- Decode s=1: ~0 (a chunk of size 1 == one `step()`). A zero-risk micro-opt in the same edit: SKIP the
  `stablehlo.while` entirely when s==1 and call `step()` directly (removes the loop wrapper for the
  common decode path; the while at nn.zig:1391-1398 runs one iteration today). Estimate a few %.
- MTP verify s=Kv (2-4): the scan stops being Kv sequential steps -> a single chunk of parallel
  matmuls. Removes the scan's contribution to the ~2x verify cost. NOTE this is necessary-but-not-
  sufficient for MTP: the PROJECTIONS still cost ~Kv x (GEMV trap) until lever 3 lands (section 5).
- Prefill s=large: the big one. O(s) sequential tiny steps -> O(s/C) chunk-steps with DPAS-friendly
  matmuls. This directly cuts TTFT and is the dominant prefill cost of the 48 GDN layers. Also the MTP
  prefill (MtpPrefill, model.zig:562-581) benefits.

### 3.5 Numerics / correctness risk

The chunked algebra is mathematically equivalent to the recurrence but REASSOCIATES float ops (the WY
inverse + chunk matmuls vs the per-step rank-1 updates). Keep ALL of g-cumsum, A, the inverse, h, and
v_new in f32 (FLA uses f32 for A and the state, chunk.py:47, chunk_delta_h.py:86; the current zml scan
is already f32, model.zig:1044-1063). Expect rel-l2 parity, NOT bit-identical, at the scan output.
Greedy argmax is usually robust to <1e-3 perturbations, so the END-TO-END token stream should stay
"...Paris." -- but this must be MEASURED, and the C=1/Kv path should be checked to reduce EXACTLY to
the current `step()` (a strong unit anchor). Gate: rel-l2 vs the current scan on random inputs (CPU),
then byte-identical greedy end-to-end on GPU.

### 3.6 If the StableHLO scan is compute-bound -> custom kernel (lever 4)

If P0/A-B shows the StableHLO chunked scan is still compute-bound (many small batched matmuls not
hitting DPAS, or reduce-window cumsum slow), fall back to a hand-written chunked-scan kernel. zml's
only custom-call surface is `zml/kernel.zig` and it currently targets `tpu_custom_call` (kernel.zig:374-379)
-- an XPU path is UNPROVEN (ZML_INT8_PERF_HANDOFF.md:102-104,119-121). The realistic route is the same
as lever 3: an XLA FFI custom call registered for the XPU backend calling a XeTLA/SYCL chunked kernel.
Defer until the StableHLO version is measured.

### 3.7 Concrete steps (lever 2)

1. Add `GatedDeltaNet.forwardChunked(...)` in nn.zig alongside the existing `forward`
   (nn.zig:1320-1404), taking a comptime/So runtime chunk size C. Implement steps 3.2(1-6) with
   `cumulativeSum`, `triangular`, `dotAcc`, and the Neumann-series inverse. Emit the final recurrent
   state (needed for the cache update at model.zig:1199-1202 and MTP rollback, model.zig:626-637).
2. Add a fast s==1 path that bypasses the while and calls `step()` (nn.zig:1298-1311) directly.
3. In `recurrentGatedDeltaRule` (model.zig:1065-1072), call `forwardChunked` instead of `forward` for
   s>1; keep the tag renames (model.zig:1074-1077) identical so the cache/out_proj wiring is unchanged.
4. CPU unit test: rel-l2 of `forwardChunked` vs `forward` over random {s in [1,4,64,256], h=48,
   k=v=128}; assert C=1 reduces to `step()` bit-exact.
5. GPU A/B (attended TP=2): prefill TTFT + decode t/s + byte-identical greedy vs 13.7 baseline; then
   re-measure the MTP verify timer (ZML_MTP_PLAN.md:13) to confirm the scan no longer dominates it.

Effort: MEDIUM (the Neumann inverse + chunk bookkeeping are the fiddly parts; everything is
CPU-validatable). Files: `zml/zml/nn.zig` (new forwardChunked), `zml/examples/llm/models/qwen3_5/model.zig`
(recurrentGatedDeltaRule call site, model.zig:1065). No BUILD/kernel/plugin change.

---

## 4. LEVER 1 -- int8-quantize the 3 big GDN bf16 projections

### 4.1 What to quantize

Only the 3 big ones matter: `in_proj_qkv` (52.4M), `in_proj_z` (31.5M), `out_proj` (31.5M). Leave
`in_proj_b`/`in_proj_a` (0.25M each) and `conv1d_weight` (41K) bf16 -- int8 buys nothing and the 48-row
per-channel scale on b/a is silly. int8'ing the 3 big projections halves 11.1 GB -> ~5.56 GB of bf16
GDN-projection reads per token (section 2), i.e. ~17% of total model bytes.

### 4.2 Checkpoint work (the GDN is bf16 today)

The w8a8-sqgptq `ignore` list excludes `linear_attn` (ZML_W8A8.md:124-127), so no int8 GDN weight
exists. Two options:
- (a) OFFLINE requant (recommended first cut): a preprocessing script produces, per GDN layer,
  `in_proj_qkv.weight` (I8 [10240,5120]) + `in_proj_qkv.weight_scale` (BF16 [10240,1]) and likewise
  for in_proj_z / out_proj, using per-output-channel SYMMETRIC RTN (`scale = max(abs(W), over in-dim)/127;
  W_i8 = round(W/scale).clamp(-127,127)`) -- exactly the scheme QuantizedLinear expects
  (common_quant.zig:1-25, ZML_W8A8.md:118-119,136-145). Merge those tensors into the checkpoint
  (mirror the vision-graft merge pattern, memory vision-retention-in-quants). RTN needs no GPU and no
  calibration; GPTQ (the project default, CLAUDE.md) is an option if RTN coherence is marginal.
- (b) quantize-on-load in the zml loader -- more invasive (zml's `io.load` reflects Tensor fields by
  name and streams bf16 directly, model.zig:106-113; there is no per-tensor transform hook), so prefer (a).

The QuantizedLinear discriminator is presence of `weight_scale` on disk (common_quant.zig:9-18): drop
in the scales and the SAME field type auto-selects the int8 path -- no model-type swap.

### 4.3 Wiring into zml (trivial once the checkpoint has scales)

Change `GatedDeltaNet.initProj` and the 3 big bindings from `zml.nn.Linear` to `common_quant.QuantizedLinear`,
mirroring how SelfAttn/Mlp already do it (model.zig:746-753, model.zig:672-698):
- `initProj` (model.zig:994-996) currently returns `zml.nn.Linear` with `weight_scale=null`. Add a
  QuantizedLinear variant binding `weight` + `maybeCreateTensor("weight_scale", {.dout,.one}, ...)`.
- `in_proj_qkv`/`in_proj_z` are COLUMN-parallel (contract `.d=.replicated`, model.zig:1002-1003) -> use
  the full-W8A8 shared-act path (act_quant=true default, common_quant.zig:69,121-188). They share the
  input `x_in` (model.zig:1110-1111,1148) -> apply the ACT-QUANT DEDUP already used in Mlp/SelfAttn
  (quantAct once, feed both, model.zig:643-664,717-719): quantize `x_in` over `.d` ONCE and share
  across in_proj_qkv + in_proj_z (+ b/a if kept int8). This amortizes the prologue over the projections
  and is the difference between ~1.1x and a useful ratio at M=1 (ZML_W8A8.md:64-71).
- `out_proj` is ROW-parallel (contract `.d=.model`, sharded, model.zig:1006,1195). Same situation as
  o_proj/down_proj: `.weightOnly()` is the DECODE-optimal default (weight int8, bf16 act, one bf16
  all_reduce(SUM)) and `.rowParallelW8A8()` is MEASURED SLOWER at M=1 (common_quant.zig:97-119,
  model.zig:762-768). But note `.weightOnly()` MATERIALIZES a bf16 weight in-graph -> no bandwidth win
  at decode (ZML_W8A8.md:64-71). So out_proj int8 helps VRAM/fit but not decode speed until lever 3.

### 4.4 The kernel dependency (why lever 1 alone is only modest at decode)

With TODAY's ops, int8 GDN projections at decode M=1 give:
- in_proj_qkv/z (column-parallel full-W8A8 `dotAcc`): ~1.1-1.3x on those two GEMVs (act-quant prologue
  tax + GEMV trap; ZML_W8A8.md:64-73). Dedup pushes toward the top of that range.
- out_proj (weightOnly): ~1x or slower at decode (materialization). VRAM win only.
Net decode: maybe +5..12% overall (projections are ~55-70% of GDN-s=1 which is 79% of decode; applying
~1.2x to ~two-thirds of that fraction). Prefill (large M) is where int8 `dotAcc` shines -- ~1.6-2x on
the GDN projections (ZML_W8A8.md:64-71 shows raw-i8 up to 1.86x at M=1024) -- so lever 1's clean,
immediate win is PREFILL/TTFT, matching lever 2.

The full decode payoff requires lever 3 (fused w8a16: bf16 act x int8 weight, dequant in the matmul
EPILOGUE, no act-quant prologue, no weight materialization -- kernels/int8_gemm_w8a16.h:20-156,
ZML_INT8_PERF_HANDOFF.md:77-93). That is the same kernel the 21% full-attn decode path wants, so it is
shared work.

### 4.5 Risk (do not skip)

The scheme AUTHORS excluded linear_attn from quant (ZML_W8A8.md:124-127). The GDN recurrent state
ACCUMULATES over the sequence, so quant error in k/v/beta/g (fed by these projections) can compound.
Per-channel W8 symmetric is usually safe, but this MUST be validated for COHERENCE, not just rel-l2:
the "Paris" greedy probe (ZML_INT8_PERF_HANDOFF.md:214-216) AND ideally a short HumanEval+/degenerate-
output check (memory sglang W8A8 HumanEval+ 0.970/0.933 is the accuracy anchor). If coherence degrades,
fall back to GPTQ (calibrated) for the GDN or keep out_proj bf16.

### 4.6 Concrete steps (lever 1)

1. Offline RTN requant script -> emit int8 weight + bf16 [dout,1] scale for in_proj_qkv/z + out_proj of
   all 48 GDN layers; merge into the checkpoint (vision-graft-style merge).
2. model.zig: QuantizedLinear `initProj` variant; bind the 3 big projections; wire the shared-act
   dedup for in_proj_qkv+in_proj_z (model.zig:1110-1151); `.weightOnly()` on out_proj (model.zig:1195).
3. CPU block-parity test (mirror the M3 block probe, ZML_W8A8.md:33-41): rel-l2 of the quantized GDN
   projections vs bf16 dequant reference.
4. GPU A/B (attended TP=2): byte-identical-or-coherent greedy vs 13.7; decode t/s; prefill TTFT.
   ACCEPT only if coherent AND faster-or-equal at decode (per the shelf rule, CLAUDE.md).

Effort: LOW wiring + MEDIUM checkpoint. Files: an offline requant/merge script; `zml/examples/llm/models/qwen3_5/model.zig`
(initProj model.zig:994, init model.zig:1002-1006, forwardImpl proj call sites model.zig:1111,1148,1195).

---

## 5. Why MTP still will not fully amortize until lever 3 (and Kv is large enough)

MTP is done + byte-exact at 98.8% accept but 0x speedup (ZML_MTP_PLAN.md:8-20). The verify at s=Kv is
~2x a decode because BOTH parts scale with s:
- the SCAN: Kv sequential `step()`s -> FIXED by lever 2 (chunked, one chunk of matmuls).
- the PROJECTIONS: M=Kv GEMV, LINEAR in M (b70-int8-xmx-roofline, ZML_W8A8.md:53) -> needs a DPAS/XMX
  kernel that tiles M=Kv into ONE matmul (lever 3). AND even DPAS int8 only leaves the GEMV regime at
  M>=~8 (ZML_INT8_PERF_HANDOFF.md:168-171), so K=1 (Kv=2) is intrinsically weak; K=3-4 (Kv=4-5) is
  where a tiled kernel starts to amortize, but accept decays 0.84/0.57/0.46 per draft position
  (docs/kernel/12_mtp_specdecode_plan.md via ZML_MTP_PLAN.md:109). Sweep K only AFTER lever 3.

So: lever 2 removes the scan from the verify cost; lever 3 removes the projection cost; together they
let the 98.8%-accept MTP deliver ~1.9x (ZML_INT8_PERF_HANDOFF.md:32,150-155). Neither alone unlocks it.

---

## 6. Profiling to do FIRST (P0 -- one attended TP=2 run decides kernel spend)

The layer-level split (GDN 79% / int8-attn 21%) exists; go one level deeper INSIDE the GDN before
committing. Instrument like the ZML_PROFILE_LAYERS timer (inference.zig Runner + session.zig,
ZML_INT8_PERF_HANDOFF.md:127-133) but bracket the sub-blocks of `GatedDeltaNet.forwardImpl`
(model.zig:1104-1204):

1. Split GDN-s=1 decode time into: (a) the 3 big projections in_proj_qkv+in_proj_z+out_proj
   (model.zig:1111,1148,1195), (b) the f32 recurrent scan (model.zig:1176-1186 -> nn.zig:1320-1404),
   (c) conv1d (model.zig:1121-1145), (d) norm/misc. This picks lever 1 vs lever 2 ordering. Hypothesis:
   proj ~55-70%, scan+conv ~30-45%.
2. Repeat at s=2 (the MTP-verify shape). Report each sub-block's s=2/s=1 ratio: ~1x = amortizes
   (bandwidth-bound); ~2x = linear-in-M (GEMV/compute-bound). This directly confirms which parts block
   MTP and whether lever 3 (kernel) is mandatory. Expect projections and scan both ~2x today.
3. Micro-A/B the StableHLO chunked scan (lever 2) in isolation vs the while-scan at s in {2,64,256}:
   is it faster, and does it hit DPAS or stay tiny-batched-matmul-bound? Decides lever 2-vs-4.
4. Confirm int8 GDN projection ratios at the real shapes M in {1,2,4,8} with `//examples/w8a8_sweep`
   (ZML_INT8_PERF_HANDOFF.md:194): in_proj_qkv (K5120 N10240), in_proj_z (K5120 N6144), out_proj
   (K6144 N5120). Validates lever 1's expected win and the M>=8 DPAS crossover for MTP-K sizing.

---

## 7. Recommended order of execution

1. P0 profiling (section 6) -- one attended TP=2 run.
2. IN PARALLEL, both CPU-validatable, no kernel: lever 2 (chunked scan in StableHLO) + lever 1 (int8 GDN
   projections via existing dotAcc + offline requant). Land whichever P0 says is the bigger GDN-s=1
   fraction first. Both also deliver immediate large PREFILL/TTFT wins.
3. Lever 3 (fused w8a16 XLA-FFI/oneDNN kernel) -- the session's main kernel deliverable; shared by the
   GDN projections and the 21% full-attn path; the prerequisite for MTP projection amortization.
4. Re-measure MTP drive (expect ~1.9x once 2+3 land); then sweep K and trim the ~10ms/iter overhead
   (ZML_INT8_PERF_HANDOFF.md:150-155).
5. Lever 4 (custom XeTLA chunked-scan kernel) ONLY if lever 2's StableHLO version is measured
   compute-bound.

Validation is non-negotiable at every step: byte-identical (or coherence-gated for the lossy int8 GDN)
greedy "The capital of France is Paris." vs the 13.7 baseline (ZML_INT8_PERF_HANDOFF.md:212-216,
CLAUDE.md shelf rules), plus microbench %-of-608GB/s (decode) and %-of-367TOPS (prefill).

---

## 8. References (file:line)

- zml GDN model: `zml/examples/llm/models/qwen3_5/model.zig` -- GatedDeltaNet :976-1205, initProj :994,
  init bindings :1002-1007, recurrentGatedDeltaRule :1032-1078, forwardImpl :1104-1204, conv1d
  :1121-1145, out_proj :1195; SelfAttn W8A8 pattern :746-768; Mlp W8A8 + act-quant dedup :672-724;
  quantAct/projLinear/Act :643-664; MTP GdnSnapshot :626-637, MtpPrefill :562-581.
- zml scan: `zml/zml/nn.zig` -- GatedDeltaNet.step :1298-1311, forward (while) :1320-1404, while
  :1391-1398.
- zml ops: `zml/zml/tensor.zig` -- dotGeneralAcc :1353, dotAcc :1418, cumulativeSum :1641, triangular
  :4098, triangularSolve (rank<=2!) :467-472.
- zml quant: `zml/examples/llm/models/common_quant.zig` -- QuantizedLinear :55, forward :121,
  quantizeActivations :44, forwardQuant :201, weightOnly :91, rowParallelW8A8 :114, usesActQuant :194.
- zml custom-call surface: `zml/zml/kernel.zig` -- tpu_custom_call :374-379.
- FLA chunked reference (/mnt/vm_8tb/b70/build/vllm/vllm/model_executor/layers/): fla/ops/chunk.py
  :23-86 (pipeline); chunk_scaled_dot_kkt.py:74-99 (A); solve_tril.py:82-89 (inverse);
  wy_fast.py:91-116 (w,u); chunk_delta_h.py:131-296 (inter-chunk h); chunk_o.py:89-139 (output);
  fused_recurrent.py:97-123 (decode step == zml step); mamba/gdn/qwen_gdn_linear_attn.py:290-416
  (ChunkGatedDeltaRule dispatch), :969-1016 (forward_xpu).
- Kernels blueprint: `kernels/int8_gemm_w8a16.h` :20-156 (fused bf16-act x int8-weight, epilogue
  dequant -- the lever-3 target); `kernels/int8_gemm_w8a8.h`.
- Docs: `zml/ZML_INT8_PERF_HANDOFF.md` (GDN=79% :56-64, verify blocker :48-53, roofline :164-171,
  integration paths :96-121, plan :123-162); `zml/ZML_MTP_PLAN.md` (STATUS :8-20, GDN wrinkle :67-79,
  accept decay :109); `zml/ZML_W8A8.md` (scheme+ignore :112-146, sweep/roofline :64-73);
  `docs/kernel/21_gdn_spec_capture_issue.md` (GDN + spec-decode FULL-capture upstream block on XPU);
  memory `b70-int8-xmx-roofline`.

---

## 9. Implementation decisions (2026-07-01, deep-session research + being implemented)

Lever 2 (chunked scan) is being implemented in `zml/nn.zig` (`GatedDeltaNet.forwardChunked`) with a
validated numpy reference (machine-precision, rel-L2 ~1e-15 vs the zml `step()` recurrence) driving a
CPU parity test vs the existing while-scan. Decisions, grounded in a dedicated FLA/GDN research pass:

- **Chunk size C=32 (not FLA's 64).** FLA's C=64 is justified by HBM<->SRAM amortization INSIDE one
  fused Triton kernel. For a PURE batched-matmul StableHLO expression (no fused kernel) that
  justification disappears -- every chunk op already round-trips HBM -- leaving raw FLOPs + the CxC
  inverse, both of which FAVOR smaller C. C=32 halves the LCd intra term and quarters the CxC-inverse
  element count vs 64, at the cost of doubling the (cheap, `stablehlo.while`) inter-chunk scan length.
  Keep C a multiple of 16 for DPAS alignment; sweep {16,32,64} on GPU later.
- **fp32 boundaries mirror FLA exactly:** keep in fp32 the log-gate cumsum, the decay exponent
  (exponentiate LATE; all exponents <=0 on the strict-lower triangle so NO overflow and no clamp
  needed), the `A = beta*KK^T*decay` matrix, the entire `(I-A)^-1` Neumann computation, and the state
  accumulation. Inputs q,k,v may be bf16 but the model already upcasts to f32 (model.zig:1044-1063).
  f32 single-chunk-whole-sequence UNDERFLOWS `d_j=exp(cumsum)` for long s (exp(-128)->0) -- which is
  exactly WHY chunking (per-chunk cumsum reset) is mandatory for prefill; short s (decode/verify) is
  safe either way.
- **Inverse = finite Neumann via fp32 iterative DOUBLING**, `ceil(log2 C)` stages (C=32 -> 5), exact by
  nilpotency (A strict-lower => A^C=0). This is O(log C)-depth pure batched-matmul (XLA-friendly),
  strictly better than porting FLA's O(C)-depth blocked forward-substitution. SIGN is load-bearing:
  FLA inverts `(I + A_fla)` with A_fla POSITIVE, so the port uses `N = -A_fla`, `T = sum N^n`.
- **Static tril masks** (`triangular(.{.i,.j}, num_diagonals)`), NOT data-dependent -- a Compiler-First
  SSD paper measured dynamic row-masking dropping throughput 82.8%. num_diagonals=-1 for the strict A
  (step 2), num_diagonals=0 INCLUSIVE for the intra output Aqk (step 6) -- the one-diagonal difference
  is load-bearing (excluding it = 84% error, confirmed empirically).
- **Where it pays / does not:** decode s=1 chunk-size is irrelevant (scan is one step; the 79% is the
  projections). The chunked scan's real wins are (a) PREFILL/TTFT (O(s)->O(s/C) sequential) and (b)
  removing the MTP-verify's sequential-scan dependency so the PROJECTIONS can amortize once the
  batched-M w8a16 GEMM lands (ZML_INT8_GEMM_OPT.md 7.4). It is NOT a single-stream decode win on its
  own -- the M=1 int8 projection bandwidth/layout problem is (ZML_INT8_GEMM_OPT.md 7.1-7.3).
- **Head config (served qwen3.6-27B dense, confirmed from config.json):** 16 k/q heads, 48 v-heads
  (GVA repeat 3), head_k=head_v=128, conv kernel 4, full-attn every 4th layer. (35B/80B variants have
  32 v-heads, ratio 2.)
