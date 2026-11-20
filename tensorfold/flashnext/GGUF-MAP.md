# GGUF-MAP: Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS GGUF -> TensorFold PR #3603 XPU Flash-Next

Author: worker M for Lead A (claude-sonnet-5-5), 2026-11-20. CPU-only, read-mostly research; no GPU touched, no source tree modified.
Status: v1. Companion: DESIGN-A.md (worker L/K consume this), tools/strata_oracle.sh + tools/strata_oracle.py (section 7).

Evidence tags: [V] verified here by reading real tensor bytes / running arithmetic (details in section 8); [R] read in source;
UNVERIFIED = not checked, said explicitly. Every mapping claim carries file:line.

Path shorthands used below:
  PR  = /mnt/vm_8tb/b70/build/tensorfold/fn-model/zig/src/families/flashnext   (xpu_load.zig, xpu_ops.zig, xpu_forward.zig, xpu_store.zig, xpu_ple.zig, cuda_norms.zig)
  CL  = /mnt/vm_8tb/b70/build/tensorfold/fn-model/zig/kernels/xpu/flashnext    (flashnext.cl, fn_mixers.cl, fn_moe_head.cl, fn_dpas.cl)
  ST  = /mnt/vm_8tb/b70/build/strata-0216/src                                  (Strata 0.2.16 source; engine/ = C++/SYCL)
  LL  = ST/engine/third_party/strata-llama/src                                 (llama.cpp fork whose models/qwen4exp.cpp is the graph the GGUF was made for)
  CV  = ST/engine/third_party/strata-gguf-py/converter/conversion              (HF -> GGUF converter, qwen4exp.py + qwen.py)
  CEN = /mnt/vm_8tb/b70/results/tf-flashnext/census/gguf_census.json           (header census: names, ggml types, shapes, offsets, metadata)
  GGUF1/GGUF2 = /mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf

The single most useful fact: the GGUF was written by CV/qwen4exp.py and is evaluated by LL/models/qwen4exp.cpp. That graph is the
semantic reference (it matches Strata, which runs this exact file). PR3603's kernels were compared against it line by line below.

----------------------------------------------------------------------------------------------------------------------------------
## 1. MISMATCH / ACTION (headline table)

Severity: CRIT = wrong output if ignored; HIGH = must change loader/kernel; MED = needs a decision; LOW = note.

| # | Item | GGUF / Strata | PR3603 | Sev | ACTION |
|---|------|---------------|--------|-----|--------|
| 1 | GDN value-head order | V heads TILED: head index hv = j*16 + k (k = key head 0..15, j = 0..2). Applied to attn_qkv V rows, attn_gate rows, ssm_alpha/beta rows, ssm_a, ssm_dt.bias, ssm_conv1d V channels, ssm_out COLUMNS (CV/qwen.py:453-476, :569-616). Strata pairs v head h with k head h % 16 (ST/engine/include/strata/kernels/gdn.hpp:54-55). | GROUPED: v head hv pairs with k head hv/(NV/NK) (CL/fn_mixers.cl:148); loader takes ONE contiguous V span per rank (PR/xpu_load.zig:384-389). | CRIT | (a) loader: each rank takes 3 V-spans, one per j (table in 3.2); (b) kernel: change fn_gdn_rows pairing to hk = hv % NK (CL/fn_mixers.cl:148). Everything else (state, conv, z, beta, alpha, ssm_a, dt, ssm_out columns) then stays in GGUF local order [j][kl] with no reorder. Do NOT try to un-permute ssm_out columns (128-wide chunks inside 256-element super-blocks). |
| 2 | ssm_a | stores -exp(A_log) (CV/qwen.py:395-396); used as gate = softplus(alpha+dt) * ssm_a (LL/models/qwen4exp.cpp:895-900) | takes A_log: g = exp(-exp(a_log) * softplus(a + dt)) (CL/fn_mixers.cl:137) | HIGH | either kernel g = exp(ssm_a * softplus(a+dt)) (preferred, 1 line), or load a_log = ln(-ssm_a) in f64 -> f32. [V] ln(-ssm_a) returns exact bf16 values (blk.0: 2.203125, 4.34375, 3.296875, ...), so the roundtrip is lossless. |
| 3 | RMSNorm convention | every norm already has the Gemma "+1" folded in, F32 (CV/qwen.py:401-402, CV/qwen4exp.py:133-136 ple + indexer, LL comment qwen4exp.cpp:280 "converter folded each gamma to (1 + w)"). ssm_norm is NOT +1 (excluded, qwen.py:401). | one global offset detected from layer-0..47 attn hc_norm means (PR/xpu_store.zig:417-443, PR/cuda_norms.zig:4-17), added to every normScale tensor. | MED | [V] detectNorms on the GGUF returns offset 0 (48/48 layers mean>0.5, median 1.041) so main-model norms are consumed raw, OK. BUT the MTP source decides: mtp-q2_0.gguf and the HF .bin tensors are zero-centered RAW (need +1), Strata rt dense.bin is +1 folded F32 (section 6). Replace the single store-wide norm_offset by an explicit per-source flag: GGUF main = 0, rt = 0, mtp-q2_0.gguf / HF raw = 1. Never auto-detect across mixed sources. |
| 4 | Weight formats | 11 quant types + BF16/F16/F32 (section 2 table) | MLX affine 4-bit g32 everywhere (Q4 struct, PR/xpu_load.zig:54-60) | HIGH | QW/QE descriptors + per-type kernels (worker K). The dequant maths is in ST/engine/metal/src/ref_quant.cpp and strata-ggml ggml-quants.c. Types per role are in the map (section 3). |
| 5 | BF16-weight GEMV | router, indexer q/k proj, ssm_alpha/beta, ple_key/value, all HC down/up/inject are BF16 in the GGUF | PR linearBf16 means "bf16 activations x Q4 weights" (PR/xpu_ops.zig:38-54); only the router is BF16 (packRouterTiles, PR/xpu_load.zig:310-332) | HIGH | need a BF16-weight linear path (rows 1..16 GEMV + DPAS tile for prefill). HC pair needs it twice per layer; Q8_0 copy is an optional speed trick (row 7). |
| 6 | dtype conversions | ssm_conv1d F32, ssm_norm F32, ple_conv1d F16, ssm_dt.bias F32, ssm_a F32 | cw bf16 [C,4] (xpu_load.zig:402-409), norm bf16 [128] (:419-421), ple_cw bf16 (:469-471), dt/a f32 | LOW | [V] conv1d F32 low 16 bits all zero (exactly bf16), ssm_norm F32 low16 zero, ple_conv1d F16 upcast has low16 zero => F32/F16 -> bf16 is LOSSLESS. dt.bias low16 zero as well. |
| 7 | HC down+inject stitch | hc_{attn,ffn}_down [10240 x 320] BF16 + hc_*_inject [10240 x 4] BF16 -> row-major [320][10240] and [4][10240] (qwen4exp.cpp:208-214) | HcW.down = 324 rows: 320 down rows then 4 inject rows, n_down=324, has_inj=1 (PR/xpu_load.zig:298-307) | MED | concat rows in that order. Strata's Q8_0 HC copies cover down/up only; INJECT STAYS BF16 (ST/engine/sycl/src/core/native_dense.cpp:393-402) and the Q8 copy is EXTRA (prefill GEMMs read the BF16 copy). DESIGN-A 2 assumes Q8 replaces BF16: keep a BF16 copy for R>16 or quantify. |
| 8 | indexer q/k stitch | indexer.q_proj [2560 x 512] + indexer.k_proj [2560 x 128] (CV/qwen4exp.py:123-131 split of index_qk_proj as q = rows[:512], k = rows[512:]) | one index_qk_proj 640 rows: iq rows 0..511 (4 heads x 128) then raw key 512..639 (CL/fn_mixers.cl:226,234) | LOW | stitch q_proj rows then k_proj rows: identical memory image. BF16 (not Q4). |
| 9 | PLE table | per_layer_token_embd IQ4_NL [160 x 320001536] in GGUF2, ONE contiguous table, no weight_scale | 128 shard tensors of MLX affine 4-bit + weight_scale (PR/xpu_ple.zig:194-200, :349, :431-438) | HIGH | replace dequantRow by IQ4_NL row dequant (split-half nibbles, codebook, fp16 d, section 4g). Row index = global row; drop Hash.locate/shards. HASH ITSELF IS IDENTICAL [V]: PR3603's formula reproduces the GGUF metadata exactly (sizes, offsets, multipliers, padded row count). |
| 10 | Config sources | config.json (PR/xpu_ple.zig:247-292) + model.safetensors.index.json (PR/xpu_store.zig:395-406) | GGUF metadata keys (section 4j) | MED | read keys from GGUF1; every constant PR hard-codes matches (section 4j table). |
| 11 | Embedding / head | token_embd IQ3_S [2560 x 248320] (row = token), output.weight Q5_K [2560 x 248320] (row = token) | embed_bf16 / embed_q4 gathers (CL/flashnext.cl:31-56); head Q4 row slices (PR/xpu_load.zig:550) | HIGH | gather+dequant IQ3_S row into 4 bf16 stream copies; Q5_K head row-sliced per rank (124160 rows each, row slicing is free). |
| 12 | MTP | NOT in the GGUF. Sources: mtp-q2_0.gguf (HF names, BF16 dense, Q2_0 experts) or Strata rt (Q8_0 dense, Q2_0 experts blobs) | MtpW from MLX names language_model.mtp.* (PR/xpu_load.zig:555-577) | HIGH | section 6 gives the field-by-field map. Experts are Q2_0 (same type as 30 main down layers). Draft vocab: int32 file, not text. |
| 13 | MTP attention | Strata drafter attends DENSELY over all cells (ST/engine/src/core/mtp.cpp:1194, no indexer) | PR runs the MTP layer through QSA attnRank with indexer (PR/xpu_forward.zig:441) | LOW | identical for context <= 2051; beyond that only acceptance rate differs, not correctness. Keep PR's QSA (cheaper). |
| 14 | Stop tokens | engine stops at 248044 AND 248046 (ST/engine/sycl/src/program/generate.cpp:13029; tokenizer.ggml.eos_token_id = 248046 = <|im_end|>, ple.eos = 248044 = <|endoftext|>) | Engine.isEos(token) == r.eos == 248044 only (PR/xpu_engine.zig:90-92) | MED | stop set {248044, 248046} from GGUF tokenizer metadata. PLE hash EOS stays 248044 (ple.eos_token_id). |
| 15 | Vision | not in GGUF1 (mmproj is a separate file); GGUF has no ple.image_token_id -> llama falls back to EOS 248044 for image positions (qwen4exp.cpp:1072-1074) | text only (native/xpu.zig:216-219 refuses --vision) | LOW | out of scope for the port; text-only. |
| 16 | QSA details | block score has no /sqrt(128) (qwen4exp.cpp:646-660) | divides by sqrt(128) (CL/fn_mixers.cl:289,306) | NONE | positive constant, same ordering. Dense threshold identical (<=2051 cells). Tie order UNVERIFIED. |
| 17 | KV precision | oracle plain mode = fp16 KV; qualified Strata = int8 KV | bf16 KV | LOW | compare at logit level (top-1 agreement/KL), not bitwise. |
| 18 | Tokenizer / template | in GGUF1 metadata (tokens 248320, merges 247587, pre qwen35, chat_template); also extracted to ST-side dir /mnt/vm_8tb/b70/build/strata-ab/packs/ista-gsq-rco-iq3-xxs/tokenizer | xpu_main takes token ids from a file (PR/xpu_main.zig:30-37); the TF server core reads the tokenizer from the model dir (UNVERIFIED which files) | MED | section 4h. |

----------------------------------------------------------------------------------------------------------------------------------
## 2. (a) Dimension and memory order, and the GGUF type table

ggml shape [ne0, ne1, ne2]: ne0 is the FASTEST axis. A 2-D weight [ne0, ne1] is ne1 rows of ne0 contiguous elements = [out][in] row-major,
exactly the MLX [out, in] layout PR3603 expects; block formats run along ne0 (the input axis). A 3-D expert tensor [ne0, ne1, ne2] is
[expert][out][in] with experts contiguous (expert e at e * ne1 * row_bytes). Evidence: ST/engine/include/strata/kernels/gdn.hpp:63-69
(ssm_conv1d ne=[4,10240] "ne0 = d_conv is the FAST axis"), ST/engine/include/strata/core/family_layout.hpp:149-236 (the shapes Strata
checks), LL/models/qwen4exp.cpp:157-258 (creation shapes), CEN (shapes below).

| GGUF tensor (CEN shape [ne0,ne1,..]) | row-major reading | PR3603 shape it replaces |
|---|---|---|
| attn_qkv [2560,10240] | 10240 rows x 2560 in | in_proj_qkv [10240,2560] (xpu_load.zig:392) |
| attn_gate [2560,6144] | 6144 x 2560 | in_proj_z [6144,2560] (:393) |
| ssm_out [6144,2560] | 2560 rows x 6144 in | out_proj [2560,6144] (:397) |
| attn_q [2560,12288] | 12288 x 2560 | q_proj [12288,2560] (:432) |
| attn_k, attn_v [2560,512] | 512 x 2560 | k_proj, v_proj (:433-434) |
| attn_output [6144,2560] | 2560 x 6144 | o_proj (:437) |
| ffn_gate_exps, ffn_up_exps [2560,640,512] | [512][640][2560] | switch_mlp.gate_proj/up_proj [512,640,*] (:475-476) |
| ffn_down_exps [640,2560,512] | [512][2560][640] | switch_mlp.down_proj (:477) |
| ffn_{gate,up}_shexp [2560,640], ffn_down_shexp [640,2560] | 640 x 2560 ; 2560 x 640 | shared_expert.{gate,up,down}_proj (:479-481) |
| ffn_gate_inp [2560,512] BF16 | 512 x 2560 | mlp.gate.weight [512,2560] (:346-348) |
| ffn_gate_inp_shexp [2560] BF16 (1-D) | 1 x 2560 | shared_expert_gate [1,2560] (:352-354) |
| hc_*_down [10240,320] | 320 x 10240 | input_mix_weight_down [320,10240] (:300) |
| hc_*_up [320,10240] | 10240 x 320 | input_mix_weight_up [10240,320] (:306) |
| hc_*_inject [10240,4] | 4 x 10240 | block_inject_weight [4,10240] (:301) |
| hc_*_norm, output_hc_norm [10240] F32 | flat [4 streams][2560] (stream-major) | hc_norm.weight [10240] -> scale (:292-294) |
| ple_key [2560,10240] BF16 | 10240 x 2560 | ple.key_proj [10240,2560] (:457) |
| ple_value [2560,2560] BF16 | 2560 x 2560 | ple.value_proj (:458) |
| ssm_conv1d [4,10240] F32 | [10240 ch][4 taps] | conv1d.weight [10240,4] bf16 (:399-409) |
| ple_conv1d [4,10240] F16 | [10240][4] | ple conv1d [10240,4] bf16 (:469-471) |
| token_embd [2560,248320] IQ3_S, output.weight [2560,248320] Q5_K | 248320 rows x 2560 | embed_tokens, lm_head |
| per_layer_token_embd [160,320001536] IQ4_NL (GGUF2) | 320001536 rows x 160 (90 B/row) | ngram shards |

ggml type sizes (bytes per block of elements; [V] bytes/element recomputed from CEN for every tensor): Q4_K 144/256, Q5_K 176/256,
Q6_K 210/256, IQ4_XS 136/256, IQ4_NL 18/32, IQ3_S 110/256, IQ3_XXS 98/256, IQ2_S 82/256, IQ2_XS 74/256, IQ2_XXS 66/256, Q2_0 (type 42)
18/64 (2 B fp16 d + 16 B of 2-bit codes), Q8_0 34/32, BF16 2, F16 2, F32 4. Alignment rule for column slicing of a quantized row: block
size of the type (256 for K/IQ3/IQ2/IQ4_XS, 32 for IQ4_NL, 64 for Q2_0).

Types per role in GGUF1 (counts over the 48 layers; from CEN):
  attn_qkv (36 GDN layers): IQ4_XS 11, Q6_K 14, Q4_K 8, Q5_K 3      attn_gate: IQ3_S 10, Q6_K 5, IQ4_XS 11, Q4_K 7, Q5_K 3
  ssm_out (36): IQ4_XS 6, IQ3_S 3, Q4_K 4, Q5_K 4, Q6_K 19          attn_q (12): IQ4_XS 4, Q4_K 3, Q6_K 4, IQ3_S 1
  attn_k (12): IQ4_XS 2, Q4_K 2, Q6_K 6, Q5_K 1, IQ3_S 1            attn_v (12): Q6_K 9, Q4_K 1, IQ4_XS 2
  attn_output (12): Q6_K 10, Q5_K 1, Q4_K 1
  ffn_gate/up_exps: IQ2_XXS 9, IQ2_XS 10, IQ2_S 10, IQ3_XXS 6, IQ3_S 13 (gate and up use the SAME type in a layer [V])
  ffn_down_exps: Q2_0 30, IQ4_NL 18      ffn_down_shexp: IQ4_NL 40, Q2_0 8      ffn_gate_shexp: IQ4_XS 21, Q4_K 12, IQ3_S 13, Q5_K 1, Q6_K 1
  ffn_up_shexp: IQ3_S 27, Q5_K 1, Q4_K 7, Q6_K 3, IQ4_XS 10        token_embd IQ3_S, output Q5_K, PLE IQ4_NL
  BF16: hc_*_down/up/inject, output_hc_down/up, ffn_gate_inp, ffn_gate_inp_shexp, ssm_alpha, ssm_beta, indexer q_proj/k_proj, ple_key, ple_value
  F32: all norms (hc_*_norm, output_hc_norm, q/k_norm, indexer norms, ple norms, ssm_norm), ssm_a, ssm_dt.bias, ssm_conv1d.   F16: ple_conv1d.
Layer kinds: full attention at layers 3,7,...,47 (blk.N has attn_q/k/v/output + q/k_norm + indexer.*), GDN elsewhere [V]; PR linearLayer(l)=l%4!=3 (PR/xpu_store.zig:42-44) matches; GGUF
compress_ratios = [0,0,0,4,...] and ple.layers = [1] (CEN metadata). PLE tensors (ple_*) exist only in blk.1 [V]; PR ple_layer=1 (xpu_store.zig:31).

----------------------------------------------------------------------------------------------------------------------------------
## 3. Per-role map: GGUF tensor -> PR3603 field -> consumer -> transform

Common transform for every quantized matrix: keep GGUF block bytes row-major, descriptor QW{t, n=ne1 rows, k=ne0, row_bytes} (DESIGN-A 8);
row slicing = byte range, column slicing only on block boundaries (section 5). PR consumers: linear = xpu_ops.zig:linearBf16/linearF32 (38-72).

### 3.1 Global tensors

| GGUF | PR3603 field (xpu_load.zig) | kernel / consumer | transform |
|---|---|---|---|
| token_embd.weight IQ3_S [2560,248320] | RankWeights.embed EmbedW (:70-75, loaded :521-535) | fn_embed_bf16 / fn_embed_q4 (CL/flashnext.cl:31-56) copy ONE row into 4 stream slots; xpu_forward.zig:195-200 | new IQ3_S gather+dequant kernel (or dequant table to bf16 at load: 1.27 GB/card). No scaling of the embedding (LL qwen4exp.cpp:362 build_inp_embd, no scale). Also the MTP embedding branch (xpu_forward.zig:421-426). |
| output.weight Q5_K [2560,248320] | RankWeights.head Q4 (:550), row slice rk*vocab_per (124160 for TP2) | linearBf16(head, mixed) xpu_forward.zig:299; logits_part/best CL/flashnext.cl:131-203 | Q5_K row slice, free. Also source of the MTP draft head: rows for draft ids (xpu_load.zig:489-508). Strata: mtp.hpp:342 "main head's rows for a token subset". |
| output_hc_norm [10240] F32 | RankWeights.mixer HcW.scale (:549, loadHcW :292-296) | fn_hc_writeback_norm scale (CL/flashnext.cl:113) | none (offset 0, section 4c) |
| output_hc_down [10240,320] BF16 | mixer.down (320 rows, no inject) (:304) | fn_hc_down_1 (CL/fn_dpas.cl:257-283) | none |
| output_hc_up [320,10240] BF16 | mixer.up (:306) | fn_hc_upmix_1 (CL/fn_dpas.cl:298-327) | none |
| (rope) rope.freq_base 1e7, rope.dimension_count 64 | RankWeights.inv_freq f32[32] (:516-517) | apply_rope (CL/fn_mixers.cl:207-213) | none, computed (section 4d) |

### 3.2 Per-layer common (all 48 layers)

| GGUF blk.N.* | PR3603 LayerW field | consumer | transform |
|---|---|---|---|
| hc_attn_norm [10240] F32 | attn_hc.scale | hcReadout -> fn_hc_writeback_norm (xpu_ops.zig:75) | none |
| hc_attn_down [10240,320] + hc_attn_inject [10240,4] BF16 | attn_hc.down (324 rows: down rows 0..319, inject rows 320..323), n_down=324, has_inj=1 | fn_hc_down_1/wide (CL/fn_dpas.cl:257-296): cols<320 -> ACT=silu(0.25*dot) ; cols 320..323 -> INJ=2*sigmoid(0.25*dot) | concatenate rows [down;inject]; BF16 (or Q8_0 for down, BF16 inject) |
| hc_attn_up [320,10240] BF16 | attn_hc.up | fn_hc_upmix_1/dpas (CL/fn_dpas.cl:298-370): mixed = 0.25 * sum_s sigmoid(up_s . act) * norm_s | none |
| hc_ffn_norm / hc_ffn_down / hc_ffn_inject / hc_ffn_up | mlp_hc.{scale,down,up} | same kernels, hcReadout(mlp_hc,...) (xpu_forward.zig:264) | same as attn_hc |
| ffn_gate_inp [2560,512] BF16 + ffn_gate_inp_shexp [2560] BF16 | LayerW.router = packRouterTiles of [513][2560]: rows 0..511 = ffn_gate_inp, row 512 = ffn_gate_inp_shexp (xpu_load.zig:344-360) | fn_router_dpas_* then fn_route_topk (CL/fn_moe_head.cl:4-58) | row concat (512 + 1) then tile pack; dtype already bf16 |
| ffn_gate_exps / ffn_up_exps [2560,640,512] (IQ2_*/IQ3_*) | routed_gate / routed_up (rank slice first_e.. local_e) | fn_moe_up_1/dpas: silu(gate)*up (CL/fn_moe_head.cl:137-240) | expert-major contiguous, split gate/up already (no fused gate_up); QE descriptor; rank slice = byte range e*n*row_bytes |
| ffn_down_exps [640,2560,512] (Q2_0 / IQ4_NL) | routed_down | fn_moe_down_* (:243-331) | same |
| ffn_gate_shexp, ffn_up_shexp [2560,640] | shared_gate/up (row slice rk*320, len 320) | fn_moe_sh_up_* (:334-417) | row slice free |
| ffn_down_shexp [640,2560] | shared_down (column slice rk*320, len 320) | linearF32 (xpu_ops.zig:232) | column slice 320 = 10 IQ4_NL blocks or 5 Q2_0 blocks, aligned for both types present [V] |
| (shared expert gate) | router row 512 (above) | WTS[10]=sigmoid(row[512]) (CL/fn_moe_head.cl:56) | none |

### 3.3 GDN layers (36 layers: l % 4 != 3)

PR3603 projection image per rank (the single linearBf16 on a.mixed, xpu_ops.zig:104) = rows [ q (NK*128) | k (NK*128) | v (NV*128) | z (NV*128) | b (NV) | a (NV) ]
(xpu_load.zig:391-396; consumed by fn_gdn_front/fn_gdn_norm CL/fn_mixers.cl:102-141,186-199: b at C+NV*128+hv, a at C+NV*128+NV+hv, z at C+hv*128+t).

| GGUF | PR3603 | transform (TP2 rank r owns key heads 8r..8r+7 = 8 key heads, 24 value heads) |
|---|---|---|
| attn_qkv [2560,10240] | in_proj_qkv rows [q 2048 | k 2048 | v 6144] (PR/xpu_load.zig:384-392) | q rows r*1024 .. +1024 ; k rows 2048 + r*1024 .. +1024 ; v rows: THREE spans, for j=0,1,2: 4096 + j*2048 + r*1024 .. +1024 (tiled order, section 4b). Local v order = [j][kl], local hv = j*8 + kl. |
| attn_gate [2560,6144] (= z) | in_proj_z | three spans j*2048 + r*1024 .. +1024 (rows) |
| ssm_beta [2560,48] BF16 (= in_proj_b) | in_proj_b slot "b" (beta = sigmoid(b)) | rows {j*16 + 8r .. +8}, j=0..2 |
| ssm_alpha [2560,48] BF16 (= in_proj_a) | in_proj_a slot "a" | same rows as beta |
| ssm_a [48] F32 | a_log (xpu_load.zig:411-416) | local index set as beta; kernel uses ssm_a directly (row 2 of section 1) |
| ssm_dt.bias [48] F32 | dt (:413-417) | same index set |
| ssm_conv1d [4,10240] F32 | cw bf16 [gdn_conv][4] (:399-409) | channels: q [r*1024,+1024), k [2048+r*1024,+1024), v: 4096 + j*2048 + r*1024 (3 spans); convert F32 -> bf16 (lossless [V]); tap order unchanged (GGUF tap index i = PR tap, oldest first, fn_mixers.cl:93-96 vs gdn.hpp:69-71) |
| ssm_norm [128] F32 | norm bf16 [128] (:419-421) | F32 -> bf16 lossless [V]; NOT +1 |
| ssm_out [6144,2560] | out (column slice, :397) | columns j*2048 + r*1024 .. +1024 for j=0..2 concatenated in j order (local input order [j][kl][128], which is the order fn_gdn_norm writes when hv_local = j*8 + kl); each span = 4 super-blocks of 256 [V alignment ok for IQ4_XS/IQ3_S/Q4_K/Q5_K/Q6_K] |

### 3.4 Full-attention layers (12 layers: 3,7,...,47)

PR3603 projection image per rank: [ q/gate (NQ*512) | k (NKV*256) | v (NKV*256) | iq (4*128) | ik (128) ] (xpu_load.zig:431-436, CL/fn_mixers.cl:225-252).

| GGUF | PR3603 | transform |
|---|---|---|
| attn_q [2560,12288] | q_proj rows rk*6144 .. +6144 (12 heads, each 512 = [q 256 | gate 256]) | row slice; interleaving already identical (qwen4exp.cpp:218,794-816; CL/fn_mixers.cl:232,481) |
| attn_k [2560,512], attn_v [2560,512] | k_proj/v_proj rows kv_idx*256 .. +256 (kv_idx = rk for TP2) | row slice |
| indexer.q_proj [2560,512] + indexer.k_proj [2560,128] BF16 | index_qk_proj 640 rows (replicated on every rank, :430,435) | stitch q rows (4 heads x 128) then k rows |
| attn_output [6144,2560] | o_proj (column slice rk*3072 .. +3072, :437) | 3072 = 12 super-blocks, aligned [V] |
| attn_q_norm [256] F32 | q_scale f32[256] | none |
| attn_k_norm [256] F32 | k_scale | none |
| indexer.q_norm [128] F32 | i_scale | none |
| indexer.k_norm [128] F32 | ik_scale | none |

### 3.5 PLE (layer 1 only: blk.1.ple_* + GGUF2)

| GGUF | PR3603 | consumer | transform |
|---|---|---|---|
| blk.1.ple_key [2560,10240] BF16 | layers[1].ple_key (:457) | linearBf16(ple_key, ple_emb) xpu_ops.zig:244 | BF16 path (row 5) |
| blk.1.ple_value [2560,2560] BF16 | ple_val (:458) | xpu_ops.zig:245 | same |
| blk.1.ple_norm_key / ple_norm_query / ple_norm_conv [10240] F32 | ple_nk / ple_nq / ple_nc | fn_ple_gate (CL/fn_mixers.cl:4-43), fn_ple_conv (:46-74) | none (+1 already folded, CV/qwen4exp.py:134-136) |
| blk.1.ple_conv1d [4,10240] F16 | ple_cw bf16 [10240][4] (:469-471) | fn_ple_conv (taps 4, dilation 3) | F16 -> bf16 lossless [V] |
| per_layer_token_embd (GGUF2) IQ4_NL [160,320001536] | Ngram table (shards) | Ngram.rows/fill (xpu_ple.zig:431-476) host gather -> [rows,2560] bf16 | section 4g |

----------------------------------------------------------------------------------------------------------------------------------
## 4. Detailed findings (a..j of the brief)

### 4b. GDN layout, q/k/v/z/a/b, conv, ssm_a, dt, norm

- attn_qkv rows are [q (16 heads x 128) | k (16 x 128) | v (48 x 128)], same span order as PR3603 (CV/qwen.py:578-586 reorders only the v part; LL
  qwen4exp.cpp:931-947 slices q at 0, k at 2048, v at 4096; PR xpu_load.zig:384-390). q/k need no permutation.
- v heads ARE permuted. HF stores v heads grouped by key head (g0_v0..v2, g1_v0..v2, ...); the converter reorders "from grouped to tiled" because
  num_k_heads (16) != num_v_heads (48): new index = j*16 + k (CV/qwen.py:455-476 docstring + _reorder_v_heads; applied at :578-616 to in_proj_qkv V
  rows, in_proj_z rows, in_proj_a/b rows, A_log/dt_bias, conv1d V channels, out_proj columns). Strata's kernels use h % h_k (ST gdn.hpp:54-55,
  ST/engine/metal/tests/gdn_parity.cpp:145 hk = h % HK). PR3603 uses hk = hv / (NV/NK) (CL/fn_mixers.cl:148) = grouped. => row 1 of section 1.
- attn_gate = in_proj_z, ssm_beta = in_proj_b, ssm_alpha = in_proj_a [R] (ST/engine/third_party/strata-gguf-py/gguf/tensor_mapping.py:393,899-901,926-928;
  graph: beta = sigmoid(ssm_beta x) qwen4exp.cpp:884-889, alpha = ssm_alpha x then softplus(alpha + dt) * ssm_a :891-900). PR3603: b -> sigmoid, a -> softplus(a+dt)
  (CL/fn_mixers.cl:134-138), with in_proj_b stitched before in_proj_a (xpu_load.zig:394-395). Same roles, same order. GDN output gate is sigmoid(z) in both
  (qwen4exp.cpp:476-486, CL/fn_mixers.cl:197). q is L2-normalised and scaled 1/sqrt(128), k L2-normalised (CL/fn_mixers.cl:124-127; Strata gdn.hpp:47).
- ssm_conv1d F32 [4,10240]: memory [channel][tap] with tap 3 = newest (gdn.hpp:63-71) == PR cw[c*4+tap], x(at=r+tap) oldest first (CL/fn_mixers.cl:93-96). Same
  orientation as HF conv1d.weight [10240,1,4] squeezed (CV/qwen.py:399-400 squeeze). Only dtype (F32 -> bf16, lossless) and the v-channel order (tiled) change.
- ssm_a = -exp(A_log) (CV/qwen.py:395-396) and A_log is recovered exactly as ln(-ssm_a) [V]. ssm_dt.bias = dt_bias (renamed dt_proj.bias at CV/qwen.py:397-398 then
  mapped to ssm_dt.bias), only reordered. ssm_norm = linear_attn.norm.weight raw (no +1, CV/qwen.py:401), PR3603 uploads it raw (xpu_load.zig:419-421). eps 1e-6.

### 4c. Norm weights (+1 offset)

CV/qwen.py:401-402: any tensor whose name ends in "norm.weight" except linear_attn.norm.weight gets +1 (covers q_norm, k_norm, hc_*_norm, output_hc_norm and the MTP
norms in the main-model converter); CV/qwen4exp.py:133-136 adds +1 to ple.norm_key/norm_query/norm_conv and indexer.q_layernorm/k_layernorm (not caught by the generic
rule). So every norm in the GGUF is "direct" (effective weight = stored value), and LL applies them as plain ggml_mul(rms_norm(x), w) (qwen4exp.cpp:281, 799, 809, 626, 638,
1220). [V] measured means of the stored F32 tensors: hc_attn_norm L0 0.937, L47 1.038, all 48 attn hc_norm means in [0.676, 1.765] (48/48 above 0.5, median 1.041);
hc_ffn_norm L5 1.336; output_hc_norm 3.750; attn_q_norm 1.283; attn_k_norm 1.273; indexer q/k norm 0.963/0.959; ple_norm_key/query 0.893, ple_norm_conv 0.891;
ssm_norm 0.967 (raw).

PR3603: Store.detectNorms (xpu_store.zig:417-433) takes the 48 attn hc_norm means and cuda_norms.offset (cuda_norms.zig:4-17): 0 if >=90% of means > 0.5 and median in
[0.75,1.5] ("direct"), 1 if <=10% above 0.5 and median in [-0.5,0.25] ("centered"); normScale adds the one offset to every normScale tensor (xpu_store.zig:435-443):
hc_norm (xpu_load.zig:292), q/k_norm (439-442), indexer q/k layernorm (443-446), ple norms (459-464), MTP pre_fc_norm_* (556-559). [V] emulated on the GGUF => offset 0 (48/48 > 0.5).

| norm | GGUF tensor | PR field | convention in GGUF |
|---|---|---|---|
| hc attn/ffn | blk.N.hc_attn_norm, hc_ffn_norm [10240] | HcW.scale | direct (+1 folded) |
| final mixer | output_hc_norm [10240] | mixer.scale | direct |
| attention q/k | attn_q_norm, attn_k_norm [256] | q_scale, k_scale | direct |
| indexer q/k | indexer.q_norm, indexer.k_norm [128] | i_scale, ik_scale | direct |
| PLE | blk.1.ple_norm_key/query/conv [10240] | ple_nk, ple_nq, ple_nc | direct |
| GDN out | ssm_norm [128] | norm (bf16, raw) | raw, never offset |
| MTP | (not in GGUF) | norm_e, norm_h + layer/mixer norms | see section 6 (source dependent) |

### 4d. Attention

- attn_q [2560,12288]: per-head [q 256 | gate 256] interleaved (qwen4exp.cpp:218 comment "wq holds [q|gate] interleaved per head"; views :794-816 with stride 2*head_dim). PR3603
  q_proj reads head*512 for q and +256 for the gate (CL/fn_mixers.cl:232, :481). Identical.
- attn_k/attn_v: rows [kv head][256] (n_head_kv=2). k_norm per head over 256, v raw (qwen4exp.cpp:808-819; CL/fn_mixers.cl:241-249). Identical.
- indexer: q_proj 512 = 4 heads x 128, k_proj 128; q: norm(128) then RoPE; k: stored RAW, mean-pooled over blocks of ratio 4, THEN normed and RoPE'd at the
  block's first position (qwen4exp.cpp:598-642; block position = pb*r, LL/llama-memory-hybrid-idx.cpp:477 sec_pos = pb*r, :484 only for ranked 2D image cells). Score = sum_h relu(q_h . pooled_b)
  (:646-660), top-k over cells of width min(n_kv, indexer_top_k + r - 1) = 2051 (:682). PR3603: qsa_pool mean(4) -> norm -> rope at RATIO*b (CL/fn_mixers.cl:256-275); scores
  /sqrt(128) (positive scale, no effect on selection; CL/fn_mixers.cl:289,306); select 512 blocks + tail, only when complete blocks > 512 (:317-417) = dense for <=2051 cells.
  Same behaviour; constants match: GGUF indexer.top_k 2048 = 512 blocks x ratio 4 (compress_ratios entry 4 on the 12 attention layers, 0 on GDN layers) vs
  PR index_ratio 4, index_top 512, index_idw 2052, max_nkeys 2051 (PR/xpu_store.zig:27-30; xpu_ops.zig:170).
- RoPE: GGUF rope.dimension_count 64 (= head_dim 256 x partial_rotary 0.25), freq_base 1e7, rope.dimension_sections [11,11,10,0] (interleaved M-RoPE, text default
  [11,11,10,0], CV/qwen.py:621-626). For text-only input t=h=w=pos (Strata fills put(c,c,c,c): ST/engine/sycl/src/program/generate.cpp:10816-10817), M-RoPE reduces to ordinary
  NEOX RoPE over the first 64 dims, pairing d with d+32. PR3603: inv_freq[i] = 1e7^(-2i/64), i<32 (xpu_load.zig:516-517), pairing d<->d+32 for d<64 (CL/fn_mixers.cl:203-213),
  q/k/iq and pooled keys all rotated. Identical for text. attn scale 1/16 (CL/fn_mixers.cl:428) = 1/sqrt(256) (qwen4exp.cpp:838). Sigmoid output gate (qwen4exp.cpp:849-852,
  CL/fn_mixers.cl:478-483).
- compress_ratios [0,0,0,4,0,0,0,4,...,4] (48 entries) = layer l has an indexer iff l%4==3.

### 4e. Hyper-connections

Reference math (qwen4exp.cpp:267-351), every step matches PR3603:
  xn = rmsnorm_per_stream(x) * gamma (gamma [4][2560], eps 1e-6)            PR: fn_hc_writeback_norm (CL/flashnext.cl:109-114)
  lo = silu( (down . xn_flat) / 4 )                      (320 values)       PR: fn_hc_down_1 ACT = silu(rbf(rbf(sum)*0.25)) (CL/fn_dpas.cl:278-279)
  gate = up . lo (10240 = 4 streams x 2560); mixed = (1/4) sum_s xn_s * sigmoid(gate_s)         PR: fn_hc_upmix_1 (CL/fn_dpas.cl:298-327)
  inject = inject_w . xn_flat ; w = 2*sigmoid(inject/4)  (4 values)         PR: INJ = 2*sigmoid(0.25*dot) (CL/fn_dpas.cl:280)
  x_s += block_out * w_s  (residual update per stream)                       PR: fn_hc_writeback_norm/only: H += branch*inj (CL/flashnext.cl:100-105, 123-127)
  final mixer = same readout without inject; output = mixed (NO further norm) -> output.weight      PR: hcReadout(mixer) then head linear (xpu_forward.zig:298-299)
So hc_{attn,ffn}_down/up/inject/norm and output_hc_down/up/norm map 1:1 onto input_mix_weight_down/up, block_inject_weight, hc_norm, hyper_connection_mixer.*; the only
required transform is the 324-row stitch (down 320 rows + inject 4 rows). The shapes in the GGUF are [hc_dim, hc_lr] = [10240,320] for down (row-major [320][10240]), confirmed
by Strata "gr_read transposes, so the file's orientation is [hc_dim, hc_lr] / [hc_lr, hc_dim]" (ST/engine/include/strata/core/family_layout.hpp:170).
Strata STRATA_HC_Q8=1 (qualified config env): Q8_0 of the BF16 values on the host (ggml reference rounding: fp16 d = amax/127, q = round-half-away(x/d), 32 values/block along the input axis;
down = 320 rows x 320 blocks, up = 10240 rows x 10 blocks), uploaded in a split layout (all quants row-major then all fp16 scales; ST/engine/sycl/include/strata/kernels/fused_gr_q8.hpp:9-12;
loader ST/engine/sycl/src/core/native_dense.cpp:393-402; mixer too, verify.cpp:2063). Inject is NOT quantized and the BF16 copies stay for the prompt path.
Default off in SYCL unless STRATA_HC_Q8=1 (native_dense.cpp:37). The oracle in plain mode sets it to 0 (BF16 reads).

### 4f. MoE

- Router: logits = ffn_gate_inp . x (512, fp32 accumulate) ; PR packs 513 rows (512 + ffn_gate_inp_shexp) (xpu_load.zig:344-360). Routing in both: top-10 of the 512 logits,
  weights = softmax over the selected 10 (LL: softmax over all 512, top-k, renormalise "norm_w=true" => identical; w_scale 0 = none: LL/llama-hparams.h:124,
  llama-graph.cpp:2149; PR: exp(v - top)/sum over the 10 selected, CL/fn_moe_head.cl:43-52). The 11th slot is sigmoid(shared_gate_logit) multiplying the shared expert output
  (LL qwen4exp.cpp:1020-1029; PR WTS[10] = sigmoid(row[512]), CL/fn_moe_head.cl:56; fn_moe_combine adds sh_out*WTS[10], :420-437). Ties: lowest expert id (PR) -- ggml top-k tie order UNVERIFIED.
- Experts: SwiGLU silu(gate)*up then down; separate gate/up/down tensors, expert-major contiguous [V]; no fused gate_up in the main GGUF. Expert count 512, used 10, ffn 640, shared 640 (CEN metadata).
- Shared expert silu(gate)*up -> down, scaled by the sigmoid gate (LL :1010-1029). PR: moe_sh_up then linearF32(shared_down) then combine with weight WTS[10] (xpu_ops.zig:222-237).
- No expert bias (exp_probs_b nullptr, LL :993-1005), no routed scaling factor. HF config also has no norm_topk_prob / routed_scaling_factor key [V /mnt/cache/b70/devmodel/Qwen/Qwen3.8-Flash-Next/config.json].

### 4g. PLE: hash, table layout, dequant (verified)

Metadata (CEN): qwen4exp.ple.layers=[1], ngram_size 3, heads_per_ngram 8, conv_kernel 4, eos_token_id 248044, embedding_length_per_layer_input 160,
layer_multipliers [23703573157769, 20109073645365, 8052911324071], head_vocab_sizes [20000003, 20000023, 20000033, ..., 20000171], head_offsets [0, 20000003, 40000026, ..., 300001275]
(16 heads = (3-1)*8). No weight_scale, no ple.image_token_id, no strata.ple.* keys. [V] PR3603's Hash.init (PR/xpu_ple.zig:56-98) with vocab=248320 (HF config vocab_size), seed=1234,
ple_index=0 (ple_layer_ids=[2] -> layer 1), base=20,000,000, divisor 128, 128 shards reproduces EXACTLY: the 16 sizes (nth prime after 19,999,999, ...), the 16 offsets
(running sums), the 3 multipliers (splitmix64 chain, xpu_ple.zig:91-96), and the padded total 320001536 = GGUF2 row count. (With any other vocab value the multipliers differ.)
The two hashes are the same function:
  ctx[0]=token ; ctx[s]=predecessor s positions back, but once a predecessor (s>=1) equals EOS 248044 (or is missing = start of sequence) that position and every OLDER one is EOS;
  the token's OWN EOS does not cut its own context.   history initial = [eos, eos] (PR/xpu_state.zig:247,351; Strata TOKEN_NULL -> EOS)
  for n in 2..3: mixed_n = (u64)ctx[0]*m[0] ^ (u64)ctx[1]*m[1] ^ ... (n terms, wrap mod 2^64)       (ST/engine/src/kernels/ngram.cpp:100-126; LL qwen4exp.cpp:1098-1124)
                 for g in 0..7: h = (n-2)*8 + g ; row[h] = mixed_n % head_vocab_sizes[h] + head_offsets[h]
PR3603 lastIds (xpu_ple.zig:140-173) computes the same (EOS prefix-max logic equals the cut rule; signed i64 arithmetic equals u64 because m[i] < 2^63/vocab so products never reach 2^63).
GGUF row index = global row of per_layer_token_embd (the converter concatenates shards in index order, CV/qwen4exp.py:145-172), so NO shard/locate step is needed; PR's per-shard
starts[] only matter for MLX shards. Eos: PR takes config.json eos_token_id (xpu_ple.zig:272-279; the local HF config has the scalar 248044 [V]) = GGUF ple.eos_token_id 248044. Note the CONVERTER takes
the LAST element of a list (CV/qwen4exp.py:99-103); generation_config.json has [248046, 248044].
Gather layout: 16 rows x 160 values flattened head-slowest into [2560] per token (head h occupies [h*160,(h+1)*160); ST ngram.hpp:236-239; LL qwen4exp.cpp:1198-1201); PR Ngram.fill writes
out[i*dh ..] with items i = token*heads + head (xpu_ple.zig:431-438, 454-476) -> identical image [tokens][2560] bf16.
Table dtype: IQ4_NL, 90 bytes/row = 5 blocks x 18 B: fp16 d then 16 bytes; SPLIT-HALF nibbles: byte j holds element j (low nibble) and j+16 (high nibble) of the 32-block;
value = d * kvalues_iq4nl[nibble], kvalues = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113} (no -8) (ST/engine/src/kernels/ngram.cpp:134-147, ngram.hpp:130-145). No row scale
(needs_scale false for IQ4_NL, ngram.hpp:66-73). Output to bf16: Strata dequantizes to float; PR rounds to bf16 (xpu_ple.zig:38) -- round-to-nearest-even of d*kv.
GGUF2: header 192 bytes (data_start=192 [V] CEN), single tensor offset 0, 28,800,138,240 bytes = 320001536 x 90 [V]; mmap GGUF2 and index row*90 + 192.
PLE gating math (identical in both): key=ple_key.emb, value=ple_value.emb, per stream s: rms(key_s)*nk . rms(h_s)*nq / sqrt(2560) -> signed sqrt -> sigmoid -> gated = sig*value;
grouped norm_conv; depthwise causal conv taps 4 dilation 3 with 9-row history, silu; h += gated + conv (LL qwen4exp.cpp:1206-1293; CL/fn_mixers.cl:4-74; ple_tail_n=9 xpu_store.zig:32).
It runs BEFORE layer 1's attn readout and after layer 0's MoE writeback (LL :397-399; PR xpu_forward.zig:231-236).

### 4h. Head, embedding, tokenizer, model-dir needs

- output.weight Q5_K 417 MiB, token_embd IQ3_S 260 MiB (CEN bytes 437043200 / 273152000 [V]); both rows = vocab ids; no tied weights (output is separate; LL falls back to tied only if absent).
- PR3603 xpu_main.zig takes MODEL_DIR + a file of token ids (xpu_main.zig:30-37) and Engine.init reads only: model.safetensors.index.json + shards (xpu_store.zig:395-406, 456-500), config.json (xpu_ple.zig:247-292),
  draft table proposal_table.text (embedded, PR/xpu_load.zig:485-487). The TF server core (native/xpu_flashnext.zig:132 passes o.dir) additionally needs a tokenizer + chat template: UNVERIFIED which files it opens.
- Tokenizer provenance: GGUF1 metadata carries tokenizer.ggml.model=gpt2, pre=qwen35, tokens (248320), token_type, merges (247587), eos_token_id 248046 (<|im_end|>), padding/bos 248044 (<|endoftext|>), add_bos_token false, chat_template.
  Strata already extracted it: /mnt/vm_8tb/b70/build/strata-ab/packs/ista-gsq-rco-iq3-xxs/tokenizer/{vocab.json (248320 entries), merges.txt, token_type.json, tokenizer.json (pre + special ids), chat_template.jinja (9993 B)}
  (reader: ST/terminal/tools/strata_tokenizer.py, pure python + regex). [V] the HF dir /mnt/cache/b70/devmodel/Qwen/Qwen3.8-Flash-Next has vocab.json with 248044 entries (added tokens live in its tokenizer.json) and a chat_template.jinja of 8952 B
  that differs from Strata's 9993 B; merges identical. Recommendation: read tokens/merges/pre/special ids from GGUF1 metadata (or the Strata dir) so the model dir needs only the two GGUF files.

### 4j. Everything else PR3603 hard-codes vs GGUF (all match [V] unless noted)

| PR constant | value | GGUF key / evidence |
|---|---|---|
| layers_n, linear_n, attn_n | 48, 36, 12 (xpu_store.zig:8-10) | block_count 48, full_attention_interval 4 (+ compress_ratios) |
| dims, streams, wide, low | 2560, 4, 10240, 320 | embedding_length, hyper_connection.count 4, .low_rank 320 |
| eps | 1e-6 (:15) | layer_norm_rms_epsilon 9.99999997e-7 (f32 of 1e-6) |
| vocab, experts_n, topk, moe_w | 248320, 512, 10, 640 | token_embd shape, expert_count/used_count, expert_feed_forward_length (shared 640) |
| q_heads, kv_heads, head_dim | 24, 2, 256 | attention.head_count/head_count_kv/key_length |
| gdn: 16 k heads, 48 v heads, 128, taps 4 | | ssm.group_count 16, time_step_rank 48, state_size 128, conv_kernel 4, inner_size 6144 |
| index_heads/dim/ratio/top | 4, 128, 4, 512 blocks | indexer.head_count 4, key_length 128, top_k 2048, compress_ratios 4 |
| ple_layer, ple_tail_n | 1, 9 | ple.layers [1], (4-1)*3 |
| RoPE | theta 1e7, 64 dims | rope.freq_base 1e7, dimension_count 64 |
| shared_expert_gate | sigmoid on a 1x2560 dot | ffn_gate_inp_shexp [2560] |
Not implemented by PR3603 / not needed: image tokens (no vision), M-RoPE sections beyond text, attention sinks (none exist in qwen4exp.cpp), per-layer special cases other than PLE@1 and QSA@l%4==3 (none found).

----------------------------------------------------------------------------------------------------------------------------------
## 5. TP2 slicing rules specific to the GGUF (alignment checklist)

| slice | span | alignment need | status |
|---|---|---|---|
| attn_qkv / attn_gate / ssm_alpha / ssm_beta rows, indexer rows, attn_q/k/v rows, router, ffn_*_shexp gate/up rows, output.weight rows, experts | rows | none (rows are independent in every GGUF type: blocks run along the input axis) | free |
| ssm_out columns, 3 spans of 1024 | 1024 | 256 (all ssm_out types are 256-block: IQ4_XS, IQ3_S, Q4_K, Q5_K, Q6_K) | ok [V] |
| attn_output columns, 3072 per rank | 3072 | 256 (Q6_K/Q5_K/Q4_K) | ok [V] |
| ffn_down_shexp columns, 320 per rank | 320 | 32 (IQ4_NL) / 64 (Q2_0) | ok [V] (320 = 10 / 5 blocks) |
| expert split by id | whole experts | none | ok (balanced placement table in DESIGN-A 4 is a separate permutation) |

TP2 rank r (0/1) numbers: NK=8, NV=24, gdn_conv = 2*1024 + 3072 = 5120 channels, projection rows gdn_pw = 5120 + 3072 + 48 = 8240 (PR/xpu_store.zig:59-97 forTp), attention
NQ=12 heads (rows 6144), NKV=1.

----------------------------------------------------------------------------------------------------------------------------------
## 6. (i) MTP drafter: sources, tensor map, experts layout, draft vocab

The GGUF has no MTP tensors [V CEN: no mtp.* / nextn.* names in either shard]. Two Strata-made sources of the SAME 31 HF tensors (mtp-inventory.md):

A. /mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/mtp-q2_0.gguf (889 MB, plain GGUF v3, HF tensor names, 31 tensors; read by the same GGUF reader as the main file):
   BF16 dense tensors (fc_*, HC, router, shared expert, attention, indexer), F32 RAW (zero-centred, need +1) norms, experts Q2_0:
   mtp.layers.0.mlp.experts.gate_up_proj Q2_0 [2560,1280,512] (per expert: rows 0..639 = gate, 640..1279 = up), mtp.layers.0.mlp.experts.down_proj Q2_0 [640,2560,512].
   [V] listed with gguf-py GGUFReader (header only).  Q2_0 error vs bf16: relative rms 0.42-0.43 on the first experts (mtp-q2_0.gguf.report.json) -- the drafter is intentionally crude.
B. .../mtp-bf16/rt/ (Strata runtime format, tools/mtp_rt.py): dense.bin (+dense.txt index) and experts.bin.
   dense.txt line = name kind rows cols offset bytes into dense.bin, 256-byte aligned. Large projections are Q8_0 (ggml reference rounding), HC + router BF16, norms F32 WITH +1 applied
   (mtp_rt.py:11,89-91,95-99). [V] rt f32 norms equal raw HF bf16 + 1 exactly for all 9 norm tensors (max|rt-(raw+1)|=0); raw HF means: mixer hc_norm 3.79, L0 attn hc_norm 0.275, q_norm 2.68,
   pre_fc_norm_embedding -0.764 (zero-centred) => raw sources NEED +1, rt does not.
   experts.bin = 512 blobs, expert e at e*1,382,400 B (BLOB = 3*(2560*640*18/64); ST/engine/include/strata/kernels/cpu/expert.hpp:49, expert_layout.hpp:35-41). One blob (mtp_rt.py:52-62):
     [ A  gate/up Q2_0 code bytes, 1280 rows interleaved (row 2r = gate r, row 2r+1 = up r), 40 blocks/row x 16 B : 819,200 B ]
     [ B  down Q2_0 code bytes, 2560 rows x 10 blocks x 16 B : 409,600 B ]
     [ C  gate/up fp16 block scales, same row order, 1280 x 40 x 2 B : 102,400 B ]
     [ D  down fp16 block scales, 2560 x 10 x 2 B : 51,200 B ]
   (a lossless relayout of the GGUF Q2_0 blocks: code plane / scale plane instead of 18-byte blocks). Recommendation: use source A (no new container/format logic in the loader; BF16 dense; same Q2_0 kernel as the 30 Q2_0 down layers).
   Dense rt order and shapes are exactly the dense.txt listing (fc_embedding, fc_hidden, hyper_connection_mixer.*, attn_hyper_connection.*, mlp.gate, mlp.shared_expert.*, mlp.shared_expert_gate,
   mlp_hyper_connection.*, self_attn.*, pre_fc_norm_*).

Field map (MtpW + its LayerW; PR/xpu_load.zig:107-117, 555-577; HF names = source A names with prefix mtp.; "rt" names are the dense.txt names):

| MtpW / LayerW field | source A tensor (mtp.*) | rt dense.txt name (kind) | transform |
|---|---|---|---|
| norm_e (f32 [2560]) | pre_fc_norm_embedding.weight F32 | pre_fc_norm_embedding.weight (f32, +1) | A: +1 ; rt: none |
| norm_h (f32 [10240]) | pre_fc_norm_hidden.weight F32 | pre_fc_norm_hidden.weight (f32, +1) | A: +1 ; rt: none. ONE rms over all 10240 (pooled; Strata default hnorm_stream_=false, mtp.hpp:283; PR forwardMtp xpu_forward.zig:431) |
| fc_e [2560,2560] | fc_embedding.weight BF16 | fc_embedding.weight (q8_0) | row-major [out][in]; applied to rmsnorm(embed row) (xpu_forward.zig:427-428; Strata mtp.cpp:1081-1083) |
| fc_h [2560,2560] | fc_hidden.weight BF16 | fc_hidden.weight (q8_0) | applied PER STREAM to the normed 10240 row viewed as 4 x 2560 (xpu_forward.zig:432 streams*m rows; mtp.cpp:1090-1093) |
| (add) | -- | -- | h_s = fc_e(...) + fc_h_s for all 4 streams (xpu_forward.zig:435; mtp.cpp:1094 add_streams_broadcast) |
| layer.attn_hc | layers.0.attn_hyper_connection.{hc_norm F32, input_mix_weight_down [10240,320], input_mix_weight_up, block_inject_weight [10240,4]} | attn_hyper_connection.* (f32 / bf16) | stitch down+inject (324 rows); norm +1 for A |
| layer.mlp_hc | layers.0.mlp_hyper_connection.* | mlp_hyper_connection.* | same |
| layer.router | layers.0.mlp.gate.weight [2560,512] + mlp.shared_expert_gate.weight [2560,1] | mlp.gate.weight (bf16 512x2560) + mlp.shared_expert_gate.weight (bf16 1x2560) | 513 rows |
| layer.routed_gate/up | layers.0.mlp.experts.gate_up_proj [2560,1280,512] | experts.bin planes A/C | split gate (rows 0..639) / up (640..1279) per expert (rt interleaves them 2r/2r+1) |
| layer.routed_down | layers.0.mlp.experts.down_proj [640,2560,512] | experts.bin planes B/D | none |
| layer.shared_gate/up/down | mlp.shared_expert.{gate,up,down}_proj.weight BF16 ([2560,640],[2560,640],[640,2560]) | same names (q8_0) | rank slices as the main layers |
| layer.proj | q_proj [2560,12288], k_proj [2560,512], v_proj [2560,512], indexer.index_qk_proj [2560,640] (already stitched q 512 rows then k 128 rows, same as PR) | same (q8_0) | stitch q|k|v|index rows as section 3.4 |
| layer.out | o_proj [6144,2560] | self_attn.o_proj.weight | column slice |
| layer.{q,k,i,ik}_scale | self_attn.q_norm / k_norm [256], indexer.q_layernorm / k_layernorm [128] F32 | same (f32 +1) | A: +1 |
| mixer (HcW, no inject) | hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up} | hyper_connection_mixer.* | down 320 rows only; norm +1 for A. Strata uses the MTP's own mixer then the MAIN head (mtp.cpp:1339-1371) |
| draft (Q4 -> Q5_K) | -- | -- | rows of the main output.weight (Q5_K) selected by the draft vocab ids; MTP shares embedding (token_embd) and lm_head with the main model [R] (mtp.cpp:1069-1080 token_embd gather; mtp.hpp:342; mtp.cpp:818-832 draft head = subset rows of the main head) |
| draft_lo/draft_len | -- | -- | vocab slice across ranks (TpShape.draftSlice) unchanged |

Draft vocab: Strata uses --mtp-draft-vocab ST/terminal/data/draft_vocab_en.bin = a raw little-endian int32 array with NO header, 40525 token ids, sorted ascending, ids 0..248319 [V]
(ST/engine/src/core/mtp.cpp:818-832, mtp.hpp:68-70). PR3603 builds draft ids from proposal_table.text (PR/xpu_load.zig:485-487; draft_vocab_default.txt: 79591 ids, sorted, max 248076 [V]).
Overlap between the two sets: 39364 ids [V]. Decision for the port: add a draftIdsFromBin(path) and use the Strata file for like-for-like acceptance against Strata; the PR text table is the alternative (2x wider head slice => slower draft head).
Differences that only affect acceptance, not correctness: Strata MTP attention is dense over all cells with a window ring (mtp.cpp:1194, ring 407-409), PR3603 uses QSA for the MTP layer (row 13).

----------------------------------------------------------------------------------------------------------------------------------
## 7. Oracle: reference logits and greedy continuations from Strata 0.2.16

### 7.1 What Strata offers (all [R], SYCL build under ST/engine/sycl)

| tap | where | what you get | used? |
|---|---|---|---|
| serve line protocol: GEN <max_new> [keys] id,id,... ; replies "T <id>" then "LP <logprob> id:logprob ..." per token, "DONE ..." at the end; keys logprobs=K (0..20), allowed=a,b,c, temperature/top_k/seed (absent = greedy) | ST/engine/sycl/src/program/generate.cpp:10700-10790 (parse), 12980-13022 (T/LP emission), 13243 (DONE); client side ST/terminal/serve/server.py:804 ([exe,"--serve",*args]), 2427-2436 (GEN line), 1422-1431 | raw token ids in; greedy tokens + log-prob of the chosen token + top-20 log-probs (from the head logits BEFORE sampling) per generated position | YES (primary) |
| STRATA_LOGITS_DUMP=<path> | ST/engine/sycl/src/core/verify.cpp:3263-3281 | EVERY verify window's full fp32 logits: record = {int32 0x53474F4C, int32 T, int32 n_vocab, int32 record_no, int64 pos0} + T*n_vocab float32; row t = distribution after the token at position pos0+t. File opened "wb" once per process, flushed per record | YES (full logits; first token + top-64/lse per position) |
| STRATA_DUMP_FIRST_LOGITS=<path> | ST/engine/sycl/src/program/generate.cpp:14304-14311 | n_vocab float32 of the FIRST decode window row 0 (= first-token logits after the whole prompt) | generate (non-serve) mode ONLY: not reachable from --serve; not used |
| --dump-logits PATH (generate mode) | generate.cpp:1886, 6124-6140, 13715-13745 | per-position logits with header; comment 6119-6121: a native pack "never reaches the per-token dump site" | NO |
| STRATA_LOGPOS=<path>, STRATA_LOGPOS_TOPK=K, STRATA_LOGPOS_EXTRA | generate.cpp:11425-11441; ST/engine/sycl/src/core/verify.cpp:3410-3463 | teacher-forced log-prob of every prompt token read through the short-read windows (tab separated; top-K list) | optional extra (prompt-side scoring) |
| STRATA_FOLLOW=<file> + STRATA_FOLLOW_LOGPOS=<path> | generate.cpp:11872-11890, 12900 | the request decodes the token ids in <file> instead of its argmax (teacher forcing inside Strata) + logprobs | not needed: TF does its own teacher forcing |
| --dump-residual / --dump-layers / --dump-halves / --dump-routing (generate mode) | generate.cpp usage text ~866-870 | final residual, per-layer R, half-layer block_out/inject, routed expert ids+weights | follow-up bisection tools if top-1 diverges; UNVERIFIED with the native EP configuration |

### 7.2 Design of tools/strata_oracle.{sh,py} (written, NOT run)

- Engine: /mnt/vm_8tb/b70/build/strata-0216/strata-dd-0216 --serve <args>, started in the strata-sycl-dev:020 container exactly like strata/flash-next/dd/nl_launch_0216.sh (same mounts, ONEAPI_DEVICE_SELECTOR=level_zero:*, SYS_PTRACE for --expert-parallel-device,
  config env from hotschmoe-dd-0216.json), under bin/gpu-run with NO --card flag = both cards leased. The client (python) wraps only "docker run" in gpu-run, so the lease lives exactly as long as the container.
- It does NOT use server_intel.py / HTTP: raw token ids go straight to the engine, so no tokenizer round trip, no chat-template differences, and no port is needed (the brief's port 18134 is only used by the optional "http" mode:
  "strata_oracle.sh http" writes a plain-args server config and prints the nl_launch_0216.sh command with port 18134; it starts nothing).
- Mode plain (default) = config args minus {--mtp, --mtp-draft-vocab, --mtp-max-t, --spec, --spec-min-p, --batch-mtp, --vision, --conversation-cache-*, --kv, --kv-resident, --max-context, --prefill}, plus
  --kv fp16 --max-context 16384 --prefill 12288 --suffix-draft 0 --lookup-chain 0 --prompt-cache 0; env: STRATA_HC_Q8=0 (BF16 HC reads, no Q8 rounding), STRATA_BATCH_MTP=0.
  Why: spec/MTP windows make the verify path run T>1 rows with drafted tokens; plain T=1 decode (spec default 0, generate.cpp:660) gives the cleanest "target model only" greedy path and the least Strata-internal noise in the logits.
  fp16 KV is closer to PR3603's 16-bit KV than int8. --prompt-cache 0 reads every prompt from position 0 so positions in the dump are absolute.
  Mode qualified = the daily-driver arguments unchanged except server-managed flags (--vision, --batch-mtp, conversation cache) -- use it once to confirm the plain tokens equal the qualified path's tokens.
- Prompts (8, fixed, tokenized by Strata's own tokenizer incl. special tokens; ids written to $OUT/prompts.json): p0 raw English, p1 python code, p2 rust code, p3/p4 chat (thinking off: <think>\n\n</think>\n\n),
  p5 multilingual (French/Chinese/German), p6 raw text with a literal <|endoftext|> in the middle and a repeated 4-word phrase (exercises the PLE EOS reset + repeated n-grams), p7 long chat prompt of 4805 tokens (> 2051 cells => the QSA top-512-block selection
  is active for every generated position; Strata reads it through the batched prompt path). Prompt lengths: 22, 24, 37, 32, 35, 23, 25, 4805 [V tokenized on CPU].
- Per prompt: GEN 256 logprobs=20 <ids>; records tokens, per-token top-20 logprobs, DONE line; extracts from the dump the rows at absolute positions n-1+i (i = 0..generated-1) -> first_logits/<name>.f32 (fp32 [n_vocab], first token) and per-position
  top-64 raw logits + logsumexp ("topk64"). Prompts <= 64 tokens are read through the verify windows so their per-position prompt logits are in the dump too (not extracted by default).
  After the 8 prompts, prompt 0 is run once more as a determinism probe (same tokens?). Output: $OUT/oracle.json (+ engine.log with stderr, engine_plan.json with the exact command, prompts.json). Default OUT=/mnt/vm_8tb/b70/results/tf-flashnext/oracle.
  The raw dump (~1 MB per row, ~2.5 GB) is deleted unless --keep-dump; --no-dump keeps only the LP top-20 (fallback).
- Stop: QUIT on stdin, wait 120 s, docker stop -t 30 as fallback, docker rm -f at start, trap in the shell wrapper; the shell prints docker ps, bin/gpu-run --status and optionally runs bin/xpu-health before/after (ORACLE_HEALTH=1 default).
- Self test without GPU [V, done]: prompts subcommand tokenized all 8 prompts with exact decode round trip; "run --engine-cmd fake-engine" exercised the protocol parser, dump parser, first-logits extraction, determinism probe and "compare"
  against a synthetic stub; "run --print-only" prints the exact docker command. Nothing GPU-side was executed.

Commands (do NOT run until the GPU is leased and hotschmoe-dd is stopped):
  tensorfold/flashnext/tools/strata_oracle.sh plan                       # print the command only
  tensorfold/flashnext/tools/strata_oracle.sh run                        # ~15-25 min (UNVERIFIED), cards 0+1
  tensorfold/flashnext/tools/strata_oracle.sh run --max-new 64 --only p0_en_story_raw,p6_eos_reset_raw --no-dump   # quick bring-up of the harness itself
  tensorfold/flashnext/tools/strata_oracle.sh compare --tf /path/to/tf_result.json --min-top1 0.98

### 7.3 Teacher-forcing contract for the TensorFold engine

For each prompt p in oracle.json: feed ids = p.ids ++ p.gen_ids[0..i) for i = 0..len(gen_ids)-1 and take the argmax (and, optionally, top-k log-probs) of the logits at the LAST fed token. Position i of the oracle is "the distribution that produced
gen_ids[i]" = Strata row at absolute position n_prompt-1+i. In one teacher-forced pass over ids = p.ids ++ p.gen_ids[0..g-1] (n+g-1 tokens) the needed logits rows are the last g rows, positions n-1 .. n+g-2 (PR3603: forwardPass with logits_mode .all,
xpu_forward.zig:292-313, chunks <= chunk_cap; or the last row of g separate forward passes). TF result file for "compare":
  {"prompts":[{"name":"p3_chat_sky","top1":[id,...],"topk":[[[id,logprob],...],...],"first_logits_f32":"/path/p3.f32"}]}   (topk and first_logits optional; top1[i] has the same indexing as gen_ids[i])
"compare" reports per prompt: top-1 agreement, misses that are near ties (oracle top1-top2 log-prob gap < 0.05 nat, --tie-gap) vs real misses, first real miss index, KL(oracle||tf) on the shared top-20 (renormalised), and, if both first-token
logit files exist, full-vocab KL and max |dlogprob| over the oracle top-64. No pass/fail threshold is claimed here: set --min-top1 only after the first matched comparison (CLAUDE.md: no speed/stability claims without matched evidence).
Expected differences even for a correct port (do not chase): bf16 activations vs Strata f32/q8_1 activations, fp16 vs bf16 KV, Q8 HC (if used), kernel summation order; the oracle is about gross errors (wrong head order, norm offsets, hash, quant decode).

### 7.4 UNVERIFIED about the oracle run
- the plain-mode argument set starts cleanly on the 2-card EP configuration (first run is a bring-up; fall back to --mode qualified);
- STRATA_LOGITS_DUMP captures the head rows on the EP two-card verifier path (verify.cpp:3263 is inside Verifier::run; EP stage routing not read); fallback = --no-dump (top-20 LP only; first-token KL then over top-20);
- STRATA_HC_Q8=0 is accepted on this path (SYCL default is off unless "1", native_dense.cpp:37, so it should be);
- wall time (engine load not measured here); whether Strata output is bit-deterministic at temperature 0 (probe included);
- that the container image has /opt/intel/oneapi/setvars.sh (copied from nl_launch_0216.sh).

----------------------------------------------------------------------------------------------------------------------------------
## 8. Verification log (what was actually read or computed, CPU only)

1. CEN loaded: 1223 tensors in GGUF1, 1 in GGUF2; role table and bytes/element per type recomputed; attention layers = {3,7,...,47}; MoE gate/up types equal within each layer.
2. Real values read from GGUF1 (tensor bytes at data_start 11,024,256 + offset): norm means as quoted in 4c; ssm_a blk.0 min -157.985 max -0.0279, ln(-ssm_a) = bf16-exact values; ssm_dt.bias, ssm_conv1d, ssm_norm: F32 with zero low 16 bits;
   ple_conv1d F16 upcast has zero low 16 bits; detectNorms emulation: 48/48 attn hc_norm means > 0.5, median 1.0409 => offset 0.
3. PLE hash: python port of PR/xpu_ple.zig Hash.init (splitmix64, nthPrimeAfter) reproduces head_vocab_sizes, head_offsets, layer_multipliers [23703573157769, 20109073645365, 8052911324071] and the padded row count 320001536 only for vocab=248320.
4. MTP: rt dense.bin f32 norms == raw HF bf16 + 1 (max diff 0) for all 9 norm tensors; mtp-q2_0.gguf header listed through gguf-py; experts.bin size 707,788,800 = 512 x 1,382,400 [V]; dense.bin 116,099,072 B.
   draft_vocab_en.bin: 162,100 B = 40,525 int32 ids, sorted; PR draft_vocab_default.txt 79,591 ids.
5. Tokenizer round trip: Strata tokenizer (venv python 3.14, regex+numpy present) encodes/decodes the 8 oracle prompts exactly; special tokens parse (<|im_start|>=248045, <|im_end|>=248046, <think>=248068, </think>=248069, <|endoftext|>=248044).
6. Local HF config /mnt/cache/b70/devmodel/Qwen/Qwen3.8-Flash-Next/config.json: vocab_size 248320, ple_layer_ids [2], eos_token_id 248044 (scalar), output_gate_type sigmoid, no norm_topk_prob/routed_scaling_factor.
Line numbers for PR3603 files cite commit 207aaae (git -C /mnt/vm_8tb/b70/build/tensorfold/fn-model show 207aaae:<path>). In the live worktree workers K/L have since added gguf_* files and 3 kernel handles to xpu_store.zig, so xpu_store.zig lines after ~143 are shifted by +3 (Kernels struct) / +6 (Dev init), e.g. detectNorms 417 -> 423. flashnext.cl only gained an include at the end.
