# 2S1 -- Ornith-1.5-122B: architecture, fit on 2x B70, reuse

Agent: Claude Sonnet 5.5 (claude-sonnet-5-5), 2026-10-20. Research only. No GPU,
no weight download, no commit. Sources: HF API and raw files for
ornith-ai/Ornith-1.5-122B (config.json, README.md, model.safetensors.index.json,
sha 346ec829), the GGUF tree listings of the quant repos, and the first 12 MB
(header only) of unsloth/Ornith-1.5-122B-GGUF UD-IQ3_XXS. Local reading:
reports 2O1, 1H1, 1C12, 1C13, 1G4, plugins/INTERFACE.md, LOWBIT_FORMAT.md,
the 35B config, and the fork registry
(/mnt/vm_8tb/b70/build/vllm-flashnext/vllm/vllm/model_executor/models/registry.py).

## 0. Verdict (short)

FEASIBLE on 2x32 GiB with a ~3-bit expert GGUF (Unsloth UD-IQ3_XXS, 46.1 GiB
including MTP) and EP-sidecar, 2x200K context fits only with fp8 KV (or
bf16 KV with zero slack). The user is right about the lineage, with one
correction: the config is exactly the Qwen3.5-MoE 122B-A10B geometry
(model_type qwen3_5_moe, hidden 3072, 48 layers, 256 experts, top-8, expert
FFN 1024). The model card does not name a base; "Qwen3.5-122B-A10B" is inferred
from identical dimensions, not stated. It is a Qwen3.5-generation architecture,
NOT Qwen3.6 or 3.8, and the 397B sibling is the Qwen3.5-397B-A17B geometry.
vLLM already has the model class. int4 does not fit for the goal. Decode is
bounded by the TRUNK, not the experts (trunk is 78 percent of active bytes in
the stock UD file), so trunk bit-width is the main decode lever.

## 1. Architecture (from config.json / index)

| item | value |
| --- | --- |
| architecture / model_type | Qwen3_5MoeForConditionalGeneration / qwen3_5_moe (text: qwen3_5_moe_text) |
| layers | 48, pattern 3 linear (GDN) + 1 full, repeated: 36 GDN + 12 full attention |
| hidden | 3072 |
| full attention | 32 Q heads, 2 KV heads, head_dim 256, attn_output_gate (q_proj is 3072x16384), partial_rotary 0.25, mrope interleaved, rope_theta 1e7 |
| GDN (linear attn) | 16 key heads x 128, 64 value heads x 128, conv kernel 4, mamba_ssm_dtype fp32. in_proj_qkv 3072x12288, in_proj_z 3072x8192, out_proj 8192x3072, tiny a/b |
| MoE | 256 routed experts, top-8, expert FFN 1024, 1 shared expert FFN 1024 with sigmoid shared_expert_gate; all 48 layers are MoE (mlp_only_layers empty). Experts are FUSED in the safetensors (experts.gate_up_proj, experts.down_proj) |
| vocab | 248320, untied embed / lm_head |
| context | 262144 native (YaRN x4 to ~1M documented) |
| MTP | yes. mtp_num_hidden_layers 1, tensors under mtp.* (fc, one full-attn + MoE layer; ~2.5B params) |
| vision | 27-layer ViT, hidden 1152, out 3072 (~0.45B params, 0.84 GiB bf16; GGUF mmproj 0.8 GiB) |
| PLE / per-layer embeddings | none (unlike Flash-Next) |
| license | MIT. README: vLLM >= 0.19.1, SGLang >= 0.5.9, "runs on a single 80GB GPU" |

Parameter accounting (computed from shapes; total matches the index size,
250.17 GB bf16 = 125.09B params):

| part | params |
| --- | ---: |
| routed experts (48 x 256 x 3 x 3072 x 1024) | 115.96B |
| shared experts | 0.45B |
| GDN projections (36 layers) | 3.19B |
| full attention (12 layers) | 0.94B |
| embed + lm_head | 0.76B + 0.76B |
| vision | 0.45B |
| MTP layer (256 experts) | 2.52B |

Active per token: 8 of 256 experts = 3.62B routed + 4.58B trunk (shared + GDN +
full) + 0.76B lm_head = 8.97B (the "A10B" with embedding rows).

Contrast with the 35B-A3B we already serve: hidden 2048, 40 layers, expert FFN
512, 16 Q heads, 32 GDN value heads. The 122B has the same expert count (256)
and top-k (8) but 1.5x hidden, 2x expert FFN, 2x GDN value heads (key:value
ratio 1:4, was 1:2).

## 2. Fit on 2 x 32 GiB

### 2.1 Weight sizes (GiB). Text model + MTP, vision separate (0.8 GiB)

Computed from shapes. "trunk" = shared + GDN + attention + lm_head + embed.

| layout | experts | trunk+embed+head | text total | + vision | + MTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 216.0 | 11.4 | 227.4 | 228.2 | 232.9 |
| int4 g128 (4.16 bpw), trunk 4.16 | 56.2 | 3.2 | 59.4 | 60.2 | 61.4 |
| int4 experts, Q8 trunk | 56.2 | 5.5 | 61.7 | 62.5 | 63.8 |
| IQ3_XXS 3.0625 bpw experts, Q8 trunk | 41.3 | 5.5 | 46.9 | 47.7 | 48.7 |
| IQ3_XXS experts, Q6_K-ish trunk | 41.3 | 4.5 | 45.8 | 46.7 | 47.6 |
| int3 g128 (3.125 bpw) experts, int4 trunk | 42.2 | 3.2 | 45.4 | 46.2 | 47.2 |
| IQ2_XXS experts (2.06 bpw), Q6 trunk | 27.8 | 4.5 | 32.3 | 33.2 | 33.8 |

Measured published GGUF sizes (HF tree API):

| repo / file | GiB |
| --- | ---: |
| unsloth UD-IQ1_M | 37.6 |
| unsloth UD-IQ2_XXS | 39.1 |
| unsloth UD-Q2_K_XL | 41.5 |
| unsloth UD-IQ3_XXS | 46.1 |
| unsloth UD-Q3_K_XL | 55.7 |
| unsloth UD-Q4_K_XL | 74.2 |
| AtomicChat AD-IQ3_XXS-IQ2_S | 44.4 |
| AtomicChat AD-IQ3_S-IQ3_XXS | 50.6 |
| AtomicChat AD-IQ4_XS-IQ3_S | 57.9 |
| AtomicChat AD-Q4_K-IQ4_XS | 66.3 |
| ornith-ai official Q4_K_M / Q8_0 | 70.6 / 123.5 |

Budget: 64 GiB minus ~2 GiB runtime per card = ~60 GiB usable. Plain int4 g128
(60.2 GiB with vision) and every Q4_K-class GGUF leave nothing for KV: ruled
out for the 2x200K goal. Sub-4-bit (IQ3) is the only class that fits.

### 2.2 Header of unsloth UD-IQ3_XXS (read via range request, 12 MB, no weights)

- general.architecture qwen35moe, block_count 49 (48 + 1 nextn/MTP block),
  nextn_predict_layers 1, expert_count 256, expert_used 8.
- Gate/up/down experts are UNFUSED (separate ffn_gate_exps, ffn_up_exps,
  ffn_down_exps), shape (3072,1024,256) and (1024,3072,256). K=3072 and
  K=1024 are multiples of 256, so every super-block type applies cleanly.
- Per-layer mixed expert types: ffn_gate/up IQ3_XXS x27, IQ2_S x9, IQ2_XXS x8,
  and type id 19 (IQ1_S) x3 layers. ffn_down IQ3_XXS x27, IQ4_XS x8, IQ2_S x7,
  Q2_K x3.
- Trunk: attn_qkv / attn_gate / ssm_out / attn_q/k/v/output mostly Q5_K with
  some Q8_0 (and a Q6_K); shared expert Q5_K or Q8_0; ssm alpha/beta Q8_0;
  output.weight Q8_0 (0.63 GiB); token_embd Q5_K (0.49 GiB); router and
  shared-gate F32.
- Computed split of the 46.1 GiB: routed experts 38.77, trunk+head 4.35,
  embed 0.49, MTP block 2.50 (experts and all).
- Active bytes per decoded token (no MTP): routed 1.30 GB + trunk 4.67 GB =
  5.97 GB. The trunk dominates (78 percent).

Type compat with b70_gguf_xpu: IQ3_XXS, IQ2_XXS, IQ2_S, IQ4_XS, Q2_K, Q5_K,
Q6_K, Q8_0 are covered (INTERFACE.md + C12 fast-path list). IQ1_S (id 19) is
NOT: 3 layers of gate/up would need a new kernel, or pick a file without it
(AtomicChat AD-IQ3_XXS-IQ2_S, or requantize).

### 2.3 KV and state

Only the 12 full-attention layers grow. Per token: 12 layers x 2 (K,V) x 2 KV
heads x 256 = 12288 elements.

| KV dtype | bytes/token | 200K tokens | 2 x 200K |
| --- | ---: | ---: | ---: |
| bf16 | 24576 | 4.58 GiB | 9.16 GiB |
| fp8 | 12288 | 2.29 GiB | 4.58 GiB |

(Native 262144 context: 6.0 GiB bf16 / 3.0 GiB fp8 per sequence.)

GDN state is fixed per sequence: 36 layers x 64 value heads x 128 x 128 x fp32
= 144 MiB, plus conv state ~2.6 MiB, ~147 MiB per sequence. Negligible.
Context scaling is hence cheap: the 122B costs 24 KiB/token, the same order as
the 35B (10 full layers x 2 x 2 x 256 x 2 B = 20 KiB). The existing
kv-parking (G4, 13.9 GB/s restore of 200063 tokens on Ornith 35B) will move
~4.9 GB bf16 per 200K sequence, about 0.35 s.

### 2.4 Layout with the EP sidecar (card 0 = vLLM rank with trunk + KV; card 1 = experts)

Using the Unsloth UD-IQ3_XXS sizes. Card 0 non-expert load: trunk+head+embed
4.84, vision 0.8, MTP 2.5 (kept on card 0 as Strata does), KV+state 2 seqs,
runtime 2, prefill/activation workspace ~2.

| | KV bf16 | KV fp8 |
| --- | ---: | ---: |
| card 0 non-expert | 4.84+0.8+2.5+9.45+2+2 = 21.6 | 4.84+0.8+2.5+4.87+2+2 = 17.0 |
| card 0 expert room (of 32) | 10.4 | 15.0 |
| card 1 expert load (38.77 - card 0) | 28.4 | 23.8 |
| card 1 room (32 - 2 - 1 workspace) | 29 | 29 |

Result: bf16 KV fits with ~0.6 GiB slack on card 1 (too tight; prefill dequant
scratch of one layer is about 2.4 GiB per half-expert set at T=12288, see C13).
fp8 KV leaves ~5 GiB on card 1 and is the recommended 2x200K configuration.
Dropping MTP from VRAM frees 2.5 GiB. A fuller AD-IQ3_S-IQ3_XXS (50.6 GiB)
works only with fp8 KV and no slack on card 1 (about 0.5 GiB).

Balance: traffic-balanced EP is possible (Strata deals experts in a snake so
decode traffic is ~50/50 even when VRAM is 30/70), but compute balance is not
solved: card 0 also reads the trunk (4.7 GB per token in the stock UD file)
and card 1 only reads ~0.65 GB of experts. Card 0 is the critical path. Moves
that could help: lm_head (0.63 GiB) and the shared experts (0.45B) to card 1,
and a lower-bit trunk (below).

## 3. Decode ceiling (roofline, 608 GB/s per card)

Bytes read per token. Card-0 path in EP = trunk + half the expert bytes.

| scenario | total active GB | card 0 GB (EP) | ms @100% BW | tok/s @100% | tok/s @70% |
| --- | ---: | ---: | ---: | ---: | ---: |
| single card, UD-IQ3_XXS as shipped (impossible, 46 GiB) | 5.97 | 5.97 | 9.8 | 102 | 71 |
| EP, UD-IQ3_XXS as shipped | 5.97 | 5.32 | 8.7 | 114 | 80 |
| EP, trunk requantized to ~4.5 bpw (Q4_K/IQ4_XS), head Q6_K | 4.5 | 3.9 | 6.4 | 156 | 110 |
| EP, same + lm_head + shared on card 1 | 4.5 | 3.0 | 4.9 | 200 | 140 |
| EP, int4 (not fitting) trunk Q8 | 7.4 | ~6.1 | 10.0 | 100 | 70 |

Flash-Next daily driver (Strata, 125B similar size) measures C1 ~141 tok/s, so
a Strata-class kernel set can reach ~70-90 percent of the "EP, trunk 4.5 bpw"
row. Our vLLM-sidecar path is far from roofline today: the 35B EP graph at
C1 is 11.36 ms against a ~2 ms roofline (about 0.25 ms of sync/launch per MoE
layer). Extrapolating 48 layers plus the larger bytes, expect first-light
vLLM numbers of roughly 50-70 tok/s at C1 on the stock UD file (estimate,
not measured). MTP (1 draft, layer is 2.5B incl. 8 active experts, ~0.25 GB
per draft) is the best lever on top. The 150 tok/s decode goal needs MTP +
low-bit trunk + the fused schedule; 100-120 is the realistic plan on this
model.

Prefill 2000+ tok/s: active FLOPs are 9B x 2 = 18 GFLOP/token (3x the 35B).
The 35B E2 low-bit path measured 1961 tok/s at 8k on one card, so the 122B
would be ~650 tok/s on that same single-card path. EP splits expert FLOPs over
two cards (experts are 40 percent of active parameters), and trunk GEMMs stay
on card 0 (about 9.2 GFLOP/token, ~6000 tok/s at 60 TFLOPS effective). A
~1500-2000 tok/s target is not ruled out but is unproven; it depends on the C13
fused dequant+bmm path being applied per card and chunked prefill overlap.

## 4. Reuse of our work

### 4.1 Upstream vLLM support

Registry in the fork (HEAD 4d73572e14) already contains
Qwen3_5MoeForConditionalGeneration (registry.py:603), Qwen3_5MoeForCausalLM
(:202), and Qwen3_5MoeMTP (:711). No new model class needed. README requires
vLLM >= 0.19.1; our fork is newer. The Ornith-35B-A3B (same model_type) is
what we already serve through the same class.

### 4.2 Plugin by plugin

| component | applies to 122B? | what is new |
| --- | --- | --- |
| b70_gguf_xpu fused MoE (decode moe_gemv, C9/C12 schedule, grouped_gemm prefill) | Yes for IQ3_XXS/IQ2_XXS/IQ2_S/IQ4_XS/Q2_K/Q5_K/Q6_K/Q8_0. Shapes: K=3072 / K=1024, N=1024 / 3072, all multiples of 256 and 64. | IQ1_S (id 19) kernel for 3 gate/up layers in UD-IQ3_XXS (or avoid the file). Re-tune C12 3D grid and entry-width chunks for the larger expert (3.1M weights per matrix vs 1.0M for the 35B; C12 was tuned on K=2560/I=640 and 35B). C13 wide dequant (270-286 GB/s) scales; dequant+bmm per layer is ~4.8 GB bf16 for the full set, so per-card half-sets (2.4 GB) are fine. |
| b70_gguf_xpu GGUF loader (fused vs unfused, qwen3_5 schema) | Yes. 122B GGUF is arch qwen35moe, unfused gate/up/down like the 35B. | Load placing: MTP block 48 (nextn.*, includes 256 experts) and the unfused eh_proj/enorm/hnorm naming. Vision from mmproj-*.gguf (0.8 GiB) or from safetensors. |
| b70_lowbit_xpu int3/int4 AutoRound (W3A16 / W4A16 g128, mixed int3/int2) | Yes for the INC/GPTQ expert format (LOWBIT_FORMAT.md: packing is shape agnostic). | The checkpoint does not exist; we would have to quantize 250 GB bf16 with AutoRound (calibration on GPU, hours). Experts are fused in the source safetensors: needs the same unfuse chunk as the 35B recipe (gate_up_proj [E,2048,3072] -> gate/up [E,1024,3072]). qkv_proj / in_proj_qkvz packed mapping stack: q_proj is 3072x16384 (gate fused), in_proj_qkvz 3072x(12288+8192). Int3 kernel was 11.29 ms C1 on the 35B vs int4; int3 trunk would need K=8192 groups (8192/128 OK). |
| b70_ep_xpu sidecar (H1/M0-M3, 128/128 split on 35B, topk 8) | Yes, directly the target. Same E=256, topk=8, so no hard-coded k=10 issue as in Strata. | Placement: uneven split (card 0 ~10-15 GiB of 38.8) per section 2.4. placement.py exists (untracked, in progress). Larger hidden: inbox payload per token is the q8_1 hidden row 3072 -> 3264 B (35B: 2048 -> 2176 B), still small; flag handshake 1.8 us is unchanged. 48 layers instead of 40 raise per-token sync work ~20 percent. The PLE guard is not needed. |
| kvpark (G4 exact-tail parking, 13.9 GB/s restore) | Yes. 24 KiB/token bf16 KV (12 attn layers) and 147 MiB GDN state per sequence. | Re-measure block size for hybrid page alignment (the 35B used block 1088); need a 200K T3 gate which is not yet measured on the 35B either. |
| D3 fused GDN decode (b70_gdn_xpu) | Probably, untested. | Heads: 16 K / 64 V, ratio 4 (35B is 16/32, ratio 2). Value-head repeat factor and state size per layer (64x128x128 fp32 = 4 MiB) must be checked in kernel; per-layer work is 2x larger. |
| b70_qsa_xpu | n/a here | Qwen3.5 head_dim 256 / 2 KV heads like the 35B; shape change is 32 Q heads (GQA ratio 16, was 8). Needs check. |
| fp8 KV (calibrated) | Yes: vLLM fp8 KV worked on hybrid in the dense NVFP4 work | Scales must be calibrated on this model; scale-1.0 was near lossless on Qwen3.6 dense, verify here. |
| GPTQ INT4 trunk of the 35B (serve_ornith_gptq.sh) | Template only | Shapes differ; int4 checkpoint absent for 122B. |

### 4.3 Quant sources on HF (as of 2026-10-20)

No GPTQ, AWQ, or AutoRound variant was found for 122B. Found: official
ornith-ai/Ornith-1.5-122B-FP8, -GGUF (Q4_K_M and above only), -MLX*; unsloth
UD-GGUF (IQ1_M through BF16; has MTP; mmproj F16/BF16);
AtomicChat AD-* GGUFs (imatrix, IQ1/IQ2/IQ3/IQ4 mixes);
Mia-AiLab EXL3 4.0 bpw; OsaurusAI JANG MLX. The ISTA GSQ-RCO ~3-bit format
we use on Flash-Next exists on HF for the 9B and 35B but not for the 122B. An
FP8 variant is 125 GB, too big (and no fast FP8 expert path here).

## 5. Verdict, layout, risks, first steps

Recommended layout: Unsloth UD-IQ3_XXS (or AtomicChat AD-IQ3_XXS-IQ2_S, which
avoids IQ1_S) on the EP sidecar. Card 0: vLLM rank with trunk (4.8 GiB), MTP
(2.5 GiB), vision tower, fp8 KV for 2 x 200K (4.6 GiB), 10-15 GiB of experts.
Card 1: the rest of the experts (24-28 GiB). Then, as a quality/speed step,
requantize the trunk to Q4_K/IQ4_XS-class (the trunk is 78 percent of decode
bytes), which also frees ~1.8 GiB on card 0.

Main risks:
- Quality: 38.8 GiB of experts at 2-3 bits (3 layers at IQ1_S, 9+8 layers at
  IQ2) on a 256-expert model; the no-tool-call/quality gate must be a real
  eval (Pi harness), not coherence only.
- Trunk-bound decode: 4.7 GB of trunk per token on card 0 caps C1 at ~80
  tok/s even with perfect expert overlap unless the trunk is requantized or
  split to card 1.
- 2x200K bf16 KV leaves ~0.6 GiB slack on card 1; use fp8 KV (calibration
  needed) or park KV to RAM.
- Prefill: 3x the FLOPs of the 35B; 2000 tok/s is unproven.
- GDN D3 and C12 kernels were tuned on other shapes (ratio 1:4 value heads;
  expert 3072x1024).
- IQ1_S has no kernel; MTP block has 256 experts (2.5 GiB) in the Unsloth file.
- Download cost: 46 GiB GGUF, 250 GB bf16 if we requantize.

First 3 steps (all CPU or short GPU probes, queue between serves):
1. Parse the full unsloth UD-IQ3_XXS header (done for the first 12 MB here),
   then download that one file (46 GiB, NVMe /mnt/cache) and extend the GGUF
   loader to the qwen35moe 122B shapes (blk.48 MTP naming, mmproj vision). Run
   a CPU/one-card dequant correctness check on one layer per expert type.
2. Run a one-layer kernel probe on card 0 at the new shape (E=256, K=3072,
   I=1024, topk 8, types IQ3_XXS/IQ2_S/IQ4_XS): moe_gemv T=1..8, C12 schedule,
   and C13 prefill at T=4096/12288, plus a D3 GDN decode probe at 16K/64V
   heads. Gate: bitwise or cosine vs CPU dequant. Add IQ1_S only if the file
   needs it.
3. Bring up the EP sidecar eager on 122B with an uneven placement (card 0
   ~12 GiB experts, fp8 KV, no MTP first), 8k context, greedy gate, then
   graph capture, then 200K x2 KV fit and MTP. Measure C1 ms/token against
   the roofline rows in section 3, and stop-start the daily driver
   (hotschmoe-dd) around the GPU windows.

Not verified: any speed number for the 122B (all projections above are
estimates), GDN/QSA kernel behaviour at the new head ratios, the quality of
any 2-3 bit quant of this model, and which exact base checkpoint Ornith was
trained from.
