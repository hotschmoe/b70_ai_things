# NVFP4 27B: prefill INT8-XMX + fp8 KV calibration -- investigation findings (2026-07-08)

Overnight research session (operator: "get NVFP4 prefill onto Intel XMX; re-enable fp8 KV for TP=1
long context; YOLO, dead-ends welcome"). This doc captures the DURABLE findings (literature + vLLM
source + hardware facts) that steer the two thrusts. Experiment results land in JOURNAL.md as they
complete. Sources: a web/vendor research pass (URLs inline) + a codex read of the extracted vLLM
v0.24.0 source (file:line inline). ASCII only.

--------------------------------------------------------------------------------
## POST-GPU CORRECTIONS (2026-07-08, after the experiments landed) -- READ FIRST

The pre-GPU THRUST 2 theory below was PARTIALLY REFUTED by the on-box results (JOURNAL "session 3 cont"):
- fp8 KV scale=1.0 is NEAR-LOSSLESS on this checkpoint, NOT lossy. Measured per-layer amax: K 11.7-21.8,
  V 7.5-133.0 -- ALL far below e4m3's max of 448, so scale=1.0 does NOT clip. And e4m3 is a FLOAT (roughly
  constant RELATIVE precision across its range), so values sitting "at the bottom of [-448,448]" are NOT
  low-precision -- the "poor precision at the bottom of the range" reasoning below is a fixed-point
  intuition that does not hold for a float format. fp8 scale=1.0 ran CLEAN over 3500 forced tokens on the
  clean config (TP=1, no-MTP, v0.24.0 fused); the earlier "repetition ~tok985" did NOT reproduce and was
  the TP=2+MTP path, not KV precision.
- Calibration IS mechanically viable (the XPU FlashAttention backend applies the scales -- proven by 3
  scales -> 3 distinct temp-0 hashes) and the offline calibrator + injector shipped (sitecustomize block
  (10), serve KV_SCALES knob, kv_scales_nvfp4_27b.json). But on THIS checkpoint calibrated scales are
  near-neutral (no clipping to fix); kept for other checkpoints / robustness.
- The REAL, shipped win of fp8 KV is CONTEXT: fp8 = 1.98x the bf16 KV context -> 128k fits on ONE card
  (bf16 caps ~71.5k; proven with a real 118,856-token needle retrieval). Binding constraint = weights
  (24.11 GiB), not KV (only ~16/64 layers cache KV).
The Thrust-1 (prefill) findings below stand (int8-XMX is the only 2x path; needs a custom K-blocked kernel).


--------------------------------------------------------------------------------
## THRUST 2 -- fp8 KV cache (re-enable, calibrated, for TP=1 long context)

### Root cause of the DD repetition, sharpened
- Our checkpoint nvidia/Qwen3.6-27B-NVFP4 (ModelOpt 0.45.0, MIXED_PRECISION) declares
  `quantization_config.kv_cache_scheme = {dynamic:false, num_bits:8, type:float}` (static fp8 KV) but
  ships NO `k_scale`/`v_scale` tensors on ANY layer. vLLM then defaults every KV scale to 1.0 and logs
  "Using KV cache scaling factor 1.0 for fp8_e4m3". Scale=1.0 maps the actual K/V range (|K|,|V| ~ 1-10)
  onto only the bottom of e4m3's [-448,448] range -> poor precision that accumulates over generation
  length -> the coherent-early / repetition-late collapse we bisected (bf16 clean 3500 tok; fp8 repeats
  ~tok985).
- WHY the scales are missing (smoking gun): ModelOpt changed the default `--kv_cache_qformat` from `fp8`
  (data-calibrated) to `fp8_cast` in v0.43; `fp8_cast`/`nvfp4_cast` skip KV calibration and use a
  CONSTANT amax = 448.0 (e4m3 max), so scale = amax/448 = 1.0 and nothing meaningful is stored. The
  OLDER nvidia/Qwen3-8B-NVFP4 (ModelOpt 0.35) still ships REAL calibrated scales named
  `model.layers.N.self_attn.k_proj.k_scale` / `v_proj.v_scale`. So this is NVIDIA's default, not a vLLM
  bug. NVIDIA does not expect you to calibrate KV for these NVFP4 checkpoints; they punt to constant-amax
  because their downstream engine (TensorRT-LLM) uses fp8 attention math where 1.0 stays in range.
  Sources: ModelOpt changelog https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html ;
  8B index https://huggingface.co/nvidia/Qwen3-8B-NVFP4/raw/main/model.safetensors.index.json ;
  27B config/index https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4/raw/main/config.json .

### Two hazards, both matching our model class
1. Shipped checkpoint => scale 1.0 => clipping/drift on long gen (above).
2. The obvious "fix" `--calculate-kv-scales` is DANGEROUS here, two independent reasons:
   - vLLM DISABLES it for hybrid GDN/Mamba models: `HybridAttentionMambaModelConfig.verify_and_update_config`
     turns off `cache_config.calculate_kv_scales` before it reaches the attention layers
     (vllm_src/model_executor/models/config.py:261-289; Qwen3_5ForConditionalGeneration is in that map).
   - Even if forced on, it is a known CORRUPTOR on hybrid GDN+Attention: vLLM issue #37554 -- the online
     scale calc's single dummy forward runs with UNINITIALIZED GDN recurrent state -> garbage activations
     feed the full-attention layers -> wildly-wrong frozen KV scales -> repetitive loops from request one.
     And its divisors are Q/K/V=200/200/100 (envs.py), not amax/448.
     Source: https://github.com/vllm-project/vllm/issues/37554 .

### The fix: offline calibration + non-destructive injection
- vLLM loads per-layer, per-tensor SCALAR k_scale/v_scale (shape () or (1,), f32) via
  `BaseKVCacheMethod` (vllm_src/.../quantization/kv_cache.py:57-146). It uses checkpoint scales ONLY when
  `is_quantized_kv_cache(kv_cache_dtype) and not calculate_kv_scales` and both > 0; else falls back to 1.0.
  The "1.0" warning fires iff k==v==1.0 (kv_cache.py:147). GUARD to verify: if
  `kv_cache_uses_per_token_head_scales(kv_cache_dtype)` is True, checkpoint scales are IGNORED (forced 1.0,
  computed per-token-head in-kernel) -- confirm our path does not hit it.
- Injection format (non-destructive): the checkpoint prefix is `model.language_model.layers.N.self_attn.*`
  (e.g. existing `...k_proj.input_scale`), and there IS a `model.safetensors.index.json`. So add ONE small
  extra shard with the ~34 scalar f32 tensors and add them to a COPY of index.json weight_map, mounted as an
  overlay (mirror the KV_CONFIG_OVERRIDE style in serve_nvfp4_27b.sh) -- base checkpoint stays pristine.
  `maybe_remap_kv_scale_name` (weight_utils.py:1358-1400) remaps proj-level names, so write them as the 8B's
  convention `model.language_model.layers.N.self_attn.k_proj.k_scale` / `v_proj.v_scale` (fallback
  `...self_attn.k_scale`). Only the 17 FULL-ATTENTION layers carry a KV cache: indices
  [0,3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]; the other 48 are GDN/linear-attn (no scales). Compute
  scale = amax/448 from per-layer max|K|,max|V| over a real calibration set.
- XPU APPLIES the scales (calibration is not futile): FlashAttention (XPU default) passes layer._k_scale/
  _v_scale into reshape_and_cache_flash on write (flash_attn.py:951-960) and as k_descale/v_descale on read
  (flash_attn.py:800-889); the Triton path visibly divides/multiplies K/V by the scale
  (triton_reshape_and_cache_flash.py:108-125, triton_unified_attention.py). Diagnostic: A/B with
  --attention-backend TRITON_ATTN if FlashAttn misbehaves post-injection.
- Payoff sizing: the hybrid has only 16-17 full-attn layers, so KV is already lean; fp8 roughly HALVES it,
  i.e. ~2x the max TP=1 context (the whole point for 128k->200k). Gate the adoption on a quality check --
  calibrated fp8 KV on a 16-KV-layer hybrid may or may not match bf16 KV per token; measure long-gen
  repetition before shipping. Alt/fallback the repo already validated: KV_FP8=0 (bf16 KV).

--------------------------------------------------------------------------------
## THRUST 1 -- NVFP4 MLP prefill onto INT8 XMX (2x compute reclaim)

Current fused path (torch.ops._xpu_C.nvfp4_gemm_w4a16) keeps weights 4-bit resident and decompresses
f4_e2m1 -> bf16 in the oneDNN JIT gemm: bf16 COMPUTE rate at prefill. INT8 XMX is 2x bf16 -> the prize.

### Corrected ceiling and dead-ends (saves GPU time)
- "int4 = 4x int8, int2 = 8x int8" is FALSE for MIXED int-weight x int8-activation. Per arXiv 2508.06753
  (Intel Xe2 low-bit inference), int2xint8->int32 and int4xint8 DPAS run at the SAME throughput as
  int8xint8; the 2x/4x MAC multiplier exists ONLY for SYMMETRIC low-bit dpas.s4.s4 (=W4A4) / s2.s2. So any
  scheme keeping bf16/int8 activations caps at int8 rate. TARGET = ~2x over bf16 (int8 vs bf16 XMX). This
  matches the operator's premise ("reclaim exactly the 2x W8A8 gets"). Source:
  https://arxiv.org/html/2508.06753v2 .
- oneDNN has NO native block-scaled s8xs8->s32 GEMM. It offers only: (a) per-tensor/per-channel int8
  (no K-group scale in general matmul; K-grouped src scale exists only in the experimental grouped-GEMM/MoE
  path), (b) low-bit WEIGHT-DECOMPRESSION to bf16/f16 then FLOAT matmul (our current path), or (c) FLOAT
  MXFP8/NVFP4 block-scaled matmul. A true integer block-scaled GEMM must be hand-written. Sources:
  https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html ,
  https://www.intel.com/content/www/us/en/developer/articles/release-notes/onednn/2026.html .
- Xe2/Battlemage DPAS has NO fp8 element type (esimd dpas_argument_type = u8/s8/u4/s4/u2/s2/bf16/fp16/tf32;
  see proto_int4/INT4_DPAS_PIONEER.md). So oneDNN's newer "float NVFP4/MXFP8 optimized on Xe2" runs fp8/fp4
  as STORAGE with bf16 COMPUTE (upconvert) = same speed as the current bf16-compute path = NO prefill
  speedup on B70. The oneDNN-float-NVFP4 route is a dead end for speed on this silicon.

### Consequence: the only 2x path is a CUSTOM K-blocked block-scaled int8 GEMM
NVFP4's weight scale is per-16-K-group (varies WITHIN the reduction), so it cannot be applied at a W8A8
epilogue. The kernel must: accumulate s32 within each 16-elem K-block, multiply by
(block_scale * global_scale / 2) * per-token-activation-scale, accumulate in fp32. E2M1x2 is exact s8
(codes {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}, |max|=12), so the f4->s8 repack is lossless and stays a per-tile
transient (weights remain 4-bit resident; a full s8 copy = the 31GB deadend). Prefill activations quantize
per-token to s8 -> prefill becomes W4A8-numerics (verify quality). Vehicles: intel/sycl-tla
(CUTLASS-for-Intel, already targets intel_gpu_bmg_g31 with int8 GEMM + fused epilogues + narrow-int types)
or hand ESIMD (proto_int4/ has a working s8/s4 DPAS harness). Reference perf: arXiv 2508.06753's XeTLA/ESIMD
low-bit-weight x bf16-act kernels reach "close to oneDNN int8 GEMM," prefill (compute-bound) benefits most.
Expectation: ~2x over bf16 at large M, plus bandwidth from 4-bit weights; go/no-go = the step-1 ceiling
probe (does a plain int8 s8xs8 with a resident s8 copy actually hit ~2x over bf16 at M=2048-8192 on real
gate/up/down shapes; confirm jit:gemm:xe src_s8/wei_s8/dst_s32 via ONEDNN_VERBOSE).

--------------------------------------------------------------------------------
## Assumption-flags (things this session corrected)
1. "NVIDIA ships calibrated fp8 KV scales" -- FALSE for the 27B (0.45, fp8_cast constant-amax); TRUE only
   for the older 8B (0.35). The DD repetition is root-caused to NVIDIA's fp8_cast default.
2. "Turn on --calculate-kv-scales to fix it" -- DANGEROUS on hybrid GDN (vLLM #37554 reproduces
   repetition; also disabled by vLLM config for hybrids). Offline calibration or bf16 KV only.
3. "oneDNN can do a NVFP4-style block-scaled int8 GEMM" -- FALSE; must be hand-written.
4. "int4 = 4x int8 DPAS rate" -- FALSE for mixed int-weight x int8-act (same as int8); the multiplier is
   W4A4-only (symmetric s4.s4) via ESIMD. B70 int8 367 TOPS is the practical prefill compute ceiling.
