# W8A8 / FP8 / INT8 download options for Qwen3.6 (27B dense + 35B-A3B MoE)

Scout report: do downloadable pre-quantized checkpoints exist so we can SKIP burning quant
hours? Priority = schemes that actually SERVE on vLLM-XPU (Intel Arc Pro B70 / Battlemage,
32 GB/card, vLLM 0.23.0+xpu, soon 2 cards).

Date: 2026-06-20. All "VERIFIED" rows below were fetched (HF page exists + size/scheme read
off the repo). Rows under "SEARCHED, NOTHING FOUND" are gaps -- no such repo exists.

-------------------------------------------------------------------------------------------
## XPU serveability cheat-sheet (what GOOD / RISKY / NO means here)
-------------------------------------------------------------------------------------------
  GOOD on XPU:
    - FP8 (block-fp8 or per-tensor)        -> upstream XPU fp8 scaled_mm kernel (proven; we run qwen3-14b-fp8)
    - compressed-tensors W8A8-INT8         -> our XPUInt8ScaledMMLinearKernel (proven on qwen3-14b-w8a8-gptq)
    - int4 AutoRound / wNa16-INT           -> proven (we already serve int4-AutoRound for both models)

  RISKY on XPU (kernel coverage WIP / vendor-specific -- a finding either way):
    - AWQ-int4                 -> XPU INT4 AWQ/GPTQ GEMM is WIP in vllm-xpu-kernels; Marlin path is CUDA-only
    - GPTQ-int4/int8           -> had an XPU load regression (vLLM #39474); moe_wna16 MoE path unproven on XPU
    - AMD Quark format         -> Quark loader is AMD/ROCm-oriented; Intel uses AutoRound/INC, NOT Quark

  NO on XPU:
    - GGUF                     -> llama.cpp, not vLLM
    - NVFP4 / MXFP4            -> Blackwell / CUDA fp4 kernels, no Xe2 path
    - MLX                      -> Apple-only
    - bitsandbytes             -> no XPU kernel
    - **any INT8/W8A8 MoE**    -> NO XPU int8-MoE kernel exists. A W8A8/int8 *35B-A3B* checkpoint
                                  would NOT serve on XPU even if one existed. (FP8-MoE is fine; the
                                  35B FP8 below uses the fp8 MoE path, not int8.)

===========================================================================================
## MODEL 1 -- Qwen3.6-27B  (DENSE, Gated-DeltaNet hybrid; Qwen3_5ForConditionalGeneration)
##   We already have: int4-AutoRound (Lorbus) + bf16 (Qwen). WANT: W8A8-int8 / FP8 / int8(W8A16).
##   BF16 baseline ~52 GB. 1 card = 32 GB.
===========================================================================================

### VERIFIED (repo fetched, exists)

| repo id                                  | scheme / format                         | size on disk | fits 1x B70? | XPU-serveable     | verdict |
|------------------------------------------|-----------------------------------------|--------------|--------------|-------------------|---------|
| Qwen/Qwen3.6-27B-FP8                      | FP8 block-fp8 (block 128), native fp8   | ~18.3 GB     | YES          | GOOD (fp8 kernel) | **DOWNLOAD THIS** -- best high-precision XPU option. DeltaNet/recurrent layers: confirm they aren't fp8-broken at load. |
| vrfai/Qwen3.6-27B-FP8                     | FP8 **W8A8 static per-tensor**, compressed-tensors (llm-compressor) | ~34 GB | NO (too big) | GOOD format, but oversized | Skip -- 34 GB won't fit 1 card; FFN/attn fp8 but DeltaNet+vision kept bf16 inflates it. Use Qwen official fp8 instead. |
| nameistoken/Qwen3.6-27B-Quark-W8A8-INT8  | **W8A8 INT8** (w=int8 per-chan static, a=int8 per-token dynamic) in **AMD Quark format** | ~29 GB | marginal (29<32 but tight w/ KV) | RISKY/likely-NO (Quark loader is AMD; XPU uses AutoRound/INC) | This is the *exact scheme we want* (int8 dyn-act) but WRONG FORMAT for XPU. Would need re-pack to compressed-tensors. Not a drop-in. |
| cyankiwi/Qwen3.6-27B-AWQ-INT4            | AWQ int4 (compressed-tensors lib tag)   | ~19.0 GB     | YES          | RISKY (XPU AWQ-int4 GEMM is WIP; Marlin=CUDA) | int4 only; we already have int4-AutoRound which is the *proven* XPU int4. Skip unless AWQ-on-XPU becomes a goal. |
| vrfai/Qwen3.6-27B-NVFP4                  | NVFP4 (compressed-tensors)              | ~26 GB       | YES (size)   | NO (Blackwell fp4) | Skip -- no Xe2 fp4 path. |
| bartowski/Qwen_Qwen3.6-27B-GGUF         | GGUF (many quants incl Q8_0/Q4)         | varies       | depends      | NO (llama.cpp)    | Skip for vLLM. |
| unsloth/Qwen3.6-27B-GGUF                 | GGUF                                    | varies       | depends      | NO                | Skip for vLLM. |
| mlx-community/Qwen3.6-27B-OptiQ-4bit     | MLX 4-bit                               | ~15 GB       | -            | NO (Apple)        | Skip. |

### SEARCHED, NOTHING FOUND (gaps -- no such repo exists as of this scout)
  - compressed-tensors **W8A8-INT8** for 27B (the format our XPUInt8ScaledMMLinearKernel wants).
    Only the AMD-Quark INT8 above exists; nobody has published a compressed-tensors int8 W8A8.
  - **int8 W8A16** (weight-only int8) for 27B. None found.
  - GPTQ-Int8 for 27B. None found (Qwen only ships GPTQ-Int4 at the 27B tier: Qwen/Qwen3.5-27B-GPTQ-Int4
    exists for the 3.5 gen, not a 3.6 int8).

-------------------------------------------------------------------------------------------
### MODEL 1 BOTTOM LINE
-------------------------------------------------------------------------------------------
  - FP8 path (recommended high-precision anchor): **DOWNLOAD `Qwen/Qwen3.6-27B-FP8`** (~18.3 GB,
    fits 1 card, proven XPU fp8 kernel). No need to quant fp8 ourselves.
        hf download Qwen/Qwen3.6-27B-FP8 --local-dir Qwen3.6-27B-FP8
  - W8A8-INT8 path (our int8 kernel): **MUST QUANT OURSELVES.** No compressed-tensors int8 W8A8
    exists. The only int8 checkpoint (nameistoken Quark) is AMD-format + 29 GB -> not XPU-serveable
    as-is. Produce it with scripts/40,49,54 -> ...-W8A8-gptq (compressed-tensors), like qwen3-14b.

===========================================================================================
## MODEL 2 -- Qwen3.6-35B-A3B  (MoE, 256 experts / 8+1 active; Qwen3_5MoeForConditionalGeneration)
##   We already have: int4-AutoRound (Intel). WANT: W8A8 / FP8 / int8.
##   *** CRITICAL: no XPU int8-MoE kernel. INT8/W8A8 on this MoE will NOT serve on XPU. ***
##   BF16 baseline ~70 GB.
===========================================================================================

### VERIFIED (repo fetched, exists)

| repo id                                  | scheme / format                          | size on disk | fits 1x B70? | XPU-serveable                | verdict |
|------------------------------------------|------------------------------------------|--------------|--------------|------------------------------|---------|
| Qwen/Qwen3.6-35B-A3B-FP8                  | FP8 block-fp8 (block 128), native fp8 MoE | ~34.5 GB    | NO (needs 2 cards / offload) | GOOD (fp8 MoE path)          | **DOWNLOAD THIS for the 2-card setup** -- the only GOOD XPU large-precision option for the MoE. ~34.5 GB > 32, so 1 card needs --cpu-offload; comfortable on 2x B70. |
| palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4     | GPTQ-Int4 g128 sym, **moe_wna16**, +MTP   | ~24.4 GB    | YES          | RISKY (moe_wna16 GPTQ on XPU unproven; GPTQ XPU had load regressions) | int4 alt to our AutoRound; worth a serveability probe but not a precision upgrade. |
| cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit        | AWQ int4 (MoE)                            | ~24.3 GB    | YES          | RISKY (XPU AWQ-int4 WIP)     | int4 only; we already have int4-AutoRound (proven). Skip unless probing AWQ-on-XPU. |
| nvidia/Qwen3.6-35B-A3B-NVFP4             | NVFP4                                     | ~20 GB      | YES (size)   | NO (Blackwell fp4)          | Skip -- no Xe2 path. |
| unsloth/Qwen3.6-35B-A3B-NVFP4           | NVFP4                                     | ~20 GB      | YES (size)   | NO                          | Skip. |
| bartowski/Qwen_Qwen3.6-35B-A3B-GGUF     | GGUF                                     | varies       | depends      | NO (llama.cpp)              | Skip for vLLM. |
| unsloth/Qwen3.6-35B-A3B-MLX-8bit        | MLX 8-bit                                | ~37 GB      | -            | NO (Apple)                  | Skip. |
| noctrex/Qwen3.6-35B-A3B-MXFP4_MOE-GGUF  | MXFP4 GGUF                               | varies       | -            | NO                          | Skip. |

### SEARCHED, NOTHING FOUND (gaps -- no such repo exists)
  - **FP8 that fits 1 card** for the 35B MoE. Official fp8 is ~34.5 GB (>32). No smaller fp8 exists.
  - W8A8-INT8 / int8 (W8A16) for the 35B MoE. None found -- AND it would not matter: there is
    **no XPU int8-MoE kernel**, so an int8 MoE checkpoint could not serve on B70 even if published.
  - GPTQ-Int8 for the 35B MoE. None found.

-------------------------------------------------------------------------------------------
### MODEL 2 BOTTOM LINE
-------------------------------------------------------------------------------------------
  - FP8 (recommended, but 2-card): **DOWNLOAD `Qwen/Qwen3.6-35B-A3B-FP8`** (~34.5 GB). It is the
    only GOOD XPU-serveable >int4 checkpoint for the MoE. Does NOT fit 1x B70 -- target the 2-card
    setup, or --cpu-offload-gb on 1 card for scoring only.
        hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir Qwen3.6-35B-A3B-FP8
  - W8A8 / INT8: **DO NOT pursue for XPU.** No int8-MoE kernel on Xe2 -> won't serve. Stay on
    int4-AutoRound (have it) for the 1-card low-precision path; use the fp8 above for precision.
  - int4 alternatives (GPTQ/AWQ) exist and fit 1 card but are RISKY on XPU and not better than the
    AutoRound we already serve -- only download if specifically probing AWQ/GPTQ-MoE on XPU.

===========================================================================================
## ONE-LINE SUMMARY
===========================================================================================
  27B dense  : DOWNLOAD Qwen/Qwen3.6-27B-FP8 (~18.3 GB, 1-card, GOOD). For W8A8-INT8 -> must quant ourselves.
  35B-A3B MoE: DOWNLOAD Qwen/Qwen3.6-35B-A3B-FP8 (~34.5 GB, 2-card, GOOD). INT8/W8A8 -> not viable on XPU (no int8-MoE kernel).
