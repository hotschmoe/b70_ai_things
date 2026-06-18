# 05 — W8A8 INT8 Quantization for Qwen3-14B on Intel Arc Pro B70 (Battlemage / vLLM-XPU)

**Date:** 2026-06-17
**Goal:** Produce a true W8A8 INT8 checkpoint of Qwen3-14B (dense, GQA) and engage the Battlemage INT8 XMX fast path (the ~367 INT8 TOPS) under vLLM-XPU.
**Environment:** Qwen3-14B BF16 local; from-source upstream vLLM-XPU `0.20.2rc1.dev2+gc51df4300`, image `vllm-xpu-env:tf`, torch `2.10+xpu`. FP8 already works (vLLM selects `XPUFP8ScaledMMLinearKernel`).

---

## ⛔ TL;DR — The honest answer up front

**As of June 2026, upstream vLLM-XPU has NO kernel that runs true INT8 W8A8 (int8 weights × int8 activations → int32 INT8 GEMM) on Battlemage.** There is no XPU entry in vLLM's INT8 scaled-MM kernel registry, the XPU `scaled_mm` kernel is **FP8-only**, and the Triton INT8 fallback is gated to CUDA. So a compressed-tensors `W8A8` (INT8) checkpoint will **not** light up the 367 INT8 TOPS XMX path on B70 today — it will either fail to find a kernel or (in a vendor build) fall back to a dequant-to-FP16 path that throws away the INT8 advantage.

**What you should actually do:**
- **For an accurate-yet-fast 8-bit path on B70 right now → keep using FP8** (`XPUFP8ScaledMMLinearKernel`). It is the only verified, kernel-accelerated 8-bit path on Battlemage, and at decode (batch-1, bandwidth-bound) INT8 would not beat it anyway (both are ~1 byte/weight).
- **If you specifically want the INT8 XMX engines lit up**, the only XPU INT8-compute kernel that exists upstream is **W4A8** (`XPUW4A8IntLinearKernel`: int4 weights, **per-token int8 activations**, oneDNN). That feeds the INT8 XMX path with int8 activations and gives you smaller weights too — but it is W4A8, not W8A8, and needs kernel-readiness verification on your build.
- **Produce the W8A8 INT8 checkpoint anyway if you want it for portability/CPU**, but treat XPU acceleration as *unverified / not-yet-supported*. The recipe is below.

> ⚠️ **Unverified / cross-arch flag:** the community "B70 ran AMD Quark W8A8 INT8" claim could not be confirmed. The Quark INT8 W8A8 reference path in vLLM is CUTLASS/ROCm; there is no evidence of a Quark INT8 XPU kernel. Treat any "Quark W8A8 on B70" report as **using a dequant fallback, not the INT8 XMX path**, unless a `XPU...Int8...Kernel` log line proves otherwise.

---

## 1. Which W8A8 format does vLLM-XPU actually accelerate on Battlemage?

Decisive evidence from the upstream source (main, June 2026):

| Candidate format | How vLLM dispatches it | Lands on XPU INT8 XMX kernel? | Verdict |
|---|---|---|---|
| **(a) compressed-tensors INT8 W8A8** (`llmcompressor --scheme W8A8`) | `CompressedTensorsW8A8Int8` → `init_int8_linear_kernel()` → `choose_scaled_mm_linear_kernel(_POSSIBLE_INT8_KERNELS)` | **NO.** `_POSSIBLE_INT8_KERNELS` has entries only for **CPU** (`Zentorch`, `CPUInt8`), **CUDA** (`CutlassInt8`, `TritonInt8`), **ROCM** (`AiterInt8`, `TritonInt8`). **There is no XPU entry.** On XPU this raises *"Failed to find a kernel that can implement the ScaledMM linear layer"* (or, in a vendor build, falls back to a generic dequant path). | ❌ Not accelerated on XPU |
| **(b) AMD Quark W8A8 INT8** | Quark linear method → same INT8 scaled-MM dispatch (CUTLASS/ROCm reference) | **NO.** No Quark INT8 XPU kernel exists. Auto-detection of pre-quantized non-FP8 models on XPU is itself buggy (see Pitfalls). | ❌ Not accelerated on XPU; unverified community claim |
| **(c) IPEX WoQ INT8 (`lowp_mode=INT8`)** | IPEX `WeightOnlyQuantizedLinear` | **NO — it's weight-only (W8A16).** IPEX WoQ stores weights quantized; **activations stay FP16/BF16**. `lowp_mode=INT8` controls the *internal compute dtype of a WoQ GEMM*, not true INT8×INT8 activation quant. This is not W8A8. | ❌ Not W8A8 (it's W8A16) |
| **(d) ✅ The ONE int8-activation XPU kernel that exists** — **W4A8** (`XPUW4A8IntLinearKernel`, `vllm.model_executor.kernels.linear.mixed_precision.xpu`) | int4 group-quantized weights (packed uint4) + **per-token symmetric int8 activations**, oneDNN `int4_gemm_w4a8` | **YES — this is the only upstream XPU path that produces int8 activations into the XMX INT8 datapath.** But it is **W4A8, not W8A8.** | ✅ Only int8-activation XPU path, but W4A8 |

**Bottom line for Q1:** None of (a)/(b)/(c) engages the Battlemage INT8 XMX path. The XPU `scaled_mm` kernel (`XPUFP8ScaledMMLinearKernel`) explicitly rejects non-FP8: *"XPUFP8ScaledMM only support FP8 weight dtype"*. The H1-2026 Intel XPU quantization roadmap (vLLM #37979) targets only **`wNa16 (INT)`** and **`w8a16 (FP8)`** scheme coverage for XPU — **INT8 W8A8 is not on the roadmap.**

### How to VERIFY which kernel is engaged (grep these in the vLLM startup log)

Run with `VLLM_LOGGING_LEVEL=DEBUG` and grep stderr:

```bash
# What you WANT to see for an accelerated 8-bit path on B70 today (FP8):
#   "Using XPUFP8ScaledMMLinearKernel"           ← FP8 XMX path, GOOD

# What proves INT8-activation XMX is engaged (only via W4A8):
#   "XPUW4A8IntLinearKernel"  / op "int4_gemm_w4a8"

# RED FLAGS — INT8 W8A8 did NOT hit an XMX kernel:
#   "Failed to find a kernel that can implement the ScaledMM linear layer"  ← hard fail
#   "CutlassInt8ScaledMMLinearKernel"   ← CUDA-only; cannot run on XPU (would error)
#   any "dequant" / weights upcast to fp16 before matmul  ← slow fallback, no INT8 TOPS
```

Concrete grep recipe:
```bash
vllm serve <model> ... 2>&1 | tee serve.log
grep -iE "ScaledMMLinearKernel|LinearKernel|kernel that can implement|w4a8|w8a8|int8|fp8|dequant" serve.log
```
The **LinearMethod / kernel class name** in the log is the ground truth. If you do not see an `XPU...Kernel` line for the quantized layers, you are not on an XMX fast path.

---

## 2. Exact recipe to quantize Qwen3-14B to W8A8 with llm-compressor

This produces a standard compressed-tensors **W8A8 INT8** checkpoint (per-channel symmetric int8 weights + per-token **dynamic** int8 activations). It is the correct, portable artifact — just be aware (Section 1) that **XPU will not currently accelerate it**; it will accelerate on CUDA/ROCm/CPU.

> **Run on CPU.** Confirmed: llm-compressor calibration on XPU is unreliable, and the int8 compute kernels assume CUDA. CPU-only calibration is the safe path. Your box (128 GB RAM, 32 threads) handles a 14B GPTQ calibration comfortably (weights ~28 GB BF16 in RAM + Hessians; well under 128 GB). Expect **30–90 min** on CPU for 512 samples with GPTQ; use SmoothQuant+RTN (drop GPTQ) for a ~5–10 min run if you accept slightly lower accuracy.

**Install:**
```bash
pip install "llmcompressor>=0.8.0" "compressed-tensors" "transformers>=4.52" datasets accelerate
# llm-compressor 0.8.0+ added explicit Qwen3 support.
```

**Expected output:** directory `Qwen3-14B-W8A8-INT8/`, ~**14–15 GB** (int8 weights ≈ ½ of the ~28 GB BF16; plus scales/config).

### `quantize_w8a8.py` (complete, runnable, CPU-only)

```python
#!/usr/bin/env python3
"""
Quantize Qwen3-14B (BF16) -> compressed-tensors W8A8 INT8.
Scheme W8A8: per-channel symmetric int8 weights + per-token DYNAMIC int8 activations.
CPU-ONLY calibration (XPU calibration is unreliable). 128 GB RAM / 32 threads is sufficient.

NOTE: The resulting checkpoint is NOT accelerated by upstream vLLM-XPU on Battlemage
as of June 2026 (no XPU INT8 W8A8 kernel). It IS accelerated on CUDA/ROCm/CPU.
For an accelerated 8-bit path on B70, prefer FP8. See 05_w8a8_recipe.md Section 1.
"""

import os
# ---- Force CPU; keep XPU/CUDA out of the calibration path ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ZE_AFFINITY_MASK"] = ""          # hide Level-Zero / XPU devices
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("HF_HUB_OFFLINE", "1") # weights are local

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# ---- Config ----
MODEL_ID = "/path/to/Qwen3-14B"          # <-- local BF16 weights
SAVE_DIR = "Qwen3-14B-W8A8-INT8"
NUM_CALIBRATION_SAMPLES = 512            # 256 is fine if you want it faster
MAX_SEQUENCE_LENGTH = 2048

# ---- Load on CPU in bf16 ----
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ---- Calibration data: instruction-style text via the model's chat template ----
ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                  split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False,
                     max_length=MAX_SEQUENCE_LENGTH, truncation=True,
                     add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# ---- Recipe: SmoothQuant (eases activation quant) -> GPTQ W8A8 ----
# scheme="W8A8" => weights: int8 per-channel symmetric; activations: int8 per-token DYNAMIC.
# ignore lm_head (kept high precision); Qwen3 needs no special module ignores for the dense 14B.
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ---- Save compressed (int8 on disk) ----
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved W8A8 INT8 checkpoint to: {SAVE_DIR}")
```

**Faster RTN variant** (no GPTQ Hessians, ~5–10 min on CPU; slightly lower accuracy): replace the `recipe` with
```python
from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```

---

## 3. How to serve the resulting checkpoint on vLLM-XPU

vLLM **auto-detects** compressed-tensors from `config.json` (`quantization_config`), so `--quantization` is usually unnecessary; pass it explicitly only to force the path or aid debugging.

```bash
# Auto-detected (preferred):
ZE_AFFINITY_MASK=0 \
VLLM_LOGGING_LEVEL=DEBUG \
vllm serve /path/to/Qwen3-14B-W8A8-INT8 \
  --device xpu \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  2>&1 | tee serve_w8a8.log

# Explicit (debugging):  --quantization compressed-tensors
# (For a Quark artifact you would use: --quantization quark)
```

**B70 env/flags worth setting** (carry over from your working FP8 setup): `--device xpu`, `--dtype float16` (Battlemage prefers fp16 over bf16 for the quant kernels), single-GPU `ZE_AFFINITY_MASK=0`. If you hit MLA issues on other models, `VLLM_MLA_DISABLE=1` — not needed for dense Qwen3-14B.

**Then immediately verify (do not trust that it "loaded"):**
```bash
grep -iE "ScaledMMLinearKernel|LinearKernel|kernel that can implement|dequant|int8|fp8" serve_w8a8.log
```
If you see *"Failed to find a kernel..."* or a CUDA-only `CutlassInt8...` reference (which cannot execute on XPU), the W8A8 INT8 path is **not** working on B70 — fall back to FP8 (Section 1).

---

## 4. Pitfalls

- **Ignored layers.** Always `ignore=["lm_head"]`. The embedding/`lm_head` and router/norm layers must stay high precision; quantizing `lm_head` to int8 noticeably degrades quality. The dense Qwen3-14B has no MoE router to ignore.
- **Activation-quant ordering.** Run **SmoothQuant before** weight quantization. SmoothQuant migrates activation outliers into the weights (which tolerate them via per-channel scales) so per-token int8 activation quant stays accurate. Reversing the order, or skipping SmoothQuant on a model with activation outliers, costs accuracy.
- **Dynamic vs static activations.** `W8A8` here uses **per-token dynamic** activation quant (scales computed at runtime) — no activation calibration scales baked in, more robust, and what vLLM's INT8 path expects. Static per-tensor activation int8 is more fragile; avoid unless you have a reason.
- **Qwen3 specifics.** Use `llmcompressor >= 0.8.0` (added Qwen3 support). For the **dense** 14B, no special module handling is needed. (The known Qwen3-*Next*-80B-A3B "random params after W8A8-int8" bug — llm-compressor #2059 — is an **MoE** issue and does **not** apply to dense Qwen3-14B.)
- **XPU auto-detect routing bug.** vLLM on XPU has mis-routed some pre-quantized non-FP8 models (e.g. AWQ-int4 → torchao → CUDA-only crash: llm-scaler #269). If auto-detect misbehaves, set `--quantization compressed-tensors` explicitly. GPTQ/W4A16 on XPU was itself **broken in v0.19.0** (missing XPU branches in `gptq.py`, #39474) and needed a patch — sanity-check your build before relying on any INT weight-only path.
- **⚠️ Is INT8 W8A8 actually faster than FP8 on Battlemage? Honestly: NO (and it doesn't even run accelerated).** Two independent reasons:
  1. **No XPU INT8 W8A8 kernel exists** (Section 1) — so it can't be faster; it isn't accelerated at all on B70.
  2. **Even if a kernel existed, decode is bandwidth-bound.** At batch-1 decode, throughput is set by bytes-of-weights-read-per-token. INT8 and FP8 are **both ~1 byte/weight**, so they have the *same* memory traffic and the INT8 TOPS advantage (a compute-side win) is irrelevant at small batch. INT8's edge over FP8 only appears in **compute-bound prefill / large-batch** regimes. The empirical data point (llm-compressor #2549: W8A8 **52.9% slower** than FP16 at batch-1 on an H20, decode-bound) reinforces this. **Conclusion: do not expect INT8 W8A8 to beat FP8 on B70 for typical single-stream serving.** FP8 is the right 8-bit choice today.

---

## Sources

- vLLM RFC — XPU kernel migration to vllm-xpu-kernels (FP8 scaled_mm done; INT4 w4a16; no INT8 W8A8): https://github.com/vllm-project/vllm/issues/33214
- vLLM RFC — Intel Quantization Support Roadmap H1 2026 (XPU scheme coverage = wNa16 INT + w8a16 FP8 only): https://github.com/vllm-project/vllm/issues/37979
- vLLM source — compressed_tensors dispatch (`CompressedTensorsW8A8Int8`): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
- vLLM source — INT8 kernel registry (`_POSSIBLE_INT8_KERNELS`, no XPU entry): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/__init__.py
- vLLM source — XPU scaled_mm kernel (FP8-only, `XPUFP8ScaledMMLinearKernel`): https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/kernels/linear/scaled_mm/xpu.py
- vLLM API — XPU mixed_precision kernel (`XPUW4A8IntLinearKernel`: int4 weight + per-token int8 activation, oneDNN): https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/mixed_precision/xpu/
- vLLM docs — INT8 W8A8 quantization (Nvidia compute-capability >7.5 noted): https://docs.vllm.ai/en/latest/features/quantization/int8/
- llm-compressor — W8A8 INT8 example (SmoothQuant + GPTQ, per-channel weight + per-token dynamic activation): https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_int8/
- llm-compressor 0.8.0 — extended Qwen3 support: https://developers.redhat.com/articles/2025/10/07/llm-compressor-080-extended-support-qwen3-and-more
- llm-compressor #2549 — "W8A8 slower than FP16" (decode bandwidth-bound, batch-1, CutlassInt8 kernel): https://github.com/vllm-project/llm-compressor/issues/2549
- llm-compressor #2059 — Qwen3-Next-80B (MoE) W8A8-int8 random-params bug (MoE only, not dense): https://github.com/vllm-project/llm-compressor/issues/2059
- vLLM #39474 — GPTQ (W4A16) broken on XPU in v0.19.0, needs patch (verified on Arc Pro B70): https://github.com/vllm-project/vllm/issues/39474
- intel/llm-scaler #269 — XPU auto-detect mis-routes pre-quantized AWQ-int4 to CUDA-only torchao: https://github.com/intel/llm-scaler/issues/269
- intel/llm-scaler README — B70 supported quant = Dynamic Online FP8 / Int4 / MXFP4 (no INT8 W8A8 listed): https://github.com/intel/llm-scaler/blob/main/README.md
- IPEX XPU WoQ docs (weight-only, "unlike w8a8 we focus on weight-only"): https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm/int4_weight_only_quantization.html
- Phoronix — Intel LLM-Scaler vLLM 0.14-b8.2 with official Arc Pro B70 support: https://www.phoronix.com/news/Intel-LLM-Scaler-vllm-0.14-b8.2
- AMD Quark on vLLM (INT8 W8A8 / MXFP4 / FP8; CUTLASS-ROCm reference, no XPU INT8 kernel): https://docs.vllm.ai/en/latest/features/quantization/quark/

> **Flagged unverified / cross-arch claims:** (1) Any "community B70 ran AMD Quark W8A8 INT8 on the XMX path" report — no Quark INT8 XPU kernel exists; treat as a dequant fallback unless an `XPU...Int8...Kernel` log line proves otherwise. (2) The llm-compressor #2549 INT8-slower benchmark is on Nvidia H20, used here only to illustrate the decode-bandwidth-bound argument, which is architecture-independent. (3) `XPUW4A8IntLinearKernel` is documented upstream but its kernel-readiness on your exact `0.20.2rc1` Battlemage build should be confirmed by grepping for the kernel name at load time.
