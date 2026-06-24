# 11 -- INT4/FP4 landscape + W4A8/W4A4 roadmap (B70, INT8-XMX-only)

**Created:** 2026-06-24. **Status:** SURVEY + PLAN (TODO -- nothing new measured here yet).
**Owns:** the low-bit format landscape (NVFP4/MXFP4/ROCmFP4/Intel), the W4A8 vs W4A4
literature synthesis, and the recommended next steps for our W4A8/W4A4 tracks.

**W4A8 is our next targeted research path.** This doc is a pointer target, not a
re-implementation. Authoritative siblings:
- [`w4a8/README.md`](../../w4a8/README.md) -- the live W4A8-INT8 branch.
- [`w4a8/AUTOROUND_W4A8_FEASIBILITY.md`](../../w4a8/AUTOROUND_W4A8_FEASIBILITY.md) -- why AutoRound cannot be our W4A8 quantizer.
- [`RESEARCH_TODO.md`](../../RESEARCH_TODO.md) Track 8 (W4A8 NEXT) / W4A4 DEFERRED.
- [`docs/quant_methods.md`](../quant_methods.md) -- method x scheme registry.

---

## 0. One-paragraph verdict

On a B70 (Battlemage / Xe2: INT8/INT4/INT2 + FP16/BF16 XMX, **no native FP4 or FP8
matrix compute**), the entire field's low-bit trajectory points at exactly the lane we
already target: **integer W4A8** -- store int4 weights, dequant in-register to int8, feed
the INT8 XMX datapath. The FP4 formats everyone is shipping (NVFP4, MXFP4) are
**Blackwell / CDNA4 silicon wins**; on hardware without FP4 tensor cores they degrade to
storage-only with on-the-fly dequant (vLLM literally dequants FP4 -> half), i.e. **zero
compute benefit**. There is no "Intel NVFP4" we can buy today; the first Intel matrix
engine with native FP8/FP4/MXFP4 is **Xe3P "Crescent Island," a 2H-2026 data-center part**,
not our cards. So: keep building integer W4A8/W8A8 kernels; do not chase FP4 formats.

---

## 1. Format landscape (verified against vendor docs + OCP spec)

### NVFP4 (NVIDIA, Blackwell)
- E2M1 4-bit element + **two-level scaling**: per-**16**-element micro-block scale in FP8
  **E4M3** (fractional), plus a per-tensor FP32 global scale.
- Hardware-accelerated **only on Blackwell 5th-gen Tensor Cores** (~20 PFLOPS FP4; GB300 ~7x
  GEMM over Hopper). Not on Hopper or earlier.
- Accuracy: DeepSeek-R1 FP8->NVFP4 <=1% degradation; ~2.3x throughput at matched accuracy vs
  FP8; pretraining a 12B/10T-token model in NVFP4 matched FP8 (MMLU-pro 62.58 vs 62.62).
- Refs: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>,
  arXiv:2509.25149.
- **B70 verdict: no compute value -- Blackwell-locked. Storage codec only.**

### MXFP4 / MX (OCP Microscaling v1.0, Sept 2023)
- Same E2M1 element, block size **32** sharing one **E8M0** scale (power-of-two only).
- vs NVFP4: block 32 vs 16, scale E8M0/pow2 vs E4M3/fractional. NVFP4's finer block + richer
  scale is why it converges to lower loss (MXFP4 needed ~1-1.5T extra tokens to catch up at 8B).
- Native on **NVIDIA Blackwell** and **AMD CDNA4 (MI355X)**. Intel native only on Xe3P.
- Refs: OCP MX v1.0 spec; arXiv:2310.10537.
- **B70 verdict: no compute value. Storage codec only.**

### AMD ROCmFP4 / Quark
- CDNA4/MI355X: native MXFP4/MXFP6/MXFP8 via scaled-MFMA (`v_mfma_scale_f32_*_f8f6f4`);
  FP4 = 10.1 PFLOPS dense. (Native FP4 is **CDNA4/MI355X**, not MI300/MI325.)
- **Quark is a quantizer, not Intel hardware.** Emits int3/4/8, fp8, mxfp4/6/8, bfp16; AWQ /
  GPTQ (incl. GPTQ-for-MXFP4) / SmoothQuant / Rotation / Qronos; consumes compressed-tensors.
- Our `quark-W8A8` 35B-A3B quant is plain integer int8 and runs on B70 XMX. **Caveat:** the
  exact compressed-tensors metadata schema was NOT primary-source-confirmed as drop-in for a
  generic vLLM loader -- verify `quantization_config` before assuming portability. Quark's
  FP8/MXFP4 outputs give B70 nothing.
- Refs: ROCm CDNA matrix-cores + MXFP4/6 blogs; Quark release notes; vLLM Quark docs.

### Intel (the decisive section)
- **Xe2 (Battlemage / B70) XMX = INT8 / INT4 / INT2 + FP16 / BF16 only. NO FP8, NO FP4.**
  INT8 runs at 2x FP16. **INT8/INT4 is the hardware ceiling on our cards.**
- Xe3 (Panther Lake) adds FP8 *dequant* but still does not *compute* FP8/FP4 -- INT8 stays the path.
- **Xe3P "Crescent Island"** (data-center, 2H-2026) is the first Intel matrix engine with native
  FP8/FP4/MXFP4/MXFP8. Not our box.
- Gaudi 2/3: native FP8, no INT8/MXFP4 (separate accelerator family).
- Software: INC + AutoRound emit INT2/3/4/8 and can *export* MXFP4/NVFP4/FP8 -- but on
  Battlemage those run **dequantized/emulated**, not on a native low-bit datapath. AutoRound's
  recommended schemes are W4A16-centric.
- Caveat: Xe2/Xe3 XMX datatype specifics rest partly on HWCooling's reconstruction of Intel
  disclosures (oneAPI XMX primary doc blocked automated fetch; it documents only INT8/BF16/FP16).
- **B70 verdict: FP4/FP8 compute is unbuyable today. Integer W4A8/W8A8 is the correct and only
  fast lane.**

---

## 2. W4A8 SOTA -- this is our headline lane

The systems literature converged on exactly our design (int4 weight x int8 activation, all
math on INT8 tensor cores). These are directly actionable for our `int4_gemm_w4a8` rewrite.

- **QServe / QoQ -- W4A8KV4 (arXiv:2405.04532, hanlab.mit.edu/projects/qserve).** The one to
  read closely (user-requested). Two-level *progressive group quantization* keeps ALL compute on
  INT8 tensor cores; dequants int4->int8 **in-register via subtract-after-multiply**; chooses
  **weight-dequant over partial-sum-dequant** (lower register pressure) + compute-aware weight
  reordering. Throughput vs TensorRT-LLM: 1.2-1.4x (Llama-3-8B), 2.4x A100 / 3.5x L40S
  (Qwen1.5-72B). Accuracy loss vs FP16: 1.03 / 0.89 / 0.40% (7B/13B/70B). **Adds KV4** on top of
  W4A8 -- the "KV4" piece is the part we have not explored.
- **QQQ (arXiv:2406.09904).** Marlin-derived W4A8 GEMM: INT4->INT8 unpack, INT8 GEMM, dequant
  INT32->FP16. Per-group W4A8 GEMM **2.51x over W8A8** and **1.71x over W4A16-Marlin**;
  end-to-end Llama-2-13B 2.10x vs W8A8 / 1.25x vs W4A16; PPL within 0.13 of W8A8.
- **LiquidGEMM (arXiv:2509.01229).** 2-instruction UINT4 unpack -- cheap dequant-on-fast-path
  technique relevant to the GEMV prologue.
- **Design rules that map onto B70 XMX:** dequant on the fast path, never spill to scalar cores;
  prefer weight-dequant; in-register int4->int8. Our native oneDNN `int4_gemm_w4a8` is the right
  *shape* but its M=1 decode/GEMV path is unoptimized (see section 4).

## 3. W4A4 SOTA -- note, do NOT start

- W8A8 (SmoothQuant): ~lossless. W4A8KV4 (QoQ): ~1%. W4A4 baseline (QuaRot): +0.63 PPL at 7B,
  **collapses on Llama-3-70B (55 ppl)**. W4A4 SOTA (FlatQuant 2025, arXiv:2410.09426): **<1% drop
  even on Llama-3-70B**.
- QuaRot (arXiv:2404.00456): Hadamard rotation, 1.97-3.33x prefill, but ~4pt zero-shot gap at 7B,
  fragile on Llama-3. SpinQuant (arXiv:2405.16406): learned rotations, narrows to ~2.9pt; beats
  SmoothQuant by 25pt at A4 -- naive 4-bit activation quant is catastrophic. Atom (arXiv:2310.19102):
  mixed INT8-outlier/INT4-rest.
- **Not production-viable as a default:** model-fragile + systems story lags (W4A4 dequant spills
  to slow scalar cores -- exactly why MIT ships W4A8KV4, not W4A4). Our "later frontier" stance is
  correct.
- **If pursued later** it needs a new **s4 x s4 -> s32 GEMM** *plus* a **Hadamard/rotation kernel**
  (online transform). Do not start until W8A8/W4A8 are robust.

---

## 4. Recommended next steps

### W4A8 (active -- Track 8)
1. **Re-quant the 27B W4A8 -- it is too big (PRIORITY, user-flagged).** The shipped 27B-W4A8 is
   ~16 GB on disk and pays a ~28 GiB unpacked-int8 GPU transient at load (mitigated today by the
   `VLLM_W4A8_PREPACKED=1` prepack). We want a **smaller targeted artifact**: re-pack int4 on
   disk (target ~9 GB, load ~39s -> ~23s per `w4a8/README.md` Win C) AND re-run the quant with a
   **good GPTQ/SmoothQuant recipe** (see #2) rather than the current sqgptq prepack. Goal: smaller
   on disk, smaller load transient, and accuracy >= w4a16-gptq (0.848).
2. **Nail down a good GPTQ/SmoothQuant recipe for our targeted W4A8 (user-flagged).** AutoRound is
   a DEAD END as a W4A8 quantizer (hard-asserts bits==8 -- `w4a8/AUTOROUND_W4A8_FEASIBILITY.md`),
   so the int8-XMX W4A8 path must come from compressed-tensors GPTQ + SmoothQuant. Build a
   repeatable recipe (group size, SmoothQuant alpha, down_proj/rotation-skip handling, calibration
   set) and validate on **harder evals**, not just the 14B HumanEval+ prior. Also scan for newer
   methods that beat plain GPTQ+SQ for W4A8 (Qronos, rotation-based pre-quant, QoQ-style progressive
   group quant) -- the QServe/QQQ papers describe quantizers, not just kernels.
3. **Optimize the `int4_gemm_w4a8` M=1 decode/GEMV path.** Biggest single-stream gap: 16.5 t/s
   eager vs w4a16's 29 at identical 9.3 GiB VRAM -> unoptimized oneDNN int4 GEMV + per-token
   act-quant. Apply QServe/QQQ/LiquidGEMM rules (in-register int4->int8, weight-dequant,
   cheap-unpack) and **fuse the per-token act-quant into the GEMM prologue** -- the same fusion is
   the last W8A8 decode headroom, so it pays off twice.
4. **Read W4A8KV4 (QServe/QoQ) closely (user-flagged)** -- specifically the KV4 piece, which we
   have not explored, and the progressive-group-quant quantizer that feeds #2.
5. **Honor the concurrency-niche measurement gate first** (`w4a8/README.md`): sweep W4A8 vs W4A16
   vs W8A8 aggregate t/s at C1-64 to confirm int8 activations buy a throughput niche before sinking
   weeks into the GEMV.

### W4A4 (deferred -- frontier)
- Keep notes current (FlatQuant is the method to watch). **Do not start kernels** until W8A8/W4A8
  are robust. Entry cost when we do: s4 x s4 -> s32 GEMM + online Hadamard/rotation kernel.

### FP4 formats (NVFP4 / MXFP4)
- **Skip entirely** until Intel native FP4 hardware (Xe3P "Crescent Island", 2H-2026) exists. On
  B70 they are storage codecs with no compute win; integer W4A8 dominates.

### Adjacent gap worth naming (quark-W8A8 MoE linear layers)
- The 35B-A3B quark-W8A8 path runs the **256 routed experts as true int8** (Triton fused-MoE),
  but the **shared-expert / attn linear layers run dequant-to-bf16 (effectively W8A16)** because
  XPU has no *registered* int8 scaled-mm linear kernel -- so those layers fall back to a
  correctness-first bf16 GEMM (`rdy_to_serve/qwen36-35b-a3b-quark-w8a8-int8/patches/quark.py`,
  `QuarkW8A8Int8DequantXPU`).
- We **already have** the dense `XPUInt8ScaledMMLinearKernel` (`contrib/vllm_int8_xpu/`).
  Registering it as the XPU scaled-mm entry for those linear layers is a **documented, contained
  upgrade** -- it makes the minority linear path true W8A8 (memory + potential compute win) without
  a new kernel. Low-risk, and it also unblocks the same registration the int8 MoE work wants.

---

## 5. Sources

NVFP4: nvidia developer blog (introducing-nvfp4...), arXiv:2509.25149. MX/MXFP4: OCP MX v1.0
spec, arXiv:2310.10537. AMD: ROCm matrix-cores + MXFP4/6 blogs, Quark release notes, vLLM Quark
docs. Intel: HWCooling Xe2/Xe3 analyses, Crescent Island/Xe3P announcement, intel/neural-compressor,
intel/auto-round. W4A4: QuaRot 2404.00456, SpinQuant 2405.16406, Atom 2310.19102, FlatQuant
2410.09426. W4A8: QServe/QoQ 2405.04532, QQQ 2406.09904, LiquidGEMM 2509.01229.
