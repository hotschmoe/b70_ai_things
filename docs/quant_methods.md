# Quant method registry -- algorithm x scheme x model (B70)

**Created:** 2026-06-20
**Purpose:** ONE place that answers two questions: (1) *which quant algorithm do we intend to use for each
precision scheme, and why* (forward plan, Tables A/B); (2) *which method have we actually measured on which
model* (evidence ledger, Table C). Table D is the load-bearing part: the **XPU kernel gate** that decides what
is even servable on the B70 -- because the 4-bit-activation methods below depend on kernels we have not written.

**Authoritative numbers live in [`../evals/results/SUMMARY.md`](../evals/results/SUMMARY.md)** -- this doc is the
method-pivoted *index*, not a second source of truth for metrics.
**Detailed per-scheme recipes** (the `scripts/49` knobs) live in [`../MTP_TODO.md`](../MTP_TODO.md) Playbook B and
[`literature/07_w8a8_int8_recovery.md`](literature/07_w8a8_int8_recovery.md). **Action plan:** [`../RESEARCH_TODO.md`](../RESEARCH_TODO.md).

---

## Lever taxonomy (read this first)

Three *independent* levers. Methods combine across them -- they do not compete one-for-one. A full recipe usually
picks one from each row that applies to the scheme:

```
WEIGHT rounding      RTN  <  GPTQ ~= AutoRound            recovers weight-quant error; matters most at int4 weights
ACTIVATION smoothing SmoothQuant / AutoSmoothQuant / OS+  migrates act outliers into weights; the dominant W8A8 lever
ROTATION             SpinQuant / QuaRot / FlatQuant /     removes outliers by rotating hidden space;
                     QServe-rotation                       marginal at >=8-bit, DECISIVE at 4-bit acts (W4A8/W4A4)
```

Our measured root cause (doc 07 S2): the int8-**activation** quant is the W8A8 cost, not int8 weights. That is why
the activation + rotation levers matter more as we push activations below 8 bits, and why rotation is a *skip* at W8A8
but a *requirement* at W4A4.

---

## Table A -- method -> scheme PLAN (which stack we intend per precision)

| scheme | weights | activations | rotation | servable on B70 today? | intended method stack |
|---|---|---|---|---|---|
| FP8 (control) | fp8 | fp8 | none | YES (emulated -> bf16) | vLLM online dynamic FP8 -- the pristine baseline |
| W8A16 (quality ref) | GPTQ/RTN int8 | fp16 | none | **NO** (XPUwNa16 is int4-only) | reference only; near-lossless but no kernel |
| **W8A8 (PRIMARY)** | **GPTQ** (or AutoRound) int8 | per-token int8 + **selective SmoothQuant** | **skip** (marginal at 8-bit) | **YES** (our `XPUInt8ScaledMMLinearKernel`) | GPTQ weights + selective SmoothQuant on the 16 full-attn layers (doc 07 S3.1) |
| W4A16 (capacity) | **AutoRound** (>= GPTQ) int4 | fp16 | none | YES (`XPUwNa16`, 32-dim caveat) | AutoRound int4 weights; no act quant |
| W4A8 (w4a8/ agent + future) | AutoRound/GPTQ int4 | per-token int8 + SmoothQuant | **QServe-QoQ or SpinQuant** | PARTIAL (`int4_gemm_w4a8` exists; **online Hadamard does NOT**) | rotation enters here to rescue int8 acts at int4 weights |
| W4A4 (frontier) | int4 | **int4** | **FlatQuant** (acc) / **QuaRot** (kernel) | **NO** (needs int4xint4 GEMM + transform kernel) | FlatQuant first (SOTA acc, has Qwen2.5 tools) / QuaRot fallback (parameter-free Hadamard = cleaner kernel) / PrefixQuant for static acts -- see the W4A4 section below |

**Reading it:** the rotation column is empty until W4A8 and full at W4A4. That matches doc 07's verdict ("skip
rotation at W8A8; reconsider only at W4A8") -- the user's QServe/SpinQuant (W4A8) and QuaRot/FlatQuant (W4A4)
choices land exactly in the regime where rotation earns its cost.

---

## Table B -- method glossary (one line + lever + XPU note)

| method | arXiv | lever | what it does | XPU status |
|---|---|---|---|---|
| RTN | -- | (weight floor) | round-to-nearest, data-free | trivially servable; our baseline |
| **GPTQ** | 2210.17323 | WEIGHT | OBQ/Hessian-guided weight rounding w/ calibration | servable (just weights); **beat our RTN: +3.0 plus W8A8, +1.9 W4A16** |
| **AutoRound** ("autoint") | 2309.05516 | WEIGHT | Intel SignRound -- learned rounding via signed-gradient descent | servable; **our int4 leader** (27B 0.927) |
| **SmoothQuant** | 2211.10438 | ACTIVATION | per-channel act->weight scale migration, fused offline (free at runtime) | currently `SMOOTHQUANT=0` (hybrid pairing bug) -> selective fix queued (doc 07 S3.1) |
| **AutoSmoothQuant** | -- (SQ variant) | ACTIVATION | SmoothQuant with *automatic per-layer/per-node alpha* search vs one global alpha | same (free) kernel cost; NNCF per-node-alpha is our reference path |
| Outlier-Suppression+ | 2304.09145 | ACTIVATION | adds per-channel *shifting* for asymmetric outliers | port the transform; no kernel needed (offline) |
| **SpinQuant** | 2405.16406 | ROTATION | *learned* rotations (Cayley/Stiefel) removing outliers; W4A4/W4A8/KV4 | R1/R2 fuse offline (free); **R3/R4 online Hadamard = KERNEL-GATED** |
| **QuaRot** | 2404.00456 | ROTATION | Hadamard rotations via computational invariance; end-to-end W4A4KV4 | offline rotations free; **online Hadamard (down_proj in, attn) = KERNEL-GATED** |
| **FlatQuant** | 2410.09426 | ROTATION/transform | learnable Kronecker-factored affine transforms ("flatness"); SOTA W4A4 | applied online -> needs fused transform kernel (CUDA-only today) |
| **QServe / QoQ** | 2405.04532 | WEIGHT+ROTATION | W4A8KV4 = progressive group quant (two-level W4 scales) + SmoothAttention + rotation, paired with the QServe CUDA serving system (`mit-han-lab/omniserve`) | algorithm portable to *produce* a W4A8 checkpoint; its **serving kernels are CUDA**. A8 not A4 -> won't exercise an INT4 fastpath |
| **PrefixQuant** | 2410.05265 | ACTIVATION (static) | removes token-wise outliers so **static** per-tensor act-quant beats dynamic; supports W4A4 | static scales = **no per-token dynamic-quant overhead** (attractive for a custom kernel); but **only fake-quant code is public**, inference kernels later -> research-only today (`ChenMnZ/PrefixQuant`) |
| LLMC / Awesome-LLM-Quant | -- | (toolkit) | benchmarking harness wrapping several of these methods | use to A/B methods on Qwen3-14B (fake-quant) before committing kernel effort |

---

## Table C -- coverage / evidence matrix (what method we ran on what model)

The "show GPTQ beat RTN" ledger. `plus` = HumanEval+ plus pass@1 (164, thinking-off, greedy). `agree` = Tier-0
top-1 token agreement vs bf16. Numbers are anchors; SUMMARY.md is authoritative. Status: DONE / BLOCKED / PLANNED /
FUTURE (kernel-gated) / OTHER-AGENT.

| model | scheme | method (calib) | plus | agree | status / note |
|---|---|---|---|---|---|
| Qwen3-14B | W8A8 | **RTN** | 0.860 | 0.881 | DONE -- baseline |
| Qwen3-14B | W8A8 | **GPTQ** | **0.890** | 0.908 | **DONE -- beats RTN +3.0 plus; ties FP8 (0.890), beats FP8 base (0.921 vs 0.915)** |
| Qwen3-14B | W8A8 | SmoothQuant | -- | -- | BLOCKED -- `SMOOTHQUANT=0` (Qwen3.6 16/64 hybrid pairing bug); selective fix queued (RESEARCH_TODO 2a) |
| Qwen3-14B | W8A8 | AutoSmoothQuant | -- | -- | PLANNED -- per-node alpha sweep (RESEARCH_TODO 2e) |
| Qwen3-14B | W8A8 | AutoRound | **0.872** | -- | **DONE 06-23 -- GPTQ slightly WINS (gptq 0.890 vs AR 0.872); ~tie at int8 weights, as predicted (Track 3b)** |
| Qwen3-14B | W4A16 | **RTN** | 0.829 | 0.841 | DONE |
| Qwen3-14B | W4A16 | **GPTQ** | **0.848** | ~0.883 | DONE -- beats RTN +1.9 plus (within HumanEval CI) |
| Qwen3-14B | W4A16 | AutoRound | -- | -- | PLANNED -- int4 is AutoRound's home turf (RESEARCH_TODO 3a) |
| Qwen3-14B | W4A8 | RTN | 0.817 | 0.822 | DONE -- dominated (slowest decode + lowest quality) |
| Qwen3-14B | W4A8 | **AutoRound** | -- | -- | **OTHER-AGENT (`w4a8/`)** -- target plus >= 0.84 |
| Qwen3-14B | W4A8 | SpinQuant / QServe | -- | -- | FUTURE -- rotation rescue; online-Hadamard KERNEL-GATED (Table D) |
| Qwen3-14B | W4A4 | FlatQuant / QuaRot | -- | -- | FUTURE -- doubly kernel-gated (Table D + W4A4 section) |
| Qwen3-14B | W4A4 | PrefixQuant (static-a) | -- | -- | FUTURE -- static-act avoids per-token overhead; fake-quant code only today |
| Qwen3.6-27B | W4A16 | **AutoRound** | **0.927** | -- | DONE -- single-card quality LEADER |
| Qwen3.6-27B | W8A8 | GPTQ + sel-SmoothQuant | -- | -- | PLANNED -- needs 2 cards to serve (RESEARCH_TODO via MTP_TODO Phase C) |
| Qwen3.6-35B-A3B | W4A16 (MoE) | AutoRound | -- | -- | LOADS+GENERATES (1 B70) via 16-line INC-XPU MoE routing patch -> MoeWNA16/Triton; was "OOM"; see contrib/vllm_moe_xpu/ |
| Qwen3-14B / 27B | FP8 | online dynamic | 0.890 | 0.968 | DONE -- control (27B FP8 needs vLLM 0.23.0) |

**The headline this table exists to show:** at every scheme we've calibrated both ways, **GPTQ/AutoRound beat RTN**,
and GPTQ-W8A8 fully closed the int8 coding gap to FP8. Rotation methods (rows marked FUTURE) are the *next*
unexplored column -- but they are blocked on kernels, not on quant-time effort (Table D).

---

## GPTQ vs AutoRound -- the decision (2026-06-23, measured + literature-grounded)

**Neither is "always better"; the gap tracks weight-quant error (tiny at high bits, large at low bits) -- and it is
ACCURACY-only, never speed.** Literature crossover (same-model, published): int8 = WASH; int4 = marginal/often-wash
(GPTQ *wins* on Llama-2-13B W4G128); int3 = AutoRound ahead ~+8 but model-dependent (GPTQ *wins* on OPT-6.7B W3G128);
int2 = AutoRound dominates (+11..+33, GPTQ can collapse). **Skeptic flag:** every "AutoRound wins" number is Intel-sourced;
no independent leaderboard ranks the two, and Intel's own table has GPTQ winning ~7/39. Refs: arXiv 2309.05516, 2411.02355, 2502.13178; jarvislabs kernel-swap bench.

| Question | Answer |
|---|---|
| Is GPTQ always better? | **No** (and AutoRound isn't either). Our `w8a8 gptq 0.890 > AR 0.872` is a pure **int8 artifact**, not 14B-specific. |
| Move all quants to GPTQ? | **No** -- method is **perf-neutral** (below), so no speed reason; W8A8 is a wash, int4 is wash-to-marginal. Don't churn. |
| **W8A8 / int8** pick | **GPTQ** -- ~tie, but 1-shot/cheaper/validated + no XPU-calib risk. |
| **int4 (W4A16/W4A8)** pick | **AutoRound** (slight edge + auto-mixed-precision + we have working int4-AR serves) -- but the repo's 0.927-vs-0.848 is CONFOUNDED (27B-AR vs 14B-gptq, different models); same-model int4 is ~a wash. **Calibrate on CPU/CUDA, never the B70** (XPU-calib corrupts int4 -> Q8 garbage; lit confirms XPU-calib unreliable). |
| Does calibration change pp/TTFT/tg? | **No.** Same export FORMAT -> same kernel -> identical speed (calibration sets weight *values* only; kernel-swap bench: 10.9x from kernel alone). Only indirect lever = format knobs (act-order/sym/group) gating kernel eligibility. |
| More extensible (spec-decode/MTP)? | **Neither** -- that's an ENGINE concern. Both reach **compressed-tensors** (GPTQ via llm-compressor), which is the format the engine's MTP/EAGLE path loads. Quantizer = spec-decode-agnostic. (Our Lorbus int4-AR captures fine GRAPH=1 via `quantization=inc`, despite the lit's generic "AutoRound-XPU needs enforce-eager" caveat -- different path.) |
| Production cost | GPTQ 1-shot (cheapest at small scale; can OOM >70B). AutoRound gradient (iters=200; `light`=2-3x faster). |

**Policy:** W8A8 -> GPTQ (don't redo); int4 -> AutoRound (home turf, marginal) ON CPU/CUDA; export everything to
compressed-tensors for MTP composability. The bf16 vs quant and the *kernel/format* matter far more than gptq-vs-autoround.

---

## Table D -- the XPU kernel gate (why W4A8/W4A4 rotation is NOT plug-and-play)

This is the part that turns "use QServe/SpinQuant/QuaRot/FlatQuant" into a sequencing problem:

1. **Offline rotation = FREE to serve.** Rotations fused into the weights at quant time (SpinQuant R1/R2,
   QuaRot's invariance rotations) just produce *different weights* -- our existing int8/int4 kernels serve them
   unchanged. Capture these first.
2. **Online rotation = needs a kernel we don't have.** Runtime Hadamard/FWHT (SpinQuant R3/R4, QuaRot's online
   down_proj/value rotation, FlatQuant's per-layer transforms) requires a fast SYCL/XMX Hadamard kernel.
   **None exists for Intel** (doc 07 S6: CUDA-only -- Dao-AILab/HadaCore + an AMD ROCm port). Until we write one,
   these methods can only run in their *offline-only* variant (weaker outlier removal) or not at all.
3. **W4A4 also needs an int4 x int4 GEMM.** We have `int4_gemm_w4a8` (int4 w x **int8** a) only. W4A4 needs
   `s4 x s4 -> s32`. XMX supports INT4, so it is buildable -- but it is a NEW kernel. **=> W4A4 is DOUBLY gated**
   (missing Hadamard kernel AND missing int4-activation GEMM).
4. **QServe / FlatQuant ship fused CUDA kernels.** We would port the *algorithm* to emit a checkpoint, then either
   (a) serve an offline-rotated-only variant on our existing kernels, or (b) write the missing XPU kernel(s).

**Practical sequencing (so B70 time isn't wasted):**
- *Quantize* these on RunPod-NVIDIA / CPU; never block the B70 on quant-time work.
- *Serving* readiness order: **W8A8 / W4A16 (now)** -> **W4A8 offline-rotation** (SpinQuant R1/R2 path, no new kernel)
  -> **W4A8 online-rotation + W4A4** (after we write a SYCL Hadamard kernel, and for W4A4 an int4xint4 GEMM).
- Rotation stays a Track-8 / w4a8-branch item, NOT a near-term W8A8 task -- it buys nothing at 8-bit.

---

## W4A4-INT4 path -- the method pick (FlatQuant vs QuaRot) + Qwen3 caveats

For a *true* W4A4-INT4 Qwen3-14B (int4 weights AND int4 activations, to light a future int4xint4 XMX path -- NOT
FP4, NOT A8), the field's open-source tools split exactly along our kernel-design fork:

- **FlatQuant** (`ruikangliu/FlatQuant`, ICML'25) -- **best accuracy starting point.** Current SOTA on W4A4; clean
  repo; ships a real W4A4KV4 script (`--w_bits 4 --a_bits 4`) and `model_tools` for LLaMA2/3/3.1 + **Qwen2.5**
  (Qwen3-14B is close enough that we *adapt*, not write from scratch). Efficient kernels live in
  `deploy/kernels/kron_matmul.py` + `block_matmul.py` -- where our B70 int4 GEMM would slot in. **Catch:** its
  transform is **Kronecker-decomposed affine matrices fused into the GEMM**, so our kernel must apply a
  *pre-transform*, not just a plain int4 matmul.
- **QuaRot** (`spcl/QuaRot`, NeurIPS'24) -- **cleaner kernel target.** Parameter-free fixed **Hadamard** rotation, so
  our kernel stays a more standard **int4xint4 -> int32 GEMM + a Hadamard pre-multiply** (easier to map to XMX than
  FlatQuant's learned affine). Accuracy a notch below FlatQuant but well-established. **Llama-centric -> we add
  Qwen3 support ourselves.**
- **PrefixQuant** (`ChenMnZ/PrefixQuant`) -- the **static-activation** angle. Removes token-wise outliers so *static*
  per-tensor act-quant beats dynamic -> **no per-token dynamic-quant overhead**, attractive for a hand-written
  kernel. But only fake-quant code is public today (inference kernels later) -> accuracy validation only for now.
- **QServe/QoQ + SpinQuant -- NOT for this goal.** Both deploy at **W4A8** (int8 acts), so they won't exercise the
  int4 fastpath. Still, **read QServe's analysis**: they chose A8 because W4A4 per-group quant forces an
  **int32 -> FP dequant of partial sums** off the systolic array -- the exact XMX-accumulate tradeoff we'd navigate.
- **LLMC / Awesome-LLM-Quantization** -- a benchmarking harness to A/B these on Qwen3-14B (fake-quant) before
  committing kernel effort.

**Recommendation:** prototype with **FlatQuant** for accuracy (Qwen support + best W4A4 numbers); if the fused-affine
transform fights the XMX kernel design, **fall back to QuaRot's Hadamard** for a cleaner int4xint4 target. Both can
generate the quantized model in **fake-quant first**, so we validate perplexity *before* the real kernel exists.

**Two B70-specific gates (from Table D), restated for W4A4:** (1) a new `s4 x s4 -> s32` GEMM (we only have
int4xint8); (2) the transform kernel -- a fixed-Hadamard FWHT (QuaRot) or a fused Kronecker-affine (FlatQuant);
neither exists for SYCL/XMX.

**Qwen3-14B adaptation caveat (check this FIRST):** Qwen3 uses **QK-norm** and a different head config than Qwen2.5,
so the **attention rotation / Hadamard insertion points** differ -- that's the most likely place a FlatQuant
`model_tools/qwen2.py` (or QuaRot Llama) adaptation breaks. First step when this track opens: **diff Qwen3 vs
Qwen2.5 in FlatQuant's `model_tools`** to size the port before writing any kernel.

---

## How to use this doc

- **Picking a method for a new run:** Table A (intended stack) -> Table D (is it servable yet?) -> Table B (what it is).
- **After a run:** add/flip a row in Table C (method + model + scheme + plus/agree + status), and mirror the headline
  into `JOURNAL.md`. Keep full metrics in `evals/results/SUMMARY.md`; this table only tracks *coverage + the delta story*.
- **Boundary:** W4A8 rows are the `w4a8/` agent's -- update by cross-reading their results, don't fork their work.
