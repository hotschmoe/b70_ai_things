# Quant quality — Qwen3-14B on the Arc Pro B70 (first campaign, 2026-06-19)

How much each quantization degrades the **same** Qwen3-14B vs its BF16 self. Served one-at-a-time on a
single B70, vLLM 0.23.0-based images, greedy/eager, eval concurrency 1, thinking **off**.

## Results

| quant | weights / acts | calib | ppl ↓ | top1-agree vs bf16 ↑ | nll-gap ↓ | gsm8k (n=150) ↑ | serves on B70 |
|---|---|---|---|---|---|---|---|
| **bf16** (reference) | 16 / 16 | — | 12.7010 | — | — | — | ❌ ~29.6 GB > one card |
| **fp8** | 8fp / 8fp | online | 12.6966 | 0.968 | 0.062 | 0.960 (144/150) | ✅ XPU FP8 |
| **w8a16** | int8 / 16 | RTN | 12.7596 | **0.981** | 0.037 | — *(can't serve)* | ❌ no XPU kernel |
| **w8a8** | int8 / int8 | RTN | 13.0839 | 0.881 | 0.250 | 0.953 (143/150) | ✅ **our int8 kernel** |
| **w4a16** | int4 / 16 | RTN | 13.5528 | 0.841 | 0.340 | 0.947 (142/150) | ✅ XPUwNa16 |
| **w4a8** | int4 / int8 | RTN | 14.1943 | 0.822 | 0.420 | 0.927 (139/150) | ✅ XPUW4A8Int |

- **top1-agree** = fraction of 1063 corpus tokens where the quant's greedy argmax == bf16's (1.0 = identical).
- **ppl** on a fixed prose+code corpus. **gsm8k**: thinking-off, greedy, `#### <n>` exact-match, first 150 test items (paired).
- bf16 + w8a16 scored **offline on CPU** (neither serves on one card); tokenization verified identical to vLLM /tokenize (0/10 misaligned).

## Headline: the int8 **activation** quant is the quality cost, not the int8 **weights**

Decompose by holding weights fixed and changing only the activation precision:

| weights | acts 16-bit | acts int8 | Δ agreement from int8 acts |
|---|---|---|---|
| **int8** | w8a16: **0.981** agree, ppl 12.76 | w8a8: 0.881 agree, ppl 13.08 | **−0.100** |
| **int4** | w4a16: 0.841 agree, ppl 13.55 | w4a8: 0.822 agree, ppl 14.19 | −0.019 |

- **int8 weights are nearly free** — W8A16 (int8 w, fp16 acts) is essentially lossless (ppl 12.76 ≈ bf16
  12.70, 98.1% token agreement — even *higher* than fp8's 96.8%).
- **Quantizing activations to int8 is what costs fidelity** — it drops int8-weight token agreement from
  98.1% → 88.1% (−10 pts). Same direction at int4.
- **…but it barely moves the task metric.** W8A8 gsm8k 95.3% ≈ fp8 96.0%. The int8 activation quant flips
  many *low-confidence* tokens (hence the agreement drop) but rarely the final answer. So the cheap
  deterministic canary (agreement) is *more* sensitive than gsm8k — which is the point of Tier 0.
- **Weight bits dominate ppl/task:** W8A8 (int8 w) > W4A16 (int4 w) on every metric, even though W4A16
  keeps full-precision activations. 4-bit weights cost more than int8 activations.

## Kernel-coverage map (B70, vLLM 0.23.0 + our kernels) → priorities

| scheme | XPU kernel | lights INT8 systolic path? |
|---|---|---|
| fp8 | ✅ XPU FP8 | no (fp) |
| **W8A8 int8** | ✅ **ours** (`XPUInt8ScaledMMLinearKernel`) | **yes** |
| W4A8 int | ✅ `XPUW4A8IntLinearKernel` | yes (int8 acts) |
| W4A16 (int4 w-only) | ✅ `XPUwNa16` | no (fp16 acts) |
| **W8A16 (int8 w-only)** | ❌ **MISSING** — `XPUwNa16` is int4-only (uint4/uint4b8) | no (fp16 acts) |

**Where to focus kernel work:**
1. **Keep optimizing W8A8 (our int8 kernel) — it's the speed sweet spot.** Near-fp8 *task* quality
   (gsm8k 95.3 vs 96.0), serves at 15 GB, and it's the only path that lights the B70's INT8 systolic
   datapath (the hardware's real advantage). The decode GEMV / quant K-loop are the remaining wins.
2. **A W8A16 (int8 weight-only) kernel is the one coverage gap, and it would be near-lossless** (0.981
   agreement). But it keeps fp16 activations → it's a *memory-savings / max-fidelity* play, NOT a
   compute-speed one (no INT8 matmul). Write it only if a quality-critical, latency-tolerant use case
   appears; for the throughput coding-server, W8A8 wins.
3. **W4A8 only when memory-bound** (9.3 GB) — it costs real quality (gsm8k 92.7%, 0.822 agreement).

## Caveats (don't over-read)

- **gsm8k n=150** → ~±2.5% per cell; the fp8↔w8a8↔w4a16 gsm8k gaps are within/near noise. The **tight
  signal is ppl + agreement** (deterministic, 1063 paired tokens) — trust those for fine ordering.
- All non-fp8 quants are **RTN (data-free)** here; GPTQ/SmoothQuant calibration would lift them somewhat
  (esp. w4a8/w4a16). This campaign compares *schemes at RTN*, not best-achievable per scheme.
- **Noise floor ≈ 0 for Tier 0** (greedy, concurrency-1): W8A8 ppl was **13.0839 identical across two
  independent runs**, so the ppl/agreement deltas here are real signal, not run-to-run wobble. (A
  bf16-vs-bf16 run isn't possible — bf16 won't serve — but quant-vs-itself reproduces exactly.)
- 14B-class, thinking-off. **May not transfer to 27B / thinking-on** — re-run when card #2 lands.

## Repro

```
# serve a quant (see ../configs/models.yaml), then:
python ../orchestrator/run_evals.py --endpoint http://192.168.10.5:18080/v1 --model <id> --quant <label> --tiers 0,2 --limit 150
# offline CPU score for non-servable / reference (bf16, w8a16): scripts/55_tier0_reference_cpu.sh SRC=... QLABEL=...
# divergence matrix: python ../orchestrator/tier0_matrix.py ../results bf16
```
