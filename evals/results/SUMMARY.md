# Quant quality — Qwen3-14B on the Arc Pro B70 (first campaign, 2026-06-19)

Measuring how much each quantization degrades the **same** Qwen3-14B vs its BF16 self. All served on
one B70 (one model at a time), vLLM 0.23.0-based images, greedy/eager, eval concurrency 1.

## Results

| quant | weights / acts | ppl ↓ | top1-agree vs bf16 ↑ | nll-gap vs bf16 ↓ | gsm8k (n=150) ↑ | serve VRAM |
|---|---|---|---|---|---|---|
| **bf16** (reference) | 16 / 16 | 12.7010 | — | — | — *(CPU ref)* | ~29.6 GB (won't fit to serve) |
| **fp8** | 8 / 8 (fp) | 12.6966 | 0.968 | 0.062 | 0.960 (144/150) | ~15.2 GB |
| **w8a8** (our int8 kernel) | int8 / int8-dyn | 13.0839 | 0.881 | 0.250 | 0.953 (143/150) | ~15.2 GB |
| **w4a8** | int4 / int8-dyn | 14.1943 | 0.822 | 0.420 | 0.927 (139/150) | ~9.3 GB |

- **top1-agree** = fraction of corpus tokens where the quant's greedy argmax == bf16's argmax (1.0 = identical decisions).
- **nll-gap** = mean |Δ logprob| of the actual token vs bf16. **ppl** on a fixed 10-passage prose+code corpus (1063 tokens).
- gsm8k: thinking **off**, greedy, `#### <n>` exact-match, first 150 test items (paired across quants).

## Read

- **FP8 is effectively lossless** — ppl ≈ bf16 (12.697 vs 12.701), 96.8% token agreement, gsm8k 96.0%.
  It is the practical high-precision anchor on this hardware (bf16 itself won't fit one card to serve).
- **W8A8 (our INT8 kernel) is the sweet spot** — only +0.38 ppl (~3%), 88% token agreement, gsm8k within
  noise of fp8 (95.3 vs 96.0, n=150). Near-fp8 quality at the same VRAM, and it lights the B70's INT8
  systolic fastpath → **this is where kernel-optimization effort pays off most.**
- **W4A8 trades real quality for footprint** — +1.49 ppl (14.19), 82% agreement, gsm8k 92.7% (−3.3 pts
  vs fp8). The int4 weights are the cost; its only edge is VRAM (9.3 GB). Worth it only when memory-bound.
- The divergence canary (ppl/agreement) and the task metric (gsm8k) **agree on the ordering**
  fp8 > w8a8 > w4a8 — the cheap deterministic Tier-0 signal predicts the expensive task outcome.

## Caveats (don't over-read)

- **gsm8k n=150** → ~±2.5% per cell; the fp8↔w8a8 gsm8k gap (0.7 pt) is within noise. The **ppl/agreement
  deltas are tight** (deterministic, 1063 paired tokens) and carry the real signal between fp8 and w8a8.
- **No formal noise floor run yet** (bf16-vs-bf16). The fp8≈bf16 ppl (and fp8's marginally *lower* ppl)
  is itself a sanity check that sub-1% ppl moves are at the noise level.
- bf16 reference scored **offline on CPU** (won't serve on one card); tokenization verified identical to
  vLLM /tokenize (0/10 passages misaligned), so agreement numbers are valid.
- 14B-class; **may not transfer to 27B** (re-run when card #2 lands). thinking-off (more quant-sensitive,
  bounded, deterministic) — a thinking-on pass would score higher and compress the gaps.

## Repro

```
# per quant: serve (see evals/configs/models.yaml), then
python evals/orchestrator/run_evals.py --endpoint http://192.168.10.5:18080/v1 \
    --model <served-id> --quant <label> --tiers 0,2 --limit 150
# bf16 anchor (offline CPU): scripts/55_tier0_reference_cpu.sh
# divergence matrix: python evals/orchestrator/tier0_matrix.py evals/results bf16
```
