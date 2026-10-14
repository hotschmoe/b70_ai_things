# Qwen3.8-27B / Qwen3.6-27B: official FP8 vs 4-bit quality review

Date: 2026-09-05
Scope: quality only (not speed). Models: Qwen3.8-27B (hybrid GDN) and
Qwen3.6-27B. Quants: official block FP8 (W8A16), W4A16, GPTQ INT4, AutoRound
INT4, GGUF Q4_K / Q4_K_M / Q4_0, and NVFP4 where quality evidence exists.
This is a literature/evidence review. No GPUs were touched.

Reading rule: HumanEval+ numbers below are local thinking-off greedy
sandbox (EvalPlus 164, seed 1234 unless noted). Publisher Terminal-Bench
numbers are thinking-on agent scores on a different harness. Do not mix
them. Local Terminal-Bench 3 pilots all scored 0.0 from runtime/policy
failure, not from a completed quality ranking.

---

## Verdict

Official Qwen3.8-27B block FP8 remains the quality-conservative default
for Terminal-Bench / agent work on vLLM-XPU.

That is not because a matched Terminal-Bench 3 campaign ranked FP8 above
4-bit. No such campaign exists. It is because:

1. The publisher states official FP8 is fine-grained block-128 FP8 whose
   "performance metrics are nearly identical to those of the original
   model" (Qwen/Qwen3.8-27B-FP8 and Qwen/Qwen3.6-27B-FP8 cards).
2. Steve's Qwen3.8 official-FP8 vLLM/XPU gate matches the Q8_0 oracle on
   seven exact canaries, including Python `sum(i * i for i in range(4))`
   = 14, plus 8-run repeat stability and a 3,829-token needle. Later
   local Neural FP8 work added 12/12 token-array exactness, 32/32 and
   64/64 concurrent exact-answer canaries, and a 30,037-token semantic
   needle. That is the strongest Qwen3.8 quality envelope on this box.
3. GPTQ INT4 of Qwen3.8 is quality-rejected as a no-loss default: the
   same Python canary answers 30 instead of 14. Q8_0 and Q4_K_M answer
   14. MTP and FP8 KV do not recover the miss.
4. Local Qwen3.8 4-bit coding proxies (Q4_K_M, AutoRound W4A16) are
   close on thinking-off HumanEval+, but they have no completed
   Terminal-Bench score, and AutoRound's fast MTP lane is not
   token-deterministic (21-22/25).
5. Qwen3.8 NVFP4 is behind Q4_K_M and W8A8 on the local HumanEval+
   board. Do not treat NVFP4 as a quality upgrade over official FP8.
6. Qwen3.6 NVFP4 is the local HumanEval+ leader (0.988/0.945), but
   publisher agent benches put Qwen3.8-27B ahead of Qwen3.6-27B
   (Terminal Bench 2.1 Terminus 73.0 vs 63.4). Local TB3 never scored
   either model. For agent work, prefer Qwen3.8 official FP8 over
   "3.6 NVFP4 won HumanEval+".

Caveats that must travel with the verdict:

- There is no local HumanEval+ for Qwen3.8 official FP8 or BF16, and
  none for Qwen3.6 official FP8. The FP8 default is canary/oracle plus
  publisher identity, not a matched 164-problem coding number.
- Local TB3-local-70 never produced a nonzero reward. CAMPAIGN_RELAUNCH
  forbids ranking quality from the four zero-score pilots.
- Thinking-off HumanEval+ did not show the publisher's Qwen3.8 agent
  jump vs Qwen3.6. Use HE+ as a coding-quant ranking, not as a TB proxy.
- Q8_0 GGUF is Steve's llama.cpp quality oracle. Q4_K_M is the best
  locally measured 4-bit coding proxy for Qwen3.8. Neither is a TB
  score.

Do not change serving from this review.

---

## Table: quant x model x quality x determinism x TB relevance

Legend:

- HE+ = local HumanEval+ pass@1 base/plus, thinking-off greedy, 164
  problems, EvalPlus sandbox 0.3.1.
- Canary-14 = Steve exact Python result `sum(i*i for i in range(4))`.
  Correct is 14.
- Det = repeat/token-array exactness under greedy/temp-0.
- TB = Terminal-Bench relevance. "none scored" means no completed
  quality ranking.

| Model | Quant | Quality evidence | Determinism | TB / agent relevance |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B | Official block FP8 W8A16 (`Qwen/Qwen3.8-27B-FP8`) | Publisher: nearly identical to BF16. No local HE+. Steve: 7/7 exact canaries match Q8_0, canary-14=14, 3829-token needle pass. Local Neural: 12/12 token arrays, 32/32 and 64/64 concurrent exact-answer, 30037-token semantic needle identical across lifetimes. Independent PPL (ithelp, llama.cpp BF16 vs vLLM FP8): EN 6.9529 -> 6.9712 (+0.26%), ZH 11.1337 -> 11.1842 (+0.45%). Huginn unofficial FP8 WikiText-2 +0.263% vs official BF16 is a different checkpoint; do not treat as official. | Strong within-lifetime: 12/12, canaries, needle. Eager vs graph not universally byte-identical on long free-form. MTP1 target-verified 12/12 vs MTP0 in Steve R54/R187 packets. | Quality-conservative vLLM-XPU default for agent. No local TB3 score. Publisher TB 2.1 73.0 is the BF16/original family, not a measured FP8-vs-INT4 TB delta. F05a 32K pass is synthetic context, not model-quality. |
| Qwen3.8-27B | BF16 reference | Publisher original. No local HE+. Weights retained as quality ceiling (`evals/configs/models.yaml`). | Not locally scored as a served identity in the retained HE+ set. | Publisher agent numbers (TB 2.1 73.0, SWE-Pro 61.7, LCB v6 90.3) attach here. Official FP8 is the claimed near-lossless stand-in. |
| Qwen3.8-27B | GGUF Q8_0 | Steve quality oracle for the 14-canary and needle. No local HE+ in the retained 3.8 set. | Q8 TP2 target-only passed the semantic gate; canary-14=14. | Quality-conservative llama.cpp control, not a TB score. |
| Qwen3.8-27B | GGUF Q4_K_M (ggml-org stock) | Local HE+ **0.970 / 0.927** (20260817T231333Z). Best 3.8 HE+ on the box. 5 base fails (32,95,130,140,145). Steve: canary-14=14, needle pass, same canary outputs as Q8. Independent PPL vs BF16: EN +0.50%, ZH +0.84% (ithelp). | MTP vs no-MTP 6/7 exact on obliterated sibling; shared modular-arithmetic miss (0 vs 71) recorded as model quality, not speculation. UD-Q4_K_XL matched 6/7 canaries and text hashes including that miss. | Best local 4-bit coding proxy. No TB3 score. Selected 2026-08-23 as quality-first llama.cpp daily driver for coding, not as a TB winner. |
| Qwen3.8-27B | GGUF UD-Q4_K_XL | No separate HE+. Same 6/7 canary result and identical hashes vs Q4_K_M, including the modular miss. | Byte-identical to Q4_K_M on the 8-prompt corpus in the matched campaign. | Treat as Q4_K_M-class quality, not independently TB-qualified. HE+ still required before any shelf claim. |
| Qwen3.8-27B | GGUF Q4_0 | No local HE+ for Qwen3.8 Q4_0. Community PPL/KLD ladders put Q4_0 below Q4_K_M (tooltd Qwen3.8 table: Q4_0-AutoRound-Code KLD 0.0266 vs Q4_K_M 0.0155). | Not locally gated. | Not a quality default. Do not substitute Q4_0 for Q4_K_M. |
| Qwen3.8-27B | AutoRound W4A16 (`devan-carlin/Qwen3.8-27B-int4-AutoRound`, local HE+ on W4A16-autoround-mtp5) | Local HE+ **0.963 / 0.915** (20260819T063444Z). One base / two plus behind Q4_K_M. Author card: "quality parity on math, logic, algorithm, and code prompts" with no numbered bench. Steve TP1 nightly: objective battery pass, canary-14=14, 8K needle. | Fast MTP5 lane not promotable: 21-22/25 pairwise, target A/B 24/25, sealed TP1 MTP5 2/4. Cross-boot graph pair 19/25. FP8 KV at TP1 changed 22/25 outputs. Graph+MTP corrupted 0/25. | "Good enough" 4-bit for thinking-off coding if you accept HE+ 0.915 plus. Rejected as quality-default agent/TB route until token-stable and TB-scored. |
| Qwen3.8-27B | GPTQ INT4 G128 (`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`) | **Rejected.** Canary-14 actual **30**, expected 14. Needle still passed. No local HE+. TB3 GPTQ pilot scored 0.0 after PIECEWISE abort before edit; that is infrastructure, not a quality number. Retained GPTQ TB/exactness/speed used FP16 KV despite BF16 labels. | MTP4 matched the bad GPTQ target exactly (still 30). Repeat of a wrong answer is not quality. | Not the no-quality-loss default. Experimental performance only. Do not use for TB ranking. |
| Qwen3.8-27B | NVFP4 Inferact ModelOpt | Local HE+ **0.939 / 0.915** (20260816T005900Z). Behind 3.6 NVFP4 by 8 base / 5 plus. Journal: not a 3.8 quality win on this bench. | 18/18 coherence in the speed campaign; HE+ is the quality row. | Not a DD vs 3.6 NVFP4. TB3 NVFP4 pilot 0.0 after graph abort at 19328 tokens; not a quality score. |
| Qwen3.8-27B | NVFP4 RadixArk mixed | Local HE+ **0.933 / 0.890** (20260817T081950Z). Worst 3.8 HE+ in the matched set. | Coherent on the speed campaign; not HE+ leader. | Not a quality upgrade. Mixed NVFP4 MLP + FP8 attn/GDN, not a pure 4-bit. |
| Qwen3.8-27B | W8A8 GPTQ (8-bit, not 4-bit; nearby control) | Local HE+ **0.957 / 0.927** (20260818T064331Z). Plus ties Q4_K_M; base 1.3 pts under. | Target-only 32/32 concurrent canary passed. MTP1 drafted 6076, accepted 0, and failed arithmetic 54 vs 60 in all 8 rounds. | 8-bit coding proxy, not 4-bit. TB3 W8A8 FULL arm 0.0 after kernel fault ~17K; reclaim500 survived but timed out. Not a TB quality ranking. |
| Qwen3.6-27B | Official block FP8 (`Qwen/Qwen3.6-27B-FP8`) | Publisher: nearly identical to original. No local HE+. Historical B70 path: official FP8 on llm-scaler failed (DeltaNet+FP8 init), so this box never produced a 3.6 official-FP8 coding score. Unsloth: Qwen3.6-27B NVFP4/FP8/BF16 MMLU-Pro/GPQA/AIME 2025 are within a point (FP8 MMLU-Pro 86.11 vs BF16 85.96; GPQA 86.87 vs 88.13). | Not locally measured as a served 3.6 official-FP8 identity. | Publisher TB 2.0 59.3 (3.6 family). No local TB3 score. Cannot claim local FP8 vs INT4 for 3.6 official FP8. |
| Qwen3.6-27B | NVFP4 NVIDIA ModelOpt mixed | Local HE+ **0.988 / 0.945** MTP3 fused (20260704T172906Z); **0.976 / 0.945** embed-INT8 MTP5 (20260721T223914Z); **0.976 / 0.939** native E4M3 M<=8 (20260722T170717Z). Local coding leader. Needle@93k 4/4 twice. Gate 18/18 + 36/36. Uncalibrated FP8 KV caused late repetition; calibrated FP8 KV was later quality-neutral on HE+. | Graph-mode HE+ had known greedy nondeterminism (one prompt, five different outputs). 139/164 solutions byte-identical vs folded-scale twin. | Strong coding/long-ctx 3.6 artifact. Publisher TB 2.1 63.4 is the 3.6 family, below 3.8 73.0. Do not promote 3.6 NVFP4 over 3.8 official FP8 for agent work on publisher TB. |
| Qwen3.6-27B | AutoRound W4A16 (Lorbus/Intel class) | Local HE+ **0.963 / 0.927** on vLLM 0.23 eager (2026-06-20). Same-stack sglang later 0.921/0.896, depressed by max_tokens=2048 truncation (9/13 base fails). Intel HF card has no numbered accuracy table. Kaitchup: Intel INT4 (keep in_proj_a/b 16-bit) stays close to BF16 on thinking-on benches; NVFP4 that quantizes linear attention is the loser. | Steve 3.6 INT4 lane closed 2026-08-18: complete-token parity vs a differently configured reference unsatisfiable at fp16; XPU batch invariance dead behind `is_cuda_alike()`. | Good 4-bit coding proxy for 3.6. Not a TB score. Not a reason to drop 3.8 official FP8 for agents. |
| Qwen3.6-27B | W4A8 sqgptq | Local HE+ **0.963 / 0.927** (20260721T195422Z). Does not displace NVFP4. Earlier sglang W4A8 graph 0.921/0.896 (truncation). | Coherent on the 0.25.1 captured+MTP3 run. | Secondary 4-bit. Not TB-scored. |
| Qwen3.6-27B | W8A8 fused (8-bit control) | Local HE+ **0.970 / 0.933** (20260628T233713Z). Face-value 1-2 plus behind NVFP4; harness rule 1-2 pt is noise. | 18/18 mixed-load. MTP+graph later hit the same NEO abort class as NVFP4, later. | 8-bit research path. Not 4-bit. Not TB-scored. |
| Qwen3.6-27B | GGUF Q4_K_M / Q4_0 | No local HE+ in the retained 3.6 set. Community PPL (bobaburger): Q8_0 6.568, Q4_K_M 6.673, Q4_0 not the top-4-bit. Unsloth 3.6 4-bit KLD 0.0227 vs 8-bit 0.0028. | Q4_0 was a DFlash target-only control in Steve's closed campaign, not a HE+ gate. | Not locally ranked vs official FP8. Q4_K_M is the usual GGUF 4-bit pick; Q4_0 is not quality-first. |
| Qwen3.6-27B | GPTQ INT4 (official/community) | `Qwen/Qwen3.6-27B-GPTQ-Int4` is listed in vLLM recipes; this review did not find a local HE+ or canary-14 row. Do not infer 3.8 GPTQ's 30-fail onto 3.6 without a measurement. | Missing. | Missing. |

Harness note: 1-2 HumanEval+ points is noise (~1-3 problems). Template/stack
moved the same 3.6 int4 weights from 0.963/0.927 (vLLM) to 0.921/0.896
(sglang, truncation). Compare only matched thinking-off greedy sandbox
rows.

---

## What 4-bit is "good enough" vs what is rejected

### Good enough as a 4-bit coding proxy (not a TB winner)

- **Qwen3.8 Q4_K_M GGUF:** HE+ 0.970/0.927, canary-14=14, needle pass,
  Q8-matched short canaries. Small PPL hit vs BF16. Best 3.8 4-bit on
  this box. Shared modular miss (0 vs 71) is a model miss, not a quant
  unique fail. LiveCodeBench/agent comparison unmeasured.
- **Qwen3.8 AutoRound W4A16:** HE+ 0.963/0.915. Canary-14=14 on Steve
  TP1 nightly. Author claims answer parity, no numbered suite. Accept
  as "close 4-bit" for thinking-off code. Do not accept the 101 tok/s
  MTP5 lane as quality-stable.
- **Qwen3.6 AutoRound W4A16:** HE+ 0.963/0.927 on the original vLLM
  row. Historically the single-card quality leader until 3.6 NVFP4.
  Keep in_proj_a/b in 16-bit (Intel/Lorbus/devan-carlin recipe).
- **Qwen3.6 W4A8 sqgptq:** HE+ 0.963/0.927, equal to AutoRound on that
  run, below NVFP4.
- **Qwen3.8 UD-Q4_K_XL:** canary-identical to Q4_K_M on 6/7. HE+ not
  rerun. Treat as Q4_K_M-class pending HE+.

### Rejected as a quality default

- **Qwen3.8 GPTQ INT4 G128:** canary-14 answers 30. One reproducible
  semantic fail is enough to reject a no-loss default. Needle pass does
  not cancel it. TB3 0.0 is a graph abort, not a second quality number.
  FP16-KV contamination further invalidates retained GPTQ "BF16" labels.
- **Qwen3.8 NVFP4 (Inferact and RadixArk):** HE+ 0.939/0.915 and
  0.933/0.890. Behind Q4_K_M, W8A8, and 3.6 NVFP4. Journal verdict: do
  not swap the 3.6 coding DD on this bench. Not "4-bit good enough" vs
  official FP8 for agent work.
- **Qwen3.8 AutoRound MTP5 as a promoted exact lane:** 21-22/25, FP8
  KV 3/25, graph+MTP 0/25. Speed observations with an output-changing
  greedy margin are withdrawn as quality evidence.
- **Q4_0 as a quality pick:** no 3.8 HE+; community KLD/PPL worse than
  Q4_K_M. Do not use Q4_0 to argue 4-bit parity with FP8.
- **Uncalibrated FP8 KV on 3.6 NVFP4:** late fluent repetition. Calibrated
  scales later looked quality-neutral on HE+. KV dtype is its own
  identity; do not fold it into weight-quant quality.
- **Qwen3.8 Flash-Next 125B-A6B official FP8:** out of 27B scope, but
  the same canary-14 miss (answers 30) is inherited on that MoE. Do not
  confuse it with Qwen3.8-27B official FP8, which answers 14.

### Nearby 8-bit, not 4-bit

W8A8 GPTQ is not 4-bit. On 3.8 it ties Q4_K_M on plus (0.927) and loses
a little on base. On 3.6 it is 0.970/0.933, practically tied with NVFP4
under the 1-2 pt rule. MTP can still break a simple arithmetic canary
(54 vs 60) even when target-only is clean. Treat MTP as a separate
quality identity.

---

## Open gaps

1. **No local HumanEval+ for Qwen3.8 official FP8 or BF16.** The FP8
   quality default is canary/oracle + publisher identity. A matched
   164-problem thinking-off row would close the coding gap vs Q4_K_M
   0.970/0.927.
2. **No local HumanEval+ for Qwen3.6 official FP8.** Official 3.6 FP8
   never served cleanly on the historical llm-scaler path. NVIDIA NVFP4
   is the measured 3.6 high-quality compressed artifact, not official
   FP8.
3. **No completed Terminal-Bench 3 quality ranking for any quant.**
   Four matched bun-sourcemap-leak pilots scored 0.0 from timeout,
   graph abort, or kernel fault. CAMPAIGN_RELAUNCH: do not rank quality
   from those jobs. Local-70 vs official-74 boundary still applies
   (four H100-only tasks).
4. **No matched thinking-on agent eval of FP8 vs Q4_K_M vs AutoRound.**
   Publisher TB 2.1 73.0 vs 63.4 is 3.8 vs 3.6 at original precision,
   not FP8 vs INT4. HE+ thinking-off hid the 3.8 agent jump.
5. **No local perplexity corpus ranking** for 3.8 official FP8 vs
   Q4_K_M vs AutoRound vs GPTQ on the retained evals harness. External
   PPL (ithelp, Huginn, tooltd, bobaburger) is secondary and not
   same-stack.
6. **No local GSM8K / MMLU-Pro / LiveCodeBench** for 3.8 quants. 14B
   GSM8K was insensitive to W8A8; do not assume 27B agent work is.
7. **Qwen3.6 GPTQ INT4:** recipe exists; no local canary-14 or HE+.
8. **Needle at native 256K** for 3.8 official FP8 vs Q4_K_M: 3.8 FP8
   has 30K semantic needle and 32K synthetic; 3.6 NVFP4 has 93k and
   190k retrieval. Not a matched pair.
9. **Tool/agent coherence beyond canaries:** Ornith W8A8 had a real
   OpenAI tool-parse pass; Qwen TB3 pilots never completed an edit on
   bun-sourcemap-leak under the failed runtimes. Missing for FP8 vs
   INT4.
10. **devan-carlin AutoRound card** asserts quality parity without
    numbered benches. Local HE+ 0.963/0.915 is the numbered row; the
    card is not independent confirmation.

---

## Sources (trust order)

### 1. Local repo

- This file: `docs/20260905_qwen38_fp8_vs_int4_quality_review.md`
- `JOURNAL.md` 2026-08-23l (3.8 Q4_K_M vs AutoRound vs NVFP4 HE+ pick);
  2026-08-23e (modular miss); 2026-08-28p / 2026-08-29c (TB3 GPTQ FP16
  KV invalidation).
- `archive/to-delete-20260826/JOURNAL.before-trim.md` 2026-06-20
  (3.6 AutoRound 0.963/0.927), 2026-06-29 (3.6 W8A8 0.970/0.933),
  2026-07-04 (3.6 NVFP4 0.988/0.945), 2026-07-21 (W4A8 0.963/0.927;
  embed-INT8 0.976/0.945), 2026-07-22 (native E4M3 0.976/0.939),
  2026-08-16 (Inferact 0.939/0.915; 3.8 HE+ does not show publisher
  jump), 2026-08-17 (RadixArk 0.933/0.890; Q4_K_M 0.970/0.927),
  2026-08-18i (W8A8 0.957/0.927).
- EvalPlus summaries:
  `archive/to-delete-20260826/repo/evals-results/*/summary.json`
  for the rows cited above.
- `evals/terminalbench/README.md` and `CAMPAIGN_RELAUNCH.md` (TB3
  0.0 pilots; do not rank quality; GPTQ FP16 KV; thinking-off adapter
  bug).
- `docs/quant_methods.md` Table C (method index; 14B numbers are not
  27B).
- `docs/20260830_f05a_qwen38_fp8_32k_long_context_pass.md` (synthetic
  32K, not model quality).
- `docs/20260830_qwen38_fp8_neural_full_graph.md` (30K needle, 64/64
  concurrent).
- `docs/20260831_qwen38_w8a8_native_int8_result.md` (32/32 target-only;
  MTP arithmetic 54 vs 60).
- `evals/configs/models.yaml` (BF16 as quality ceiling, no shelf).
- `models/manifest.yaml` (live identities: 3.8 BF16/FP8/NVFP4/W8A8/
  GPTQ-INT4; 3.6 NVFP4 only).

### 2. neural.download and steveseguin/b70-optimization-lab

Local mirror: `/mnt/vm_8tb/github/xe2x2/refs/b70-optimization-lab/`

- `community/sergiiob-qwen38-27b-vllm-xpu/validation/2026-08-16-quality-kv-dtype-decision.md`
  and `.../2026-08-16-quality-kv-dtype-summary.json` (GPTQ 30 vs Q8/Q4
  14).
- `repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/README.md` and
  `experiments/qwen38-27b-b70/notes/2026-08-16-official-fp8-vllm-graph-tp2.md`
  (official FP8 7/7, canary 14, needle, Q8 oracle).
- `repro/qwen38-27b-autoround-int4-b70/README.md` (devan-carlin pin
  `bce40cacab0a4535b92fb3d57615c2bea9adf3d1`; MTP5 21-22/25).
- `experiments/qwen38-27b-b70/notes/2026-08-22-qwen38-tp1-vllm-nightly-bringup-finding.md`
  (AutoRound canary 14, 8K needle; FP8 KV 3/25; graph+MTP 0/25).
- `scripts/qwen38-text-quality-suite.py` (canary-14 expected 14).
- Public: https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html
- Public: https://github.com/steveseguin/b70-optimization-lab

### 3. Hugging Face cards / community quants

- https://huggingface.co/Qwen/Qwen3.8-27B
  (TB 2.1 73.0, SWE-Pro 61.7, LCB v6 90.3 vs 3.6 63.4 / 53.5 / 83.9).
- https://huggingface.co/Qwen/Qwen3.8-27B-FP8
  (block-128 FP8, "nearly identical" to original).
- https://huggingface.co/Qwen/Qwen3.6-27B-FP8
  (same FP8 claim; TB 2.0 59.3 on the 3.6 family table).
- https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound
  (W4A16 g128, keep in_proj_a/b 16-bit; qualitative parity claim).
- https://huggingface.co/Intel/Qwen3.6-27B-int4-AutoRound
  (recipe card, no numbered accuracy table).
- https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound
  (same recipe class; local HE+ used this family).
- https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
  (the GPTQ that failed canary-14).
- Pilcothink MixedInt4 AutoRound MMLU/GSM8K table is a different mixed
  checkpoint, not the local W4A16-autoround-mtp5 row.

### 4. Papers/blogs about these exact models

- Unsloth Qwen3.6-27B NVFP4 vs FP8 vs BF16: MMLU-Pro/GPQA/AIME 2025
  within ~1 point (https://unsloth.ai/docs/models/qwen3.6). Qwen3.8
  NVFP4 KLD/top-1 vs BF16 92-97% recovery
  (https://unsloth.ai/docs/models/qwen3.8).
- Kaitchup, "Qwen3.6 27B Quantization: FP8 vs INT4 vs NVFP4"
  (2026-05-12): Intel INT4 (16-bit in_proj_a/b) close to BF16;
  NVFP4 that quantizes linear attention is the clear loser.
- ithelp Day 14 (2026-09-05): Qwen3.8-27B PPL ladder BF16 / q8_0 /
  official FP8 / q4_k_m (cited above). Different host/runtime than B70.
- Huginn DGX WikiText-2: unofficial FP8 +0.263% vs official BF16.
  Not the Qwen official FP8 artifact.
- bobaburger r/LocalLLaMA Qwen3.6 GGUF PPL/KLD ladder (2026-05-29).
- jcartu Blackwell HumanEval 79.3% for 3.6 FP8+MTP3 is a different
  harness (130/164, not EvalPlus plus) on NVIDIA; do not join it to
  local 0.988/0.945.

### 5. xe2x2

`/mnt/vm_8tb/github/xe2x2` kernel campaign notes treat FP8 W8A16 as
the dense incumbent control and INT4 as a capacity/XMX research target.
No additional 27B HumanEval+ or TB quality numbers beyond the mirrored
b70-optimization-lab tree.

---

## Bottom line, restated

For Terminal-Bench / agent work on this stack, keep official Qwen3.8-27B
block FP8 as the quality default until a completed TB3-local-70 (or
official-74) run with a healthy endpoint produces a verifier ranking.
Q4_K_M is the best measured 4-bit coding stand-in for Qwen3.8. AutoRound
W4A16 is close on HE+ if you ignore token-exactness. GPTQ INT4 is
rejected. Qwen3.6 NVFP4 is the local coding champion and is not a
substitute for Qwen3.8 agent quality. Speed is out of scope.
