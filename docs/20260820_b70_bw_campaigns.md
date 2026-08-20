# Five one-lever campaigns + Intel-quant / DSpark / 4x-MoE notes

**Created:** 2026-08-20
**Operator:** spend an entire campaign on *each* A-E lever. Do not mix
levers in one loop. Accuracy recovery is later; A-E may ship a faster
but slightly worse checkpoint if the BW/%-of-roof move is real.

Standing holds at write: Ornith GRAPH 34.9, W8A8 k1bar **31.9**,
W1 GPTQ-Int4+draft 65.08, llama.cpp Q4_K_M 43.8, Pliny Q8 32.03.
W8A8-gptq HE+ 0.957/0.927. Hot decode bytes on that file: 30.2 GiB
(58% INT8, 42% BF16). No-spec TP=2 18.45 = 49% of 37.5 DRAM roof.

Ledger: append `## LOOP N` to `docs/20260820_b70_bw_campaigns_loops.md`
when a campaign starts (create that file on LOOP 1). Dead-ends go in
`docs/20260820_b70_bw_campaigns_deadends.md` (create on first packet).

Do not start DD. P2PACCESS=0. One GPU pick at a time.

---

## Campaign order (serial)

| id | campaign | why this order |
|---|---|---|
| **A** | INT8 the GDN `linear_attn` projections | Biggest byte cut (10.4 GiB BF16). Changes the roof. |
| **B** | Kill/fuse 101 us per-token INT8 quant | Layer tax already measured; fusedq source exists. |
| **C** | Launch/replay (XPUGraph, MRV2) | Post-capture is launch-bound. No quant change. |
| **D** | Small-M INT8 VNNI16 tile | Kernel-only; maybe +10-20% INT8 GEMV, not 2x. |
| **E** | DSpark accept 2.45 -> ~3.3 | Does not raise GB/s; sells more tokens per walk. |

F (Intel-shaped W4A8 / XMX) and G (4x MoE shopping) are **after** A-E
unless the operator jumps. W4A4 integer kernels stay note-taking until
W4A8 M=1 is not a trap (quant_methods.md Table A).

Each campaign: one hypothesis, G1, one metric (`bench_code` c1 and a
kernel BW probe where relevant), HE+ only if weights or sampling change.

---

## A -- INT8 GDN projections

**Hypothesis:** 10.36 GiB of decode-hot `linear_attn` is BF16 because
GPTQ `ignore: re:.*linear_attn.*`. INT8 those Linears (keep conv1d,
A_log, dt_bias, norms in BF16/FP32). Hot bytes 30.2 -> ~25 GiB. Roof
37.5 -> ~45 tok/s no-spec at 100% BW; realistic GEMV ~23.

**Do:** new GPTQ (or SQ+GPTQ) on `models/files/qwen3.8-27b/w8a8-gptq`
with linear_attn Linear **not** ignored. Copy to a new number, do not
rewrite 150. Serve same k1bar recipe. Census tensors before claiming
INT8. HE+ vs 0.957/0.927.

**Done when:** tensor census shows GDN `in_proj_*` / `out_proj` as I8
and c1 is measured. Fail closed if G1 bangs or HE+ drops >3 plus pts
without a recovery plan.

**Not this campaign:** MTP INT8, visual INT8, embed INT8, lm_head
(lm_head is campaign-adjacent; A/B only if GDN INT8 lands).

---

## B -- Fuse the 101 us per-token quant

**Hypothesis:** `dynamic_per_token_int8_quant` is 101 us on [1,17408],
35% of `down_proj` layer time (docs/20260703_faster_dd_plan.md).
`int8_gemm_w8a8_fusedq` already exists (`kernels/README.md`,
`research/w8a8/FUSEDQ_NOTES.md`) and is not the default e2e path.

**Do:** enable fusedq on the live 3.8 W8A8 serve (GRAPH=1). Isolated
layer time + e2e c1 vs k1bar 31.9 / MTP-off 18.45. Do not retune GPTQ.

**Done when:** layer probe shows the 101 us gone or fused into the
matmul, and e2e is measured. NO-GO if fusedq is a no-op at M=1.

---

## C -- Launch / replay (XPUGraph, MRV2)

**Hypothesis:** after PIECEWISE capture, the step is launch/Python
bound, not GEMM (GEMMs already 88-100% of 581 GB/s on 3.6 W8A8).
MRV2 async runner and XPUGraph replay are the remaining host tax.

**Do:** A/B `VLLM_USE_V2_MODEL_RUNNER` and XPUGraph replay knobs on
the same W8A8-gptq GRAPH=1 serve. No weight change. Watch c1 and
tpot, not just kernel BW.

**Done when:** one A/B with launch stats. Prior note: MRV2 overlap
may not materialize on XPU -- still run it.

---

## D -- Small-M INT8 VNNI16 tile

**Hypothesis:** M=1 INT8 GEMV is 361/608 GB/s (59%). BF16 GEMV is 76%.
Intel arXiv:2508.06753: VNNI16 prepack + rectangular subgroup tiles
that reuse each dequant weight ~8x. Target 59% -> ~70%+, not 100%.

**Do:** isolated GEMV microbench on real 3.8 shapes (MLP N=17408/5120,
attn N=1024/6144, GDN proj). Then e2e only if isolated >=1.10x.

**Done when:** isolated number vs 361 GB/s. Closed 2026-07-21 as a
2-3x e2e lever; this campaign is the remaining 10-20%, not a retry
of "write a 3x GEMV".

**Intel-shaped sizes to prefer (DPAS/VNNI):**
- M tile 8 or 16 (decode M=1 cannot fill DPAS; this campaign is
  about making M=1 less bad, not pretending M=16).
- K % 32 == 0 (5120, 17408 both ok).
- N % 16 == 0 (watch GDN/attn 24-head / N=24 traps; Marlin-class
  tiles wanted N%64).
- Prefill and DSpark verify (M=2..8) are where XMX actually pays.
  Campaign D measures decode M=1 anyway so we know the residual.

---

## E -- DSpark accept 2.45 -> ~3.3

**Hypothesis:** k1bar 31.9 with greedy accept 2.45 is an FP8-trained
drafter on an INT8 target. Matched hidden-state train should raise
accept toward RadixArk-on-FP8 (~3.3). Tokens per weight-walk, not GB/s.

**Do:** the existing campaign `docs/20260818_qwen38_w8a8_dspark_campaign.md`
P1 train on W8A8 hiddens. Do not start until A-D have a LOOP entry
or the operator jumps. Gate: 10-sample overfit then HE+/c1.

**Done when:** accept_len greedy >= 3.0 on the code harness with G1
GO, or a dead-end packet that matched training did not move accept.

---

## F -- Steal NVIDIA's *idea*, not their NVFP4 file (notes, not a loop yet)

NVIDIA on 5090: NVFP4 (4-bit weights, often 4-bit or FP8 acts) + cheap
verify on tensor cores + DSpark. Xe2 has **no FP4/FP8 XMX**. Stealing
the file (Unsloth/RadixArk NVFP4) gives us `nvfp4_gemm_w4a16` (E2M1
decompress into a W4A16 matmul), which we already ran. Stealing the
*shape* means: fewer bytes + INT act GEMV that DPAS can eat.

### What 3.8 W4A16 / W4A4 exists on HF

**W4A16 (int4 weights, 16-bit acts) -- several, this is W4A16 not XMX:**
- `devan-carlin/Qwen3.8-27B-int4-AutoRound` -- **on disk**
  `models/files/qwen3.8-27b/int4-autoround`. Steve's 101.9 stack.
  GRAPH=0 we got 13.4; GRAPH=1 hung (D16).
- `Frozenlock/Qwen3.8-27B-int4-AutoRound` -- MTP quantized, 18 GB.
- `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`
- `soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ` (g128, ultrachat calib)
- `cyankiwi/Qwen3.8-27B-AWQ-INT4`, `dbirks/Qwen3.8-27B-W4A16-AutoRound`,
  `philbert440/Qwen3.8-27B-W4A16-AWQ`
- `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` -- mixed 4-bit + MTP

**W4A4 integer (int4 w x int4 a):** no quality 3.8 checkpoint we trust.
**NVFP4 labeled W4A4** (sakamakismile, Unsloth, Inferact-deleted):
NVIDIA E2M1, not INT4 DPAS. On B70 this is the W4A16 decompress path.

### Which of those lights XMX?

| Scheme | Kernel we have | XMX DPAS? |
|---|---|---|
| W8A8 | `int8_gemm_w8a8` / `w8a16` | **YES** (s8 x s8 or s8 x f16) |
| W4A8 | `int4_gemm_w4a8` (upstream vllm-xpu-kernels) | **YES if** unpack int4->int8 in register then DPAS s8xs8 |
| W4A16 | `int4_gemm_w4a16` | **NO.** Dequant to BF16, then BF16 GEMM. Bytes win only. |
| NVFP4 | `nvfp4_gemm_w4a16` | **NO.** Same decompress-to-matmul family. |
| W4A4 int | none | **NO** until an int4xint4 GEMM exists |

The "funky W4A16 -> int8xint8" trick **is W4A8**: keep 4-bit storage,
unpack to int8 in registers, DPAS. Do not unpack to INT8 and *store*
INT8 (that is W8A8, 2x the bytes). Do not leave acts in BF16 (W4A16).

W4A16 is still useful as a **capacity** and "walk fewer bytes at 76%
BF16-GEMV efficiency" experiment (D16 hang is the serve blocker, not
the math). It will not raise % of INT8 XMX utilization.

**BW-first (accuracy later) target stack:** W4A8 GPTQ/SQ on 3.8 MLP+attn
(+ GDN Linears if A has a recipe), group 128, ignore conv/mtp/visual
until census. Measure isolated GEMV GB/s and e2e c1. HE+ after the
speed number exists. Rotation (QServe/SpinQuant) is the recovery
track in `docs/quant_methods.md`, not the first F loop.

---

## G -- 4x B70 / 128 GiB MoE (shopping, not a loop)

Qwen3.8-27B Terminal-Bench 2.1 = **73.0**. No released Qwen3.8 35B-A3B.

| model | total / active | TB (card, mix of 2.0/2.1) | 4x32 GiB fit? |
|---|---|---|---|
| Qwen3.8-27B dense | 27 / 27 | **73.0** (2.1) | already |
| Ornith-1.5-35B-A3B | 35 / ~3 | **67.8** (2.1) | 2x today; 4x easy. Closest MoE, still loses DeepSWE 22 vs 42 |
| Qwen3.6-35B-A3B | 35 / 3 | 51.5 (2.0) | yes; we serve it. Weaker TB than 3.8-27B |
| Qwen3-Coder-Next | 80 / 3 | 36.2 (2.0) | Q4 ~40-50 GiB, yes. Not 73-class |
| Qwen3.5-122B-A10B | 122 / 10 | older 3.5, not 73 | W4A16 ~60 GiB, **fits 4x**. 10B active is the 3-17B band |
| Qwen3-Next-80B-A3B | 80 / 3 | instruct, not TB-73 | fits |
| GLM-4.7-Flash | ~31B MoE | not 73 | fits |
| MiniMax-M2.7 | 229B | Steve's lane | W4 ~110 GiB, **tight on 128** |
| 70B dense W4A16 | 70 / 70 | depends | ~35-40 GiB, 2x is enough; 4x is TP for speed not fit |

**There is no 3B-17B-active MoE that matches 3.8-27B on Terminal-Bench
today.** Ornith 1.5 is the honest "almost" and still trails. 4x B70
unlocks **122B-A10B class** (10B active, ~60 GiB W4) or a comfortable
80B-A3B, not a drop-in faster 3.8. 70B MoE would be ideal and is not
on the shelf; a 70B *dense* W4A16 already fits 2x.

If 4x arrives: first serve is Qwen3.5-122B-A10B W4A16 or Coder-Next
Q4 as a speed toy, plus Ornith NVFP4 TP=4. Do not expect TB 73.

---

## What not to do

- Do not mix A-E in one fire.
- Do not treat NVFP4 HF cards as INT4 XMX.
- Do not enable vLLM `CCL_TOPO_P2P_ACCESS=1`.
- Do not rewrite `scripts/150`; copy to a new number.
- Do not start W4A4 int4xint4 kernels in A-E.
- Do not fake Steve 101.922 (withdrawn; D16 hang).
