# Qwen3.6 quant eval on a single Arc Pro B70 (2026-06-19)

> **The real targets are Qwen3.6-27B and Qwen3.6-35B-A3B** (quality + real-world speed). The Qwen3-14B
> campaign below was harness verification + a quant-delta study (don't read it as the headline).

> ## [!] DECODE LEADERBOARD UPDATE (2026-06-20): PIECEWISE graph capture ~DOUBLES int4 decode
> **Every decode t/s in the tables below was measured EAGER (`--enforce-eager`). That is no longer the right
> config.** PIECEWISE XPU graph capture (image `:int8g`, `w4a8/30_serve_w4a8_graph.sh GRAPH=1`, torch 2.11+xpu)
> works on the B70 and is the single biggest decode lever found. Captured single-stream decode (Qwen3-14B, same
> probe), eager -> PIECEWISE:
>
> | model / quant | VRAM | eager t/s | **PIECEWISE t/s** | gain |
> |---|---|---|---|---|
> | **27B int4 (AutoRound)** | 17.6 GB | 7.84 | **30.84** | **+293% (3.93x!)** -- erases the density tax |
> | **14B w4a16-gptq** | 9.3 GB | 28.0 | **54.6** | **+95%** (14B decode leader) |
> | **14B w4a8-gptq**  | 9.3 GB | 16.8 | **48.2** | **+187%** (also best prefill/TTFT, int8-XMX) |
> | 14B w8a8-gptq | 15.3 GB | 23.6 | 26.7 | +13% (fused quant already lean) |
> | 14B fp8 | 15.3 GB | ~32 | (capped ~40 by 15.3 GB) | -- |
> | **35B-A3B MoE int4** | 19.6 GB | 7.93 | **56.84** | **+617% (7.17x!!)** -- now the FASTEST config on the board |
>
> **[!] BOTH flagships transformed; the "higher-density tax" is ERASED.** Old story: the big models win quality
> but decode ~4x slower than the 14B fp8 (27B 7.9, 35B ~6 vs fp8 32). With PIECEWISE capture: **27B -> 30.84 t/s
> (~= 14B fp8) keeping +4 quality pts; 35B-A3B MoE -> 56.84 t/s = the FASTEST single-card decode measured.** The
> capture gain scales with how dispatch-bound the eager path is: **MoE 7.17x > 27B-GDN 3.93x > w4a8 2.87x >
> w4a16 1.95x > w8a8 1.13x** (monotonic in eager-op-count). On one B70 the big models went from "great but slow"
> to "great AND fast" -- the campaign's highest-impact outcome. (35B serves on `:v0230moe` = `:v0230` + the INC
> MoE routing patch; the masked `fused_moe` kernel is routing-agnostic so it captures correctly.)
>
> **Revised picks (single B70, captured):** decode-heavy/interactive -> **w4a16-gptq (54.6 t/s, near-lossless,
> 9.3 GB)**; prefill-heavy/long-context/agentic -> **w4a8-gptq** (48 decode + best prefill/TTFT, 9.3 GB). w4a8 is
> NO LONGER "dominated" -- it co-leads with w4a16, split by the decode-vs-prefill axis. Both int4 paths ~2x with
> capture (they were eager-dispatch-bound; w4a8 most, due to its unfused act-quant). See JOURNAL 2026-06-20 +
> FINDINGS + docs/kernel/04. (One gotcha: torch.compile on XPU needs the `pass_config` fix in 30_serve to dodge
> a `NameError: MLARoPEKVCacheCatFusionPass` from CUDA-only fusion passes.)

## REAL TARGETS — Qwen3.6 on one B70 (vLLM 0.23.0, `:v0230` build, greedy/eager)

| model (quant) | serves on 1xB70? | gsm8k | HumanEval+ b/+ | ppl | decode t/s | TTFT ms | prefill t/s | VRAM |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-27B** (**NVFP4** ModelOpt, vLLM v0.24.0 GRAPH+MTP) | **yes** | 96% (48/50) | **0.988 / 0.945** | - | **38.7-41.8** | 1203 | 1702 | 24.1 GB + 2.3 graphs |
| **Qwen3.6-27B** (AutoRound int4) | **yes** | **100% (50/50)** | 0.963 / 0.927 | 6.60 | 7.59 | 305 | 1369 | 17.6 GB |
| **Qwen3.6-35B-A3B** (Intel int4 AutoRound, 256-expert MoE) | **yes** -- needs INC-XPU MoE patch | - | - | - | ~6 (eager, untuned) | - | - | 19.6 GiB load + 6.24 KV |

> ## [!] W4A8 ACCURACY GATE -- PASS (2026-06-28, sglang-XPU, HumanEval+ 164, thinking-off, greedy, sandboxed)
> The vision-retaining **W4A8/W4A16 hybrid** daily driver (`rdy_to_serve/qwen36-27b-w4a8-graph`, int4_gemm ops:
> int8-act prefill + fp16-act decode) was eval'd vs the int4-woqgemm champion on the **SAME sglang stack +
> SAME Lorbus int4 weights** (the apples-to-apples test that isolates the int8-act prefill from the stack):
>
> | config (sglang GRAPH=1, card 0, same int4 weights) | pass@1 base | pass@1 plus | base-fails |
> |---|---|---|---|
> | **W4A8 hybrid** (int8-act prefill) | **0.921** | **0.896** | 13 |
> | int4-graph champion (woqgemm, fp16-act) | **0.921** | **0.896** | 13 |
>
> **DELTA 0.000 / 0.000 -- the int8-act prefill causes ZERO coding-accuracy loss.** Failed sets overlap 9/13;
> the 4 that differ are marginal problems flipped by greedy non-determinism across kernels (nets to zero).
> So W4A8 = int4 on accuracy AND faster (decode 25.15 vs 23.5 t/s, lower TTFT, vision retained) -> a real
> daily-driver upgrade. The absolute 0.921 (vs the vLLM 0.963 row below, SAME weights) is depressed by output
> verbosity: **9 of 13 base failures are truncation at max_tokens=2048** (verbose comment-heavy solutions, fence
> never closed), not quant damage; sglang's chat template is more verbose than vLLM's -> more truncation. The
> gap is a stack/methodology artifact, not the kernel. Result dirs under `evals/results/...__w4a8-graph` and
> `...__int4-graph-sglang`; full analysis in `sglang/W4A8_PLAN.md` (ACCURACY GATE 2026-06-28d).

- **The higher-density tradeoff, quantified (2026-06-20, HumanEval+ 164, thinking-off):** the 27B int4 hits
  **0.963 / 0.927** vs the best 14B (fp8 **0.915 / 0.890**) -- **+4.8 base / +3.7 plus** -- but decodes at
  **7.9 t/s vs 32** (fresh `perf_probe`: 7.94 t/s, TTFT 283 ms, prefill 1376 t/s; confirms the 7.6 below).
  So **~4x slower decode buys ~+4 pts pass@1.** Note HumanEval near-saturates here, so this *understates*
  the 27B's real edge (its gsm8k is a clean 100% vs the 14B's ~95%); on harder/agentic code the gap widens.
- **27B runs great on one card** (aces gsm8k, much stronger than the 14B) but **decode is slow (7.6 t/s)** —
  big dense model + Gated-DeltaNet + unoptimized int4 decode. Prefill 1369 t/s, TTFT 305 ms.
  - Requires the **`:v0230` full build** — our `:int8` image was built minimal (`GDN_ENABLED=OFF`) so it
    **lacks `gdn_attention`** and crashes on the first token. Also: copy the base chat_template into the
    AutoRound tokenizer_config (it ships without one).
  - Our own **compressed-tensors W4A16 27B fits (25 GB) but won't serve**: `XPUwNa16` needs input dims ÷32,
    and the 27B's gated attention has a 4304 dim (the 14B never hit this). Needs a 32-pad / ignore / kernel fix.
- **[2026-06-20] 35B-A3B int4 NOW LOADS + GENERATES on one card** -- the "MoE-on-XPU gap" was a routing bug, not a
  missing kernel. vLLM 0.23.0+xpu's INC integration (`auto-round` -> `INCConfig`) returned `None` for `RoutedExperts`
  on XPU -> FusedMoE fell back to bf16 (`UnquantizedFusedMoEMethod`) -> 256 experts dequantized toward ~70 GB ->
  `OUT_OF_DEVICE_MEMORY` at load. A **16-line patch** routes the experts to `MoeWNA16Config` (the pure-Triton
  `fused_moe_kernel_gptq_awq`, no CUDA-only op since `should_moe_wna16_use_cuda` is `is_cuda`-gated). Now:
  **load 19.6 GiB** (int4 packed) + 6.24 GiB KV (122,880 tok, 60x@2048), coherent output, **~6 t/s eager
  (untuned)**. Patch + load-test: `contrib/vllm_moe_xpu/` + `scripts/53_loadtest_35b_moe_xpu.sh`. NOT yet: perf
  tuning (no `E=256,N=512,int4_w4a16` config -> "sub-optimal" warning), graph capture, accuracy eval.

---

> ## [!] NEW #1 (2026-07-04): NVFP4 27B = 0.988 / 0.945 -- beats int4-AutoRound on quality AND is 5x faster
> The `nvidia/Qwen3.6-27B-NVFP4` ModelOpt MIXED_PRECISION checkpoint (W4A16_NVFP4 MLP + FP8 attention),
> served on ONE B70 via our custom `nvfp4_gemm_w4a16` oneDNN op + PIECEWISE capture + stock NEXTN MTP
> (`vllm/nvfp4/serve_nvfp4_27b.sh`, MODE=fused GRAPH=1 MTPTOK=3): HumanEval+ **0.988 base / 0.945 plus**
> (164, thinking-off, greedy, sandboxed; result dir `...__nvfp4-fused-graph-mtp3`). That is **+2.5 base /
> +1.8 plus over the long-standing int4-AutoRound champion** (0.963/0.927) at **38.7-41.8 t/s vs 7.9** --
> quality AND speed, no tradeoff. Two honesty notes: (a) MTP is output-invariant at greedy, so the number
> reflects the NVFP4 numerics, not spec decode; (b) stack differs from the 0.963 row (vLLM v0.24.0
> captured vs v0.23 eager) -- template/stack artifacts historically move this metric by a few points
> (see the W4A8 gate note below). A same-stack re-run was ATTEMPTED 2026-07-04 and is currently
> IMPOSSIBLE: the v0230 int4 serve on card 1 soft-poisons to "!!!!" in BOTH captured AND eager modes
> (post-kernel-7.1 suspect; JOURNAL NEG entries) -- deferred to an int4-on-v0.24.0 port. Still, the
> direction is consistent: FP8-attn + 4-bit-MLP with NVIDIA's
> calibration is simply a higher-fidelity 4-bit recipe than int4-AutoRound-everywhere. gsm8k (tier2,
> 50 items): 96% (48/50) vs int4's 100% (50/50) -- 2 misses, small-sample noise territory (p~0.15),
> but recorded honestly; ppl pending.
> Bonus: the eval GENERATED 3.3x faster than previous 27B runs (700 s vs ~2300 s) -- MTP accept on code
> is ~99% (per-position 1.00/1.00/0.97), i.e. coding workloads decode at ~50 t/s on one card.

## Code-quality leaderboard (HumanEval+) + decode speed -- all single-B70 configs (2026-06-20)

> Every config that serves on ONE Arc Pro B70, sorted by HumanEval+ plus pass@1 (164 problems, thinking-off,
> greedy, sandboxed grading). Same base models under different quant+calibration; speed is fresh single-stream
> `perf_probe` (greedy, eager). 14B base bf16 does not fit one card, so fp8 is the practical 14B ceiling.

| rank | model | quant (calib) | HumanEval+ base / plus | decode t/s | TTFT ms | prefill t/s | VRAM |
|---|---|---|---|---|---|---|---|
| **1** | Qwen3.6-27B | **NVFP4 (NVIDIA ModelOpt)**, vLLM v0.24.0 GRAPH+MTP | **0.988 / 0.945** | **38.7-41.8** | 1203 | 1702 | 26.4 GB |
| 2 | Qwen3.6-27B | int4 (AutoRound) | 0.963 / 0.927 | 7.9 | 283 | 1376 | 17.6 GB |
| 2 | Qwen3-14B | w8a8 (GPTQ) | 0.921 / 0.890 | 23.5 | 101 | 5740 | ~15 GB |
| 2 | Qwen3-14B | fp8 (online) | 0.915 / 0.890 | **32.1** | **85** | 3525 | ~15 GB |
| ~4 | Qwen3-14B | w8a8 (AutoRound) | 0.909 / 0.872 | ~23.5 | -- | -- | ~15 GB |
| 4 | Qwen3-14B | w8a8 (RTN) | 0.902 / 0.860 | 23.8 | 101 | **5780** | ~15 GB |
| 5 | Qwen3-14B | w4a16 (GPTQ) | 0.872 / 0.848 | 29.0 | 84 | 2920 | ~9.3 GB |
| 6 | Qwen3-14B | **w4a8 (GPTQ)** | **0.872 / 0.835** | ~16.5 | ~139 | ~4403 | 9.3 VRAM / 16 disk* |
| 7 | Qwen3-14B | w4a16 (RTN) | 0.866 / 0.829 | 29.1 | 79 | 2921 | ~9.3 GB |
| 8 | Qwen3-14B | w4a8 (RTN, archived) | 0.860 / 0.817 | 16.5 | 139 | 4403 | 9.3 VRAM / 16 disk* |

- **Quality:** the 27B int4 leads by ~+4 plus pts but decodes ~4x slower (7.9 vs 32 t/s) -- the higher-density
  tradeoff. Among the 14B, **w8a8-gptq ties fp8 at the top (0.890 plus)**; **GPTQ beats RTN at both schemes**
  (w8a8 +3.0 plus, w4a16 +1.9 plus -- the latter within HumanEval's CI; **w4a8 +1.8 plus, 0.817->0.835**). bf16
  14B ceiling is unmeasured here (does not fit one card); fp8 is the practical anchor.
- **[2026-06-20] w4a8-gptq closes most of the int8-act gap:** GPTQ-W4A8 = **0.872 / 0.835** -- base **TIES**
  w4a16-gptq (0.872), plus within ~1 CI (0.835 vs 0.848). So w4a8 is now accuracy-viable, NOT "dominated"; its
  weak spot is decode (16.5 t/s) while its EDGE is prefill/TTFT (int8-XMX: +51% prefill, ~-32% TTFT vs w4a16 --
  see the w4a8-vs-w4a16 head-to-head). Accuracy no longer blocks w4a8; the decode kernel does (see docs/kernel/04).
- **Speed:** fp8 fastest decode (32.1) + lowest TTFT; w8a8 best prefill (5780, the int8 systolic path);
  w4a16 best quality-per-VRAM (9.3 GB at 29 t/s). **w4a8-gptq: slowest single-stream decode (16.5) but the
  int8-XMX prefill/TTFT edge at 9.3 GB** (kernel-limited decode, fixable -- see docs/kernel/04).
- **Picks:** interactive/chat -> **fp8** (or **w8a8-gptq** for matching quality + 1.6x prefill); VRAM-tight ->
  **w4a16-gptq** (best small-footprint quality, fast decode); max quality on one card -> **27B int4** if 7.9
  t/s decode is tolerable; **w4a8-gptq** when prefill/TTFT/throughput-heavy (long-context/agentic, the int8-XMX
edge), else w4a16-gptq for decode-heavy. HumanEval near-saturates -- treat sub-2pt gaps as ties.
- *VRAM = vLLM "Model loading took" (weight memory on the GPU), not on-disk size. **w4a8 is the exception:**
  its int4 weights are stored **UNPACKED** on disk (single 16 GB safetensors, ~1 byte/int4-weight), but vLLM
  packs them to **9.3 GiB in VRAM** (verified 2026-06-20; also the slowest load at 39 s vs ~23 s). Repack-to-
  4bit on disk is pending -- it cuts disk + load time, not VRAM. (W8A8-gptq is also 16 GB on disk, but that is
  int8 = naturally ~1 byte/weight, so disk ~= its 15.3 GiB VRAM; no repack to gain there.)*

---

# (Secondary) Quant-delta study — Qwen3-14B (harness verification, 2026-06-19)

How much each quantization degrades the **same** Qwen3-14B vs its BF16 self. Served one-at-a-time on a
single B70, vLLM 0.23.0-based images, greedy/eager, eval concurrency 1, thinking **off**.

## Results

| quant | weights / acts | calib | ppl ↓ | top1-agree vs bf16 ↑ | nll-gap ↓ | gsm8k (n=150) ↑ | HumanEval+ pass@1 base/+ ↑ | serves on B70 |
|---|---|---|---|---|---|---|---|---|
| **bf16** (reference) | 16 / 16 | — | 12.7010 | — | — | — | — *(can't serve)* | ❌ ~29.6 GB > one card |
| **fp8** | 8fp / 8fp | online | 12.6966 | 0.968 | 0.062 | 0.960 (144/150) | **0.915 / 0.890** | ✅ XPU FP8 |
| **w8a16** | int8 / 16 | RTN | 12.7596 | **0.981** | 0.037 | — *(can't serve)* | — *(can't serve)* | ❌ no XPU kernel |
| **w8a8** | int8 / int8 | RTN | 13.0839 | 0.881 | 0.250 | 0.953 (143/150) | 0.902 / 0.860 | ✅ **our int8 kernel** |
| **w4a16** | int4 / 16 | RTN | 13.5528 | 0.841 | 0.340 | 0.947 (142/150) | 0.866 / 0.829 | ✅ XPUwNa16 |
| **w4a8** | int4 / int8 | RTN | 14.1943 | 0.822 | 0.420 | 0.927 (139/150) | 0.860 / 0.817 | ✅ XPUW4A8Int |

- **top1-agree** = fraction of 1063 corpus tokens where the quant's greedy argmax == bf16's (1.0 = identical).
- **ppl** on a fixed prose+code corpus. **gsm8k**: thinking-off, greedy, `#### <n>` exact-match, first 150 test items (paired).
- bf16 + w8a16 scored **offline on CPU** (neither serves on one card); tokenization verified identical to vLLM /tokenize (0/10 misaligned).
- **Tier 1 (execution-graded code): full 14B sweep done** (2026-06-20) — HumanEval+ column above, 164
  problems, thinking-off, greedy, sandboxed Docker grading (`--network none`, non-root, throwaway cache;
  harness README §11). **Code spreads ~7 pts (fp8 0.890+ → w4a8 0.817+) where gsm8k moved ~3** — the
  long-generation signal the harness is built to surface. Ordering **fp8 > w8a8 > w4a16 > w4a8** agrees with
  ppl/agreement. HumanEval is contamination-prone → treat as directional; Tier 0 is the precise rank.
  *(Qwen3.6-27B int4 = 0.963/0.927 -- see the REAL TARGETS table up top for the higher-density jump.)*

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

## Performance (single-stream, greedy, eager, max-len 4096) — `perf_probe.py`

| quant | decode t/s ↑ | TTFT ms ↓ | prefill t/s ↑ | serve VRAM |
|---|---|---|---|---|
| fp8 | **29.96** | 82 | 3531 | ~15 GB |
| w8a8 | 21.86 | 121 | **5787 (1.64× fp8)** | ~15 GB |
| w4a16 | 26.40 | 89 | 2939 | **~9.3 GB** |
| w4a8 | 16.47 | 139 | 4403 | **9.3 GB** (16 on disk, unpacked) |

> A fresh `perf_probe` pass during the 2026-06-20 Tier-1 sweep re-confirmed the ordering (fp8 32.1 ·
> w4a16 29.1 · w8a8 23.8 · w4a8 16.5 t/s decode — same ranks, ~5-10% run-to-run higher). **w4a8 decodes
> *slowest* despite the smallest weights:** int8-activation dynamic quant per token adds decode overhead
> that int4-weight bandwidth savings don't recover — so w4a8 is a pure memory play (9.3 GB), not a speed one.

- **Decode (bandwidth-bound):** fp8 > **w4a16 26.4** > w8a8 21.9. int4 weights (9.3 GB) stream less per
  token, so w4a16 *out-decodes* w8a8 despite lower quality. (W8A8 + PIECEWISE graph → ~27, closes the gap.)
- **Prefill (compute-bound):** **w8a8 wins 1.64× fp8** — the INT8 systolic datapath. w4a16 is *worst* at
  prefill (int4 weight-only dequant → fp16 matmul, no systolic).
- **Pick by workload:** decode/chat → fp8 (or w4a16 if VRAM-tight); long-context/prefill/batch → **w8a8**;
  smallest footprint → w4a16. This is exactly why W8A8 (prefill + INT8 path) is the coding-server kernel target.

## Calibration: RTN vs GPTQ+SmoothQuant — does it matter? (2026-06-19)

The matrix above is all **RTN** (data-free). We re-quantized two schemes with calibration (128 samples ×
2048 tok ≈ 260k activation rows/Hessian — the GPTQ-paper default): **W8A8 = SmoothQuant+GPTQ**, **W4A16 =
GPTQ** (weight-only → SmoothQuant is a no-op, skipped).

| quant | recipe | ppl | agree vs bf16 | gsm8k | HumanEval+ b/+ |
|---|---|---|---|---|---|
| W8A8 | RTN | 13.08 | 0.881 | 95.3% | 0.902 / 0.860 |
| W8A8 | **SmoothQuant+GPTQ@128** | 13.05 | **0.908** (+2.7) | 94.7% | **0.921 / 0.890** |
| W8A8 | SmoothQuant+GPTQ**@512** | 13.15 | 0.900 | 95.3% | - |
| W4A16 | RTN | 13.55 | 0.841 | 94.7% | 0.866 / 0.829 |
| W4A16 | **GPTQ@128** | **13.34** | **0.883** (+4.2) | **96.7%** (+2.0) | **0.872 / 0.848** |

- **GPTQ shows up MORE on code than gsm8k.** W8A8 RTN->GPTQ: gsm8k moved ~0 (saturated) but HumanEval+ plus
  jumped **0.860 -> 0.890 (+3.0)** and base **0.902 -> 0.921 (+1.9)** -- GPTQ-W8A8 plus now *matches* fp8
  (0.890) and base *beats* it (0.921 vs 0.915). Calibration is free at inference (decode 23.5 t/s, = RTN).
  Confirms why we weight long-generation tiers: the calibration lift is nearly invisible on saturated gsm8k.
- **But on code, int8 GPTQ helped MORE than int4 GPTQ -- opposite of the agreement metric.** W4A16 RTN->GPTQ
  on HumanEval+ = **0.829 -> 0.848 (+1.9 plus) / 0.866 -> 0.872 (+0.6 base)**, smaller than W8A8's +3.0,
  even though int4's *agreement* lift (+4.2) was bigger than int8's (+2.7). Caveat: both code deltas are near
  HumanEval's 164-item CI (a few problems), so read direction-not-magnitude; Tier 0 agreement stays the tight
  rank. Net: GPTQ is worth it both places (free at inference, never hurts), most clearly for W8A8.

- **Calibration's lift scales with quantization error.** W8A8 (int8 weights, already near-lossless) gains
  +2.7 agreement pts and ~0 ppl. W4A16 (int4 weights) gains +4.2 agreement, −0.21 ppl, +2 gsm8k — int4 has
  real weight error for GPTQ to recover.
- **GPTQ-W4A16 ≈ RTN-W8A8 in token fidelity** (0.883 vs 0.881): good int4 calibration buys back roughly an
  activation-bit of fidelity.
- For **W8A8 it's SmoothQuant, not GPTQ**, doing the work — it sharpens the int8 *activation* quant (the W8A8
  bottleneck), so agreement tightens even though weights/ppl barely move. gsm8k stays within noise (saturated).
- **Sample count: 128 ≈ 512, use 128.** W8A8 SmoothQuant+GPTQ@512 (ppl 13.15, agree 0.900, gsm8k 95.3%) is
  **within noise of @128** (13.05 / 0.908 / 94.7%) — actually marginally worse, i.e. pure run variance. More
  samples bought nothing. And it was NOT cheap: **@512 took ~99 min vs @128's ~30 min (~3×)** — the Cholesky
  inverse is sample-independent, but the calibration forward passes AND the Hessian accumulation (Σxxᵀ) both
  scale with samples. 128 seqs × 2048 tok ≈ 260k activation rows already conditions the Hessian. **Default 128.**

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
