# evals/ — quantization quality harness

Measuring **how much a quantization scheme degrades a *fixed* base model**, so we can pick the
right quant for the B70 coding/research server with eyes open.

> Start here, then read `configs/models.yaml` (the quant matrix) and run the Quickstart.
> This harness is **scaffolding under active development** — see the Roadmap at the bottom.

> ### ⚠️ ALWAYS verify which checkpoint is actually served — RTN vs GPTQ
> Quant dirs that differ only by calibration (e.g. `Qwen3-14B-W8A8-rtn` vs `…-gptq`) are a silent
> foot-gun: serving the wrong one mislabels the result. **This already bit us** — the Tier-1 HumanEval+
> `w8a8` number was served from the **RTN** checkpoint, not SmoothQuant+GPTQ (re-run pending). Before
> trusting any result:
> 1. `curl -s http://192.168.10.5:18080/v1/models | python3 -m json.tool` → confirm the **served id**.
> 2. Cross-check that id against `configs/models.yaml` → the **exact model path** it maps to.
> 3. The `served_model_id` must encode the calibration method (`…-gptq` / `…-rtn`), never a bare
>    `qwen3-14b-w8a8`. Less-performant dups are parked in `models/archive/`.

---

## 1. INTENT (read this first — it changes every design choice)

We are **not** asking "is Qwen3-14B good at code?" We are asking:

> **"How much did W8A8 (or FP8, or W4A8) degrade *this same model* versus its FP16/BF16 self?"**

That is a **small-delta measurement**, and it drives everything:

- **The BF16/FP16 run is the reference ceiling.** Every quant is scored as **retention vs that reference**,
  on **identical inputs**. We report `Δ vs bf16`, not just absolute scores.
- The benchmark must be **sensitive** (won't saturate, won't drown the delta in noise) and
  **deterministic** (greedy, batch-1, pinned seed).
- Our real use case is a **long-form coding/research server** (agentic, multi-step, long outputs).
  Quant damage **compounds** over long generations and is nearly invisible on short multiple-choice.
  So we **weight long-generation tasks** (code, reasoning chains, creative builds) over MC trivia.

The first campaign is **Qwen3-14B across {BF16, FP8, W8A8-INT8, W4A8}** — we already have all those
checkpoints on the box. Goals: (a) characterize quant effects in the 14B class, and (b) iron out this
orchestrator before pointing it at the 27B (which needs card #2 to serve W8A8).

⚠️ **14B-class findings may NOT transfer to larger models.** Quant sensitivity changes with scale.
Treat 14B results as "directional for this class," not universal law. Re-run on 27B when it serves.

---

## 2. How it runs (topology)

```
┌──────────────────────────┐         LAN (OpenAI API)          ┌─────────────────────────────┐
│  ms-r1  (this dev box)   │  ── http://192.168.10.5:18080 ──▶ │  Unraid "b70" (Threadripper) │
│  orchestrator + graders  │   /v1/completions, /v1/chat/...   │  vLLM serving one quant      │
│  headless browser (T3)   │                                   │  1x Arc Pro B70 (32 GB)      │
└──────────────────────────┘                                   └─────────────────────────────┘
```

- **The GPU box only serves.** It has no Python (by design) — all eval logic lives here on ms-r1 and
  hits the model over the network. This keeps eval code in the repo and decoupled from the GPU host.
- **Serving** is done by the existing `scripts/NN_serve_*.sh` via `scripts/runremote.sh`. The
  orchestrator can either (a) assume an endpoint is already up (`--endpoint`), or (b) cycle serve
  configs itself (`--serve-script ... --serve-env ...`), wait for `/health`, eval, tear down, repeat.
- Could also run the orchestrator as a container (here or on the box). LAN-API mode is the default
  because it's the simplest and matches how the server is actually used in production.

---

## 3. The tiers (run cheapest → most expensive)

| Tier | Name | What it measures | Grading | Why |
|---|---|---|---|---|
| **0** | **Divergence** | perplexity + **top-1 token agreement** + NLL vs the BF16 reference on a fixed corpus | deterministic, no judge | **The canary.** Cheapest, most sensitive. Catches distribution shift that task accuracy misses. Run for every quant. |
| **1** | **Code (exec)** | HumanEval+ / MBPP+ pass@1; (later) LiveCodeBench, Aider-polyglot | unit tests | What users feel. Execution-graded = objective. |
| **2** | **Reasoning** | GSM8K (+ MATH subset) exact-match | string match | Long reasoning chains compound per-token quant error → sensitive. |
| **3** | **Creative / visual** | curated front-end build prompts (e.g. "ink drop dispersing in water") | semi-objective + pairwise | Real for a coding server. Objective sub-signal: **renders with zero console errors**. Taste: **pairwise A/B vs bf16**. |

**Tier 0 is the headline.** It's fully deterministic, needs no grader, and gives millions of token
comparisons → an extremely tight ranking. If you only have time for one tier, do Tier 0.

---

## 4. THE GOLDEN RULE — measure the noise floor first

Before believing *any* delta, run the reference **against itself**: **BF16 vs BF16** (two runs, or two
seeds). Whatever wobble you see there is your **noise floor**. A quant "regression" smaller than the
noise floor is **noise, not a finding.** Most homegrown quant evals skip this and report noise as signal.

The orchestrator has a `--noise-floor` mode that runs a config twice and reports run-to-run variance.

---

## 5. DOs

- **DO** anchor everything to a single BF16/FP16 reference and report `Δ vs reference` on identical inputs.
- **DO** run the **noise floor** (§4) and only trust deltas that clear it.
- **DO** decode **greedily** (`temperature=0`), `batch/concurrency=1`, fixed `max_tokens`, fixed `seed`.
- **DO** weight **long-generation** tasks (code, reasoning, creative) — that's where quant damage shows.
- **DO** record full provenance per run: model id, quant scheme, vLLM image+version, serve flags,
  KV-cache dtype, git SHA of this repo, endpoint, date, sampling params. (The orchestrator does this.)
- **DO** prefer **contamination-resistant** code benches (LiveCodeBench, recent problems) for headline claims.
- **DO** keep raw generations (for Tier-3 pairwise judging and for auditing graders later).

## 6. DON'Ts

- **DON'T** compare across **different base models** in the same table and call it a "quant" effect.
- **DON'T** lean on **MMLU / multiple-choice** as the primary signal — MC accuracy is fairly quant-robust
  and will under-report degradation. Use generative/exec/reasoning tasks.
- **DON'T** report a 1–2 point pass@1 gap on HumanEval (164 items) as real — its confidence interval is
  wider than that. Either use enough items or lean on Tier 0's tight signal.
- **DON'T** use a sampling temperature > 0 for the graded tiers (kills reproducibility). Temp>0 is fine
  *only* for a separate "real-use feel" pass, with a pinned seed.
- **DON'T** trust a single LLM-judge score for Tier 3 — use **pairwise, position-swapped**, ideally 2+ judges.
- **DON'T** let the chat template silently differ between configs — a template change masquerades as a
  quant effect. Pin the template / use the same request path for all quants.

---

## 7. Pitfalls (the subtle ones that bite quant-delta studies)

1. **Determinism is not free.** Even `temperature=0` is **not bitwise-stable** on GPU across batch sizes
   and kernel choices. vLLM continuous batching means *what else is in the batch* can change your tokens —
   vLLM explicitly does **not** guarantee reproducibility by default. Mitigate: serve with
   **`VLLM_BATCH_INVARIANT=1`** (batch-invariant kernels), eval at **concurrency 1**, set `seed`, fixed
   `max_tokens`, **no speculative decoding**, and pin the same vLLM version + tokenizer revision across
   quants. Then still run the noise floor to quantify residual wobble. Don't claim sub-noise deltas.
   (Refs: vLLM docs `usage/reproducibility` and `features/batch_invariance`.)
2. **Sample size / confidence intervals.** Small benches (HumanEval=164) have wide CIs. Report CIs or use
   Tier 0 (huge effective N at the token level) for the precise ranking.
3. **Contamination & saturation.** Classic benches leak into training and saturate on strong models →
   no headroom to see degradation. Prefer time-stamped/recent problems for headline numbers.
4. **MC insensitivity.** (see DON'Ts) — quantization can wreck long generations while MC barely moves.
5. **Chat-template / prompt drift** across configs looks exactly like a quant effect. Hold it constant.
6. **Apples-to-apples serving.** Same KV-cache dtype, same max-model-len, same vLLM version where possible.
   If you *must* vary the serve config (e.g. W8A8 needs the int8 image), **record it** and treat
   cross-image comparisons with caution.
7. **Tier-3 judge bias.** LLM judges favor longer/first answers. Use pairwise + swap positions + average.

---

## 8. Quant matrix (first campaign: Qwen3-14B)

See `configs/models.yaml` for the live, authoritative list. As of writing, on the box:

| Label | Checkpoint (on box) | Weights | Acts | B70 serveable? (kernel) |
|---|---|---|---|---|
| `bf16` (REFERENCE) | `Qwen3-14B` (specula-build) | 16 | 16 | ❌ ~29.6 GB won't fit one card (v1 engine). Scored offline on CPU. |
| `fp8` | online `--quantization fp8` | 8 fp | 8 fp | ✅ XPU FP8 kernel |
| `w8a8` | `Qwen3-14B-W8A8-INT8` | int8 | int8 dyn | ✅ **our** `XPUInt8ScaledMMLinearKernel` |
| `w4a8` | `Qwen3-14B-W4A8-INT` | int4 | int8 dyn | ✅ `XPUW4A8IntLinearKernel` |
| `w4a16` | `Qwen3-14B-W4A16` (scripts/54 RTN) | int4 | 16 | ✅ `XPUwNa16LinearKernel` (int4 weight-only) |
| `w8a16` | `Qwen3-14B-W8A16` (scripts/54 RTN) | int8 | 16 | ❌ **NO XPU kernel** — `XPUwNa16` is int4-only (uint4/uint4b8); our int8 GEMM needs int8 acts. Scored offline on CPU. |

**Kernel-coverage finding (2026-06-19):** the B70 (vLLM 0.23.0 + our kernels) serves fp8, W8A8-int8,
W4A8-int, and **W4A16 (int4 weight-only)** — but **NOT W8A16 (int8 weight-only)**: the `XPUwNa16` kernel
only accepts int4 weights. **W8A16 is the one missing kernel, and the eval says it would be near-lossless**
(ppl 12.76, 0.981 token-agreement vs bf16 — better fidelity than our W8A8's 0.881). The catch: W8A16 keeps
fp16 activations, so it does **not** light the INT8 systolic path — it's a memory-savings play, not a
compute-speed one, whereas W8A8 (int8 acts) does use the fastpath. So the kernel call is a genuine
tradeoff: write a W8A16 int8-weight-only kernel for max fidelity, vs. keep optimizing W8A8 for speed
(its task accuracy is already ≈ fp8). See [results/SUMMARY.md](results/SUMMARY.md) for the full analysis.

⚠️ `w4a16` ≠ `w4a8` (different schemes — int4-w/fp16-a vs int4-w/int8-a). Both now exist + are evaluated.

---

## 9. Outputs

Each run writes a self-describing directory under `evals/results/` (raw gens gitignored, summaries kept):

```
evals/results/<UTC-stamp>__<model>__<quant>/
  config.json        # full provenance (model, quant, serve flags, git SHA, sampling, endpoint)
  tier0_divergence.json
  tier1_code.json
  tier2_reasoning.json
  tier3_creative/    # raw generations + rendered screenshots + console-error log
  summary.json       # the headline numbers for this (model, quant)
```

`orchestrator/report.py` rolls all summaries into a single **"% retention vs reference"** markdown table
(the deliverable that feeds the repo's `FINDINGS.md` scoreboard).

---

## 10. Quickstart

```bash
# 0. one-time: deps on this dev box (ms-r1)
python3 -m venv evals/.venv && source evals/.venv/bin/activate
pip install -r evals/requirements.txt          # openai, lm-eval, evalplus, playwright, pyyaml, numpy
playwright install chromium                     # for Tier-3 headless render

# 0b. one-time: build the Tier-1 code-execution sandbox image (Docker; see §11)
bash evals/sandbox/build.sh                     # -> evalplus-sandbox:0.3.1

# 1. (on the box) serve a quant — e.g. W8A8 + PIECEWISE graph:
#    scripts/runremote.sh scripts/51_serve_int8_specdecode.sh IMG=vllm-xpu-env:int8g GRAPH=1 CGMODE=PIECEWISE SPEC=0

# 2. smoke the endpoint from here
python3 evals/orchestrator/run_evals.py --endpoint http://192.168.10.5:18080/v1 --check

# 3. run a tier (Tier 0 = the canary; needs the BF16 reference endpoint too for agreement/KLD)
python3 evals/orchestrator/run_evals.py \
    --endpoint http://192.168.10.5:18080/v1 --model Qwen3-14B-W8A8-INT8 --quant w8a8 \
    --reference-endpoint http://192.168.10.5:18080/v1 --reference-model Qwen3-14B \
    --tiers 0,2

# 3b. Tier 1 (execution-graded code, sandboxed). Smoke first with --limit, then drop it for all 164.
python3 evals/orchestrator/run_evals.py \
    --endpoint http://192.168.10.5:18080/v1 --model qwen3-14b-fp8 --quant fp8 \
    --tiers 1 --tier1-dataset humaneval --limit 5      # smoke; full run = omit --limit

# 4. roll up everything into a retention table
python3 evals/orchestrator/report.py evals/results/ > evals/results/SUMMARY.md
```

---

## 11. Tier 1 — the code-execution sandbox (Docker)

Tier 1 is the only tier that **executes model-generated code** (to grade HumanEval+/MBPP+ pass@1 with
EvalPlus). That code is untrusted, so the pipeline splits into three steps and isolates the one that runs it:

```
1. GENERATE  (host, SAFE)   our own OpenAI-client loop → raw markdown responses
                            • same discipline as tiers 2/3: greedy (temp 0), fixed seed, concurrency 1,
                              enable_thinking OFF by default (flip with --tier1-think)
                            • EvalPlus's exact chat prompt, so samples drop straight into its grader
2. SANITIZE  (host, SAFE)   `evalplus.sanitize` — tree-sitter text extraction, no code is run
3. EVALUATE  (DOCKER)       `evalplus.evaluate` RUNS the code vs the +tests → pass@1. Sandboxed:
                              --network none · non-root --user · throwaway cache copy · mem/pids caps
```

Why our own generator instead of `evalplus.codegen`: codegen's OpenAI backend can't pass
`chat_template_kwargs.enable_thinking`, so it would silently run **thinking-ON** and diverge from the
rest of the harness. We replicate its prompt exactly and own the request.

**The sandbox guarantees** (see `evals/sandbox/Dockerfile` + `orchestrator/tier1_code.py`):
- **`--network none`** — generated code gets no network.
- **non-root `--user $(id -u):$(id -g)`** — no root in-container; output files are owned by you.
- **throwaway cache** — we copy `~/.cache/evalplus` into the run dir and mount *that* (read-write), so
  the untrusted code (and the ground-truth `.pkl` EvalPlus writes) can't touch your real cache.
- **`--memory 8g --pids-limit 512`** — blast-radius caps.

```bash
bash evals/sandbox/build.sh        # one-time: build evalplus-sandbox:0.3.1 (pinned to host evalplus)
# then run Tier 1 (see Quickstart §10 step 3b). --limit N trims a throwaway dataset copy for fast smokes;
# omit --limit for a real run (EvalPlus then asserts full 164-problem coverage — a useful drop-check).
```

Escape hatch: `--allow-code-exec` runs `evalplus.evaluate` **on the host, UNSANDBOXED** (only if Docker
is unavailable and you trust the environment). Default is always the Docker sandbox.

---

## 12. Roadmap / TODO

- [ ] Tier 0: full-vocab KLD via an offline forward-pass script (API gives only top-k → approximate now).
- [ ] Tier 1: wire LiveCodeBench (contamination-resistant) + Aider-polyglot (edit-correctness, most
      relevant to the coding-server use case).
- [x] Tier 3 **validated** on W8A8 (2026-06-19): 8 builds generated + headless-rendered, 6/8 renders-clean.
      Warts to fix: (a) screenshot at a fixed 2.5 s can catch an animation mid-intro (ink-drop snapped
      before dispersion) → capture multiple frames or a later T; (b) `non_blank` heuristic false-negatives
      CSS-only pages (no canvas/svg/img) → count visible styled DOM nodes instead.
- [ ] Tier 3: vision-LLM auto-judge on screenshots (pairwise, swapped) to cut manual viewing.
- [x] Tier 1 (code/EvalPlus) **WIRED + sandboxed** (2026-06-19): generate(host)→sanitize(host)→evaluate
      (Docker, --network none, non-root, throwaway cache). Validated on Qwen3-14B-fp8 HumanEval+ (§11).
- [ ] Auto-cycle serve configs from `models.yaml` (serve → wait healthy → eval → tear down → next).
- [ ] CIs / bootstrap on all task scores; flag any delta below the noise floor.
- [ ] Optional: containerize the orchestrator for one-command runs.
