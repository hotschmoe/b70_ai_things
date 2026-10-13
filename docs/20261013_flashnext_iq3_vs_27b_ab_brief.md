# Flash-Next IQ3 (all experts in VRAM) vs Qwen3.8-27B INT4: A/B brief

Date: 2026-10-13. Orchestrator: Claude. Workers: Codex and Sonnet subagents.

## Decision recorded

- The user dropped the "no substitution" rule of the 2026-10-08 campaign on
  2026-10-13. Other publishers, lower-bit checkpoints and pruned variants are
  allowed, provided the served ID and result paths name the exact artifact.
- The UD-Q4_K_XL RAM-spill engine work is PAUSED. Last measured state:
  about 255 tok/s prefill and 20-22 tok/s decode, ctx 4096, one request, no
  cache (JOURNAL.md, "optimized56 Direct model timings finish"). Its portable
  findings stay; no new harness, admission or source-plan work on it.
- No custom VRAM+RAM expert spillover in vLLM or SGLang. Estimated 23-36
  engineer-weeks for a 25-40 tok/s ceiling.

## Question

Does a ~3-bit Qwen3.8-Flash-Next with every expert resident on the two B70s
beat the current Qwen3.8-27B INT4 AutoRound daily driver on agentic coding by
enough to justify lower speed and lower concurrency?

Known going in: Flash-Next cannot win on speed here. The 27B does 90+ tok/s
decode, 2000+ tok/s prefill, four streams to 200k (user-reported, to be
remeasured in the matched arm). Published dual-B70 all-VRAM Flash-Next IQ3 is
53-66 tok/s decode on Strata (community reports, unverified). The only reason
to migrate is quality on long-horizon agentic work (BF16 DeepSWE 58.7 vs 42.2,
SWE-bench Pro 62.5 vs 61.7). Nobody has measured whether 3 bits keeps that.

## Arms

A. Baseline: Qwen3.8-27B INT4 AutoRound, the current daily-driver recipe,
   unchanged. Weights models/files/qwen3.8-27b/int4-autoround-bce40cac
   (confirm against the live recipe before the run).

B. Candidate: Qwen3.8-Flash-Next, unpruned, ~3-bit, all experts on GPU, n-gram
   (PLE) table off-GPU on NVMe (/mnt/cache; /mnt/vm_8tb is SATA).
   - Primary artifact: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_S
     (83.6 GB, about 55 GB on the cards). Strata-native, so the Unsloth
     `--compat-bf16` Q8 hyper-connection gap should not apply; verify.
   - Fallback if IQ3_S leaves too little room for KV and workspace:
     IQ3_XXS (75.8 GB, about 47 GB on the cards).
   - Engine: fresh upstream Niko1221/Strata (a0116b27 or newer) built from a
     clean clone, SYCL, stock. Explicit `--layer-split`, not auto (#1440).
     `STRATA_KV_WIDE_HOST=0` until #2157 is fixed; fall back to v0.1.42 on
     DEVICE_LOST. No local engine patches unless a blocker requires one.
   - Do not reuse the old /mnt/vm_8tb/github/strata pin or its build trees.

C. Optional, only after B: ISTA GSQ-RCO Coder (256 of 512 experts, 58.4 GB).

## Stages and gates

S0 Intake (no GPU). Download and hash the artifact, write a model lock, build
   Strata. Record host/driver identity once.
S1 Bring-up (GPU lease, both cards). Health before and after. One coherent
   OpenAI-compatible completion with served names `hotschmoe-dd` first and a
   detailed alias such as `qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_S-strata`.
   Gate: coherent output, tool-call round trip, clean teardown.
S2 Speed and capacity. Decode and prefill tok/s at P512, P4096, P32768;
   largest context that loads; whether 2 concurrent streams work; MTP on/off
   if the stock build supports it. Same probe on arm A for a matched table.
   Gate: none. This is a measurement, reported as-is.
S3 Quality A/B. Same harness, same agent policy, same context window, same
   thinking policy, one request at a time, both arms:
   - quick sanity: evals tier1 code (HumanEval+/MBPP+ class), both arms;
   - main: Terminal-Bench 3 with Pi on a fixed, pre-registered subset of the
     local-70 lock (target 20-24 tasks, chosen before either arm runs),
     paired per task. Endpoint failures are recorded separately from task
     failures.
S4 Only if S3 is ambiguous: extend to the full local-70 lock.

## Decision rule (set before running)

- Continue Flash-Next only if arm B beats arm A on the paired Terminal-Bench
  subset by a margin that is not explained by one or two task flips (roughly
  +10 points absolute or more, and more paired wins than losses), with no
  endpoint instability the baseline does not have.
- Tie or loss: close the Flash-Next campaign, keep the 27B, archive portable
  findings.
- If B wins: next engine work is expert-parallel (Strata PR #1798/#2205) and
  concurrency, not spillover.

## Rules for workers

- CLAUDE.md applies: bin/gpu-run for every GPU touch, health around risky
  runs, MAX_JOBS<=4 and <=90 GiB for builds, ASCII only, identity checks.
- Evidence format is one JOURNAL.md entry per stage:
  CONFIG -> COMMAND -> RESULT -> VERDICT. No admission/adjudication/receipt
  machinery, no source-plan JSON, no per-step design docs. A result is a
  number, the command that produced it, and where the raw log lives.
- Raw output under /mnt/vm_8tb/b70/results/flashnext_iq3_ab_20261013/.
- Stop and report on a hard blocker instead of building a harness around it.

## Amendment 1 (2026-10-13, user direction)

- Harness: newest Pi release through Harbor, one configuration for both arms.
- Primary score: Terminal-Bench 3 on the pre-registered 24-task subset
  (evals/terminalbench/tb3_ab24_manifest.json). Secondary: SWE-bench Verified
  30-instance subset through Harbor + Pi. Aider polyglot optional.
- Context 131072 for both arms, prefix caching on. Stretch for arm B: 200k per
  stream if it loads. If arm B cannot hold 131072 or cannot reuse a prefix
  under layer split, that is a finding and is reported as such, and both arms
  run at the largest window arm B holds.
- Report time-to-completion per task (agent wall time, turns, tokens, how the
  task ended) alongside the score; paired per-task table for the two arms.
- Terminal-Bench has never scored above 0 locally (old endpoints at 20-30
  tok/s hit the 1800 s agent timeout). A 3-task calibration on arm A, on tasks
  outside the 24, fixes the thinking policy and timeout multiplier before the
  real run. The same policy then applies to both arms.
- Arm A is one vLLM TP=2 endpoint: hotschmoe-dd.service recipe
  (vllm/int4/native5802_trial/serve.py), MTP3, calibrated fp8 KV, 200k, 4
  streams. Launch with XPU_HEALTH_TIMEOUT=400 while the SATA disk is busy.

## Amendment 2 (2026-10-13, decision eval changed)

- Terminal-Bench through Harbor has still not produced a trustworthy local
  score (calibration runs scored 0 with harness errors). The decision eval is
  now evals/abfast: 19 validated commit-replay tasks (6 from the owner's
  repos, 13 from post-July-2026 open-source commits), Pi 1.2.0 run directly
  in a throwaway container, context window 131072, thinking on, hidden tests
  graded after the agent stops. Terminal-Bench and SWE-bench stay as optional
  overnight confirmation.
- The decision rule is unchanged but applied to abfast: arm B must win by
  more than one or two task flips, i.e. at least +2 tasks net with more
  paired wins than losses, and no extra endpoint instability.
- Timeout: the first arm A run used a 900 s agent cap at 4 parallel tasks and
  11 of 19 tasks hit the cap (3 of the 4 failures were timeouts). Quality is
  therefore scored as "passes within 2700 s": arm A tasks that failed by
  timeout at 900 s are rerun at 2700 s; arm B runs at 2700 s throughout.
  Time-to-completion is reported separately and is only comparable at
  matched parallelism.
- Arm A server load in that run: 7.24 M prompt tokens (5.84 M cache hits,
  1.40 M uncached) and 0.216 M completion tokens in 4134 s. Uncached prefill
  of tool output is the larger share of server time, so prefix reuse and
  prefill speed on arm B matter as much as decode speed.
- Output cap: with thinking on and a 16384 max-output cap, one reasoning
  turn can use the whole cap and Pi ends the run (stop reason "length").
  That is what failed moreitertools-numeric-range on arm A, and the
  Terminal-Bench calibration saw the same thing. The scored protocol is
  therefore 2700 s and 32768 max output tokens: arm A's four first-pass
  failures are rerun under it, arm B runs under it throughout.
- Terminal-Bench calibration on arm A (2 tasks outside the 24): reward 0 on
  both, best partial 23 of 27 verifier tests in 50 min. Plumbing and
  SWE-bench via Harbor + Pi 1.2.0 work; runners and report are in
  evals/ab and evals/terminalbench/ab_report.py. Estimated 12-24 h per arm
  for the 24 tasks, so it stays an optional overnight confirmation.

## Outcome (2026-10-13)

- Arm B as run: ISTA GSQ-RCO IQ3_XXS on stock Strata v0.1.43.1, two 131072
  slots, MTP4, conversation parking. IQ3_S fits one stream only (200k
  proved without MTP, 131072 with MTP).
- Matched abfast (parallel 2, 2700 s, 32768 output): 18/19 each, 19 paired
  ties, same single failure. Median agent wall on tasks both passed: 526 s
  (A) vs 491 s (B).
- Decision rule result: bar not met. 27B INT4 stays the daily driver.
- What changed from the going-in assumptions: Flash-Next is not slower per
  task here. Prefill 1.3-2.0k tok/s and near-total prefix reuse offset its
  lower concurrency (2 streams at 131072 vs 4 at 200k).
- Limit of the evidence: abfast is near saturation, so a long-horizon
  quality advantage is untested. Terminal-Bench 24 through Harbor is the
  available harder eval (12-24 h per arm).
