# Experimental design

Why these harnesses, this matrix, and this analysis. Condensed from the model/quant investigation
(repo map + Qwen3.6 literature + a codex consult on eval design).

## The question, stated to be falsifiable

H1: int4 (W4A16) degrades **long multi-step agentic** trajectories more than **single-shot codegen**,
relative to W8A8/int8. Literature anchor: ACBench (arXiv:2505.19433) reports 4-bit "preserves workflow
generation and tool use (1-3% drop) but degrades real-world application accuracy by 10-15%"; W8A8/int8
is near-lossless (~0.2-1%). Mix-Quant attributes the gap to decode-stage quant error accumulating over
long trajectories. If H1 holds on our models, the spectrum shows a small int4 hit on `aider` and a
larger one on `bfcl`/`tau2`/`swe`.

## Within-architecture only

The four configs are two architectures x two schemes. Quant effects are read **within** an arch:

- dense: `27b-int4` - `27b-w8a8`
- moe:   `35b-int4` - `35b-w8a8`

A dense-vs-MoE difference is a model-choice question (already addressed: the 27B dense wins quality
benchmarks; the 35B-A3B wins throughput) and must NOT be read as a quantization effect. MoE quant
sensitivity is special (router/gate/shared-expert precision drives it), which is exactly why the
within-arch MoE pair is its own contrast.

## The spectrum (each harness has a distinct job)

1. `aider` polyglot -- **control.** 225 Exercism edit tasks, 6 languages, shallow multi-turn
   (edit -> run tests -> fix once). Reports `pass_rate_2` and `percent_cases_well_formed` (catches
   quant-induced diff-format breakage). H1 predicts this barely moves under int4.
2. `bfcl` multi-turn -- **isolator.** Hundreds of structured tool-call cases (base/miss_func/
   miss_param/long_context). Cleanest failure attribution: invalid JSON / wrong function / wrong arg /
   missing arg / premature-final / no-op loops. Best statistical power for a small delta.
3. `tau2` retail/airline -- **multi-turn tool + state.** `pass^k` compares final DB state to a goal;
   a user-simulator drives the conversation. Hold the user-sim model FIXED across configs (else both
   sides change). Surfaces policy/state drift over a real dialogue.
4. `swe` (mini-swe-agent + SWE-bench/rebench grader) -- **real agentic coding headline.** Resolved %.
   Confounded by agent scaffold/localization, so it is the downstream confirmation, not the isolator.
   Use rebench (decontaminated) where possible since Qwen3.6 may have SWE-bench Verified in training.

## How many tasks make a delta real (power)

From the codex consult (paired, same tasks, randomized order):

- aider 225 greedy: good for ~8-10 pp deltas; weak at 5 pp.
- bfcl (hundreds-thousands of cases): best shot at calling a ~5 pp structured delta real.
- tau2 89-ish tasks x 1 seed: only large deltas; x3-5 seeds for ~6-10 pp.
- swe verified/rebench slice: statistically OK but attribution is the weakest.

Lead conclusions with `bfcl`; treat `swe`/`tau2` single-seed deltas as directional unless large.

## Determinism + analysis

- Primary regime: `temperature=0, top_p=1, seed=1234`. Greedy isolates quant from sampling noise.
  Scores are concurrency-invariant under greedy, so we run at a fixed concurrency (4) purely for
  wall-clock/throughput -- the score is unaffected.
- Stress regime (future, not v1): `temperature=0.2 x 3-5 seeds` on a subset, to test whether int4
  *widens brittleness* under realistic sampling.
- Significance: `lib/stats.py` pairs per-task pass/fail by `task_id`, runs McNemar's exact test on the
  discordant pairs + a paired bootstrap CI on the accuracy difference. Report the CI, not just the point.
- Also worth reporting (future): output flip-rate vs the W8A8 baseline ("Accuracy is Not All You Need").

## Thinking mode (the primary axis)

Qwen3.6 is a hybrid reasoner. `EVAL_THINKING` (configs.sh) selects the regime; default `on`.

- **thinking-on (default)** -- the real agentic-coding workload: the model emits `<think>` traces before
  tool calls / edits. Serve runs the `qwen3` reasoning parser + prefix caching (PREFIXCACHE=1). Sizing:
  MAXLEN 65536 (64k, fp16 KV -- no fp8 hack), max_tokens 8192, MAXSEQS 1, concurrency 1. The bigger budget
  is mandatory -- the 2026-06-25 smoke showed aider at MAXLEN 16384 / max_tokens 2048 hitting
  `exhausted_context_windows` mid-think and scoring an artifactual 0.0 (a sizing artifact, not a capability
  or plumbing result); 32k still exhausted a few hard cases, so 64k. 64k x 1 seq fits the tight single-card
  27B-int4 (identical KV footprint to the 32k x 2 seq config the smoke already fit). Prefix caching recovers
  most of the concurrency-1 wall-clock on multi-turn harnesses (shared growing prefix skips re-prefill).
- **thinking-off (`EVAL_THINKING=off`)** -- a deliberate SECOND axis (faster, far fewer tokens; some
  models hold up well without thinking). Serve drops the reasoning parser; sizing reverts to MAXLEN 16384
  / max_tokens 2048 / MAXSEQS 4. STATUS: EXPERIMENTAL -- fully suppressing Qwen3.6 thinking needs the
  no-think switch (`enable_thinking=false` / `/no_think`), which is not yet threaded through all four
  harnesses, so off-mode numbers are not trustworthy until a live no-think run is validated. `EVAL_NO_THINK`
  is exported for harnesses to honor once wired.

Compare within a single thinking regime; do not mix thinking-on and thinking-off scores in one delta.
Every emitted result records `meta.thinking` / `meta.max_len` / `meta.max_tokens` so runs are auditable.

## Serving choices that affect comparability

- One config served at a time on port 18080 (serial) so wall-clock/token numbers are uncontended.
- MAXLEN/MAXSEQS/max_tokens are set by the thinking regime (above) and held constant across all four
  configs. The thinking-on MAXSEQS 2 is bounded by the tightest config (single-card fp16-KV 27B int4 at
  32k). Long SWE trajectories may still truncate, but identically across configs, so the A/B stays fair.
- Each recipe keeps its shipped, verified defaults (GRAPH/MTP/PUSH_AR/KV dtype) -- the realistic "what
  you would actually serve" config -- so the speed numbers are honest rather than handicapped to match.
- MTP on `27b-w8a8` is lossless at temp=0 (speculative tokens are accepted only if they equal the target
  greedy choice), so it changes speed, not score.

## Known limitations

- `tau2` needs a fixed external user-sim model; if none is configured it is skipped (documented in its
  README) rather than run with a confounded same-model user-sim.
- Eval wall-clock is a proxy for latency under a fixed synthetic load, not a production SLO.
- `swe` resolved% at MAXLEN=16384 will be lower than a long-context serve; it is a relative number here.
