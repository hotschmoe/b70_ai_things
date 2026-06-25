# tau2 harness -- tau2-bench (multi-turn tool-use, user-simulator)

Measures **multi-turn, tool-using agent quality** on tau2-bench's `retail` domain: the
model-under-test acts as a customer-service agent, calls domain tools, and must satisfy a customer
whose turns are produced by a SEPARATE user-simulator LLM. Each task has a programmatic
`evaluation_criteria`; a task earns `reward == 1.0` iff all DB writes / required actions / response
checks pass. Our primary score is **pass^1** (the upstream pass^k with k=1): the fraction of tasks
solved in a single greedy (temp=0) trial.

This is the agentic counterpart to single-shot codegen: it exercises whether int4 vs w8a8 degrades
*trajectories* (tool-call formatting, state tracking, recovery) more than one-shot accuracy.

## Upstream version (PINNED)

- Repo: <https://github.com/sierra-research/tau2-bench> (the older `tau-bench` redirects here; this
  is **tau2**). Actively maintained.
- **Pinned commit: `8ebb7499622fc2be9b9d510d6f7a7653461f4f29`** (2026-06-22), package `tau2==1.0.0`.
- Installed editable into a uv python-3.12 venv (tau2 requires `>=3.12,<3.14`).
- Domain task data is bundled in the clone under `src/tau2-bench/data/tau2/domains/` and located via
  the `TAU2_DATA_DIR` env var (run.sh sets it). retail = **114** tasks (ids `0`..`113`), airline = 50.

## Setup

```bash
cd agentic-eval/harnesses/tau2
bash setup.sh         # idempotent: uv venv py3.12, clone @ pinned commit, uv pip install -e
```
Prints `DONE tau2 version=1.0.0 pinned_commit=8ebb74...` and confirms the run flags exist.

## Run

```bash
bash run.sh <CONFIG_LABEL> <SUBSET>
# CONFIG_LABEL in: 27b-int4 27b-w8a8 35b-int4 35b-w8a8   (from configs.sh)
# SUBSET in:       smoke | standard | full
```

Subset -> retail task count (single trial, greedy, pass^1):

| subset   | tasks | trials | notes |
|----------|-------|--------|-------|
| smoke    | 3     | 1      | plumbing shakeout, ~minutes |
| standard | 20    | 1      | default campaign size |
| full     | 114   | 1      | complete retail domain |

`TAU2_DOMAIN=airline` (etc.) overrides the domain. Task ids are stable across configs (the tau2 task
id, e.g. `"7"`), so `lib/stats.py` can pair runs for McNemar/bootstrap.

### Rough run time

Each task is a multi-turn conversation (default up to 200 steps; typically a handful of agent +
user-sim round-trips). At concurrency 4 against a local vLLM B70 serve: smoke ~2-5 min,
standard ~15-30 min, full ~1.5-3 h -- dominated by the agent-side decode and the user-sim latency.

## THE USER-SIMULATOR REQUIREMENT (read this -- it is a real confound)

tau2 runs TWO LLMs per conversation:
- `--agent-llm` = the **model-under-test** (our served quant), and
- `--user-llm`  = the **user-simulator** that plays the customer.

For a clean quant A/B the user-sim **MUST be a single FIXED model held constant across all four
configs**. It must **NOT** be the model-under-test: if the user-sim were also the served quant, then
switching int4 -> w8a8 would change BOTH sides of every conversation, and you could no longer
attribute a score delta to the agent. That is a confound, not a measurement.

Because we may not have an external API key on the box, run.sh makes the user-sim explicit and
**refuses to guess**:

- Configure it via env:
  - `USER_SIM_MODEL`     -- litellm model id for the fixed user-sim (e.g. `gpt-4.1`,
    `anthropic/claude-...`, or `openai/<some-served-id>` on a DIFFERENT, fixed endpoint).
  - `USER_SIM_BASE_URL`  -- (optional) `api_base` if the user-sim is an OpenAI-compatible endpoint.
  - `USER_SIM_API_KEY`   -- (optional) its key. For cloud providers you may instead export the
    provider's standard key env var (e.g. `OPENAI_API_KEY`) and leave this unset.
- If `USER_SIM_MODEL` is **unset**, run.sh **SKIPS cleanly**: it emits a result JSON with
  `score=null` and `extra.skipped="no fixed user-sim configured (USER_SIM_MODEL unset)"` via
  `evallib.py emit`, so the scoreboard shows tau2 as *not-run* rather than crashing. It does **NOT**
  silently fall back to the model-under-test.

Recommended: pick ONE strong, cheap, deterministic cloud model you have access to (e.g. a GPT-4.1 /
Claude tier) as `USER_SIM_MODEL` and use the **same** id for all four configs. Run all four in the
same campaign so the user-sim is identical. Example:

```bash
export USER_SIM_MODEL=gpt-4.1
export OPENAI_API_KEY=sk-...        # user-sim provider key (NOT the local vLLM)
for c in 27b-int4 27b-w8a8 35b-int4 35b-w8a8; do
  # (serve config $c first, under the GPU lease) then:
  bash run.sh "$c" standard
done
```

## Endpoint / arg path (SOURCE-VERIFIED -- smoke-test first)

tau2's `--agent-llm-args` is a JSON string parsed with `json.loads` and forwarded **verbatim** to
`litellm.completion(...)`. Verified in `src/tau2/utils/llm_utils.py::generate()` (it splats
`**kwargs` straight into `completion()`), reached from `src/tau2/agent/llm_agent.py` which splats
`self.llm_args`. This is **not** documented in the upstream README, so it is treated as load-bearing
and smoke-tested. run.sh sends:

```text
--agent-llm      openai/$EVAL_SERVED
--agent-llm-args {"api_base":"$EVAL_ENDPOINT","api_key":"EMPTY","temperature":0.0,
                  "top_p":1.0,"max_tokens":2048,"seed":1234}
```

The `openai/` prefix routes litellm's OpenAI-compatible provider; `api_base` points it at the local
vLLM `/v1`. Determinism knobs come from `configs.sh` (`AE_TEMPERATURE/TOP_P/MAX_TOKENS/SEED`).

### Token accounting caveat

`ae_snap` reads vLLM `/metrics`, which only counts traffic to the **local** serve. The token delta
recorded for a tau2 run therefore reflects the **AGENT side only** -- the user-sim runs on a
separate (cloud) endpoint and is correctly NOT counted. That is what we want: the cost number is the
model-under-test's cost. (If you ever point the user-sim at the SAME local vLLM, the delta would
double-count and the A/B would be confounded anyway -- don't.)

## Output

- Native tau2 results: `$TAU2_DATA_DIR/simulations/ae_<label>_<domain>_<subset>/results.json`
  (tau2 hardcodes `<data>/simulations/<save_to>/`); a copy + the run log land in `runs/` (gitignored).
- `parse.py <results.json>` -> `results/<label>/.tau2.parsed.json` (HARNESS_CONTRACT schema:
  `score`=pass^1, `score_name`="pass^1", `per_task[{task_id, passed}]`,
  `extra={avg_reward, pass^2, num_trials, n_simulations}`).
- `evallib.py emit` -> `results/<label>/tau2.json` (canonical per-(config,harness) result).

`passed` per task = `reward == 1` (single-trial) / passed-all-trials (multi-trial). pass^k uses the
upstream formula `C(success,k)/C(trials,k)` averaged over tasks.

## Caveats / residual risks

- **User-sim is the whole ballgame.** Results are only comparable if the SAME fixed `USER_SIM_MODEL`
  was used for every config in the campaign. The emitted JSON records `meta.user_sim` -- check it
  matches across the four `results/*/tau2.json` before trusting any delta.
- **Greedy user-sim is still stochastic in practice.** Even at temp=0 a cloud user-sim can drift
  between runs (provider nondeterminism); the agent A/B is still valid because the user-sim is held
  constant per campaign, but absolute pass^1 may wobble run-to-run. The greedy regime is primary.
- **`MAX_TOKENS=2048` + `MAXLEN=16384`** (campaign-wide, identical across configs): a few long retail
  trajectories may hit the step/context cap and fail for reasons unrelated to the quant. Identical
  across configs, so the A/B stays fair, but absolute pass^1 is a floor.
- **Tool-call parser**: the serve must have tool-calling enabled (configs.sh forces
  `EVAL_TOOLCALL=1` + `qwen3_coder` parser). If a config serves without a working tool parser, tau2
  agents will fail to emit tool calls and score ~0 -- that is a serve-config bug, not a model result.
  Verify with the smoke subset first.
- **Infra errors score 0, not null.** If the endpoint is unreachable mid-run, those tasks get
  `reward_info=null` -> `passed=false` -> they drag pass^1 down rather than aborting. A *fully* failed
  run (rc!=0 or no results.json) emits `score=null`. Inspect `runs/*.log` if pass^1 is unexpectedly 0.
- **`litellm` is pinned by tau2** (`>=1.80.15,<1.82.7`); resolved version is recorded in the venv. If
  upstream tau2 bumps it, re-pin the commit and re-run setup.sh.
