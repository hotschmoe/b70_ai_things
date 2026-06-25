# aider polyglot harness (single-shot codegen CONTROL)

This harness runs the **Aider polyglot benchmark** as the single-shot-codegen *control* in the
4-way quant comparison. The literature predicts int4 costs only ~1-3 pp here (vs ~10-15 pp on long
agentic trajectories), so a roughly-flat `pass_rate_2` under int4 is the expected baseline that the
other harnesses (bfcl/tau2/swe) are read against.

## What it measures

The benchmark gives the model an Exercism exercise (instructions + stub solution files), asks for a
solution, runs the language's unit tests, and on failure feeds the test output back **once** (shallow
multi-turn: edit -> run tests -> fix once, i.e. `--tries 2`). Metrics:

- **`pass_rate_2`** (headline, our `score`): fraction of exercises whose tests pass within 2 tries.
- `pass_rate_1`: fraction passing on the first try.
- `percent_cases_well_formed`: fraction of cases with **no malformed edit responses** -- this is the
  quant-sensitive signal. A quant that breaks the `diff` edit format shows up here as a drop even if
  `pass_rate` looks ok. We force `--edit-format diff` precisely so this metric is live (the default
  `whole` format is almost never malformed and would make the metric blind).

## Pinned versions

| component | pin |
|---|---|
| aider repo (benchmark code + Docker image) | tag **v0.86.2**, commit `253f0368b873ba30d8ee26e463718f0c03614ddf` |
| aider-chat (host driver venv) | **0.86.2** |
| litellm (baked into the benchmark image) | **1.81.10** (from aider v0.86.2 `requirements.txt`) |
| polyglot-benchmark dataset (225 exercises) | commit `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f` |
| benchmark Docker image tag | `aider-benchmark` |

The benchmark code (`benchmark/benchmark.py`) is **not** shipped in the `aider-chat` PyPI wheel; it
only exists in the git repo, and the Docker image is built `FROM` that repo (`COPY . /aider`). So we
vendor a pinned checkout at `./vendor/aider` and build the image from it.

## The dockerized-run decision

Running the 225 polyglot exercises needs six language toolchains (g++, go, java, node, rust, python)
to execute the tests. We do **not** install those on the host. Instead the official aider benchmark
Docker image (`benchmark/Dockerfile`) bundles them, and `run.sh` runs the benchmark **inside** that
image with **`--network host`**, so `http://localhost:18080/v1` reaches the host vLLM directly.

- Custom endpoint: `OPENAI_API_BASE=$EVAL_ENDPOINT`, `OPENAI_API_KEY=dummy`, model
  `openai/$EVAL_SERVED` (standard LiteLLM openai-compatible routing).
- `AIDER_DOCKER=1` (benchmark.py refuses to run model code outside a container without it).
- Dataset bind-mounted read-only at `/benchmarks/polyglot-benchmark`; the per-run output dir is the
  writable `/benchmarks` mount under `runs/<id>/tmp.benchmarks`.
- `--user $(id -u):$(id -g)` so output files are owned by the host user and `runs/` is cleanable
  without sudo. That requires a writable `HOME=/tmp/h` and a runtime
  `git config --global --add safe.directory /aider` (benchmark.py reads the repo commit hash);
  `run.sh` wraps the benchmark invocation in `bash -lc` to set both up.

The host `.venv` (aider-chat 0.86.2) is used only to drive dataset/subset selection and `parse.py`.

## How subset selection stays STABLE across configs

`benchmark.py` selects tasks with an **unseeded** `random.shuffle(test_dnames)` then `[:num_tests]`,
so its own `--num-tests` is **not** stable across runs/configs. We therefore do **not** use it.
Instead `select_subset.py` computes the exercise set ourselves:

1. Enumerate all 225 `<lang>/exercises/practice/<exercise>` paths, `sorted()` for a canonical base.
2. One seeded `random.Random(AE_SEED).shuffle(...)` (AE_SEED=1234) -> one canonical ordering.
3. Subsets are **nested prefixes** of that ordering: `smoke=first 5`, `standard=first 50`, `full=225`.

We hand the exact selected paths to `benchmark.py` via `--keywords` (comma-joined). Its keyword
filter is `keyword in dn` over the relative path; a full path `<lang>/exercises/practice/<ex>` is
unique, so each keyword matches exactly one exercise (verified: 5/5 and 50/50 exact, no over-match).

Consequences:
- The **same** exercises run for all four configs -> `task_id`s pair for McNemar/bootstrap stats.
- `smoke` is a subset of `standard` is a subset of `full`, so a smoke task is always a member of the
  larger sets.
- `task_id` is `"<lang>/<exercise>"` (lang-qualified), **not** the bare exercise name: 58 of the 100
  exercise names recur across languages, so the bare name would collide.

| subset | exercises | language spread (seed 1234) |
|---|---|---|
| `smoke`    | 5   | js, rust, cpp, java, go (one each) |
| `standard` | 50  | all 6 (java 15, js 10, go 8, py 8, rust 6, cpp 3) |
| `full`     | 225 | all |

## Determinism knobs

`benchmark.py` has no `--temperature`/`--seed` flag; aider reads them from model settings. `run.sh`
writes a per-run `model-settings.yml` registering `openai/$EVAL_SERVED` with
`use_temperature: $AE_TEMPERATURE` (0.0 -> greedy) and `extra_params: {top_p, seed, max_tokens}`
(passed straight to the litellm completion), and forces `edit_format: diff`. Greedy + seed makes
output reproducible per request; `--threads $AE_CONCURRENCY` only affects wall-clock.

## Commands

```bash
# one-time, idempotent (creates .venv, vendors aider@v0.86.2, clones dataset, builds the image)
bash setup.sh

# run one config x subset (invoked by run/run_config.sh; endpoint must be live)
bash run.sh 27b-int4 smoke       # 5 exercises, plumbing shakeout (a couple minutes)
bash run.sh 27b-w8a8 standard    # 50 exercises (campaign size)
bash run.sh 35b-int4 full        # all 225

# standalone parse of a native run dir
.venv/bin/python parse.py runs/<id>/tmp.benchmarks/<date>--<id> --selected runs/<id>/.selected.txt
```

Outputs: `results/<label>/aider.json` (standard emit) + `results/<label>/.aider.parsed.json`.
Raw run data (per-exercise `.aider.results.json`, logs) lands under `runs/<id>/` (gitignored).

## Run time (rough, greedy, threads=4)

Per exercise is one or two model completions plus a test compile/run. Expect roughly:
- `smoke` (5): a few minutes.
- `standard` (50): ~30-60 min depending on model speed and how often the 2nd try fires.
- `full` (225): a few hours.

These are dominated by model latency on the box, not the harness.

## Caveats / residual risks for the live run

- **Endpoint reachability from the container.** `--network host` assumes the host networking driver;
  validated only offline here. If the box's docker uses a non-default network mode, fall back to
  `OPENAI_API_BASE=http://<host-ip>:18080/v1`. (TP=2 W8A8 configs serve on the same 18080.)
- **edit_format=diff vs model capability.** We force `diff` to make `percent_cases_well_formed`
  meaningful. If a heavily-quantized config produces a flood of malformed diffs, `pass_rate` will
  drop partly for format reasons, not pure reasoning -- that drop IS the signal we want, but read it
  together with `percent_cases_well_formed` (don't attribute it all to reasoning).
- **max_tokens=2048.** Inherited from the campaign determinism knobs. Some solutions (or the test
  feedback turn) may want more; if you see truncation in `runs/<id>/benchmark.log`, raise
  `AE_MAX_TOKENS`. Held constant across configs so the A/B stays fair.
- **Missing/crashed exercises = fail.** If the container crashes mid-exercise (no
  `.aider.results.json`), `parse.py` counts that task as `passed:false` so `n_tasks` equals the
  requested subset size (config-stable denominator). `extra.n_missing` reports how many. Upstream's
  own `pass_rate` divides by *completed* tests instead -- check `n_missing` before trusting a score.
- **12g container memory cap** (from upstream `docker.sh`). The C++/Java/Rust builds fit; if a test
  OOMs it shows as a fail. Raise `--memory` in `run.sh` if needed.
- **litellm seed support.** vLLM honors `seed`; if a served config ignores it, greedy (temp 0) still
  makes decoding deterministic, so pairing is unaffected.
