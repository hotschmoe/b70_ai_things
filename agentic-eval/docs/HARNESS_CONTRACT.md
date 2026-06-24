# Harness integration contract

Every harness lives in `agentic-eval/harnesses/<name>/` and exposes the same small interface so the
orchestrator can drive all four identically and the summarizer can build one table. Implement these
files; nothing else is required.

```
harnesses/<name>/
  setup.sh        # idempotent: create .venv (uv, python 3.12), install PINNED deps, fetch dataset
  run.sh          # bash run.sh <config_label> <subset>   (the contract below)
  parse.py        # native harness output -> parsed.json (schema below)
  .venv/          # uv venv (gitignored)
  README.md       # what it measures, the exact upstream version, how to run standalone, caveats
```

## Environment your `run.sh` receives

`run.sh` is invoked as `bash run.sh <CONFIG_LABEL> <SUBSET>` from `run_config.sh`, which has already
sourced `lib/common.sh` and called `ae_set_config`. Source common.sh yourself at the top to get:

```bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)/lib/common.sh"
ae_set_config "$1"        # sets the EVAL_*/AE_* vars below for this config
SUBSET="${2:-standard}"
```

| var | meaning |
|---|---|
| `EVAL_SERVED` | served-model-id to send as the `model` field (e.g. `qwen36-27b-int4`) |
| `EVAL_ENDPOINT` | `http://localhost:18080/v1` (OpenAI-compatible) |
| `EVAL_BASE_URL` | `http://localhost:18080` |
| `EVAL_PORT` | `18080` |
| `EVAL_ARCH` / `EVAL_SCHEME` | `dense`/`moe`, `int4`/`w8a8` (for labeling only) |
| `RESULTS_DIR` | `agentic-eval/results/<label>` (write your JSON here) |
| `AE_TEMPERATURE` `AE_TOP_P` `AE_SEED` `AE_MAX_TOKENS` | determinism knobs: 0.0 / 1.0 / 1234 / 2048 |
| `AE_CONCURRENCY` | fixed request concurrency (4) -- greedy => scores are concurrency-invariant |
| `AE_ROOT` `AE_LIB` | `agentic-eval/` and `agentic-eval/lib/` |

No API key is needed (local vLLM); pass `dummy`/`EMPTY` where a key is required.

## Subset levels (your `run.sh` must honor `$SUBSET`)

- `smoke`    -- 3-5 tasks. Must finish in a couple minutes. For plumbing shakeout.
- `standard` -- the default campaign size (tractable, meaningful). Suggested per harness:
  aider ~50, bfcl `multi_turn` full (~800), tau2 retail ~20, swe rebench/verified slice ~20.
- `full`     -- the complete benchmark (aider 225, bfcl all multi_turn, tau2 retail 114, swe 100+).

## What `run.sh` must do (the wrapper)

```bash
B=$(ae_snap); START=$(ae_now)
#  ... run the harness in its .venv against $EVAL_ENDPOINT, model $EVAL_SERVED,
#      temperature $AE_TEMPERATURE, concurrency $AE_CONCURRENCY, the $SUBSET task set ...
END=$(ae_now); A=$(ae_snap)
python parse.py <native-output-dir> > "$RESULTS_DIR/.<name>.parsed.json"
"$AE_PY" "$AE_LIB/evallib.py" emit \
  --config "$EVAL_LABEL" --harness <name> --subset "$SUBSET" --served "$EVAL_SERVED" \
  --parsed "$RESULTS_DIR/.<name>.parsed.json" \
  --tok-before "$B" --tok-after "$A" --start "$START" --end "$END" \
  --out "$RESULTS_DIR/<name>.json" --meta "temperature=$AE_TEMPERATURE"
```

`ae_snap` / `ae_now` come from common.sh. Token accounting is automatic via vLLM `/metrics` -- you do
not compute tokens yourself.

## `parsed.json` schema (what `parse.py` prints to stdout)

```json
{
  "score": 0.82,                 // primary metric in [0,1] (pass rate / resolved rate / accuracy)
  "score_name": "pass_rate_2",   // short name of that metric
  "n_tasks": 50,
  "per_task": [                  // REQUIRED for the paired McNemar/bootstrap stats
    {"task_id": "exercise-name-or-instance-id", "passed": true},
    {"task_id": "...", "passed": false}
  ],
  "extra": {"well_formed_pct": 0.99, "any_secondary_metric": 1.23}
}
```

`task_id` must be STABLE across configs (same task -> same id) so `lib/stats.py` can pair them.

## Rules

- Pin every dependency to an exact version in `setup.sh` (the research found current ones; record them
  in your README). Repeatability is the whole point.
- `setup.sh` must be idempotent and must NOT touch system python (use `uv venv --python 3.12`).
- Do not require the GPU or a live endpoint to *install*. The live run happens later under the lease.
- Greedy/temp=0 is the primary regime. (A temp=0.2 stress pass is a later add-on, not v1.)
- Do not `git commit`. The orchestrator owner commits.
- Sandbox containers (swebench grader, mini-swe task env, aider `--docker`) run on the HOST docker
  daemon -- you are on the host (uv venv), so just use docker normally.
