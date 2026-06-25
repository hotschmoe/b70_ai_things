# swe harness -- mini-swe-agent + SWE-bench Verified grader

Real agentic coding, the long-horizon end of the spectrum. The model drives a per-instance
docker container by emitting shell, tries to fix a real GitHub issue, submits a git patch, and the
official SWE-bench grader runs the repo's hidden tests in a clean eval container. Headline metric:
**resolved %** (fraction of attempted instances whose patch makes the tests pass).

## What it measures

`score = resolved / n_submitted`, `score_name = "resolved"`. `n_submitted` is the number of instances
in the requested slice (what mini attempted). A submitted instance the grader cannot resolve -- failed
tests, empty patch, or a harness error -- counts as a fail. `task_id == instance_id`, taken in dataset
order with no shuffle, so the same slice gives the same ids across all four quant configs and
`lib/stats.py` can pair them for McNemar / bootstrap.

## Versions (pinned in setup.sh)

| component | version | role |
|---|---|---|
| `mini-swe-agent` | **2.4.2** (PyPI) | trajectory generator: model emits bash, runs in a docker container |
| `swebench` | **4.1.0** (PyPI) | official grader: per-instance eval image, runs hidden tests, writes report json |
| python | 3.12.13 (uv venv) | isolated from system python |

## Dataset: SWE-bench Verified (not rebench), and why

`princeton-nlp/SWE-bench_Verified`, split `test`, 500 human-validated instances. Chosen over
`nebius/SWE-rebench` (decontaminated) for this campaign because:

- It is the **field-standard headline number** -- every model card / paper reports Verified, so our
  4-way quant deltas are directly interpretable against the literature.
- Its eval **docker images are public and prebuilt** (`docker.io/swebench/sweb.eval.x86_64.*`), so a
  slice grades reliably without building images from scratch.
- Verified is human-curated (solvable, well-specified), which lowers task noise -- exactly what you
  want when the question is a 5-10 pp *quant* delta, not absolute capability.

Decontamination (train-on-test leakage) is a real concern for *absolute* scores, but this campaign is
a **within-family A/B** (int4 vs w8a8 of the *same* base model): both arms see the identical
(possibly-contaminated) tasks, so leakage cancels in the delta. If a later run needs a clean absolute
number, switch the subset to rebench -- mini supports `--subset rebench` and the grader takes
`--dataset_name nebius/SWE-rebench` (the run.sh `SUBSET_NAME`/`--dataset_name` pair is the only change).

## Bash-only (text) mode -- no tool-parser confound

mini runs in **text/backticks** mode (`-c swebench_backticks.yaml` + `model.model_class:
litellm_textbased`): the model emits ```` ```mswea_bash_command ... ``` ```` fenced shell parsed from
plain text, NOT vLLM native tool-calls. So a quant's resolved% reflects coding ability, not whether
the XML/JSON tool parser fired -- the right isolation for the quant comparison. (The shared
`EVAL_TOOLCALL=1` serve setting still applies for bfcl/tau2; this harness simply does not use it.)

## Run it standalone

```bash
# one-time install (idempotent; ~6s warm, prints versions; prefetches Verified metadata)
bash agentic-eval/harnesses/swe/setup.sh

# then, with a config's endpoint already live on :18080 (run_all.sh handles the serve+lease):
bash agentic-eval/harnesses/swe/run.sh 27b-w8a8 smoke      # 3 instances
bash agentic-eval/harnesses/swe/run.sh 27b-w8a8 standard   # 20 instances
bash agentic-eval/harnesses/swe/run.sh 27b-w8a8 full       # 100 instances
```

`run.sh <config_label> <subset>` sources `lib/common.sh`, renders `config.yaml.tmpl` with the live
`$EVAL_SERVED` / `$EVAL_BASE_URL` / determinism knobs, runs `mini-extra swebench` -> `preds.json`,
grades with `swebench.harness.run_evaluation` -> report json, then `parse.py` + `evallib.py emit`
produce `results/<label>/swe.json`. Raw outputs (predictions, trajectories, grader logs, the rendered
config) land under `runs/<label>/<subset>-<stamp>/` (gitignored via `harnesses/*/runs/`).

### Subset -> slice (stable, dataset order)

| subset | slice | instances |
|---|---|---|
| `smoke` | `0:3` | 3 |
| `standard` | `0:20` | 20 |
| `full` | `0:100` | 100 |

### Single-instance sanity (no grading)

```bash
OPENAI_API_KEY=dummy .venv/bin/mini-extra swebench-single \
  --subset verified -i astropy__astropy-12907 -c swebench_backticks.yaml -c <rendered config.yaml>
```

## Cost: disk / time / image pulls

- **Grader docker images:** each instance pulls a prebuilt eval image
  (`swebench/sweb.eval.x86_64.<id>`), commonly ~0.5-2 GB each. Budget **~120 GB** for the `full`
  (100-instance) run if every image is distinct; `--cache_level env` keeps the (shared) base/env
  layers and drops per-instance layers between runs to bound disk. ~6.5 TB free on `/mnt/vm_8tb`, so
  this is comfortable.
- **Time:** dominated by (a) per-instance image pull on first touch and (b) model trajectory length.
  A smoke (3) run is minutes once images are cached; standard (20) and full (100) are mostly bounded
  by how long the model takes per trajectory at temp 0 with `step_limit=250`.
- **mini docker:** mini also starts a container per instance (the SWE-bench eval image, reused) to run
  the model's shell. Same images as the grader, so no extra pull.
- **Concurrency:** both mini (`--workers`) and the grader (`--max_workers`) use `AE_CONCURRENCY` (4).
  Greedy/temp=0 -> per-instance result is concurrency-invariant; concurrency only moves wall-clock.

## MAXLEN=16384 caveat

The campaign serves all four configs at `EVAL_MAXLEN=16384` (the largest context that fits the
tightest config -- 27B dense int4 single-card fp16-KV). SWE trajectories are the longest in the
spectrum and **can hit this ceiling**: once the running transcript exceeds the context window, earlier
turns fall out / the request truncates, which can cap how much a model can accomplish on a hard
instance. This is a real ceiling on absolute resolved%. It is **identical across all four configs**,
so the within-architecture int4-vs-w8a8 delta stays fair -- but do not read this harness's absolute
resolved% as a leaderboard number; read the *delta*. (mini's `step_limit=250` and `cost_limit` are not
the binding constraint here; for a local model cost is 0 and the context window is the practical cap.)

## Files

```
setup.sh           idempotent install (pinned mini 2.4.2 + swebench 4.1.0), prefetch Verified
config.yaml.tmpl   mini model config template; @@SERVED@@/@@BASE_URL@@/@@TEMPERATURE@@/... substituted by run.sh
run.sh             bash run.sh <config_label> <subset>  -- generate -> grade -> parse -> emit
parse.py           grader report + preds -> parsed.json (resolved fraction, per_task, extras)
.venv/  data/  runs/   gitignored (venv, HF dataset cache, raw per-run outputs)
```

## Validation done (offline / no vLLM endpoint)

- `setup.sh` clean + idempotent; both CLIs (`mini-extra swebench`, `swebench.harness.run_evaluation`)
  resolve; Verified test split (500) cached.
- `config.yaml.tmpl` renders to valid YAML; mini's config loader merges it onto `swebench_backticks.yaml`
  and selects `LitellmTextbasedModel` (bash/text mode, no tool-calls) with the `hosted_vllm/` route and
  temp/max_tokens wired.
- `parse.py` verified against a synthesized grader report (happy path: resolved fraction, per_task,
  avg_turns from a trajectory) and the no-report graceful-degrade path; full `parse.py -> evallib.py emit`
  chain produces a schema-correct `swe.json`.
- **Gold-grade docker pipeline:** ran `--predictions_path gold` on 1 Verified instance
  (`astropy__astropy-12907`) end-to-end on the host docker daemon to prove image pull + reference-patch
  apply + hidden-test run + report json all work on this box. (See the run report in git history / notes;
  re-run with the single-instance gold command above to reconfirm.)

## Residual risks / friction

- The full live A/B needs the vLLM endpoint up (not available during harness build); only an
  endpoint-dependent smoke can confirm mini<->vLLM wire format end-to-end.
- mini's text mode expects the model to consistently emit a single ```` ```mswea_bash_command ```` block;
  a weak quant that mis-formats will burn steps on `format_error` retries (counted as the run's cost,
  fair across configs).
- Grader disk grows with distinct instances; `--cache_level env` bounds it but a very large `full`
  sweep across all 4 configs still pulls many images -- monitor `/mnt/vm_8tb` if expanding past 100.
