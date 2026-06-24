# agentic-eval -- repeatable 4-way quant comparison for Qwen3.6 serving

A self-contained, repeatable harness to answer one question for the B70 serving decision:

> **Does int4 quantization degrade real multi-step agentic coding / tool-use more than single-shot
> codegen, and by how much -- for the Qwen3.6 27B dense vs the 35B-A3B MoE?**

It serves each of four on-disk configs in turn on the box, runs a spectrum of pull-and-run agentic
harnesses against the live vLLM endpoint, and produces one scoreboard: **score, wall-clock, and token
cost per config**. The literature predicts int4 costs ~1-3 pp on single-shot codegen but ~10-15 pp on
long agentic trajectories (ACBench 2505.19433); this measures whether that holds on *our* models/box.

## The four configs (all already on the shelf -- no quantization work)

| label | arch | scheme | served id | cards | recipe |
|---|---|---|---|---|---|
| `27b-int4` | dense | W4A16 int4 (AutoRound) | `qwen36-27b-int4` | 1 | `rdy_to_serve/qwen36-27b-int4` |
| `27b-w8a8` | dense | W8A8 int8 (SmoothQuant+GPTQ)+MTP | `qwen36-27b-w8a8-sqgptq-mtp` | 2 (TP) | `rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp` |
| `35b-int4` | moe | W4A16 int4 (AutoRound) | `qwen36-35b-a3b-int4` | 1 | `rdy_to_serve/qwen36-35b-a3b-int4` |
| `35b-w8a8` | moe | W8A8 int8 (Quark) | `qwen36-35b-a3b-quark-w8a8-int8` | 2 (TP) | `rdy_to_serve/qwen36-35b-a3b-quark-w8a8-int8` |

The load-bearing contrast is **within architecture**: `27b-int4` vs `27b-w8a8`, and `35b-int4` vs
`35b-w8a8`. A dense-vs-MoE gap is a model-choice question, not a quant effect -- do not conflate them.

## The harness spectrum (single-shot -> long-horizon)

| harness | role | primary metric | why |
|---|---|---|---|
| `aider` | single-shot codegen **control** | `pass_rate_2` | should stay ~flat under int4 -- the baseline |
| `bfcl` | structured tool-use **isolator** | multi-turn accuracy | most cases => best power to call a 5 pp delta |
| `tau2` | multi-turn tool+state | `pass^1` | conversational tool-use with state drift |
| `swe` | real agentic coding | resolved % | mini-swe-agent + SWE-bench(/rebench) grader headline |

See `docs/DESIGN.md` for the experimental design (within-arch deltas, temp regimes, McNemar/bootstrap,
how many tasks make a delta real) and `docs/HARNESS_CONTRACT.md` for the per-harness interface.

## Run it

```bash
cd /mnt/vm_8tb/github/b70_ai_things

# 0) one-time: install each harness's pinned venv (uv, python 3.12) + datasets
for h in aider bfcl tau2 swe; do bash agentic-eval/harnesses/$h/setup.sh; done

# 1) plumbing shakeout on ONE config (tiny task counts, ~minutes) -- DO THIS FIRST
./bin/gpu-run bash agentic-eval/run/smoke.sh 27b-w8a8

# 2) the full campaign (serves all 4 in turn under one lease, then summarizes)
SUBSET=standard ./bin/gpu-run bash agentic-eval/run/run_all.sh

# narrower runs
CONFIGS="35b-int4 35b-w8a8" HARNESSES="bfcl" SUBSET=standard ./bin/gpu-run bash agentic-eval/run/run_all.sh

# 3) is a specific delta significant?
python3 agentic-eval/lib/stats.py --results agentic-eval/results --harness bfcl --a 35b-int4 --b 35b-w8a8
```

`run_all.sh` MUST be launched under `gpu-run` (it holds the box for the whole serial campaign so
wall-clock/token numbers are uncontended). It serves one config at a time on port 18080, runs the
harnesses, tears down (wedge-guarded), and regenerates the scoreboard below.

## Does total time / total tokens matter? (yes -- it is half the decision)

Score alone does not pick a serving config; the int4-vs-w8a8 tradeoff is **quality vs speed**. int4 is
the faster, single-card, DP=2-capable option; w8a8 is the higher-fidelity, both-cards (TP=2) option.
So the scoreboard reports, per config: the **score** (quality), the **wall-clock** to complete the
eval (a proxy for end-to-end latency under a fixed workload), and **total generation tokens + tok/s**
(throughput). That lets you read "quality per second" and "quality per token", which is exactly the
decision: if int4 is e.g. 4 pp worse but 40% faster and frees a card for DP=2, that is a real,
quantified call rather than a vibe. (Eval-wall-clock is a within-this-harness proxy, not a production
SLO; it is comparable across configs because the task set and concurrency are held fixed.)

## Results

<!-- RESULTS:START -->
_(no runs yet -- `run_all.sh` regenerates this section)_
<!-- RESULTS:END -->

## Layout

```
agentic-eval/
  configs.sh                 canonical 4-config registry + shared serve/determinism knobs
  serve/serve_config.sh      label -> rdy_to_serve recipe (+ eval overrides); no lease (caller holds it)
  run/{run_all,run_config,smoke}.sh   campaign / per-config / shakeout drivers
  lib/{common.sh,evallib.py,summarize.py,stats.py}   token accounting, emit, scoreboard, significance
  harnesses/{aider,bfcl,tau2,swe}/   one pinned uv venv each (HARNESS_CONTRACT.md)
  results/<config>/<harness>.json    standard per-run records -> results/{scores.json,SUMMARY.md}
  docs/{DESIGN.md,HARNESS_CONTRACT.md}
```
