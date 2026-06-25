# BFCL multi_turn harness (agentic-eval)

Measures **multi-turn agentic tool-calling accuracy** of a served Qwen3.6 config using the
Berkeley Function Calling Leaderboard (BFCL) `multi_turn` category, driven against our
ALREADY-RUNNING vLLM OpenAI-compatible endpoint (no in-harness serving).

## Exact upstream version

- Package: `bfcl-eval==2026.3.23` (PyPI). Pinned in `setup.sh`.
- Dataset version baked into that package: **BFCL v4** (`VERSION_PREFIX = "BFCL_v4"`). The
  multi_turn data ships INSIDE the wheel (`bfcl_eval/data/BFCL_v4_multi_turn_*.json` +
  `data/possible_answer/...`), so there is **no separate dataset download** -- `setup.sh` only
  verifies the four files are present.
- Extra pinned dep: `soundfile==0.13.1`. BFCL imports `qwen_agent` at CLI startup; `qwen_agent`
  imports `soundfile` lazily and crashes the whole CLI if it is missing. Not declared by BFCL, so
  we install it explicitly.
- We do NOT install the `oss-eval-vllm` extra (it pins `vllm==0.8.5` for LOCAL serving). We use
  `--skip-server-setup` against our own endpoint, so that heavy extra is unnecessary.

## The MODEL_KEY / handler decision (the load-bearing part)

BFCL resolves `--model <KEY>` against a STATIC internal registry
(`bfcl_eval.constants.model_config.MODEL_CONFIG_MAPPING`) that picks the prompt format + tool-call
parser (the "handler") AND the HuggingFace id sent as the OpenAI `model` field. Our served ids
(`qwen36-27b-int4`, etc.) are not in that registry, and there is **no user-config hook** to add one.

Two existing Qwen3 key families exist:

| Key family | Handler | Path | Usable here? |
|---|---|---|---|
| `qwen3-32b`, `qwen3-30b-a3b-instruct-2507`, ... | `QwenAPIHandler` | Alibaba DashScope API (needs `QWEN_API_KEY`) | No |
| `Qwen/Qwen3-32B`, `Qwen/Qwen3-30B-A3B-Instruct-2507`(`-FC`) | `QwenHandler` / `QwenFCHandler` | OSS local handler (`base_oss_handler.OSSHandler`) | Yes, but `model=` field = `Qwen/Qwen3-...` |

The OSS handlers are perfect (they call `/v1/completions`, build the Qwen3 prompt themselves,
embed tools as `<tools>...</tools>`, and parse `<tool_call>{...}</tool_call>` XML out of the raw
completion -- i.e. BFCL "prompt-FC" mode that works against ANY plain completions endpoint). The
**only** problem: they send `model=<the registry HF id>`, which our vLLM rejects because it serves
`qwen36-27b-int4`.

**Decision: register a one-entry model config at runtime whose KEY == our served id**, reusing the
stock `QwenFCHandler`. This is done by `sitecustomize.py` (placed on `PYTHONPATH`, so it runs at
interpreter startup INCLUDING inside the `bfcl` subprocess). It reads `BFCL_REGISTER_MODEL`
(=`$EVAL_SERVED`) and inserts:

```
MODEL_CONFIG_MAPPING[$EVAL_SERVED] = ModelConfig(
    model_name=$EVAL_SERVED,          # <-- sent as the OpenAI `model` field; now matches vLLM
    model_handler=QwenFCHandler,      # prompt-FC: <tool_call> XML in/out (Qwen3 native format)
    is_fc_model=True, underscore_to_dot=False, ...)
```

Why `QwenFCHandler` (FC) and not the plain `QwenHandler` (prompt): FC injects the tool schemas as
`<tools>` and parses `<tool_call>` XML -- which is exactly Qwen3.6's native tool-call format and
matches what our server emits (we verified the parser handles multi-call + dotted names like
`spotify.play`). Set `BFCL_REGISTER_FC=0` to fall back to the plain-prompt handler.

Note on NATIVE vLLM tool-calling: our server runs `--enable-auto-tool-choice --tool-call-parser
qwen3_coder`, but BFCL's OSS path uses `/v1/completions` (not `/v1/chat/completions`) and does its
own prompt/parsing, so vLLM's native parser is simply not exercised by this harness. That is fine
and is the standard BFCL way to score an OSS model; the model still produces the same `<tool_call>`
text either way.

### Endpoint + tokenizer env (confirmed from `base_oss_handler.py`)

- `REMOTE_OPENAI_BASE_URL` -> our `$EVAL_ENDPOINT` (the `.../v1` root). **Confirmed** var name.
- `REMOTE_OPENAI_API_KEY` -> `dummy`. Confirmed.
- `--skip-server-setup` -> use the existing endpoint (do not spin up vLLM). Confirmed flag.
- Even with `--skip-server-setup`, the handler's `spin_up_local_server` still (a) loads a
  **tokenizer** to cap `max_tokens`, and (b) polls `GET {base}/models` until 200. So:
  - `REMOTE_OPENAI_TOKENIZER_PATH` -> the on-disk model dir for the config (so it does NOT try to
    download `qwen36-27b-int4` from HF). `run.sh` maps served-id -> local dir and sets
    `HF_HUB_OFFLINE=1`. The tokenizer is used ONLY for a token-count cap, so any Qwen3.6 tokenizer
    works; we use the exact per-config one.
  - Our real vLLM serves `/v1/models`, so the readiness poll passes immediately.
- `BFCL_PROJECT_ROOT` -> redirects BFCL's `result/`, `score/`, `.env`, id-file out of site-packages
  into `harnesses/bfcl/work/<config>/`. Confirmed env var.

## Commands (what `run.sh` does)

```
bash run.sh <config_label> <smoke|standard|full>
```

Per subset:
- generate: `bfcl generate --model $EVAL_SERVED --test-category multi_turn --skip-server-setup
  --num-threads $AE_CONCURRENCY --temperature $AE_TEMPERATURE -o [--run-ids]`
- evaluate: `bfcl evaluate --model $EVAL_SERVED --test-category multi_turn [--partial-eval]`
- then `parse.py <score_dir> --result-dir <result_dir>` -> `.bfcl.parsed.json`, then the standard
  `evallib.py emit` (token/wall accounting) -> `results/<label>/bfcl.json`.

## What the metric means

- `score` / `score_name = multi_turn_acc`: the **unweighted (macro) mean of the four subcategory
  accuracies** -- exactly BFCL's official `multi_turn` overall (`calculate_unweighted_accuracy` over
  `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`).
  For a full run each subcat is 200 cases, so macro == micro; `extra.overall_acc_micro` carries the
  case-weighted figure too. A case "passes" only if the model produces the correct sequence of tool
  calls AND the resulting simulated API state matches ground truth across ALL turns -- this is a
  strict, execution-checked, multi-step metric (the point of using BFCL vs single-shot codegen).
- `extra.subcategories`: per-subcat accuracy/correct/total.
- `extra.failure_types`: structured failure breakdown -- the whole reason to use BFCL. Keys seen in
  multi_turn:
  - `multi_turn:force_terminated` (hit the 20-step cap without resolving),
  - `multi_turn:inference_error` (model returned a non-list / broken response),
  - `multi_turn:empty_turn_model_response` (emitted no tool call when one was needed),
  - `multi_turn:instance_state_mismatch` (wrong end state),
  - `multi_turn:execution_response_mismatch` (right calls, wrong/extra results),
  - `multi_turn:method_invoke_order_mismatch` (right calls, wrong order),
  - `multi_turn:irrelevance_error:decoder_success` (called a tool when it should not).
  `extra` rolls these into `force_terminated_rate`, `inference_error_rate`, and
  `wrong_func_or_state_rate` (over the full task set).
- `per_task`: one `{task_id, passed}` per case. `task_id` is the BFCL case id
  (`multi_turn_base_0`, ...), STABLE across configs so `lib/stats.py` can pair runs. (BFCL's score
  file lists only FAILED entries; passing ids are recovered from the result file via `--result-dir`.)

## Subset sizes

| subset | selection | cases |
|---|---|---|
| smoke | fixed id slice via `--run-ids` (3 base + 1 miss_func + 1 miss_param) | 5 |
| standard | full `multi_turn` (all 4 subcats) | 800 |
| full | full `multi_turn` (all 4 subcats) | 800 |

(BFCL multi_turn has no natural "medium" size; standard == full == the whole 800-case category, per
the contract's "bfcl multi_turn full (~800)".)

## Run-time / cost estimate

Multi-turn is the most expensive BFCL category: each case runs up to 4 user turns x up to 20
agentic steps, each step a `/v1/completions` round-trip, with the trajectory re-sent every step.
Expect on the order of tens of requests per case and long prompts in `long_context`. At
`AE_CONCURRENCY=4` against our local box, budget **multiple hours** for the full 800-case standard
run per config (exact time depends on the served config's decode speed; the w8a8 TP=2 configs and
the int4 single-card configs differ). smoke (5 cases) finishes in a couple of minutes once decode
is warm. No API cost (local vLLM); cost is GPU wall-time under the lease.

## Caveats / friction hit (all resolved)

1. **`bfcl version` is broken upstream** -- it looks up dist metadata for `bfcl` but the package is
   `bfcl-eval` (`PackageNotFoundError`). `setup.sh` uses `bfcl test-categories` for the
   liveness/import probe instead.
2. **`uv venv` is not idempotent** -- it errors if `.venv` exists. `setup.sh` only creates it when
   missing.
3. **Score files are JSON-LINES, not a JSON array** -- line 0 is the header
   `{accuracy,correct_count,total_count}`, lines 1..n are the FAILED entries only. `parse.py` reads
   JSONL (and tolerates an array for forward-compat). Validated on real BFCL output.
4. **`--run-ids` id file is a DICT** keyed by subcategory: `{"multi_turn_base": ["multi_turn_base_0",
   ...], ...}` (not a flat list). `run.sh` writes it in that shape for smoke.
5. **Subset evaluation needs `--partial-eval`** -- `bfcl evaluate` raises on a result/dataset length
   mismatch otherwise. `run.sh` passes it for `smoke` only (standard/full cover the whole category).
6. **`spin_up_local_server` runs even with `--skip-server-setup`** -- it loads a tokenizer and polls
   `GET {base}/models`. Hence the `REMOTE_OPENAI_TOKENIZER_PATH` + `HF_HUB_OFFLINE` handling, and the
   reliance on our server exposing `/v1/models` (it does).
7. **Registry injection** uses `sitecustomize.py` on `PYTHONPATH` (no upstream user-config hook
   exists). It is a silent no-op in any interpreter where `bfcl_eval` is absent (e.g. the system
   python3 that runs `evallib.py`).

## What was validated offline (no live endpoint)

- `setup.sh` runs clean and idempotently; `bfcl test-categories` / `bfcl models` work; bundled
  multi_turn dataset present (4 subcats x 200 = 800).
- The runtime registry injection makes `bfcl models` / handler resolution see `qwen36-27b-int4` ->
  `QwenFCHandler`, model_name == served id.
- `QwenFCHandler` correctly parses realistic multi-call Qwen3 `<tool_call>` XML (incl. dotted names).
- **Full generate -> evaluate -> parse -> emit pipeline run end-to-end against a local stub
  OpenAI-completions server** (real BFCL multi-turn execution engine, real score files): produced
  the canonical `results/<label>/bfcl.json` with stable BFCL ids, per-subcat accuracy, and the
  failure-type breakdown. `parse.py` verified on the actual JSONL score files (and on a synthesized
  full-size fixture).

## Unresolved risk for the live run

- **Prompt-FC vs native-FC mismatch**: BFCL prompt-FC injects tools as `<tools>` in a
  `/v1/completions` prompt; it does not use our `--tool-call-parser qwen3_coder`. If a config's
  served chat template differs materially from BFCL's hand-written Qwen3 template, scores could be
  depressed by formatting, not capability. Mitigation if scores look anomalously low: try
  `BFCL_REGISTER_FC=0` (plain prompt) or sanity-check one trajectory's `inference_log`.
- **max_tokens cap = min(4096, ctx - prompt_tokens - 2)** inside the handler; our serve `MAXLEN` is
  16384. Long `long_context` cases could truncate. Identical across configs, so the A/B stays fair,
  but absolute long_context numbers may be low.
- Run time for the full 800-case standard pass is multi-hour per config under the GPU lease; plan
  campaign scheduling accordingly.
- The stub validation could not exercise CORRECT trajectories (it returns no/dummy tool calls), so
  the "passed=true" path through the real checker is only validated structurally (via the
  result-file id recovery + the synthesized fixture), not against a live model producing correct
  calls. First live smoke should confirm at least one case passes.
