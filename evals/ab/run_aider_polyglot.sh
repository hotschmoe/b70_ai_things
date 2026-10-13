#!/bin/bash
# Aider polyglot benchmark against any OpenAI-compatible endpoint (no GPU touched).
# Usage: BASE_URL=http://192.168.10.5:18080/v1 API_KEY=... MODEL=hotschmoe-dd \
#        ARM_LABEL=armA THREADS=4 THINKING=on|off|default MAX_TOKENS=16384 \
#        [LIMIT=N] [LANGS=python,go] [EDIT_FORMAT=diff|whole] [CTX=200000] [SEED=1234] \
#        evals/ab/run_aider_polyglot.sh
# LIMIT picks N exercises with a SEEDED shuffle, so both arms get the identical subset.
set -euo pipefail
AB=${AB_ROOT:-/mnt/vm_8tb/b70/evals/ab}
: "${BASE_URL:?}" "${MODEL:?}" "${ARM_LABEL:?}"
API_KEY=${API_KEY:-none}; THREADS=${THREADS:-1}; THINKING=${THINKING:-off}
MAX_TOKENS=${MAX_TOKENS:-16384}; EDIT_FORMAT=${EDIT_FORMAT:-diff}; CTX=${CTX:-131072}
SEED=${SEED:-1234}; TRIES=${TRIES:-2}; LIMIT=${LIMIT:--1}; LANGS=${LANGS:-}
TIMEOUT=${TIMEOUT:-3600}   # per-LLM-request seconds (aider default 600 is too low for slow thinking)
STAMP=$(date +%Y%m%dT%H%M%S); RUN="$AB/runs/aider-$ARM_LABEL-$STAMP"
mkdir -p "$RUN/benchmarks"; cp -r "$AB/aider/tmp.benchmarks/polyglot-benchmark" "$RUN/benchmarks/"
case "$THINKING" in
  on)  EB='extra_body: {chat_template_kwargs: {enable_thinking: true}}';;
  off) EB='extra_body: {chat_template_kwargs: {enable_thinking: false}}';;
  default) EB='';;
  *) echo "THINKING must be on|off|default" >&2; exit 2;;
esac
cat > "$RUN/model-settings.yml" <<Y
- name: openai/$MODEL
  edit_format: $EDIT_FORMAT
  use_repo_map: false
  extra_params:
    max_tokens: $MAX_TOKENS
    $EB
Y
cat > "$RUN/model-metadata.json" <<J
{"openai/$MODEL": {"max_tokens": $MAX_TOKENS, "max_input_tokens": $CTX, "max_output_tokens": $MAX_TOKENS,
 "input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}}
J
cat > "$RUN/wrapper.py" <<'P'
import random, sys, os
sys.path.insert(0, "/aider")
from aider import models
from benchmark.benchmark import app
models.request_timeout = int(os.environ["AB_TIMEOUT"])
models.register_litellm_models(["/run/model-metadata.json"])
random.seed(int(os.environ["AB_SEED"]))   # benchmark.py shuffles unseeded; seed => same subset per arm
app()
P
ARGS=(--model "openai/$MODEL" --edit-format "$EDIT_FORMAT" --tries "$TRIES" --threads "$THREADS"
      --read-model-settings /run/model-settings.yml --new --num-tests "$LIMIT")
[ -n "$LANGS" ] && ARGS+=(--languages "$LANGS")
{ echo "arm=$ARM_LABEL model=$MODEL base_url=$BASE_URL thinking=$THINKING max_tokens=$MAX_TOKENS ctx=$CTX"
  echo "edit_format=$EDIT_FORMAT tries=$TRIES threads=$THREADS limit=$LIMIT langs=$LANGS seed=$SEED"
  echo "aider_commit=$(git -C "$AB/aider" rev-parse HEAD) polyglot=$(git -C "$AB/aider/tmp.benchmarks/polyglot-benchmark" rev-parse HEAD)"; } | tee "$RUN/config.txt"
docker run --rm --network host --memory=32g --memory-swap=32g \
  -v "$AB/aider":/aider -v "$RUN/benchmarks":/benchmarks -v "$RUN":/run \
  -e OPENAI_API_BASE="$BASE_URL" -e OPENAI_API_KEY="$API_KEY" -e AIDER_DOCKER=1 \
  -e AIDER_BENCHMARK_DIR=/benchmarks -e AB_TIMEOUT="$TIMEOUT" -e AB_SEED="$SEED" \
  -e LITELLM_LOG=ERROR -e PYTHONDONTWRITEBYTECODE=1 \
  aider-benchmark:5dc9490 python3 /run/wrapper.py "${ARGS[@]}" "aider-$ARM_LABEL" 2>&1 | tee "$RUN/run.log"
RC=${PIPESTATUS[0]}
docker run --rm -v "$RUN":/d aider-benchmark:5dc9490 chown -R "$(id -u):$(id -g)" /d >/dev/null 2>&1 || true
python3 - "$RUN" <<'P'
import json, glob, sys, statistics as st
run = sys.argv[1]; rs = []
for f in glob.glob(run + "/benchmarks/*/*/exercises/practice/*/.aider.results.json"):
    try: rs.append(json.load(open(f)))
    except Exception: pass
n = len(rs); tries = max([len(r.get("tests_outcomes", [])) for r in rs] + [0])
passed = lambda r, k: any(r["tests_outcomes"][:k]) if r.get("tests_outcomes") else False
out = {"completed": n, "pass_rate_1": sum(passed(r, 1) for r in rs) / max(n, 1),
       "pass_rate_2": sum(passed(r, 2) for r in rs) / max(n, 1),
       "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rs), "completion_tokens": sum(r.get("completion_tokens", 0) for r in rs),
       "mean_duration_s": st.mean([r.get("duration", 0) for r in rs]) if rs else None,
       "malformed": sum(r.get("num_malformed_responses", 0) for r in rs), "timeouts": sum(r.get("test_timeouts", 0) for r in rs),
       "context_exhausted": sum(r.get("num_exhausted_context_windows", 0) for r in rs),
       "per_exercise": {r["testdir"].split("benchmarks/")[-1]: r.get("tests_outcomes") for r in rs}}
json.dump(out, open(run + "/summary.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "per_exercise"}); print("summary:", run + "/summary.json")
P
exit "${RC:-0}"
