#!/usr/bin/env bash
# agentic-eval/harnesses/aider/run.sh -- run the Aider polyglot benchmark (single-shot
# codegen CONTROL) against the live vLLM endpoint for one config, then emit the standard JSON.
#
#   bash run.sh <config_label> <subset>      subset = smoke | standard | full
#
# The benchmark runs INSIDE the aider-benchmark Docker image (bundles g++/go/java/node/rust/
# python) with --network host so http://localhost:18080/v1 reaches the host vLLM. We pin the
# EXACT exercise set ourselves (deterministic seeded selection) and pass it via --keywords so the
# same task_ids pair across all four configs (benchmark.py's own --num-tests is an UNSEEDED
# shuffle -> not config-stable; we do not use it).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "$HERE/../.." && pwd)/lib/common.sh"
ae_set_config "$1"
SUBSET="${2:-standard}"

PY="$HERE/.venv/bin/python"
DATA_POLY="$HERE/data/polyglot-benchmark"
IMAGE_TAG="aider-benchmark"

[ -x "$PY" ]            || { ae_log "aider: missing .venv -- run setup.sh first"; exit 3; }
[ -d "$DATA_POLY" ]     || { ae_log "aider: missing dataset $DATA_POLY -- run setup.sh"; exit 3; }
docker image inspect "$IMAGE_TAG" >/dev/null 2>&1 || { ae_log "aider: docker image '$IMAGE_TAG' missing -- run setup.sh"; exit 3; }

# ---- per-run scratch dir (gitignored) ----------------------------------------------------
RUN_ID="$(date +%Y%m%d-%H%M%S)-${EVAL_LABEL}-${SUBSET}"
RUN_DIR="$HERE/runs/$RUN_ID"
BENCH_DIR="$RUN_DIR/tmp.benchmarks"          # AIDER_BENCHMARK_DIR inside the container
mkdir -p "$BENCH_DIR"
# benchmark.py wants the dataset at $AIDER_BENCHMARK_DIR/polyglot-benchmark; bind-mount it RO.
# (We mount, rather than copy, so we keep the single pinned dataset checkout.)

# ---- deterministic, config-invariant exercise selection ----------------------------------
mapfile -t SELECTED < <("$PY" "$HERE/select_subset.py" "$DATA_POLY" "$SUBSET" --seed "$AE_SEED")
[ "${#SELECTED[@]}" -gt 0 ] || { ae_log "aider: subset selection empty"; exit 4; }
KEYWORDS="$(IFS=,; echo "${SELECTED[*]}")"
ae_log "aider: subset=$SUBSET -> ${#SELECTED[@]} exercises (seed=$AE_SEED)"

# ---- model-settings YAML: pin determinism for the served id (temp/top_p/seed/max_tokens) --
# benchmark.py has no --temperature flag; aider takes it from model settings. We register the
# served id as a known model with use_temperature=$AE_TEMPERATURE and extra_params for the rest,
# and force edit_format=diff so percent_cases_well_formed actually catches diff breakage (the
# default "whole" format is ~never malformed, making that metric blind).
MODEL_SETTINGS="$RUN_DIR/model-settings.yml"
TEMP_FIELD="$AE_TEMPERATURE"
# aider treats use_temperature: false (no temperature sent) vs a float (sent). 0.0 is a valid float.
cat > "$MODEL_SETTINGS" <<YAML
- name: openai/${EVAL_SERVED}
  edit_format: diff
  use_repo_map: false
  use_temperature: ${TEMP_FIELD}
  streaming: false
  extra_params:
    top_p: ${AE_TOP_P}
    seed: ${AE_SEED}
    max_tokens: ${AE_MAX_TOKENS}
YAML

ae_log "aider: config=$EVAL_LABEL served=$EVAL_SERVED endpoint=$EVAL_ENDPOINT temp=$AE_TEMPERATURE"

# ---- run the dockerized benchmark --------------------------------------------------------
# --network host  : http://localhost:18080/v1 resolves to the host vLLM from inside.
# OPENAI_API_BASE : LiteLLM routes openai/<id> to our endpoint. Key is a dummy (local vLLM).
# AIDER_DOCKER=1  : benchmark.py refuses to run unvetted model code outside a container w/o it.
# --user host uid : output files are owned by us so runs/ is cleanable without sudo. That needs a
#                   writable HOME and a runtime git safe.directory (benchmark.py reads the commit).
# We mount the dataset (RO), the run's tmp.benchmarks (RW), and the model-settings file (RO).
B=$(ae_snap); START=$(ae_now)

INNER="export HOME=/tmp/h; mkdir -p \$HOME; git config --global --add safe.directory /aider; \
python3 /aider/benchmark/benchmark.py '$RUN_ID' \
  --model 'openai/${EVAL_SERVED}' \
  --edit-format diff \
  --read-model-settings /aider/model-settings.yml \
  --tries 2 \
  --threads '$AE_CONCURRENCY' \
  --keywords '$KEYWORDS' \
  --new \
  --exercises-dir polyglot-benchmark"

set +e
docker run --rm \
  --network host \
  --memory=12g --memory-swap=12g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/h \
  -e AIDER_DOCKER=1 \
  -e AIDER_BENCHMARK_DIR=/benchmarks \
  -e OPENAI_API_BASE="$EVAL_ENDPOINT" \
  -e OPENAI_API_KEY=dummy \
  -e AIDER_CHECK_UPDATE=false \
  -v "$DATA_POLY":/benchmarks/polyglot-benchmark:ro \
  -v "$BENCH_DIR":/benchmarks \
  -v "$MODEL_SETTINGS":/aider/model-settings.yml:ro \
  "$IMAGE_TAG" \
  bash -lc "$INNER" \
  2>&1 | tee "$RUN_DIR/benchmark.log"
RC=${PIPESTATUS[0]}
set -e

END=$(ae_now); A=$(ae_snap)
ae_log "aider: benchmark container exited rc=$RC"

# benchmark.py writes one run dir under $AIDER_BENCHMARK_DIR named "<commit>--$RUN_ID" (or RUN_ID).
# Locate the dir that holds the per-exercise .aider.results.json files.
NATIVE_DIR="$(find "$BENCH_DIR" -maxdepth 1 -type d -name "*${RUN_ID}*" | head -1)"
[ -n "$NATIVE_DIR" ] || NATIVE_DIR="$BENCH_DIR"
ae_log "aider: parsing native results from $NATIVE_DIR"

# Record the selected task set so parse.py can account for non-run/missing exercises (= fail).
printf '%s\n' "${SELECTED[@]}" > "$RUN_DIR/.selected.txt"
PARSED="$RESULTS_DIR/.aider.parsed.json"
"$PY" "$HERE/parse.py" "$NATIVE_DIR" --selected "$RUN_DIR/.selected.txt" > "$PARSED"

"$AE_PY" "$AE_LIB/evallib.py" emit \
  --config "$EVAL_LABEL" --harness aider --subset "$SUBSET" --served "$EVAL_SERVED" \
  --parsed "$PARSED" \
  --tok-before "$B" --tok-after "$A" --start "$START" --end "$END" \
  --out "$RESULTS_DIR/aider.json" --meta "temperature=$AE_TEMPERATURE"

ae_log "aider: done ($EVAL_LABEL/$SUBSET) -> $RESULTS_DIR/aider.json"
