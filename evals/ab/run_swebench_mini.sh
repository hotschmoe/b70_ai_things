#!/bin/bash
# mini-swe-agent on the pre-registered 30-instance SWE-bench Verified subset, then official swebench scoring.
# Usage: BASE_URL=http://192.168.10.5:18080/v1 API_KEY=... MODEL=hotschmoe-dd ARM_LABEL=armA \
#        THREADS=2 THINKING=on|off|default MAX_TOKENS=8192 STEP_LIMIT=100 WALL_LIMIT=2400 \
#        [LIMIT=N] [INSTANCES=file] [TEXTBASED=1] [SKIP_SCORE=1] evals/ab/run_swebench_mini.sh
# Runs on the host (agent) + per-instance docker containers (repo env); no GPU touched.
set -euo pipefail
AB=${AB_ROOT:-/mnt/vm_8tb/b70/evals/ab}; V=$AB/venv-swe/bin
: "${BASE_URL:?}" "${MODEL:?}" "${ARM_LABEL:?}"
API_KEY=${API_KEY:-none}; THREADS=${THREADS:-1}; THINKING=${THINKING:-off}; MAX_TOKENS=${MAX_TOKENS:-8192}
STEP_LIMIT=${STEP_LIMIT:-100}; WALL_LIMIT=${WALL_LIMIT:-2400}; LIMIT=${LIMIT:-0}; TIMEOUT=${TIMEOUT:-1800}
INSTANCES=${INSTANCES:-$AB/swebench_verified_subset30.txt}
export HF_HOME=$AB/hf HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} MSWEA_COST_TRACKING=ignore_errors MSWEA_SILENT_STARTUP=1
RUN=$AB/runs/swe-$ARM_LABEL-$(date +%Y%m%dT%H%M%S); mkdir -p "$RUN"; export MSWEA_GLOBAL_CONFIG_DIR=$RUN/cfg; mkdir -p "$MSWEA_GLOBAL_CONFIG_DIR"
IDS=$(grep -v '^\s*$' "$INSTANCES"); [ "$LIMIT" -gt 0 ] && IDS=$(echo "$IDS" | head -n "$LIMIT")
echo "$IDS" > "$RUN/instances.txt"; FILTER="^($(echo "$IDS" | paste -sd'|'))\$"
CFG=swebench.yaml; MC=litellm; [ "${TEXTBASED:-0}" = 1 ] && { CFG=swebench_backticks.yaml; MC=litellm_textbased; }
CT=()
case "$THINKING" in
  on)  CT=(-c 'model.model_kwargs.extra_body={"chat_template_kwargs":{"enable_thinking":true}}');;
  off) CT=(-c 'model.model_kwargs.extra_body={"chat_template_kwargs":{"enable_thinking":false}}');;
  default) ;; *) echo "THINKING must be on|off|default" >&2; exit 2;;
esac
echo "arm=$ARM_LABEL model=$MODEL base_url=$BASE_URL thinking=$THINKING max_tokens=$MAX_TOKENS steps=$STEP_LIMIT wall=$WALL_LIMIT threads=$THREADS textbased=${TEXTBASED:-0} n=$(echo "$IDS" | wc -l)
mini_swe_agent=$(git -C "$AB/mini-swe-agent" rev-parse HEAD) swebench=$($V/python -c 'import importlib.metadata as m;print(m.version("swebench"))')" | tee "$RUN/config.txt"
START=$(date +%s)
"$V/python" -m minisweagent.run.benchmarks.swebench --subset verified --split test --filter "$FILTER" \
  -w "$THREADS" -o "$RUN/traj" --model "openai/$MODEL" --model-class "$MC" -c "$CFG" \
  -c "model.model_kwargs.api_base=\"$BASE_URL\"" -c "model.model_kwargs.api_key=\"$API_KEY\"" \
  -c "model.model_kwargs.max_tokens=$MAX_TOKENS" -c "model.model_kwargs.timeout=$TIMEOUT" "${CT[@]}" \
  -c "agent.step_limit=$STEP_LIMIT" -c "agent.wall_time_limit_seconds=$WALL_LIMIT" 2>&1 | tee "$RUN/run.log" || true
echo "agent_wall_s=$(( $(date +%s) - START ))" | tee -a "$RUN/config.txt"
[ "${SKIP_SCORE:-0}" = 1 ] && exit 0
cd "$RUN"
"$V/python" -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-Bench_Verified --split test \
  --predictions_path "$RUN/traj/preds.json" --instance_ids $IDS --max_workers 2 --run_id "$ARM_LABEL" --timeout 1800 2>&1 | tee "$RUN/score.log"
"$V/python" - "$RUN" <<'P'
import json, glob, sys, collections
run = sys.argv[1]; ids = open(run + "/instances.txt").read().split()
rep = {}
for f in glob.glob(run + "/*.json"):
    d = json.load(open(f))
    if "resolved_ids" in d: rep = d
res = set(rep.get("resolved_ids", [])); tok_in = tok_out = steps = 0; ex = collections.Counter(); per = {}
for i in ids:
    try:
        t = json.load(open(f"{run}/traj/{i}/{i}.traj.json"))
        ex[t["info"].get("exit_status")] += 1
        u = [m.get("extra", {}).get("response", {}).get("usage", {}) for m in t["messages"] if m.get("role") == "assistant"]
        tin = sum(x.get("prompt_tokens", 0) for x in u); tout = sum(x.get("completion_tokens", 0) for x in u)
        tok_in += tin; tok_out += tout; steps += len(u); per[i] = {"resolved": i in res, "steps": len(u), "exit": t["info"].get("exit_status")}
    except Exception as e: per[i] = {"resolved": i in res, "error": str(e)[:80]}
out = {"n": len(ids), "resolved": len(res & set(ids)), "rate": len(res & set(ids)) / len(ids), "empty_patch": rep.get("empty_patch_ids", []),
       "exit_status": dict(ex), "prompt_tokens": tok_in, "completion_tokens": tok_out, "agent_steps": steps, "per_instance": per}
json.dump(out, open(run + "/summary.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "per_instance"}); print("summary:", run + "/summary.json")
P
