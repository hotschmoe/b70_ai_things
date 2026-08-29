#!/usr/bin/env bash
# Run the bounded M04 Qwen3.8 W8A8 paired-rank graph census.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/m04_graph_census/$STAMP}"
PROFILE_ROOT="m04_graph_census_$STAMP"
HOST_PROFILE="$ROOT/sgl_cache/$PROFILE_ROOT"
CONTAINER_PROFILE="/sgl_cache/$PROFILE_ROOT"
SERVE="$SCRIPT_DIR/serve_qwen38_w8a8.sh"
WORKLOAD="$SCRIPT_DIR/graph_census_workload.py"
CENSUS="$REPO/vllm/w8a8/graph_boundary_census.py"
NAME="m04-qwen38-w8a8-$STAMP"
SERVED="qwen3.8-27b-W8A8-gptq-gdn-rtn-m04-breakable-tp2"
PORT="${PORT:-18080}"
PROFILE_STEPS="${PROFILE_STEPS:-4}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"

case "${1:-}" in
  --leased) shift ;;
  "")
    exec env B70_AGENT=m04-graph-census \
      "$REPO/bin/gpu-run" bash "$0" --leased
    ;;
  *) echo "usage: $0" >&2; exit 2 ;;
esac

mkdir -p "$RESULT_DIR" "$HOST_PROFILE"
active=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$active" = 1 ]; then
    curl -fsS --max-time 15 "http://127.0.0.1:$PORT/health" \
      >"$RESULT_DIR/endpoint_before_teardown.txt" 2>&1 || rc=1
    docker logs "$NAME" >"$RESULT_DIR/server.log" 2>&1 || true
    NAME="$NAME" PORT="$PORT" SERVED="$SERVED" LOG="$RESULT_DIR/server.log" \
      bash "$SERVE" stop || rc=1
    if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "endpoint remained live after teardown" >&2
      rc=1
    fi
  fi
  "$REPO/bin/xpu-health" >"$RESULT_DIR/health_post_card.log" 2>&1 || rc=1
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT" \
    >"$RESULT_DIR/health_post_collective.log" 2>&1 || rc=1
  cat "$RESULT_DIR/health_post_card.log"
  cat "$RESULT_DIR/health_post_collective.log"
  echo "VERDICT -> exit=$rc result_dir=$RESULT_DIR"
  exit "$rc"
}
trap cleanup EXIT INT TERM

{
  echo "utc_start=$STAMP"
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "kernel=$(uname -r)"
  echo "served=$SERVED"
  echo "tp=2"
  echo "p2p=0"
  echo "target_dtype=bfloat16"
  echo "kv_dtype=bfloat16"
  echo "decode_graph=breakable"
  echo "graph_bs=1"
  echo "graph_reclaim=500"
  echo "mtp=0"
  echo "radix_cache=off"
  echo "profile_steps=$PROFILE_STEPS"
  echo "minimum_profiled_ratio=0.75"
} >"$RESULT_DIR/config.txt"

"$REPO/bin/xpu-health" | tee "$RESULT_DIR/health_pre_card.log"
"$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT" \
  | tee "$RESULT_DIR/health_pre_collective.log"

DECODE_GRAPH=breakable GRAPH_BS=1 CG_RECLAIM=500 \
  CTX=4096 MEMFRAC=0.75 MAXREQ=1 MTP=0 TOOLPARSER=none \
  NAME="$NAME" PORT="$PORT" SERVED="$SERVED" LOG="$RESULT_DIR/server.log" \
  bash "$SERVE" start | tee "$RESULT_DIR/start.log"
active=1

curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" \
  >"$RESULT_DIR/models.json"
docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | sort >"$RESULT_DIR/container_env.txt"
rg -q '^CCL_TOPO_P2P_ACCESS=0$' "$RESULT_DIR/container_env.txt"

python3 "$WORKLOAD" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --nonce 100000 --output-tokens 64 --json-out "$RESULT_DIR/warmup.json"
python3 "$WORKLOAD" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --nonce 100001 --output-tokens 128 --json-out "$RESULT_DIR/control_a.json"

first_token_signal="$RESULT_DIR/profiled_first_token.signal"
rm -f "$first_token_signal"
python3 "$WORKLOAD" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --nonce 100002 --output-tokens 128 --json-out "$RESULT_DIR/profiled.json" \
  --first-token-signal "$first_token_signal" &
profiled_pid=$!
signaled=0
for _ in $(seq 1 600); do
  if [ -f "$first_token_signal" ]; then
    signaled=1
    break
  fi
  kill -0 "$profiled_pid" 2>/dev/null || break
  sleep 0.05
done
[ "$signaled" = 1 ]
curl -fsS -X POST "http://127.0.0.1:$PORT/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"output_dir\":\"$CONTAINER_PROFILE\",\"num_steps\":$PROFILE_STEPS,\"activities\":[\"CPU\",\"XPU\"],\"profile_by_stage\":true,\"record_shapes\":true,\"with_stack\":false,\"profile_prefix\":\"m04_qwen38\"}" \
  >"$RESULT_DIR/start_profile_response.txt"
wait "$profiled_pid"

found=0
for _ in $(seq 1 90); do
  if [ "$(find "$HOST_PROFILE" -maxdepth 1 -type f -name '*DECODE.trace.json.gz' | wc -l)" = 2 ]; then
    found=1
    break
  fi
  sleep 1
done
[ "$found" = 1 ]

python3 "$WORKLOAD" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --nonce 100003 --output-tokens 128 --json-out "$RESULT_DIR/control_b.json"

mapfile -t traces < <(
  find "$HOST_PROFILE" -maxdepth 1 -type f -name '*DECODE.trace.json.gz' | sort
)
[ "${#traces[@]}" = 2 ]
control_value="$(jq -s '([.[0].post_first_tok_s, .[1].post_first_tok_s] | sort) | add / length' "$RESULT_DIR/control_a.json" "$RESULT_DIR/control_b.json")"
profiled_value="$(jq -r '.post_first_tok_s' "$RESULT_DIR/profiled.json")"
python3 "$CENSUS" \
  --trace "0=${traces[0]}" --trace "1=${traces[1]}" \
  --skip-iterations 0 \
  --control-value "$control_value" --profiled-value "$profiled_value" \
  --minimum-profiled-ratio 0.75 \
  --json-out "$RESULT_DIR/graph_boundary_census.json" \
  | tee "$RESULT_DIR/graph_boundary_census.log"

docker logs "$NAME" >"$RESULT_DIR/server.log" 2>&1
if rg -i 'device_lost|out_of_resources|ur_result_error|enginedead|gpu vm fault|(^|[^a-z])nan([^a-z]|$)' \
  "$RESULT_DIR/server.log"; then
  echo "fatal server marker" >&2
  exit 1
fi
echo "RESULT -> M04 runtime census passed result_dir=$RESULT_DIR"
