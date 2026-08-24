#!/usr/bin/env bash
# Profile the corrected Ornith MTP1 Sglang path with semantic Kineto ranges.
# Run only through:
#   ./bin/gpu-run bash sglang/w8a8/profile_ornith15_w8a8.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SERVE="$REPO/sglang/w8a8/serve_ornith15_w8a8.sh"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/ornith15_w8a8_profile_$STAMP}"
PROFILE_ROOT="${PROFILE_ROOT:-ornith15_w8a8_profile_$STAMP}"
HOST_PROFILE="$ROOT/sgl_cache/$PROFILE_ROOT"
CONTAINER_PROFILE="/sgl_cache/$PROFILE_ROOT"
NAME="${NAME:-sglang_ornith15_w8a8_profile}"
PORT="${PORT:-18080}"
SERVED="${SERVED:-ornith-1.5-35b-a3b-W8A8-rtn-mtp1-profile}"
CTX="${CTX:-8192}"
PROFILE_STEPS="${PROFILE_STEPS:-5}"
BENCH_REPS="${BENCH_REPS:-3}"

mkdir -p "$OUT" "$HOST_PROFILE"
exec > >(tee -a "$OUT/campaign.log") 2>&1

active=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$active" = 1 ]; then
    docker logs "$NAME" >"$OUT/server.log" 2>&1 || true
    docker inspect "$NAME" >"$OUT/container_inspect.json" 2>/dev/null || true
    NAME="$NAME" PORT="$PORT" SERVED="$SERVED" bash "$SERVE" stop || true
  fi
  "$REPO/bin/xpu-health" >"$OUT/health_post.log" 2>&1 || rc=1
  cat "$OUT/health_post.log"
  echo "VERDICT -> exit=$rc artifacts=$OUT traces=$HOST_PROFILE"
  exit "$rc"
}
trap cleanup EXIT INT TERM

{
  echo "utc_start=$STAMP"
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "kernel=$(uname -r)"
  echo "image=sglang-xpu:mtp-0515"
  echo "model=/models/ornith-1.5-35b-a3b/w8a8-rtn-mtp-shisa"
  echo "served=$SERVED"
  echo "tp=2"
  echo "ctx=$CTX"
  echo "mtp_steps=1"
  echo "draft_tokens=2"
  echo "graph=disabled"
  echo "overlap=disabled"
  echo "radix=disabled"
  echo "ccl_topo_p2p_access=0"
  docker image inspect sglang-xpu:mtp-0515 --format 'image_id={{.Id}}' 2>/dev/null || true
} >"$OUT/manifest.txt"

"$REPO/bin/xpu-health" | tee "$OUT/health_pre.log"
for power_cap in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do
  [ -f "$power_cap" ] || continue
  printf '%s ' "$power_cap"
  tr -d '\n' <"$power_cap"
  echo
done >"$OUT/power_caps_microwatts.txt"

echo "CONFIG -> Sglang TP2 eager, true routed-expert W8A8, MTP1, P2P off"
CTX="$CTX" RADIX=0 MAXREQ=1 MTP=1 SPEC_STEPS=1 SPEC_DRAFT=2 \
  PROFILE_RANGES=1 PORT="$PORT" NAME="$NAME" SERVED="$SERVED" \
  bash "$SERVE" start | tee "$OUT/start.log"
active=1

curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$OUT/models.json"
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" >"$OUT/server_info.json"
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/metrics" >"$OUT/metrics_before.txt"
docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | sort >"$OUT/container_env.txt"
rg -q '^B70_ORNITH_PROFILE_RANGES=1$' "$OUT/container_env.txt"
rg -q '^CCL_TOPO_P2P_ACCESS=0$' "$OUT/container_env.txt"

echo "COMMAND -> phase_bench p512 g128 n=$BENCH_REPS ignore-eos"
python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
  --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --prompt-tokens 512 --gen-tokens 128 --n "$BENCH_REPS" --ignore-eos \
  --label ornith15-w8a8-sglang-eager-mtp1 --out "$OUT/phase_bench.json" \
  2>&1 | tee "$OUT/phase_bench.log"

echo "COMMAND -> Kineto CPU+XPU five-step staged decode profile"
curl -fsS -X POST "http://127.0.0.1:$PORT/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"output_dir\":\"$CONTAINER_PROFILE\",\"num_steps\":$PROFILE_STEPS,\"activities\":[\"CPU\",\"XPU\"],\"profile_by_stage\":true,\"record_shapes\":true,\"with_stack\":false,\"profile_prefix\":\"ornith_mtp1\"}" \
  >"$OUT/start_profile_response.txt"

python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
  "http://127.0.0.1:$PORT/v1" "$SERVED" 1 128 32 1 \
  2>&1 | tee "$OUT/profile_trigger.log"

found=0
for _ in $(seq 1 90); do
  if [ "$(find "$HOST_PROFILE" -maxdepth 1 -type f -name '*DECODE.trace.json.gz' | wc -l)" = 2 ]; then
    found=1
    break
  fi
  sleep 1
done
[ "$found" = 1 ]

mapfile -t traces < <(find "$HOST_PROFILE" -maxdepth 1 -type f -name '*DECODE.trace.json.gz' | sort)
python3 "$REPO/sglang/w8a8/analyze_ornith_trace.py" "${traces[@]}" \
  --json-out "$OUT/semantic_trace.json" | tee "$OUT/semantic_trace.log"
python3 "$REPO/sglang/graph_mtp/analyze_trace.py" "${traces[0]}" 50 \
  >"$OUT/rank0_launch_gap_profile.txt"
python3 "$REPO/sglang/graph_mtp/analyze_trace.py" "${traces[1]}" 50 \
  >"$OUT/rank1_launch_gap_profile.txt"
python3 "$REPO/scripts/112_parse_trace.py" "$HOST_PROFILE" \
  >"$OUT/kernel_bucket_profile.txt"
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/metrics" >"$OUT/metrics_after.txt"

docker logs "$NAME" >"$OUT/server.log" 2>&1
if rg -i 'device_lost|out_of_resources|ur_result_error|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' \
  "$OUT/server.log"; then
  echo "VERDICT -> FAIL fatal server marker"
  exit 1
fi

echo "RESULT -> profile complete"
