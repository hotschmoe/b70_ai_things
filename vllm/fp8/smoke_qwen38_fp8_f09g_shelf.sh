#!/usr/bin/env bash
# F09g: live smoke of the promoted Qwen3.8 FP8 MTP1 shelf wrapper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/f09g_qwen38_fp8_shelf_smoke/$STAMP}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/qwen38-fp8-daily-mtp1}"
IMAGE=b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a
SHELF="$REPO/rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh"
NAME=qwen38_fp8_f09g_shelf
PORT=18209
SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f09g-shelf

case "${1:-}" in
  --leased) shift ;;
  '') exec env B70_AGENT=f09g-qwen38-fp8-shelf "$REPO/bin/gpu-run" bash "$0" --leased ;;
  *) echo "usage: $0" >&2; exit 2 ;;
esac

[ ! -e "$RESULT_DIR" ] || { echo "RESULT_DIR must be new: $RESULT_DIR" >&2; exit 1; }
mkdir -p "$RESULT_DIR" "$CACHE_DIR"
server_pid=""

stop_server() {
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [ -n "$server_pid" ]; then
    wait "$server_pid" 2>/dev/null || true
  fi
}

run_health() {
  local label="$1"
  "$REPO/bin/xpu-health" --img "$IMAGE" \
    >"$RESULT_DIR/${label}-card.log" 2>&1
  env IMG="$IMAGE" "$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180 \
    >"$RESULT_DIR/${label}-collective.log" 2>&1
}

cleanup() {
  local rc=$?
  set +e
  stop_server
  if [ "$rc" -ne 0 ]; then
    "$REPO/bin/xe-reset" --method rebind >"$RESULT_DIR/recovery.log" 2>&1 || rc=1
  fi
  run_health failure-post >"$RESULT_DIR/failure-post.log" 2>&1 || rc=1
  echo "$rc" >"$RESULT_DIR/smoke.rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "CONFIG -> PROFILE=daily MTP1 262144/c4 direct-P2P FULL graph shelf" \
  | tee "$RESULT_DIR/config.txt"
echo "COMMAND -> bin/gpu-run bash vllm/fp8/smoke_qwen38_fp8_f09g_shelf.sh --leased" \
  | tee "$RESULT_DIR/command.txt"
run_health pre

env PROFILE=daily NAME="$NAME" PORT="$PORT" SERVED="$SERVED" \
  CACHE_DIR="$CACHE_DIR" bash "$SHELF" start >"$RESULT_DIR/server.log" 2>&1 &
server_pid=$!

for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  kill -0 "$server_pid" 2>/dev/null || {
    tail -120 "$RESULT_DIR/server.log" >&2
    exit 1
  }
  sleep 5
done
curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null
curl -fsS --max-time 15 "http://127.0.0.1:${PORT}/v1/models" \
  >"$RESULT_DIR/models.json"
python3 - "$RESULT_DIR/models.json" "$SERVED" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="ascii"))
assert [item["id"] for item in data["data"]] == [sys.argv[2]]
PY
python3 - "$PORT" "$SERVED" "$RESULT_DIR/completion.json" <<'PY'
import json
import sys
import urllib.request

port, served, output = sys.argv[1:]
payload = json.dumps(
    {
        "model": served,
        "prompt": "Reply with exactly READY.",
        "max_tokens": 16,
        "temperature": 0,
        "top_p": 1,
    }
).encode("ascii")
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    data = json.load(response)
assert data["model"] == served
assert len(data["choices"]) == 1 and data["choices"][0]["text"]
with open(output, "w", encoding="ascii") as handle:
    json.dump(data, handle, indent=2, ensure_ascii=True, sort_keys=True)
    handle.write("\n")
PY

stop_server
[ -z "$(docker ps -aq --filter "name=^/${NAME}$")" ]
! ss -ltn | grep -Eq ":${PORT}[[:space:]]"
run_health post
echo "RESULT -> model identity, completion, teardown, card health, and collective health passed" \
  | tee "$RESULT_DIR/result.txt"
echo "VERDICT -> F09g PASS: promoted MTP1 shelf wrapper is live" \
  | tee "$RESULT_DIR/verdict.txt"
echo 0 >"$RESULT_DIR/smoke.rc"
trap - EXIT INT TERM HUP
