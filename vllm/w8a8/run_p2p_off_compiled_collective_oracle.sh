#!/usr/bin/env bash
# Run the M02 P2P-off compiled collective oracle in three fresh lifetimes.
# Invoke this script directly. It acquires both B70 leases through bin/gpu-run;
# --leased is an internal recursion guard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/m02_p2p_off_collective/$STAMP}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/m02_p2p_off_collective}"
LIFETIME_TIMEOUT="${LIFETIME_TIMEOUT:-600}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
EAGER_ITERATIONS="${EAGER_ITERATIONS:-8}"
COMPILED_ITERATIONS="${COMPILED_ITERATIONS:-8}"
GRAPH_ITERATIONS="${GRAPH_ITERATIONS:-16}"

print_config() {
  echo "image=$IMG"
  echo "p2p=0"
  echo "world_size=2"
  echo "lifetimes=3"
  echo "all_reduce_shapes=[1,5120],[4,5120]"
  echo "all_gather_input_shapes=[4,2560]"
  echo "modes=eager_direct,compiled_functional_wait_tensor,compiled_xpu_graph"
  echo "result_dir=$RESULT_DIR"
}

case "${1:-}" in
  --print-config) print_config; exit 0 ;;
  --leased) shift ;;
  -h|--help)
    sed -n '2,4s/^# *//p' "$0"
    exit 0
    ;;
  "")
    exec env B70_AGENT=m02-p2p-off-collective \
      "$REPO/bin/gpu-run" bash "$0" --leased
    ;;
  *) echo "usage: $0 [--print-config]" >&2; exit 2 ;;
esac

case "$IMG" in
  *@sha256:*) ;;
  *) echo "IMG must be digest pinned, got: $IMG" >&2; exit 2 ;;
esac
case "$LIFETIME_TIMEOUT:$HEALTH_TIMEOUT:$EAGER_ITERATIONS:$COMPILED_ITERATIONS:$GRAPH_ITERATIONS" in
  *[!0-9:]*|*::*|:*|*:)
    echo "timeouts and iteration counts must be positive integers" >&2
    exit 2
    ;;
esac
for value in "$LIFETIME_TIMEOUT" "$HEALTH_TIMEOUT" "$EAGER_ITERATIONS" \
  "$COMPILED_ITERATIONS" "$GRAPH_ITERATIONS"; do
  [ "$value" -gt 0 ] || {
    echo "timeouts and iteration counts must be positive integers" >&2
    exit 2
  }
done

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
docker image inspect "$IMG" >/dev/null 2>&1 || {
  echo "pinned image is not local: $IMG" >&2
  exit 2
}
mkdir -p "$RESULT_DIR" "$CACHE_DIR"

current_name=""
cleanup() {
  if [ -n "$current_name" ]; then
    docker rm -f "$current_name" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

run_health() {
  "$REPO/bin/xpu-health"
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT"
}

print_config
echo "pre-health -> begin"
run_health
echo "pre-health -> pass"

for lifetime in 1 2 3; do
  current_name="m02-p2p-off-${STAMP}-${lifetime}"
  lifetime_cache="$CACHE_DIR/$STAMP/lifetime-$lifetime"
  mkdir -p "$lifetime_cache"
  echo "lifetime=$lifetime -> entry container=$current_name"
  set +e
  timeout --signal=TERM --kill-after=30 "$LIFETIME_TIMEOUT" \
    docker run --rm --name "$current_name" --device /dev/dri \
      -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
      -v "$SCRIPT_DIR/p2p_off_compiled_collective_oracle.py:/opt/p2p_off_compiled_collective_oracle.py:ro" \
      -v "$RESULT_DIR:/results" \
      -v "$lifetime_cache:/m02-cache" \
      -e B70_ORACLE_IMAGE="$IMG" \
      -e CCL_ATL_TRANSPORT=ofi \
      -e CCL_TOPO_P2P_ACCESS=0 \
      -e CCL_LOG_LEVEL=info \
      -e FI_TCP_IFACE=eth0 \
      -e CCL_KVS_IFACE=eth0 \
      -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
      -e ZE_AFFINITY_MASK=0,1 \
      -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
      -e TORCHINDUCTOR_CACHE_DIR=/m02-cache/torchinductor \
      -e TRITON_CACHE_DIR=/m02-cache/triton \
      --entrypoint /opt/venv/bin/torchrun "$IMG" \
      --standalone --nproc-per-node=2 \
      /opt/p2p_off_compiled_collective_oracle.py \
      --output-dir /results --lifetime "$lifetime" \
      --eager-iterations "$EAGER_ITERATIONS" \
      --compiled-iterations "$COMPILED_ITERATIONS" \
      --graph-iterations "$GRAPH_ITERATIONS" \
      --timeout "$HEALTH_TIMEOUT"
  lifetime_rc=$?
  set -e

  docker rm -f "$current_name" >/dev/null 2>&1 || true
  if docker inspect "$current_name" >/dev/null 2>&1; then
    echo "lifetime=$lifetime -> teardown failed container=$current_name" >&2
    lifetime_rc=1
  else
    echo "lifetime=$lifetime -> teardown pass"
  fi
  current_name=""

  echo "lifetime=$lifetime -> post-health begin"
  set +e
  run_health
  health_rc=$?
  set -e
  if [ "$health_rc" -eq 0 ]; then
    echo "lifetime=$lifetime -> post-health pass"
  else
    echo "lifetime=$lifetime -> post-health fail rc=$health_rc" >&2
  fi
  if [ "$lifetime_rc" -ne 0 ] || [ "$health_rc" -ne 0 ]; then
    echo "lifetime=$lifetime -> fail oracle_rc=$lifetime_rc health_rc=$health_rc" >&2
    exit 1
  fi
  python3 - "$RESULT_DIR/lifetime-$lifetime.json" "$lifetime" <<'PY'
import json
import sys

path, expected_lifetime = sys.argv[1], int(sys.argv[2])
with open(path, encoding="ascii") as source:
    result = json.load(source)
assert result["passed"] is True, result
assert result["lifetime"] == expected_lifetime, result
assert result["matched_rank_call_signatures"] is True, result
PY
  echo "lifetime=$lifetime -> pass result=$RESULT_DIR/lifetime-$lifetime.json"
done

python3 - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
documents = [json.loads((root / f"lifetime-{index}.json").read_text(encoding="ascii")) for index in range(1, 4)]
assert [document["lifetime"] for document in documents] == [1, 2, 3]
assert all(document["passed"] for document in documents)
print("M02_OK lifetimes=3 p2p=0 matched_rank_entry_return=true")
PY
echo "result -> $RESULT_DIR"
