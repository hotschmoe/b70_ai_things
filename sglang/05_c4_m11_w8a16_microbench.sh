#!/usr/bin/env bash
# One-card exact-M=11 W8A16 routing microbenchmark. No server is started,
# stopped, queried, or restored. Invoke through the external single-card lease:
#
#   ./bin/gpu-run --card 0 bash sglang/05_c4_m11_w8a16_microbench.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SELF="$REPO/sglang/05_c4_m11_w8a16_microbench.sh"
BENCH="$REPO/sglang/bench_c4_m11_w8a16.py"
CARD="${CARD:-0}"
IMAGE="${IMAGE:-sglang-xpu:mtp}"
KDIR="${KDIR:-$ROOT/w8a8_kernel}"
SO="$KDIR/_xpu_C.abi3.so"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c4_m11_w8a16_microbench_$STAMP}"
NAME="c4_m11_w8a16_microbench_${CARD}_$$"
LOCK_BASE="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"
LEASE_CHECK_ONLY="${B70_C4_LEASE_CHECK_ONLY:-0}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
REPEATS="${REPEATS:-2}"
lease_proven=0

case "$CARD" in 0|1) ;; *) echo "CARD must be 0 or 1" >&2; exit 2 ;; esac
case "$LEASE_CHECK_ONLY" in
  0|1) ;;
  *) echo "B70_C4_LEASE_CHECK_ONLY must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
exec > >(tee -a "$OUT/gate.log") 2>&1
say() { echo "[c4-m11-w8a16 $(date -u +%H:%M:%S)] $*"; }

require_external_single_card_lease() {
  local fd expected actual
  if [ "$CARD" = 0 ]; then fd=8; else fd=9; fi
  expected="$LOCK_BASE.$CARD"
  actual="$(readlink "/proc/$$/fd/$fd" 2>/dev/null || true)"
  if [ "$actual" != "$expected" ]; then
    echo "refusing GPU work: invoke through ./bin/gpu-run --card $CARD" >&2
    echo "fd$fd=$actual expected=$expected" >&2
    return 2
  fi
  flock -n "$fd" || {
    echo "inherited card-$CARD lease fd is not locked" >&2
    return 2
  }
  {
    echo "LEASE_CHECK PASS card=$CARD"
    echo "fd$fd=$actual"
    echo "owner=$(cat "$expected.owner" 2>/dev/null || true)"
  } | tee "$OUT/lease_check.txt"
}

save_container() {
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker logs "$NAME" >"$OUT/container.log" 2>&1 || true
    docker inspect "$NAME" >"$OUT/container_inspect.json" 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$lease_proven" != 1 ]; then
    say "exit rc=$rc before lease proof; no GPU/container cleanup attempted"
    exit "$rc"
  fi
  save_container
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [ ! -s "$OUT/health_post.log" ]; then
    "$REPO/bin/xpu-health" --card "$CARD" --img "$IMAGE" \
      >"$OUT/health_post.log" 2>&1 || rc=1
  fi
  cat "$OUT/health_post.log" || true
  say "exit rc=$rc artifacts=$OUT endpoint_policy=untouched"
  exit "$rc"
}
trap cleanup EXIT INT TERM

require_external_single_card_lease
lease_proven=1
if [ "$LEASE_CHECK_ONLY" = 1 ]; then
  trap - EXIT INT TERM
  say "lease-check-only PASS; no GPU/container work attempted"
  exit 0
fi

for artifact in "$SELF" "$BENCH" "$SO" \
  "$REPO/sglang/patches/w8a8_shim.py" \
  "$REPO/kernels/int8_gemm_w8a16.h" \
  "$REPO/kernels/int8_gemm_w8a8.h" \
  "$REPO/kernels/int8_quant_common.hpp"; do
  [ -s "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 2; }
done
for op in int8_gemm_w8a16 int8_gemm_w8a8 dynamic_per_token_int8_quant; do
  rg -a -q "$op" "$SO" || { echo "kernel SO missing op: $op" >&2; exit 2; }
done
docker inspect "$NAME" >/dev/null 2>&1 && {
  echo "benchmark container already exists: $NAME" >&2
  exit 2
}

sha256sum "$SELF" "$BENCH" "$SO" \
  "$REPO/sglang/patches/w8a8_shim.py" \
  "$REPO/kernels/int8_gemm_w8a16.h" \
  "$REPO/kernels/int8_gemm_w8a8.h" \
  "$REPO/kernels/int8_quant_common.hpp" >"$OUT/artifacts.sha256"
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
{
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "utc_start=$STAMP"
  echo "claim=c4_exact_m11_w8a16_microbench"
  echo "card=$CARD"
  echo "image=$IMAGE"
  echo "image_id=$IMAGE_ID"
  echo "so=$SO"
  echo "warmup=$WARMUP"
  echo "iterations=$ITERS"
  echo "repeats=$REPEATS"
  echo "minimum_gain_pct=5.0"
  echo "maximum_cv_pct=5.0"
  echo "p2p=0"
  echo "queue_profiling=0"
  echo "endpoint_policy=untouched_no_server_actions"
} >"$OUT/manifest.txt"

say "leased pre-health card=$CARD"
"$REPO/bin/xpu-health" --card "$CARD" --img "$IMAGE" \
  2>&1 | tee "$OUT/health_pre.log"

say "run exact-M=11 benchmark"
docker create --name "$NAME" --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  -e "ZE_AFFINITY_MASK=$CARD" \
  -e CCL_TOPO_P2P_ACCESS=0 \
  -e PYTHONUNBUFFERED=1 \
  -v "$KDIR:/work/kernel:ro" \
  -v "$BENCH:/work/bench_c4_m11_w8a16.py:ro" \
  -v "$OUT:/out" \
  "$IMAGE" bash -lc \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
   export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\${LD_LIBRARY_PATH:-}; \
   python3 /work/bench_c4_m11_w8a16.py \
     --so /work/kernel/_xpu_C.abi3.so --output /out/result.json \
     --warmup '$WARMUP' --iterations '$ITERS' --repeats '$REPEATS' \
     --minimum-gain-pct 5.0 --maximum-cv-pct 5.0" \
  >"$OUT/create.log"
docker inspect "$NAME" >"$OUT/container_inspect.json"
docker start -a "$NAME" 2>&1 | tee "$OUT/benchmark.log"
save_container

python3 - "$OUT/result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="ascii") as handle:
    result = json.load(handle)
if result.get("schema") != "c4_m11_w8a16_microbench_v1":
    raise SystemExit("bad result schema")
if result.get("analysis", {}).get("pass") is not True:
    raise SystemExit("benchmark result gate failed")
print("RESULT_GATE PASS")
PY
if rg -i \
  'device_lost|out_of_resources|ur_result_error|enginedead|segmentation fault|(^|[^a-z])nan([^a-z]|$)' \
  "$OUT/benchmark.log"; then
  echo "fatal marker in benchmark log" >&2
  exit 1
fi

say "leased post-health card=$CARD"
"$REPO/bin/xpu-health" --card "$CARD" --img "$IMAGE" \
  2>&1 | tee "$OUT/health_post.log"
sha256sum --check "$OUT/artifacts.sha256"
say "PASS artifacts=$OUT endpoint_policy=untouched"
