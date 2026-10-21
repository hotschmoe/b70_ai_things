#!/usr/bin/env bash
# PREPARED ONLY. Execute solely after root GPU grant; physicalcard1 <=120sbody.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
S="$(cd "$(dirname "$0")" && pwd)"
TAG=${TAG:?fresh tag required}
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
BUILD=$B/d1/hc_subgroup_r1
OUT=$BUILD/$TAG
[[ ! -e "$OUT" ]] || exit 5
BASE_SO=${B70_O4_SO:-$R/vllm/flashnext/plugins/b70_o4_xpu/b70_o4_xpu/_C.so}
[[ -s "$BUILD/libd1_hc_sg_r1.so" && -s "$BASE_SO" ]] || exit 3
mkdir -p "$OUT"
exec > >(tee "$OUT/window.log") 2>&1
export B70_AGENT=D1 B70_GPU_LOCK_TIMEOUT=600
sha256sum "$S/upcol.cpp" "$S/probe.py" "$S/build.sh" "$S/window.sh" "$BUILD/libd1_hc_sg_r1.so" "$BASE_SO" \
 "$B/vllm/vllm/models/qwen4_exp/xpu/hc_decode.py" "$R/vllm/flashnext/harness/xpu-run.sh" \
 "$R/vllm/flashnext/harness/xpu-entrypoint.sh" > "$OUT/source-before.sha256"
docker image inspect vllm-flashnext-xpu:py312 --format '{{.Id}}' > "$OUT/image-id.txt"
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1
PRE=$?
[[ "$PRE" == 0 ]] && rg -q 'xpu-health: HEALTHY' "$OUT/health-before.txt" || exit 3
"$R/bin/cpu-run" --quiet "$R/vllm/flashnext/harness/xpu-run.sh" --card 1 \
 --name "vllm-fn-d1-hcsg-$TAG" --memory 8g \
 --env "B70_O4_SO=$BASE_SO" --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
 --env SYCL_CACHE_PERSISTENT=0 -- /opt/venv/bin/python "$S/probe.py" --output "$OUT" > "$OUT/probe.log" 2>&1
BODY=$?
if docker inspect "vllm-fn-d1-hcsg-$TAG" >/dev/null 2>&1; then
  echo 'CONTAINER_LEFTOVER:1'
  docker stop -t 10 "vllm-fn-d1-hcsg-$TAG" > "$OUT/stop.txt" 2>&1 || true
else echo 'CONTAINER_LEFTOVER:0'; fi
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
POST=$?
sha256sum --check "$OUT/source-before.sha256" > "$OUT/source-after-check.txt" 2>&1
SOURCE=$?
printf 'BODY_RC:%s POST_RC:%s SOURCE_RC:%s\n' "$BODY" "$POST" "$SOURCE" | tee "$OUT/status.txt"
echo "WINDOW_DONE $(date -u +%FT%TZ)"
[[ "$BODY" == 0 && "$POST" == 0 && "$SOURCE" == 0 ]]
