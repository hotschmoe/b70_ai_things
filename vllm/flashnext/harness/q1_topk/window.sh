#!/usr/bin/env bash
# Only execute after explicit parent GPU grant. Physical card1, 240s body.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
S="$(cd "$(dirname "$0")" && pwd)"
BUILD=${Q1_TOPK_ROOT:-$B/q1/topk-r1}
TAG=${Q1_PROFILE_TAG:-profile-r1}
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
OUT=$BUILD/$TAG
[[ ! -e "$OUT" ]] || { echo "ABORT existing output $OUT"; exit 5; }
mkdir -p "$OUT/trace"
exec > >(tee "$OUT/window.log") 2>&1
export B70_AGENT=Q1 B70_GPU_LOCK_TIMEOUT=600
echo "Q1_TOPK_WINDOW_START $(date -u +%FT%TZ) card=1 body_cap_s=240"
rg --files "$R/vllm/flashnext/plugins/b70_qsa_xpu" | sort | xargs sha256sum > "$OUT/source-before.sha256"
sha256sum "$S/q1_topk.cpp" "$BUILD/libq1_topk.so" "$S/probe.py" "$S/oracle.py" \
 "$S/inner.sh" "$S/window.sh" "$R/vllm/flashnext/harness/xpu-run.sh" \
 "$R/vllm/flashnext/harness/xpu-entrypoint.sh" \
 "$B/b70_qsa_xpu/libb70_qsa_xpu.so" >> "$OUT/source-before.sha256"
docker image inspect vllm-flashnext-xpu:py312 --format '{{.Id}}' > "$OUT/image-id.txt"
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1
PRE=$?
echo "HEALTH_BEFORE_RC:$PRE"
[[ "$PRE" == 0 ]] && grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt" || exit 3
B70_AGENT=Q1 "$R/bin/cpu-run" --quiet \
 "$R/vllm/flashnext/harness/xpu-run.sh" --card 1 --name "vllm-fn-q1-topk-$TAG" --memory 12g \
 --env "PYTHONPATH=$S:$R/vllm/flashnext/plugins/b70_qsa_xpu" \
 --env "Q1_TOPK_ROOT=$BUILD" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
 --env "Q1_UT_SESSION=q1topk$TAG" --env "Q1_PROFILE_OUT=$OUT" \
 --env B70_QSA=0 --env SYCL_CACHE_PERSISTENT=0 \
 -- bash "$S/inner.sh" > "$OUT/run.log" 2>&1
BODY=$?
echo "BODY_RC:$BODY"
if docker inspect "vllm-fn-q1-topk-$TAG" > /dev/null 2>&1; then
  echo 'CONTAINER_LEFTOVER:1'
  docker stop -t 10 "vllm-fn-q1-topk-$TAG" > "$OUT/stop.txt" 2>&1 || true
else
  echo 'CONTAINER_LEFTOVER:0'
fi
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
POST=$?
echo "HEALTH_AFTER_RC:$POST"
sha256sum --check "$OUT/source-before.sha256" > "$OUT/source-after-check.txt"
SOURCE=$?
echo "SOURCE_CHECK_RC:$SOURCE"
echo "WINDOW_DONE $(date -u +%FT%TZ)"
[[ "$BODY" == 0 && "$POST" == 0 && "$SOURCE" == 0 ]] || exit 1
