#!/usr/bin/env bash
# Short counts-only quant-route census for the Unsloth XL TP=2 MTP3 shelf.
# Caller must hold both GPU leases. The endpoint is left down.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-ud-q4-k-xl/serve.sh"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/qwen38_ud_q4k_xl_quant_census_$STAMP}"
NAME="${NAME:-qwen38_ud_q4k_xl_quant_census}"
PORT="${PORT:-31004}"
SERVED="${SERVED:-qwen38-27b-ud-q4-k-xl-mtp3-quant-census}"
IMG="${IMG:-qwen38-b70:quant-census}"
IMG_ID="${IMG_ID:-sha256:44d657a92f4c1d59d264c680a7dd0e66a8547a21b0c71133a63347f67c2cda5c}"
OVERLAY="${OVERLAY:-$REPO/llamacpp/qwen38_ud_q4k_xl_profile_entrypoint.sh}"
API_KEY_FILE="${API_KEY_FILE:-/mnt/vm_8tb/b70/secrets/dd_api_key}"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if docker inspect "$NAME" >/dev/null 2>&1; then
        docker stop --time 60 "$NAME" >"$OUT/stop.log" 2>&1 || true
        docker logs "$NAME" >"$OUT/server.log" 2>&1 || true
        docker rm -f "$NAME" >/dev/null 2>&1 || true
    fi
    "$REPO/bin/xpu-health" >"$OUT/health_final.log" 2>&1 || rc=1
    cat "$OUT/health_final.log"
    printf 'quant census exit rc=%s artifacts=%s endpoint=down\n' "$rc" "$OUT"
    exit "$rc"
}
trap cleanup EXIT INT TERM

docker image inspect "$IMG" --format '{{.Id}}' | tee "$OUT/image_id.txt"
[ "$(<"$OUT/image_id.txt")" = "$IMG_ID" ]
"$REPO/bin/xpu-health" | tee "$OUT/health_pre.log"

NAME="$NAME" PORT="$PORT" SERVED="$SERVED" IMG="$IMG" IMG_ID="$IMG_ID" \
    OVERLAY="$OVERLAY" ENABLE_MTP=1 MTP_DRAFT_MAX=3 \
    LAB_DOORS=0 GGML_SYCL_QUANT_CENSUS=1 \
    bash "$SHELF" start 2>&1 | tee "$OUT/start.log"

docker inspect "$NAME" >"$OUT/container_inspect.json"
curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$OUT/models.json"

auth=()
if [ -s "$API_KEY_FILE" ]; then
    auth=(-H "Authorization: Bearer $(<"$API_KEY_FILE")")
fi
curl -fsS --max-time 900 "${auth[@]}" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that merges two sorted lists, explain its complexity, and include tests.\"}],\"max_tokens\":512,\"temperature\":0,\"ignore_eos\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" >"$OUT/response.json"

docker stop --time 60 "$NAME" >"$OUT/stop.log" 2>&1
docker logs "$NAME" >"$OUT/server.log" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
python3 "$REPO/llamacpp/parse_quant_census.py" "$OUT/server.log" \
    --write "$OUT/quant_census.json" | tee "$OUT/quant_census.stdout.json"
"$REPO/bin/xpu-health" | tee "$OUT/health_post.log"
