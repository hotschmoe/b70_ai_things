#!/usr/bin/env bash
# Candidate-only mechanism gate for repaired shared INT8 lm_head storage.
# Caller must hold both cards:
#   ./bin/gpu-run bash sglang/run_c4_lmhead_int8_mechanism.sh
# The daily-driver and experiment endpoints remain down after every exit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
ANALYZER="$REPO/sglang/analyze_c4_lmhead_int8_mechanism.py"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c4_lmhead_int8_mechanism_$STAMP}"
NAME="${NAME:-c4_lmhead_int8_mechanism}"
PORT="${PORT:-31003}"
PROD_NAME="${PROD_NAME:-qwen38_unsloth_ud_q4k_xl_tp2}"
PROD_PORT="${PROD_PORT:-18080}"
SERVED="qwen36-27b-w8a8-gptq-mtp-c4-lmhead-int8-mechanism"
CTX=131072
MAXREQ=4
KDIR="/mnt/vm_8tb/b70/w8a8_kernel"
PUSHDIR="$REPO/vllm/contrib/vllm_push_allreduce/prebuilt"
PUSH_AR_SO="/work/push_ar/libxpu_push_ar_graph.so"
PUSH_AR_MAXB=536870912
CKPT_HOST="$REPO/models/files/qwen3.6-27b/w8a8-sqgptq"

KERNEL_SO="$KDIR/_xpu_C.abi3.so"
W8A16_SOURCE="$REPO/kernels/int8_gemm_w8a16.h"
W8A8_SHIM="$REPO/sglang/patches/w8a8_shim.py"
WOQ_SHIM="$REPO/sglang/patches/woq_shim.py"
MTP_PATCH="$REPO/sglang/patches/mtp_replicated_embedding.py"
PUSH_PATCH="$REPO/sglang/patches/push_ar_xpu.py"
PUSH_SOURCE="$REPO/vllm/contrib/vllm_push_allreduce/118_xpu_push_ar_graph.cpp"
PUSH_HOST_SO="$PUSHDIR/libxpu_push_ar_graph.so"
MODEL_CONFIG="$CKPT_HOST/config.json"
MODEL_INDEX="$CKPT_HOST/model.safetensors.index.json"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/gate.log") 2>&1

say() { echo "[c4-lmhead-mechanism $(date -u +%H:%M:%S)] $*"; }

save_candidate() {
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker logs "$NAME" >"$OUT/server.log" 2>&1 || true
    docker inspect "$NAME" >"$OUT/container_inspect.json" 2>/dev/null || true
  fi
}

stop_candidate() {
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker stop -t 60 "$NAME" >"$OUT/stop.log" 2>&1 || true
    docker rm "$NAME" >>"$OUT/stop.log" 2>&1 || true
  fi
}

ensure_endpoints_down() {
  local ok=0
  if docker inspect "$NAME" >/dev/null 2>&1; then
    ok=1
  fi
  if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
    | rg -qx "$PROD_NAME"; then
    docker stop -t 60 "$PROD_NAME" >/dev/null 2>&1 || true
    docker rm "$PROD_NAME" >/dev/null 2>&1 || true
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ok=1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PROD_PORT/health" >/dev/null 2>&1; then
    ok=1
  fi
  if [ "$ok" = 0 ]; then
    echo "down" >"$OUT/endpoint_state.txt"
  else
    echo "not-down" >"$OUT/endpoint_state.txt"
  fi
  return "$ok"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  save_candidate
  stop_candidate
  if [ ! -s "$OUT/health_post.log" ]; then
    "$REPO/bin/xpu-health" >"$OUT/health_post.log" 2>&1 || rc=1
  fi
  ensure_endpoints_down || rc=1
  cat "$OUT/health_post.log" || true
  say "exit rc=$rc artifacts=$OUT endpoint=down"
  exit "$rc"
}
trap cleanup EXIT INT TERM

artifacts=(
  "$SHELF"
  "$ANALYZER"
  "$W8A8_SHIM"
  "$WOQ_SHIM"
  "$MTP_PATCH"
  "$KERNEL_SO"
  "$W8A16_SOURCE"
  "$PUSH_PATCH"
  "$PUSH_SOURCE"
  "$PUSH_HOST_SO"
  "$MODEL_CONFIG"
  "$MODEL_INDEX"
)
for artifact in "${artifacts[@]}"; do
  [ -s "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 2; }
done
rg -a -q 'int8_gemm_w8a16' "$KERNEL_SO" || {
  echo "kernel SO is missing int8_gemm_w8a16" >&2
  exit 2
}
sha256sum "${artifacts[@]}" >"$OUT/artifacts.sha256"

{
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "utc_start=$STAMP"
  echo "claim=repaired_lmhead_int8_mechanism"
  echo "context_length=$CTX"
  echo "max_running_requests=$MAXREQ"
  echo "baseline_capacity=143360"
  echo "minimum_capacity=136192"
  echo "push_ar_min_numel=0"
  echo "replicate_mtp_embed=1"
  echo "delay_mlp_ar=0"
  echo "fused_mlp_ar_norm=0"
  echo "lmhead_int8=1"
  echo "lmhead_compute=w8a16_only"
  echo "fixed_generation_tokens=640"
  echo "accept_rate_floor=0.20"
  echo "max_consecutive_zero_accept_samples=2"
  echo "endpoint_policy=down_after_gate_no_restore"
  docker image inspect sglang-xpu:mtp --format 'image_id={{.Id}}' \
    2>/dev/null || true
} >"$OUT/manifest.txt"

say "snapshot and stop the daily-driver endpoint if present"
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' \
  >"$OUT/docker_ps_before.txt"
if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
  | rg -qx "$PROD_NAME"; then
  curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
    >"$OUT/production_models_before.json" || true
  docker inspect "$PROD_NAME" >"$OUT/production_inspect_before.json"
  docker stop -t 60 "$PROD_NAME" >"$OUT/production_stop.log"
  docker rm "$PROD_NAME" >>"$OUT/production_stop.log"
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_pre.log"

say "start exact 131072 candidate"
CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$NAME" \
  SERVED="$SERVED" IMG=sglang-xpu:mtp \
  CKPT=/models/qwen3.6-27b/w8a8-sqgptq API_KEY= CONFIG_OVERLAY= \
  MEMFRAC=0.90 TOOLCALL=1 TOOLPARSER=qwen3_coder \
  REASONPARSER=qwen3 METRICS=1 THINKCAP=4096 \
  KDIR="$KDIR" PUSHDIR="$PUSHDIR" PUSH_AR_SO="$PUSH_AR_SO" \
  SPEC_STEPS=10 SPEC_DRAFT=11 REPLICATE_MTP_EMBED=1 \
  PUSH_AR=1 PUSH_AR_MIN_NUMEL=0 PUSH_AR_MAXB="$PUSH_AR_MAXB" \
  DELAY_MLP_AR=0 FUSED_MLP_AR_NORM=0 LMHEAD_INT8=1 \
  bash "$SHELF" start 2>&1 | tee "$OUT/start.log"

curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" \
  >"$OUT/models.json"
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" \
  >"$OUT/server_info_before.json"
docker inspect "$NAME" >"$OUT/container_inspect.json"

docker logs "$NAME" >"$OUT/server_before_fixed.log" 2>&1
before_lines="$(wc -l <"$OUT/server_before_fixed.log")"
say "run fixed 640-token acceptance and route trigger"
python3 - "$PORT" "$SERVED" "$OUT/fixed_generation.json" <<'PY'
import json
import sys
import urllib.request

port, model, output = sys.argv[1:]
prompt = (
    "Write a continuous detailed technical essay of at least 1200 words about "
    "GPU memory hierarchy, tensor parallel inference, collective communication, "
    "and INT8 quantization. Use complete sentences and do not use a table."
)
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "top_p": 1,
    "seed": 1234,
    "max_tokens": 640,
    "ignore_eos": True,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"content-type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.load(response)
with open(output, "w", encoding="ascii") as handle:
    json.dump(result, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
print(
    "FIXED -> completion_tokens="
    f"{result.get('usage', {}).get('completion_tokens')} "
    f"finish={result.get('choices', [{}])[0].get('finish_reason')}"
)
PY
docker logs "$NAME" >"$OUT/server_after_fixed.log" 2>&1
tail -n "+$((before_lines + 1))" "$OUT/server_after_fixed.log" \
  >"$OUT/fixed_server_delta.log"

say "run deterministic corpus and four-stream coherence gate"
python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
  --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --max-tokens 64 --out "$OUT/deterministic.json" \
  2>&1 | tee "$OUT/deterministic.log"
python3 "$REPO/vllm/gate_concurrent_coherence.py" \
  "http://127.0.0.1:$PORT/v1" "$SERVED" 2 2 128 \
  2>&1 | tee "$OUT/coherence.log"

curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" \
  >"$OUT/server_info_after.json"
save_candidate
sha256sum "${artifacts[@]}" >"$OUT/artifacts_after.sha256"
cmp "$OUT/artifacts.sha256" "$OUT/artifacts_after.sha256"

say "stop candidate before analysis"
stop_candidate
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_post.log"
ensure_endpoints_down

python3 "$ANALYZER" "$OUT"
say "mechanism gate complete; endpoints remain down"
