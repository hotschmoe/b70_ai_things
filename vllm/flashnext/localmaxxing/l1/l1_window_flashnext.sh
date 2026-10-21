#!/usr/bin/env bash
# L1 localmaxxing window: Flash-Next IQ3_XXS (ISTA GSQ-RCO) on the EP
# sidecar, BOTH cards (card 0 = model + 138 experts/layer, card 1 = 374
# remote experts/layer via the b70_ep plugin; world size 1, no oneCCL),
# no speculation. O4's decode recipe (env file), frozen L1 snapshot:
#   fork  = worktree vllm-lmx-20261021 (branch b70/lmx-20261021)
#   repo  = lmx-l1/frozen-repo (harness/plugins/bin/registry copy)
# health -> integrity -> serve -> promo gate -> lmx measure-local -> stop -> health.
# Never submits.
#
#   RECIPE=lmx-l1/recipe-flashnext.env MAX_LEN=131072 TAG=w3-fn bash l1_window_flashnext.sh
set -uo pipefail
L=/mnt/vm_8tb/b70/build/vllm-flashnext/lmx-l1
FR=$L/frozen-repo
H=$FR/vllm/flashnext/harness
LMX=$FR/vllm/flashnext/localmaxxing
W=/mnt/vm_8tb/b70/build/vllm-flashnext/vllm-lmx-20261021
B=/mnt/vm_8tb/b70/build/vllm-flashnext
ENGINE_COMMIT=cfd2b6ffdb8fc2a96c09712e7d04c946b149ca95
RECIPE=${RECIPE:?RECIPE}
MAX_LEN=${MAX_LEN:?MAX_LEN}
TAG=${TAG:?TAG}
PORT=${PORT:-18161}
MAX_SEQS=${MAX_SEQS:-4}
MAX_BATCHED=${MAX_BATCHED:-2048}
OUT=$L/runs/$TAG
NAME=vllm-fn-l1-$TAG
ALIAS=qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-vllm-sidecar-mtp0
HF=/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext
GGUF=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
HARD_KB=8388608
mkdir -p "$OUT"
test ! -e "$OUT/started" || { echo "refusing to overwrite $OUT"; exit 2; }
touch "$OUT/started"
exec > >(tee -a "$OUT/window.log") 2>&1
echo "=== L1 Flash-Next window $TAG MAX_LEN=$MAX_LEN $(date -u +%FT%TZ) ==="
export B70_AGENT=L1 B70_GPU_LOCK_TIMEOUT=14400 B70_CPU_LOCK_TIMEOUT=1800
export PYTHONIOENCODING=ascii:backslashreplace
if systemctl --user is-active -q hotschmoe-dd; then echo DD_ACTIVE_ABORT; exit 3; fi
AVAIL=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo); echo "MEM_AVAILABLE_KB:$AVAIL"
[ "$AVAIL" -lt 60000000 ] && { echo ABORT_LOW_RAM_START; exit 4; }

( cd "$W" && test "$(git rev-parse HEAD)" = "$ENGINE_COMMIT" && test -z "$(git status --porcelain)" ) \
  || { echo ENGINE_NOT_FROZEN; exit 5; }
( cd "$FR" && sha256sum -c --quiet "$L/frozen-repo.final.sha256" ) || { echo FROZEN_REPO_CHANGED; exit 5; }
cp -p "$RECIPE" "$OUT/recipe.env"
ENVS=()
for kv in $(grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | xargs); do ENVS+=(--env "$kv"); done
GGUF_SO=$(grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | grep '^B70_GGUF_SO=' | cut -d= -f2)
EP_SO=$(grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | grep '^B70_EP_SO=' | cut -d= -f2)
EXT=("$GGUF_SO" "$EP_SO" "$FR/vllm/flashnext/plugins/b70_o4_xpu/b70_o4_xpu/_C.so" "$L/l1_api.py")
for so in $(find "$B/c7b-fa-overlay" -name '*.so' | sort); do EXT+=("$so"); done
sha256sum "${EXT[@]}" | tee "$OUT/so-before.sha256"
stat -c '%n %s %Y' "$GGUF" "${GGUF%00001-of-00002.gguf}00002-of-00002.gguf" | tee "$OUT/weights.stat"
echo "engine_commit=$ENGINE_COMMIT" > "$OUT/engine.txt"

"$FR/bin/gpu-run" "$FR/bin/xpu-health" > "$OUT/health-before.txt" 2>&1
echo "HEALTH_BEFORE_RC:$?"; tail -n 1 "$OUT/health-before.txt"
grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt" || { echo HEALTH_BEFORE_FAIL; exit 6; }

PYPATH="$B/c7b-fa-overlay:$W:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$FR/vllm/flashnext/plugins/b70_gguf_xpu:$FR/vllm/flashnext/plugins/b70_ep_xpu:$FR/vllm/flashnext/plugins/b70_qsa_xpu:$FR/vllm/flashnext/plugins/b70_o4_xpu"
CFG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,8]}'
( while true; do a=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$a" -lt "$HARD_KB" ]; then echo "ABORT_LOW_RAM $a"; docker stop -t 5 "$NAME" >/dev/null 2>&1; exit 0; fi
    sleep 2; done ) &
WD=$!
timeout --signal=TERM --kill-after=30 90m \
"$H/xpu-run.sh" --card both --memory 100g --name "$NAME" --port "$PORT" \
  --env B70_GGUF=1 --env B70_EP=1 --env B70_QSA=0 --env "B70_GGUF_CPU_DEQUANT=$FR/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/libdequant_cpu.so" \
  --env "B70_GGUF_PATH=$GGUF" \
  --env "B70_EP_PROFILE=/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin" \
  --env B70_EP_Q0=138 --env B70_EP_SPLIT=profile \
  --env B70_PLE_NBYTES=28800138240 --env B70_EP_MEM_MIN_KB=$HARD_KB --env B70_EP_MAX_T=8192 \
  --env B70_XPU_CG_RECLAIM=500 --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 \
  --env "PYTHONPATH=$PYPATH" --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 --env VLLM_NO_USAGE_STATS=1 --env VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  --env SYCL_CACHE_PERSISTENT=0 --env "SYCL_CACHE_DIR=$B/cache/sycl-l1" \
  "${ENVS[@]}" \
  -- /opt/venv/bin/python "$L/l1_api.py" serve "$HF" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name hotschmoe-dd "$ALIAS" \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len "$MAX_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization 0.98 \
  --language-model-only --load-format b70_gguf --quantization b70_gguf --trust-remote-code \
  --attention-backend FLASH_ATTN --no-enable-prefix-caching --mamba-cache-mode align \
  --compilation-config "$CFG" > "$OUT/serve.log" 2>&1 &
SPID=$!
ready=0
for i in $(seq 1 720); do
  kill -0 $SPID 2>/dev/null || { echo "serve exited before ready"; break; }
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" \
     && grep -q "$ALIAS" "$OUT/models.json"; then ready=1; echo "READY $(date -u +%T)"; break; fi
  sleep 5
done
GATE_RC=9; LMX_RC=9
if [ $ready = 1 ]; then
  awk '/MemAvailable:|AnonPages:/{print}' /proc/meminfo | tee "$OUT/rss-ready.txt"
  grep -E "KV cache size|Maximum concurrency|PLE device|split layer 47" "$OUT/serve.log" | head -n 6
  curl -sf "http://127.0.0.1:$PORT/version" > "$OUT/version.json"; cat "$OUT/version.json"; echo
  "$FR/bin/cpu-run" --quiet python3 "$H/promo_gate.py" --base-url "http://127.0.0.1:$PORT" \
    --profile flashnext --expect-alias "$ALIAS" --timeout 900 --out "$OUT/gate.json" > "$OUT/gate.log" 2>&1
  GATE_RC=$?; echo "GATE_RC:$GATE_RC"; cat "$OUT/gate.log"
  VERSION=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$OUT/version.json")
  {
    echo "# vLLM (fork branch b70/lmx-20261021 @ ${ENGINE_COMMIT:0:12}) + out-of-tree b70_gguf_xpu / b70_ep_xpu / b70_o4_xpu plugins, 2x Arc Pro B70, expert-parallel sidecar (card 1 runs 374 of 512 experts per layer), no speculative decoding"
    echo -n "B70_GGUF=1 B70_EP=1 B70_EP_SPLIT=profile B70_EP_Q0=138 B70_XPU_CG_RECLAIM=500 "
    grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | grep -v '_SO=\|_RT=' | xargs | tr -d '\n'; echo " \\"
    echo "vllm serve Qwen3.8-Flash-Next (hf config) --load-format b70_gguf --quantization b70_gguf (GGUF Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf; PLE table in host RAM) \\"
    echo "  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS --max-num-batched-tokens $MAX_BATCHED \\"
    echo "  --gpu-memory-utilization 0.98 --language-model-only --attention-backend FLASH_ATTN --no-enable-prefix-caching --mamba-cache-mode align \\"
    echo "  --compilation-config '$CFG'  (no --speculative-config)"
  } > "$OUT/lmx-command.txt"
  "$FR/bin/cpu-run" --quiet python3 "$LMX/lmx_vllm.py" --measure-local \
    --base-url "http://127.0.0.1:$PORT" --served-model hotschmoe-dd --alias "$ALIAS" \
    --hf-id ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --quantization IQ3_XXS \
    --hardware "$LMX/hardware_pair.json" --engine-version "$VERSION" \
    --engine-commit "$ENGINE_COMMIT" --engine-build "b70-lmx-20261021 (vllm fork b70/lmx-20261021, O4 sidecar decode recipe)" \
    --command-file "$OUT/lmx-command.txt" --mode no-spec --out "$OUT/localmaxxing"
  LMX_RC=$?; echo "LMX_RC:$LMX_RC"
  for f in "$OUT"/localmaxxing/*/validation.json; do echo "$f"; cat "$f"; done
  curl -sf "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics-end.txt"
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/models-end.json"
else
  tail -n 40 "$OUT/serve.log"
fi
kill $WD 2>/dev/null
docker stop -t 20 "$NAME" >/dev/null 2>&1 || true
wait $SPID; SERVE_RC=$?; echo "SERVE_RC:$SERVE_RC"
sha256sum "${EXT[@]}" > "$OUT/so-after.sha256"
cmp -s "$OUT/so-before.sha256" "$OUT/so-after.sha256" && echo SO_UNCHANGED || echo SO_CHANGED
( cd "$W" && test "$(git rev-parse HEAD)" = "$ENGINE_COMMIT" && test -z "$(git status --porcelain)" ) && echo ENGINE_STILL_FROZEN || echo ENGINE_CHANGED
( cd "$FR" && sha256sum -c --quiet "$L/frozen-repo.final.sha256" ) && echo FROZEN_REPO_UNCHANGED || echo FROZEN_REPO_CHANGED
"$FR/bin/gpu-run" "$FR/bin/xpu-health" > "$OUT/health-after.txt" 2>&1
echo "HEALTH_AFTER_RC:$?"; tail -n 1 "$OUT/health-after.txt"
echo "L1_WINDOW_DONE ready=$ready gate=$GATE_RC lmx=$LMX_RC serve=$SERVE_RC"
